"""Native O(p^2)-per-step coefficients via incremental inverse maintenance.

The base estimators in :mod:`sufficient_regression.ols` maintain the sufficient
statistics ``XtX``, ``Xty`` and ``yty`` incrementally, but recover coefficients
with a fresh dense ``np.linalg.solve`` every time they are requested. That solve
is ``Theta(p^3)`` per call, so the performance note correctly observes that
"end-to-end wall-clock speedups are bounded by the unchanged dense solve cost
unless the implementation also updates or reuses a factorization"
(``sufficient_statistics_ols_performance.tex``).

This module implements the missing piece. It maintains the inverse of the
regularized Gram matrix

    A = XtX + R          (R is the fixed ridge penalty matrix)
    M = A^{-1}

directly, using rank-1 Sherman-Morrison updates. Appending a weighted row, or
dropping one from a rolling window, is a rank-1 (down)date of ``A`` and therefore
a rank-1 update of ``M`` costing ``Theta(p^2)``. Coefficients are then

    beta = M @ Xty

which is ``Theta(p^2)``, not ``Theta(p^3)``. This is the classic recursive least
squares / Sherman-Morrison trick behind ``statsmodels.RecursiveLS`` and friends.

Implementation
--------------
Small and medium sequential updates are fused in a compiled Cython kernel:
validation crosses the Python/native boundary once, then sufficient statistics,
inverse state, and coefficients are updated together without p-by-p Python
temporaries. Wide systems use NumPy's optimized BLAS kernels for higher
matrix-vector throughput. The dense estimators in :mod:`ols` remain the
correctness reference. Sherman-Morrison updates accumulate floating-point error
across a stream (downdates especially), so these estimators:

- rebuild ``M`` exactly from ``XtX`` on a bounded cadence to cap drift, and
- still expose the exact maintained ``XtX``/``Xty`` for any caller that wants a
  cold dense solve.

Assumptions / scope
-------------------
- The maintained inverse is the inverse of ``XtX + R``. ``R`` is fixed, so only
  rank-1 row contributions move ``A``; ridge >= 0 is supported.
- The inverse is built lazily on the first solve (so warm-up rows that leave the
  unregularized system singular simply defer the build). Once it exists, every
  row add/drop updates it in place.
- A near-singular downdate (the removed row would make ``A`` singular) is not
  silently absorbed: the inverse is invalidated so the next solve rebuilds from
  ``XtX`` and fails loudly via :class:`SingularRegressionError` if truly
  singular. This is a rebuild trigger, not a different-answer fallback.
- ``ForgettingOLS`` is intentionally not implemented here: its ridge term is not
  decayed alongside ``XtX``, so the regularized system does not reduce to a clean
  single rank-1 recursion. Pure (ridge=0) forgetting would, but is left out to
  keep the native kernel's invariants simple.
"""

from __future__ import annotations

import numpy as np

from . import _native
from .ols import (
    RollingOLS,
    SingularRegressionError,
    _SufficientOLSBase,
    _as_single_row,
    _as_single_value,
    _as_single_weight,
    _regularization_matrix,
    _require_full_rank_unregularized,
    _validate_positive_integer,
    _weighted_stats,
)

# Downdates can drive the Sherman-Morrison denominator to zero when the removed
# row would make the regularized system singular. Below this magnitude we refuse
# the in-place update and rebuild instead; the rebuild raises loudly if the
# system really is singular. The threshold is on the dimensionless quantity
# ``1 + c * z^T M z`` (c = +/- weight), which is O(1) for well-posed systems.
_SHERMAN_MORRISON_MIN_DENOM = 1e-10
# Scalar native loops avoid dispatch overhead for small and medium systems.
# At this width, optimized BLAS matrix-vector kernels overtake the scalar loop
# on supported NumPy builds. A fixed boundary keeps execution deterministic.
_BLAS_TRANSITION_MIN_PARAMS = 256


def _scalar_weight(sample_weight: float | None) -> float:
    """Validate and return one row weight without allocating a length-one array."""

    if sample_weight is None:
        return 1.0
    weight_array = np.asarray(sample_weight, dtype=float)
    if weight_array.ndim == 0:
        weight = float(weight_array)
    elif weight_array.ndim == 1 and weight_array.shape[0] == 1:
        weight = float(weight_array[0])
    else:
        raise ValueError("sample_weight must be one scalar value for push.")
    if not np.isfinite(weight):
        raise ValueError("sample_weight must be finite.")
    if weight < 0:
        raise ValueError("sample_weight cannot be negative.")
    return weight


def _stage_streaming_row(
    model: _SufficientOLSBase,
    x,
    y: float,
    sample_weight: float | None,
) -> tuple[np.ndarray, float, float]:
    """Stage one public ``push`` input in the estimator's reusable row buffer.

    Shape mismatches and bad weights raise here. Finiteness of the feature row
    and the target is checked inside the native kernel instead, which already
    loads every element on its way through: the NumPy ``isfinite`` scan this
    replaces cost more per row than the update arithmetic it guarded. Wide
    systems skip the kernel, so they are validated explicitly below.

    The returned row is owned by the estimator and is overwritten by the next
    push. Every consumer either feeds it straight to an update or copies it into
    the rolling ring buffer, so no caller retains it.
    """

    if model.n_features_in_ is None:
        raise RuntimeError("Single-row native preparation requires fitted state.")

    features = x if type(x) is np.ndarray else np.asarray(x, dtype=float)
    if features.ndim == 2 and features.shape[0] == 1:
        features = features[0]
    elif features.ndim != 1:
        raise ValueError("x must be one row.")
    if features.shape[0] != model.n_features_in_:
        raise ValueError(
            "X has a different number of features than the fitted state; "
            f"expected {model.n_features_in_}, got {features.shape[0]}."
        )

    row = model._push_scratch
    if row is None:
        row = np.zeros(model.n_params_, dtype=float)
        if model.fit_intercept:
            # The intercept column is constant, so it is written once at
            # allocation rather than on every push.
            row[0] = 1.0
        model._push_scratch = row
    if model.fit_intercept:
        row[1:] = features
    else:
        row[:] = features

    y_type = type(y)
    if y_type is float or y_type is np.float64:
        y_value = float(y)
    else:
        y_value = float(_as_single_value(y, name="y")[0])
    weight = _scalar_weight(sample_weight)

    if model.n_params_ >= _BLAS_TRANSITION_MIN_PARAMS:
        _native.validate_streaming_row(row, y_value)
    return row, y_value, weight


def _add_scalar_statistics(
    model: _SufficientOLSBase,
    y: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Update non-matrix statistics after a native batch transition."""

    weighted_y = weights * y
    model._yty += float(weighted_y @ y)
    model._y_sum += float(weights @ y)
    model.weight_sum_ += float(np.sum(weights))
    model.n_observations_ += int(y.shape[0])


def _invert_regularized_gram(A: np.ndarray) -> np.ndarray:
    """Dense inverse of the regularized Gram matrix, failing loudly if singular.

    Uses a solve against the identity rather than ``np.linalg.inv`` so the
    singular case raises the package's :class:`SingularRegressionError` with the
    same guidance the dense estimators give.
    """

    try:
        return np.linalg.solve(A, np.eye(A.shape[0], dtype=float))
    except np.linalg.LinAlgError as exc:
        raise SingularRegressionError(
            "The regression system is singular. Add more independent rows, "
            "remove collinear features, or use ridge > 0."
        ) from exc


class _RecursiveInverseMixin:
    """Maintain ``M = (XtX + R)^{-1}`` under rank-1 row (down)dates.

    Mixed in ahead of a :class:`_SufficientOLSBase` subclass so that the exact
    sufficient statistics are still maintained by the base, while the coefficient
    solve reuses the incrementally maintained inverse.
    """

    def _reset_state(self) -> None:
        super()._reset_state()
        # ``None`` means "not currently maintained"; the next solve rebuilds it
        # from XtX. Rows added while it is None update only XtX (cheap and
        # exact), so the eventual rebuild is still correct.
        self._inv: np.ndarray | None = None
        self._inv_updates: int = 0
        # Reusable augmented row for ``push``, sized on first use. Reallocated
        # implicitly on refit because ``fit`` resets state.
        self._push_scratch: np.ndarray | None = None

    def _regularized_gram(self) -> np.ndarray:
        return self._xtx + _regularization_matrix(
            self.n_params_,
            ridge=self.ridge,
            fit_intercept=self.fit_intercept,
            regularize_intercept=self.regularize_intercept,
        )

    def _ensure_inverse(self) -> np.ndarray:
        """Return ``M``, building it from XtX with a dense inverse if needed."""

        if self._inv is None:
            _require_full_rank_unregularized(
                self._xtx,
                n_params=self.n_params_,
                ridge=self.ridge,
            )
            self._inv = _invert_regularized_gram(self._regularized_gram())
            # The solve can differ by a few ulps across triangles. The native
            # kernel preserves exact symmetry after this one-time projection.
            self._inv = np.ascontiguousarray(
                0.5 * (self._inv + self._inv.T),
                dtype=float,
            )
            self._inv_updates = 0
        return self._inv

    def _blas_recursive_update(
        self,
        z: np.ndarray,
        y_value: float,
        weight: float,
        *,
        sign: float,
    ) -> bool:
        """Update inverse and coefficients using wide-matrix BLAS operations."""

        if weight == 0.0:
            return True
        c = sign * weight
        update_direction = self._inv @ z
        denominator = 1.0 + c * float(z @ update_direction)
        if (
            not np.isfinite(denominator)
            or abs(denominator) <= _SHERMAN_MORRISON_MIN_DENOM
        ):
            return False
        residual = y_value - float(z @ self._params_cache)
        scale = c / denominator
        self._params_cache += scale * update_direction * residual
        # The initial inverse is projected to exact symmetry, and the outer
        # correction is bitwise symmetric, so per-update symmetrization would
        # only add another full p-by-p allocation.
        self._inv -= scale * np.outer(update_direction, update_direction)
        return True

    def _solve_params(self) -> np.ndarray:
        self._require_stats()
        if self.weight_sum_ <= 0:
            raise SingularRegressionError("Cannot solve with zero total sample weight.")
        return self._ensure_inverse() @ self._xty

    @property
    def coefficient_covariance_(self) -> np.ndarray:
        """Classical covariance for ``params_``, reusing the maintained inverse.

        Identical result to the dense base property, but when ``ridge=0`` the
        regularized Gram matrix is exactly ``XtX``, so ``M`` already holds
        ``(XtX)^{-1}`` and the covariance is ``sigma^2 * M`` with no extra solve.
        """

        # Raises for ridge != 0, nonpositive dof, or rank deficiency before we
        # touch the inverse, matching the base contract.
        sigma_squared = self.residual_variance_
        inverse_xtx = self._ensure_inverse()
        return np.array(sigma_squared * inverse_xtx, dtype=float, copy=True)


class RankOneIncrementalOLS(_RecursiveInverseMixin, _SufficientOLSBase):
    """Native append-only OLS with O(p^2)-per-step coefficients.

    Behaves like :class:`~sufficient_regression.IncrementalOLS` but maintains the
    inverse of the regularized Gram matrix so that both row appends and
    coefficient reads cost ``Theta(p^2)`` instead of a fresh ``Theta(p^3)`` solve.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        ridge: float = 0.0,
        regularize_intercept: bool = False,
        refresh_every: int | None = None,
        missing: str = "raise",
    ) -> None:
        # Append-only Sherman-Morrison updates have denom >= 1 and no
        # cancellation, so they are numerically benign; periodic rebuilds are
        # off by default. Callers streaming millions of rows can set a cadence to
        # cap any residual drift.
        if refresh_every is not None:
            self.refresh_every: int | None = _validate_positive_integer(
                refresh_every, name="refresh_every"
            )
        else:
            self.refresh_every = None
        super().__init__(
            fit_intercept=fit_intercept,
            ridge=ridge,
            regularize_intercept=regularize_intercept,
            missing=missing,
        )

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
    ) -> "RankOneIncrementalOLS":
        self._reset_state()
        _, X_aug, y_arr, weights = self._validate_X_y(X, y, sample_weight, reset=True)
        (
            self._xtx,
            self._xty,
            self._yty,
            self._y_sum,
            self.weight_sum_,
        ) = _weighted_stats(X_aug, y_arr, weights)
        self.n_observations_ = int(y_arr.shape[0])
        # Build the inverse lazily on first solve: a fit batch may not yet be
        # full rank, and deferring keeps the failure at coefficient-read time.
        self._inv = None
        self._inv_updates = 0
        self._dirty = True
        return self

    def partial_fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
    ) -> "RankOneIncrementalOLS":
        if self.n_features_in_ is None:
            return self.fit(X, y, sample_weight=sample_weight)
        _, X_aug, y_arr, weights = self._validate_X_y(X, y, sample_weight, reset=False)
        X_aug = np.ascontiguousarray(X_aug, dtype=float)
        y_arr = np.ascontiguousarray(y_arr, dtype=float)
        weights = np.ascontiguousarray(weights, dtype=float)
        if self._inv is None:
            self._add_stats(
                *_weighted_stats(X_aug, y_arr, weights),
                n_rows=y_arr.shape[0],
            )
            return self

        if self._params_cache is None:
            # Internal callers can build the inverse directly; keep the native
            # state invariant explicit rather than assuming params_ did it.
            self._params_cache = self._inv @ self._xty
        if self.n_params_ >= _BLAS_TRANSITION_MIN_PARAMS:
            processed = 0
            inverse_valid = True
            inverse_updates = 0
            for row, y_value, weight in zip(
                X_aug,
                y_arr,
                weights,
                strict=True,
            ):
                self._xtx += float(weight) * np.outer(row, row)
                self._xty += float(weight) * row * float(y_value)
                processed += 1
                inverse_valid = self._blas_recursive_update(
                    row,
                    float(y_value),
                    float(weight),
                    sign=1.0,
                )
                if not inverse_valid:
                    break
                inverse_updates += int(weight != 0.0)
        else:
            processed, inverse_valid, inverse_updates = _native.append_update(
                self._xtx,
                self._xty,
                self._inv,
                self._params_cache,
                X_aug,
                y_arr,
                weights,
                _SHERMAN_MORRISON_MIN_DENOM,
            )
        if processed < y_arr.shape[0]:
            remaining = slice(processed, None)
            xtx, xty, _, _, _ = _weighted_stats(
                X_aug[remaining],
                y_arr[remaining],
                weights[remaining],
            )
            self._xtx += xtx
            self._xty += xty
        _add_scalar_statistics(self, y_arr, weights)
        self._inv_updates += int(inverse_updates)

        refresh_due = (
            self.refresh_every is not None
            and self._inv_updates >= self.refresh_every
        )
        if not inverse_valid or refresh_due:
            # A refresh is part of the numerical contract: exact statistics
            # remain current and the next coefficient read rebuilds from them.
            self._inv = None
            self._params_cache = None
            self._inv_updates = 0
            self._dirty = True
        else:
            self._dirty = False
        return self

    append = partial_fit

    def push(
        self,
        x,
        y: float,
        *,
        sample_weight: float | None = None,
    ) -> "RankOneIncrementalOLS":
        if self.n_features_in_ is None:
            return self.fit(
                _as_single_row(x, name="x"),
                _as_single_value(y, name="y"),
                sample_weight=_as_single_weight(sample_weight),
            )
        row, y_value, weight = _stage_streaming_row(
            self,
            x,
            y,
            sample_weight,
        )
        if self._inv is None:
            _native.stats_add(self._xtx, self._xty, row, y_value, weight)
            self._yty += weight * y_value * y_value
            self._y_sum += weight * y_value
            self.weight_sum_ += weight
            self.n_observations_ += 1
            self._dirty = True
            return self

        if self._params_cache is None:
            self._params_cache = self._inv @ self._xty
        if self.n_params_ >= _BLAS_TRANSITION_MIN_PARAMS:
            self._xtx += weight * np.outer(row, row)
            self._xty += weight * row * y_value
            inverse_valid = self._blas_recursive_update(
                row,
                y_value,
                weight,
                sign=1.0,
            )
            inverse_updates = int(weight != 0.0 and inverse_valid)
        else:
            inverse_valid, inverse_updates = _native.append_update_one(
                self._xtx,
                self._xty,
                self._inv,
                self._params_cache,
                row,
                y_value,
                weight,
                _SHERMAN_MORRISON_MIN_DENOM,
            )
        self._yty += weight * y_value * y_value
        self._y_sum += weight * y_value
        self.weight_sum_ += weight
        self.n_observations_ += 1
        self._inv_updates += int(inverse_updates)
        refresh_due = (
            self.refresh_every is not None
            and self._inv_updates >= self.refresh_every
        )
        if not inverse_valid or refresh_due:
            self._inv = None
            self._params_cache = None
            self._inv_updates = 0
            self._dirty = True
        else:
            self._dirty = False
        return self


class RankOneRollingOLS(_RecursiveInverseMixin, RollingOLS):
    """Native fixed-window OLS with O(p^2)-per-slide coefficients.

    Each slide is one rank-1 downdate (departing row) and one rank-1 update
    (arriving row) of the maintained inverse, so a full ``params_`` read after
    every slide costs ``Theta(p^2)`` rather than a fresh ``Theta(p^3)`` solve.

    The inverse is rebuilt from ``XtX`` whenever the base recomputes its
    sufficient statistics (``recompute_every``, defaulting to one full window),
    so Sherman-Morrison drift is bounded by the same cadence that bounds add/drop
    roundoff in the dense estimator.
    """

    # Rolling rows live in contiguous arrays rather than the base class's deque.
    # The native transition can therefore consume both rows without constructing
    # Python tuples or allocating p-by-p temporaries.
    refresh_every: int | None = None

    def _reset_state(self) -> None:
        super()._reset_state()
        self._row_buffer: np.ndarray | None = None
        self._target_buffer: np.ndarray | None = None
        self._weight_buffer: np.ndarray | None = None
        self._buffer_start = 0
        self._buffer_length = 0

    @property
    def window_size_(self) -> int:
        return self._buffer_length

    def _allocate_buffers(self) -> None:
        self._row_buffer = np.empty((self.window, self.n_params_), dtype=float)
        self._target_buffer = np.empty(self.window, dtype=float)
        self._weight_buffer = np.empty(self.window, dtype=float)
        self._buffer_start = 0
        self._buffer_length = 0

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
    ) -> "RankOneRollingOLS":
        self._reset_state()
        _, X_aug, y_arr, weights = self._validate_X_y(
            X,
            y,
            sample_weight,
            reset=True,
        )
        X_aug = np.ascontiguousarray(X_aug[-self.window :], dtype=float)
        y_arr = np.ascontiguousarray(y_arr[-self.window :], dtype=float)
        weights = np.ascontiguousarray(weights[-self.window :], dtype=float)
        self._allocate_buffers()
        self._buffer_length = y_arr.shape[0]
        self._row_buffer[: self._buffer_length] = X_aug
        self._target_buffer[: self._buffer_length] = y_arr
        self._weight_buffer[: self._buffer_length] = weights
        (
            self._xtx,
            self._xty,
            self._yty,
            self._y_sum,
            self.weight_sum_,
        ) = _weighted_stats(X_aug, y_arr, weights)
        self._xtx = np.ascontiguousarray(self._xtx, dtype=float)
        self._xty = np.ascontiguousarray(self._xty, dtype=float)
        self.n_observations_ = int(y_arr.shape[0])
        self._updates_since_recompute = 0
        self._dirty = True
        return self

    def partial_fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
    ) -> "RankOneRollingOLS":
        if self.n_features_in_ is None:
            return self.fit(X, y, sample_weight=sample_weight)
        _, X_aug, y_arr, weights = self._validate_X_y(
            X,
            y,
            sample_weight,
            reset=False,
        )
        X_aug = np.ascontiguousarray(X_aug, dtype=float)
        for row, y_value, weight in zip(X_aug, y_arr, weights, strict=True):
            self._push_prepared_native(row, float(y_value), float(weight))
        return self

    append = partial_fit

    def push(
        self,
        x,
        y: float,
        *,
        sample_weight: float | None = None,
    ) -> "RankOneRollingOLS":
        if self.n_features_in_ is None:
            return self.fit(
                _as_single_row(x, name="x"),
                _as_single_value(y, name="y"),
                sample_weight=_as_single_weight(sample_weight),
            )
        row, y_value, weight = _stage_streaming_row(
            self,
            x,
            y,
            sample_weight,
        )
        self._push_prepared_native(row, y_value, weight)
        return self

    def _push_prepared_native(
        self,
        row: np.ndarray,
        y_value: float,
        weight: float,
    ) -> None:
        self._require_stats()
        if self._row_buffer is None:
            raise RuntimeError("Rolling native buffers are not initialized.")

        if self._buffer_length < self.window:
            insert_at = (
                self._buffer_start + self._buffer_length
            ) % self.window
            if self._inv is None:
                if self.n_params_ >= _BLAS_TRANSITION_MIN_PARAMS:
                    self._xtx += weight * np.outer(row, row)
                    self._xty += weight * row * y_value
                else:
                    _native.stats_add(
                        self._xtx,
                        self._xty,
                        row,
                        y_value,
                        weight,
                    )
                self._dirty = True
            else:
                if self._params_cache is None:
                    self._params_cache = self._inv @ self._xty
                if self.n_params_ >= _BLAS_TRANSITION_MIN_PARAMS:
                    self._xtx += weight * np.outer(row, row)
                    self._xty += weight * row * y_value
                    inverse_valid = self._blas_recursive_update(
                        row,
                        y_value,
                        weight,
                        sign=1.0,
                    )
                    inverse_updates = int(weight != 0.0 and inverse_valid)
                else:
                    inverse_valid, inverse_updates = _native.append_update_one(
                        self._xtx,
                        self._xty,
                        self._inv,
                        self._params_cache,
                        row,
                        y_value,
                        weight,
                        _SHERMAN_MORRISON_MIN_DENOM,
                    )
                self._inv_updates += int(inverse_updates)
                if inverse_valid:
                    self._dirty = False
                else:
                    self._invalidate_inverse()
            self._buffer_length += 1
            self.n_observations_ += 1
        else:
            insert_at = self._buffer_start
            old_row = self._row_buffer[insert_at]
            old_y = float(self._target_buffer[insert_at])
            old_weight = float(self._weight_buffer[insert_at])
            if self._inv is None:
                if self.n_params_ >= _BLAS_TRANSITION_MIN_PARAMS:
                    self._xtx -= old_weight * np.outer(old_row, old_row)
                    self._xty -= old_weight * old_row * old_y
                    self._xtx += weight * np.outer(row, row)
                    self._xty += weight * row * y_value
                else:
                    _native.stats_slide(
                        self._xtx,
                        self._xty,
                        old_row,
                        old_y,
                        old_weight,
                        row,
                        y_value,
                        weight,
                    )
                self._dirty = True
            else:
                if self._params_cache is None:
                    self._params_cache = self._inv @ self._xty
                if self.n_params_ >= _BLAS_TRANSITION_MIN_PARAMS:
                    self._xtx -= old_weight * np.outer(old_row, old_row)
                    self._xty -= old_weight * old_row * old_y
                    self._xtx += weight * np.outer(row, row)
                    self._xty += weight * row * y_value
                    inverse_valid = self._blas_recursive_update(
                        old_row,
                        old_y,
                        old_weight,
                        sign=-1.0,
                    )
                    if inverse_valid:
                        inverse_valid = self._blas_recursive_update(
                            row,
                            y_value,
                            weight,
                            sign=1.0,
                        )
                else:
                    inverse_valid = _native.rolling_update(
                        self._xtx,
                        self._xty,
                        self._inv,
                        self._params_cache,
                        old_row,
                        old_y,
                        old_weight,
                        row,
                        y_value,
                        weight,
                        _SHERMAN_MORRISON_MIN_DENOM,
                    )
                if inverse_valid:
                    self._inv_updates += int(old_weight != 0.0)
                    self._inv_updates += int(weight != 0.0)
                    self._dirty = False
                else:
                    self._invalidate_inverse()
            self._yty -= old_weight * old_y * old_y
            self._y_sum -= old_weight * old_y
            self.weight_sum_ -= old_weight
            self._buffer_start = (self._buffer_start + 1) % self.window

        self._row_buffer[insert_at] = row
        self._target_buffer[insert_at] = y_value
        self._weight_buffer[insert_at] = weight
        self._yty += weight * y_value * y_value
        self._y_sum += weight * y_value
        self.weight_sum_ += weight
        self._updates_since_recompute += 1
        if self._updates_since_recompute >= self.recompute_every:
            self.recompute()

    def _invalidate_inverse(self) -> None:
        self._inv = None
        self._params_cache = None
        self._inv_updates = 0
        self._dirty = True

    def _ordered_buffer_views(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._row_buffer is None or self._buffer_length == 0:
            raise SingularRegressionError("The rolling window is empty.")
        indices = (
            self._buffer_start + np.arange(self._buffer_length)
        ) % self.window
        return (
            np.ascontiguousarray(self._row_buffer[indices]),
            np.ascontiguousarray(self._target_buffer[indices]),
            np.ascontiguousarray(self._weight_buffer[indices]),
        )

    def recompute(self) -> "RankOneRollingOLS":
        self._require_stats()
        rows, targets, weights = self._ordered_buffer_views()
        (
            self._xtx,
            self._xty,
            self._yty,
            self._y_sum,
            self.weight_sum_,
        ) = _weighted_stats(rows, targets, weights)
        self._xtx = np.ascontiguousarray(self._xtx, dtype=float)
        self._xty = np.ascontiguousarray(self._xty, dtype=float)
        self.n_observations_ = self._buffer_length
        self._updates_since_recompute = 0
        self._invalidate_inverse()
        return self

    def current_window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_stats()
        rows, targets, weights = self._ordered_buffer_views()
        if self.fit_intercept:
            rows = rows[:, 1:]
        return rows.copy(), targets.copy(), weights.copy()
