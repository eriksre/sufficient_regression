"""Prototype: O(p^2)-per-step coefficients via incremental inverse maintenance.

The base estimators in :mod:`sufficient_regression.ols` maintain the sufficient
statistics ``XtX``, ``Xty`` and ``yty`` incrementally, but recover coefficients
with a fresh dense ``np.linalg.solve`` every time they are requested. That solve
is ``Theta(p^3)`` per call, so the performance note correctly observes that
"end-to-end wall-clock speedups are bounded by the unchanged dense solve cost
unless the implementation also updates or reuses a factorization"
(``sufficient_statistics_ols_performance.tex``).

This module prototypes the missing piece. It maintains the inverse of the
regularized Gram matrix

    A = XtX + R          (R is the fixed ridge penalty matrix)
    M = A^{-1}

directly, using rank-1 Sherman-Morrison updates. Appending a weighted row, or
dropping one from a rolling window, is a rank-1 (down)date of ``A`` and therefore
a rank-1 update of ``M`` costing ``Theta(p^2)``. Coefficients are then

    beta = M @ Xty

which is ``Theta(p^2)``, not ``Theta(p^3)``. This is the classic recursive least
squares / Sherman-Morrison trick behind ``statsmodels.RecursiveLS`` and friends.

Status
------
This is a prototype. The dense estimators in :mod:`ols` remain the reference
implementation and are exact up to a single solve's roundoff. Sherman-Morrison
updates accumulate additional floating-point error across a stream (downdates
especially), so these estimators:

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
- ``ForgettingOLS`` is intentionally not prototyped here: its ridge term is not
  decayed alongside ``XtX``, so the regularized system does not reduce to a clean
  single rank-1 recursion. Pure (ridge=0) forgetting would, but is left out to
  keep the prototype's invariants simple.
"""

from __future__ import annotations

import numpy as np

from .ols import (
    RollingOLS,
    SingularRegressionError,
    _SufficientOLSBase,
    _as_single_row,
    _as_single_value,
    _as_single_weight,
    _regularization_matrix,
    _validate_positive_integer,
    _weighted_stats,
)

# Downdates can drive the Sherman-Morrison denominator to zero when the removed
# row would make the regularized system singular. Below this magnitude we refuse
# the in-place update and rebuild instead; the rebuild raises loudly if the
# system really is singular. The threshold is on the dimensionless quantity
# ``1 + c * z^T M z`` (c = +/- weight), which is O(1) for well-posed systems.
_SHERMAN_MORRISON_MIN_DENOM = 1e-10


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

    The subclass is responsible for calling :meth:`_sherman_morrison` once per
    weighted row contribution (sign ``+1`` to add, ``-1`` to drop) and for
    setting ``self.refresh_every`` (``None`` disables the periodic exact
    rebuild).
    """

    def _reset_state(self) -> None:
        super()._reset_state()
        # ``None`` means "not currently maintained"; the next solve rebuilds it
        # from XtX. Rows added while it is None update only XtX (cheap and
        # exact), so the eventual rebuild is still correct.
        self._inv: np.ndarray | None = None
        self._inv_updates: int = 0

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
            self._inv = _invert_regularized_gram(self._regularized_gram())
            self._inv_updates = 0
        return self._inv

    def _sherman_morrison(self, z: np.ndarray, weight: float, sign: float) -> None:
        """Apply ``A -> A + sign * weight * z z^T`` to the maintained inverse.

        Sherman-Morrison: with ``c = sign * weight``, ``u = M z`` and
        ``denom = 1 + c * z^T u``,

            M <- M - (c / denom) * u u^T.

        For an add (``sign=+1``, weight>=0) ``denom >= 1`` always. For a drop the
        denominator can collapse if removing the row makes ``A`` singular; see the
        guard below.
        """

        if self._inv is None or weight == 0.0:
            return
        c = sign * float(weight)
        u = self._inv @ z
        denom = 1.0 + c * float(z @ u)
        if not np.isfinite(denom) or abs(denom) <= _SHERMAN_MORRISON_MIN_DENOM:
            # The (down)date drove the regularized system to (near-)singular.
            # Invalidate so the next solve rebuilds from XtX and raises loudly if
            # the system is genuinely singular, rather than propagating a blown-up
            # inverse silently.
            self._inv = None
            self._inv_updates = 0
            return
        self._inv = self._inv - (c / denom) * np.outer(u, u)
        self._inv_updates += 1
        if self.refresh_every is not None and self._inv_updates >= self.refresh_every:
            # Periodic exact rebuild caps Sherman-Morrison drift over long
            # streams. Deferred to the next solve by invalidating rather than
            # rebuilding here, so callers that never read coefficients never pay.
            self._inv = None
            self._inv_updates = 0

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
    """Prototype append-only OLS with O(p^2)-per-step coefficients.

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
        self._add_stats(
            *_weighted_stats(X_aug, y_arr, weights), n_rows=y_arr.shape[0]
        )
        # Update the maintained inverse one row at a time. Skipped while the
        # inverse is unbuilt (warm-up); the lazy rebuild from XtX stays exact.
        if self._inv is not None:
            for z, weight in zip(X_aug, weights, strict=True):
                self._sherman_morrison(z, float(weight), sign=1.0)
        return self

    append = partial_fit

    def push(
        self,
        x,
        y: float,
        *,
        sample_weight: float | None = None,
    ) -> "RankOneIncrementalOLS":
        return self.partial_fit(
            _as_single_row(x, name="x"),
            _as_single_value(y, name="y"),
            sample_weight=_as_single_weight(sample_weight),
        )


class RankOneRollingOLS(_RecursiveInverseMixin, RollingOLS):
    """Prototype fixed-window OLS with O(p^2)-per-slide coefficients.

    Each slide is one rank-1 downdate (departing row) and one rank-1 update
    (arriving row) of the maintained inverse, so a full ``params_`` read after
    every slide costs ``Theta(p^2)`` rather than a fresh ``Theta(p^3)`` solve.

    The inverse is rebuilt from ``XtX`` whenever the base recomputes its
    sufficient statistics (``recompute_every``, defaulting to one full window),
    so Sherman-Morrison drift is bounded by the same cadence that bounds add/drop
    roundoff in the dense estimator.
    """

    # The base RollingOLS already rebuilds XtX every recompute_every pushes;
    # tying the inverse rebuild to that cadence (via recompute()) is sufficient,
    # so no independent periodic refresh is needed here.
    refresh_every: int | None = None

    def _add_row_stats(self, z: np.ndarray, y_value: float, weight: float) -> None:
        super()._add_row_stats(z, y_value, weight)
        self._sherman_morrison(z, weight, sign=1.0)

    def _drop_row_stats(self, z: np.ndarray, y_value: float, weight: float) -> None:
        super()._drop_row_stats(z, y_value, weight)
        self._sherman_morrison(z, weight, sign=-1.0)

    def recompute(self) -> "RankOneRollingOLS":
        result = super().recompute()
        # XtX was just rebuilt exactly from the ring buffer; rebuild M from it on
        # the next solve so the maintained inverse cannot drift past one window.
        self._inv = None
        self._inv_updates = 0
        return result
