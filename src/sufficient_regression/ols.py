"""Exact online ordinary least squares using sufficient statistics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Deque, Literal, Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray
MissingPolicy = Literal["raise"]


class RegressionError(Exception):
    """Base exception for this package."""


class NotFittedError(RegressionError):
    """Raised when an estimator needs fitted state but has none."""


class SingularRegressionError(RegressionError):
    """Raised when the normal-equation system cannot be solved."""


@dataclass(frozen=True)
class RegressionDiagnostics:
    """Small set of exact diagnostics derived from maintained statistics."""

    n_observations: int
    weight_sum: float
    rank: int
    residual_sum_squares: float
    weighted_mean_y: float
    total_sum_squares: float
    r_squared: float


def _as_2d_float_array(X: ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_1d_float_array(y: ArrayLike, *, name: str, n_rows: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    # A 2D target is only valid as an explicit single column; flattening row
    # vectors would silently reinterpret columns as observations.
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.reshape(-1)
    elif arr.ndim == 2 and arr.shape[0] == 1:
        raise ValueError(
            f"{name} must not be a row-vector target; use shape (n,) or (n, 1)."
        )
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar, 1D array, or single-column 2D array."
        )
    if arr.shape[0] != n_rows:
        raise ValueError(
            f"{name} must have {n_rows} values to match X; got {arr.shape[0]}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_single_row(X: ArrayLike, *, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2 and arr.shape[0] == 1:
        pass
    else:
        raise ValueError(f"{name} must be one row.")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_single_value(y: float, *, name: str) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 1 and arr.shape[0] == 1:
        pass
    else:
        raise ValueError(f"{name} must be one scalar value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _as_single_weight(sample_weight: float | None) -> np.ndarray | None:
    if sample_weight is None:
        return None
    arr = np.asarray(sample_weight, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1 and arr.shape[0] == 1:
        return arr
    raise ValueError("sample_weight must be one scalar value for push.")


def _as_weights(sample_weight: ArrayLike | None, *, n_rows: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_rows, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim != 1:
        raise ValueError("sample_weight must be a 1D array.")
    if weights.shape[0] != n_rows:
        raise ValueError(
            "sample_weight must have one value per row; "
            f"expected {n_rows}, got {weights.shape[0]}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("sample_weight cannot contain negative values.")
    return weights


def _validate_ridge(ridge: float) -> float:
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be a finite nonnegative value.")
    return ridge


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _augment_intercept(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    if not fit_intercept:
        return X
    return np.column_stack((np.ones(X.shape[0], dtype=float), X))


def _weighted_stats(
    X_aug: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    weighted_X = X_aug * weights[:, None]
    xtx = X_aug.T @ weighted_X
    xty = X_aug.T @ (weights * y)
    yty = float(np.dot(weights * y, y))
    y_sum = float(np.dot(weights, y))
    weight_sum = float(np.sum(weights))
    return xtx, xty, yty, y_sum, weight_sum


def _regularization_matrix(
    n_params: int, *, ridge: float, fit_intercept: bool, regularize_intercept: bool
) -> np.ndarray:
    if ridge == 0:
        return np.zeros((n_params, n_params), dtype=float)
    diagonal = np.ones(n_params, dtype=float)
    # Matching common ML-library behavior: the intercept represents a location
    # shift, so it is not penalized unless the caller explicitly opts in.
    if fit_intercept and not regularize_intercept:
        diagonal[0] = 0.0
    return np.diag(ridge * diagonal)


def _require_full_rank_unregularized(
    xtx: np.ndarray,
    *,
    n_params: int,
    ridge: float,
) -> None:
    """Fail explicitly when an unregularized normal system is rank deficient.

    ``np.linalg.solve`` is not a portable singularity detector: some LAPACK
    builds return an unstable value for exactly dependent columns. A cold solve
    is already cubic, so an explicit rank check preserves loud failure semantics
    without changing the hot recursive-update complexity.
    """

    if ridge == 0 and np.linalg.matrix_rank(xtx) != n_params:
        raise SingularRegressionError(
            "The regression system is singular. Add more independent rows, "
            "remove collinear features, or use ridge > 0."
        )


def _rank_one_cholesky_update(lower: np.ndarray, update: np.ndarray) -> np.ndarray:
    """Return the Cholesky factor for ``lower @ lower.T + update update.T``."""

    updated = np.array(lower, dtype=float, copy=True)
    vector = np.array(update, dtype=float, copy=True)
    if updated.ndim != 2 or updated.shape[0] != updated.shape[1]:
        raise ValueError("lower must be a square matrix.")
    if vector.ndim != 1 or vector.shape[0] != updated.shape[0]:
        raise ValueError("update must be a vector matching lower.")

    for index in range(vector.shape[0]):
        diagonal = updated[index, index]
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            raise SingularRegressionError(
                "Cannot rank-one update a non-positive Cholesky factor."
            )
        radius = np.hypot(diagonal, vector[index])
        cosine = radius / diagonal
        sine = vector[index] / diagonal
        updated[index, index] = radius
        if index + 1 < vector.shape[0]:
            updated[index + 1 :, index] = (
                updated[index + 1 :, index] + sine * vector[index + 1 :]
            ) / cosine
            vector[index + 1 :] = (
                cosine * vector[index + 1 :]
                - sine * updated[index + 1 :, index]
            )
    return updated


def _solve_lower_triangular(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    solution = np.empty_like(rhs, dtype=float)
    for row in range(lower.shape[0]):
        diagonal = lower[row, row]
        if not np.isfinite(diagonal) or diagonal == 0.0:
            raise SingularRegressionError("Cannot solve with a singular factor.")
        solution[row] = (
            rhs[row] - float(lower[row, :row] @ solution[:row])
        ) / diagonal
    return solution


def _solve_upper_from_lower_transpose(
    lower: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    solution = np.empty_like(rhs, dtype=float)
    for row in range(lower.shape[0] - 1, -1, -1):
        diagonal = lower[row, row]
        if not np.isfinite(diagonal) or diagonal == 0.0:
            raise SingularRegressionError("Cannot solve with a singular factor.")
        solution[row] = (
            rhs[row] - float(lower[row + 1 :, row] @ solution[row + 1 :])
        ) / diagonal
    return solution


def _solve_cholesky_factor(lower: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    intermediate = _solve_lower_triangular(lower, rhs)
    return _solve_upper_from_lower_transpose(lower, intermediate)


class _SufficientOLSBase:
    """Shared solving and validation code for sufficient-stat estimators."""

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        ridge: float = 0.0,
        regularize_intercept: bool = False,
        missing: MissingPolicy = "raise",
    ) -> None:
        if missing != "raise":
            raise ValueError("Only missing='raise' is supported.")
        self.fit_intercept = bool(fit_intercept)
        self.ridge = _validate_ridge(ridge)
        self.regularize_intercept = bool(regularize_intercept)
        self.missing = missing
        self._reset_state()

    def _reset_state(self) -> None:
        self.n_features_in_: int | None = None
        self.n_params_: int | None = None
        self.n_observations_: int = 0
        self.weight_sum_: float = 0.0
        self._xtx: np.ndarray | None = None
        self._xty: np.ndarray | None = None
        self._yty: float = 0.0
        self._y_sum: float = 0.0
        self._params_cache: np.ndarray | None = None
        self._dirty: bool = True

    @property
    def xtx_(self) -> np.ndarray:
        self._require_stats()
        return self._xtx.copy()

    @property
    def xty_(self) -> np.ndarray:
        self._require_stats()
        return self._xty.copy()

    @property
    def yty_(self) -> float:
        self._require_stats()
        return float(self._yty)

    def _require_stats(self) -> None:
        if self._xtx is None or self._xty is None or self.n_params_ is None:
            raise NotFittedError("This estimator has not been fitted yet.")

    def _validate_X_y(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None,
        *,
        reset: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_arr = _as_2d_float_array(X, name="X")
        y_arr = _as_1d_float_array(y, name="y", n_rows=X_arr.shape[0])
        weights = _as_weights(sample_weight, n_rows=X_arr.shape[0])
        if reset or self.n_features_in_ is None:
            self.n_features_in_ = X_arr.shape[1]
            self.n_params_ = self.n_features_in_ + int(self.fit_intercept)
        elif X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted state; "
                f"expected {self.n_features_in_}, got {X_arr.shape[1]}."
            )
        X_aug = _augment_intercept(X_arr, self.fit_intercept)
        return X_arr, X_aug, y_arr, weights

    def _add_stats(
        self,
        xtx: np.ndarray,
        xty: np.ndarray,
        yty: float,
        y_sum: float,
        weight_sum: float,
        n_rows: int,
    ) -> None:
        self._require_stats()
        self._xtx += xtx
        self._xty += xty
        self._yty += yty
        self._y_sum += y_sum
        self.weight_sum_ += weight_sum
        self.n_observations_ += int(n_rows)
        self._dirty = True

    def _subtract_stats(
        self,
        xtx: np.ndarray,
        xty: np.ndarray,
        yty: float,
        y_sum: float,
        weight_sum: float,
        n_rows: int,
    ) -> None:
        self._require_stats()
        self._xtx -= xtx
        self._xty -= xty
        self._yty -= yty
        self._y_sum -= y_sum
        self.weight_sum_ -= weight_sum
        self.n_observations_ -= int(n_rows)
        self._dirty = True

    def _regularized_system(self) -> np.ndarray:
        self._require_stats()
        return self._xtx + _regularization_matrix(
            self.n_params_,
            ridge=self.ridge,
            fit_intercept=self.fit_intercept,
            regularize_intercept=self.regularize_intercept,
        )

    def _solve_params(self) -> np.ndarray:
        self._require_stats()
        if self.weight_sum_ <= 0:
            raise SingularRegressionError("Cannot solve with zero total sample weight.")
        _require_full_rank_unregularized(
            self._xtx,
            n_params=self.n_params_,
            ridge=self.ridge,
        )
        system = self._regularized_system()
        try:
            return np.linalg.solve(system, self._xty)
        except np.linalg.LinAlgError as exc:
            raise SingularRegressionError(
                "The regression system is singular. Add more independent rows, "
                "remove collinear features, or use ridge > 0."
            ) from exc

    @property
    def params_(self) -> np.ndarray:
        if self._dirty or self._params_cache is None:
            self._params_cache = self._solve_params()
            self._dirty = False
        return self._params_cache.copy()

    @property
    def coef_(self) -> np.ndarray:
        params = self.params_
        if self.fit_intercept:
            return params[1:].copy()
        return params.copy()

    @property
    def intercept_(self) -> float:
        self._require_stats()
        if self.fit_intercept:
            return float(self.params_[0])
        return 0.0

    def predict(self, X: ArrayLike) -> np.ndarray:
        self._require_stats()
        X_arr = _as_2d_float_array(X, name="X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted state; "
                f"expected {self.n_features_in_}, got {X_arr.shape[1]}."
            )
        X_aug = _augment_intercept(X_arr, self.fit_intercept)
        return X_aug @ self.params_

    @property
    def residual_sum_squares_(self) -> float:
        beta = self.params_
        rss = self._yty - 2.0 * float(beta @ self._xty) + float(beta @ self._xtx @ beta)
        return float(max(rss, 0.0))

    @property
    def rank_(self) -> int:
        self._require_stats()
        return int(np.linalg.matrix_rank(self._xtx))

    def _require_classical_ols_inference(self) -> int:
        self._require_stats()
        if self.ridge != 0:
            raise RegressionError(
                "Classical OLS coefficient covariance is only defined when ridge=0."
            )
        if self.weight_sum_ <= 0:
            raise SingularRegressionError(
                "Cannot estimate coefficient covariance with zero total sample weight."
            )
        rank = self.rank_
        if rank != self.n_params_:
            raise SingularRegressionError(
                "Cannot estimate coefficient covariance for a rank-deficient "
                "regression system."
            )
        return rank

    @property
    def residual_degrees_of_freedom_(self) -> float:
        """Residual degrees of freedom used by classical OLS inference."""

        rank = self._require_classical_ols_inference()
        # sample_weight is treated as effective observation mass throughout the
        # estimator, so the uncertainty calculation uses the same mass rather
        # than the raw row count.
        degrees_of_freedom = float(self.weight_sum_ - rank)
        if degrees_of_freedom <= 0:
            raise SingularRegressionError(
                "Cannot estimate residual variance with nonpositive residual "
                "degrees of freedom."
            )
        return degrees_of_freedom

    @property
    def residual_variance_(self) -> float:
        """Unbiased classical OLS estimate of sigma squared."""

        degrees_of_freedom = self.residual_degrees_of_freedom_
        return self.residual_sum_squares_ / degrees_of_freedom

    @property
    def coefficient_covariance_(self) -> np.ndarray:
        """Classical covariance matrix for ``params_``.

        The returned matrix is aligned with ``params_``: the intercept is first
        when ``fit_intercept=True``. This is the homoskedastic OLS covariance
        ``sigma^2 * (X.T @ X)^-1`` computed from the maintained sufficient
        statistics.
        """

        sigma_squared = self.residual_variance_
        try:
            inverse_xtx = np.linalg.solve(
                self._xtx,
                np.eye(self.n_params_, dtype=float),
            )
        except np.linalg.LinAlgError as exc:
            raise SingularRegressionError(
                "Cannot estimate coefficient covariance for a singular "
                "regression system."
            ) from exc
        covariance = sigma_squared * inverse_xtx
        return np.array(covariance, dtype=float, copy=True)

    @property
    def standard_errors_(self) -> np.ndarray:
        """Classical standard errors aligned with ``params_``."""

        variances = np.diag(self.coefficient_covariance_)
        if np.any(variances < 0):
            raise SingularRegressionError(
                "Coefficient covariance has negative diagonal variances."
            )
        return np.sqrt(variances)

    @property
    def diagnostics_(self) -> RegressionDiagnostics:
        self._require_stats()
        if self.weight_sum_ <= 0:
            weighted_mean = float("nan")
            tss = float("nan")
            rss = float("nan")
            r_squared = float("nan")
        else:
            weighted_mean = self._y_sum / self.weight_sum_
            tss = self._yty - (self._y_sum * self._y_sum / self.weight_sum_)
            tss = float(max(tss, 0.0))
            rss = self.residual_sum_squares_
            r_squared = float("nan") if tss == 0 else 1.0 - rss / tss
        return RegressionDiagnostics(
            n_observations=self.n_observations_,
            weight_sum=float(self.weight_sum_),
            rank=self.rank_,
            residual_sum_squares=float(rss),
            weighted_mean_y=float(weighted_mean),
            total_sum_squares=float(tss),
            r_squared=float(r_squared),
        )


class IncrementalOLS(_SufficientOLSBase):
    """Append-only exact ordinary least squares.

    The estimator stores only sufficient statistics, not historical rows. Calling
    :meth:`partial_fit` adds new rows exactly as if the model had been refit on
    the concatenated dataset, up to floating point roundoff.
    """

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "IncrementalOLS":
        self._reset_state()
        _, X_aug, y_arr, weights = self._validate_X_y(
            X, y, sample_weight, reset=True
        )
        self._xtx, self._xty, self._yty, self._y_sum, self.weight_sum_ = _weighted_stats(
            X_aug, y_arr, weights
        )
        self.n_observations_ = int(y_arr.shape[0])
        self._dirty = True
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "IncrementalOLS":
        if self.n_features_in_ is None:
            return self.fit(X, y, sample_weight=sample_weight)
        _, X_aug, y_arr, weights = self._validate_X_y(
            X, y, sample_weight, reset=False
        )
        self._add_stats(*_weighted_stats(X_aug, y_arr, weights), n_rows=y_arr.shape[0])
        return self

    append = partial_fit

    def push(
        self,
        x: ArrayLike,
        y: float,
        *,
        sample_weight: float | None = None,
    ) -> "IncrementalOLS":
        return self.partial_fit(
            _as_single_row(x, name="x"),
            _as_single_value(y, name="y"),
            sample_weight=_as_single_weight(sample_weight),
        )


class CholeskyIncrementalOLS(IncrementalOLS):
    """Append-only OLS with an incrementally maintained Cholesky factor.

    This prototype keeps the same exact sufficient statistics as
    :class:`IncrementalOLS`. After the first successful coefficient solve, it
    updates the Cholesky factor of the regularized normal-equation system with
    one rank-one update per appended row, reducing the post-factorization
    coefficient path from a dense solve to triangular solves.
    """

    def _reset_state(self) -> None:
        super()._reset_state()
        self._cholesky_factor: np.ndarray | None = None

    @property
    def cholesky_factor_(self) -> np.ndarray:
        """Lower Cholesky factor of the current regularized solve system."""

        return self._ensure_cholesky_factor().copy()

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "CholeskyIncrementalOLS":
        if self.n_features_in_ is None:
            return self.fit(X, y, sample_weight=sample_weight)
        _, X_aug, y_arr, weights = self._validate_X_y(
            X, y, sample_weight, reset=False
        )
        updated_factor = self._updated_factor_for_batch(X_aug, weights)
        self._add_stats(*_weighted_stats(X_aug, y_arr, weights), n_rows=y_arr.shape[0])
        if updated_factor is not None:
            self._cholesky_factor = updated_factor
        return self

    append = partial_fit

    def _ensure_cholesky_factor(self) -> np.ndarray:
        self._require_stats()
        if self._cholesky_factor is None:
            try:
                self._cholesky_factor = np.linalg.cholesky(
                    self._regularized_system()
                )
            except np.linalg.LinAlgError as exc:
                raise SingularRegressionError(
                    "The regression system is singular or not positive definite. "
                    "Add more independent rows, remove collinear features, or "
                    "use ridge > 0."
                ) from exc
        return self._cholesky_factor

    def _updated_factor_for_batch(
        self, X_aug: np.ndarray, weights: np.ndarray
    ) -> np.ndarray | None:
        if self._cholesky_factor is None:
            return None
        factor = np.array(self._cholesky_factor, dtype=float, copy=True)
        for z, weight in zip(X_aug, weights, strict=True):
            if weight == 0.0:
                continue
            # The ridge penalty is fixed after initialization, so each
            # appended weighted row changes the solve system by exactly uu^T.
            update = np.sqrt(weight) * z
            factor = _rank_one_cholesky_update(factor, update)
        return factor

    def _solve_params(self) -> np.ndarray:
        self._require_stats()
        if self.weight_sum_ <= 0:
            raise SingularRegressionError("Cannot solve with zero total sample weight.")
        return _solve_cholesky_factor(self._ensure_cholesky_factor(), self._xty)


class RollingOLS(_SufficientOLSBase):
    """Exact fixed-window ordinary least squares with add/drop updates."""

    def __init__(
        self,
        window: int,
        *,
        fit_intercept: bool = True,
        ridge: float = 0.0,
        regularize_intercept: bool = False,
        recompute_every: int | None = None,
        missing: MissingPolicy = "raise",
    ) -> None:
        self.window = _validate_positive_integer(window, name="window")
        if recompute_every is not None:
            self.recompute_every = _validate_positive_integer(
                recompute_every,
                name="recompute_every",
            )
        else:
            # Rebuilding once per full window bounds add/drop roundoff without
            # asking stream callers to choose a numerical drift guard up front.
            self.recompute_every = self.window
        self._buffer: Deque[tuple[np.ndarray, float, float]] = deque()
        self._updates_since_recompute = 0
        super().__init__(
            fit_intercept=fit_intercept,
            ridge=ridge,
            regularize_intercept=regularize_intercept,
            missing=missing,
        )

    def _reset_state(self) -> None:
        super()._reset_state()
        self._buffer = deque()
        self._updates_since_recompute = 0

    @property
    def window_size_(self) -> int:
        return len(self._buffer)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "RollingOLS":
        self._reset_state()
        _, X_aug, y_arr, weights = self._validate_X_y(
            X, y, sample_weight, reset=True
        )
        self._xtx = np.zeros((self.n_params_, self.n_params_), dtype=float)
        self._xty = np.zeros(self.n_params_, dtype=float)
        self._yty = 0.0
        self._y_sum = 0.0
        self.weight_sum_ = 0.0
        self.n_observations_ = 0
        for z, y_value, weight in zip(X_aug, y_arr, weights, strict=True):
            self._push_prepared(z, float(y_value), float(weight))
        self._dirty = True
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "RollingOLS":
        if self.n_features_in_ is None:
            return self.fit(X, y, sample_weight=sample_weight)
        _, X_aug, y_arr, weights = self._validate_X_y(
            X, y, sample_weight, reset=False
        )
        for z, y_value, weight in zip(X_aug, y_arr, weights, strict=True):
            self._push_prepared(z, float(y_value), float(weight))
        return self

    append = partial_fit

    def push(
        self,
        x: ArrayLike,
        y: float,
        *,
        sample_weight: float | None = None,
    ) -> "RollingOLS":
        return self.partial_fit(
            _as_single_row(x, name="x"),
            _as_single_value(y, name="y"),
            sample_weight=_as_single_weight(sample_weight),
        )

    def recompute(self) -> "RollingOLS":
        self._require_stats()
        self._xtx = np.zeros((self.n_params_, self.n_params_), dtype=float)
        self._xty = np.zeros(self.n_params_, dtype=float)
        self._yty = 0.0
        self._y_sum = 0.0
        self.weight_sum_ = 0.0
        self.n_observations_ = 0
        for z, y_value, weight in self._buffer:
            self._add_row_stats(z, y_value, weight)
        self._updates_since_recompute = 0
        self._dirty = True
        return self

    def current_window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return copies of the current unaugmented X, y, and weights."""

        self._require_stats()
        if not self._buffer:
            raise NotFittedError("The rolling window is empty.")
        rows = []
        y_values = []
        weights = []
        for z, y_value, weight in self._buffer:
            rows.append(z[1:] if self.fit_intercept else z)
            y_values.append(y_value)
            weights.append(weight)
        return np.vstack(rows), np.asarray(y_values), np.asarray(weights)

    def _push_prepared(self, z: np.ndarray, y_value: float, weight: float) -> None:
        self._require_stats()
        if len(self._buffer) == self.window:
            old_z, old_y, old_weight = self._buffer.popleft()
            self._drop_row_stats(old_z, old_y, old_weight)
        z_copy = np.array(z, dtype=float, copy=True)
        self._buffer.append((z_copy, y_value, weight))
        self._add_row_stats(z_copy, y_value, weight)
        self._updates_since_recompute += 1
        if self._updates_since_recompute >= self.recompute_every:
            self.recompute()
        self._dirty = True

    def _row_stats(
        self, z: np.ndarray, y_value: float, weight: float
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        xtx = weight * np.outer(z, z)
        xty = weight * z * y_value
        yty = weight * y_value * y_value
        y_sum = weight * y_value
        return xtx, xty, float(yty), float(y_sum), float(weight)

    def _add_row_stats(self, z: np.ndarray, y_value: float, weight: float) -> None:
        self._add_stats(*self._row_stats(z, y_value, weight), n_rows=1)

    def _drop_row_stats(self, z: np.ndarray, y_value: float, weight: float) -> None:
        self._subtract_stats(*self._row_stats(z, y_value, weight), n_rows=1)


class ForgettingOLS(_SufficientOLSBase):
    """Exact exponentially weighted OLS for sequential data."""

    def __init__(
        self,
        decay: float,
        *,
        fit_intercept: bool = True,
        ridge: float = 0.0,
        regularize_intercept: bool = False,
        missing: MissingPolicy = "raise",
    ) -> None:
        decay = float(decay)
        if not np.isfinite(decay) or not (0.0 < decay <= 1.0):
            raise ValueError("decay must be finite and in the interval (0, 1].")
        self.decay = decay
        super().__init__(
            fit_intercept=fit_intercept,
            ridge=ridge,
            regularize_intercept=regularize_intercept,
            missing=missing,
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "ForgettingOLS":
        self._reset_state()
        _, X_aug, y_arr, weights = self._validate_X_y(
            X, y, sample_weight, reset=True
        )
        self._xtx = np.zeros((self.n_params_, self.n_params_), dtype=float)
        self._xty = np.zeros(self.n_params_, dtype=float)
        self._yty = 0.0
        self._y_sum = 0.0
        self.weight_sum_ = 0.0
        self.n_observations_ = 0
        self._add_decayed_batch(X_aug, y_arr, weights)
        self._dirty = True
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
    ) -> "ForgettingOLS":
        if self.n_features_in_ is None:
            return self.fit(X, y, sample_weight=sample_weight)
        _, X_aug, y_arr, weights = self._validate_X_y(
            X, y, sample_weight, reset=False
        )
        self._add_decayed_batch(X_aug, y_arr, weights)
        self._dirty = True
        return self

    append = partial_fit

    def push(
        self,
        x: ArrayLike,
        y: float,
        *,
        sample_weight: float | None = None,
    ) -> "ForgettingOLS":
        return self.partial_fit(
            _as_single_row(x, name="x"),
            _as_single_value(y, name="y"),
            sample_weight=_as_single_weight(sample_weight),
        )

    def _add_decayed_batch(
        self, X_aug: np.ndarray, y: np.ndarray, weights: np.ndarray
    ) -> None:
        self._require_stats()
        n_rows = y.shape[0]
        self._xtx *= self.decay**n_rows
        self._xty *= self.decay**n_rows
        self._yty *= self.decay**n_rows
        self._y_sum *= self.decay**n_rows
        self.weight_sum_ *= self.decay**n_rows

        row_decay = self.decay ** np.arange(n_rows - 1, -1, -1, dtype=float)
        effective_weights = weights * row_decay
        xtx, xty, yty, y_sum, weight_sum = _weighted_stats(
            X_aug, y, effective_weights
        )
        self._xtx += xtx
        self._xty += xty
        self._yty += yty
        self._y_sum += y_sum
        self.weight_sum_ += weight_sum
        self.n_observations_ += int(n_rows)
