# sufficient-regression

Exact incremental and rolling ordinary least-squares regression using
sufficient statistics.

The package is built for cases where a batch OLS fit is correct but repeatedly
reprocessing historical rows is too expensive. It maintains:

```text
XtX = X.T @ X
Xty = X.T @ y
yty = y.T @ y
```

and solves the normal-equation system lazily when coefficients are requested.
For OLS and ridge regression, appending rows or adding/dropping rolling-window
rows is exact up to floating point roundoff.

## Estimators

- `IncrementalOLS`: append-only exact OLS / ridge updates.
- `RollingOLS`: fixed-window exact add/drop updates with an internal ring
  buffer.
- `ForgettingOLS`: exponentially weighted OLS for evolving processes.

This is not stochastic gradient descent. `partial_fit` and `push` update exact
sufficient statistics, not approximate optimizer state.

### Prototype: rank-1 inverse maintenance

- `RankOneIncrementalOLS` and `RankOneRollingOLS` additionally maintain the
  inverse of the regularized Gram matrix with Sherman–Morrison rank-1 updates
  (classical recursive least squares). Coefficients become a `Theta(p^2)`
  matrix–vector product instead of a fresh `Theta(p^3)` dense solve on every
  read, so appending a row or sliding a window and then reading coefficients is
  `Theta(p^2)` per step. They reproduce the dense estimators' coefficients,
  covariance, and loud-failure behavior, rebuilding the inverse exactly on a
  bounded cadence to cap Sherman–Morrison drift. See
  [`scripts/bench_rank_one.py`](scripts/bench_rank_one.py) and the rank-1
  section of the
  [performance note](docs/math/sufficient_statistics_ols_performance.pdf).

## Examples

Append-only regression:

```python
from sufficient_regression import IncrementalOLS

model = IncrementalOLS(fit_intercept=True)
model.fit(X_initial, y_initial)
model.partial_fit(X_new, y_new)

predictions = model.predict(X_test)
coefficients = model.coef_
intercept = model.intercept_
standard_errors = model.standard_errors_
covariance = model.coefficient_covariance_
```

Rolling-window regression:

```python
from sufficient_regression import RollingOLS

model = RollingOLS(window=1000, ridge=1e-6)

for x_row, y_value in stream:
    model.push(x_row, y_value)
    current_beta = model.params_
```

Exponential forgetting:

```python
from sufficient_regression import ForgettingOLS

model = ForgettingOLS(decay=0.995)
model.partial_fit(X_batch, y_batch)
```

## Design Notes

- `params_` is `[intercept, *coef_]` when `fit_intercept=True`.
- `coefficient_covariance_` is the classical homoskedastic OLS covariance for
  `params_`, and `standard_errors_` is aligned with `params_`. Both require
  `ridge=0`, full-rank sufficient statistics, and positive residual degrees of
  freedom.
- Ridge does not penalize the intercept by default, matching common ML-library
  behavior. Set `regularize_intercept=True` to include it.
- Missing, NaN, infinite, negative-weight, shape-mismatch, and singular
  unregularized systems fail loudly.
- Weighted covariance treats `sample_weight` as effective observation mass.
  Heteroskedasticity-robust standard errors are intentionally not exposed
  because the estimators do not retain row-level residuals or leverage values.
- `IncrementalOLS.n_observations_` and `ForgettingOLS.n_observations_` count
  raw rows processed. `RollingOLS.n_observations_` counts active-window rows.
  `weight_sum_` is the current effective weighted sample mass, so it decays in
  `ForgettingOLS`.
- `RollingOLS.fit` on more rows than the window retains only the final window.
- `RollingOLS` recomputes sufficient statistics from its active ring buffer
  once per full window by default, bounding add/drop roundoff in long streams.
- The implementation intentionally does not use a pseudoinverse fallback.
  Singular systems raise `SingularRegressionError`; use ridge when you want a
  regularized solution.

## Math Documents

- [Correctness proof](docs/math/sufficient_statistics_ols_correctness.pdf)
- [Performance analysis](docs/math/sufficient_statistics_ols_performance.pdf)

## Development

Bootstrap a fresh workspace with a local virtual environment and editable dev
install:

```bash
./scripts/setup-dev.sh
```

Run tests with:

```bash
./scripts/test.sh
```

The scripts create `.venv` when it is absent, then activate it before running
Python. The `.venv` directory is intentionally gitignored and should not be
committed.
