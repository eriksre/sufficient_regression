# Usage Examples

These examples show the two main streaming workflows the package is built for:
updating an existing regression without rereading old rows, and maintaining a
fixed-size rolling regression window.

## Choosing An Estimator

| Workflow | Estimator | Use when |
| --- | --- | --- |
| Append rows, coefficients read occasionally | `IncrementalOLS` | You want exact sufficient-statistic updates and a dense solve only when coefficients are requested. |
| Append rows, coefficients read after most updates | `RankOneIncrementalOLS` | You want recursive least squares style inverse maintenance so each update plus coefficient read is `Theta(p^2)` after the inverse is built. |
| Append rows, conservative factor maintenance | `CholeskyIncrementalOLS` | You want append-only rank-one Cholesky updates and strong numerical behavior. |
| Most-recent-N rolling regression | `RollingOLS` | You want exact add/drop sufficient statistics and can tolerate a dense solve when coefficients are requested. |
| Most-recent-N rolling regression, coefficients read after each slide | `RankOneRollingOLS` | You want add/drop inverse maintenance for the moving-window hot path. |
| Exponentially decayed history | `ForgettingOLS` | You want old rows to decay smoothly rather than fall out of a hard window. |

The rank-one inverse estimators maintain `M = (X.T @ X + R)^-1` directly. They
are the closest match for workloads where coefficients are read frequently after
each update. `RankOneRollingOLS` rebuilds from exact sufficient statistics on
the rolling recompute cadence, and `RankOneIncrementalOLS` does the same when
`refresh_every` is set. Both fail loudly on singular systems.

## Large Existing Fit Plus New Rows

Suppose you already have a large starting pool and want to append new rows
without refitting from raw historical data.

```python
import numpy as np

from sufficient_regression import IncrementalOLS

rng = np.random.default_rng(123)
X_initial = rng.normal(size=(1_000_000, 8))
true_beta = rng.normal(size=8)
y_initial = 2.0 + X_initial @ true_beta + rng.normal(size=1_000_000)

X_new = rng.normal(size=(1_000, 8))
y_new = 2.0 + X_new @ true_beta + rng.normal(size=1_000)

model = IncrementalOLS(ridge=1e-6)
model.fit(X_initial, y_initial)

# Only the 1,000 new rows are summarized here. The original 1,000,000 rows are
# not reread.
model.partial_fit(X_new, y_new)

params = model.params_
predictions = model.predict(X_new[:10])
```

`IncrementalOLS` is a good default when you update many rows and read
coefficients occasionally. It maintains `X.T @ X`, `X.T @ y`, and `y.T @ y`
exactly up to floating point roundoff.

## Append Stream With Frequent Coefficient Reads

If you read coefficients after nearly every new row, a fresh dense solve on each
read can dominate runtime. `RankOneIncrementalOLS` maintains the inverse once it
has been built.

```python
import numpy as np

from sufficient_regression import RankOneIncrementalOLS

rng = np.random.default_rng(456)
X_initial = rng.normal(size=(20_000, 12))
true_beta = rng.normal(size=12)
y_initial = 0.5 + X_initial @ true_beta + rng.normal(scale=0.1, size=20_000)

stream_X = rng.normal(size=(5_000, 12))
stream_y = 0.5 + stream_X @ true_beta + rng.normal(scale=0.1, size=5_000)

model = RankOneIncrementalOLS(ridge=1e-6)
model.fit(X_initial, y_initial)

# First coefficient read builds the inverse from the current sufficient stats.
current_params = model.params_

for x_row, y_value in zip(stream_X, stream_y, strict=True):
    model.push(x_row, float(y_value))
    # After the inverse exists, this read uses maintained inverse state instead
    # of a fresh dense normal-equation solve.
    current_params = model.params_
```

Use `refresh_every` if you want periodic exact inverse rebuilds in very long
append streams:

```python
model = RankOneIncrementalOLS(ridge=1e-6, refresh_every=10_000)
```

## Rolling Window Regression

For a fixed-size moving window, each incoming row evicts the oldest row once the
window is full. `RollingOLS` updates the sufficient statistics by subtracting the
old row contribution and adding the new row contribution.

```python
import numpy as np

from sufficient_regression import RollingOLS

rng = np.random.default_rng(789)
stream_X = rng.normal(size=(10_000, 6))
beta = rng.normal(size=6)
stream_y = 1.0 + stream_X @ beta + rng.normal(scale=0.2, size=10_000)

model = RollingOLS(window=1_000, ridge=1e-6)

for x_row, y_value in zip(stream_X, stream_y, strict=True):
    model.push(x_row, float(y_value))
    if model.window_size_ == model.window:
        window_params = model.params_

X_window, y_window, weights_window = model.current_window()
```

This is the exact fixed-window analogue of fitting on `stream_X[-1000:]`, but
without rebuilding the sufficient statistics from all 1,000 active rows on every
slide.

## Rolling Window With Frequent Coefficient Reads

If you read coefficients after every slide, use `RankOneRollingOLS` to maintain
the inverse across one row drop and one row add.

```python
import numpy as np

from sufficient_regression import RankOneRollingOLS

rng = np.random.default_rng(101112)
stream_X = rng.normal(size=(10_000, 6))
beta = rng.normal(size=6)
stream_y = 1.0 + stream_X @ beta + rng.normal(scale=0.2, size=10_000)

model = RankOneRollingOLS(window=1_000, ridge=1e-6)

for x_row, y_value in zip(stream_X, stream_y, strict=True):
    model.push(x_row, float(y_value))
    if model.window_size_ == model.window:
        # The first read builds the inverse; later slide-plus-read iterations
        # use the rank-one inverse-maintenance hot path.
        window_params = model.params_
```

`RankOneRollingOLS` rebuilds its maintained inverse whenever the base rolling
estimator recomputes exact sufficient statistics from the active ring buffer.
The default recompute cadence is one full window.

## Sample Weights

All estimators support nonnegative sample weights. A zero-weight row is accepted
and has no effect on the sufficient statistics.

```python
from sufficient_regression import RankOneRollingOLS

model = RankOneRollingOLS(window=500, ridge=1e-6)

for x_row, y_value, weight in weighted_stream:
    model.push(x_row, y_value, sample_weight=weight)
```

Weighted covariance treats `sample_weight` as effective observation mass.

## Classical Standard Errors

Classical OLS inference is available when `ridge=0`, the data Gram matrix is
full rank, and residual degrees of freedom are positive.

```python
from sufficient_regression import IncrementalOLS

model = IncrementalOLS(ridge=0.0).fit(X, y)

params = model.params_
standard_errors = model.standard_errors_
covariance = model.coefficient_covariance_
```

If those conditions are not met, the estimator raises a package exception rather
than silently returning a pseudoinverse-based answer.

## Handling Singular Systems

Unregularized systems can be singular during warm-up or when features are
collinear. The package fails loudly:

```python
from sufficient_regression import RankOneIncrementalOLS, SingularRegressionError

model = RankOneIncrementalOLS(ridge=0.0)
model.push([1.0, 2.0], 3.0)

try:
    params = model.params_
except SingularRegressionError:
    # Add more independent rows, remove collinear features, or use ridge > 0.
    pass
```

For online systems that may be underdetermined early in the stream, a small
ridge value is usually the cleanest way to request a regularized solution:

```python
model = RankOneIncrementalOLS(ridge=1e-6)
```

The implementation intentionally does not use a pseudoinverse fallback.
