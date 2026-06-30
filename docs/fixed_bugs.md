# Fixed Bugs

## 2026-07-01

- Made `scripts/test.sh` bootstrap dev dependencies when an existing `.venv`
  lacks pytest.
- Added tracked setup and test scripts that create `.venv` when absent,
  activate it before Python commands, and install development dependencies.
- Added classical OLS coefficient covariance, standard errors, residual
  variance, and residual degrees-of-freedom properties with loud validation for
  ridge, rank-deficient, zero-weight, and nonpositive-degree cases.
- Changed `RollingOLS` to recompute sufficient statistics from the active ring
  buffer once per full window by default.
- Rejected row-vector and multi-column batch `y` targets instead of flattening
  them.

## 2026-06-30

- Fixed independent-oracle gaps in incremental streaming tests.
- Added coverage for ridge intercept regularization.
- Expanded diagnostics tests across weighted, no-intercept, rolling, forgetting,
  and zero-variance cases.
- Added long-stream reference checks for rolling and forgetting estimators.
- Added strict pytest configuration and coverage tooling.
- Made unfitted `intercept_` raise `NotFittedError` consistently.
- Changed `rank_` to report data Gram rank instead of regularized-system rank.
- Rejected zero-feature batch inputs.
- Made zero-effective-weight diagnostics return NaN diagnostics without solving.
- Honored `RollingOLS.recompute_every` during initial `fit`.
- Tightened rolling integer validation for `window` and `recompute_every`.
- Clarified README count semantics for `n_observations_`.
- Aligned the ridge cost statement in the performance note with the dense
  implementation.
