# Fixed Bugs

## 2026-07-24

- Replaced fragmented Python/NumPy rank-one streaming transitions with a fused
  compiled kernel that maintains sufficient statistics, inverse state, and
  coefficients together.
- Replaced the rolling estimator's deque of per-row arrays with contiguous ring
  storage and vectorized numerical refreshes.
- Added the missing distribution-build dependency and cross-platform native
  wheel build/test automation.
- Included the native Cython source in source distributions and added an
  install check for the built archive.
- Restricted Windows wheel builds to AMD64 after the 32-bit test environment
  failed to resolve supported scientific reference dependencies.

## 2026-07-01

- Invalidated `RankOneRollingOLS`'s maintained inverse before recomputing
  rolling sufficient statistics, preventing stale Sherman-Morrison updates
  during ring-buffer replay.
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
