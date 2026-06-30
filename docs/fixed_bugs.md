# Fixed Bugs

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
