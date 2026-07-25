# Fixed Bugs

## 2026-07-25

- Moved streaming-row finiteness validation out of the Python push path and into
  the native kernel, which already loads every element. The two NumPy
  `isfinite` scans it replaced cost more per row than the update arithmetic
  itself. Validation runs before any mutation, so a rejected row leaves state
  untouched.
- Replaced the per-push augmented row allocation with a reusable per-estimator
  buffer whose constant intercept entry is written once, and added a scalar
  `append_update_one` kernel entry point so the single-row append path no longer
  reshapes the row and allocates two length-one arrays per call.
- Removed the redundant per-element triangle averaging from
  `_symmetric_rank_one`; symmetry is established when the inverse is built and
  preserved because every update writes both triangles from one value.
- Added streaming validation coverage: non-finite features and targets on both
  estimators, state preservation after a rejected push, the wide NumPy/BLAS
  path's explicit validation, ring-buffer independence from the reused push
  buffer, and non-canonical row input forms.

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
- Made unregularized singularity detection explicit instead of relying on
  platform-specific `np.linalg.solve` failure behavior.
- Made the Linux binary support boundary explicit at manylinux x86_64 instead
  of accidentally building unclaimed 32-bit and musl variants.

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
