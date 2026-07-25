# Bugs And Review Findings

## Open

- The `p >= 256` NumPy/BLAS update path costs roughly 100 us per append and 200
  us per rolling slide, against 12 us and 22 us at `p = 128`. That is a ~9x jump
  for a 2x width increase, worse than the `Theta(p^2)` the path is supposed to
  achieve, because each row is processed through Python-level `np.outer`
  allocations instead of the fused kernel. Calling BLAS `dsymv`/`dsyr` from
  inside the Cython kernel would remove the cliff and the separate code path.
- The native kernel rejects a Sherman-Morrison downdate on
  `denominator <= min_denominator`, but the NumPy/BLAS path at `p >= 256`
  rejects on `abs(denominator) <= min_denominator`. A negative denominator
  (a downdate that makes the regularized system indefinite) is therefore
  refused by one path and applied by the other. Both should use the native
  kernel's one-sided test.
- `_require_full_rank_unregularized` is gated on `ridge == 0`, so any nonzero
  ridge disables the only conditioning guard. On badly scaled designs the
  estimators then return coefficients with double-digit relative error and no
  exception. `docs/usage.md` recommends `ridge=1e-6`, which walks users into
  this gap.
- `_ensure_inverse` runs its rank check only when the maintained inverse is
  absent, so `RankOneIncrementalOLS` can return coefficients on a rank-deficient
  system where `IncrementalOLS` raises. Whether the caller gets an exception
  depends on refresh phase.
- Nothing in the README or the math notes states that forming `XtX` squares the
  condition number, and `diagnostics_` exposes `rank` but no conditioning
  measure, so a caller cannot detect the regime where results silently degrade.

## Review Findings Closed In This Build

- Fixed: single-row `push` validated every streaming row with two
  `np.all(np.isfinite(...))` scans, which cost about 1.73 us per row at `p = 8`
  against 1.00 us for the entire fused kernel update. Validation is 1.7x the
  mathematics it guarded. Finiteness is now checked inside the kernel, which
  already loads every element, and always before any state is mutated so a
  rejected row leaves the estimator untouched. Wide systems skip the kernel and
  are validated explicitly.
- Fixed: `push` allocated a fresh augmented row per call and, on the append
  path, reshaped it and allocated two length-one arrays purely to satisfy the
  batch kernel signature. Rows are now staged in a reusable per-estimator
  buffer with the constant intercept written once, and a scalar
  `append_update_one` kernel entry point replaces the batch call.
- Fixed: `_symmetric_rank_one` re-averaged both triangles of the maintained
  inverse on every element, but symmetry is already a state invariant: the
  freshly built inverse is projected to exact symmetry and every update writes
  both triangles from one value. The averaging was a no-op costing an extra
  load, add and multiply in the innermost loop of the hot path.
- Fixed: the rank-one estimators reduced arithmetic complexity to
  `Theta(p^2)` but executed each row through dozens of Python/NumPy calls and
  temporary arrays, making 1,000-row rolling windows slower than vectorized
  full refits. The production path now fuses statistics, inverse, and direct
  coefficient updates in a compiled kernel and stores rolling rows contiguously.
- Fixed: the development environment did not include the `build` frontend
  needed to verify wheel and source distributions. It is now a development
  dependency, and native wheels are built and tested in CI.
- Fixed: the first native source distribution omitted `_native.pyx`, so an
  installation from that archive could not compile the required extension. The
  Cython source is now explicitly included and the built archive is install-tested.
- Fixed: the first Windows wheel matrix attempted unsupported 32-bit CPython
  builds, where the scientific test dependencies do not publish compatible
  wheels. Windows production artifacts are now explicitly AMD64.
- Fixed: Windows LAPACK could return an unstable value instead of raising for
  an exactly rank-deficient unregularized system. Cold unregularized solves now
  perform an explicit rank check, making loud singular failure cross-platform.
- Fixed: the initial Linux wheel configuration implicitly expanded into
  32-bit and musl artifacts that were outside the supported release targets.
  Linux wheel support is now explicit: manylinux x86_64 on CPython 3.10-3.14.
- Fixed: `RankOneRollingOLS.recompute()` invalidated the maintained inverse only
  after replaying buffered rows, so recompute could apply Sherman-Morrison
  updates to a stale inverse before discarding it. It now invalidates before
  rebuilding sufficient statistics.
- Fixed: fresh workspaces lacked the gitignored `.venv` expected by the
  documented development workflow. Tracked setup and test scripts now create
  and activate `.venv` before Python commands.
- Fixed: `scripts/test.sh` trusted any readable `.venv` as ready for tests, so
  an existing but unbootstrapped environment failed with `No module named
  pytest`. It now installs dev dependencies when pytest is absent.
- Fixed: OLS models exposed coefficients and sufficient statistics but did not
  expose classical coefficient covariance or standard errors. The estimators
  now provide loud classical-OLS-only inference properties.
- Fixed: `RollingOLS` defaulted `recompute_every` to `None`, disabling any
  automatic drift guard for subtractive add/drop streams. It now recomputes from
  the active ring buffer once per full window by default.
- Fixed: batch `y` validation flattened row-vector targets shaped `(1, n)`,
  silently treating columns as observations. Row-vector and multi-column targets
  now raise `ValueError`.
- Fixed: chunked and row-wise incremental tests used `IncrementalOLS.fit` as
  the oracle. They now use independent direct normal-equation references.
- Fixed: `regularize_intercept=True` ridge behavior was untested. It now has
  direct-solve coverage.
- Fixed: diagnostics coverage was too narrow. Tests now cover weighted,
  no-intercept, constant-target, rolling-window, and forgetting diagnostics.
- Fixed: long-stream stability tests did not compare final state against
  recomputed references. They now compare rolling and forgetting parameters and
  sufficient statistics against independent references.
- Fixed: pytest configuration lacked strict marker/config enforcement and
  coverage tooling. Strict pytest options and a coverage threshold are now
  configured.
- Fixed: unfitted `intercept_` returned `0.0` when `fit_intercept=False`.
  It now raises `NotFittedError`, matching learned-attribute behavior.
- Fixed: `rank_` reported the rank of the regularized solve system. It now
  reports the rank of the maintained data Gram matrix.
- Fixed: batch `fit` accepted zero feature columns. It now raises `ValueError`.
- Fixed: zero-effective-weight diagnostics attempted to solve parameters and
  raised despite the explicit NaN diagnostics branch. They now return NaN
  diagnostics without solving.
- Fixed: `RollingOLS.fit` did not honor `recompute_every` during initial
  loading. It now recomputes before returning when the threshold is reached.
- Fixed: invalid rolling window types such as `True` and `math.inf` did not
  consistently raise the documented `ValueError`. Window and recompute interval
  validation now requires positive non-boolean integers.
- Fixed: README count semantics implied `n_observations_` was raw processed
  rows for every estimator. It now distinguishes append/forgetting raw counts
  from rolling active-window counts.
- Fixed: the performance note described diagonal ridge setup as `Theta(p)`,
  but the current dense implementation materializes a dense penalty matrix. The
  note now states the implemented pre-solve ridge work is `Theta(p^2)`.
