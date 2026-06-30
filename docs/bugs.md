# Bugs And Review Findings

No open bugs are currently known.

## Review Findings Closed In This Build

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
