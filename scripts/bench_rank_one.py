#!/usr/bin/env python
"""Benchmark the rank-1 incremental-inverse prototype against the dense solve.

The performance note shows that maintaining sufficient statistics removes
repeated row scans but leaves a ``Theta(p^3)`` dense solve on every coefficient
read. This script measures the hot path that exposes that bound -- coefficients
requested after *every* row -- and compares:

    IncrementalOLS         : dense np.linalg.solve per params read   (Theta(p^3))
    RankOneIncrementalOLS  : Sherman-Morrison maintained inverse     (Theta(p^2))

It also checks rolling windows and reports the worst coefficient discrepancy so
the speedup is never mistaken for a correctness regression.

Run with the repo virtualenv active:

    source .venv/bin/activate && python scripts/bench_rank_one.py
"""

from __future__ import annotations

import time

import numpy as np

from sufficient_regression import (
    IncrementalOLS,
    RankOneIncrementalOLS,
    RollingOLS,
    RankOneRollingOLS,
)


def _make_data(n: int, p: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = 0.5 + X @ beta + rng.normal(scale=0.1, size=n)
    return X, y


def _time_append_stream(model, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Push every row and read coefficients after each, returning (seconds, beta)."""

    warmup = X.shape[1] + 2  # skip rank-deficient warm-up reads
    start = time.perf_counter()
    beta = None
    for i in range(X.shape[0]):
        model.push(X[i], y[i])
        if i >= warmup:
            beta = model.params_
    return time.perf_counter() - start, beta


def _time_rolling_stream(model, X, y) -> tuple[float, np.ndarray]:
    window = model.window
    start = time.perf_counter()
    beta = None
    for i in range(X.shape[0]):
        model.push(X[i], y[i])
        if i >= window:
            beta = model.params_
    return time.perf_counter() - start, beta


def bench_append(n: int, dims: list[int]) -> None:
    print(f"\nAppend stream: n={n} rows, coefficients read after every row")
    print(f"{'p':>5} {'dense (s)':>12} {'rank-1 (s)':>12} {'speedup':>9} {'max|Δβ|':>12}")
    for p in dims:
        X, y = _make_data(n, p, seed=p)
        t_dense, b_dense = _time_append_stream(IncrementalOLS(), X, y)
        t_fast, b_fast = _time_append_stream(RankOneIncrementalOLS(), X, y)
        diff = float(np.max(np.abs(b_dense - b_fast)))
        print(f"{p:>5} {t_dense:>12.4f} {t_fast:>12.4f} {t_dense / t_fast:>8.1f}x {diff:>12.2e}")


def bench_rolling(n: int, window: int, dims: list[int]) -> None:
    print(f"\nRolling stream: n={n} rows, window={window}, coefficients read after every slide")
    print(f"{'p':>5} {'dense (s)':>12} {'rank-1 (s)':>12} {'speedup':>9} {'max|Δβ|':>12}")
    for p in dims:
        X, y = _make_data(n, p, seed=100 + p)
        t_dense, b_dense = _time_rolling_stream(RollingOLS(window=window), X, y)
        t_fast, b_fast = _time_rolling_stream(RankOneRollingOLS(window=window), X, y)
        diff = float(np.max(np.abs(b_dense - b_fast)))
        print(f"{p:>5} {t_dense:>12.4f} {t_fast:>12.4f} {t_dense / t_fast:>8.1f}x {diff:>12.2e}")


def bench_solve_step(dims: list[int], steps: int = 200) -> None:
    """Isolate the per-step linear algebra: append one row, then read coefficients.

    This strips away the push() validation/buffer overhead (identical for both
    estimators) so the measurement reflects only the algorithmic difference:
    a fresh dense solve (Theta(p^3)) versus a Sherman-Morrison inverse update
    plus a matrix-vector product (Theta(p^2)).
    """

    print(f"\nIsolated per-step cost: one row appended + one coefficient read, x{steps}")
    print(f"{'p':>5} {'dense (ms)':>12} {'rank-1 (ms)':>12} {'speedup':>9} {'max|Δβ|':>12}")
    for p in dims:
        rng = np.random.default_rng(1000 + p)
        # Seed a well-conditioned full-rank system so both paths start from the
        # same state; then measure identical append+solve steps on each.
        seed_rows = max(4 * p, 64)
        Xs = rng.normal(size=(seed_rows, p))
        ys = rng.normal(size=seed_rows)
        new_X = rng.normal(size=(steps, p))
        new_y = rng.normal(size=steps)

        # Dense baseline: maintain XtX/Xty, solve from scratch each read.
        xtx = Xs.T @ Xs
        xty = Xs.T @ ys
        start = time.perf_counter()
        for i in range(steps):
            z = new_X[i]
            xtx = xtx + np.outer(z, z)
            xty = xty + z * new_y[i]
            beta_dense = np.linalg.solve(xtx, xty)
        t_dense = time.perf_counter() - start

        # Rank-1: maintain the inverse, read via a matrix-vector product.
        xtx2 = Xs.T @ Xs
        xty2 = Xs.T @ ys
        inv = np.linalg.solve(xtx2, np.eye(p))
        start = time.perf_counter()
        for i in range(steps):
            z = new_X[i]
            u = inv @ z
            denom = 1.0 + float(z @ u)
            inv = inv - (1.0 / denom) * np.outer(u, u)
            xty2 = xty2 + z * new_y[i]
            beta_fast = inv @ xty2
        t_fast = time.perf_counter() - start

        diff = float(np.max(np.abs(beta_dense - beta_fast)))
        print(
            f"{p:>5} {t_dense * 1e3:>12.2f} {t_fast * 1e3:>12.2f} "
            f"{t_dense / t_fast:>8.1f}x {diff:>12.2e}"
        )


def main() -> None:
    bench_solve_step(dims=[32, 64, 128, 256, 512, 1024])
    bench_append(n=4000, dims=[8, 32, 128, 512])
    # window must stay comfortably larger than p, else the active-window system
    # is rank-deficient and both estimators solve a singular system.
    bench_rolling(n=4000, window=2000, dims=[8, 32, 128, 512])
    print(
        "\nThe isolated table shows the algorithmic win: the dense path pays a "
        "Theta(p^3) solve on every read while the rank-1 path pays Theta(p^2), so "
        "the speedup grows ~linearly in p. The end-to-end streaming tables fold in "
        "per-row Python/validation overhead common to both paths, so the win there "
        "only emerges once p is large enough for the solve to dominate that overhead."
    )


if __name__ == "__main__":
    main()
