#!/usr/bin/env python
"""Benchmark the public rank-one estimators and emit machine-readable results.

Run this script from two checkouts to compare an implementation change:

    python scripts/bench_production.py --label before --output before.json
    python scripts/bench_production.py --label native --output native.json

The benchmark separates initialization from the timed streaming region and
reports medians. Coefficients are read after every update because this is the
workload where maintained inverse state should remove repeated dense solves.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from sufficient_regression import RankOneIncrementalOLS, RankOneRollingOLS


def _make_data(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = 0.5 + X @ beta + rng.normal(scale=0.1, size=n)
    return X, y


def _median_seconds(prepare, run, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        state = prepare()
        start = time.perf_counter()
        run(state)
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def _append_seconds(n: int, p: int, repeats: int) -> float:
    seed_rows = max(4 * p, 64)
    X, y = _make_data(seed_rows + n, p, seed=1_000 + p)

    def prepare():
        model = RankOneIncrementalOLS(fit_intercept=False).fit(
            X[:seed_rows],
            y[:seed_rows],
        )
        _ = model.params_
        return model

    def run(model) -> None:
        for index in range(seed_rows, X.shape[0]):
            model.push(X[index], y[index])
            _ = model.params_

    return _median_seconds(prepare, run, repeats)


def _rolling_seconds(
    *,
    slides: int,
    window: int,
    p: int,
    repeats: int,
) -> float:
    X, y = _make_data(window + slides, p, seed=2_000 + p + window)

    def prepare():
        model = RankOneRollingOLS(
            window=window,
            fit_intercept=False,
            ridge=1e-8,
        ).fit(X[:window], y[:window])
        _ = model.params_
        return model

    def run(model) -> None:
        for index in range(window, X.shape[0]):
            model.push(X[index], y[index])
            _ = model.params_

    return _median_seconds(prepare, run, repeats)


def _refit_seconds(
    *,
    slides: int,
    window: int,
    p: int,
    repeats: int,
) -> float:
    X, y = _make_data(window + slides, p, seed=2_000 + p + window)
    penalty = 1e-8 * np.eye(p)

    def run(_state) -> None:
        for index in range(window, X.shape[0]):
            X_window = X[index - window + 1 : index + 1]
            y_window = y[index - window + 1 : index + 1]
            _ = np.linalg.solve(
                X_window.T @ X_window + penalty,
                X_window.T @ y_window,
            )

    return _median_seconds(lambda: None, run, repeats)


def benchmark(*, quick: bool, repeats: int) -> dict[str, list[dict[str, float]]]:
    dimensions = [8, 32, 128] if quick else [8, 32, 128, 256]
    stream_rows = 1_000 if quick else 4_000
    rolling_window = 750 if quick else 2_000
    rolling_slides = 750 if quick else 2_000
    crossover_windows = [1_000, 10_000] if quick else [1_000, 10_000, 100_000]
    crossover_slides = 300 if quick else 1_000

    append = [
        {
            "features": p,
            "rows": stream_rows,
            "seconds": _append_seconds(stream_rows, p, repeats),
        }
        for p in dimensions
    ]
    rolling = [
        {
            "features": p,
            "window": rolling_window,
            "slides": rolling_slides,
            "seconds": _rolling_seconds(
                slides=rolling_slides,
                window=rolling_window,
                p=p,
                repeats=repeats,
            ),
        }
        for p in dimensions
    ]
    crossover = []
    for window in crossover_windows:
        native_seconds = _rolling_seconds(
            slides=crossover_slides,
            window=window,
            p=8,
            repeats=repeats,
        )
        refit_seconds = _refit_seconds(
            slides=crossover_slides,
            window=window,
            p=8,
            repeats=repeats,
        )
        crossover.append(
            {
                "features": 8,
                "window": window,
                "slides": crossover_slides,
                "rank_one_seconds": native_seconds,
                "refit_seconds": refit_seconds,
                "speedup": refit_seconds / native_seconds,
            }
        )
    return {
        "append": append,
        "rolling": rolling,
        "refit_crossover": crossover,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")

    payload = {
        "label": args.label,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "config": {
            "quick": args.quick,
            "repeats": args.repeats,
        },
        "results": benchmark(quick=args.quick, repeats=args.repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
