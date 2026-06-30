"""Exact online and rolling least-squares regression."""

from .ols import (
    ForgettingOLS,
    IncrementalOLS,
    NotFittedError,
    RegressionDiagnostics,
    RegressionError,
    RollingOLS,
    SingularRegressionError,
)

__all__ = [
    "ForgettingOLS",
    "IncrementalOLS",
    "NotFittedError",
    "RegressionDiagnostics",
    "RegressionError",
    "RollingOLS",
    "SingularRegressionError",
]
