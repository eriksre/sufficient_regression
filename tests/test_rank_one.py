"""Tests for the Sherman-Morrison rank-1 inverse-maintenance prototype.

The dense estimators in :mod:`sufficient_regression.ols` are the reference: the
rank-1 estimators must reproduce their coefficients, covariance, and loud-failure
behavior. Batch fits rebuild the inverse with the same dense solve and therefore
match to a single inversion's roundoff; long incremental/rolling streams accrue
extra Sherman-Morrison roundoff and are checked at a looser-but-tight tolerance.
"""

import numpy as np
import pytest

from sufficient_regression import (
    IncrementalOLS,
    NotFittedError,
    RankOneIncrementalOLS,
    RankOneRollingOLS,
    RollingOLS,
    RegressionError,
    SingularRegressionError,
)


def make_data(n=200, p=5, *, seed=123, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = 1.75 + X @ beta + rng.normal(scale=noise, size=n)
    return X, y


# --------------------------------------------------------------------------- #
# Batch parity with the dense estimators                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("ridge", [0.0, 0.5])
def test_append_batch_matches_dense(fit_intercept, ridge):
    X, y = make_data()
    dense = IncrementalOLS(fit_intercept=fit_intercept, ridge=ridge).fit(X, y)
    fast = RankOneIncrementalOLS(fit_intercept=fit_intercept, ridge=ridge).fit(X, y)
    np.testing.assert_allclose(fast.params_, dense.params_, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(fast.coef_, dense.coef_, atol=1e-11, rtol=1e-11)
    assert fast.intercept_ == pytest.approx(dense.intercept_, abs=1e-11)


def test_append_weighted_matches_dense():
    X, y = make_data(n=120, p=4, seed=51)
    rng = np.random.default_rng(9)
    weights = rng.uniform(0.2, 3.0, size=X.shape[0])
    dense = IncrementalOLS().fit(X, y, sample_weight=weights)
    fast = RankOneIncrementalOLS().fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(fast.params_, dense.params_, atol=1e-11, rtol=1e-11)


def test_predict_matches_dense():
    X, y = make_data()
    dense = IncrementalOLS().fit(X, y)
    fast = RankOneIncrementalOLS().fit(X, y)
    np.testing.assert_allclose(fast.predict(X[:10]), dense.predict(X[:10]), atol=1e-10)


# --------------------------------------------------------------------------- #
# Streaming parity: the hot path the prototype targets                        #
# --------------------------------------------------------------------------- #


def test_append_stream_read_after_each_matches_dense():
    """Push one row at a time, reading coefficients after every full-rank step."""

    X, y = make_data(n=600, p=6, seed=202)
    dense = IncrementalOLS()
    fast = RankOneIncrementalOLS()
    worst = 0.0
    for i in range(X.shape[0]):
        dense.push(X[i], y[i])
        fast.push(X[i], y[i])
        if i >= X.shape[1] + 2:
            worst = max(worst, float(np.max(np.abs(dense.params_ - fast.params_))))
    assert worst < 1e-7


def test_inverse_built_midstream_then_maintained():
    """Reading coefficients mid-stream builds the inverse; later pushes maintain it."""

    X, y = make_data(n=400, p=5, seed=303)
    dense = IncrementalOLS().fit(X[:30], y[:30])
    fast = RankOneIncrementalOLS().fit(X[:30], y[:30])
    _ = fast.params_  # forces the lazy inverse build now
    for i in range(30, X.shape[0]):
        dense.push(X[i], y[i])
        fast.push(X[i], y[i])
    np.testing.assert_allclose(fast.params_, dense.params_, atol=1e-8, rtol=1e-8)


def test_refresh_every_preserves_correctness():
    """A periodic exact rebuild must not change the answer, only bound drift.

    Coefficients are read after every step so the inverse is actually maintained
    in place; otherwise the lazy build would defer all work to a single final
    solve and never exercise the refresh path.
    """

    X, y = make_data(n=800, p=6, seed=404)
    dense = IncrementalOLS()
    fast = RankOneIncrementalOLS(refresh_every=50)
    for i in range(X.shape[0]):
        dense.push(X[i], y[i])
        fast.push(X[i], y[i])
        if i >= X.shape[1] + 2:
            np.testing.assert_allclose(
                fast.params_, dense.params_, atol=1e-9, rtol=1e-9
            )
    # The periodic rebuild must have fired and reset the update counter.
    assert fast._inv_updates < fast.refresh_every


# --------------------------------------------------------------------------- #
# Rolling parity                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ridge", [0.0, 1e-6])
def test_rolling_stream_matches_dense(ridge):
    X, y = make_data(n=700, p=5, seed=505)
    window = 60
    dense = RollingOLS(window=window, ridge=ridge)
    fast = RankOneRollingOLS(window=window, ridge=ridge)
    worst = 0.0
    for i in range(X.shape[0]):
        dense.push(X[i], y[i])
        fast.push(X[i], y[i])
        if i >= window:
            worst = max(worst, float(np.max(np.abs(dense.params_ - fast.params_))))
    assert worst < 1e-8


def test_rolling_recompute_rebuilds_inverse():
    """After the base recomputes XtX, the maintained inverse is rebuilt from it."""

    X, y = make_data(n=300, p=4, seed=606)
    window = 40
    fast = RankOneRollingOLS(window=window, recompute_every=20)
    dense = RollingOLS(window=window, recompute_every=20)
    for i in range(X.shape[0]):
        fast.push(X[i], y[i])
        dense.push(X[i], y[i])
    np.testing.assert_allclose(fast.params_, dense.params_, atol=1e-10, rtol=1e-10)


# --------------------------------------------------------------------------- #
# Inference reuse of the maintained inverse                                   #
# --------------------------------------------------------------------------- #


def test_covariance_and_standard_errors_match_dense():
    X, y = make_data(n=150, p=4, seed=707)
    dense = IncrementalOLS().fit(X, y)
    fast = RankOneIncrementalOLS().fit(X, y)
    np.testing.assert_allclose(
        fast.coefficient_covariance_, dense.coefficient_covariance_, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        fast.standard_errors_, dense.standard_errors_, atol=1e-10, rtol=1e-10
    )


def test_covariance_requires_zero_ridge():
    X, y = make_data(n=100, p=3, seed=808)
    fast = RankOneIncrementalOLS(ridge=0.5).fit(X, y)
    with pytest.raises(RegressionError):
        _ = fast.coefficient_covariance_


# --------------------------------------------------------------------------- #
# Loud failure                                                                #
# --------------------------------------------------------------------------- #


def test_not_fitted_raises():
    with pytest.raises(NotFittedError):
        _ = RankOneIncrementalOLS().params_


def test_exactly_singular_raises_loudly():
    """A duplicated feature column makes XtX exactly singular; the solve must fail."""

    rng = np.random.default_rng(909)
    n = 150
    a = rng.normal(size=n)
    X = np.column_stack([a, a, rng.normal(size=n)])  # column 0 == column 1
    y = rng.normal(size=n)
    fast = RankOneIncrementalOLS(fit_intercept=True).fit(X, y)
    with pytest.raises(SingularRegressionError):
        _ = fast.params_


def test_invalid_refresh_every_rejected():
    with pytest.raises(ValueError):
        RankOneIncrementalOLS(refresh_every=0)
    with pytest.raises(ValueError):
        RankOneIncrementalOLS(refresh_every=-3)


def test_singular_downdate_invalidates_and_rebuilds():
    """A drop that drives the system singular must invalidate, not blow up.

    In a 2-row, 2-parameter window each active row has leverage exactly 1, so
    dropping one makes the Sherman-Morrison denominator collapse to ~0. The guard
    must invalidate the maintained inverse so the next read rebuilds from the
    (now full-rank) window and still matches the dense estimator.
    """

    rng = np.random.default_rng(1212)
    n = 40
    X = rng.normal(size=(n, 2))
    y = rng.normal(size=n)
    window = 2
    fast = RankOneRollingOLS(window=window, fit_intercept=False)
    dense = RollingOLS(window=window, fit_intercept=False)
    saw_invalidation = False
    for i in range(n):
        fast.push(X[i], y[i])
        dense.push(X[i], y[i])
        if i >= window:
            # The drop inside this push collapses the denominator of the inverse
            # built by the previous read; check before the read rebuilds it.
            if fast._inv is None:
                saw_invalidation = True
            np.testing.assert_allclose(
                fast.params_, dense.params_, atol=1e-9, rtol=1e-9
            )
    assert saw_invalidation


def test_zero_total_weight_raises_loudly():
    X, y = make_data(n=50, p=3, seed=1313)
    fast = RankOneIncrementalOLS().fit(X, y, sample_weight=np.zeros(X.shape[0]))
    with pytest.raises(SingularRegressionError):
        _ = fast.params_
