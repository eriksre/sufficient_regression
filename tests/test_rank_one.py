"""Tests for native Sherman-Morrison rank-1 inverse maintenance.

The dense estimators in :mod:`sufficient_regression.ols` are the reference: the
rank-1 estimators must reproduce their coefficients, covariance, and loud-failure
behavior. Batch fits rebuild the inverse with the same dense solve and therefore
match to a single inversion's roundoff; long incremental/rolling streams accrue
extra Sherman-Morrison roundoff and are checked at a looser-but-tight tolerance.
"""

import numpy as np
import pytest

from sufficient_regression import _native
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


def test_append_batch_after_inverse_build_uses_native_direct_beta_update():
    X, y = make_data(n=280, p=7, seed=313)
    weights = np.linspace(0.2, 2.0, X.shape[0])
    dense = IncrementalOLS(ridge=1e-5).fit(
        X[:80],
        y[:80],
        sample_weight=weights[:80],
    )
    fast = RankOneIncrementalOLS(ridge=1e-5).fit(
        X[:80],
        y[:80],
        sample_weight=weights[:80],
    )
    _ = fast.params_

    dense.partial_fit(X[80:], y[80:], sample_weight=weights[80:])
    fast.partial_fit(X[80:], y[80:], sample_weight=weights[80:])

    assert not fast._dirty
    np.testing.assert_allclose(fast.params_, dense.params_, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(fast.xtx_, dense.xtx_, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(fast.xty_, dense.xty_, atol=1e-10, rtol=1e-10)


def test_wide_blas_transition_matches_direct_solve_for_append_and_rolling():
    rng = np.random.default_rng(317)
    n, p, window = 330, 256, 300
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    ridge = 1e-3

    incremental = RankOneIncrementalOLS(
        fit_intercept=False,
        ridge=ridge,
    ).fit(X[:window], y[:window])
    _ = incremental.params_
    incremental.partial_fit(X[window:], y[window:])
    expected_append = np.linalg.solve(
        X.T @ X + ridge * np.eye(p),
        X.T @ y,
    )
    np.testing.assert_allclose(
        incremental.params_,
        expected_append,
        atol=1e-9,
        rtol=1e-9,
    )

    rolling = RankOneRollingOLS(
        window=window,
        fit_intercept=False,
        ridge=ridge,
        recompute_every=10_000,
    ).fit(X[:window], y[:window])
    _ = rolling.params_
    rolling.partial_fit(X[window:], y[window:])
    X_window = X[-window:]
    y_window = y[-window:]
    expected_rolling = np.linalg.solve(
        X_window.T @ X_window + ridge * np.eye(p),
        X_window.T @ y_window,
    )
    np.testing.assert_allclose(
        rolling.params_,
        expected_rolling,
        atol=1e-8,
        rtol=1e-8,
    )


@pytest.mark.parametrize("p", [3, 31, 255])
def test_fused_native_rolling_matches_sequential_rank_one_equations(p):
    """The one-pass kernel must preserve the prior remove-then-add equations."""

    rng = np.random.default_rng(318 + p)
    base = rng.normal(size=(2 * p + 20, p))
    initial_xtx = base.T @ base + 100.0 * np.eye(p)
    initial_xty = rng.normal(size=p)
    initial_inverse = np.linalg.solve(initial_xtx, np.eye(p))
    initial_inverse = 0.5 * (initial_inverse + initial_inverse.T)
    initial_beta = initial_inverse @ initial_xty
    old_row = np.ascontiguousarray(rng.normal(size=p))
    new_row = np.ascontiguousarray(rng.normal(size=p))
    old_target = float(rng.normal())
    new_target = float(rng.normal())

    for old_weight, new_weight in ((0.0, 1.3), (1.1, 0.0), (0.7, 1.2)):
        expected_xtx = initial_xtx.copy()
        expected_xty = initial_xty.copy()
        expected_inverse = initial_inverse.copy()
        expected_beta = initial_beta.copy()

        expected_xtx -= old_weight * np.outer(old_row, old_row)
        expected_xty -= old_weight * old_row * old_target
        expected_xtx += new_weight * np.outer(new_row, new_row)
        expected_xty += new_weight * new_row * new_target

        if old_weight != 0.0:
            direction = expected_inverse @ old_row
            denominator = 1.0 - old_weight * float(old_row @ direction)
            assert denominator > 1e-10
            residual = old_target - float(old_row @ expected_beta)
            scale = old_weight / denominator
            expected_beta -= scale * direction * residual
            expected_inverse += scale * np.outer(direction, direction)

        if new_weight != 0.0:
            direction = expected_inverse @ new_row
            denominator = 1.0 + new_weight * float(new_row @ direction)
            assert denominator > 1e-10
            residual = new_target - float(new_row @ expected_beta)
            scale = new_weight / denominator
            expected_beta += scale * direction * residual
            expected_inverse -= scale * np.outer(direction, direction)

        actual_xtx = np.ascontiguousarray(initial_xtx.copy())
        actual_xty = np.ascontiguousarray(initial_xty.copy())
        actual_inverse = np.ascontiguousarray(initial_inverse.copy())
        actual_beta = np.ascontiguousarray(initial_beta.copy())
        valid = _native.rolling_update(
            actual_xtx,
            actual_xty,
            actual_inverse,
            actual_beta,
            old_row,
            old_target,
            old_weight,
            new_row,
            new_target,
            new_weight,
            1e-10,
        )

        assert valid
        np.testing.assert_allclose(
            actual_xtx,
            expected_xtx,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            actual_xty,
            expected_xty,
            atol=1e-13,
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            actual_inverse,
            expected_inverse,
            atol=1e-13,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            actual_beta,
            expected_beta,
            atol=1e-13,
            rtol=1e-12,
        )
        np.testing.assert_array_equal(actual_inverse, actual_inverse.T)


def test_append_batch_refresh_invalidates_after_native_transition():
    X, y = make_data(n=180, p=5, seed=323)
    fast = RankOneIncrementalOLS(refresh_every=10).fit(X[:40], y[:40])
    _ = fast.params_

    fast.partial_fit(X[40:90], y[40:90])

    assert fast._inv is None
    np.testing.assert_allclose(
        fast.params_,
        IncrementalOLS().fit(X[:90], y[:90]).params_,
        atol=1e-10,
        rtol=1e-10,
    )


def test_append_hot_path_reuses_inverse_after_build(monkeypatch):
    X, y = make_data(n=90, p=5, seed=333)
    ridge = 1e-4
    fast = RankOneIncrementalOLS(ridge=ridge).fit(X[:40], y[:40])
    _ = fast.params_

    dense = IncrementalOLS(ridge=ridge).fit(X[:40], y[:40])
    dense.push(X[40], y[40])
    expected = dense.params_

    def fail_solve(*args, **kwargs):
        raise AssertionError("unexpected dense solve after inverse build")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)

    fast.push(X[40], y[40])
    np.testing.assert_allclose(fast.params_, expected, atol=1e-10, rtol=1e-10)


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


def test_append_inverse_stays_symmetric_over_weighted_stream():
    X, y = make_data(n=250, p=5, seed=414)
    weights = np.linspace(0.25, 2.5, X.shape[0])
    fast = RankOneIncrementalOLS(ridge=1e-5).fit(
        X[:40],
        y[:40],
        sample_weight=weights[:40],
    )
    _ = fast.params_

    for i in range(40, X.shape[0]):
        fast.push(X[i], y[i], sample_weight=float(weights[i]))
        np.testing.assert_allclose(fast._inv, fast._inv.T, atol=1e-14, rtol=1e-14)


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


def test_rolling_hot_path_reuses_inverse_after_build(monkeypatch):
    X, y = make_data(n=90, p=4, seed=616)
    window = 35
    ridge = 1e-5
    fast = RankOneRollingOLS(window=window, ridge=ridge, recompute_every=10_000)
    dense = RollingOLS(window=window, ridge=ridge, recompute_every=10_000)
    for i in range(window):
        fast.push(X[i], y[i])
        dense.push(X[i], y[i])
    _ = fast.params_

    dense.push(X[window], y[window])
    expected = dense.params_

    def fail_solve(*args, **kwargs):
        raise AssertionError("unexpected dense solve after inverse build")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)

    fast.push(X[window], y[window])
    np.testing.assert_allclose(fast.params_, expected, atol=1e-9, rtol=1e-9)


def test_rolling_weighted_stream_matches_dense():
    X, y = make_data(n=360, p=5, seed=626)
    weights = np.linspace(0.1, 2.0, X.shape[0])
    window = 70
    dense = RollingOLS(
        window=window,
        ridge=0.2,
        regularize_intercept=True,
    )
    fast = RankOneRollingOLS(
        window=window,
        ridge=0.2,
        regularize_intercept=True,
    )
    worst = 0.0
    for i in range(X.shape[0]):
        dense.push(X[i], y[i], sample_weight=float(weights[i]))
        fast.push(X[i], y[i], sample_weight=float(weights[i]))
        if i >= window:
            worst = max(worst, float(np.max(np.abs(dense.params_ - fast.params_))))
    assert worst < 1e-8


def test_native_ring_buffer_preserves_wrapped_window_order():
    X, y = make_data(n=95, p=4, seed=631)
    weights = np.linspace(0.3, 1.7, X.shape[0])
    window = 17
    fast = RankOneRollingOLS(window=window, ridge=1e-5).fit(
        X[:window],
        y[:window],
        sample_weight=weights[:window],
    )
    for index in range(window, X.shape[0]):
        fast.push(
            X[index],
            y[index],
            sample_weight=float(weights[index]),
        )

    current_X, current_y, current_weights = fast.current_window()
    np.testing.assert_array_equal(current_X, X[-window:])
    np.testing.assert_array_equal(current_y, y[-window:])
    np.testing.assert_array_equal(current_weights, weights[-window:])
    np.testing.assert_allclose(
        fast.params_,
        IncrementalOLS(ridge=1e-5)
        .fit(X[-window:], y[-window:], sample_weight=weights[-window:])
        .params_,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("bad_weight", [-1.0, np.nan, np.inf])
def test_native_push_rejects_invalid_weight(bad_weight):
    X, y = make_data(n=30, p=3, seed=633)
    fast = RankOneIncrementalOLS().fit(X[:10], y[:10])
    with pytest.raises(ValueError):
        fast.push(X[10], y[10], sample_weight=bad_weight)


def test_rolling_recompute_invalidates_inverse_before_exact_rebuild():
    X, y = make_data(n=120, p=4, seed=636)
    fast = RankOneRollingOLS(window=30, recompute_every=10_000)
    for i in range(45):
        fast.push(X[i], y[i])
    _ = fast.params_
    fast.recompute()
    assert fast._inv is None
    np.testing.assert_allclose(
        fast.params_,
        IncrementalOLS().fit(X[15:45], y[15:45]).params_,
        atol=1e-10,
        rtol=1e-10,
    )


def test_rolling_inverse_stays_symmetric_over_long_stream():
    X, y = make_data(n=320, p=5, seed=646)
    fast = RankOneRollingOLS(window=80, ridge=1e-5, recompute_every=10_000)

    for i in range(X.shape[0]):
        fast.push(X[i], y[i])
        if i >= 80:
            _ = fast.params_
            np.testing.assert_allclose(
                fast._inv,
                fast._inv.T,
                atol=1e-14,
                rtol=1e-14,
            )


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


def test_exactly_singular_raises_loudly_even_when_lapack_does_not(monkeypatch):
    """Rank detection must not depend on platform-specific solve behavior."""

    rng = np.random.default_rng(909)
    n = 150
    a = rng.normal(size=n)
    X = np.column_stack([a, a, rng.normal(size=n)])  # column 0 == column 1
    y = rng.normal(size=n)
    fast = RankOneIncrementalOLS(fit_intercept=True).fit(X, y)
    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda matrix, rhs: np.zeros_like(rhs, dtype=float),
    )
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


def test_singular_rolling_window_fails_loudly_after_rebuild():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    y = np.array([1.0, 2.0, 3.0])
    fast = RankOneRollingOLS(window=2, fit_intercept=False)
    fast.push(X[0], y[0])
    fast.push(X[1], y[1])
    _ = fast.params_

    fast.push(X[2], y[2])
    assert fast._inv is None
    with pytest.raises(SingularRegressionError):
        _ = fast.params_


def test_zero_total_weight_raises_loudly():
    X, y = make_data(n=50, p=3, seed=1313)
    fast = RankOneIncrementalOLS().fit(X, y, sample_weight=np.zeros(X.shape[0]))
    with pytest.raises(SingularRegressionError):
        _ = fast.params_


# --------------------------------------------------------------------------- #
# Streaming row validation, which the native kernel performs on the way        #
# through rather than with a NumPy scan in the push path                       #
# --------------------------------------------------------------------------- #


def _streaming_models(window):
    X, y = make_data(n=60, p=4, seed=4242)
    return [
        (RankOneIncrementalOLS().fit(X[:20], y[:20]), X, y),
        (RankOneRollingOLS(window=window).fit(X[:20], y[:20]), X, y),
    ]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_native_push_rejects_non_finite_features(bad_value):
    for model, X, y in _streaming_models(window=15):
        bad_row = X[20].copy()
        bad_row[2] = bad_value
        with pytest.raises(ValueError, match="x must contain only finite values"):
            model.push(bad_row, y[20])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_native_push_rejects_non_finite_target(bad_value):
    for model, X, y in _streaming_models(window=15):
        with pytest.raises(ValueError, match="y must contain only finite values"):
            model.push(X[20], bad_value)


def test_rejected_push_leaves_state_untouched():
    """A rejected row must not leave partially updated statistics behind."""

    for model, X, y in _streaming_models(window=15):
        _ = model.params_
        before_params = model.params_.copy()
        before_xtx = model.xtx_.copy()
        before_xty = model.xty_.copy()
        before_rows = model.n_observations_
        bad_row = X[20].copy()
        bad_row[0] = np.nan
        for bad_args in ((bad_row, y[20]), (X[20], np.nan)):
            with pytest.raises(ValueError):
                model.push(*bad_args)
        np.testing.assert_array_equal(model.xtx_, before_xtx)
        np.testing.assert_array_equal(model.xty_, before_xty)
        np.testing.assert_array_equal(model.params_, before_params)
        assert model.n_observations_ == before_rows


def test_wide_blas_push_still_validates_non_finite_rows():
    """The wide path skips the kernel, so it validates the row explicitly."""

    n_features = 260
    X, y = make_data(n=700, p=n_features, seed=515)
    fast = RankOneIncrementalOLS(fit_intercept=False, ridge=1e-9).fit(
        X[:600], y[:600]
    )
    _ = fast.params_
    bad_row = X[600].copy()
    bad_row[7] = np.nan
    with pytest.raises(ValueError, match="x must contain only finite values"):
        fast.push(bad_row, y[600])


def test_push_scratch_reuse_does_not_alias_rolling_window():
    """``push`` stages rows in a reused buffer; the ring buffer must copy them."""

    X, y = make_data(n=400, p=3, seed=717)
    window = 40
    fast = RankOneRollingOLS(window=window).fit(X[:window], y[:window])
    _ = fast.params_
    for index in range(window, X.shape[0]):
        fast.push(X[index], y[index])
        _ = fast.params_
    rows, targets, weights = fast.current_window()
    np.testing.assert_allclose(rows, X[-window:], rtol=0, atol=0)
    np.testing.assert_allclose(targets, y[-window:], rtol=0, atol=0)
    np.testing.assert_allclose(weights, np.ones(window), rtol=0, atol=0)


@pytest.mark.parametrize(
    "row_factory",
    [
        lambda row: list(row),
        lambda row: row.reshape(1, -1),
        lambda row: row.astype(np.float32),
        lambda row: [int(value) for value in np.round(row)],
    ],
)
def test_push_accepts_non_canonical_row_forms(row_factory):
    X, y = make_data(n=80, p=4, seed=818)
    fast = RankOneIncrementalOLS().fit(X[:30], y[:30])
    dense = IncrementalOLS().fit(X[:30], y[:30])
    _ = fast.params_
    for index in range(30, 40):
        fast.push(row_factory(X[index]), float(y[index]))
        dense.push(row_factory(X[index]), float(y[index]))
    np.testing.assert_allclose(fast.params_, dense.params_, rtol=1e-10)


def test_push_rejects_wrong_width_and_multi_row_input():
    X, y = make_data(n=40, p=4, seed=919)
    fast = RankOneIncrementalOLS().fit(X[:20], y[:20])
    with pytest.raises(ValueError, match="different number of features"):
        fast.push(np.ones(7), 1.0)
    with pytest.raises(ValueError, match="must be one row"):
        fast.push(X[20:23], 1.0)
