import math

import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge

from sufficient_regression import (
    ForgettingOLS,
    IncrementalOLS,
    NotFittedError,
    RollingOLS,
    RegressionError,
    SingularRegressionError,
)


def make_data(n=120, p=4, *, seed=123, noise=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    intercept = 1.75
    y = intercept + X @ beta + rng.normal(scale=noise, size=n)
    return X, y, intercept, beta


def augment(X, fit_intercept):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not fit_intercept:
        return X
    return np.column_stack((np.ones(X.shape[0]), X))


def ridge_penalty(n_params, ridge, fit_intercept, regularize_intercept=False):
    penalty = np.eye(n_params) * ridge
    if fit_intercept and not regularize_intercept:
        penalty[0, 0] = 0.0
    return penalty


def reference_params(
    X,
    y,
    *,
    fit_intercept=True,
    sample_weight=None,
    ridge=0.0,
    regularize_intercept=False,
):
    X_aug = augment(X, fit_intercept)
    y = np.asarray(y, dtype=float)
    weights = np.ones(y.shape[0]) if sample_weight is None else np.asarray(sample_weight)
    XtX = X_aug.T @ (weights[:, None] * X_aug)
    Xty = X_aug.T @ (weights * y)
    return np.linalg.solve(
        XtX
        + ridge_penalty(
            X_aug.shape[1],
            ridge,
            fit_intercept,
            regularize_intercept=regularize_intercept,
        ),
        Xty,
    )


def reference_stats(X, y, *, fit_intercept=True, sample_weight=None):
    X_aug = augment(X, fit_intercept)
    y = np.asarray(y, dtype=float)
    weights = np.ones(y.shape[0]) if sample_weight is None else np.asarray(sample_weight)
    XtX = X_aug.T @ (weights[:, None] * X_aug)
    Xty = X_aug.T @ (weights * y)
    yty = float(np.dot(weights * y, y))
    return XtX, Xty, yty


def assert_params_equal(model, expected, *, atol=1e-10, rtol=1e-10):
    np.testing.assert_allclose(model.params_, expected, atol=atol, rtol=rtol)
    if model.fit_intercept:
        assert model.intercept_ == pytest.approx(expected[0], abs=atol, rel=rtol)
        np.testing.assert_allclose(model.coef_, expected[1:], atol=atol, rtol=rtol)
    else:
        assert model.intercept_ == 0.0
        np.testing.assert_allclose(model.coef_, expected, atol=atol, rtol=rtol)


def test_public_api_exports():
    import sufficient_regression

    assert sufficient_regression.__all__ == [
        "ForgettingOLS",
        "IncrementalOLS",
        "NotFittedError",
        "RegressionDiagnostics",
        "RegressionError",
        "RollingOLS",
        "SingularRegressionError",
    ]


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_incremental_fit_matches_numpy_statsmodels_and_sklearn(fit_intercept):
    X, y, _, _ = make_data()
    model = IncrementalOLS(fit_intercept=fit_intercept).fit(X, y)

    expected = reference_params(X, y, fit_intercept=fit_intercept)
    assert_params_equal(model, expected)

    if fit_intercept:
        sm_expected = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit().params
    else:
        sm_expected = sm.OLS(y, X).fit().params
    np.testing.assert_allclose(model.params_, sm_expected, atol=1e-10, rtol=1e-10)

    sklearn = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    np.testing.assert_allclose(model.coef_, sklearn.coef_, atol=1e-10, rtol=1e-10)
    assert model.intercept_ == pytest.approx(float(sklearn.intercept_), abs=1e-10)


def test_incremental_chunked_and_row_push_match_batch():
    X, y, _, _ = make_data(n=150, p=5, seed=321)
    expected = reference_params(X, y)

    chunked = IncrementalOLS()
    start = 0
    for size in [1, 2, 7, 31, 109]:
        chunked.partial_fit(X[start : start + size], y[start : start + size])
        start += size
    assert_params_equal(chunked, expected)

    rowwise = IncrementalOLS()
    for x_row, y_value in zip(X, y, strict=True):
        rowwise.push(x_row, y_value)
    assert_params_equal(rowwise, expected)
    np.testing.assert_allclose(
        rowwise.predict(X[:10]), augment(X[:10], True) @ expected
    )


def test_incremental_weighted_matches_statsmodels_wls():
    X, y, _, _ = make_data(n=90, p=3, seed=777)
    weights = np.linspace(0.1, 2.5, X.shape[0])

    model = IncrementalOLS().fit(X, y, sample_weight=weights)
    expected = sm.WLS(y, sm.add_constant(X, has_constant="add"), weights=weights).fit().params
    assert_params_equal(model, expected)

    chunked = IncrementalOLS()
    chunked.partial_fit(X[:25], y[:25], sample_weight=weights[:25])
    chunked.partial_fit(X[25:], y[25:], sample_weight=weights[25:])
    assert_params_equal(chunked, expected)

    XtX, Xty, yty = reference_stats(X, y, sample_weight=weights)
    np.testing.assert_allclose(model.xtx_, XtX, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(model.xty_, Xty, atol=1e-12, rtol=1e-12)
    assert model.yty_ == pytest.approx(yty, abs=1e-12)


def test_zero_weight_rows_have_no_effect_and_all_zero_weight_raises_on_solve():
    X, y, _, _ = make_data(n=80, p=3, seed=555)
    weights = np.ones(X.shape[0])
    weights[10:30] = 0.0

    weighted = IncrementalOLS().fit(X, y, sample_weight=weights)
    expected = IncrementalOLS().fit(X[weights > 0], y[weights > 0]).params_
    assert_params_equal(weighted, expected)

    all_zero = IncrementalOLS().fit(X, y, sample_weight=np.zeros(X.shape[0]))
    with pytest.raises(SingularRegressionError, match="zero total sample weight"):
        _ = all_zero.params_
    zero_diag = all_zero.diagnostics_
    assert zero_diag.weight_sum == 0.0
    assert math.isnan(zero_diag.residual_sum_squares)
    assert math.isnan(zero_diag.weighted_mean_y)
    assert math.isnan(zero_diag.total_sum_squares)
    assert math.isnan(zero_diag.r_squared)


def test_ridge_matches_direct_solve_and_sklearn_without_intercept_penalty():
    X, y, _, _ = make_data(n=100, p=4, seed=987)
    ridge = 3.25
    model = IncrementalOLS(ridge=ridge).fit(X, y)
    expected = reference_params(X, y, ridge=ridge)
    assert_params_equal(model, expected, atol=1e-10, rtol=1e-10)

    sklearn = Ridge(alpha=ridge, fit_intercept=True, solver="cholesky").fit(X, y)
    np.testing.assert_allclose(model.coef_, sklearn.coef_, atol=1e-10, rtol=1e-10)
    assert model.intercept_ == pytest.approx(float(sklearn.intercept_), abs=1e-10)


def test_ridge_can_regularize_intercept_when_requested():
    X, y, _, _ = make_data(n=100, p=4, seed=654)
    ridge = 2.0
    model = IncrementalOLS(ridge=ridge, regularize_intercept=True).fit(X, y)
    expected = reference_params(
        X,
        y,
        ridge=ridge,
        regularize_intercept=True,
    )
    assert_params_equal(model, expected, atol=1e-10, rtol=1e-10)

    unregularized_intercept = IncrementalOLS(ridge=ridge).fit(X, y)
    assert not np.allclose(model.params_, unregularized_intercept.params_)


def test_singular_unregularized_system_raises_and_ridge_solves():
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = IncrementalOLS(fit_intercept=False).fit(X, y)
    with pytest.raises(SingularRegressionError, match="singular"):
        _ = model.params_

    ridge = IncrementalOLS(fit_intercept=False, ridge=0.1).fit(X, y)
    assert np.all(np.isfinite(ridge.params_))
    assert ridge.rank_ == 1
    assert ridge.diagnostics_.rank == 1


def test_fit_replaces_prior_state_and_validation_raises_loudly():
    X1, y1, _, _ = make_data(n=40, p=2, seed=1)
    X2, y2, _, _ = make_data(n=60, p=2, seed=2)
    model = IncrementalOLS().fit(X1, y1)
    first_params = model.params_
    model.fit(X2, y2)
    assert not np.allclose(model.params_, first_params)
    assert_params_equal(model, reference_params(X2, y2))

    with pytest.raises(NotFittedError):
        IncrementalOLS().predict(X1)
    with pytest.raises(NotFittedError):
        _ = IncrementalOLS(fit_intercept=False).intercept_
    with pytest.raises(ValueError, match="different number of features"):
        model.partial_fit(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        model.partial_fit([[math.nan, 1.0]], [1.0])
    with pytest.raises(ValueError, match="negative"):
        model.partial_fit(X1[:2], y1[:2], sample_weight=[1.0, -1.0])
    with pytest.raises(ValueError, match="at least one row"):
        model.partial_fit(np.empty((0, 2)), np.empty(0))
    with pytest.raises(ValueError, match="at least one feature"):
        IncrementalOLS(fit_intercept=False).fit(np.empty((3, 0)), np.ones(3))


def test_validation_edge_cases_are_explicit():
    one_dimensional = IncrementalOLS().fit(np.arange(5.0), np.arange(5.0))
    assert one_dimensional.n_features_in_ == 1

    column_y = IncrementalOLS().fit(
        np.arange(10.0).reshape(5, 2),
        np.arange(5.0).reshape(-1, 1),
    )
    assert column_y.n_observations_ == 5

    scalar_y = IncrementalOLS(ridge=0.1).fit([[1.0, 2.0]], 3.0)
    assert scalar_y.n_observations_ == 1

    single_row = IncrementalOLS(ridge=0.1).push(
        [[1.0, 2.0]],
        [3.0],
        sample_weight=[1.5],
    )
    assert single_row.weight_sum_ == pytest.approx(1.5)

    with pytest.raises(ValueError, match="1D or 2D"):
        IncrementalOLS().fit(np.ones((2, 2, 1)), np.ones(2))
    with pytest.raises(ValueError, match="single-column"):
        IncrementalOLS().fit(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="row-vector"):
        IncrementalOLS().fit(
            np.arange(10.0).reshape(5, 2),
            np.arange(5.0).reshape(1, -1),
        )
    with pytest.raises(ValueError, match="must have 2 values"):
        IncrementalOLS().fit(np.ones((2, 2)), np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        IncrementalOLS().fit(np.ones((2, 2)), [1.0, math.inf])
    with pytest.raises(ValueError, match="one row"):
        IncrementalOLS().push(np.ones((2, 2)), 1.0)
    with pytest.raises(ValueError, match="at least one feature"):
        IncrementalOLS().push(np.empty((1, 0)), 1.0)
    with pytest.raises(ValueError, match="finite"):
        IncrementalOLS().push([math.inf], 1.0)
    with pytest.raises(ValueError, match="one scalar"):
        IncrementalOLS().push([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        IncrementalOLS().push([1.0], math.nan)
    with pytest.raises(ValueError, match="one scalar value for push"):
        IncrementalOLS().push([1.0], 2.0, sample_weight=[1.0, 2.0])
    with pytest.raises(ValueError, match="1D array"):
        IncrementalOLS().fit(np.ones((2, 2)), np.ones(2), sample_weight=np.ones((2, 1)))
    with pytest.raises(ValueError, match="one value per row"):
        IncrementalOLS().fit(np.ones((2, 2)), np.ones(2), sample_weight=np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        IncrementalOLS().fit(
            np.ones((2, 2)),
            np.ones(2),
            sample_weight=[1.0, math.inf],
        )
    with pytest.raises(ValueError, match="ridge"):
        IncrementalOLS(ridge=-1.0)
    with pytest.raises(ValueError, match="missing"):
        IncrementalOLS(missing="drop")

    fitted = IncrementalOLS().fit(np.ones((3, 2)), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="different number of features"):
        fitted.predict(np.ones((1, 3)))


def test_diagnostics_match_batch_reference():
    X, y, _, _ = make_data(n=75, p=3, seed=44)
    model = IncrementalOLS().fit(X, y)
    params = reference_params(X, y)
    X_aug = augment(X, True)
    residuals = y - X_aug @ params
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) @ (y - y.mean())))

    diag = model.diagnostics_
    assert diag.n_observations == len(y)
    assert diag.weight_sum == pytest.approx(float(len(y)))
    assert diag.rank == X_aug.shape[1]
    assert diag.residual_sum_squares == pytest.approx(rss, abs=1e-10)
    assert diag.total_sum_squares == pytest.approx(tss, abs=1e-10)
    assert diag.r_squared == pytest.approx(1.0 - rss / tss, abs=1e-10)


def test_weighted_and_no_intercept_diagnostics_match_reference():
    X, y, _, _ = make_data(n=85, p=3, seed=45)
    weights = np.linspace(0.2, 1.8, len(y))
    model = IncrementalOLS(fit_intercept=False).fit(X, y, sample_weight=weights)
    params = reference_params(
        X,
        y,
        fit_intercept=False,
        sample_weight=weights,
    )
    XtX, Xty, yty = reference_stats(
        X,
        y,
        fit_intercept=False,
        sample_weight=weights,
    )
    ysum = float(np.dot(weights, y))
    wsum = float(weights.sum())
    rss = float(yty - 2.0 * params @ Xty + params @ XtX @ params)
    tss = float(yty - ysum * ysum / wsum)

    diag = model.diagnostics_
    assert diag.n_observations == len(y)
    assert diag.weight_sum == pytest.approx(wsum, abs=1e-12)
    assert diag.weighted_mean_y == pytest.approx(ysum / wsum, abs=1e-12)
    assert diag.residual_sum_squares == pytest.approx(rss, abs=1e-10)
    assert diag.total_sum_squares == pytest.approx(tss, abs=1e-10)
    assert diag.r_squared == pytest.approx(1.0 - rss / tss, abs=1e-10)


def test_constant_target_diagnostics_have_nan_r_squared():
    X = np.arange(20, dtype=float).reshape(-1, 2)
    y = np.full(10, 3.5)
    model = IncrementalOLS(ridge=0.1).fit(X, y)
    diag = model.diagnostics_
    assert diag.total_sum_squares == 0.0
    assert math.isnan(diag.r_squared)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_classical_ols_uncertainty_matches_statsmodels(fit_intercept):
    X, y, _, _ = make_data(n=140, p=4, seed=411, noise=0.35)
    model = IncrementalOLS(fit_intercept=fit_intercept).fit(X, y)

    if fit_intercept:
        X_design = sm.add_constant(X, has_constant="add")
    else:
        X_design = X
    expected = sm.OLS(y, X_design).fit()

    assert model.residual_degrees_of_freedom_ == pytest.approx(expected.df_resid)
    assert model.residual_variance_ == pytest.approx(expected.scale, abs=1e-12)
    np.testing.assert_allclose(
        model.coefficient_covariance_,
        expected.cov_params(),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.standard_errors_,
        expected.bse,
        atol=1e-12,
        rtol=1e-12,
    )

    covariance_copy = model.coefficient_covariance_
    covariance_copy[0, 0] = -1.0
    assert model.coefficient_covariance_[0, 0] != -1.0


def test_weighted_uncertainty_treats_weights_as_effective_observation_mass():
    X, y, _, _ = make_data(n=28, p=3, seed=412, noise=0.4)
    weights = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 7)
    model = IncrementalOLS().fit(X, y, sample_weight=weights)

    repeated_X = np.repeat(X, weights.astype(int), axis=0)
    repeated_y = np.repeat(y, weights.astype(int))
    expected = IncrementalOLS().fit(repeated_X, repeated_y)

    assert model.residual_degrees_of_freedom_ == pytest.approx(
        expected.residual_degrees_of_freedom_
    )
    assert model.residual_variance_ == pytest.approx(
        expected.residual_variance_, abs=1e-12
    )
    np.testing.assert_allclose(
        model.coefficient_covariance_,
        expected.coefficient_covariance_,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.standard_errors_,
        expected.standard_errors_,
        atol=1e-12,
        rtol=1e-12,
    )


def test_rolling_and_forgetting_uncertainty_use_current_sufficient_statistics():
    X, y, _, _ = make_data(n=80, p=3, seed=413, noise=0.3)
    window = 22

    rolling = RollingOLS(window=window).fit(X, y)
    rolling_expected = IncrementalOLS().fit(X[-window:], y[-window:])
    np.testing.assert_allclose(
        rolling.coefficient_covariance_,
        rolling_expected.coefficient_covariance_,
        atol=1e-12,
        rtol=1e-12,
    )

    forgetting = ForgettingOLS(decay=1.0).fit(X[:35], y[:35])
    forgetting.partial_fit(X[35:], y[35:])
    incremental = IncrementalOLS().fit(X, y)
    np.testing.assert_allclose(
        forgetting.standard_errors_,
        incremental.standard_errors_,
        atol=1e-12,
        rtol=1e-12,
    )


def test_uncertainty_raises_when_classical_covariance_is_not_defined():
    with pytest.raises(NotFittedError):
        _ = IncrementalOLS().coefficient_covariance_

    X, y, _, _ = make_data(n=30, p=3, seed=414)
    ridge = IncrementalOLS(ridge=0.1).fit(X, y)
    with pytest.raises(RegressionError, match="ridge=0"):
        _ = ridge.coefficient_covariance_

    rank_deficient_X = np.array(
        [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
    )
    rank_deficient = IncrementalOLS(fit_intercept=False).fit(
        rank_deficient_X,
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    with pytest.raises(SingularRegressionError, match="rank-deficient"):
        _ = rank_deficient.standard_errors_

    exact_fit = IncrementalOLS(fit_intercept=False).fit(
        np.eye(3),
        np.array([1.0, 2.0, 3.0]),
    )
    with pytest.raises(SingularRegressionError, match="nonpositive"):
        _ = exact_fit.residual_variance_

    zero_weight = IncrementalOLS().fit(X, y, sample_weight=np.zeros(len(y)))
    with pytest.raises(SingularRegressionError, match="zero total sample weight"):
        _ = zero_weight.coefficient_covariance_


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_rolling_matches_static_ols_after_every_push(fit_intercept):
    X, y, _, _ = make_data(n=70, p=3, seed=202)
    window = 18
    model = RollingOLS(window=window, fit_intercept=fit_intercept, ridge=0.5)

    for i, (x_row, y_value) in enumerate(zip(X, y, strict=True), start=1):
        model.push(x_row, y_value)
        start = max(0, i - window)
        expected = reference_params(
            X[start:i],
            y[start:i],
            fit_intercept=fit_intercept,
            ridge=0.5,
        )
        assert model.window_size_ == i - start
        assert_params_equal(model, expected, atol=1e-10, rtol=1e-10)


def test_rolling_fit_retains_final_window_and_batch_partial_fit_matches_rowwise():
    X, y, _, _ = make_data(n=100, p=4, seed=303)
    weights = np.linspace(0.5, 1.5, len(y))
    window = 25

    batch = RollingOLS(window=window, ridge=0.2).fit(X, y, sample_weight=weights)
    expected = reference_params(
        X[-window:],
        y[-window:],
        sample_weight=weights[-window:],
        ridge=0.2,
    )
    assert_params_equal(batch, expected)
    current_X, current_y, current_w = batch.current_window()
    np.testing.assert_allclose(current_X, X[-window:])
    np.testing.assert_allclose(current_y, y[-window:])
    np.testing.assert_allclose(current_w, weights[-window:])

    rowwise = RollingOLS(window=window, ridge=0.2)
    for x_row, y_value, weight in zip(X, y, weights, strict=True):
        rowwise.push(x_row, y_value, sample_weight=weight)
    assert_params_equal(rowwise, batch.params_)

    chunked = RollingOLS(window=window, ridge=0.2)
    chunked.partial_fit(X[:37], y[:37], sample_weight=weights[:37])
    chunked.partial_fit(X[37:], y[37:], sample_weight=weights[37:])
    assert_params_equal(chunked, batch.params_)


def test_rolling_recompute_preserves_state_and_window_validation():
    X, y, _, _ = make_data(n=60, p=3, seed=88)
    model = RollingOLS(window=17, recompute_every=5, ridge=0.1).fit(X[:30], y[:30])
    assert model._updates_since_recompute == 0
    before = model.params_
    model.recompute()
    assert_params_equal(model, before)
    model.partial_fit(X[30:], y[30:])
    assert_params_equal(model, reference_params(X[-17:], y[-17:], ridge=0.1))

    with pytest.raises(ValueError, match="positive integer"):
        RollingOLS(window=0)
    with pytest.raises(ValueError, match="positive integer"):
        RollingOLS(window=2.5)
    with pytest.raises(ValueError, match="positive integer"):
        RollingOLS(window=True)
    with pytest.raises(ValueError, match="positive integer"):
        RollingOLS(window=math.inf)
    with pytest.raises(ValueError, match="positive integer"):
        RollingOLS(window=5, recompute_every=True)


def test_rolling_default_recompute_cadence_is_one_window():
    X, y, _, _ = make_data(n=10, p=3, seed=89)
    model = RollingOLS(window=4, ridge=0.1)
    assert model.recompute_every == 4

    for x_row, y_value in zip(X[:4], y[:4], strict=True):
        model.push(x_row, y_value)
    assert model._updates_since_recompute == 0

    model.push(X[4], y[4])
    assert model._updates_since_recompute == 1
    assert_params_equal(model, reference_params(X[1:5], y[1:5], ridge=0.1))


def test_rolling_diagnostics_use_active_window_only():
    X, y, _, _ = make_data(n=70, p=3, seed=707)
    weights = np.linspace(0.3, 1.7, len(y))
    window = 16
    model = RollingOLS(window=window, ridge=0.25).fit(
        X,
        y,
        sample_weight=weights,
    )
    current_X = X[-window:]
    current_y = y[-window:]
    current_weights = weights[-window:]
    params = reference_params(
        current_X,
        current_y,
        sample_weight=current_weights,
        ridge=0.25,
    )
    XtX, Xty, yty = reference_stats(
        current_X,
        current_y,
        sample_weight=current_weights,
    )
    ysum = float(np.dot(current_weights, current_y))
    wsum = float(current_weights.sum())
    rss = float(yty - 2.0 * params @ Xty + params @ XtX @ params)
    tss = float(yty - ysum * ysum / wsum)

    diag = model.diagnostics_
    assert diag.n_observations == window
    assert diag.weight_sum == pytest.approx(wsum, abs=1e-12)
    assert diag.weighted_mean_y == pytest.approx(ysum / wsum, abs=1e-12)
    assert diag.residual_sum_squares == pytest.approx(rss, abs=1e-10)
    assert diag.total_sum_squares == pytest.approx(tss, abs=1e-10)


def explicit_forgetting_reference(
    X,
    y,
    *,
    decay,
    fit_intercept=True,
    sample_weight=None,
    ridge=0.0,
):
    X_aug = augment(X, fit_intercept)
    y = np.asarray(y, dtype=float)
    weights = np.ones(y.shape[0]) if sample_weight is None else np.asarray(sample_weight)
    n_params = X_aug.shape[1]
    XtX = np.zeros((n_params, n_params))
    Xty = np.zeros(n_params)
    yty = 0.0
    ysum = 0.0
    wsum = 0.0
    for z, y_value, weight in zip(X_aug, y, weights, strict=True):
        XtX *= decay
        Xty *= decay
        yty *= decay
        ysum *= decay
        wsum *= decay
        XtX += weight * np.outer(z, z)
        Xty += weight * z * y_value
        yty += weight * y_value * y_value
        ysum += weight * y_value
        wsum += weight
    params = np.linalg.solve(
        XtX + ridge_penalty(n_params, ridge, fit_intercept),
        Xty,
    )
    return params, XtX, Xty, yty, ysum, wsum


def test_forgetting_matches_explicit_recursion_for_rows_and_chunks():
    X, y, _, _ = make_data(n=100, p=3, seed=909)
    weights = np.linspace(0.25, 2.0, len(y))
    decay = 0.97
    ridge = 0.4
    expected, XtX, Xty, yty, ysum, wsum = explicit_forgetting_reference(
        X, y, decay=decay, sample_weight=weights, ridge=ridge
    )

    rowwise = ForgettingOLS(decay=decay, ridge=ridge)
    for x_row, y_value, weight in zip(X, y, weights, strict=True):
        rowwise.push(x_row, y_value, sample_weight=weight)
    assert_params_equal(rowwise, expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rowwise.xtx_, XtX, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rowwise.xty_, Xty, atol=1e-10, rtol=1e-10)
    assert rowwise.yty_ == pytest.approx(yty, abs=1e-10)
    assert rowwise.weight_sum_ == pytest.approx(wsum, abs=1e-10)
    rss = float(yty - 2.0 * expected @ Xty + expected @ XtX @ expected)
    tss = float(yty - ysum * ysum / wsum)
    diag = rowwise.diagnostics_
    assert diag.n_observations == len(y)
    assert diag.weight_sum == pytest.approx(wsum, abs=1e-10)
    assert diag.weighted_mean_y == pytest.approx(ysum / wsum, abs=1e-10)
    assert diag.residual_sum_squares == pytest.approx(rss, abs=1e-10)
    assert diag.total_sum_squares == pytest.approx(tss, abs=1e-10)

    chunked = ForgettingOLS(decay=decay, ridge=ridge)
    chunked.partial_fit(X[:20], y[:20], sample_weight=weights[:20])
    chunked.partial_fit(X[20:77], y[20:77], sample_weight=weights[20:77])
    chunked.partial_fit(X[77:], y[77:], sample_weight=weights[77:])
    assert_params_equal(chunked, expected, atol=1e-10, rtol=1e-10)


def test_forgetting_decay_one_matches_incremental_and_invalid_decay_raises():
    X, y, _, _ = make_data(n=80, p=3, seed=606)
    inc = IncrementalOLS(ridge=0.3).fit(X, y)
    forget = ForgettingOLS(decay=1.0, ridge=0.3).fit(X[:30], y[:30])
    forget.partial_fit(X[30:], y[30:])
    assert_params_equal(forget, inc.params_)

    for decay in [-0.1, 0.0, 1.1, math.inf]:
        with pytest.raises(ValueError, match="decay"):
            ForgettingOLS(decay=decay)


def test_long_rolling_and_forgetting_stream_stays_finite_and_symmetric():
    X, y, _, _ = make_data(n=1_500, p=4, seed=1234, noise=0.2)
    rolling = RollingOLS(window=75, ridge=0.01, recompute_every=200)
    forgetting = ForgettingOLS(decay=0.995, ridge=0.01)

    for x_row, y_value in zip(X, y, strict=True):
        rolling.push(x_row, y_value)
        forgetting.push(x_row, y_value)

    for model in [rolling, forgetting]:
        assert np.all(np.isfinite(model.params_))
        np.testing.assert_allclose(model.xtx_, model.xtx_.T, atol=1e-9, rtol=1e-12)
        assert model.yty_ >= 0

    rolling_expected = reference_params(X[-75:], y[-75:], ridge=0.01)
    assert_params_equal(rolling, rolling_expected, atol=1e-9, rtol=1e-9)

    (
        forgetting_expected,
        forgetting_XtX,
        forgetting_Xty,
        forgetting_yty,
        _,
        _,
    ) = explicit_forgetting_reference(X, y, decay=0.995, ridge=0.01)
    assert_params_equal(forgetting, forgetting_expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(forgetting.xtx_, forgetting_XtX, atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(forgetting.xty_, forgetting_Xty, atol=1e-8, rtol=1e-9)
    assert forgetting.yty_ == pytest.approx(forgetting_yty, abs=1e-8)
