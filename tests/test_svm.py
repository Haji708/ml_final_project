import numpy as np
import pytest

from src.svm import PegasosSVM


def make_data():
    X = np.array([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [2.0, 1.0],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_to_pm1_maps_two_classes():
    y_pm1, classes = PegasosSVM._to_pm1(np.array(["standard", "premium"]))

    np.testing.assert_array_equal(classes, ["premium", "standard"])
    np.testing.assert_array_equal(y_pm1, [1, -1])


def test_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="exactly 2"):
        PegasosSVM._to_pm1([0, 1, 2])


def test_linear_svm_trains_and_predicts():
    X, y = make_data()

    model = PegasosSVM(
        lambda_=1e-3,
        n_iters=5_000,
        random_state=42,
        record_every=100,
    )
    model.fit(X, y)

    predictions = model.predict(X)
    scores = model.decision_function(X)

    assert predictions.shape == y.shape
    assert scores.shape == y.shape
    assert set(predictions).issubset({0, 1})
    assert model.w.shape == (X.shape[1],)
    assert len(model.objective_history_) > 1


def test_objective_history_is_finite_and_improves():
    X, y = make_data()

    model = PegasosSVM(
        lambda_=1e-3,
        n_iters=5_000,
        random_state=42,
        record_every=100,
    ).fit(X, y)

    history = np.asarray(model.objective_history_)

    assert np.isfinite(history).all()
    assert history[-1] < history[0]


def test_margin_width_and_support_vectors():
    X, y = make_data()

    model = PegasosSVM(
        lambda_=1e-3,
        n_iters=5_000,
        random_state=42,
    ).fit(X, y)

    expected_width = 2.0 / np.linalg.norm(model.w)
    assert np.isclose(model.margin_width_, expected_width)

    y_pm1, _ = model._to_pm1(y)
    margins = y_pm1 * model.decision_function(X)

    np.testing.assert_array_equal(
        model.support_indices_,
        np.flatnonzero(margins <= 1.0 + 1e-6),
    )
    np.testing.assert_allclose(
        model.support_margins_,
        margins[model.support_indices_],
    )


def test_rff_svm_trains_and_predicts():
    X, y = make_data()

    model = PegasosSVM(
        lambda_=1e-3,
        n_iters=5_000,
        random_state=42,
        feature_map="rff",
        n_components=50,
        gamma=1.0,
    ).fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert model.w.shape == (50,)
    assert model.rff_weights_.shape == (X.shape[1], 50)
    assert model.rff_offset_.shape == (50,)
    y_pm1, _ = model._to_pm1(y)
    margins = y_pm1 * model.decision_function(X)

    np.testing.assert_array_equal(
        model.support_indices_,
        np.flatnonzero(margins <= 1.0 + 1e-6),
    )
    np.testing.assert_allclose(
        model.support_margins_,
        margins[model.support_indices_],
    )

def test_same_seed_is_reproducible():
    X, y = make_data()

    first = PegasosSVM(
        lambda_=1e-3,
        n_iters=2_000,
        random_state=42,
    ).fit(X, y)

    second = PegasosSVM(
        lambda_=1e-3,
        n_iters=2_000,
        random_state=42,
    ).fit(X, y)

    np.testing.assert_allclose(first.w, second.w)
    assert first.b == second.b
    np.testing.assert_allclose(
        first.objective_history_,
        second.objective_history_,
    )


def test_wrong_feature_count_is_rejected():
    X, y = make_data()

    model = PegasosSVM(n_iters=100).fit(X, y)

    with pytest.raises(ValueError, match="Expected input with"):
        model.predict(np.ones((2, 3)))