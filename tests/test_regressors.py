"""Unit tests for Member 4's coordinate regressors and target scaler."""

import numpy as np
import pytest

from src.models.ml.regressors import (
    CoordinateTargetScaler,
    HierarchicalCoordinateRegressor,
    compute_position_metrics,
    make_base_regressor,
)


def test_coordinate_target_scaler_standardization_and_inversion():
    y = np.array(
        [
            [-7500.0, 4864800.0],
            [-7400.0, 4864900.0],
            [-7300.0, 4865000.0],
        ]
    )
    scaler = CoordinateTargetScaler()
    scaled = scaler.fit_transform(y)

    assert scaled.shape == (3, 2)
    # Check mean is approximately zero and std is approximately 1
    np.testing.assert_allclose(np.mean(scaled, axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(np.std(scaled, axis=0), [1.0, 1.0], atol=1e-7)

    # Invert and check exact recovery
    recovered = scaler.inverse_transform(scaled)
    np.testing.assert_allclose(recovered, y, atol=1e-5)


def test_scaler_refuses_transform_before_fit():
    scaler = CoordinateTargetScaler()
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        scaler.transform(np.zeros((2, 2)))


def test_position_metrics_computation():
    y_true = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],  # dist = 5.0
        ]
    )
    y_pred = np.array(
        [
            [0.0, 0.0],  # dist = 0.0
            [0.0, 0.0],
        ]
    )
    metrics = compute_position_metrics(y_true, y_pred)
    assert metrics["mean_error"] == 2.5
    assert metrics["median_error"] == 2.5
    assert metrics["mae_longitude"] == 1.5
    assert metrics["mae_latitude"] == 2.0


def test_hierarchical_regressor_routing_and_fallbacks():
    X = np.random.RandomState(42).randn(30, 5)
    # Coordinates centered around 100, 200
    y = np.random.RandomState(42).randn(30, 2) + np.array([100.0, 200.0])
    buildings = np.array([0] * 15 + [1] * 15)
    floors = np.array([0] * 8 + [1] * 7 + [0] * 15)

    base = make_base_regressor("knn", n_neighbors=2)
    model = HierarchicalCoordinateRegressor(
        base, routing="per_building_floor", min_samples=5
    )
    model.fit(X, y, buildings=buildings, floors=floors)

    # Prediction with matching building & floor
    preds = model.predict(X[:5], route_buildings=buildings[:5], route_floors=floors[:5])
    assert preds.shape == (5, 2)

    # Fallback when test has unseen building or floor
    unseen_b = np.array([99])
    unseen_f = np.array([99])
    fallback_pred = model.predict(
        X[:1], route_buildings=unseen_b, route_floors=unseen_f
    )
    assert fallback_pred.shape == (1, 2)
