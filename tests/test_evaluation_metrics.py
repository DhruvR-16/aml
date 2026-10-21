import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_predictions


def test_metrics_on_hand_computed_hierarchical_example():
    rows = pd.DataFrame(
        {
            "BUILDINGID_TRUE": [0, 0, 1],
            "BUILDINGID_PRED": [0, 1, 1],
            "FLOOR_TRUE": [0, 1, 1],
            "FLOOR_PRED": [1, 0, 1],
            "LONGITUDE_TRUE": [0.0, 0.0, 3.0],
            "LATITUDE_TRUE": [0.0, 0.0, 4.0],
            "LONGITUDE_PRED": [3.0, 0.0, 0.0],
            "LATITUDE_PRED": [4.0, 2.0, 0.0],
        }
    )

    metrics = evaluate_predictions(rows)

    assert metrics["building_accuracy"] == 2 / 3
    assert metrics["floor_accuracy"] == 1 / 3
    assert metrics["floor_accuracy_given_correct_building"] == 1 / 2
    assert metrics["mae_longitude"] == 2.0
    assert np.isclose(metrics["mae_latitude"], 10 / 3)
    assert np.isclose(metrics["mean_position_error"], (5 + 2 + 5) / 3)
    assert metrics["building_wrong_share"] == 1 / 3
    assert metrics["building_right_floor_wrong_share"] == 1 / 3
    assert metrics["building_and_floor_right_share"] == 1 / 3
    assert np.isclose(metrics["evaal_style_score"], (5 + 4 + 2 + 50 + 4 + 5) / 3)


def test_classifier_only_rows_return_class_metrics_and_nan_for_coordinates():
    metrics = evaluate_predictions(
        pd.DataFrame(
            {
                "BUILDINGID": [0, 1],
                "BUILDINGID_PRED": [0, 0],
                "FLOOR": [0, 1],
                "FLOOR_PRED": [0, 1],
            }
        )
    )
    assert metrics["building_accuracy"] == 0.5
    assert metrics["floor_accuracy"] == 1.0
    assert np.isnan(metrics["mean_position_error"])
    assert np.isnan(metrics["evaal_style_score"])


def test_oracle_coordinates_do_not_claim_classifier_metrics():
    metrics = evaluate_predictions(
        pd.DataFrame(
            {
                "BUILDINGID_TRUE": [0],
                "BUILDINGID_PRED": [np.nan],
                "FLOOR_TRUE": [0],
                "FLOOR_PRED": [np.nan],
                "LONGITUDE_TRUE": [0.0],
                "LONGITUDE_PRED": [3.0],
                "LATITUDE_TRUE": [0.0],
                "LATITUDE_PRED": [4.0],
            }
        )
    )
    assert np.isnan(metrics["building_accuracy"])
    assert np.isnan(metrics["floor_accuracy"])
    assert np.isnan(metrics["evaal_style_score"])
    assert metrics["mean_position_error"] == 5.0
