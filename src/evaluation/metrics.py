"""Common classification and indoor-localization metrics.

The evaluator accepts either canonical prediction names (``*_TRUE`` and
``*_PRED``) or the classifier CSV contract (``BUILDINGID``/``FLOOR`` for
truth). Coordinate values and all derived distances are in metres.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _column(frame: pd.DataFrame, *candidates: str) -> pd.Series | None:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return None


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Compute all available metrics from one model's prediction rows.

    Missing prediction types produce NaN metrics. Hierarchical shares and mean
    errors are conditional on both coordinate and categorical predictions being
    available. The EvAAL-style score is the mean position error plus 50 m for a
    wrong building and 4 m for a wrong floor, as specified in the project guide.
    """
    if predictions.empty:
        raise ValueError("predictions must contain at least one row")

    result: dict[str, float | int] = {"n_samples": len(predictions)}
    b_true = _column(predictions, "BUILDINGID_TRUE", "BUILDINGID")
    b_pred = _column(predictions, "BUILDINGID_PRED")
    f_true = _column(predictions, "FLOOR_TRUE", "FLOOR")
    f_pred = _column(predictions, "FLOOR_PRED")
    lon_true = _column(predictions, "LONGITUDE_TRUE")
    lat_true = _column(predictions, "LATITUDE_TRUE")
    lon_pred = _column(predictions, "LONGITUDE_PRED")
    lat_pred = _column(predictions, "LATITUDE_PRED")

    building_ok = floor_ok = None
    building_valid = floor_valid = None
    if b_true is not None and b_pred is not None:
        building_valid = b_true.notna().to_numpy() & b_pred.notna().to_numpy()
        building_ok = np.zeros(len(predictions), dtype=bool)
        building_ok[building_valid] = (
            b_true.to_numpy()[building_valid] == b_pred.to_numpy()[building_valid]
        )
        result["building_accuracy"] = (
            float(np.mean(building_ok[building_valid]))
            if np.any(building_valid)
            else float("nan")
        )
    else:
        result["building_accuracy"] = float("nan")
    if f_true is not None and f_pred is not None:
        floor_valid = f_true.notna().to_numpy() & f_pred.notna().to_numpy()
        floor_ok = np.zeros(len(predictions), dtype=bool)
        floor_ok[floor_valid] = (
            f_true.to_numpy()[floor_valid] == f_pred.to_numpy()[floor_valid]
        )
        result["floor_accuracy"] = (
            float(np.mean(floor_ok[floor_valid]))
            if np.any(floor_valid)
            else float("nan")
        )
        common_valid = (
            floor_valid & building_valid
            if building_valid is not None
            else np.zeros(len(predictions), dtype=bool)
        )
        correct_building = (
            common_valid & building_ok
            if building_ok is not None
            else np.zeros(len(predictions), dtype=bool)
        )
        if np.any(correct_building):
            result["floor_accuracy_given_correct_building"] = float(
                np.mean(floor_ok[correct_building])
            )
        else:
            result["floor_accuracy_given_correct_building"] = float("nan")
    else:
        result["floor_accuracy"] = float("nan")
        result["floor_accuracy_given_correct_building"] = float("nan")

    has_coords = all(v is not None for v in (lon_true, lat_true, lon_pred, lat_pred))
    if has_coords:
        delta_lon = lon_pred.to_numpy(dtype=float) - lon_true.to_numpy(dtype=float)
        delta_lat = lat_pred.to_numpy(dtype=float) - lat_true.to_numpy(dtype=float)
        errors = np.hypot(delta_lon, delta_lat)
        result.update(
            {
                "mae_longitude": float(np.mean(np.abs(delta_lon))),
                "mae_latitude": float(np.mean(np.abs(delta_lat))),
                "mean_position_error": float(np.mean(errors)),
                "median_position_error": float(np.median(errors)),
                "p75_position_error": float(np.percentile(errors, 75)),
                "p95_position_error": float(np.percentile(errors, 95)),
            }
        )
    else:
        errors = None
        for name in (
            "mae_longitude",
            "mae_latitude",
            "mean_position_error",
            "median_position_error",
            "p75_position_error",
            "p95_position_error",
        ):
            result[name] = float("nan")

    hierarchy_valid = (
        building_valid & floor_valid
        if building_valid is not None and floor_valid is not None
        else np.zeros(len(predictions), dtype=bool)
    )
    has_hierarchy = (
        building_ok is not None
        and floor_ok is not None
        and errors is not None
        and np.any(hierarchy_valid)
    )
    if has_hierarchy:
        wrong_building = hierarchy_valid & ~building_ok
        building_right_floor_wrong = hierarchy_valid & building_ok & ~floor_ok
        both_right = hierarchy_valid & building_ok & floor_ok
        for name, mask in (
            ("building_wrong", wrong_building),
            ("building_right_floor_wrong", building_right_floor_wrong),
            ("building_and_floor_right", both_right),
        ):
            result[f"{name}_share"] = float(np.mean(mask[hierarchy_valid]))
            result[f"{name}_mean_position_error"] = (
                float(np.mean(errors[mask])) if np.any(mask) else float("nan")
            )
        result["evaal_style_score"] = float(
            np.mean(
                errors[hierarchy_valid]
                + 50.0 * wrong_building[hierarchy_valid]
                + 4.0 * ~floor_ok[hierarchy_valid]
            )
        )
    else:
        for name in (
            "building_wrong_share",
            "building_wrong_mean_position_error",
            "building_right_floor_wrong_share",
            "building_right_floor_wrong_mean_position_error",
            "building_and_floor_right_share",
            "building_and_floor_right_mean_position_error",
            "evaal_style_score",
        ):
            result[name] = float("nan")
    return result


def metrics_from_columns(
    predictions: pd.DataFrame, *, aliases: Mapping[str, str] | None = None
) -> dict[str, float | int]:
    """Convenience wrapper for dataframes whose columns need explicit aliases."""
    if not aliases:
        return evaluate_predictions(predictions)
    renamed = predictions.rename(columns=dict(aliases))
    return evaluate_predictions(renamed)
