"""Combine the validation-selected M3 building and floor classifiers by task."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build() -> pd.DataFrame:
    path = ROOT / "results/preds/classifiers_tuned_grouped_cv.csv"
    predictions = pd.read_csv(path)
    building = predictions[
        (predictions.model == "random_forest")
        & (predictions.feature_set == "tuned_variance_filtered")
    ][["fold", "row_index", "BUILDINGID", "BUILDINGID_PRED"]].rename(
        columns={"BUILDINGID_PRED": "BUILDINGID_PRED_RF"}
    )
    floor = predictions[
        (predictions.model == "xgboost")
        & (predictions.feature_set == "tuned_variance_filtered")
    ][["fold", "row_index", "FLOOR", "FLOOR_PRED"]].rename(
        columns={"FLOOR_PRED": "FLOOR_PRED_XGB"}
    )
    result = building.merge(floor, on=["fold", "row_index"], validate="one_to_one")
    result["model"] = "task_best"
    result["feature_set"] = "tuned_task_best"
    result["BUILDINGID_PRED"] = result.pop("BUILDINGID_PRED_RF")
    result["FLOOR_PRED"] = result.pop("FLOOR_PRED_XGB")
    order = [
        "model",
        "feature_set",
        "fold",
        "row_index",
        "BUILDINGID",
        "BUILDINGID_PRED",
        "FLOOR",
        "FLOOR_PRED",
    ]
    result = result[order]
    out = ROOT / "results/preds/classifiers_task_best_grouped_cv.csv"
    result.to_csv(out, index=False)
    return result


if __name__ == "__main__":
    output = build()
    print(f"Saved {len(output):,} task-wise out-of-fold routes")
