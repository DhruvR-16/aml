import pandas as pd

from scripts.build_results_table import _normalize_m4


def test_global_m4_oof_rows_join_m3_predictions_and_oracle_is_not_scored_hierarchically():
    regressions = pd.DataFrame(
        {
            "model": ["knn", "knn"],
            "routing": ["global", "per_building"],
            "routing_mode": ["global", "oracle"],
            "fold": [0, 0],
            "row_index": [5, 5],
            "BUILDINGID_ROUTE": [-1, 0],
            "FLOOR_ROUTE": [-1, 1],
            "BUILDINGID_TRUE": [0, 0],
            "FLOOR_TRUE": [1, 1],
        }
    )
    classifier = pd.DataFrame(
        {
            "model": ["random_forest"],
            "feature_set": ["tuned_variance_filtered"],
            "fold": [0],
            "row_index": [5],
            "BUILDINGID_PRED": [0],
            "FLOOR_PRED": [1],
        }
    )

    result = _normalize_m4(regressions, classifier)

    assert result.loc[0, "BUILDINGID_PRED"] == 0
    assert result.loc[0, "FLOOR_PRED"] == 1
    assert pd.isna(result.loc[1, "BUILDINGID_PRED"])
    assert pd.isna(result.loc[1, "FLOOR_PRED"])
