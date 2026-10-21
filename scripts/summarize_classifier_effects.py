"""Summarize whether zero-variance filtering changed classifier CV metrics."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run() -> pd.DataFrame:
    source = ROOT / "results/tables/classifier_grouped_cv_summary.csv"
    frame = pd.read_csv(source)
    values = ["accuracy_mean", "macro_f1_mean", "floor_accuracy_given_correct_building"]
    full = frame[frame.feature_set == "full"].set_index(["model", "target"])
    filtered = frame[frame.feature_set == "variance_filtered"].set_index(
        ["model", "target"]
    )
    rows = []
    for key in full.index:
        row = {"model": key[0], "target": key[1]}
        for metric in values:
            before, after = full.loc[key, metric], filtered.loc[key, metric]
            row[f"{metric}_full"] = before
            row[f"{metric}_filtered"] = after
            row[f"{metric}_delta"] = after - before
        row["features_full"] = full.loc[key, "n_features_mean"]
        row["features_filtered_mean"] = filtered.loc[key, "n_features_mean"]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results/tables/classifier_variance_effect.csv", index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
