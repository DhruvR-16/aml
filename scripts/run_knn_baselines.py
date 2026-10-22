"""Compare the three Member 2 RSSI representations with grouped-CV kNN."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.load import load_data  # noqa: E402
from src.features.transform import FoldFeatureBuilder  # noqa: E402


def run(k: int = 5) -> pd.DataFrame:
    train, _, _ = load_data()
    fold_file = ROOT / "data/splits/folds.json"
    if not fold_file.exists():
        raise FileNotFoundError("Run `python scripts/build_splits.py` first")
    folds = json.loads(fold_file.read_text())["folds"]
    rows = []
    for representation in ("linear_positive", "exponential", "powed"):
        for fold in folds:
            tr, va = np.asarray(fold["train_idx"]), np.asarray(fold["valid_idx"])
            builder = FoldFeatureBuilder(representation=representation)
            xtr = builder.fit_transform(train.iloc[tr])
            xva = builder.transform(train.iloc[va])
            building = KNeighborsClassifier(
                n_neighbors=k, weights="distance", n_jobs=-1
            )
            floor = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
            coords = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
            building.fit(xtr, train.BUILDINGID.iloc[tr])
            floor.fit(xtr, train.FLOOR.iloc[tr])
            coords.fit(xtr, train[["LONGITUDE", "LATITUDE"]].iloc[tr])
            pred_xy = coords.predict(xva)
            truth_xy = train[["LONGITUDE", "LATITUDE"]].to_numpy()[va]
            error = np.sqrt(np.square(pred_xy - truth_xy).sum(axis=1))
            rows.append(
                {
                    "representation": representation,
                    "fold": fold["fold"],
                    "k": k,
                    "building_accuracy": building.score(xva, train.BUILDINGID.iloc[va]),
                    "floor_accuracy": floor.score(xva, train.FLOOR.iloc[va]),
                    "coordinate_mae_longitude": np.abs(
                        pred_xy[:, 0] - truth_xy[:, 0]
                    ).mean(),
                    "coordinate_mae_latitude": np.abs(
                        pred_xy[:, 1] - truth_xy[:, 1]
                    ).mean(),
                    "mean_position_error_m": error.mean(),
                }
            )
    result = pd.DataFrame(rows)
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "knn_representation_folds.csv", index=False)
    metrics = [
        "building_accuracy",
        "floor_accuracy",
        "coordinate_mae_longitude",
        "coordinate_mae_latitude",
        "mean_position_error_m",
    ]
    summary = result.groupby("representation")[metrics].agg(["mean", "std"])
    summary.to_csv(out / "knn_representation_summary.csv")
    print(summary.to_string())
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    run(parser.parse_args().k)
