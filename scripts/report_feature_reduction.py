"""Report fold-0 PCA variance and zero-variance feature selection counts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.load import load_data  # noqa: E402
from src.features.transform import FoldFeatureBuilder  # noqa: E402


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, _, _ = load_data()
    folds = json.loads((ROOT / "data/splits/folds.json").read_text())["folds"]
    fold = folds[0]
    fit = train.iloc[fold["train_idx"]]
    validation = train.iloc[fold["valid_idx"]]
    builder = FoldFeatureBuilder(representation="linear_positive")
    x_fit = builder.fit_transform(fit)
    # Apply fitted state to the held-out inner fold to make the no-refit contract explicit.
    x_valid = builder.transform(validation)
    pca = PCA(n_components=200, random_state=42).fit(x_fit)
    variance = pd.DataFrame(
        {
            "fold": [0, 0, 0],
            "components": [50, 100, 200],
            "cumulative_explained_variance": [
                float(pca.explained_variance_ratio_[:k].sum()) for k in (50, 100, 200)
            ],
            "fit_rows": [len(fit)] * 3,
            "input_features": [x_fit.shape[1]] * 3,
        }
    )
    selector = VarianceThreshold(threshold=0.0).fit(x_fit)
    selection = pd.DataFrame(
        [
            {
                "fold": 0,
                "threshold": 0.0,
                "input_features": x_fit.shape[1],
                "selected_features": int(selector.get_support().sum()),
                "validation_rows_transformed": x_valid.shape[0],
            }
        ]
    )
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    variance.to_csv(out / "pca_explained_variance.csv", index=False)
    selection.to_csv(out / "variance_threshold_counts.csv", index=False)
    print(variance.to_string(index=False))
    print(selection.to_string(index=False))
    return variance, selection


if __name__ == "__main__":
    run()
