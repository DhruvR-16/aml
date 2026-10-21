"""Create pooled out-of-fold confusion matrices and class-support tables for M3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build(
    prediction_name: str = "classifiers_grouped_cv.csv",
    output_prefix: str = "classifier",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_path = ROOT / "results/preds" / prediction_name
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing grouped classifier predictions: {prediction_path}"
        )
    predictions = pd.read_csv(prediction_path)
    figure_dir = ROOT / "results/figures"
    table_dir = ROOT / "results/tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    long_rows: list[dict] = []
    support_rows: list[dict] = []

    for (model, feature_set), part in predictions.groupby(["model", "feature_set"]):
        for task, truth_col, pred_col in (
            ("building", "BUILDINGID", "BUILDINGID_PRED"),
            ("floor", "FLOOR", "FLOOR_PRED"),
        ):
            labels = sorted(
                set(part[truth_col].dropna()) | set(part[pred_col].dropna())
            )
            raw = confusion_matrix(part[truth_col], part[pred_col], labels=labels)
            row_sums = raw.sum(axis=1, keepdims=True)
            normalized = raw / row_sums.clip(min=1)
            for i, actual in enumerate(labels):
                support_rows.append(
                    {
                        "model": model,
                        "feature_set": feature_set,
                        "target": task,
                        "class": actual,
                        "support": int(raw[i].sum()),
                        "recall": float(normalized[i, i]),
                    }
                )
                for j, predicted in enumerate(labels):
                    long_rows.append(
                        {
                            "model": model,
                            "feature_set": feature_set,
                            "target": task,
                            "actual": actual,
                            "predicted": predicted,
                            "count": int(raw[i, j]),
                            "row_normalized_rate": float(normalized[i, j]),
                        }
                    )

            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(
                normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0,
                vmax=1,
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={"label": "Recall fraction"},
                ax=ax,
            )
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(
                f"{model.replace('_', ' ').title()} · {feature_set} · {task.title()} (OOF)"
            )
            fig.tight_layout()
            fig.savefig(
                figure_dir / f"confusion_{model}_{feature_set}_{task}.png", dpi=180
            )
            plt.close(fig)

    matrices = pd.DataFrame(long_rows)
    support = pd.DataFrame(support_rows)
    matrices.to_csv(table_dir / f"{output_prefix}_confusion_matrices.csv", index=False)
    support.to_csv(table_dir / f"{output_prefix}_class_support.csv", index=False)
    return matrices, support


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="classifiers_grouped_cv.csv")
    parser.add_argument("--output-prefix", default="classifier")
    args = parser.parse_args()
    _, support = build(args.predictions, args.output_prefix)
    print(support.to_string(index=False))
