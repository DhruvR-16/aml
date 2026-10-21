"""Train Phase 1 building/floor classifiers on shared USERID folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.load import load_data  # noqa: E402
from src.features.transform import FoldFeatureBuilder  # noqa: E402
from src.models.ml.classifiers import make_classifier  # noqa: E402

TARGETS = {"building": "BUILDINGID", "floor": "FLOOR"}


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, _, _ = load_data()
    folds = json.loads((ROOT / "data/splits/folds.json").read_text())["folds"]
    metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    for model_name in ("random_forest", "xgboost"):
        for feature_set, threshold in (("full", None), ("variance_filtered", 0.0)):
            for fold in folds:
                fold_id = fold["fold"]
                tr = np.asarray(fold["train_idx"])
                va = np.asarray(fold["valid_idx"])
                fit_df, valid_df = train.iloc[tr], train.iloc[va]
                builder = FoldFeatureBuilder(
                    representation="linear_positive", variance_threshold=threshold
                )
                x_train = builder.fit_transform(fit_df)
                x_valid = builder.transform(valid_df)
                feature_names = builder.get_feature_names_out(fit_df)
                fold_predictions: dict[str, np.ndarray] = {}
                fold_importances: dict[str, np.ndarray] = {}
                for task, target in TARGETS.items():
                    estimator = make_classifier(model_name)
                    y_train = fit_df[target].to_numpy()
                    y_valid = valid_df[target].to_numpy()
                    estimator.fit(x_train, y_train)
                    y_hat = estimator.predict(x_valid)
                    fold_predictions[task] = y_hat
                    fold_importances[task] = estimator.feature_importances_
                    metrics.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set,
                            "target": task,
                            "fold": fold_id,
                            "accuracy": accuracy_score(y_valid, y_hat),
                            "macro_f1": f1_score(
                                y_valid, y_hat, average="macro", zero_division=0
                            ),
                            "n_features": x_train.shape[1],
                            "train_rows": len(tr),
                            "validation_rows": len(va),
                        }
                    )
                    importance_kind = (
                        "mean_decrease_impurity"
                        if model_name == "random_forest"
                        else "gain"
                    )
                    importances.append(
                        pd.DataFrame(
                            {
                                "model": model_name,
                                "feature_set": feature_set,
                                "target": task,
                                "fold": fold_id,
                                "feature": feature_names,
                                "importance": estimator.feature_importances_,
                                "importance_kind": importance_kind,
                            }
                        )
                    )
                pred_frame = pd.DataFrame(
                    {
                        "model": model_name,
                        "feature_set": feature_set,
                        "fold": fold_id,
                        "row_index": va,
                        "BUILDINGID": valid_df.BUILDINGID.to_numpy(),
                        "BUILDINGID_PRED": fold_predictions["building"],
                        "FLOOR": valid_df.FLOOR.to_numpy(),
                        "FLOOR_PRED": fold_predictions["floor"],
                    }
                )
                predictions.append(pred_frame)
                print(
                    f"finished {model_name} / {feature_set} / fold {fold_id} ({x_train.shape[1]} features)",
                    flush=True,
                )

    metric_frame = pd.DataFrame(metrics)
    pred_frame = pd.concat(predictions, ignore_index=True)
    pred_frame["building_correct"] = pred_frame.BUILDINGID.eq(
        pred_frame.BUILDINGID_PRED
    )
    floor_correct = pred_frame.FLOOR.eq(pred_frame.FLOOR_PRED)
    pred_frame["floor_correct"] = floor_correct
    conditional_values = (
        pred_frame.loc[pred_frame.building_correct]
        .groupby(["model", "feature_set", "fold"])
        .floor_correct.mean()
    )
    metric_rows = []
    for model_name in ("random_forest", "xgboost"):
        for feature_set in ("full", "variance_filtered"):
            for fold in folds:
                fold_id = fold["fold"]
                for task in TARGETS:
                    row = (
                        metric_frame[
                            (metric_frame.model == model_name)
                            & (metric_frame.feature_set == feature_set)
                            & (metric_frame.target == task)
                            & (metric_frame.fold == fold_id)
                        ]
                        .iloc[0]
                        .to_dict()
                    )
                    row["floor_accuracy_given_correct_building"] = (
                        conditional_values.get(
                            (model_name, feature_set, fold_id), np.nan
                        )
                        if task == "floor"
                        else np.nan
                    )
                    metric_rows.append(row)
    metric_frame = pd.DataFrame(metric_rows)

    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = ROOT / "results/preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(out / "classifier_grouped_cv_folds.csv", index=False)
    pred_frame.drop(columns=["building_correct", "floor_correct"]).to_csv(
        pred_dir / "classifiers_grouped_cv.csv", index=False
    )
    metric_summary = (
        metric_frame.groupby(["model", "feature_set", "target"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_features_mean=("n_features", "mean"),
            floor_accuracy_given_correct_building=(
                "floor_accuracy_given_correct_building",
                "mean",
            ),
        )
        .reset_index()
    )
    metric_summary.to_csv(out / "classifier_grouped_cv_summary.csv", index=False)

    importance_frame = pd.concat(importances, ignore_index=True)
    importance_summary = importance_frame.groupby(
        ["model", "feature_set", "target", "feature", "importance_kind"], as_index=False
    ).importance.mean()
    importance_summary.to_csv(out / "classifier_feature_importance.csv", index=False)
    plot_importance(importance_summary)
    print(metric_summary.to_string(index=False))
    return metric_frame, importance_summary


def plot_importance(importance: pd.DataFrame, top_n: int = 15) -> None:
    selected = importance[importance.feature_set == "full"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for row, model in enumerate(("random_forest", "xgboost")):
        for col, task in enumerate(("building", "floor")):
            ax = axes[row, col]
            part = selected[(selected.model == model) & (selected.target == task)]
            top = part.nlargest(top_n, "importance").sort_values("importance")
            ax.barh(top.feature, top.importance, color="#3977a8")
            ax.set_title(f"{model.replace('_', ' ').title()} · {task.title()}")
            ax.set_xlabel("Mean feature importance across folds")
            ax.set_ylabel("")
    fig.suptitle("Phase 1 classification feature importance (full feature set)")
    fig.tight_layout()
    output = ROOT / "results/figures/classifier_feature_importance.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
