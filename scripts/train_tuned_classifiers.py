"""Nested USERID-grouped tuning and OOF evaluation for M3 Phase 1 classifiers.

Hyperparameters are selected by inner-fold macro-F1. Every inner split fits its
own feature builder, so feature scaling and variance filtering never see the
inner validation users. Outer folds remain untouched for the reported OOF scores.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_data  # noqa: E402
from src.features.transform import FoldFeatureBuilder  # noqa: E402
from src.models.ml.classifiers import make_classifier  # noqa: E402

TARGETS = {
    "building": ("BUILDINGID", "BUILDINGID_PRED"),
    "floor": ("FLOOR", "FLOOR_PRED"),
}
GRIDS = {
    "random_forest": [
        {"n_estimators": 200, "max_features": "sqrt", "min_samples_leaf": 1},
        {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 2},
        {"n_estimators": 300, "max_features": 0.2, "min_samples_leaf": 1},
    ],
    "xgboost": [
        {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05},
        {"n_estimators": 250, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
    ],
}


def _fit_predict(
    model: str,
    params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    seed: int,
) -> np.ndarray:
    y_train = np.asarray(y_train)
    if model == "xgboost":
        # Some USERID inner folds contain only a subset of building labels.
        # XGBoost requires encoded labels to be consecutive starting from zero.
        classes = np.unique(y_train)
        if len(classes) == 2:
            estimator = make_classifier(
                model, seed=seed, objective="binary:logistic", num_class=None, **params
            )
        else:
            estimator = make_classifier(
                model,
                seed=seed,
                objective="multi:softprob",
                num_class=len(classes),
                **params,
            )
        estimator.fit(x_train, np.searchsorted(classes, y_train))
        encoded = np.asarray(estimator.predict(x_valid)).astype(int).reshape(-1)
        return classes[encoded]
    estimator = make_classifier(model, seed=seed, **params)
    estimator.fit(x_train, y_train)
    return estimator.predict(x_valid)


def _score_params(
    model: str,
    target: str,
    fit_df: pd.DataFrame,
    variance_threshold: float | None,
    params: dict,
    inner_splits: list,
    seed: int,
) -> tuple[float, float]:
    labels, _ = TARGETS[target]
    true_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    for in_tr, in_va in inner_splits:
        tr_df, va_df = fit_df.iloc[in_tr], fit_df.iloc[in_va]
        builder = FoldFeatureBuilder(
            representation="linear_positive", variance_threshold=variance_threshold
        )
        x_tr = builder.fit_transform(tr_df)
        x_va = builder.transform(va_df)
        y_true = va_df[labels].to_numpy()
        y_pred = _fit_predict(model, params, x_tr, tr_df[labels].to_numpy(), x_va, seed)
        true_all.append(y_true)
        pred_all.append(y_pred)
    y_true, y_pred = np.concatenate(true_all), np.concatenate(pred_all)
    return (
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        float(accuracy_score(y_true, y_pred)),
    )


def run(
    seed: int = 42,
    inner_splits_count: int = 3,
    variance_threshold: float | None = 0.0,
    models_to_run: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, _, _ = load_data()
    folds = json.loads((ROOT / "data/splits/folds.json").read_text())["folds"]
    metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []
    selection: list[dict] = []
    started = time.time()

    selected_models = models_to_run or list(GRIDS)
    for model in selected_models:
        if model not in GRIDS:
            raise ValueError(f"Unknown model {model}; choose from {sorted(GRIDS)}")
        grid = GRIDS[model]
        model_metrics: list[dict] = []
        model_predictions: list[pd.DataFrame] = []
        model_selection: list[dict] = []
        for outer in folds:
            fold_id = int(outer["fold"])
            tr_idx = np.asarray(outer["train_idx"], dtype=int)
            va_idx = np.asarray(outer["valid_idx"], dtype=int)
            fit_df, valid_df = train.iloc[tr_idx], train.iloc[va_idx]
            groups = fit_df.USERID.to_numpy()
            n_inner = min(inner_splits_count, len(np.unique(groups)))
            inner_cv = GroupKFold(n_splits=n_inner)
            inner = list(inner_cv.split(fit_df, groups=groups))

            best_by_target: dict[str, dict] = {}
            for target in TARGETS:
                scored = []
                for params in grid:
                    macro_f1, accuracy = _score_params(
                        model,
                        target,
                        fit_df,
                        variance_threshold,
                        params,
                        inner,
                        seed,
                    )
                    scored.append((macro_f1, accuracy, params))
                best_f1, best_acc, best_params = max(
                    scored, key=lambda item: (item[0], item[1])
                )
                best_by_target[target] = best_params
                selected_record = {
                    "model": model,
                    "feature_set": "tuned_variance_filtered",
                    "target": target,
                    "fold": fold_id,
                    "inner_macro_f1": best_f1,
                    "inner_accuracy": best_acc,
                    **best_params,
                }
                selection.append(selected_record)
                model_selection.append(selected_record)

            builder = FoldFeatureBuilder(
                representation="linear_positive", variance_threshold=variance_threshold
            )
            x_train = builder.fit_transform(fit_df)
            x_valid = builder.transform(valid_df)
            predicted: dict[str, np.ndarray] = {}
            for target, (true_col, pred_col) in TARGETS.items():
                y_true = valid_df[true_col].to_numpy()
                y_pred = _fit_predict(
                    model,
                    best_by_target[target],
                    x_train,
                    fit_df[true_col].to_numpy(),
                    x_valid,
                    seed,
                )
                predicted[target] = y_pred
                metric_record = {
                    "model": model,
                    "feature_set": "tuned_variance_filtered",
                    "target": target,
                    "fold": fold_id,
                    "accuracy": accuracy_score(y_true, y_pred),
                    "macro_f1": f1_score(
                        y_true, y_pred, average="macro", zero_division=0
                    ),
                    "n_features": x_train.shape[1],
                    "train_rows": len(tr_idx),
                    "validation_rows": len(va_idx),
                    **{f"selected_{k}": v for k, v in best_by_target[target].items()},
                }
                metrics.append(metric_record)
                model_metrics.append(metric_record)

            pred_frame = pd.DataFrame(
                {
                    "model": model,
                    "feature_set": "tuned_variance_filtered",
                    "fold": fold_id,
                    "row_index": va_idx,
                    "BUILDINGID": valid_df.BUILDINGID.to_numpy(),
                    "BUILDINGID_PRED": predicted["building"],
                    "FLOOR": valid_df.FLOOR.to_numpy(),
                    "FLOOR_PRED": predicted["floor"],
                }
            )
            predictions.append(pred_frame)
            model_predictions.append(pred_frame)
            print(
                f"finished nested {model} / fold {fold_id}; elapsed {time.time()-started:.1f}s",
                flush=True,
            )

        # Save checkpoints so completed models survive an interrupted run.
        out_preds = ROOT / "results/preds"
        out_tables = ROOT / "results/tables"
        pd.concat(model_predictions, ignore_index=True).to_csv(
            out_preds / f"classifiers_tuned_{model}_grouped_cv.csv", index=False
        )
        pd.DataFrame(model_metrics).to_csv(
            out_tables / f"classifier_tuned_{model}_grouped_cv_folds.csv", index=False
        )
        (
            out_tables / f"classifier_tuned_{model}_selected_hyperparameters.json"
        ).write_text(json.dumps(model_selection, indent=2) + "\n")

    # Reassemble completed model checkpoints, allowing an interrupted model
    # search to resume without discarding previous folds/models.
    completed = [
        model
        for model in GRIDS
        if (ROOT / f"results/preds/classifiers_tuned_{model}_grouped_cv.csv").exists()
    ]
    metric_frame = pd.concat(
        [
            pd.read_csv(
                ROOT / f"results/tables/classifier_tuned_{model}_grouped_cv_folds.csv"
            )
            for model in completed
        ],
        ignore_index=True,
    )
    pred_frame = pd.concat(
        [
            pd.read_csv(
                ROOT / f"results/preds/classifiers_tuned_{model}_grouped_cv.csv"
            )
            for model in completed
        ],
        ignore_index=True,
    )
    pred_frame.to_csv(
        ROOT / "results/preds/classifiers_tuned_grouped_cv.csv", index=False
    )
    metric_frame.to_csv(
        ROOT / "results/tables/classifier_tuned_grouped_cv_folds.csv", index=False
    )
    summary = metric_frame.groupby(
        ["model", "feature_set", "target"], as_index=False
    ).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        n_features_mean=("n_features", "mean"),
    )
    summary.to_csv(
        ROOT / "results/tables/classifier_tuned_grouped_cv_summary.csv", index=False
    )
    (ROOT / "results/tables/classifier_tuned_selected_hyperparameters.json").write_text(
        json.dumps(
            [
                item
                for model in completed
                for item in json.loads(
                    (
                        ROOT
                        / f"results/tables/classifier_tuned_{model}_selected_hyperparameters.json"
                    ).read_text()
                )
            ],
            indent=2,
        )
        + "\n"
    )
    return metric_frame, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=sorted(GRIDS), default=sorted(GRIDS)
    )
    args = parser.parse_args()
    run(models_to_run=args.models)
