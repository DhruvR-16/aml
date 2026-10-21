"""Train and evaluate Phase 1 coordinate regressors across 5 USERID folds.

Performs:
  1. Coordinate target scaling per training fold.
  2. Inner GroupKFold (by USERID) hyperparameter search for kNN, RF, and XGBoost.
  3. Fitting global, per-building, and per-building-floor models.
  4. Evaluating oracle vs end-to-end routing using Member 3's out-of-fold classifier predictions.
  5. Exporting fold-level predictions, metrics, and summary tables.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_data  # noqa: E402
from src.features.transform import FoldFeatureBuilder  # noqa: E402
from src.models.ml.regressors import (  # noqa: E402
    CoordinateTargetScaler,
    HierarchicalCoordinateRegressor,
    compute_position_metrics,
    make_base_regressor,
)


def tune_regressor_inner_grouped(
    model_name: str,
    fit_df: pd.DataFrame,
    user_ids: np.ndarray,
    param_grid: dict[str, list],
    variance_threshold: float | None = 0.0,
    n_splits: int = 3,
    seed: int = 42,
) -> dict:
    """Select parameters with fully nested, fold-fitted features and targets."""
    unique_users = np.unique(user_ids)
    if len(unique_users) < n_splits:
        # Not enough users for n_splits, return default first combination
        return {k: v[0] for k, v in param_grid.items()}

    keys = list(param_grid.keys())
    combinations = [dict(zip(keys, combo)) for combo in product(*param_grid.values())]

    # If only one combination, skip search
    if len(combinations) <= 1:
        return combinations[0] if combinations else {}

    gkf = GroupKFold(n_splits=n_splits)
    inner_splits = list(gkf.split(fit_df, groups=user_ids))
    y_raw = fit_df[["LONGITUDE", "LATITUDE"]].to_numpy()

    best_score = float("inf")
    best_params = combinations[0]

    for params in combinations:
        fold_errors = []
        for in_tr, in_va in inner_splits:
            inner_train = fit_df.iloc[in_tr]
            inner_valid = fit_df.iloc[in_va]
            builder = FoldFeatureBuilder(
                representation="linear_positive",
                variance_threshold=variance_threshold,
            )
            X_inner_train = builder.fit_transform(inner_train)
            X_inner_valid = builder.transform(inner_valid)
            target_scaler = CoordinateTargetScaler()
            y_inner_train = target_scaler.fit_transform(y_raw[in_tr])
            est = make_base_regressor(model_name, seed=seed, **params)
            est.fit(X_inner_train, y_inner_train)
            pred_scaled = est.predict(X_inner_valid)
            pred_m = target_scaler.inverse_transform(pred_scaled)
            true_m = y_raw[in_va]
            diff = pred_m - true_m
            err = np.mean(np.sqrt(np.sum(diff**2, axis=1)))
            fold_errors.append(err)

        mean_cv_err = float(np.mean(fold_errors))
        if mean_cv_err < best_score:
            best_score = mean_cv_err
            best_params = params

    return best_params


def run(
    models: list[str] | None = None,
    routings: list[str] | None = None,
    variance_threshold: float | None = 0.0,
    tune: bool = True,
    upstream_classifier: str = "random_forest",
    upstream_features: str = "variance_filtered",
    upstream_predictions: str = "classifiers_grouped_cv.csv",
    output_suffix: str = "",
    selected_hyperparameters: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if models is None:
        models = ["knn", "random_forest", "xgboost"]
    if routings is None:
        routings = ["global", "per_building", "per_building_floor"]

    train_df, _, _ = load_data()
    folds_path = ROOT / "data/splits/folds.json"
    folds = json.loads(folds_path.read_text())["folds"]

    # Load upstream classifier predictions for end-to-end routing
    preds_file = ROOT / "results/preds" / upstream_predictions
    if not preds_file.exists():
        raise FileNotFoundError(
            f"Classifier predictions not found at {preds_file}. "
            "Run scripts/train_classifiers.py first."
        )
    classifier_preds = pd.read_csv(preds_file)
    classifier_preds = classifier_preds[
        (classifier_preds.model == upstream_classifier)
        & (classifier_preds.feature_set == upstream_features)
    ].set_index(["fold", "row_index"])
    frozen_parameters = None
    if selected_hyperparameters:
        parameter_path = ROOT / "results/tables" / selected_hyperparameters
        frozen_parameters = json.loads(parameter_path.read_text())

    metrics_records: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    scaler_metadata: list[dict] = []
    selected_hyperparameters: list[dict] = []

    # Candidate grids for inner GroupKFold tuning
    param_grids = {
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["distance", "uniform"],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_features": ["sqrt", 0.2],
        },
        "xgboost": {
            "n_estimators": [100, 150],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        },
    }

    t_start = time.time()
    for model_name in models:
        for fold in folds:
            fold_id = fold["fold"]
            tr_idx = np.asarray(fold["train_idx"])
            va_idx = np.asarray(fold["valid_idx"])

            fit_df = train_df.iloc[tr_idx]
            valid_df = train_df.iloc[va_idx]

            # Fit feature builder strictly on training fold
            builder = FoldFeatureBuilder(
                representation="linear_positive",
                variance_threshold=variance_threshold,
            )
            X_train = builder.fit_transform(fit_df)
            X_valid = builder.transform(valid_df)

            # Fit coordinate target scaler strictly on training fold
            target_scaler = CoordinateTargetScaler()
            y_train_scaled = target_scaler.fit_transform(
                fit_df[["LONGITUDE", "LATITUDE"]].to_numpy()
            )
            y_valid_true = valid_df[["LONGITUDE", "LATITUDE"]].to_numpy()

            scaler_metadata.append(
                {
                    "fold": fold_id,
                    "mean_longitude": target_scaler.mean_[0],
                    "mean_latitude": target_scaler.mean_[1],
                    "scale_longitude": target_scaler.scale_[0],
                    "scale_latitude": target_scaler.scale_[1],
                }
            )

            # Inner GroupKFold hyperparameter tuning
            best_params: dict = {}
            if tune and model_name in param_grids:
                print(
                    f"[{model_name}] Tuning on fold {fold_id} (inner GroupKFold by USERID)...",
                    flush=True,
                )
                best_params = tune_regressor_inner_grouped(
                    model_name=model_name,
                    fit_df=fit_df,
                    user_ids=fit_df["USERID"].to_numpy(),
                    param_grid=param_grids[model_name],
                    variance_threshold=variance_threshold,
                    n_splits=3,
                )
                print(
                    f"[{model_name}] Fold {fold_id} selected params: {best_params}",
                    flush=True,
                )
            elif frozen_parameters is not None:
                match = next(
                    (
                        item
                        for item in frozen_parameters
                        if item.get("model") == model_name
                        and int(item.get("fold", -1)) == int(fold_id)
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(
                        f"No frozen parameters for {model_name}, fold {fold_id}"
                    )
                best_params = {
                    k: v for k, v in match.items() if k not in {"model", "fold"}
                }

            selected_hyperparameters.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    **best_params,
                }
            )

            # Retrieve upstream classifier predictions for this fold
            val_preds_fold = classifier_preds.loc[fold_id].loc[va_idx]
            b_pred = val_preds_fold["BUILDINGID_PRED"].to_numpy()
            f_pred = val_preds_fold["FLOOR_PRED"].to_numpy()

            b_train = fit_df["BUILDINGID"].to_numpy()
            f_train = fit_df["FLOOR"].to_numpy()
            b_true = valid_df["BUILDINGID"].to_numpy()
            f_true = valid_df["FLOOR"].to_numpy()

            for routing in routings:
                base_est = make_base_regressor(model_name, **best_params)
                hierarchical_model = HierarchicalCoordinateRegressor(
                    base_estimator=base_est,
                    routing=routing,
                    min_samples=10,
                )
                hierarchical_model.fit(
                    X_train,
                    y_train_scaled,
                    buildings=b_train,
                    floors=f_train,
                )

                # Determine modes to evaluate
                if routing == "global":
                    eval_modes = [("global", None, None)]
                else:
                    eval_modes = [
                        ("oracle", b_true, f_true),
                        ("end_to_end", b_pred, f_pred),
                    ]

                for mode_name, r_b, r_f in eval_modes:
                    pred_scaled = hierarchical_model.predict(
                        X_valid,
                        route_buildings=r_b,
                        route_floors=r_f,
                    )
                    pred_m = target_scaler.inverse_transform(pred_scaled)
                    diff = pred_m - y_valid_true
                    errors_m = np.sqrt(np.sum(diff**2, axis=1))
                    pos_metrics = compute_position_metrics(y_valid_true, pred_m)

                    metrics_records.append(
                        {
                            "model": model_name,
                            "routing": routing,
                            "routing_mode": mode_name,
                            "fold": fold_id,
                            "n_train_samples": len(tr_idx),
                            "n_valid_samples": len(va_idx),
                            **pos_metrics,
                        }
                    )

                    # Store row-level predictions
                    pred_df = pd.DataFrame(
                        {
                            "model": model_name,
                            "routing": routing,
                            "routing_mode": mode_name,
                            "fold": fold_id,
                            "row_index": va_idx,
                            "USERID": valid_df["USERID"].to_numpy(),
                            "BUILDINGID_TRUE": b_true,
                            "FLOOR_TRUE": f_true,
                            "BUILDINGID_ROUTE": r_b if r_b is not None else -1,
                            "FLOOR_ROUTE": r_f if r_f is not None else -1,
                            "LONGITUDE_TRUE": y_valid_true[:, 0],
                            "LATITUDE_TRUE": y_valid_true[:, 1],
                            "LONGITUDE_PRED": pred_m[:, 0],
                            "LATITUDE_PRED": pred_m[:, 1],
                            "ERROR_METRES": errors_m,
                        }
                    )
                    all_predictions.append(pred_df)

            print(
                f"Finished {model_name} / fold {fold_id} in {time.time()-t_start:.1f}s",
                flush=True,
            )

    # Save results
    tables_dir = ROOT / "results/tables"
    preds_dir = ROOT / "results/preds"
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_records)
    suffix = f"_{output_suffix}" if output_suffix else ""
    metrics_df.to_csv(
        tables_dir / f"regressor_grouped_cv_folds{suffix}.csv", index=False
    )

    summary_df = metrics_df.groupby(
        ["model", "routing", "routing_mode"], as_index=False
    ).agg(
        mean_error_mean=("mean_error", "mean"),
        mean_error_std=("mean_error", "std"),
        median_error_mean=("median_error", "mean"),
        median_error_std=("median_error", "std"),
        p75_error_mean=("p75_error", "mean"),
        p75_error_std=("p75_error", "std"),
        p95_error_mean=("p95_error", "mean"),
        p95_error_std=("p95_error", "std"),
        mae_longitude_mean=("mae_longitude", "mean"),
        mae_longitude_std=("mae_longitude", "std"),
        mae_latitude_mean=("mae_latitude", "mean"),
        mae_latitude_std=("mae_latitude", "std"),
    )
    summary_df.to_csv(
        tables_dir / f"regressor_grouped_cv_summary{suffix}.csv", index=False
    )

    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    all_preds_df.to_csv(preds_dir / f"regressors{suffix}_grouped_cv.csv", index=False)

    # Save metadata
    (tables_dir / f"regressor_target_scalers{suffix}.json").write_text(
        json.dumps(scaler_metadata, indent=2)
    )
    (tables_dir / f"regressor_selected_hyperparameters{suffix}.json").write_text(
        json.dumps(selected_hyperparameters, indent=2)
    )

    print("\n=== Phase 1 Regressors Grouped-CV Summary (in metres) ===")
    print(
        summary_df[
            [
                "model",
                "routing",
                "routing_mode",
                "mean_error_mean",
                "mean_error_std",
                "median_error_mean",
                "p75_error_mean",
                "p95_error_mean",
            ]
        ].to_string(index=False)
    )

    return metrics_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["knn", "random_forest", "xgboost"]
    )
    parser.add_argument(
        "--routings",
        nargs="+",
        default=["global", "per_building", "per_building_floor"],
    )
    parser.add_argument(
        "--no-tune",
        dest="tune",
        action="store_false",
        help="Skip inner hyperparameter search",
    )
    parser.add_argument("--upstream-predictions", default="classifiers_grouped_cv.csv")
    parser.add_argument("--upstream-classifier", default="random_forest")
    parser.add_argument("--upstream-features", default="variance_filtered")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--selected-hyperparameters")
    args = parser.parse_args()

    run(
        models=args.models,
        routings=args.routings,
        tune=args.tune,
        upstream_predictions=args.upstream_predictions,
        upstream_classifier=args.upstream_classifier,
        upstream_features=args.upstream_features,
        output_suffix=args.output_suffix,
        selected_hyperparameters=args.selected_hyperparameters,
    )
