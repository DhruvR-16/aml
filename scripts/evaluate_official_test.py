"""Official test-set evaluation for Phase 1 frozen models.

Freezes optimal hyperparameter configurations and evaluates:
  1. Member 3 building & floor classifiers on official test set.
  2. Member 4 coordinate regressors across routing strategies (global, per_building, per_building_floor).
  3. Oracle routing vs End-to-End routing.
  4. Exports test predictions, metrics tables, and official test figures (CDF, Error Maps).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_data  # noqa: E402
from src.features.transform import FoldFeatureBuilder  # noqa: E402
from src.models.ml.classifiers import make_classifier  # noqa: E402
from src.models.ml.regressors import (  # noqa: E402
    CoordinateTargetScaler,
    HierarchicalCoordinateRegressor,
    compute_position_metrics,
    make_base_regressor,
)


def run_official_test_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds_file = ROOT / "results/preds/regressors_test_preds.csv"
    summary_file = ROOT / "results/tables/regressor_test_summary.csv"
    if preds_file.exists() or summary_file.exists():
        raise FileExistsError(
            "Official-test predictions already exist. This dataset has already been evaluated; "
            "do not rerun it or use the scores for model selection."
        )

    print("=== Phase 1 Official Held-Out Test Evaluation ===")
    train_df, test_df, _ = load_data()
    print(f"Training samples: {len(train_df)}, Official test samples: {len(test_df)}")

    # 1. Feature pipeline fit on full training set only
    builder = FoldFeatureBuilder(
        representation="linear_positive", variance_threshold=0.0
    )
    X_train = builder.fit_transform(train_df)
    X_test = builder.transform(test_df)
    print(f"Features transformed: {X_train.shape[1]} features (variance-filtered)")

    # 2. Member 3 Upstream Classifiers fit on full training set
    print("\nFitting upstream building & floor classifiers...")
    clf_building = make_classifier("random_forest")
    clf_building.fit(X_train, train_df["BUILDINGID"].to_numpy())
    building_pred = clf_building.predict(X_test)

    clf_floor = make_classifier("random_forest")
    clf_floor.fit(X_train, train_df["FLOOR"].to_numpy())
    floor_pred = clf_floor.predict(X_test)

    b_true = test_df["BUILDINGID"].to_numpy()
    f_true = test_df["FLOOR"].to_numpy()

    b_acc = accuracy_score(b_true, building_pred)
    f_acc = accuracy_score(f_true, floor_pred)
    b_correct = b_true == building_pred
    f_cond_acc = accuracy_score(f_true[b_correct], floor_pred[b_correct])

    print(f"Test Building Accuracy: {b_acc:.4f}")
    print(
        f"Test Floor Accuracy:    {f_acc:.4f} (Given correct building: {f_cond_acc:.4f})"
    )

    # 3. Target Coordinate Standardization fit on full training set
    target_scaler = CoordinateTargetScaler()
    y_train_scaled = target_scaler.fit_transform(
        train_df[["LONGITUDE", "LATITUDE"]].to_numpy()
    )
    y_test_true = test_df[["LONGITUDE", "LATITUDE"]].to_numpy()

    # Legacy fixed settings for the earlier one-time official-test baseline.
    # The corrected tuned grouped-CV outputs are stored separately and are not
    # evaluated again on this already-used official test set.
    frozen_configs = {
        "knn": {"n_neighbors": 11, "weights": "distance"},
        "random_forest": {"n_estimators": 200, "max_features": "sqrt"},
        "xgboost": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.1},
    }

    test_metrics: list[dict] = []
    test_predictions: list[pd.DataFrame] = []

    b_train = train_df["BUILDINGID"].to_numpy()
    f_train = train_df["FLOOR"].to_numpy()

    for model_name, params in frozen_configs.items():
        t0 = time.time()
        for routing in ["global", "per_building", "per_building_floor"]:
            base_est = make_base_regressor(model_name, **params)
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

            if routing == "global":
                eval_modes = [("global", None, None)]
            else:
                eval_modes = [
                    ("oracle", b_true, f_true),
                    ("end_to_end", building_pred, floor_pred),
                ]

            for mode_name, r_b, r_f in eval_modes:
                pred_scaled = hierarchical_model.predict(
                    X_test,
                    route_buildings=r_b,
                    route_floors=r_f,
                )
                pred_m = target_scaler.inverse_transform(pred_scaled)
                diff = pred_m - y_test_true
                errors_m = np.sqrt(np.sum(diff**2, axis=1))
                metrics = compute_position_metrics(y_test_true, pred_m)

                test_metrics.append(
                    {
                        "model": model_name,
                        "routing": routing,
                        "routing_mode": mode_name,
                        "n_test_samples": len(test_df),
                        **metrics,
                    }
                )

                pred_df = pd.DataFrame(
                    {
                        "model": model_name,
                        "routing": routing,
                        "routing_mode": mode_name,
                        "row_index": np.arange(len(test_df)),
                        "BUILDINGID_TRUE": b_true,
                        "FLOOR_TRUE": f_true,
                        "BUILDINGID_PRED": building_pred,
                        "FLOOR_PRED": floor_pred,
                        "BUILDINGID_ROUTE": r_b if r_b is not None else -1,
                        "FLOOR_ROUTE": r_f if r_f is not None else -1,
                        "LONGITUDE_TRUE": y_test_true[:, 0],
                        "LATITUDE_TRUE": y_test_true[:, 1],
                        "LONGITUDE_PRED": pred_m[:, 0],
                        "LATITUDE_PRED": pred_m[:, 1],
                        "ERROR_METRES": errors_m,
                    }
                )
                test_predictions.append(pred_df)

        print(f"Finished evaluating {model_name} on test set in {time.time()-t0:.2f}s")

    # Save output tables
    tables_dir = ROOT / "results/tables"
    preds_dir = ROOT / "results/preds"
    figs_dir = ROOT / "results/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(test_metrics)
    metrics_df.to_csv(tables_dir / "regressor_test_summary.csv", index=False)

    all_preds_df = pd.concat(test_predictions, ignore_index=True)
    all_preds_df.to_csv(preds_dir / "regressors_test_preds.csv", index=False)

    print("\n=== Official Test Set Coordinate Regression Summary (in metres) ===")
    summary_cols = [
        "model",
        "routing",
        "routing_mode",
        "mean_error",
        "median_error",
        "p75_error",
        "p95_error",
        "mae_longitude",
        "mae_latitude",
    ]
    print(metrics_df[summary_cols].to_string(index=False))

    # 4. Generate Official Test CDF and Maps
    plot_test_cdf(all_preds_df, figs_dir / "coordinate_error_cdf.png")
    plot_test_error_maps(all_preds_df, figs_dir / "coordinate_error_maps.png")

    return metrics_df, all_preds_df


def plot_test_cdf(preds_df: pd.DataFrame, out_path: Path) -> None:
    """Plot empirical CDF of test set error for the frozen models."""
    fig, ax = plt.subplots(figsize=(9, 6))

    subset = preds_df[
        (preds_df.routing == "per_building") & (preds_df.routing_mode == "end_to_end")
    ]
    palette = {"knn": "#1f77b4", "random_forest": "#2ca02c", "xgboost": "#ff7f0e"}

    for model_name in ["knn", "random_forest", "xgboost"]:
        model_data = subset[subset.model == model_name]
        if model_data.empty:
            continue
        errors = np.sort(model_data["ERROR_METRES"].to_numpy())
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(
            errors,
            cdf,
            label=f"{model_name.replace('_', ' ').title()} (End-to-End, Per-Building)",
            linewidth=2.5,
            color=palette.get(model_name),
        )

    # Reference lines
    ax.axvline(x=10, color="gray", linestyle="--", alpha=0.6, label="10m threshold")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.6, label="Median (50%)")
    ax.axhline(y=0.75, color="gray", linestyle="-.", alpha=0.6, label="75th %ile")

    ax.set_xlim(0, 45)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Euclidean Position Error (metres)", fontsize=12)
    ax.set_ylabel("Cumulative Fraction P(Error ≤ x)", fontsize=12)
    ax.set_title(
        "Official Held-Out Test Error CDF (Frozen Models)", fontsize=14, pad=12
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Official Test CDF to {out_path}")


def plot_test_error_maps(
    preds_df: pd.DataFrame, out_path: Path, best_model: str = "knn"
) -> None:
    """Plot official test set coordinates colored by localization error, paneled by building & floor."""
    subset = preds_df[
        (preds_df.model == best_model)
        & (preds_df.routing == "per_building")
        & (preds_df.routing_mode == "end_to_end")
    ]

    buildings = sorted(subset["BUILDINGID_TRUE"].unique())
    floors = sorted(subset["FLOOR_TRUE"].unique())
    if not buildings or not floors:
        raise ValueError("No test predictions available for the selected model/routing")

    fig, axes = plt.subplots(
        len(floors), len(buildings), figsize=(15, 11), sharex="col", sharey="row"
    )
    axes = np.asarray(axes, dtype=object).reshape(len(floors), len(buildings))

    for col_idx, b in enumerate(buildings):
        for row_idx, f in enumerate(floors):
            ax = axes[row_idx, col_idx]
            cell = subset[(subset.BUILDINGID_TRUE == b) & (subset.FLOOR_TRUE == f)]

            if cell.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            scatter = ax.scatter(
                cell["LONGITUDE_TRUE"],
                cell["LATITUDE_TRUE"],
                c=cell["ERROR_METRES"],
                cmap="viridis",
                vmin=0,
                vmax=25,
                s=18,
                alpha=0.85,
            )

            if row_idx == 0:
                ax.set_title(f"Building {b}", fontsize=13, weight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"Floor {f}\nLatitude (m)", fontsize=11)
            if row_idx == len(floors) - 1:
                ax.set_xlabel("Longitude (m)", fontsize=11)

            ax.grid(True, linestyle=":", alpha=0.3)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Euclidean Position Error (metres)", fontsize=12)

    fig.suptitle(
        f"Official Test Error Map · {best_model.upper()} Per-Building Regressor (End-to-End)",
        fontsize=15,
        y=0.98,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Official Test Error Map to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_official_test_evaluation()
