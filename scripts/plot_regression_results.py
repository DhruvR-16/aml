"""Generate report figures for Phase 1 coordinate regression:
1. Coordinate error CDF (fraction of scans <= x metres).
2. Coordinate error spatial scatter maps paneled by building and floor.
3. Routing strategy comparison bar chart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _cv_selected_routes(summary_df: pd.DataFrame) -> dict[str, str]:
    candidates = summary_df[summary_df.routing_mode == "end_to_end"]
    if candidates.empty:
        return {}
    best_rows = candidates.loc[candidates.groupby("model").mean_error_mean.idxmin()]
    return dict(zip(best_rows.model, best_rows.routing))


def plot_error_cdf(
    preds_df: pd.DataFrame,
    out_path: Path,
    summary_df: pd.DataFrame,
    title: str = "Official Held-Out Test",
    selected_routes: dict[str, str] | None = None,
) -> None:
    """Plot empirical cumulative distribution function (CDF) of position error."""
    fig, ax = plt.subplots(figsize=(9, 6))

    selected_routes = selected_routes or _cv_selected_routes(summary_df)

    palette = {"knn": "#1f77b4", "random_forest": "#2ca02c", "xgboost": "#ff7f0e"}

    for model_name in ["knn", "random_forest", "xgboost"]:
        route = selected_routes.get(model_name, "global")
        route_mode = "end_to_end" if route != "global" else "global"
        model_data = preds_df[
            (preds_df.model == model_name)
            & (preds_df.routing == route)
            & (preds_df.routing_mode == route_mode)
        ]
        if model_data.empty:
            continue
        errors = np.sort(model_data["ERROR_METRES"].to_numpy())
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(
            errors,
            cdf,
            label=f"{model_name.replace('_', ' ').title()} ({route.replace('_', ' ')})",
            linewidth=2,
            color=palette.get(model_name, None),
        )

    # Reference lines
    ax.axvline(x=10, color="gray", linestyle="--", alpha=0.6, label="10m threshold")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.6, label="Median (50%)")
    ax.axhline(y=0.75, color="gray", linestyle="-.", alpha=0.6, label="75th percentile")

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Euclidean Position Error (metres)", fontsize=12)
    ax.set_ylabel("Cumulative Probability P(Error ≤ x)", fontsize=12)
    ax.set_title(f"Phase 1 Coordinate Error CDF ({title})", fontsize=14, pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=11)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CDF plot to {out_path}")


def plot_error_maps(
    preds_df: pd.DataFrame, out_path: Path, best_model: str = "knn"
) -> None:
    """Plot held-out coordinates coloured by error for all floors."""
    subset = preds_df[
        (preds_df.model == best_model)
        & (preds_df.routing == "per_building")
        & (preds_df.routing_mode == "end_to_end")
    ]
    if subset.empty:
        subset = preds_df[preds_df.model == best_model]

    buildings = sorted(subset["BUILDINGID_TRUE"].unique())
    floors = sorted(subset["FLOOR_TRUE"].unique())

    n_rows = len(floors)
    n_cols = len(buildings)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 12), sharex="col", sharey="row"
    )
    axes = np.asarray(axes, dtype=object).reshape(n_rows, n_cols)

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
                vmax=30,
                s=12,
                alpha=0.8,
            )

            if row_idx == 0:
                ax.set_title(f"Building {b}", fontsize=13, weight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"Floor {f}\nLatitude (m)", fontsize=11)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Longitude (m)", fontsize=11)

            ax.grid(True, linestyle=":", alpha=0.3)

    # Colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Euclidean Position Error (metres)", fontsize=12)

    fig.suptitle(
        f"Spatial Distribution of Localization Errors · {best_model.replace('_', ' ').title()}",
        fontsize=16,
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Error Map to {out_path}")


def plot_routing_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Compare global vs per-building vs per-building-floor (oracle & end-to-end)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Format labels
    plot_df = summary_df.copy()
    plot_df["Strategy"] = (
        (plot_df["routing"] + " (" + plot_df["routing_mode"] + ")")
        .str.replace("_", " ")
        .str.title()
    )
    plot_df["Model"] = plot_df["model"].str.replace("_", " ").str.title()

    order = [
        "Global (Global)",
        "Per Building (Oracle)",
        "Per Building (End To End)",
        "Per Building Floor (Oracle)",
        "Per Building Floor (End To End)",
    ]
    # Filter to existing
    order = [o for o in order if o in plot_df["Strategy"].values]

    sns.barplot(
        data=plot_df,
        x="Strategy",
        y="mean_error_mean",
        hue="Model",
        order=order,
        palette="tab10",
        ax=ax,
    )

    ax.set_ylabel("Mean Position Error (metres)", fontsize=12)
    ax.set_xlabel("Routing Strategy & Mode", fontsize=12)
    ax.set_title(
        "Hierarchical Routing Comparison Across Models (Grouped CV)",
        fontsize=14,
        pad=12,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Model", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Routing Comparison to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    preds_file = ROOT / "results/preds/regressors_test_preds.csv"
    cv_preds_file = ROOT / "results/preds/regressors_tuned_grouped_cv.csv"
    summary_file = ROOT / "results/tables/regressor_grouped_cv_summary_tuned.csv"
    if (
        not preds_file.exists()
        or not cv_preds_file.exists()
        or not summary_file.exists()
    ):
        print("Required grouped-CV and official-test predictions are missing.")
        return

    preds_df = pd.read_csv(preds_file)
    cv_preds_df = pd.read_csv(cv_preds_file)
    summary_df = pd.read_csv(summary_file)

    fig_dir = ROOT / "results/figures"
    plot_error_cdf(
        preds_df,
        fig_dir / "coordinate_error_cdf.png",
        summary_df,
        title="Earlier One-Time Official-Test Baseline (Per-Building)",
        selected_routes={
            "knn": "per_building",
            "random_forest": "per_building",
            "xgboost": "per_building",
        },
    )
    plot_error_cdf(
        cv_preds_df,
        fig_dir / "coordinate_error_cdf_grouped_cv.png",
        summary_df,
        title="Grouped CV OOF",
    )
    plot_error_maps(preds_df, fig_dir / "coordinate_error_maps.png")
    plot_routing_comparison(
        summary_df, fig_dir / "regressor_routing_comparison_tuned.png"
    )


if __name__ == "__main__":
    main()
