"""Aggregate every saved prediction CSV into a comparable Phase 1 table."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_predictions  # noqa: E402


def _attach_classifier_routes(
    frame: pd.DataFrame, classifier_preds: pd.DataFrame
) -> pd.DataFrame:
    """Attach M3 OOF predictions to global M4 OOF coordinates by stable row keys."""
    out = frame.copy()
    has_predictions = "BUILDINGID_PRED" in out.columns and "FLOOR_PRED" in out.columns
    if has_predictions and out[["BUILDINGID_PRED", "FLOOR_PRED"]].notna().all(
        axis=None
    ):
        return out
    if "fold" not in out or "row_index" not in out:
        return out
    if classifier_preds.empty:
        return out
    # Match the exact classifier used to create M4's end-to-end route inputs.
    model_name = (
        "task_best"
        if "task_best" in classifier_preds.model.unique()
        else "random_forest"
    )
    feature_values = set(classifier_preds.feature_set.dropna().unique())
    feature_name = next(
        (
            candidate
            for candidate in (
                "tuned_task_best",
                "tuned_variance_filtered",
                "variance_filtered",
            )
            if candidate in feature_values
        ),
        "variance_filtered",
    )
    selected = classifier_preds[
        (classifier_preds.model == model_name)
        & (classifier_preds.feature_set == feature_name)
    ][["fold", "row_index", "BUILDINGID_PRED", "FLOOR_PRED"]].drop_duplicates(
        ["fold", "row_index"]
    )
    out = out.merge(
        selected,
        on=["fold", "row_index"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_m3"),
    )
    for col in ("BUILDINGID_PRED", "FLOOR_PRED"):
        m3_col = f"{col}_m3"
        if m3_col in out:
            if col in out:
                out[col] = out[col].where(out[col].notna(), out[m3_col])
                out.drop(columns=m3_col, inplace=True)
            else:
                out.rename(columns={m3_col: col}, inplace=True)
    return out


def _normalize_m4(frame: pd.DataFrame, classifier_preds: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    has_direct_classifier_predictions = (
        "BUILDINGID_PRED" in out.columns and "FLOOR_PRED" in out.columns
    )
    oracle_mask = (
        out.routing_mode.eq("oracle")
        if "routing_mode" in out
        else pd.Series(False, index=out.index)
    )
    # Routed end-to-end rows use predicted building/floor as their model inputs.
    # Oracle rows are intentionally excluded from categorical and penalized metrics.
    if {"BUILDINGID_ROUTE", "FLOOR_ROUTE", "routing_mode"}.issubset(out.columns):
        e2e = out.routing_mode.eq("end_to_end")
        routed = out.routing.ne("global")
        out.loc[e2e & routed, "BUILDINGID_PRED"] = out.loc[
            e2e & routed, "BUILDINGID_ROUTE"
        ]
        out.loc[e2e & routed, "FLOOR_PRED"] = out.loc[e2e & routed, "FLOOR_ROUTE"]
    out = _attach_classifier_routes(out, classifier_preds)
    if not has_direct_classifier_predictions and oracle_mask.any():
        out.loc[oracle_mask, ["BUILDINGID_PRED", "FLOOR_PRED"]] = np.nan
    return out


def build_results_table(
    preds_dir: Path | str | None = None,
    output: Path | str | None = None,
    by_fold_output: Path | str | None = None,
) -> pd.DataFrame:
    preds_dir = Path(preds_dir or ROOT / "results/preds")
    output = Path(output or ROOT / "results/tables/phase1_comparison.csv")
    by_fold_output = Path(
        by_fold_output or ROOT / "results/tables/phase1_results_by_fold.csv"
    )
    classifier_path = preds_dir / "classifiers_grouped_cv.csv"
    classifier_preds = (
        pd.read_csv(classifier_path) if classifier_path.exists() else pd.DataFrame()
    )
    records: list[dict] = []
    fold_records: list[dict] = []

    for path in sorted(preds_dir.glob("*.csv")):
        if path.name == "phase1_results.csv":
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        if path.name.startswith("regressors_"):
            # Pair tuned M4 runs with tuned M3 OOF routes; baseline runs continue
            # to consume the original fixed-setting classifier predictions.
            if "task_best" in path.stem:
                task_best_path = preds_dir / "classifiers_task_best_grouped_cv.csv"
                route_preds = (
                    pd.read_csv(task_best_path)
                    if task_best_path.exists()
                    else pd.DataFrame()
                )
            elif "tuned" in path.stem:
                tuned_path = preds_dir / "classifiers_tuned_grouped_cv.csv"
                route_preds = (
                    pd.read_csv(tuned_path) if tuned_path.exists() else pd.DataFrame()
                )
            else:
                route_preds = classifier_preds
            frame = _normalize_m4(frame, route_preds)
        # Keep comparable partitions separate: model, feature set, routing, and
        # split. Pooled OOF metrics are computed from all out-of-fold rows, while
        # a companion file preserves each fold's metrics.
        base_keys = [
            key
            for key in ("model", "feature_set", "routing", "routing_mode")
            if key in frame
        ]
        if not base_keys:
            base_keys = ["model"] if "model" in frame else []
        for split, split_rows in (
            ("official_test" if "test" in path.name else "grouped_cv", frame),
        ):
            groups = (
                split_rows.groupby(base_keys, dropna=False, sort=True)
                if base_keys
                else [((), split_rows)]
            )
            for key, rows in groups:
                values = key if isinstance(key, tuple) else (key,)
                record = {"source_file": path.name, "split": split}
                record.update(dict(zip(base_keys, values)))
                record.update(evaluate_predictions(rows))
                if "fold" in rows:
                    record["n_folds"] = int(rows["fold"].nunique())
                records.append(record)

                if "fold" not in rows:
                    fold_groups = [(None, rows)]
                else:
                    fold_groups = list(rows.groupby("fold", dropna=False, sort=True))
                for fold, fold_rows in fold_groups:
                    fold_record = {
                        "source_file": path.name,
                        "split": split,
                        "fold": fold,
                    }
                    fold_record.update(dict(zip(base_keys, values)))
                    fold_record.update(evaluate_predictions(fold_rows))
                    fold_records.append(fold_record)

    result = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    pd.DataFrame(fold_records).to_csv(by_fold_output, index=False)
    return result


if __name__ == "__main__":
    table = build_results_table()
    print(table.to_string(index=False))
