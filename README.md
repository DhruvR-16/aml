# Indoor Wi-Fi Localization (Project 9)

Phase 1 foundation for building, floor, and coordinate prediction from the UCI UJIIndoorLoc Wi-Fi fingerprints. Raw data is downloaded locally and is excluded from Git.

## Phase 1 work order and owners

1. **Member 1 — Data and infrastructure lead:** repository setup, UCI download, schema validation, training duplicate and constant-WAP cleanup, suspicious near-zero RSSI audit, USERID GroupKFold splits, official test-set designation, and the EDA notebook/data-quality report.
2. **Member 2 — Features and representation lead:** shared `+100` to `-110 dBm` convention, three RSSI representations, engineered/missingness features, fold-only scaling/PCA/variance filtering, and grouped-CV kNN comparison.
3. **Member 3 — Classification lead** consumes the cleaned splits/features for building and floor models; **Member 4 — Regression and validation lead** consumes them for coordinate models. Member 2's feature importance work is integrated with Member 3's tree models.

The guide uses Member 1–5 role labels. Use the agreed team roster to confirm each person’s member-number mapping before presenting or submitting.

## Setup and data

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_splits.py
python scripts/export_quality_report.py
```

The loader removes duplicate training rows only, detects constant WAP columns from training only and drops those columns from both splits, and flags observed readings from -5 to 0 dBm with their USERIDs. It does not modify or deduplicate the official `validationData.csv` held-out test set. Splits are five deterministic `GroupKFold` folds grouped by `USERID`; indices refer to the cleaned training frame.

## Phase 1 artifacts

- `notebooks/01_eda.ipynb`: data quality, class balance, detected AP counts, observed RSSI, USERID/PHONEID counts, coordinate maps, and train-vs-test shift.
- `notebooks/02_features.ipynb`: feature conventions, representations, engineered features, scaling/PCA and grouped kNN comparison.
- `src/data/load.py`: common loaders and cleaning.
- `src/features/transform.py`: fold-safe feature building.
- `data/splits/folds.json`: reproducible fold indices and quality metadata; raw data itself is ignored.
- `results/tables/`: data-quality, grouped kNN representation, PCA variance, and variance-filter outputs produced by the report scripts.
- `docs/theory_phase1.md`, `docs/data_quality.md`: methodology notes and data-quality summary.

Run `python scripts/run_knn_baselines.py` for the five-fold representation comparison and `python scripts/report_feature_reduction.py` for PCA/variance-filter reports. Both operate on USERID-held-out training folds; neither uses the official test set. The baseline metrics are in `results/tables/knn_representation_summary.csv`, with per-fold metrics alongside it. `results/tables/pca_explained_variance.csv` reports fold-0 cumulative explained variance, and `variance_threshold_counts.csv` reports zero-variance selection. Member 2's fold-fitted selector is used in Member 3's RF/XGBoost comparisons; see the Member 3 results section below.

## Source

UJIIndoorLoc, UCI Machine Learning Repository, DOI [10.24432/C5MS59](https://doi.org/10.24432/C5MS59), CC BY 4.0. Follow the dataset license and cite UCI in the report.

## Member 3 — Phase 1 classifier work

Random Forest and XGBoost building/floor classifiers use the shared feature builder and USERID-grouped folds. `src/models/ml/classifiers.py` contains the estimator definitions. The fixed-setting comparison is reproduced with `python scripts/train_classifiers.py`; corrected nested USERID tuning is reproduced with `python scripts/train_tuned_classifiers.py`. Both fit features within the relevant training folds and write fold summaries and OOF predictions. The checked-in classifier summary, class-support, confusion, and selected-parameter tables are under `results/tables/`. To regenerate tuned confusion figures after running the tuner, use `python scripts/plot_classifier_confusion_matrices.py --predictions results/preds/classifiers_tuned_grouped_cv.csv --output-prefix results/figures/classifier_tuned`. Interpret XGBoost gain and Random Forest impurity importance within their own model families, not as directly comparable magnitudes.

## Member 4 — Phase 1 coordinate regression and validation

`src/models/ml/regressors.py` defines coordinate scalers, regressors, routing, and position metrics. `scripts/train_regressors.py` compares kNN, Random Forest, and XGBoost with global, per-building, and per-building-floor routing. For the tuned primary route, first generate the M3 nested OOF predictions, then run:

```bash
python scripts/train_tuned_classifiers.py
python scripts/train_regressors.py --upstream-predictions classifiers_tuned_grouped_cv.csv --upstream-features tuned_variance_filtered --output-suffix tuned
```

The second command fits its feature transforms and coordinate-target scalers inside the relevant training splits and evaluates end-to-end routes using M3 OOF predictions. Oracle routing with true labels is diagnostic only. Grouped-CV summaries, fold metrics, and selected settings are in `results/tables/`; row-level predictions are generated locally under `results/preds/`. Existing official-test outputs are from the earlier fixed-setting run; do not rerun the official test or use those saved results to select a tuned model. `scripts/plot_regression_results.py` regenerates regression figures from the corresponding saved prediction CSV.

## Member 5 — Phase 1 evaluation and report integration

Run the shared evaluator and comparison-table generator after model scripts have produced local prediction CSVs:

```bash
python scripts/build_results_table.py
```

It reads prediction CSVs in `results/preds/`, writes pooled grouped-CV and official-test metrics to `results/tables/phase1_comparison.csv`, and preserves fold-level rows in `results/tables/phase1_results_by_fold.csv`. Row-level predictions are not required to review the committed summary tables and are generated locally when models are rerun. Metrics include building/floor accuracy, conditional floor accuracy, coordinate MAE and error quantiles, hierarchical breakdown, and the guide's EvAAL-style score. Unit tests are in `tests/test_evaluation_metrics.py`.

The evaluator preserves the saved official-test outputs and does not rerun them after tuning. The saved test results therefore represent the earlier fixed-setting baseline; use grouped-CV results to compare tuned models.

