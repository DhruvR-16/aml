# Phase 1 classification: building and floor

## Random Forest

A Random Forest trains decision trees on bootstrap samples and considers a random subset of features at each split. Averaging many trees reduces the variance of an individual tree and can capture nonlinear interactions between access-point readings without requiring a linear decision boundary. Its feature importance here is mean decrease in impurity; correlated WAP features can divide importance across one another, so it is a diagnostic rather than a causal explanation.

## XGBoost

XGBoost builds trees sequentially, with each new tree improving the loss left by the preceding ensemble. Gradient-based optimization, shrinkage, row/column subsampling, and regularization help control overfitting while modeling nonlinear feature interactions. The classifier uses multiclass log loss for building and floor. Its gain importance reports split improvement and is not directly comparable in scale to Random Forest impurity importance.

## Validation design

All reported classification results use the saved five-fold `GroupKFold` partition by `USERID`. The feature builder, observed-RSSI scalers, and optional zero-variance selector are fit independently on each training fold. The official `validationData.csv` remains reserved for final evaluation. Overall floor accuracy is supplemented with floor accuracy among scans whose building was predicted correctly, to separate floor errors from upstream building errors.
