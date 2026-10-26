"""Phase 1 building and floor classifiers."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def make_classifier(name: str, seed: int = 42, **overrides):
    """Create the configured classifier with deterministic, CPU-friendly settings."""
    if name == "random_forest":
        params = {
            "n_estimators": 300,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": seed,
        }
        params.update(overrides)
        return RandomForestClassifier(**params)
    if name == "xgboost":
        params = {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": 4,
            "random_state": seed,
        }
        params.update(overrides)
        return XGBClassifier(**params)
    raise ValueError(f"Unknown classifier {name!r}; use random_forest or xgboost")
