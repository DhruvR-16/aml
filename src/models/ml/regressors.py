"""Phase 1 coordinate regressors, target scaling, and hierarchical routing."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


class CoordinateTargetScaler:
    """Standardize 2D coordinates (LONGITUDE, LATITUDE) on training folds only.

    All error metrics in localization must be computed on unstandardized coordinates
    (in metres), so inverse_transform is applied before evaluation.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, y: np.ndarray | pd.DataFrame) -> CoordinateTargetScaler:
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected 2D target matrix of shape (N, 2), got {arr.shape}"
            )
        self.mean_ = np.mean(arr, axis=0)
        scale = np.std(arr, axis=0)
        # Avoid division by zero if coordinates are constant
        self.scale_ = np.where(scale > 0, scale, 1.0)
        return self

    def transform(self, y: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CoordinateTargetScaler must be fit before transform")
        arr = np.asarray(y, dtype=float)
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, y: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y_scaled: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "CoordinateTargetScaler must be fit before inverse_transform"
            )
        arr = np.asarray(y_scaled, dtype=float)
        return (arr * self.scale_) + self.mean_

    def to_dict(self) -> dict[str, list[float]]:
        if self.mean_ is None or self.scale_ is None:
            return {}
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }


def make_base_regressor(name: str, seed: int = 42, **kwargs: Any) -> BaseEstimator:
    """Create a coordinate regressor instance with reproducible defaults or custom kwargs."""
    if name == "knn":
        params = {
            "n_neighbors": kwargs.get("n_neighbors", 5),
            "weights": kwargs.get("weights", "distance"),
            "metric": kwargs.get("metric", "minkowski"),
            "p": kwargs.get("p", 2),
            "n_jobs": kwargs.get("n_jobs", -1),
        }
        return KNeighborsRegressor(**params)

    if name == "random_forest":
        params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", None),
            "max_features": kwargs.get("max_features", "sqrt"),
            "min_samples_leaf": kwargs.get("min_samples_leaf", 1),
            "random_state": seed,
            "n_jobs": kwargs.get("n_jobs", -1),
        }
        return RandomForestRegressor(**params)

    if name == "xgboost":
        if XGBRegressor is None:
            raise ImportError(
                "xgboost is not installed. Please install xgboost in your environment."
            )
        params = {
            "n_estimators": kwargs.get("n_estimators", 200),
            "max_depth": kwargs.get("max_depth", 6),
            "learning_rate": kwargs.get("learning_rate", 0.05),
            "subsample": kwargs.get("subsample", 0.8),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
            "reg_lambda": kwargs.get("reg_lambda", 1.0),
            "tree_method": kwargs.get("tree_method", "hist"),
            "random_state": seed,
            "n_jobs": kwargs.get("n_jobs", 4),
        }
        base_xgb = XGBRegressor(**params)
        return MultiOutputRegressor(base_xgb)

    raise ValueError(f"Unknown regressor {name!r}; use knn, random_forest, or xgboost")


def compute_position_metrics(
    y_true_m: np.ndarray,
    y_pred_m: np.ndarray,
) -> dict[str, float]:
    """Compute coordinate MAE and Euclidean position error distribution in metres."""
    y_true_m = np.asarray(y_true_m, dtype=float)
    y_pred_m = np.asarray(y_pred_m, dtype=float)
    diff = y_pred_m - y_true_m
    mae_lon = float(np.mean(np.abs(diff[:, 0])))
    mae_lat = float(np.mean(np.abs(diff[:, 1])))
    euclidean = np.sqrt(np.sum(diff**2, axis=1))

    return {
        "mae_longitude": mae_lon,
        "mae_latitude": mae_lat,
        "mean_error": float(np.mean(euclidean)),
        "median_error": float(np.median(euclidean)),
        "p75_error": float(np.percentile(euclidean, 75)),
        "p95_error": float(np.percentile(euclidean, 95)),
    }


class HierarchicalCoordinateRegressor(BaseEstimator, RegressorMixin):
    """Hierarchical coordinate regressor supporting global, per-building, and per-building-floor routing.

    Routing strategies:
      - 'global': Single regressor trained on all training samples.
      - 'per_building': Regressors fit independently per building ID.
      - 'per_building_floor': Regressors fit independently per (building ID, floor ID).
        Falls back to the parent building model if a floor has fewer than min_samples.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        routing: Literal["global", "per_building", "per_building_floor"] = "global",
        min_samples: int = 10,
    ) -> None:
        self.base_estimator = base_estimator
        self.routing = routing
        self.min_samples = min_samples

        self.global_model_: BaseEstimator | None = None
        self.building_models_: dict[int, BaseEstimator] = {}
        self.floor_models_: dict[tuple[int, int], BaseEstimator] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        buildings: np.ndarray | pd.Series,
        floors: np.ndarray | pd.Series | None = None,
    ) -> HierarchicalCoordinateRegressor:
        X = np.asarray(X)
        y = np.asarray(y)
        b_arr = np.asarray(buildings)
        f_arr = np.asarray(floors) if floors is not None else None

        # Always fit a global model as top-level fallback
        self.global_model_ = clone(self.base_estimator)
        self.global_model_.fit(X, y)

        if self.routing == "global":
            return self

        # Fit per-building models
        unique_buildings = np.unique(b_arr)
        for b in unique_buildings:
            idx = np.where(b_arr == b)[0]
            if len(idx) >= self.min_samples:
                b_model = clone(self.base_estimator)
                b_model.fit(X[idx], y[idx])
                self.building_models_[int(b)] = b_model

        if self.routing == "per_building":
            return self

        # Fit per-building-floor models
        if f_arr is None:
            raise ValueError(
                "floors array must be provided for per_building_floor routing"
            )

        groups = pd.DataFrame({"b": b_arr, "f": f_arr}).groupby(["b", "f"]).indices
        for (b, f), idx in groups.items():
            if len(idx) >= self.min_samples:
                bf_model = clone(self.base_estimator)
                bf_model.fit(X[idx], y[idx])
                self.floor_models_[(int(b), int(f))] = bf_model

        return self

    def predict(
        self,
        X: np.ndarray,
        route_buildings: np.ndarray | pd.Series | None = None,
        route_floors: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        X = np.asarray(X)
        n_samples = len(X)
        preds = np.zeros((n_samples, 2), dtype=float)

        if self.routing == "global" or route_buildings is None:
            return self.global_model_.predict(X)

        b_arr = np.asarray(route_buildings)
        f_arr = np.asarray(route_floors) if route_floors is not None else None

        if self.routing == "per_building":
            for b, model in self.building_models_.items():
                mask = b_arr == b
                if np.any(mask):
                    preds[mask] = model.predict(X[mask])
            unassigned = np.ones(n_samples, dtype=bool)
            for b in self.building_models_:
                unassigned[b_arr == b] = False
            if np.any(unassigned):
                preds[unassigned] = self.global_model_.predict(X[unassigned])
            return preds

        if self.routing == "per_building_floor":
            if f_arr is None:
                raise ValueError(
                    "route_floors required for per_building_floor prediction"
                )

            assigned = np.zeros(n_samples, dtype=bool)
            # Try most specific floor models first
            for (b, f), model in self.floor_models_.items():
                mask = (b_arr == b) & (f_arr == f)
                if np.any(mask):
                    preds[mask] = model.predict(X[mask])
                    assigned[mask] = True

            # Fallback to parent building model
            remaining = ~assigned
            if np.any(remaining):
                for b, model in self.building_models_.items():
                    mask = remaining & (b_arr == b)
                    if np.any(mask):
                        preds[mask] = model.predict(X[mask])
                        assigned[mask] = True

            # Fallback to global model
            final_remaining = ~assigned
            if np.any(final_remaining):
                preds[final_remaining] = self.global_model_.predict(X[final_remaining])

            return preds

        raise ValueError(f"Unknown routing strategy {self.routing!r}")
