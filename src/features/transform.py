"""Fold-safe RSSI representations and engineered fingerprint features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

MISSING_RSSI = 100
FLOOR_RSSI = -110.0
WAP_PREFIX = "WAP"


class ObservedSignalScaler:
    """Scale detected RSSI values per WAP; keep missing signal zeros at zero."""

    def fit(self, x: np.ndarray, observed: np.ndarray) -> ObservedSignalScaler:
        self.signal_width_ = observed.shape[1]
        self.signal_mean_ = np.zeros(self.signal_width_, dtype=float)
        self.signal_scale_ = np.ones(self.signal_width_, dtype=float)
        for j in range(self.signal_width_):
            values = x[observed[:, j], j]
            if values.size:
                self.signal_mean_[j] = values.mean()
                scale = values.std()
                self.signal_scale_[j] = scale if scale > 0 else 1.0
        other = x[:, self.signal_width_ :]
        if other.shape[1] > 0:
            self.other_scaler_ = StandardScaler().fit(other)
        else:
            self.other_scaler_ = None
        return self

    def transform(self, x: np.ndarray, observed: np.ndarray) -> np.ndarray:
        result = x.copy()
        signal = result[:, : self.signal_width_]
        signal[observed] = ((signal - self.signal_mean_) / self.signal_scale_)[observed]
        if self.other_scaler_ is not None:
            result[:, self.signal_width_ :] = self.other_scaler_.transform(
                x[:, self.signal_width_ :]
            )
        return result


def wap_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith(WAP_PREFIX)]
    return sorted(cols, key=lambda c: int(c[3:]))


def _observed(frame: pd.DataFrame) -> np.ndarray:
    return frame[wap_columns(frame)].to_numpy(dtype=float) != MISSING_RSSI


def build_rssi_features(
    frame: pd.DataFrame, representation: str = "linear_positive"
) -> pd.DataFrame:
    """Convert +100 sentinels with one shared convention and apply an RSSI mapping.

    linear_positive maps [-110, 0] to [0, 1]. exponential uses exp(RSSI/24),
    and powed raises the normalized positive signal to beta=e. Missing entries stay
    zero in every representation; use the explicit mask to distinguish absence.
    """
    cols = wap_columns(frame)
    raw = frame[cols].to_numpy(dtype=float)
    mask = raw != MISSING_RSSI
    dbm = np.where(mask, raw, FLOOR_RSSI)
    if representation == "linear_positive":
        values = np.clip((dbm - FLOOR_RSSI) / (0.0 - FLOOR_RSSI), 0.0, 1.0)
    elif representation == "exponential":
        values = np.exp(dbm / 24.0)
    elif representation == "powed":
        values = np.power(np.clip((dbm - FLOOR_RSSI) / 110.0, 0.0, 1.0), np.e)
    else:
        raise ValueError(f"Unknown representation {representation!r}")
    values[~mask] = 0.0
    return pd.DataFrame(values, columns=cols, index=frame.index)


def build_engineered_features(frame: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """Add detection count, RSSI summaries, strongest AP and top-k AP ranks."""
    cols = wap_columns(frame)
    raw = frame[cols].to_numpy(dtype=float)
    mask = raw != MISSING_RSSI
    count = mask.sum(axis=1)
    safe = np.where(mask, raw, -np.inf)
    strongest_idx = safe.argmax(axis=1)
    strongest = safe[np.arange(len(raw)), strongest_idx]
    strongest[count == 0] = FLOOR_RSSI
    observed_sum = np.where(mask, raw, 0.0).sum(axis=1)
    observed_mean = np.divide(
        observed_sum,
        count,
        out=np.full(len(raw), FLOOR_RSSI, dtype=float),
        where=count > 0,
    )
    centered = np.where(mask, raw - observed_mean[:, None], 0.0)
    observed_std = np.sqrt(
        np.divide(
            np.square(centered).sum(axis=1),
            count,
            out=np.zeros(len(raw), dtype=float),
            where=count > 0,
        )
    )
    k = min(top_k, raw.shape[1])
    order = np.argsort(-safe, axis=1, kind="stable")[:, :k]
    data: dict[str, np.ndarray] = {
        "detected_ap_count": count,
        "strongest_rssi": strongest,
        "strongest_wap_index": np.where(count > 0, strongest_idx + 1, 0),
        "mean_observed_rssi": observed_mean,
        "std_observed_rssi": observed_std,
    }
    for rank in range(top_k):
        if rank < k:
            present = count > rank
            data[f"top{rank + 1}_wap_index"] = np.where(present, order[:, rank] + 1, 0)
            data[f"top{rank + 1}_rssi"] = np.where(
                present, safe[np.arange(len(raw)), order[:, rank]], FLOOR_RSSI
            )
        else:
            data[f"top{rank + 1}_wap_index"] = np.zeros(len(raw), dtype=int)
            data[f"top{rank + 1}_rssi"] = np.full(len(raw), FLOOR_RSSI, dtype=float)
    for j, col in enumerate(cols):
        data[f"{col}_missing"] = (~mask[:, j]).astype(np.uint8)
    return pd.DataFrame(data, index=frame.index)


class FoldFeatureBuilder:
    """Fit scale/PCA/variance filters on fit_df only, then transform any frame."""

    def __init__(
        self,
        representation: str = "linear_positive",
        scale: bool = True,
        n_components: int | None = None,
        variance_threshold: float | None = None,
        include_engineered: bool = True,
        include_missingness: bool = True,
    ):
        self.representation = representation
        self.scale = scale
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.include_engineered = include_engineered
        self.include_missingness = include_missingness
        self.scaler = ObservedSignalScaler() if scale else None
        self._signal_width = 0
        self.pca = (
            PCA(n_components=n_components, random_state=42) if n_components else None
        )
        self.selector = (
            VarianceThreshold(variance_threshold)
            if variance_threshold is not None
            else None
        )
        self._fit = False

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        parts = [build_rssi_features(frame, self.representation).to_numpy()]
        engineered = build_engineered_features(frame)
        if self.include_missingness:
            parts.append(engineered.filter(regex="_missing$").to_numpy())
        if self.include_engineered:
            parts.append(
                engineered.loc[
                    :, [c for c in engineered if not c.endswith("_missing")]
                ].to_numpy(dtype=float)
            )
        return np.concatenate(parts, axis=1)

    def fit(self, fit_df: pd.DataFrame) -> FoldFeatureBuilder:
        x = self._matrix(fit_df)
        self._signal_width = len(wap_columns(fit_df))
        if self.scaler:
            self.scaler.fit(x, _observed(fit_df))
            x = self.scaler.transform(x, _observed(fit_df))
        if self.selector:
            self.selector.fit(x)
            x = self.selector.transform(x)
        if self.pca:
            self.pca.fit(x)
        self._fit = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fit:
            raise RuntimeError(
                "Feature builder must be fit on a training fold before transform"
            )
        x = self._matrix(frame)
        if self.scaler:
            x = self.scaler.transform(x, _observed(frame))
        if self.selector:
            x = self.selector.transform(x)
        if self.pca:
            x = self.pca.transform(x)
        return x

    def get_feature_names_out(self, frame: pd.DataFrame) -> np.ndarray:
        """Return transformed feature names (PCA components if PCA is enabled)."""
        engineered = build_engineered_features(frame)
        names = list(wap_columns(frame))
        if self.include_missingness:
            names.extend(engineered.filter(regex="_missing$").columns.tolist())
        if self.include_engineered:
            names.extend([c for c in engineered if not c.endswith("_missing")])
        if self.selector is not None and self._fit:
            names = np.asarray(names)[self.selector.get_support()].tolist()
        if self.pca is not None:
            return np.asarray([f"PC{i + 1}" for i in range(self.pca.n_components_)])
        return np.asarray(names, dtype=object)

    def fit_transform(self, fit_df: pd.DataFrame) -> np.ndarray:
        self.fit(fit_df)
        return self.transform(fit_df)
