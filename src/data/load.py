"""Load and clean the official UJIIndoorLoc train/test files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_FILE = RAW_DIR / "trainingData.csv"
TEST_FILE = RAW_DIR / "validationData.csv"
WAP_COLUMNS = [f"WAP{i:03d}" for i in range(1, 521)]
TARGET_COLUMNS = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"]
META_COLUMNS = ["SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]
MISSING_RSSI = 100


def load_raw(data_dir: str | Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load official train and held-out test files without changing their rows."""
    data_dir = Path(data_dir)
    train_path, test_path = (
        data_dir / "trainingData.csv",
        data_dir / "validationData.csv",
    )
    missing = [str(p) for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "UJIIndoorLoc files are missing: "
            + ", ".join(missing)
            + ". Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def validate_schema(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail early if a file is not the expected 529-column UJIIndoorLoc table."""
    expected = set(WAP_COLUMNS + TARGET_COLUMNS + META_COLUMNS)
    for name, frame in (("training", train), ("validation", test)):
        absent = expected.difference(frame.columns)
        if absent:
            raise ValueError(
                f"{name} data is missing expected columns: {sorted(absent)}"
            )
        if len(frame.columns) != 529:
            raise ValueError(
                f"{name} data has {len(frame.columns)} columns; expected 529"
            )


def prepare_data(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Deduplicate train, remove train-constant WAPs from both splits, and report quality.

    Constant detection is fit on training scans only. Validation rows remain untouched;
    this preserves the official held-out test set for final evaluation.
    """
    validate_schema(train, test)
    train = train.copy()
    test = test.copy()
    n_train_before, n_test_before = len(train), len(test)
    duplicate_mask = train.duplicated(keep="first")
    duplicate_count = int(duplicate_mask.sum())
    train = train.loc[~duplicate_mask].reset_index(drop=True)
    never_detected_waps = [c for c in WAP_COLUMNS if train[c].eq(MISSING_RSSI).all()]
    constant_waps = [c for c in WAP_COLUMNS if train[c].nunique(dropna=False) == 1]
    train.drop(columns=constant_waps, inplace=True)
    test.drop(columns=constant_waps, inplace=True)

    remaining_waps = [c for c in WAP_COLUMNS if c in train.columns]
    observed = train[remaining_waps].ne(MISSING_RSSI)
    suspicious = train[remaining_waps].ge(-5) & train[remaining_waps].le(0)
    suspicious_cells = int(suspicious.to_numpy().sum())
    suspicious_users = sorted(
        int(v) for v in train.loc[suspicious.any(axis=1), "USERID"].dropna().unique()
    )
    quality = {
        "train_rows_before": n_train_before,
        "train_duplicate_rows_removed": duplicate_count,
        "train_rows_after": len(train),
        "test_rows_unchanged": len(test) == n_test_before,
        "test_rows": len(test),
        "never_detected_waps": never_detected_waps,
        "never_detected_wap_count": len(never_detected_waps),
        "constant_waps_removed": constant_waps,
        "constant_wap_count": len(constant_waps),
        "suspicious_near_zero_readings": suspicious_cells,
        "suspicious_users": suspicious_users,
        "train_detected_ap_counts": observed.sum(axis=1).describe().to_dict(),
    }
    return train, test, quality


def load_data(
    data_dir: str | Path = RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Convenience loader returning cleaned frames and a quality report."""
    train, test = load_raw(data_dir)
    return prepare_data(train, test)
