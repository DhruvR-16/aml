"""Write deterministic grouped folds and the official held-out test split indices."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.data.load import load_data  # noqa: E402


def build(output: Path = ROOT / "data/splits/folds.json") -> dict:
    train, test, quality = load_data()
    splitter = GroupKFold(n_splits=5)
    folds = []
    for fold, (fit_idx, valid_idx) in enumerate(
        splitter.split(train, groups=train["USERID"]), start=0
    ):
        folds.append(
            {
                "fold": fold,
                "train_idx": fit_idx.tolist(),
                "valid_idx": valid_idx.tolist(),
            }
        )
    payload = {
        "dataset": "UJIIndoorLoc",
        "seed": 42,
        "splitter": "GroupKFold(n_splits=5)",
        "group_column": "USERID",
        "train_rows": len(train),
        "official_test_rows": len(test),
        "train_user_ids": sorted(int(v) for v in train.USERID.unique()),
        "official_test_is_validationData_csv": True,
        "folds": folds,
        "quality": quality,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"Saved {len(folds)} folds for {len(train):,} train rows; test rows={len(test):,} to {output}"
    )
    return payload


if __name__ == "__main__":
    build()
