"""Export the Member 1 report data-quality table as CSV and Markdown."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.load import load_data  # noqa: E402


def export() -> None:
    train, test, q = load_data()
    rows = [
        (
            "Exact duplicate training rows",
            q["train_duplicate_rows_removed"],
            "Drop later copies from training; preserve first occurrence.",
        ),
        (
            "WAPs never detected in training (constant columns)",
            q["never_detected_wap_count"],
            "Drop these WAP columns from train and official test using training-only detection.",
        ),
        (
            "Observed RSSI readings in [-5, 0] dBm",
            q["suspicious_near_zero_readings"],
            f"Flag only; retain pending source/user investigation. USERIDs: {q['suspicious_users']}",
        ),
        (
            "Official held-out test rows",
            len(test),
            "Keep validationData.csv untouched for final evaluation only.",
        ),
        (
            "Training scans after duplicate removal",
            len(train),
            "Use for grouped inner cross-validation.",
        ),
    ]
    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    import csv

    with (out / "data_quality.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["issue", "count", "action"])
        writer.writerows(rows)
    md = ["| Issue | Count | Action |", "|---|---:|---|"]
    md += [f"| {issue} | {count} | {action} |" for issue, count, action in rows]
    (ROOT / "docs/data_quality.md").write_text("\n".join(md) + "\n")
    (ROOT / "data/processed/quality_report.json").write_text(
        json.dumps(q, indent=2) + "\n"
    )
    print(
        f"Exported quality report for {len(train):,} training and {len(test):,} test rows"
    )


if __name__ == "__main__":
    export()
