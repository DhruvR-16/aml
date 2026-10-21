| Issue | Count | Action |
|---|---:|---|
| Exact duplicate training rows | 637 | Drop later copies from training; preserve first occurrence. |
| WAPs never detected in training (constant columns) | 55 | Drop these WAP columns from train and official test using training-only detection. |
| Observed RSSI readings in [-5, 0] dBm | 224 | Flag only; retain pending source/user investigation. USERIDs: [6, 14] |
| Official held-out test rows | 1111 | Keep validationData.csv untouched for final evaluation only. |
| Training scans after duplicate removal | 19300 | Use for grouped inner cross-validation. |
