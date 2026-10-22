# Phase 1 theory notes

## Member 1: propagation and missing access points

A common indoor propagation approximation is the log-distance path-loss model:

`P_r(d) = P_r(d_0) - 10 n log10(d / d_0) + X_sigma`,

where received power is in dBm, `d_0` is a reference distance, `n` is the environment-dependent path-loss exponent, and `X_sigma` captures shadowing and measurement variation. Walls, floors, multipath, device antenna differences, and changing occupancy violate the idealized assumptions, so a Wi-Fi fingerprint is an empirical pattern rather than a direct range measurement.

UJIIndoorLoc stores `+100` when an access point was not detected. That is a categorical observation about receiver visibility, not a measured RSSI of `+100 dBm`, and it is not equivalent to a very weak but detected signal. A shared numeric convention of `-110 dBm` places absence below the weakest recorded signal while remaining outside the observed `[-104, 0] dBm` range. The per-WAP missingness flags preserve the distinction after numeric transformation.

## Member 2: similarity transforms and PCA

Raw Euclidean distance treats a 1 dBm change as equally meaningful at every power level and mixes signal magnitude with access-point absence. A monotonic or power transform changes the relative influence of strong and weak readings; explicit missingness flags preserve the detection pattern. The linear-positive transform is `(rssi + 110) / 110`; the exponential representation is `exp(rssi / 24)`; `powed` is the normalized positive value raised to `beta=e`, following Torres-Sospedra et al. (2015). Missing entries are first mapped to `-110 dBm` and represented as zero in each signal view. These are feature hypotheses, so select among them by USERID-grouped validation, not the official test set.

For a centered training-fold feature matrix `X`, PCA computes eigenvectors of `X.T @ X` (equivalently right singular vectors of `X`). The first `k` components maximize retained variance subject to orthogonality. The explained-variance ratio for component `j` is `lambda_j / sum(lambda)`, and cumulative retained variance is its sum over the selected components. PCA and all scaling parameters must be fit on the training fold and reused unchanged for its validation fold and the official test set.

Reference: Torres-Sospedra et al., “Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems,” Expert Systems with Applications, 2015, doi:10.1016/j.eswa.2015.08.013.
