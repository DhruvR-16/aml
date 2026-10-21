"""Leakage and sentinel behavior checks for Member 2's shared feature builder."""

import numpy as np
import pandas as pd

from src.features.transform import FoldFeatureBuilder, build_rssi_features


def small_frame(values, index=None):
    return pd.DataFrame({"WAP001": values, "WAP002": [100] * len(values)}, index=index)


def test_missing_signal_maps_below_real_values_and_masks_as_zero():
    frame = small_frame([-50, 100])
    x = build_rssi_features(frame, "linear_positive")
    assert x.loc[0, "WAP001"] == 60 / 110
    assert x.loc[1, "WAP001"] == 0


def test_fit_statistics_do_not_include_validation_or_test_rows():
    fit = small_frame([-100, -50])
    held_out = small_frame([0])
    builder = FoldFeatureBuilder(include_engineered=False, include_missingness=False)
    builder.fit(fit)
    fitted_mean = builder.scaler.signal_mean_.copy()
    builder.transform(held_out)
    np.testing.assert_array_equal(builder.scaler.signal_mean_, fitted_mean)
    np.testing.assert_allclose(fitted_mean, np.array([35 / 110, 0.0]))


def test_transform_requires_fit():
    builder = FoldFeatureBuilder()
    try:
        builder.transform(small_frame([-50]))
    except RuntimeError as exc:
        assert "fit" in str(exc)
    else:
        raise AssertionError("transform should refuse to use unfitted feature state")
