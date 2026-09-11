import numpy as np
import pandas as pd

import lstm_release_train as release


def _series(values, start="2024-01-01"):
    dates = pd.date_range(start, periods=len(values), freq="MS")
    frame = pd.DataFrame({
        "date": dates,
        "price": np.asarray(values, dtype=float),
        "month_sin": np.sin(2 * np.pi * dates.month / 12),
        "month_cos": np.cos(2 * np.pi * dates.month / 12),
        "year_norm": (dates.year - 2000) / 26.0,
        "region_enc": np.zeros(len(dates)),
        "pt_enc": np.zeros(len(dates)),
    })
    return frame


def test_scaled_targets_and_windows_use_one_chronological_series():
    frame = _series(np.arange(1, 25, dtype=float))
    scaler = release.MeanScale().fit(frame.loc[:17, "price"].to_numpy())
    X, y, dates = release.create_sequences(
        frame,
        scaler,
        target_start=pd.Period("2025-07", freq="M"),
        target_end=pd.Period("2025-12", freq="M"),
    )
    assert X.shape == (6, release.SEQ_LEN, release.N_FEATURES)
    expected = scaler.transform(frame.loc[18:23, "price"].to_numpy()).ravel()
    np.testing.assert_allclose(y, expected.astype(np.float32), rtol=1e-6)
    assert dates == ["2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]


def test_future_outlier_cannot_change_train_only_scaler():
    base = _series(np.arange(1, 25, dtype=float))
    altered = base.copy()
    altered.loc[18:, "price"] = 1_000_000.0
    cutoff = pd.Period("2025-06", freq="M")
    mask_a = base.date.dt.to_period("M") <= cutoff
    mask_b = altered.date.dt.to_period("M") <= cutoff
    scaler_a = release.MeanScale().fit(base.loc[mask_a, "price"].to_numpy())
    scaler_b = release.MeanScale().fit(altered.loc[mask_b, "price"].to_numpy())
    np.testing.assert_allclose(scaler_a.mean_, scaler_b.mean_)
    np.testing.assert_allclose(scaler_a.scale_, scaler_b.scale_)


def test_commodity_aggregation_produces_one_row_per_month():
    dates = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01", "2025-02-01"])
    monthly = pd.DataFrame({
        "commodity": ["Rice"] * 4,
        "date": dates,
        "price": [10.0, 20.0, 12.0, 22.0],
        "month_sin": [0.5, 0.5, 0.866, 0.866],
        "month_cos": [0.866, 0.866, 0.5, 0.5],
        "year_norm": [1.0] * 4,
        "region_enc": [0, 1, 0, 1],
        "pt_enc": [0, 0, 0, 0],
    })
    result = release.commodity_monthly_series(monthly, "Rice", pd.Period("2025-02", freq="M"))
    assert list(result["price"]) == [15.0, 17.0]
    assert len(result) == 2
    assert result["date"].is_monotonic_increasing


def test_mean_scale_inverse_round_trip():
    values = np.array([10.0, 12.0, 14.0, 16.0])
    scaler = release.MeanScale().fit(values)
    restored = scaler.inverse_transform(scaler.transform(values))
    np.testing.assert_allclose(restored, values)
