from pathlib import Path

import numpy as np
import pandas as pd

import gpu_forecast_driver as driver


def test_window_has_expected_shape_and_finite_values():
    normalized = np.linspace(-1, 1, driver.SEQUENCE_LENGTH)
    context = {"year_min": 2000, "year_span": 26, "region_enc": 3, "pt_enc": 1}
    window = driver._window(normalized, pd.Timestamp("2026-07-01"), context)
    assert window.shape == (driver.SEQUENCE_LENGTH, driver.FEATURE_COUNT)
    assert np.isfinite(window).all()


def test_percentile_uses_nearest_rank():
    assert driver._percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_safe_name_matches_checkpoint_convention():
    assert driver._safe_name("Rice (regular, milled)") == "Rice_(regular,_milled)"


def test_default_data_path_points_to_ml_wfp_folder():
    assert driver.DEFAULT_DATA == Path(driver.ROOT).parent / "WFP" / "wfp_food_prices_phl_latest.csv"


def test_seed_rejects_short_current_history_with_typed_reason():
    monthly = pd.DataFrame(
        {
            "commodity": ["Garlic (large)"] * 10,
            "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
            "price": np.arange(10, dtype=float) + 1,
            "region_enc": [0] * 10,
            "pt_enc": [0] * 10,
            "year": [2020] * 10,
        }
    )
    try:
        driver._seed_for_commodity(monthly, "Garlic (large)", 1.0, 1.0)
    except ValueError as exc:
        assert "Need 12 monthly observations" in str(exc)
    else:
        raise AssertionError("short histories must not be forecast")


def test_seed_rejects_series_older_than_dataset_latest():
    monthly = pd.DataFrame(
        {
            "commodity": ["Rice (paddy)"] * 12,
            "date": pd.date_range("2012-04-01", periods=12, freq="MS"),
            "price": np.arange(12, dtype=float) + 1,
            "region_enc": [0] * 12,
            "pt_enc": [0] * 12,
            "year": [2012] * 9 + [2013] * 3,
        }
    )
    try:
        driver._seed_for_commodity(
            monthly,
            "Rice (paddy)",
            1.0,
            1.0,
            required_latest=pd.Timestamp("2026-06-01"),
        )
    except ValueError as exc:
        assert "current dataset maximum is 2026-06" in str(exc)
    else:
        raise AssertionError("stale commodity histories must not be projected as current")
