import numpy as np
import pandas as pd

import gpu_forecast_driver as driver


def test_window_ends_on_month_before_target():
    normalized = np.zeros(driver.SEQUENCE_LENGTH)
    context = {"year_min": 2000, "year_span": 26, "region_enc": 0, "pt_enc": 0}
    target = pd.Timestamp("2026-07-01")
    window = driver._window(normalized, target, context)
    expected_month = 6
    assert np.isclose(window[-1, 1], np.sin(2 * np.pi * expected_month / 12))
    assert np.isclose(window[-1, 2], np.cos(2 * np.pi * expected_month / 12))
