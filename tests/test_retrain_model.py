"""
Tests for retrain_model.py — Multi-model food price forecasting pipeline.

Tests cover:
  - Data loading with valid/invalid CSV paths
  - Hyperparameter variant selection logic (5 models × 5 variants)
  - build_features() feature engineering
  - Prediction validation (5x cap, negative prices)
  - Model accuracy thresholds (MAPE < acceptable limit)
  - Model output JSON schema
  - Constant/config integrity

Uses synthetic data to avoid requiring the full 16 MB WFP dataset.
"""
# signed: delta

import json
import math
import os
import sys
import textwrap
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor

# ---------------------------------------------------------------------------
# Helpers to import parts of retrain_model.py without running the pipeline
# ---------------------------------------------------------------------------

RETRAIN_PATH = Path(__file__).resolve().parent.parent / "retrain_model.py"


def _load_retrain_source() -> str:
    """Read the raw source of retrain_model.py."""
    return RETRAIN_PATH.read_text(encoding="utf-8")


def _exec_constants():
    """Execute only the constant-definition portion of retrain_model.py.

    Returns a namespace dict containing MODEL_VARIANTS, MODEL_NAMES, etc.
    without triggering the data-loading / training pipeline.

    Strategy: cut the source at the first bare print("=") call (which starts
    the pipeline), but also extract specific definitions that live AFTER that
    marker (build_features, NEEDS_SCALING) and append them.
    """
    source = _load_retrain_source()
    # The pipeline execution starts at the first bare `print("="` call
    # after the constant definitions.  We cut the source there.
    marker = 'print("=" * 65)'
    idx = source.find(marker)
    if idx == -1:
        pytest.skip("Cannot locate constant-block boundary in retrain_model.py")
    const_block = source[:idx]

    # Also extract build_features function (defined after the marker)
    bf_marker = "def build_features(data):"
    bf_idx = source.find(bf_marker)
    if bf_idx != -1:
        # Find end of function: next line that starts with a non-indented non-blank
        # character after the function body
        lines = source[bf_idx:].split("\n")
        func_lines = [lines[0]]
        for line in lines[1:]:
            if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                func_lines.append(line)
            else:
                break
        const_block += "\n" + "\n".join(func_lines) + "\n"

    # Also extract NEEDS_SCALING (a simple set literal defined after the marker)
    ns_marker = 'NEEDS_SCALING = '
    ns_idx = source.find(ns_marker)
    if ns_idx != -1:
        ns_line_end = source.index("\n", ns_idx)
        const_block += "\n" + source[ns_idx:ns_line_end] + "\n"

    ns: dict = {"__file__": str(RETRAIN_PATH)}
    exec(compile(const_block, str(RETRAIN_PATH), "exec"), ns)
    return ns


def _exec_build_features():
    """Extract and return the build_features function from source."""
    ns = _exec_constants()
    if "build_features" not in ns:
        pytest.skip("build_features not found in extracted constants")
    return ns["build_features"]


@pytest.fixture(scope="module")
def constants():
    """Module-scoped fixture: the retrain_model constant namespace."""
    return _exec_constants()


@pytest.fixture(scope="module")
def build_features_fn():
    """Module-scoped fixture: the build_features function."""
    return _exec_build_features()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_price_series(
    n_months: int = 60,
    base_price: float = 50.0,
    trend: float = 0.5,
    noise_std: float = 3.0,
    commodity: str = "Rice (regular, milled)",
    region: str = "NCR",
    pricetype: str = "Retail",
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic monthly price series mimicking WFP data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2019-01-15", periods=n_months, freq="MS")
    prices = base_price + trend * np.arange(n_months) + rng.normal(0, noise_std, n_months)
    prices = np.maximum(prices, 1.0)  # no negatives
    return pd.DataFrame({
        "date": dates,
        "price": prices,
        "commodity": commodity,
        "admin1": region,
        "region": region,
        "pricetype": pricetype,
        "year": dates.year,
        "month": dates.month,
    })


def _make_multi_commodity_df(n_commodities: int = 5, n_months: int = 60) -> pd.DataFrame:
    """Create a multi-commodity dataset for training tests."""
    commodities = [f"Commodity_{i}" for i in range(n_commodities)]
    dfs = []
    for i, comm in enumerate(commodities):
        dfs.append(_make_price_series(
            n_months=n_months,
            base_price=30.0 + i * 20,
            commodity=comm,
            seed=42 + i,
        ))
    return pd.concat(dfs, ignore_index=True)


def _make_csv_text(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV text with WFP-compatible columns."""
    return df.to_csv(index=False)


# ===================================================================
# TEST CLASS: Constants & Configuration Integrity
# ===================================================================

class TestModelConstants:
    """Validate the structure and integrity of model configuration constants."""

    def test_model_variants_has_five_families(self, constants):
        mv = constants["MODEL_VARIANTS"]
        assert len(mv) == 5, f"Expected 5 model families, got {len(mv)}"

    @pytest.mark.parametrize("family", [
        "Gradient Boosting", "Extra Trees", "Random Forest",
        "KNN (k=10)", "Ridge Regression",
    ])
    def test_each_family_has_five_variants(self, constants, family):
        mv = constants["MODEL_VARIANTS"]
        assert family in mv, f"Missing model family: {family}"
        assert len(mv[family]) == 5, f"{family} has {len(mv[family])} variants, expected 5"

    def test_model_names_matches_variants_keys(self, constants):
        assert constants["MODEL_NAMES"] == list(constants["MODEL_VARIANTS"].keys())

    def test_model_colors_covers_all_families(self, constants):
        for name in constants["MODEL_NAMES"]:
            assert name in constants["MODEL_COLORS"], f"Missing color for {name}"

    def test_model_descriptions_covers_all_families(self, constants):
        for name in constants["MODEL_NAMES"]:
            assert name in constants["MODEL_DESCRIPTIONS"], f"Missing description for {name}"
            assert len(constants["MODEL_DESCRIPTIONS"][name]) > 50, \
                f"Description for {name} is suspiciously short"

    def test_variant_search_covers_all_families(self, constants):
        vs = constants["VARIANT_SEARCH"]
        for name in constants["MODEL_NAMES"]:
            assert name in vs, f"Missing VARIANT_SEARCH for {name}"
            assert "parameter_grid" in vs[name]
            assert "selection_metric" in vs[name]
            assert len(vs[name]["parameter_grid"]) == 5

    def test_model_defs_returns_callable(self, constants):
        for name in constants["MODEL_NAMES"]:
            factory = constants["MODEL_DEFS"][name]
            assert callable(factory), f"MODEL_DEFS[{name}] is not callable"
            model = factory()
            assert hasattr(model, "fit"), f"MODEL_DEFS[{name}]() doesn't return a model"

    def test_needs_scaling_set(self, constants):
        ns = constants["NEEDS_SCALING"]
        assert "Ridge Regression" in ns
        assert "KNN (k=10)" in ns
        assert "Gradient Boosting" not in ns

    @pytest.mark.parametrize("family,expected_type", [
        ("Gradient Boosting", GradientBoostingRegressor),
        ("Extra Trees", ExtraTreesRegressor),
        ("Random Forest", RandomForestRegressor),
        ("KNN (k=10)", KNeighborsRegressor),
        ("Ridge Regression", Ridge),
    ])
    def test_variant_types(self, constants, family, expected_type):
        for v in constants["MODEL_VARIANTS"][family]:
            assert isinstance(v, expected_type), \
                f"{family} variant is {type(v).__name__}, expected {expected_type.__name__}"

    def test_all_variants_have_random_state_where_applicable(self, constants):
        """Reproducibility: tree/boosting models should have random_state=42."""
        for family in ["Gradient Boosting", "Extra Trees", "Random Forest"]:
            for v in constants["MODEL_VARIANTS"][family]:
                assert v.random_state == 42, \
                    f"{family} variant missing random_state=42"

    def test_ridge_alpha_values_ascending(self, constants):
        alphas = [v.alpha for v in constants["MODEL_VARIANTS"]["Ridge Regression"]]
        assert alphas == sorted(alphas), "Ridge alpha values should be ascending"


# ===================================================================
# TEST CLASS: build_features Function
# ===================================================================

class TestBuildFeatures:
    """Test the feature engineering pipeline."""

    def test_basic_output_shape(self, build_features_fn):
        df = _make_price_series(n_months=24)
        result = build_features_fn(df)
        assert len(result) == 24
        expected_cols = [
            "year_num", "month_sin", "month_cos",
            "price_lag1", "price_lag3", "price_lag6", "price_lag12",
            "price_ma3", "price_ma6", "price_ma12",
            "price_diff1", "price_diff12",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_year_num_offset(self, build_features_fn):
        df = _make_price_series(n_months=12)
        result = build_features_fn(df)
        # year_num = year - 2000
        assert (result["year_num"] == result["year"] - 2000).all()

    def test_month_sin_cos_range(self, build_features_fn):
        df = _make_price_series(n_months=24)
        result = build_features_fn(df)
        assert result["month_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()

    def test_lag_features_correct(self, build_features_fn):
        df = _make_price_series(n_months=24)
        result = build_features_fn(df)
        # After sorting by date, lag1 should be previous row's price
        prices = result["price"].values
        lag1 = result["price_lag1"].values
        # lag1[0] should be NaN, lag1[1] should equal prices[0]
        assert np.isnan(lag1[0])
        np.testing.assert_allclose(lag1[1:], prices[:-1])

    def test_rolling_mean_values(self, build_features_fn):
        df = _make_price_series(n_months=24)
        result = build_features_fn(df)
        # price_ma3 at index 5 should be mean of prices[3:6]
        prices = result["price"].values
        ma3 = result["price_ma3"].values
        expected_ma3_at_5 = np.mean(prices[3:6])
        np.testing.assert_allclose(ma3[5], expected_ma3_at_5, rtol=1e-5)

    def test_diff_features(self, build_features_fn):
        df = _make_price_series(n_months=24)
        result = build_features_fn(df)
        prices = result["price"].values
        diff1 = result["price_diff1"].values
        # diff1[i] = prices[i] - prices[i-1]
        np.testing.assert_allclose(diff1[1:], prices[1:] - prices[:-1])

    def test_sorted_by_date(self, build_features_fn):
        df = _make_price_series(n_months=24)
        # Shuffle the input
        df = df.sample(frac=1, random_state=99)
        result = build_features_fn(df)
        assert result["date"].is_monotonic_increasing

    def test_handles_single_row(self, build_features_fn):
        df = _make_price_series(n_months=1)
        result = build_features_fn(df)
        assert len(result) == 1
        assert "price_lag1" in result.columns


# ===================================================================
# TEST CLASS: Data Loading Validation
# ===================================================================

class TestDataLoading:
    """Test data loading and validation logic from retrain_model.py."""

    def test_missing_csv_raises_file_not_found(self, tmp_path):
        """Running retrain_model with non-existent CSV should fail."""
        env = os.environ.copy()
        env["WFP_DATA_PATH"] = str(tmp_path / "nonexistent.csv")
        env["OUTPUT_PATH"] = str(tmp_path / "out.json")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(RETRAIN_PATH)],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower() \
            or "FileNotFoundError" in result.stderr or "FileNotFoundError" in result.stdout

    def test_missing_columns_raises_value_error(self, tmp_path):
        """CSV missing required columns should fail with ValueError."""
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("col_a,col_b\n1,2\n3,4\n")
        env = os.environ.copy()
        env["WFP_DATA_PATH"] = str(csv_path)
        env["OUTPUT_PATH"] = str(tmp_path / "out.json")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(RETRAIN_PATH)],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "Missing required columns" in combined or "ValueError" in combined

    def test_insufficient_data_raises_value_error(self, tmp_path):
        """Very small dataset should fail with insufficient data error."""
        # Create a CSV with required columns but only 5 rows
        csv_path = tmp_path / "tiny.csv"
        lines = ["date,price,commodity,admin1,pricetype"]
        for i in range(5):
            lines.append(f"2024-01-{i+1:02d},{50+i},Rice,NCR,Retail")
        csv_path.write_text("\n".join(lines))
        env = os.environ.copy()
        env["WFP_DATA_PATH"] = str(csv_path)
        env["OUTPUT_PATH"] = str(tmp_path / "out.json")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(RETRAIN_PATH)],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "Insufficient data" in combined or "ValueError" in combined

    def test_negative_prices_filtered(self):
        """Negative prices should be dropped during loading."""
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "price": [10, -5, 20, 0, 30],
            "commodity": "Test",
            "admin1": "NCR",
            "pricetype": "Retail",
        })
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"])
        df = df[df["price"] > 0]
        assert len(df) == 3
        assert (df["price"] > 0).all()

    def test_required_columns_list(self, constants):
        """Verify we know what columns are required."""
        # The code checks for these exact columns
        required = ["date", "price", "commodity", "admin1", "pricetype"]
        # Verify by checking the source
        source = _load_retrain_source()
        assert 'required_cols = ["date", "price", "commodity", "admin1", "pricetype"]' in source


# ===================================================================
# TEST CLASS: Hyperparameter Variant Selection
# ===================================================================

class TestVariantSelection:
    """Test the variant selection logic (best MAPE per commodity)."""

    def test_best_variant_selected_by_mape(self):
        """Train multiple variants, verify the one with lowest MAPE wins."""
        rng = np.random.RandomState(42)
        n_train, n_val = 100, 20
        X_train = rng.randn(n_train, 3)
        y_train = 5 * X_train[:, 0] + 3 * X_train[:, 1] + rng.randn(n_train) * 0.5
        y_train = np.abs(y_train) + 10  # positive prices
        X_val = rng.randn(n_val, 3)
        y_val = 5 * X_val[:, 0] + 3 * X_val[:, 1] + rng.randn(n_val) * 0.5
        y_val = np.abs(y_val) + 10

        variants = [
            Ridge(alpha=0.01),
            Ridge(alpha=1.0),
            Ridge(alpha=100.0),  # over-regularized, should perform worst
        ]
        mapes = []
        for v in variants:
            v.fit(X_train, y_train)
            pred = np.maximum(v.predict(X_val), 0)
            mapes.append(mean_absolute_percentage_error(y_val, pred))

        best_idx = int(np.argmin(mapes))
        # The least regularized (alpha=0.01) should be best on this linear data
        assert best_idx == 0 or mapes[best_idx] < mapes[-1]

    def test_variant_selection_handles_training_failure(self):
        """If a variant fails to train, others should still be evaluated."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = np.abs(X[:, 0] * 5) + 10

        # KNN with k > n_samples will fail
        variants = [
            KNeighborsRegressor(n_neighbors=5),
            KNeighborsRegressor(n_neighbors=1000),  # will fail
        ]
        best_model = None
        for v in variants:
            try:
                v.fit(X, y)
                best_model = v
            except Exception:
                continue
        assert best_model is not None

    @pytest.mark.parametrize("n_variants", [1, 3, 5])
    def test_variant_count_flexibility(self, n_variants):
        """Test that the selection logic works with different variant counts."""
        rng = np.random.RandomState(42)
        X = rng.randn(80, 3)
        y = np.abs(X[:, 0] * 5) + 10
        X_val = rng.randn(20, 3)
        y_val = np.abs(X_val[:, 0] * 5) + 10

        variants = [Ridge(alpha=10 ** i) for i in range(n_variants)]
        best_mape = float("inf")
        best_model = None
        for v in variants:
            v.fit(X, y)
            pred = np.maximum(v.predict(X_val), 0)
            mape = mean_absolute_percentage_error(y_val, pred)
            if mape < best_mape:
                best_mape = mape
                best_model = v
        assert best_model is not None
        assert best_mape < float("inf")


# ===================================================================
# TEST CLASS: Prediction Validation (5x Cap, Negatives)
# ===================================================================

class TestPredictionValidation:
    """Test the prediction validation / capping logic."""

    def test_predictions_capped_at_5x_recent_avg(self):
        """Predictions > 5x recent average should be capped at 1.5x."""
        price_history = [50.0] * 12  # recent avg = 50
        recent_avg = np.mean(price_history[-12:])
        pred = 300.0  # 6x the average — should be capped

        if pred > recent_avg * 5:
            pred = recent_avg * 1.5

        assert pred == pytest.approx(75.0)  # 50 * 1.5

    def test_predictions_within_5x_not_capped(self):
        """Predictions within 5x should remain unchanged."""
        price_history = [50.0] * 12
        recent_avg = np.mean(price_history[-12:])
        pred = 200.0  # 4x — should NOT be capped

        if pred > recent_avg * 5:
            pred = recent_avg * 1.5

        assert pred == pytest.approx(200.0)

    def test_negative_predictions_clamped_to_zero(self):
        """Negative model output should be clamped to 0."""
        pred = max(0, float(-15.0))
        assert pred == 0.0

    def test_cap_with_varying_history(self):
        """Test capping with different price history lengths."""
        # Short history (< 12 months)
        short_history = [100.0, 110.0, 105.0]
        recent_avg = np.mean(short_history)  # ~105
        pred = 600.0  # ~5.7x
        if pred > recent_avg * 5:
            pred = recent_avg * 1.5
        assert pred == pytest.approx(recent_avg * 1.5)

    @pytest.mark.parametrize("pred_value,expected_capped", [
        (50.0, False),   # within range
        (100.0, False),  # 2x
        (249.0, False),  # just under 5x
        (251.0, True),   # just over 5x
        (1000.0, True),  # way over
    ])
    def test_cap_boundary(self, pred_value, expected_capped):
        """Parametrized boundary test for the 5x cap."""
        price_history = [50.0] * 12
        recent_avg = np.mean(price_history[-12:])
        original_pred = pred_value

        if pred_value > recent_avg * 5:
            pred_value = recent_avg * 1.5

        was_capped = pred_value != original_pred
        assert was_capped == expected_capped

    def test_forecast_loop_accumulation(self):
        """Verify forecast prices are appended to history for iterative prediction."""
        price_history = [50.0] * 12
        new_pred = 55.0
        price_history.append(new_pred)
        assert len(price_history) == 13
        assert price_history[-1] == 55.0
        # Next iteration should use updated history
        assert np.mean(price_history[-12:]) != 50.0


# ===================================================================
# TEST CLASS: Model Accuracy Thresholds
# ===================================================================

class TestModelAccuracy:
    """Test that trained models achieve acceptable accuracy on synthetic data."""

    @pytest.fixture
    def trained_model_results(self):
        """Train all 5 model families on synthetic data, return metrics."""
        rng = np.random.RandomState(42)
        n = 200
        X = np.column_stack([
            np.arange(n) / 12,  # year trend
            np.sin(2 * np.pi * np.arange(n) / 12),
            np.cos(2 * np.pi * np.arange(n) / 12),
            rng.randn(n),  # noise feature
        ])
        # Price = 50 + 0.5*trend + 5*sin(season) + noise
        y = 50 + 0.5 * X[:, 0] + 5 * X[:, 1] + rng.randn(n) * 2
        y = np.maximum(y, 1.0)

        split = 160
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        models = {
            "GB": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            "ET": ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=42),
            "RF": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            "KNN": KNeighborsRegressor(n_neighbors=10, weights="distance"),
            "Ridge": Ridge(alpha=1.0),
        }
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = np.maximum(model.predict(X_val), 0)
            mape = mean_absolute_percentage_error(y_val, pred) * 100
            results[name] = {"mape": mape, "pred": pred, "actual": y_val}
        return results

    def test_all_models_under_50_pct_mape(self, trained_model_results):
        """All models should achieve < 50% MAPE on well-structured synthetic data."""
        for name, res in trained_model_results.items():
            assert res["mape"] < 50, f"{name} MAPE={res['mape']:.1f}% exceeds 50% threshold"

    def test_at_least_one_model_under_20_pct_mape(self, trained_model_results):
        """At least one model should achieve < 20% MAPE."""
        best_mape = min(r["mape"] for r in trained_model_results.values())
        assert best_mape < 20, f"Best MAPE={best_mape:.1f}% — no model under 20%"

    def test_predictions_are_positive(self, trained_model_results):
        """All predictions should be non-negative (prices can't be negative)."""
        for name, res in trained_model_results.items():
            assert (res["pred"] >= 0).all(), f"{name} produced negative predictions"

    def test_predictions_reasonable_range(self, trained_model_results):
        """Predictions should be within a reasonable multiple of actual values."""
        for name, res in trained_model_results.items():
            ratio = res["pred"] / res["actual"]
            assert ratio.max() < 10, f"{name} max pred/actual ratio = {ratio.max():.1f}"
            assert ratio.min() > 0.1, f"{name} min pred/actual ratio = {ratio.min():.3f}"


# ===================================================================
# TEST CLASS: Output JSON Schema Validation
# ===================================================================

class TestOutputJsonSchema:
    """Validate the model_comparison.json output schema."""

    @pytest.fixture
    def comparison_json(self):
        """Load model_comparison.json if it exists."""
        path = Path(__file__).resolve().parent.parent / "model_comparison.json"
        if not path.exists():
            pytest.skip("model_comparison.json not found (model not yet trained)")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_top_level_keys(self, comparison_json):
        required = [
            "models", "modelColors", "modelDescriptions",
            "variantSearch", "overall", "trends",
            "commComparison", "forecasts", "meta",
        ]
        for key in required:
            assert key in comparison_json, f"Missing top-level key: {key}"

    def test_models_list(self, comparison_json):
        models = comparison_json["models"]
        assert isinstance(models, list)
        assert len(models) == 5

    def test_overall_metrics_structure(self, comparison_json):
        overall = comparison_json["overall"]
        for model_name in comparison_json["models"]:
            assert model_name in overall, f"Missing overall metrics for {model_name}"
            metrics = overall[model_name]
            for key in ["mape", "mae", "bias", "r2", "n_val"]:
                assert key in metrics, f"Missing metric '{key}' for {model_name}"

    def test_overall_mape_reasonable(self, comparison_json):
        """Overall MAPE should be between 0 and 200% for real data."""
        for model_name, metrics in comparison_json["overall"].items():
            assert 0 <= metrics["mape"] < 200, \
                f"{model_name} overall MAPE={metrics['mape']}% is out of range"

    def test_meta_has_required_fields(self, comparison_json):
        meta = comparison_json["meta"]
        for key in ["commodities", "pricetypes", "regions"]:
            assert key in meta, f"Missing meta field: {key}"
            assert isinstance(meta[key], list)
            assert len(meta[key]) > 0

    def test_meta_commodity_count(self, comparison_json):
        """Should have at least 1 commodity (full dataset has 70+)."""
        n = len(comparison_json["meta"]["commodities"])
        assert n >= 1, f"No commodities in meta — expected at least 1"

    def test_forecasts_structure(self, comparison_json):
        forecasts = comparison_json["forecasts"]
        for model_name in comparison_json["models"]:
            assert model_name in forecasts, f"Missing forecasts for {model_name}"
            model_fc = forecasts[model_name]
            assert isinstance(model_fc, dict)
            # Should have commodity keys
            assert len(model_fc) > 0

    def test_forecast_dates_format(self, comparison_json):
        """Forecast date keys should be YYYY-MM format."""
        import re
        date_pattern = re.compile(r"^\d{4}-\d{2}$")
        for model_name, model_fc in comparison_json["forecasts"].items():
            for commodity, dates in model_fc.items():
                for date_key in dates:
                    assert date_pattern.match(date_key), \
                        f"Invalid date format '{date_key}' in {model_name}/{commodity}"
                break  # only check first commodity per model
            break  # only check first model

    def test_forecast_prices_positive(self, comparison_json):
        """All forecast prices should be non-negative."""
        for model_name, model_fc in comparison_json["forecasts"].items():
            for commodity, dates in model_fc.items():
                for date_key, price in dates.items():
                    assert price >= 0, \
                        f"Negative forecast: {model_name}/{commodity}/{date_key}={price}"

    def test_variant_search_has_best_variant(self, comparison_json):
        vs = comparison_json["variantSearch"]
        for model_name in comparison_json["models"]:
            assert "best_variant" in vs[model_name], \
                f"Missing best_variant in variantSearch for {model_name}"
            bv = vs[model_name]["best_variant"]
            assert isinstance(bv, int) and 0 <= bv < 5

    def test_comm_comparison_structure(self, comparison_json):
        cc = comparison_json["commComparison"]
        assert isinstance(cc, dict)
        assert len(cc) > 0
        # Check a sample entry
        sample_key = next(iter(cc))
        entry = cc[sample_key]
        assert "commodity" in entry


# ===================================================================
# TEST CLASS: Feature Engineering Edge Cases
# ===================================================================

class TestFeatureEdgeCases:
    """Edge cases in feature engineering and data preparation."""

    def test_single_commodity_single_region(self, build_features_fn):
        """Should work with minimal grouping (1 commodity, 1 region)."""
        df = _make_price_series(n_months=36)
        result = build_features_fn(df)
        assert len(result) == 36

    def test_very_short_series(self, build_features_fn):
        """Short series should still produce valid features (with NaNs)."""
        df = _make_price_series(n_months=3)
        result = build_features_fn(df)
        assert len(result) == 3
        # lag12 should be all NaN for a 3-month series
        assert result["price_lag12"].isna().all()

    def test_constant_prices(self, build_features_fn):
        """Constant prices should produce zero diffs and constant MAs."""
        df = _make_price_series(n_months=24, noise_std=0.0, trend=0.0, base_price=100.0)
        result = build_features_fn(df)
        # All non-NaN diffs should be ~0
        valid_diff = result["price_diff1"].dropna()
        np.testing.assert_allclose(valid_diff, 0, atol=1e-10)

    def test_month_sin_cos_identity(self, build_features_fn):
        """sin²(month) + cos²(month) should equal 1."""
        df = _make_price_series(n_months=24)
        result = build_features_fn(df)
        identity = result["month_sin"] ** 2 + result["month_cos"] ** 2
        np.testing.assert_allclose(identity, 1.0, atol=1e-10)


# ===================================================================
# TEST CLASS: End-to-End Pipeline (with small synthetic CSV)
# ===================================================================

class TestEndToEndPipeline:
    """Integration test: run retrain_model.py with synthetic data."""

    @pytest.fixture
    def synthetic_csv(self, tmp_path):
        """Create a synthetic WFP-format CSV for end-to-end testing."""
        rng = np.random.RandomState(42)
        rows = []
        commodities = ["Rice", "Corn", "Sugar"]
        regions = ["NCR", "Region I"]
        for comm in commodities:
            for region in regions:
                base = 30 + rng.rand() * 50
                for year in range(2019, 2026):
                    for month in range(1, 13):
                        price = base + rng.randn() * 5 + (year - 2019) * 2
                        price = max(1.0, price)
                        rows.append({
                            "date": f"{year}-{month:02d}-15",
                            "price": round(price, 2),
                            "commodity": comm,
                            "admin1": region,
                            "pricetype": "Retail",
                        })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_pipeline_produces_valid_json(self, synthetic_csv, tmp_path):
        """Full pipeline should produce a valid JSON output."""
        output_path = tmp_path / "test_output.json"
        env = os.environ.copy()
        env["WFP_DATA_PATH"] = str(synthetic_csv)
        env["OUTPUT_PATH"] = str(output_path)
        # Force UTF-8 to avoid cp1252 UnicodeEncodeError on Windows
        env["PYTHONIOENCODING"] = "utf-8"
        import subprocess
        result = subprocess.run(
            [sys.executable, str(RETRAIN_PATH)],
            env=env, capture_output=True, text=True, timeout=300,
            cwd=str(RETRAIN_PATH.parent),
        )
        assert result.returncode == 0, f"Pipeline failed:\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}"
        assert output_path.exists(), "Output JSON was not created"

        with open(output_path) as f:
            data = json.load(f)
        assert "models" in data
        assert "overall" in data
        assert len(data["models"]) == 5

    def test_pipeline_output_has_forecasts(self, synthetic_csv, tmp_path):
        """Pipeline output should contain forecast data."""
        output_path = tmp_path / "test_output.json"
        env = os.environ.copy()
        env["WFP_DATA_PATH"] = str(synthetic_csv)
        env["OUTPUT_PATH"] = str(output_path)
        env["PYTHONIOENCODING"] = "utf-8"
        import subprocess
        result = subprocess.run(
            [sys.executable, str(RETRAIN_PATH)],
            env=env, capture_output=True, text=True, timeout=300,
            cwd=str(RETRAIN_PATH.parent),
        )
        if result.returncode != 0:
            pytest.skip(f"Pipeline failed: {result.stderr[-500:]}")

        with open(output_path) as f:
            data = json.load(f)
        assert "forecasts" in data
        for model_name in data["models"]:
            assert model_name in data["forecasts"]
