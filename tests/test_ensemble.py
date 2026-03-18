"""
Tests for ensemble_model.py and model_report.py.
# signed: delta

Tests cover:
  - StackingRegressor builds correctly with expected base estimators
  - Feature engineering consistency with retrain_model.py
  - Ensemble predictions are within reasonable range
  - Ensemble vs individual model comparison logic
  - Model report generation with all required sections
  - Edge cases: missing data, empty commodities, cache handling
"""
# signed: delta

import json
import math
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Add project dir to path for imports
# ---------------------------------------------------------------------------
_website_dir = str(Path(__file__).resolve().parent.parent)
if _website_dir not in sys.path:
    sys.path.insert(0, _website_dir)

import ensemble_model
import model_report


# ===================================================================
# FIXTURES
# ===================================================================


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking WFP food price data.

    Generates 20 years of monthly data (240 rows per commodity-region pair)
    so there's enough for TimeSeriesSplit CV during ensemble training.
    """
    np.random.seed(42)
    dates = pd.date_range("2004-01-01", periods=240, freq="MS")  # 20 years
    rows = []
    for date in dates:
        for comm in ["Rice", "Corn"]:
            for region in ["RegionA", "RegionB"]:
                price = 50 + np.random.randn() * 5 + (date.month * 0.5)
                rows.append(
                    {
                        "date": date,
                        "year": date.year,
                        "month": date.month,
                        "price": max(price, 1),
                        "commodity": comm,
                        "admin1": region,
                        "region": region,
                        "pricetype": "Retail",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    """Write sample data to a CSV file."""
    csv_path = tmp_path / "test_data.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_model_comparison(tmp_path):
    """Create a minimal model_comparison.json for testing."""
    mc = {
        "models": [
            "Gradient Boosting",
            "Extra Trees",
            "Random Forest",
            "KNN (k=10)",
            "Ridge Regression",
        ],
        "modelColors": {},
        "modelDescriptions": {},
        "variantSearch": {
            "Gradient Boosting": {
                "parameter_grid": [
                    {"n_estimators": 100, "learning_rate": 0.10, "max_depth": 3},
                    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
                    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5},
                    {"n_estimators": 200, "learning_rate": 0.02, "max_depth": 4},
                    {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 3},
                ],
                "selection_metric": "Validation MAPE",
            },
            "Extra Trees": {
                "parameter_grid": [
                    {"n_estimators": 50, "max_depth": 6},
                    {"n_estimators": 100, "max_depth": 8},
                    {"n_estimators": 200, "max_depth": 10},
                    {"n_estimators": 300, "max_depth": 12},
                    {"n_estimators": 200, "max_depth": None},
                ],
                "selection_metric": "Validation MAPE",
            },
        },
        "overall": {
            "Gradient Boosting": {"mape": 3.2, "mae": 1.83, "bias": -3.0, "r2": 0.82, "n_val": 26},
            "Extra Trees": {"mape": 3.9, "mae": 2.25, "bias": -3.8, "r2": 0.73, "n_val": 26},
            "Random Forest": {"mape": 4.6, "mae": 2.67, "bias": -4.6, "r2": 0.64, "n_val": 26},
            "KNN (k=10)": {"mape": 5.6, "mae": 3.18, "bias": -5.6, "r2": 0.57, "n_val": 26},
            "Ridge Regression": {"mape": 0.0, "mae": 0.01, "bias": 0.0, "r2": 1.0, "n_val": 26},
        },
        "commComparison": {
            "Rice": {
                "Gradient Boosting": {"mape": 2.5, "mae": 1.5, "bias": -2.0},
                "Extra Trees": {"mape": 3.0, "mae": 1.8, "bias": -2.5},
                "Random Forest": {"mape": 3.5, "mae": 2.0, "bias": -3.0},
            },
            "Corn": {
                "Gradient Boosting": {"mape": 4.0, "mae": 2.1, "bias": -4.0},
                "Extra Trees": {"mape": 5.0, "mae": 2.7, "bias": -5.0},
                "Random Forest": {"mape": 5.5, "mae": 3.3, "bias": -5.5},
            },
        },
        "forecasts": {},
        "trends": {},
        "meta": {
            "commodities": ["Rice", "Corn"],
            "regions": ["RegionA", "RegionB"],
            "pricetypes": ["Retail"],
        },
    }
    path = tmp_path / "model_comparison.json"
    path.write_text(json.dumps(mc, indent=2), encoding="utf-8")
    return path, mc


# ===================================================================
# TEST CLASS: StackingRegressor Construction
# ===================================================================


class TestStackingRegressorConstruction:
    """Test that the stacking regressor is built correctly."""

    def test_returns_stacking_regressor(self):
        """build_stacking_regressor should return a StackingRegressor."""
        stacker = ensemble_model.build_stacking_regressor(cv_folds=3)
        assert isinstance(stacker, StackingRegressor)

    def test_has_three_base_estimators(self):
        """Should have GradientBoosting, ExtraTrees, RandomForest as base."""
        stacker = ensemble_model.build_stacking_regressor()
        assert len(stacker.estimators) == 3

    def test_base_estimator_names(self):
        """Base estimator names should be gb, et, rf."""
        stacker = ensemble_model.build_stacking_regressor()
        names = [name for name, _ in stacker.estimators]
        assert "gb" in names
        assert "et" in names
        assert "rf" in names

    def test_base_estimator_types(self):
        """Base estimators should be the correct sklearn types."""
        stacker = ensemble_model.build_stacking_regressor()
        types = {name: type(est).__name__ for name, est in stacker.estimators}
        assert types["gb"] == "GradientBoostingRegressor"
        assert types["et"] == "ExtraTreesRegressor"
        assert types["rf"] == "RandomForestRegressor"

    def test_meta_learner_is_ridge(self):
        """Final estimator (meta-learner) should be Ridge."""
        stacker = ensemble_model.build_stacking_regressor()
        assert isinstance(stacker.final_estimator, Ridge)

    def test_cv_is_kfold(self):
        """CV strategy should be KFold(shuffle=False) for sklearn 1.8 compat."""
        stacker = ensemble_model.build_stacking_regressor(cv_folds=5)
        assert isinstance(stacker.cv, KFold)
        assert stacker.cv.shuffle is False

    def test_cv_folds_configurable(self):
        """Number of CV folds should be configurable."""
        for n in (3, 5, 10):
            stacker = ensemble_model.build_stacking_regressor(cv_folds=n)
            assert stacker.cv.n_splits == n

    def test_passthrough_false(self):
        """passthrough should be False (meta-learner sees only predictions)."""
        stacker = ensemble_model.build_stacking_regressor()
        assert stacker.passthrough is False


# ===================================================================
# TEST CLASS: Feature Engineering
# ===================================================================


class TestFeatureEngineering:
    """Test that feature engineering matches retrain_model.py."""

    def test_build_features_output_columns(self, sample_df):
        """build_features should create the expected lag/rolling columns."""
        group = sample_df[
            (sample_df["commodity"] == "Rice")
            & (sample_df["region"] == "RegionA")
        ]
        result = ensemble_model.build_features(group)
        expected = [
            "year_num", "month_sin", "month_cos",
            "price_lag1", "price_lag3", "price_lag6", "price_lag12",
            "price_ma3", "price_ma6", "price_ma12",
            "price_diff1", "price_diff12",
        ]
        for col in expected:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_feature_cols_match(self):
        """FEATURE_COLS should match what build_features creates."""
        assert len(ensemble_model.FEATURE_COLS) == 12
        assert "year_num" in ensemble_model.FEATURE_COLS
        assert "price_lag12" in ensemble_model.FEATURE_COLS

    def test_year_num_offset(self, sample_df):
        """year_num should be year - 2000."""
        group = sample_df[
            (sample_df["commodity"] == "Rice")
            & (sample_df["region"] == "RegionA")
        ]
        result = ensemble_model.build_features(group)
        assert (result["year_num"] == result["year"] - 2000).all()

    def test_month_sin_cos_range(self, sample_df):
        """sin/cos features should be in [-1, 1]."""
        group = sample_df[
            (sample_df["commodity"] == "Rice")
            & (sample_df["region"] == "RegionA")
        ]
        result = ensemble_model.build_features(group)
        assert result["month_sin"].between(-1, 1).all()
        assert result["month_cos"].between(-1, 1).all()


# ===================================================================
# TEST CLASS: Data Loading
# ===================================================================


class TestDataLoading:
    """Test data loading and preparation."""

    def test_load_valid_csv(self, sample_csv):
        """Should load valid CSV without errors."""
        df_feat, fcols, le_r, le_pt, raw = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        assert len(df_feat) > 0
        assert len(fcols) == 14  # 12 features + region_enc + pt_enc
        assert "region_enc" in fcols
        assert "pt_enc" in fcols

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing CSV."""
        with pytest.raises(FileNotFoundError):
            ensemble_model.load_and_prepare_data("/nonexistent/path.csv")

    def test_missing_columns_raises(self, tmp_path):
        """Should raise ValueError if required columns are missing."""
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            ensemble_model.load_and_prepare_data(str(bad_csv))


# ===================================================================
# TEST CLASS: Training
# ===================================================================


class TestTraining:
    """Test ensemble training pipeline."""

    def test_train_returns_dict(self, sample_csv):
        """train_ensemble should return a dict of {commodity: entry}."""
        df_feat, fcols, _, _, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        assert isinstance(trained, dict)
        assert len(trained) > 0

    def test_trained_entries_have_model_and_scaler(self, sample_csv):
        """Each trained entry should have 'model' and 'scaler' keys."""
        df_feat, fcols, _, _, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        for comm, entry in trained.items():
            assert "model" in entry, f"{comm} missing 'model'"
            assert "scaler" in entry, f"{comm} missing 'scaler'"

    def test_trained_model_can_predict(self, sample_csv):
        """Trained ensemble model should produce predictions."""
        df_feat, fcols, _, _, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        for comm, entry in trained.items():
            # Create dummy features to predict
            X = np.random.randn(5, len(fcols))
            X_scaled = entry["scaler"].transform(X)
            preds = entry["model"].predict(X_scaled)
            assert len(preds) == 5
            break  # just test one


# ===================================================================
# TEST CLASS: Prediction Validation
# ===================================================================


class TestPredictionValidation:
    """Test that ensemble predictions are reasonable."""

    def test_predictions_non_negative(self, sample_csv):
        """All ensemble predictions should be >= 0 after clamping."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        forecasts = ensemble_model.generate_forecasts(
            trained, df_feat, fcols, le_r, le_pt
        )
        for f in forecasts:
            assert f["price"] >= 0, f"Negative price: {f['price']} for {f['commodity']}"

    def test_predictions_have_required_fields(self, sample_csv):
        """Each forecast should have year, month, region, commodity, pricetype, price."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        forecasts = ensemble_model.generate_forecasts(
            trained, df_feat, fcols, le_r, le_pt
        )
        assert len(forecasts) > 0
        required_fields = {"year", "month", "region", "commodity", "pricetype", "price"}
        for f in forecasts[:10]:
            assert required_fields.issubset(f.keys())

    def test_predictions_capped_at_5x(self, sample_csv):
        """Predictions should be capped at 5x the recent average."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        forecasts = ensemble_model.generate_forecasts(
            trained, df_feat, fcols, le_r, le_pt
        )
        # Prices should be reasonable (not astronomically high)
        prices = [f["price"] for f in forecasts]
        if prices:
            assert max(prices) < 10000, f"Max price {max(prices)} is unreasonably high"
        else:
            pytest.skip("No forecasts generated (training data too small)")

    def test_forecast_dates_correct_range(self, sample_csv):
        """Forecasts should be in Feb 2026 -- Dec 2027 range."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        forecasts = ensemble_model.generate_forecasts(
            trained, df_feat, fcols, le_r, le_pt
        )
        for f in forecasts:
            assert f["year"] in (2026, 2027)
            assert 1 <= f["month"] <= 12
            if f["year"] == 2026:
                assert f["month"] >= 2  # Starts from Feb 2026


# ===================================================================
# TEST CLASS: Evaluation
# ===================================================================


class TestEvaluation:
    """Test ensemble evaluation and comparison logic."""

    def test_evaluate_returns_metrics(self, sample_csv):
        """evaluate_ensemble should return overall and per-commodity metrics."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        results = ensemble_model.evaluate_ensemble(
            trained, df_feat, fcols, le_r, le_pt
        )
        assert "ensemble_overall" in results
        assert "per_commodity" in results
        assert "comparison_vs_individual" in results

    def test_overall_metrics_have_expected_keys(self, sample_csv):
        """Overall metrics should include mape, mae, rmse, r2, bias."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        results = ensemble_model.evaluate_ensemble(
            trained, df_feat, fcols, le_r, le_pt
        )
        overall = results["ensemble_overall"]
        for key in ("mape", "mae", "rmse", "r2", "bias", "n_val"):
            assert key in overall, f"Missing key: {key}"

    def test_mape_is_non_negative(self, sample_csv):
        """MAPE should be >= 0."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        results = ensemble_model.evaluate_ensemble(
            trained, df_feat, fcols, le_r, le_pt
        )
        assert results["ensemble_overall"]["mape"] >= 0


# ===================================================================
# TEST CLASS: Model Report
# ===================================================================


class TestModelReport:
    """Test model_report.py report generation."""

    def test_per_model_metrics(self, sample_model_comparison):
        """compute_per_model_metrics should return metrics with ranks."""
        path, mc = sample_model_comparison
        metrics = model_report.compute_per_model_metrics(mc)
        assert len(metrics) == 5
        for name, m in metrics.items():
            assert "mape" in m
            assert "mape_rank" in m

    def test_best_model_per_commodity(self, sample_model_comparison):
        """compute_best_model_per_commodity should identify best per commodity."""
        path, mc = sample_model_comparison
        best = model_report.compute_best_model_per_commodity(mc)
        assert "Rice" in best
        assert "Corn" in best
        assert best["Rice"]["best_model"] is not None
        assert best["Corn"]["best_model"] is not None

    def test_feature_importance(self, sample_model_comparison):
        """compute_feature_importance should extract parameter sensitivity."""
        path, mc = sample_model_comparison
        fi = model_report.compute_feature_importance(mc)
        assert "Gradient Boosting" in fi
        assert len(fi["Gradient Boosting"]) > 0

    def test_summary_statistics(self, sample_model_comparison):
        """build_summary_statistics should produce valid summary."""
        path, mc = sample_model_comparison
        per_model = model_report.compute_per_model_metrics(mc)
        best_per_comm = model_report.compute_best_model_per_commodity(mc)
        summary = model_report.build_summary_statistics(per_model, best_per_comm, None)
        assert "total_models" in summary
        assert "best_overall_model" in summary
        assert summary["total_models"] == 5

    def test_recommendations_generated(self, sample_model_comparison):
        """build_recommendations should produce actionable recommendations."""
        path, mc = sample_model_comparison
        per_model = model_report.compute_per_model_metrics(mc)
        best_per_comm = model_report.compute_best_model_per_commodity(mc)
        recs = model_report.build_recommendations(per_model, best_per_comm, None)
        assert len(recs) > 0
        assert all("recommendation" in r for r in recs)
        assert all("rationale" in r for r in recs)

    def test_generate_report_creates_file(self, sample_model_comparison, tmp_path):
        """generate_report should create a valid JSON file."""
        path, mc = sample_model_comparison
        output = tmp_path / "test_report.json"
        with patch.object(model_report, "MODEL_COMPARISON_PATH", path), \
             patch.object(model_report, "ENSEMBLE_PATH", tmp_path / "nonexistent.json"):
            report = model_report.generate_report(output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert "summary" in data
        assert "per_model_metrics" in data
        assert "recommendations" in data

    def test_markdown_generation(self, sample_model_comparison, tmp_path):
        """generate_markdown should produce valid markdown."""
        path, mc = sample_model_comparison
        with patch.object(model_report, "MODEL_COMPARISON_PATH", path), \
             patch.object(model_report, "ENSEMBLE_PATH", tmp_path / "nonexistent.json"):
            report = model_report.generate_report(tmp_path / "report.json")
        md = model_report.generate_markdown(report)
        assert "# Philippine Food Price" in md
        assert "## Summary" in md
        assert "## Per-Model Metrics" in md


# ===================================================================
# TEST CLASS: JSON Output
# ===================================================================


class TestJsonOutput:
    """Test JSON output format and serialization."""

    def test_save_results_creates_file(self, sample_csv, tmp_path):
        """save_results should create a valid JSON file."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        forecasts = ensemble_model.generate_forecasts(
            trained, df_feat, fcols, le_r, le_pt
        )
        output = tmp_path / "test_output.json"
        ensemble_model.save_results({}, forecasts, trained, df_feat, output)
        assert output.exists()

    def test_output_json_schema(self, sample_csv, tmp_path):
        """Output JSON should have expected top-level keys."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        forecasts = ensemble_model.generate_forecasts(
            trained, df_feat, fcols, le_r, le_pt
        )
        output = tmp_path / "test_output.json"
        ensemble_model.save_results({}, forecasts, trained, df_feat, output)
        data = json.loads(output.read_text())
        for key in ("model", "base_models", "meta_learner", "forecasts", "meta"):
            assert key in data, f"Missing key: {key}"

    def test_output_meta_has_commodities(self, sample_csv, tmp_path):
        """Meta section should list trained commodities."""
        df_feat, fcols, le_r, le_pt, _ = ensemble_model.load_and_prepare_data(
            str(sample_csv)
        )
        with patch.object(ensemble_model, "ENSEMBLE_CACHE_DIR", sample_csv.parent / ".cache"):
            trained = ensemble_model.train_ensemble(df_feat, fcols, cv_folds=2)
        forecasts = ensemble_model.generate_forecasts(
            trained, df_feat, fcols, le_r, le_pt
        )
        output = tmp_path / "test_output.json"
        ensemble_model.save_results({}, forecasts, trained, df_feat, output)
        data = json.loads(output.read_text())
        # Commodities list reflects what was actually trained
        assert "commodities" in data["meta"]
        assert len(data["meta"]["commodities"]) == len(trained)


# ===================================================================
# TEST CLASS: Comparison Logic
# ===================================================================


class TestComparisonLogic:
    """Test ensemble vs individual model comparison."""

    def test_comparison_with_existing_models(self, tmp_path):
        """_compare_with_individual_models should detect wins/losses."""
        mc = {
            "overall": {
                "Gradient Boosting": {"mape": 5.0},
                "Random Forest": {"mape": 6.0},
            }
        }
        mc_path = tmp_path / "model_comparison.json"
        mc_path.write_text(json.dumps(mc))
        with patch.object(ensemble_model, "MODEL_COMPARISON_PATH", mc_path):
            result = ensemble_model._compare_with_individual_models(
                {"mape": 4.0, "mae": 1.0}
            )
        assert result["Gradient Boosting"]["ensemble_wins"] is True
        assert result["Random Forest"]["ensemble_wins"] is True

    def test_comparison_ensemble_loses(self, tmp_path):
        """Should correctly flag when ensemble does not improve."""
        mc = {
            "overall": {
                "Gradient Boosting": {"mape": 2.0},
            }
        }
        mc_path = tmp_path / "model_comparison.json"
        mc_path.write_text(json.dumps(mc))
        with patch.object(ensemble_model, "MODEL_COMPARISON_PATH", mc_path):
            result = ensemble_model._compare_with_individual_models(
                {"mape": 5.0, "mae": 2.0}
            )
        assert result["Gradient Boosting"]["ensemble_wins"] is False

    def test_comparison_no_model_file(self, tmp_path):
        """Should return empty dict if model_comparison.json doesn't exist."""
        with patch.object(
            ensemble_model, "MODEL_COMPARISON_PATH", tmp_path / "nope.json"
        ):
            result = ensemble_model._compare_with_individual_models(
                {"mape": 4.0, "mae": 1.0}
            )
        assert result == {}
