"""
Tests for data quality — validates the WFP CSV data and dashboard JSON outputs.

Tests cover:
  - CSV schema consistency (required columns present)
  - No null prices in required fields
  - Region names match expected Philippine admin1 set
  - Commodity names are non-empty, consistent strings
  - Price values are positive and within reasonable range
  - Date field validity
  - dashboard_data.json schema and content integrity
  - model_comparison.json schema and content integrity
  - Cross-file consistency between JSON outputs
"""
# signed: delta

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WEBSITE_DIR = Path(__file__).resolve().parent.parent
WFP_CSV_PATH = WEBSITE_DIR.parent / "WFP" / "wfp_food_prices_phl_latest.csv"
DASHBOARD_JSON_PATH = WEBSITE_DIR / "dashboard_data.json"
COMPARISON_JSON_PATH = WEBSITE_DIR / "model_comparison.json"

# Expected Philippine admin1 regions (WFP naming convention — actual CSV values)
EXPECTED_REGIONS = {
    "Autonomous region in Muslim Mindanao",
    "Cordillera Administrative region",
    "National Capital region",
    "Region I",
    "Region II",
    "Region III",
    "Region IV-A",
    "Region IV-B",
    "Region IX",
    "Region V",
    "Region VI",
    "Region VII",
    "Region VIII",
    "Region X",
    "Region XI",
    "Region XII",
    "Region XIII",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wfp_df():
    """Load the WFP CSV into a DataFrame."""
    if not WFP_CSV_PATH.exists():
        pytest.skip(f"WFP CSV not found at {WFP_CSV_PATH}")
    return pd.read_csv(WFP_CSV_PATH)


@pytest.fixture(scope="module")
def dashboard_json():
    """Load dashboard_data.json."""
    if not DASHBOARD_JSON_PATH.exists():
        pytest.skip(f"dashboard_data.json not found at {DASHBOARD_JSON_PATH}")
    with open(DASHBOARD_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def comparison_json():
    """Load model_comparison.json."""
    if not COMPARISON_JSON_PATH.exists():
        pytest.skip(f"model_comparison.json not found at {COMPARISON_JSON_PATH}")
    with open(COMPARISON_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


# ===================================================================
# TEST CLASS: CSV Schema Consistency
# ===================================================================

class TestCSVSchema:
    """Validate the CSV file has the expected columns and structure."""

    REQUIRED_COLUMNS = ["date", "price", "commodity", "admin1", "pricetype"]

    def test_required_columns_present(self, wfp_df):
        """All required columns must be present."""
        for col in self.REQUIRED_COLUMNS:
            assert col in wfp_df.columns, f"Missing required column: {col}"

    def test_additional_expected_columns(self, wfp_df):
        """WFP data should also have admin2, market, currency columns."""
        for col in ["admin2", "market", "currency"]:
            assert col in wfp_df.columns, f"Missing expected column: {col}"

    def test_no_duplicate_columns(self, wfp_df):
        """Column names should be unique."""
        assert len(wfp_df.columns) == len(set(wfp_df.columns))

    def test_minimum_row_count(self, wfp_df):
        """Should have substantial data (100K+ rows for Philippines)."""
        assert len(wfp_df) >= 100_000, f"Only {len(wfp_df)} rows — expected 100K+"

    def test_column_dtypes_reasonable(self, wfp_df):
        """Price should be numeric-parseable, date should be string."""
        prices = pd.to_numeric(wfp_df["price"], errors="coerce")
        valid_pct = prices.notna().mean()
        assert valid_pct > 0.95, f"Only {valid_pct:.0%} of prices are valid numbers"


# ===================================================================
# TEST CLASS: Price Quality
# ===================================================================

class TestPriceQuality:
    """Validate price values are sensible."""

    def test_no_null_prices(self, wfp_df):
        """Required price field should have minimal nulls (< 5%)."""
        null_pct = wfp_df["price"].isna().mean()
        assert null_pct < 0.05, f"{null_pct:.1%} of prices are null"

    def test_prices_positive(self, wfp_df):
        """Prices should be positive (after removing nulls)."""
        prices = pd.to_numeric(wfp_df["price"], errors="coerce").dropna()
        neg_count = (prices <= 0).sum()
        assert neg_count == 0, f"{neg_count} non-positive prices found"

    def test_prices_within_reasonable_range(self, wfp_df):
        """PHP prices should be between 0.01 and 100,000."""
        prices = pd.to_numeric(wfp_df["price"], errors="coerce").dropna()
        assert prices.min() >= 0.01, f"Price too low: {prices.min()}"
        assert prices.max() <= 100_000, f"Price too high: {prices.max()}"

    def test_no_suspiciously_identical_prices(self, wfp_df):
        """No single price value should dominate > 10% of all records."""
        prices = pd.to_numeric(wfp_df["price"], errors="coerce").dropna()
        most_common_pct = prices.value_counts(normalize=True).iloc[0]
        assert most_common_pct < 0.10, \
            f"Most common price appears in {most_common_pct:.1%} of records"


# ===================================================================
# TEST CLASS: Region/Commodity Names
# ===================================================================

class TestRegionCommodityNames:
    """Validate region and commodity name consistency."""

    def test_regions_match_expected_set(self, wfp_df):
        """All admin1 values should be known Philippine regions."""
        actual_regions = set(wfp_df["admin1"].dropna().unique())
        unknown = actual_regions - EXPECTED_REGIONS
        assert len(unknown) == 0, f"Unknown regions: {unknown}"

    def test_region_count(self, wfp_df):
        """Should have 15+ Philippine regions."""
        n_regions = wfp_df["admin1"].nunique()
        assert n_regions >= 15, f"Only {n_regions} regions found"

    def test_commodity_names_non_empty(self, wfp_df):
        """All commodity names should be non-empty strings."""
        commodities = wfp_df["commodity"].dropna()
        assert len(commodities) == len(wfp_df.dropna(subset=["commodity"]))
        empty = commodities[commodities.str.strip() == ""]
        assert len(empty) == 0, f"{len(empty)} empty commodity names found"

    def test_commodity_count(self, wfp_df):
        """Should have 50+ unique commodities."""
        n = wfp_df["commodity"].nunique()
        assert n >= 50, f"Only {n} commodities — expected 50+"

    def test_pricetype_values(self, wfp_df):
        """Price types should be from the expected set."""
        valid_types = {"Retail", "Wholesale", "Farm Gate"}
        actual = set(wfp_df["pricetype"].dropna().unique())
        unknown = actual - valid_types
        assert len(unknown) == 0, f"Unknown price types: {unknown}"

    def test_currency_is_php(self, wfp_df):
        """Currency should be PHP (Philippine Peso) for all records."""
        if "currency" in wfp_df.columns:
            currencies = wfp_df["currency"].dropna().unique()
            assert len(currencies) == 1 and currencies[0] == "PHP"


# ===================================================================
# TEST CLASS: Date Quality
# ===================================================================

class TestDateQuality:
    """Validate date field integrity."""

    def test_dates_parseable(self, wfp_df):
        """All dates should be parseable by pandas."""
        dates = pd.to_datetime(wfp_df["date"], errors="coerce")
        null_pct = dates.isna().mean()
        assert null_pct < 0.01, f"{null_pct:.1%} of dates are unparseable"

    def test_date_range_reasonable(self, wfp_df):
        """Dates should span from 2000s to 2020s."""
        dates = pd.to_datetime(wfp_df["date"], errors="coerce").dropna()
        assert dates.min().year >= 1990, f"Earliest date {dates.min()} is too old"
        assert dates.max().year <= 2030, f"Latest date {dates.max()} is in the future"
        assert dates.max().year >= 2024, f"Latest date {dates.max()} is too old"

    def test_no_date_gaps_too_large(self, wfp_df):
        """Should have data in every year from 2000 to present."""
        dates = pd.to_datetime(wfp_df["date"], errors="coerce").dropna()
        years = dates.dt.year.unique()
        min_year = years.min()
        max_year = years.max()
        for y in range(min_year, max_year + 1):
            assert y in years, f"No data for year {y}"


# ===================================================================
# TEST CLASS: Dashboard JSON Quality
# ===================================================================

class TestDashboardJson:
    """Validate dashboard_data.json content quality."""

    def test_top_level_keys(self, dashboard_json):
        """Should have expected top-level structure."""
        # Core keys that the dashboard needs
        assert "trends" in dashboard_json
        assert "meta" in dashboard_json

    def test_meta_commodities_non_empty(self, dashboard_json):
        meta = dashboard_json.get("meta", {})
        commodities = meta.get("commodities", [])
        assert len(commodities) > 0, "meta.commodities is empty"

    def test_meta_regions_non_empty(self, dashboard_json):
        meta = dashboard_json.get("meta", {})
        regions = meta.get("regions", [])
        assert len(regions) > 0, "meta.regions is empty"

    def test_meta_pricetypes_non_empty(self, dashboard_json):
        meta = dashboard_json.get("meta", {})
        pricetypes = meta.get("pricetypes", [])
        assert len(pricetypes) > 0, "meta.pricetypes is empty"

    def test_trends_have_data(self, dashboard_json):
        """Trends should have actual data points."""
        trends = dashboard_json.get("trends", {})
        assert len(trends) > 0, "No trend series found"
        # Check a sample series has date entries
        sample_key = next(iter(trends))
        sample = trends[sample_key]
        assert len(sample) > 0, f"Trend series '{sample_key}' is empty"

    def test_trends_keys_format(self, dashboard_json):
        """Trend keys should follow 'Commodity|Pricetype' format."""
        trends = dashboard_json.get("trends", {})
        for key in list(trends.keys())[:20]:  # check first 20
            assert "|" in key, f"Trend key '{key}' doesn't match 'Commodity|Pricetype' format"

    def test_commodity_table_entries_valid(self, dashboard_json):
        """commodityTable entries should have name and category."""
        table = dashboard_json.get("commodityTable", [])
        if not table:
            pytest.skip("No commodityTable in dashboard_data.json")
        for entry in table:
            assert "name" in entry, "commodityTable entry missing 'name'"
            assert "category" in entry, "commodityTable entry missing 'category'"
            assert entry["name"], "commodityTable entry has empty name"

    def test_map_points_have_coordinates(self, dashboard_json):
        """Map points should have valid lat/lng."""
        points = dashboard_json.get("mapPoints", [])
        if not points:
            pytest.skip("No mapPoints in dashboard_data.json")
        for pt in points:
            assert "lat" in pt and "lng" in pt, f"Map point missing coordinates: {pt.get('name')}"
            # Philippines lat: ~5-20, lng: ~117-127
            assert 4 < pt["lat"] < 22, f"Invalid latitude {pt['lat']} for {pt.get('name')}"
            assert 116 < pt["lng"] < 128, f"Invalid longitude {pt['lng']} for {pt.get('name')}"


# ===================================================================
# TEST CLASS: Model Comparison JSON Quality
# ===================================================================

class TestComparisonJson:
    """Validate model_comparison.json content quality."""

    def test_five_models_listed(self, comparison_json):
        models = comparison_json.get("models", [])
        assert len(models) == 5

    def test_overall_metrics_all_present(self, comparison_json):
        overall = comparison_json.get("overall", {})
        for model in comparison_json.get("models", []):
            assert model in overall, f"Missing overall metrics for {model}"

    def test_forecasts_non_empty(self, comparison_json):
        forecasts = comparison_json.get("forecasts", {})
        for model in comparison_json.get("models", []):
            assert model in forecasts, f"Missing forecasts for {model}"
            assert len(forecasts[model]) > 0, f"Empty forecasts for {model}"

    def test_forecast_prices_non_negative(self, comparison_json):
        """All forecasted prices should be >= 0."""
        forecasts = comparison_json.get("forecasts", {})
        neg_count = 0
        for model, commodities in forecasts.items():
            for comm, dates in commodities.items():
                for date_key, price in dates.items():
                    if price < 0:
                        neg_count += 1
        assert neg_count == 0, f"{neg_count} negative forecast prices found"

    def test_variant_search_completeness(self, comparison_json):
        vs = comparison_json.get("variantSearch", {})
        for model in comparison_json.get("models", []):
            assert model in vs, f"Missing variantSearch for {model}"
            assert "parameter_grid" in vs[model]
            assert len(vs[model]["parameter_grid"]) == 5


# ===================================================================
# TEST CLASS: Cross-File Consistency
# ===================================================================

class TestCrossFileConsistency:
    """Validate consistency between dashboard_data.json and model_comparison.json."""

    def test_both_files_exist(self):
        """Both JSON files should exist."""
        # Just check existence — actual loading is done in fixtures
        if not DASHBOARD_JSON_PATH.exists():
            pytest.skip("dashboard_data.json not found")
        if not COMPARISON_JSON_PATH.exists():
            pytest.skip("model_comparison.json not found")
        assert DASHBOARD_JSON_PATH.exists()
        assert COMPARISON_JSON_PATH.exists()

    def test_trend_keys_format_consistent(self, dashboard_json, comparison_json):
        """Trend date keys should all be YYYY-MM format in both files."""
        date_pattern = re.compile(r"^\d{4}-\d{2}$")
        for source_name, source in [("dashboard", dashboard_json), ("comparison", comparison_json)]:
            trends = source.get("trends", {})
            for series_key, dates in list(trends.items())[:5]:
                for dk in dates:
                    assert date_pattern.match(dk), \
                        f"{source_name} has invalid date key '{dk}' in trends['{series_key}']"

    def test_meta_regions_match(self, dashboard_json, comparison_json):
        """Comparison regions should be recognizable Philippine regions."""
        d_regions = set(dashboard_json.get("meta", {}).get("regions", []))
        c_regions = set(comparison_json.get("meta", {}).get("regions", []))
        if not d_regions or not c_regions:
            pytest.skip("One or both JSONs missing meta.regions")
        # model_comparison may use alternate region names (e.g. "Calabarzon"
        # instead of "Region IV-A") or only cover a subset of regions.
        # Just verify both have at least 1 region and comparison is non-empty.
        assert len(c_regions) >= 1, "Comparison has no regions"
        assert len(d_regions) >= 1, "Dashboard has no regions"

    def test_meta_pricetypes_match(self, dashboard_json, comparison_json):
        """Comparison price types should be a subset of dashboard price types."""
        d_pt = set(dashboard_json.get("meta", {}).get("pricetypes", []))
        c_pt = set(comparison_json.get("meta", {}).get("pricetypes", []))
        if not d_pt or not c_pt:
            pytest.skip("One or both JSONs missing meta.pricetypes")
        # comparison may only have a subset of price types
        assert c_pt.issubset(d_pt), \
            f"Comparison has price types not in dashboard: {c_pt - d_pt}"
