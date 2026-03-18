"""
Climate scenario analysis for the Philippine Food Price Dashboard.

Computes the historical relationship between ENSO (El Nino / La Nina)
states and Philippine food prices. Generates scenarios like:
  "If El Nino +1.5C, Region III rice prices +8-12% in Q3"

Uses:
  - WFP food price data (wfp_food_prices_phl_latest.csv)
  - ENSO ONI data from exogenous_features.py
  - Optional: exogenous_data.json if already fetched

Usage:
  python climate_scenarios.py --analyze                 # Full analysis
  python climate_scenarios.py --scenario "strong_el_nino" # Specific scenario
  python climate_scenarios.py --all-scenarios            # All ENSO states
  python climate_scenarios.py --summary                  # Quick table
"""
# signed: beta

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
WFP_DATA = BASE_DIR.parent / "WFP" / "wfp_food_prices_phl_latest.csv"
EXO_DATA = BASE_DIR / "exogenous_data.json"
OUTPUT_PATH = BASE_DIR / "climate_scenarios.json"

# ─── ENSO state thresholds (ONI-based) ──────────────────────
# signed: beta
ENSO_STATES = {
    "strong_el_nino":   {"label": "Strong El Niño",       "oni_min": 1.5,  "oni_max": 99.0},
    "moderate_el_nino": {"label": "Moderate El Niño",     "oni_min": 1.0,  "oni_max": 1.5},
    "weak_el_nino":     {"label": "Weak El Niño",         "oni_min": 0.5,  "oni_max": 1.0},
    "neutral":          {"label": "ENSO Neutral",         "oni_min": -0.5, "oni_max": 0.5},
    "weak_la_nina":     {"label": "Weak La Niña",         "oni_min": -1.0, "oni_max": -0.5},
    "moderate_la_nina": {"label": "Moderate La Niña",     "oni_min": -1.5, "oni_max": -1.0},
    "strong_la_nina":   {"label": "Strong La Niña",       "oni_min": -99.0,"oni_max": -1.5},
}

# Lag months: ENSO impact on PH food prices is typically delayed 3-6 months
IMPACT_LAG_MONTHS = [0, 3, 6]


def load_price_data() -> pd.DataFrame:
    """Load WFP Philippine food price data."""
    # signed: beta
    if not WFP_DATA.exists():
        print(f"  ERROR: WFP data not found at {WFP_DATA}")
        print(f"  Run daily_update.py first to download the data.")
        sys.exit(1)

    df = pd.read_csv(WFP_DATA, low_memory=False)
    # Standardize column names (WFP CSV has varying formats)
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("date", "mp_year"):
            col_map[col] = "date"
        elif cl in ("commodity", "cm_name"):
            col_map[col] = "commodity"
        elif cl in ("price", "mp_price"):
            col_map[col] = "price"
        elif cl in ("admin1", "adm1_name"):
            col_map[col] = "region"
        elif cl in ("pricetype", "pt_name"):
            col_map[col] = "pricetype"

    df = df.rename(columns=col_map)

    if "date" not in df.columns or "price" not in df.columns:
        print("  ERROR: Cannot find date/price columns in WFP data")
        sys.exit(1)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    df["year_month"] = df["date"].dt.strftime("%Y-%m")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    if "region" not in df.columns:
        df["region"] = "National"
    if "commodity" not in df.columns:
        df["commodity"] = "All"

    print(f"  Loaded {len(df):,} price records ({df['year_month'].min()} to {df['year_month'].max()})")
    return df


def load_oni_data() -> pd.DataFrame:
    """Load ONI data from exogenous_data.json or fetch fresh."""
    # signed: beta
    if EXO_DATA.exists():
        with open(EXO_DATA) as f:
            data = json.load(f)
        oni_df = pd.DataFrame(data["data"])[["year_month", "oni_value", "enso_state"]]
        oni_df = oni_df.dropna(subset=["oni_value"])
        print(f"  ONI: loaded {len(oni_df)} months from cached exogenous data")
        return oni_df

    # Fallback: fetch directly
    print("  No cached exogenous data. Fetching ONI directly...")
    try:
        from exogenous_features import fetch_oni
        return fetch_oni(2000)
    except ImportError:
        print("  ERROR: Cannot import exogenous_features.py and no cached data")
        sys.exit(1)


def classify_oni(oni_value: float) -> str:
    """Classify an ONI value into an ENSO state key."""
    for key, cfg in ENSO_STATES.items():
        if cfg["oni_min"] <= oni_value < cfg["oni_max"]:
            return key
        # Handle negative ranges (La Nina) where min > max in absolute terms
        if cfg["oni_max"] < cfg["oni_min"]:
            if cfg["oni_max"] <= oni_value < cfg["oni_min"]:
                return key
    return "neutral"


def compute_price_changes(
    price_df: pd.DataFrame,
    oni_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute monthly price changes aligned with ENSO state.

    For each commodity-region pair, calculates month-over-month
    and year-over-year price change percentages, then aligns with
    ONI values (including lagged ONI for delayed impact).
    """
    # signed: beta
    # Monthly average prices per commodity-region
    monthly = (
        price_df
        .groupby(["commodity", "region", "year_month"])["price"]
        .mean()
        .reset_index()
    )
    monthly = monthly.sort_values(["commodity", "region", "year_month"])

    # YoY price change per commodity-region
    monthly["price_yoy_pct"] = (
        monthly
        .groupby(["commodity", "region"])["price"]
        .pct_change(periods=12)
        .mul(100)
    )

    # MoM price change
    monthly["price_mom_pct"] = (
        monthly
        .groupby(["commodity", "region"])["price"]
        .pct_change(periods=1)
        .mul(100)
    )

    # Merge with ONI data
    merged = monthly.merge(oni_df[["year_month", "oni_value"]], on="year_month", how="left")

    # Add lagged ONI values (climate impact is delayed)
    oni_indexed = oni_df.set_index("year_month")["oni_value"]
    all_ym = sorted(monthly["year_month"].unique())
    ym_to_idx = {ym: i for i, ym in enumerate(all_ym)}

    for lag in IMPACT_LAG_MONTHS:
        if lag == 0:
            continue
        lag_col = f"oni_lag{lag}"
        lag_map = {}
        for ym in all_ym:
            idx = ym_to_idx[ym] - lag
            if 0 <= idx < len(all_ym):
                lagged_ym = all_ym[idx]
                lag_map[ym] = oni_indexed.get(lagged_ym, np.nan)
            else:
                lag_map[ym] = np.nan
        merged[lag_col] = merged["year_month"].map(lag_map)

    return merged


def analyze_enso_impact(
    price_changes: pd.DataFrame,
    state_key: str,
    lag: int = 3,
) -> dict:
    """Analyze price impact for a specific ENSO state.

    Returns dict with per-commodity, per-region impact statistics.
    """
    # signed: beta
    cfg = ENSO_STATES[state_key]
    oni_col = f"oni_lag{lag}" if lag > 0 else "oni_value"

    if oni_col not in price_changes.columns:
        oni_col = "oni_value"

    # Select months matching this ENSO state
    # oni_min < oni_max for all states (including La Niña), so a single
    # condition works universally.  The previous if/else branch inverted
    # the inequality for negative oni_min, creating impossible masks for
    # all La Niña scenarios (0 observations returned).  # signed: alpha
    mask = (price_changes[oni_col] >= cfg["oni_min"]) & (price_changes[oni_col] < cfg["oni_max"])

    subset = price_changes[mask].dropna(subset=["price_yoy_pct"])

    if len(subset) == 0:
        return {
            "state": cfg["label"],
            "oni_range": f"{cfg['oni_min']} to {cfg['oni_max']}",
            "impact_lag_months": lag,
            "n_observations": 0,
            "commodities": {},
            "summary": f"No data for {cfg['label']} with lag={lag}",
        }

    # Per-commodity analysis
    commodity_impacts = {}
    for commodity, grp in subset.groupby("commodity"):
        yoy = grp["price_yoy_pct"].dropna()
        if len(yoy) < 3:
            continue

        # Per-region breakdown
        region_impacts = {}
        for region, rgrp in grp.groupby("region"):
            ryoy = rgrp["price_yoy_pct"].dropna()
            if len(ryoy) < 2:
                continue
            region_impacts[region] = {
                "mean_yoy_pct": round(float(ryoy.mean()), 1),
                "median_yoy_pct": round(float(ryoy.median()), 1),
                "std_yoy_pct": round(float(ryoy.std()), 1),
                "n_months": int(len(ryoy)),
            }

        commodity_impacts[commodity] = {
            "mean_yoy_pct": round(float(yoy.mean()), 1),
            "median_yoy_pct": round(float(yoy.median()), 1),
            "std_yoy_pct": round(float(yoy.std()), 1),
            "min_yoy_pct": round(float(yoy.min()), 1),
            "max_yoy_pct": round(float(yoy.max()), 1),
            "n_months": int(len(yoy)),
            "regions": region_impacts,
        }

    # Generate narrative scenarios
    scenarios = []
    for commodity, data in sorted(commodity_impacts.items(), key=lambda x: abs(x[1]["mean_yoy_pct"]), reverse=True):
        mean_pct = data["mean_yoy_pct"]
        std_pct = data["std_yoy_pct"]
        direction = "increase" if mean_pct > 0 else "decrease"

        lo = round(abs(mean_pct) - std_pct, 1) if std_pct else round(abs(mean_pct) * 0.8, 1)
        hi = round(abs(mean_pct) + std_pct, 1) if std_pct else round(abs(mean_pct) * 1.2, 1)
        lo = max(lo, 0.1)

        # Top affected region
        top_region = None
        if data["regions"]:
            top_region = max(data["regions"].items(), key=lambda x: abs(x[1]["mean_yoy_pct"]))

        scenario_text = f"If {cfg['label']}, {commodity} prices may {direction} {lo}-{hi}%"
        if top_region:
            scenario_text += f" (strongest in {top_region[0]}: {top_region[1]['mean_yoy_pct']:+.1f}%)"

        scenarios.append(scenario_text)

    return {
        "state": cfg["label"],
        "state_key": state_key,
        "oni_range": f"{cfg['oni_min']} to {cfg['oni_max']}",
        "impact_lag_months": lag,
        "n_observations": int(len(subset)),
        "n_commodities_affected": len(commodity_impacts),
        "commodities": commodity_impacts,
        "scenarios": scenarios[:10],  # Top 10 most impactful
    }


def compute_neutral_baseline(price_changes: pd.DataFrame) -> dict:
    """Compute price change statistics during ENSO neutral periods."""
    # signed: beta
    neutral = ENSO_STATES["neutral"]
    mask = (price_changes["oni_value"] >= neutral["oni_min"]) & (price_changes["oni_value"] < neutral["oni_max"])
    subset = price_changes[mask].dropna(subset=["price_yoy_pct"])

    baseline = {}
    for commodity, grp in subset.groupby("commodity"):
        yoy = grp["price_yoy_pct"].dropna()
        if len(yoy) < 3:
            continue
        baseline[commodity] = {
            "mean_yoy_pct": round(float(yoy.mean()), 1),
            "median_yoy_pct": round(float(yoy.median()), 1),
            "n_months": int(len(yoy)),
        }
    return baseline


def generate_all_scenarios(
    price_df: pd.DataFrame,
    oni_df: pd.DataFrame,
) -> dict:
    """Generate scenarios for all ENSO states with all lag values."""
    # signed: beta
    print("\n[1/3] Computing price changes...")
    price_changes = compute_price_changes(price_df, oni_df)

    print("[2/3] Computing neutral baseline...")
    baseline = compute_neutral_baseline(price_changes)

    print("[3/3] Analyzing ENSO state impacts...")
    all_scenarios = {}
    for state_key in ENSO_STATES:
        if state_key == "neutral":
            continue
        best_lag_result = None
        for lag in IMPACT_LAG_MONTHS:
            result = analyze_enso_impact(price_changes, state_key, lag)
            if result["n_observations"] > 0:
                if best_lag_result is None or result["n_observations"] > best_lag_result["n_observations"]:
                    best_lag_result = result
        if best_lag_result:
            all_scenarios[state_key] = best_lag_result
            n_scen = len(best_lag_result.get("scenarios", []))
            print(f"  {best_lag_result['state']}: {best_lag_result['n_observations']} obs, "
                  f"{best_lag_result['n_commodities_affected']} commodities, {n_scen} scenarios")
        else:
            print(f"  {ENSO_STATES[state_key]['label']}: insufficient data")

    return {
        "description": "ENSO climate impact scenarios for Philippine food prices",
        "methodology": (
            "Historical correlation analysis between ENSO ONI index and WFP food "
            "price year-over-year changes. Impact lags of 0, 3, and 6 months tested; "
            "best lag selected per ENSO state. Scenarios show expected price change "
            "range (mean ± 1 std dev) during each ENSO phase."
        ),
        "data_sources": {
            "prices": "WFP Food Prices for Philippines",
            "climate": "NOAA CPC Oceanic Nino Index (ONI v5)",
        },
        "neutral_baseline": baseline,
        "scenarios": all_scenarios,
        "generated_at": datetime.now().isoformat(),
    }


def print_summary(results: dict) -> None:
    """Print a human-readable summary table."""
    # signed: beta
    print("\n" + "=" * 75)
    print("  ENSO Impact on Philippine Food Prices — Scenario Summary")
    print("=" * 75)

    # Baseline
    baseline = results.get("neutral_baseline", {})
    if baseline:
        print("\n  Neutral Baseline (annual price change during ENSO-neutral months):")
        for commodity, stats in sorted(baseline.items()):
            print(f"    {commodity:<30s}  {stats['mean_yoy_pct']:+5.1f}%  (n={stats['n_months']})")

    # Scenarios
    for state_key, data in results.get("scenarios", {}).items():
        print(f"\n  ── {data['state']} (ONI {data['oni_range']}, lag={data['impact_lag_months']}mo) ──")
        print(f"     Observations: {data['n_observations']}, Commodities: {data['n_commodities_affected']}")

        if data.get("scenarios"):
            for i, s in enumerate(data["scenarios"][:5], 1):
                print(f"     {i}. {s}")
        elif data.get("commodities"):
            for commodity, stats in sorted(data["commodities"].items(), key=lambda x: abs(x[1]["mean_yoy_pct"]), reverse=True)[:5]:
                baseline_pct = baseline.get(commodity, {}).get("mean_yoy_pct", 0)
                delta = stats["mean_yoy_pct"] - baseline_pct
                print(f"     {commodity:<30s}  {stats['mean_yoy_pct']:+5.1f}%  (Δ{delta:+.1f}% vs neutral)")

    print("\n" + "=" * 75)


def save_scenarios(results: dict) -> None:
    """Save scenarios to JSON."""
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUTPUT_PATH}")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="ENSO climate impact scenario analysis")
    parser.add_argument("--analyze", action="store_true", help="Full analysis (all states)")
    parser.add_argument("--scenario", metavar="STATE", help="Analyze specific ENSO state")
    parser.add_argument("--all-scenarios", action="store_true", help="Generate all scenarios")
    parser.add_argument("--summary", action="store_true", help="Print summary table")
    parser.add_argument("--save", action="store_true", help="Save to climate_scenarios.json")
    args = parser.parse_args()

    if not any([args.analyze, args.scenario, args.all_scenarios, args.summary]):
        parser.print_help()
        return

    print("=" * 65)
    print("  Climate Scenario Analysis — ENSO Impact on Philippine Prices")
    print("=" * 65)

    print("\nLoading data...")
    price_df = load_price_data()
    oni_df = load_oni_data()

    if args.scenario:
        if args.scenario not in ENSO_STATES:
            print(f"  ERROR: Unknown state '{args.scenario}'")
            print(f"  Valid: {', '.join(ENSO_STATES.keys())}")
            sys.exit(1)
        price_changes = compute_price_changes(price_df, oni_df)
        result = analyze_enso_impact(price_changes, args.scenario, lag=3)
        print(f"\n  {result['state']}: {result['n_observations']} observations")
        for s in result.get("scenarios", []):
            print(f"    → {s}")
        if args.save:
            save_scenarios({"scenarios": {args.scenario: result}, "generated_at": datetime.now().isoformat()})
        return

    results = generate_all_scenarios(price_df, oni_df)

    if args.summary or args.analyze or args.all_scenarios:
        print_summary(results)

    if args.save or args.analyze:
        save_scenarios(results)

    print("\nDone.")


if __name__ == "__main__":
    main()
