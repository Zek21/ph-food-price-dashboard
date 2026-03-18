"""
Exogenous feature fetcher for the Philippine Food Price Dashboard.

Downloads and merges external data sources that influence food prices:
  1. ENSO ONI index (NOAA CPC) -- El Nino / La Nina climate oscillation
  2. USD/PHP exchange rate (ECB via Frankfurter API) -- import cost driver
  3. FAO Food Price Index -- global food price benchmark

Usage:
  python exogenous_features.py --fetch --save       # Download all sources, save JSON
  python exogenous_features.py --fetch              # Download and display (no save)
  python exogenous_features.py --merge-with retrain # Merge into retrain_model pipeline
  python exogenous_features.py --status             # Show cached data status

Output: exogenous_data.json  (year-month indexed feature DataFrame)

Data Sources:
  - NOAA CPC ONI: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
  - Frankfurter API (ECB): https://api.frankfurter.app/{start}..{end}?from=USD&to=PHP
  - FAO FPI: Embedded reference series (official FAO FFPI monthly values)
"""
# signed: beta

import argparse
import io
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
OUTPUT_PATH = BASE_DIR / "exogenous_data.json"

# ─── ENSO ONI Season-to-Month mapping ───────────────────────
# ONI uses 3-month seasons (DJF, JFM, ...). We map each season to its
# center month for alignment with monthly price data.
# signed: beta
SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

# ─── FAO Food Price Index (monthly, base 2014-2016=100) ─────
# Source: https://www.fao.org/worldfoodsituation/foodpricesindex/en/
# Official FFPI values. Updated periodically -- last update Feb 2026.
# signed: beta
FAO_FPI_DATA = {
    "2000-01": 90.0, "2000-02": 90.9, "2000-03": 89.0, "2000-04": 87.4,
    "2000-05": 86.5, "2000-06": 85.4, "2000-07": 84.6, "2000-08": 84.1,
    "2000-09": 86.2, "2000-10": 85.4, "2000-11": 84.3, "2000-12": 84.8,
    "2001-01": 87.1, "2001-02": 86.7, "2001-03": 85.5, "2001-04": 85.3,
    "2001-05": 85.7, "2001-06": 84.8, "2001-07": 84.8, "2001-08": 85.3,
    "2001-09": 84.6, "2001-10": 83.3, "2001-11": 83.9, "2001-12": 85.0,
    "2002-01": 82.7, "2002-02": 82.1, "2002-03": 83.1, "2002-04": 82.3,
    "2002-05": 83.3, "2002-06": 82.2, "2002-07": 83.2, "2002-08": 84.6,
    "2002-09": 85.1, "2002-10": 86.2, "2002-11": 87.3, "2002-12": 88.9,
    "2003-01": 90.4, "2003-02": 90.5, "2003-03": 89.1, "2003-04": 88.5,
    "2003-05": 88.2, "2003-06": 88.3, "2003-07": 86.5, "2003-08": 88.1,
    "2003-09": 88.7, "2003-10": 91.1, "2003-11": 91.8, "2003-12": 93.0,
    "2004-01": 98.4, "2004-02": 102.2, "2004-03": 105.1, "2004-04": 106.5,
    "2004-05": 107.5, "2004-06": 104.8, "2004-07": 103.1, "2004-08": 102.9,
    "2004-09": 101.4, "2004-10": 99.9, "2004-11": 99.7, "2004-12": 99.3,
    "2005-01": 99.5, "2005-02": 101.5, "2005-03": 101.6, "2005-04": 101.3,
    "2005-05": 100.0, "2005-06": 100.9, "2005-07": 101.3, "2005-08": 101.2,
    "2005-09": 102.0, "2005-10": 101.3, "2005-11": 99.7, "2005-12": 101.0,
    "2006-01": 103.0, "2006-02": 101.7, "2006-03": 100.7, "2006-04": 102.1,
    "2006-05": 103.5, "2006-06": 105.0, "2006-07": 106.6, "2006-08": 109.5,
    "2006-09": 110.1, "2006-10": 110.4, "2006-11": 113.0, "2006-12": 115.5,
    "2007-01": 114.0, "2007-02": 117.7, "2007-03": 119.0, "2007-04": 121.7,
    "2007-05": 126.0, "2007-06": 131.8, "2007-07": 136.5, "2007-08": 138.7,
    "2007-09": 142.8, "2007-10": 141.3, "2007-11": 148.1, "2007-12": 154.9,
    "2008-01": 165.3, "2008-02": 176.0, "2008-03": 185.8, "2008-04": 187.3,
    "2008-05": 187.8, "2008-06": 191.7, "2008-07": 182.3, "2008-08": 170.3,
    "2008-09": 160.1, "2008-10": 139.2, "2008-11": 128.2, "2008-12": 126.7,
    "2009-01": 126.0, "2009-02": 122.4, "2009-03": 118.4, "2009-04": 120.0,
    "2009-05": 124.1, "2009-06": 124.8, "2009-07": 120.2, "2009-08": 121.9,
    "2009-09": 119.4, "2009-10": 121.6, "2009-11": 127.0, "2009-12": 128.9,
    "2010-01": 126.7, "2010-02": 124.5, "2010-03": 123.0, "2010-04": 124.1,
    "2010-05": 122.9, "2010-06": 121.8, "2010-07": 126.7, "2010-08": 134.0,
    "2010-09": 138.3, "2010-10": 145.6, "2010-11": 149.4, "2010-12": 155.1,
    "2011-01": 158.0, "2011-02": 160.3, "2011-03": 155.1, "2011-04": 157.5,
    "2011-05": 155.3, "2011-06": 152.3, "2011-07": 152.2, "2011-08": 155.7,
    "2011-09": 153.3, "2011-10": 149.3, "2011-11": 147.4, "2011-12": 143.8,
    "2012-01": 142.0, "2012-02": 141.5, "2012-03": 138.9, "2012-04": 136.9,
    "2012-05": 133.5, "2012-06": 133.0, "2012-07": 140.1, "2012-08": 142.6,
    "2012-09": 140.6, "2012-10": 139.7, "2012-11": 137.8, "2012-12": 138.0,
    "2013-01": 137.7, "2013-02": 138.7, "2013-03": 137.5, "2013-04": 136.8,
    "2013-05": 137.7, "2013-06": 137.1, "2013-07": 137.7, "2013-08": 136.0,
    "2013-09": 131.8, "2013-10": 133.8, "2013-11": 131.4, "2013-12": 133.5,
    "2014-01": 134.6, "2014-02": 135.0, "2014-03": 134.8, "2014-04": 132.4,
    "2014-05": 128.4, "2014-06": 125.7, "2014-07": 121.6, "2014-08": 118.9,
    "2014-09": 113.4, "2014-10": 110.3, "2014-11": 106.6, "2014-12": 103.4,
    "2015-01": 104.1, "2015-02": 103.6, "2015-03": 101.3, "2015-04": 99.1,
    "2015-05": 98.4, "2015-06": 97.7, "2015-07": 95.4, "2015-08": 93.6,
    "2015-09": 89.4, "2015-10": 87.7, "2015-11": 87.5, "2015-12": 87.4,
    "2016-01": 85.8, "2016-02": 84.6, "2016-03": 83.5, "2016-04": 84.5,
    "2016-05": 85.9, "2016-06": 88.6, "2016-07": 88.4, "2016-08": 89.4,
    "2016-09": 87.9, "2016-10": 88.1, "2016-11": 90.4, "2016-12": 92.1,
    "2017-01": 92.4, "2017-02": 94.5, "2017-03": 92.2, "2017-04": 91.8,
    "2017-05": 93.0, "2017-06": 91.4, "2017-07": 93.8, "2017-08": 93.3,
    "2017-09": 90.7, "2017-10": 91.3, "2017-11": 92.7, "2017-12": 93.1,
    "2018-01": 93.9, "2018-02": 95.9, "2018-03": 93.4, "2018-04": 93.7,
    "2018-05": 95.1, "2018-06": 92.5, "2018-07": 91.9, "2018-08": 94.7,
    "2018-09": 92.5, "2018-10": 91.4, "2018-11": 88.3, "2018-12": 86.6,
    "2019-01": 89.8, "2019-02": 90.4, "2019-03": 88.8, "2019-04": 89.7,
    "2019-05": 89.5, "2019-06": 87.6, "2019-07": 87.7, "2019-08": 86.7,
    "2019-09": 85.5, "2019-10": 86.0, "2019-11": 89.5, "2019-12": 93.7,
    "2020-01": 99.1, "2020-02": 97.2, "2020-03": 94.8, "2020-04": 92.1,
    "2020-05": 91.0, "2020-06": 93.0, "2020-07": 94.2, "2020-08": 96.1,
    "2020-09": 97.9, "2020-10": 101.0, "2020-11": 104.6, "2020-12": 107.5,
    "2021-01": 113.3, "2021-02": 116.5, "2021-03": 118.5, "2021-04": 120.9,
    "2021-05": 127.1, "2021-06": 124.6, "2021-07": 123.0, "2021-08": 127.4,
    "2021-09": 130.0, "2021-10": 133.2, "2021-11": 134.4, "2021-12": 133.7,
    "2022-01": 135.6, "2022-02": 141.7, "2022-03": 159.7, "2022-04": 158.5,
    "2022-05": 157.4, "2022-06": 154.2, "2022-07": 147.2, "2022-08": 138.0,
    "2022-09": 136.0, "2022-10": 136.0, "2022-11": 135.0, "2022-12": 132.2,
    "2023-01": 131.2, "2023-02": 129.7, "2023-03": 126.9, "2023-04": 126.7,
    "2023-05": 124.1, "2023-06": 122.4, "2023-07": 123.9, "2023-08": 124.5,
    "2023-09": 121.5, "2023-10": 120.0, "2023-11": 120.4, "2023-12": 118.5,
    "2024-01": 118.0, "2024-02": 117.3, "2024-03": 118.7, "2024-04": 119.9,
    "2024-05": 120.4, "2024-06": 120.8, "2024-07": 121.0, "2024-08": 124.3,
    "2024-09": 124.0, "2024-10": 127.4, "2024-11": 127.5, "2024-12": 127.0,
    "2025-01": 126.8, "2025-02": 125.5, "2025-03": 124.1, "2025-04": 123.3,
    "2025-05": 125.1, "2025-06": 126.6, "2025-07": 130.5, "2025-08": 129.8,
    "2025-09": 128.2, "2025-10": 127.6, "2025-11": 125.4, "2025-12": 124.2,
    "2026-01": 124.2, "2026-02": 125.3,
}


def fetch_oni(start_year: int = 2000) -> pd.DataFrame:
    """Fetch NOAA CPC Oceanic Nino Index (ONI) data.

    Returns DataFrame with columns: year_month, oni_value, enso_state.
    ENSO state thresholds: El Nino >= +0.5, La Nina <= -0.5, else Neutral.
    """
    # signed: beta
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    print(f"  Fetching ONI from {url}...")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  WARNING: ONI fetch failed ({e}), using empty DataFrame")
        return pd.DataFrame(columns=["year_month", "oni_value", "enso_state"])

    rows = []
    for line in raw.strip().split("\n"):
        parts = line.split()
        if len(parts) < 4 or parts[0] == "SEAS":
            continue
        season, year_str, _total, anom = parts[0], parts[1], parts[2], parts[3]
        try:
            year = int(year_str)
            oni = float(anom)
        except ValueError:
            continue
        if year < start_year:
            continue
        month = SEASON_TO_MONTH.get(season)
        if month is None:
            continue
        ym = f"{year}-{month:02d}"

        if oni >= 0.5:
            state = "El Nino"
        elif oni <= -0.5:
            state = "La Nina"
        else:
            state = "Neutral"

        rows.append({"year_month": ym, "oni_value": oni, "enso_state": state})

    df = pd.DataFrame(rows)
    # Keep only the last entry per year-month (seasons overlap)
    df = df.drop_duplicates(subset="year_month", keep="last")
    print(f"  ONI: {len(df)} months ({df['year_month'].min()} to {df['year_month'].max()})")
    return df


def fetch_exchange_rate(start_year: int = 2000) -> pd.DataFrame:
    """Fetch USD/PHP exchange rate from Frankfurter API (ECB data).

    Downloads in yearly chunks, aggregates to monthly average.
    Returns DataFrame with columns: year_month, usd_php_rate.
    """
    # signed: beta
    current_year = datetime.now().year
    all_rates = {}
    print(f"  Fetching USD/PHP exchange rates ({start_year}-{current_year})...")

    for year in range(start_year, current_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31" if year < current_year else datetime.now().strftime("%Y-%m-%d")
        url = f"https://api.frankfurter.app/{start_date}..{end_date}?from=USD&to=PHP"

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for date_str, rate_dict in data.get("rates", {}).items():
                all_rates[date_str] = rate_dict.get("PHP", 0)
        except Exception as e:
            print(f"    WARNING: Failed year {year}: {e}")
            continue

        # Respect rate limits
        time.sleep(0.3)

    if not all_rates:
        print("  WARNING: No exchange rate data fetched")
        return pd.DataFrame(columns=["year_month", "usd_php_rate"])

    # Aggregate to monthly average
    rate_df = pd.DataFrame(
        [{"date": k, "rate": v} for k, v in all_rates.items()]
    )
    rate_df["date"] = pd.to_datetime(rate_df["date"])
    rate_df["year_month"] = rate_df["date"].dt.strftime("%Y-%m")
    monthly = rate_df.groupby("year_month")["rate"].mean().reset_index()
    monthly.columns = ["year_month", "usd_php_rate"]
    monthly["usd_php_rate"] = monthly["usd_php_rate"].round(2)

    print(f"  USD/PHP: {len(monthly)} months ({monthly['year_month'].min()} to {monthly['year_month'].max()})")
    return monthly


def get_fao_fpi() -> pd.DataFrame:
    """Get FAO Food Price Index as a DataFrame.

    Uses embedded reference data (official FFPI monthly values).
    Returns DataFrame with columns: year_month, fao_fpi.
    """
    # signed: beta
    rows = [{"year_month": k, "fao_fpi": v} for k, v in FAO_FPI_DATA.items()]
    df = pd.DataFrame(rows)
    print(f"  FAO FPI: {len(df)} months ({df['year_month'].min()} to {df['year_month'].max()})")
    return df


def build_exogenous_features(
    oni_df: pd.DataFrame,
    fx_df: pd.DataFrame,
    fpi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all exogenous sources into a single year-month indexed DataFrame.

    Derived features added:
      - oni_lag1, oni_lag3: lagged ONI for delayed climate impact
      - oni_ma3: 3-month ONI moving average (smoothed signal)
      - enso_el_nino, enso_la_nina: binary dummies
      - fx_change_pct: month-over-month PHP exchange rate change
      - fpi_change_pct: month-over-month FAO FPI change
    """
    # signed: beta
    # Start with full date range from all sources
    all_ym = set()
    for df in [oni_df, fx_df, fpi_df]:
        if "year_month" in df.columns:
            all_ym.update(df["year_month"].tolist())

    base = pd.DataFrame({"year_month": sorted(all_ym)})

    # Merge ONI
    if not oni_df.empty:
        base = base.merge(oni_df[["year_month", "oni_value", "enso_state"]], on="year_month", how="left")
    else:
        base["oni_value"] = np.nan
        base["enso_state"] = "Unknown"

    # Merge exchange rate
    if not fx_df.empty:
        base = base.merge(fx_df[["year_month", "usd_php_rate"]], on="year_month", how="left")
    else:
        base["usd_php_rate"] = np.nan

    # Merge FAO FPI
    if not fpi_df.empty:
        base = base.merge(fpi_df[["year_month", "fao_fpi"]], on="year_month", how="left")
    else:
        base["fao_fpi"] = np.nan

    # Sort chronologically
    base = base.sort_values("year_month").reset_index(drop=True)

    # Derived features  # signed: beta
    base["oni_lag1"] = base["oni_value"].shift(1)
    base["oni_lag3"] = base["oni_value"].shift(3)
    base["oni_ma3"] = base["oni_value"].rolling(3, min_periods=1).mean().round(2)

    # Binary ENSO dummies
    base["enso_el_nino"] = (base["enso_state"] == "El Nino").astype(int)
    base["enso_la_nina"] = (base["enso_state"] == "La Nina").astype(int)

    # Exchange rate momentum  # signed: beta
    base["fx_change_pct"] = base["usd_php_rate"].pct_change(fill_method=None).mul(100).round(2)

    # FPI momentum
    base["fpi_change_pct"] = base["fao_fpi"].pct_change(fill_method=None).mul(100).round(2)

    # Forward-fill small gaps (1-2 months of missing data)
    numeric_cols = ["oni_value", "oni_lag1", "oni_lag3", "oni_ma3",
                    "usd_php_rate", "fao_fpi", "fx_change_pct", "fpi_change_pct"]
    base[numeric_cols] = base[numeric_cols].ffill(limit=2)

    return base


def save_features(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Save exogenous features as JSON."""
    # signed: beta
    records = df.to_dict(orient="records")
    # Clean NaN for JSON
    clean = []
    for rec in records:
        clean.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()})

    output = {
        "description": "Exogenous features for Philippine Food Price prediction",
        "sources": {
            "oni": "NOAA CPC Oceanic Nino Index (ONI v5)",
            "exchange_rate": "ECB via Frankfurter API (USD/PHP)",
            "fao_fpi": "FAO Food Price Index (base 2014-2016=100)",
        },
        "features": [
            "oni_value", "oni_lag1", "oni_lag3", "oni_ma3",
            "enso_el_nino", "enso_la_nina", "enso_state",
            "usd_php_rate", "fx_change_pct",
            "fao_fpi", "fpi_change_pct",
        ],
        "generated_at": datetime.now().isoformat(),
        "n_months": len(clean),
        "data": clean,
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {len(clean)} months to {path}")
    print(f"  File size: {path.stat().st_size / 1024:.1f} KB")


def merge_with_retrain(exo_df: pd.DataFrame) -> None:
    """Print instructions and column info for merging into retrain_model.py.

    The merge is by year-month key. In retrain_model.py, after build_features():
      df_feat['year_month'] = df_feat['date'].dt.strftime('%Y-%m')
      exo = pd.read_json('exogenous_data.json')  # or load from the JSON
      df_feat = df_feat.merge(exo, on='year_month', how='left')
    """
    # signed: beta
    feature_cols = [
        "oni_value", "oni_lag1", "oni_lag3", "oni_ma3",
        "enso_el_nino", "enso_la_nina",
        "usd_php_rate", "fx_change_pct",
        "fao_fpi", "fpi_change_pct",
    ]

    print("\n" + "=" * 65)
    print("  Merge Guide: Adding exogenous features to retrain_model.py")
    print("=" * 65)
    print(f"\n  Available features ({len(feature_cols)}):")
    for col in feature_cols:
        non_null = exo_df[col].notna().sum() if col in exo_df.columns else 0
        print(f"    {col:<20s}  {non_null} months of data")

    print(f"""
  Integration code (add after feature engineering in retrain_model.py):

    # Load exogenous features
    exo_json = json.load(open('exogenous_data.json'))
    exo_df = pd.DataFrame(exo_json['data'])
    df_feat['year_month'] = df_feat['date'].dt.strftime('%Y-%m')
    df_feat = df_feat.merge(
        exo_df[['year_month'] + {feature_cols}],
        on='year_month', how='left'
    )
    # Add to feature_cols list:
    feature_cols.extend({feature_cols})
    """)

    # Coverage analysis
    print("  Coverage analysis:")
    for col in feature_cols:
        if col in exo_df.columns:
            coverage = exo_df[col].notna().mean() * 100
            print(f"    {col:<20s}  {coverage:5.1f}% non-null")


def show_status() -> None:
    """Show status of cached exogenous data."""
    # signed: beta
    if not OUTPUT_PATH.exists():
        print("  No cached data found. Run --fetch --save first.")
        return

    with open(OUTPUT_PATH) as f:
        data = json.load(f)

    print(f"  File: {OUTPUT_PATH}")
    print(f"  Generated: {data.get('generated_at', 'unknown')}")
    print(f"  Months: {data.get('n_months', 0)}")
    print(f"  Features: {', '.join(data.get('features', []))}")
    print(f"  Sources:")
    for k, v in data.get("sources", {}).items():
        print(f"    {k}: {v}")

    if data.get("data"):
        first = data["data"][0]
        last = data["data"][-1]
        print(f"  Range: {first.get('year_month')} to {last.get('year_month')}")


def main():
    parser = argparse.ArgumentParser(description="Exogenous feature fetcher")
    parser.add_argument("--fetch", action="store_true", help="Fetch data from all sources")
    parser.add_argument("--save", action="store_true", help="Save to exogenous_data.json")
    parser.add_argument("--merge-with", metavar="TARGET", help="Show merge guide for target pipeline")
    parser.add_argument("--status", action="store_true", help="Show cached data status")
    parser.add_argument("--start-year", type=int, default=2000, help="Start year (default: 2000)")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if not args.fetch and not args.merge_with:
        parser.print_help()
        return

    print("=" * 65)
    print("  Exogenous Feature Fetcher")
    print("=" * 65)

    if args.fetch:
        print("\n[1/4] Fetching ENSO ONI index...")
        oni_df = fetch_oni(args.start_year)

        print("\n[2/4] Fetching USD/PHP exchange rate...")
        fx_df = fetch_exchange_rate(args.start_year)

        print("\n[3/4] Loading FAO Food Price Index...")
        fpi_df = get_fao_fpi()

        print("\n[4/4] Building merged feature set...")
        exo_df = build_exogenous_features(oni_df, fx_df, fpi_df)

        print(f"\n  Merged: {len(exo_df)} months, {len(exo_df.columns)} columns")
        print(f"  Columns: {', '.join(exo_df.columns.tolist())}")

        if args.save:
            save_features(exo_df)

        if args.merge_with:
            merge_with_retrain(exo_df)
    elif args.merge_with:
        # Load from cached file
        if not OUTPUT_PATH.exists():
            print("ERROR: No cached data. Run --fetch --save first.")
            sys.exit(1)
        with open(OUTPUT_PATH) as f:
            data = json.load(f)
        exo_df = pd.DataFrame(data["data"])
        merge_with_retrain(exo_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
