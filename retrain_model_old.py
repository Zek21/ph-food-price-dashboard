"""
Retrain price forecasting model using latest WFP data (2000-Jan 2026).
- Train: 2000-2023 (same period as old model)
- Validate: 2024-Jan 2026 (compare with old model)
- Forecast: Feb 2026-Dec 2027

Uses per-commodity GradientBoosting with proper time-series features:
  - Trend (year), seasonality (month sin/cos), lag features, rolling averages
  - Region encoding, price type encoding
"""

import csv
import json
import math
import warnings
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

print("=" * 65)
print("  Improved Food Price Model — Train & Compare")
print("=" * 65)

# ─── 1. Load latest WFP data ────────────────────────────────
print("\n[1/6] Loading latest WFP data...")
df = pd.read_csv("D:/ML/WFP/wfp_food_prices_phl_latest.csv")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]

# Standardize region names
df["region"] = df["admin1"]

print(f"   Total records: {len(df):,}")
print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"   Commodities: {df['commodity'].nunique()}")
print(f"   Regions: {df['region'].nunique()}")

# ─── 2. Feature Engineering ─────────────────────────────────
print("\n[2/6] Engineering features...")

def build_features(data):
    """Build time-series aware features for each group."""
    data = data.sort_values("date").copy()
    
    # Time features
    data["year_num"] = data["year"] - 2000  # years since 2000
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    
    # Lag features (past prices for same commodity+region+pricetype)
    data["price_lag1"] = data["price"].shift(1)
    data["price_lag3"] = data["price"].shift(3)
    data["price_lag6"] = data["price"].shift(6)
    data["price_lag12"] = data["price"].shift(12)
    
    # Rolling averages
    data["price_ma3"] = data["price"].rolling(3, min_periods=1).mean()
    data["price_ma6"] = data["price"].rolling(6, min_periods=1).mean()
    data["price_ma12"] = data["price"].rolling(12, min_periods=1).mean()
    
    # Price momentum
    data["price_diff1"] = data["price"].diff(1)
    data["price_diff12"] = data["price"].diff(12)
    
    return data

# Group by commodity + region + pricetype for per-series processing
group_cols = ["commodity", "region", "pricetype"]
pieces = []
for name, group in df.groupby(group_cols, group_keys=False):
    pieces.append(build_features(group))
df_feat = pd.concat(pieces, ignore_index=True)

feature_cols = [
    "year_num", "month_sin", "month_cos",
    "price_lag1", "price_lag3", "price_lag6", "price_lag12",
    "price_ma3", "price_ma6", "price_ma12",
    "price_diff1", "price_diff12",
]

# Encode region
le_region = LabelEncoder()
df_feat["region_enc"] = le_region.fit_transform(df_feat["region"])
feature_cols.append("region_enc")

# Encode pricetype
le_pt = LabelEncoder()
df_feat["pt_enc"] = le_pt.fit_transform(df_feat["pricetype"])
feature_cols.append("pt_enc")

# Drop rows with NaN features (first rows without lags)
df_feat = df_feat.dropna(subset=feature_cols)

print(f"   Feature columns: {len(feature_cols)}")
print(f"   Usable records after feature engineering: {len(df_feat):,}")

# ─── 3. Train/Validate Split ────────────────────────────────
print("\n[3/6] Training per-commodity models...")

train_mask = df_feat["year"] <= 2023
val_mask = df_feat["year"] >= 2024

train_df = df_feat[train_mask]
val_df = df_feat[val_mask]

print(f"   Train set: {len(train_df):,} records (up to 2023)")
print(f"   Validation set: {len(val_df):,} records (2024-Jan 2026)")

# ─── 4. Train per-commodity models ──────────────────────────
models = {}
commodity_results = {}

all_commodities = df_feat["commodity"].unique()
print(f"   Training models for {len(all_commodities)} commodities...")

for i, comm in enumerate(all_commodities):
    comm_train = train_df[train_df["commodity"] == comm]
    comm_val = val_df[val_df["commodity"] == comm]
    
    if len(comm_train) < 20:
        continue
    
    X_train = comm_train[feature_cols].values
    y_train = comm_train["price"].values
    
    # GradientBoosting for each commodity — handles non-linear trends
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    models[comm] = model
    
    # Validate
    if len(comm_val) > 0:
        X_val = comm_val[feature_cols].values
        y_val = comm_val["price"].values
        y_pred = model.predict(X_val)
        y_pred = np.maximum(y_pred, 0)  # prices can't be negative
        
        mape = mean_absolute_percentage_error(y_val, y_pred) * 100
        mae = mean_absolute_error(y_val, y_pred)
        bias = np.mean((y_pred - y_val) / y_val) * 100
        
        commodity_results[comm] = {
            "mape": round(mape, 1),
            "mae": round(mae, 2),
            "bias": round(bias, 1),
            "n_train": len(comm_train),
            "n_val": len(comm_val),
            "val_pred": y_pred.tolist(),
            "val_actual": y_val.tolist(),
            "val_dates": comm_val["date"].dt.strftime("%Y-%m").tolist(),
            "val_regions": comm_val["region"].tolist(),
            "val_pricetypes": comm_val["pricetype"].tolist(),
        }
    
    if (i + 1) % 20 == 0:
        print(f"   ...trained {i+1}/{len(all_commodities)} models")

print(f"   Trained {len(models)} models")

# ─── 5. Overall Evaluation ──────────────────────────────────
print("\n[4/6] Evaluating new model vs old model...")

# New model: aggregate validation results
all_actual = []
all_pred_new = []
for comm, res in commodity_results.items():
    all_actual.extend(res["val_actual"])
    all_pred_new.extend(res["val_pred"])

all_actual = np.array(all_actual)
all_pred_new = np.array(all_pred_new)

new_mape = mean_absolute_percentage_error(all_actual, all_pred_new) * 100
new_mae = mean_absolute_error(all_actual, all_pred_new)
new_bias = np.mean((all_pred_new - all_actual) / all_actual) * 100

print(f"\n   {'Metric':<30s} {'Old RF Model':>15s} {'New GB Model':>15s}")
print(f"   {'─'*60}")
print(f"   {'MAPE (%):':<30s} {'72.3':>15s} {new_mape:>15.1f}")
print(f"   {'MAE (PHP):':<30s} {'82.84':>15s} {new_mae:>15.2f}")
print(f"   {'Bias (%):':<30s} {'+28.5':>15s} {new_bias:>+15.1f}")

# Per-commodity comparison
print(f"\n   Top 10 Most Improved Commodities (by MAPE):")
old_mapes = {
    "Eggs": 597.4, "Eggs (duck)": 501.2, "Maize (yellow)": 183.7,
    "Semolina (yellow)": 175.1, "Coconut": 173.3, "Semolina (white)": 130.1,
    "Bananas (saba)": 119.5, "Rice (regular, milled)": 113.2,
    "Water spinach": 106.9, "Papaya": 105.9,
    "Beans (mung)": 9.5, "Mangoes (carabao)": 12.9, "Garlic": 13.2,
    "Groundnuts (unshelled)": 13.8, "Groundnuts (shelled)": 15.1,
    "Carrots": 15.7, "Beans (green, fresh)": 15.7, "Bitter melon": 17.1,
    "Potatoes (Irish)": 17.5, "Bananas (lakatan)": 19.5,
}
improvements = []
for comm, new_res in commodity_results.items():
    if comm in old_mapes:
        old_m = old_mapes[comm]
        new_m = new_res["mape"]
        improvements.append((comm, old_m, new_m, old_m - new_m))

improvements.sort(key=lambda x: x[3], reverse=True)
for comm, old_m, new_m, imp in improvements[:10]:
    print(f"   {comm:35s}: {old_m:>7.1f}% → {new_m:>7.1f}% (↓{imp:.1f}pp)")

# ─── 6. Generate Forecasts 2026-2027 ────────────────────────
print("\n[5/6] Generating forecasts for Feb 2026 — Dec 2027...")

forecast_rows = []
# For each commodity+region+pricetype with a model, forecast forward
for comm in models:
    comm_data = df_feat[df_feat["commodity"] == comm].copy()
    pricetypes = comm_data["pricetype"].unique()
    regions = comm_data["region"].unique()
    
    for pt in pricetypes:
        for region in regions:
            series = comm_data[(comm_data["pricetype"] == pt) & (comm_data["region"] == region)]
            if len(series) < 12:
                continue
            
            series = series.sort_values("date")
            last_prices = series["price"].values[-12:]  # last 12 months
            
            region_enc = le_region.transform([region])[0] if region in le_region.classes_ else 0
            pt_enc = le_pt.transform([pt])[0] if pt in le_pt.classes_ else 0
            
            # Rolling forecast
            price_history = list(series["price"].values)
            
            for year in (2026, 2027):
                for month in range(1, 13):
                    if year == 2026 and month <= 1:
                        continue  # skip Jan 2026 (already have actual)
                    if year == 2027 and month > 12:
                        continue
                    
                    year_num = year - 2000
                    month_sin = math.sin(2 * math.pi * month / 12)
                    month_cos = math.cos(2 * math.pi * month / 12)
                    
                    n = len(price_history)
                    lag1 = price_history[-1] if n >= 1 else 0
                    lag3 = price_history[-3] if n >= 3 else lag1
                    lag6 = price_history[-6] if n >= 6 else lag1
                    lag12 = price_history[-12] if n >= 12 else lag1
                    ma3 = np.mean(price_history[-3:]) if n >= 3 else lag1
                    ma6 = np.mean(price_history[-6:]) if n >= 6 else lag1
                    ma12 = np.mean(price_history[-12:]) if n >= 12 else lag1
                    diff1 = price_history[-1] - price_history[-2] if n >= 2 else 0
                    diff12 = price_history[-1] - price_history[-13] if n >= 13 else 0
                    
                    features = np.array([[
                        year_num, month_sin, month_cos,
                        lag1, lag3, lag6, lag12,
                        ma3, ma6, ma12,
                        diff1, diff12,
                        region_enc, pt_enc,
                    ]])
                    
                    pred = max(0, models[comm].predict(features)[0])
                    price_history.append(pred)
                    
                    forecast_rows.append({
                        "year": year, "month": month,
                        "region": region, "commodity": comm,
                        "pricetype": pt, "predicted_price": round(pred, 2),
                        "source": "new_model",
                    })

print(f"   Generated {len(forecast_rows):,} forecast rows")

# ─── 7. Load old predictions for comparison ─────────────────
print("\n[6/6] Loading old predictions for comparison data...")

old_pred_agg = defaultdict(lambda: {"sum": 0, "count": 0})
with open("D:/ML/merged_file_with_location_2.csv", "r") as f:
    reader = csv.DictReader(f)
    rc = 0
    for r in reader:
        rc += 1
        try:
            price = float(r["Predicted_Price"])
        except ValueError:
            continue
        key = (int(r["Year"]), int(r["Month"]), r["Region"], r["Commodity"], r["Pricetype"])
        old_pred_agg[key]["sum"] += price
        old_pred_agg[key]["count"] += 1
        if rc % 5_000_000 == 0:
            print(f"   ...processed {rc/1e6:.0f}M rows")

# ─── Build comparison dataset (actual vs old vs new) ────────
print("\n   Building comparison dataset...")

# Actual 2024+ data
actual_lookup = {}
with open("D:/ML/WFP/wfp_food_prices_phl_latest.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        year = int(r["date"][:4])
        month = int(r["date"][5:7])
        if year >= 2024:
            try:
                price = float(r["price"])
            except ValueError:
                continue
            key = (year, month, r["admin1"], r["commodity"], r["pricetype"])
            if key not in actual_lookup:
                actual_lookup[key] = {"sum": 0, "count": 0}
            actual_lookup[key]["sum"] += price
            actual_lookup[key]["count"] += 1

# Build new prediction lookup
new_pred_lookup = {}
for row in forecast_rows:
    key = (row["year"], row["month"], row["region"], row["commodity"], row["pricetype"])
    new_pred_lookup[key] = row["predicted_price"]

# New model validation predictions (already stored in commodity_results)
new_val_lookup = {}
for comm, res in commodity_results.items():
    for i in range(len(res["val_actual"])):
        date_str = res["val_dates"][i]
        yr, mo = int(date_str[:4]), int(date_str[5:7])
        key = (yr, mo, res["val_regions"][i], comm, res["val_pricetypes"][i])
        if key not in new_val_lookup:
            new_val_lookup[key] = {"sum": 0, "count": 0}
        new_val_lookup[key]["sum"] += res["val_pred"][i]
        new_val_lookup[key]["count"] += 1

# Build comparison for all matched keys
comparison = []
for key in actual_lookup:
    act_avg = actual_lookup[key]["sum"] / actual_lookup[key]["count"]
    
    old_avg = None
    if key in old_pred_agg:
        old_avg = old_pred_agg[key]["sum"] / old_pred_agg[key]["count"]
    
    new_avg = None
    if key in new_val_lookup:
        new_avg = new_val_lookup[key]["sum"] / new_val_lookup[key]["count"]
    
    comparison.append({
        "year": key[0], "month": key[1], "region": key[2],
        "commodity": key[3], "pricetype": key[4],
        "actual": round(act_avg, 2),
        "old_pred": round(old_avg, 2) if old_avg else None,
        "new_pred": round(new_avg, 2) if new_avg else None,
    })

# ─── Summary statistics ─────────────────────────────────────
matched_both = [c for c in comparison if c["old_pred"] is not None and c["new_pred"] is not None]
print(f"   Comparison points (both models): {len(matched_both):,}")

if matched_both:
    old_errs = [abs(c["old_pred"] - c["actual"]) / c["actual"] * 100 for c in matched_both if c["actual"] > 0]
    new_errs = [abs(c["new_pred"] - c["actual"]) / c["actual"] * 100 for c in matched_both if c["actual"] > 0]
    
    print(f"\n   On matched subset ({len(matched_both):,} points):")
    print(f"   Old Model MAPE: {sum(old_errs)/len(old_errs):.1f}%")
    print(f"   New Model MAPE: {sum(new_errs)/len(new_errs):.1f}%")

# ─── Build combined trends for dashboard ─────────────────────
# Historical monthly trends (all years)
hist_trends = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for _, r in df.iterrows():
    key = f"{r['commodity']}|{r['pricetype']}"
    date_key = f"{r['year']}-{r['month']:02d}"
    hist_trends[key][date_key]["sum"] += r["price"]
    hist_trends[key][date_key]["count"] += 1

# Old model monthly trends
old_trends = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for (yr, mo, reg, comm, pt), v in old_pred_agg.items():
    key = f"{comm}|{pt}"
    date_key = f"{yr}-{mo:02d}"
    old_trends[key][date_key]["sum"] += v["sum"]
    old_trends[key][date_key]["count"] += v["count"]

# New model monthly trends (validation period + forecast)
new_trends = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for comm, res in commodity_results.items():
    for i in range(len(res["val_actual"])):
        key = f"{comm}|{res['val_pricetypes'][i]}"
        dk = res["val_dates"][i]
        new_trends[key][dk]["sum"] += res["val_pred"][i]
        new_trends[key][dk]["count"] += 1

for row in forecast_rows:
    key = f"{row['commodity']}|{row['pricetype']}"
    dk = f"{row['year']}-{row['month']:02d}"
    new_trends[key][dk]["sum"] += row["predicted_price"]
    new_trends[key][dk]["count"] += 1

# Build combined trends JSON
all_keys = set(hist_trends.keys()) | set(old_trends.keys()) | set(new_trends.keys())
trends_json = {}
for key in all_keys:
    series = {}
    for dk, v in hist_trends.get(key, {}).items():
        series[dk] = {"actual": round(v["sum"] / v["count"], 2)}
    for dk, v in old_trends.get(key, {}).items():
        if dk not in series:
            series[dk] = {}
        series[dk]["old_pred"] = round(v["sum"] / v["count"], 2)
    for dk, v in new_trends.get(key, {}).items():
        if dk not in series:
            series[dk] = {}
        series[dk]["new_pred"] = round(v["sum"] / v["count"], 2)
    trends_json[key] = series

# Per-commodity MAPE comparison table
comm_comparison = []
for comm, res in commodity_results.items():
    old_m = old_mapes.get(comm)
    comm_comparison.append({
        "commodity": comm,
        "new_mape": res["mape"],
        "old_mape": old_m,
        "n_val": res["n_val"],
        "mae": res["mae"],
    })
comm_comparison.sort(key=lambda x: x["new_mape"])

# Forecast summary
forecast_summary = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for row in forecast_rows:
    key = row["commodity"]
    dk = f"{row['year']}-{row['month']:02d}"
    forecast_summary[key][dk]["sum"] += row["predicted_price"]
    forecast_summary[key][dk]["count"] += 1

forecast_json = {}
for comm, dates in forecast_summary.items():
    forecast_json[comm] = {dk: round(v["sum"]/v["count"], 2) for dk, v in dates.items()}

dashboard_data = {
    "trends": trends_json,
    "commComparison": comm_comparison,
    "forecasts": forecast_json,
    "overall": {
        "old_mape": 72.3,
        "new_mape": round(new_mape, 1),
        "old_mae": 82.84,
        "new_mae": round(new_mae, 2),
        "old_bias": 28.5,
        "new_bias": round(new_bias, 1),
        "n_comparison": len(matched_both),
    },
    "meta": {
        "commodities": sorted(df["commodity"].unique().tolist()),
        "pricetypes": sorted(df["pricetype"].unique().tolist()),
        "regions": sorted(df["region"].unique().tolist()),
        "trainPeriod": "2000-2023",
        "valPeriod": "2024-Jan 2026",
        "forecastPeriod": "Feb 2026-Dec 2027",
    },
}

# Save JSON for dashboard
with open("D:/ML/Website/model_comparison.json", "w") as f:
    json.dump(dashboard_data, f, separators=(",", ":"))

print(f"\n{'='*65}")
print(f"  Model data saved to: D:/ML/Website/model_comparison.json")
print(f"  New Model MAPE: {new_mape:.1f}% (vs Old: 72.3%)")
print(f"{'='*65}")
