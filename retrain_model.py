"""
Multi-model food price forecasting pipeline.

Trains 5 different ML models on WFP Philippines price data, evaluates each
against held-out 2024–Jan 2026 actuals, generates forecasts through Dec 2027,
and outputs a single JSON consumed by the comparison dashboard.

Models:
  1. Gradient Boosting  — ensemble of sequential weak learners, good at trends
  2. Random Forest      — bagged decision trees, robust to outliers
  3. Extra Trees        — extremely randomized trees, fast and low-variance
  4. Ridge Regression   — regularized linear model, captures linear trends
  5. K-Nearest Neighbors — instance-based, captures local price similarity
"""

import json
import math
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

# ─── Model definitions ──────────────────────────────────────
MODEL_DEFS = {
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    ),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    ),
    "Extra Trees": lambda: ExtraTreesRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    ),
    "Ridge Regression": lambda: Ridge(alpha=1.0),
    "KNN (k=10)": lambda: KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1),
}

MODEL_COLORS = {
    "Gradient Boosting": "#22c55e",
    "Random Forest":     "#6366f1",
    "Extra Trees":       "#22d3ee",
    "Ridge Regression":  "#f59e0b",
    "KNN (k=10)":        "#ec4899",
}

MODEL_DESCRIPTIONS = {
    "Gradient Boosting": "Builds an ensemble of small decision trees sequentially, with each new tree correcting the errors of the previous ones. Strong at capturing non-linear price trends and complex feature interactions. Generally the most accurate for structured tabular data.",
    "Random Forest": "Trains many decision trees independently on random subsets of the data, then averages their predictions. Robust to noise and outliers, and less prone to overfitting than a single tree. A solid general-purpose baseline.",
    "Extra Trees": "Similar to Random Forest but uses random split thresholds instead of searching for optimal ones. This makes it faster to train and often produces smoother predictions with lower variance, though it may sacrifice some accuracy.",
    "Ridge Regression": "A regularized linear model that fits a straight-line relationship between features and price. Best suited for commodities with stable, linear trends. Struggles with sudden price shocks or non-linear seasonal patterns, but is highly interpretable.",
    "KNN (k=10)": "Predicts a price by averaging the 10 most similar historical data points (weighted by distance). Captures local patterns well but can struggle with extrapolation into future time periods where no similar neighbors exist.",
}

print("=" * 65)
print("  Multi-Model Food Price Forecasting Pipeline")
print("=" * 65)

# ─── 1. Load WFP data ───────────────────────────────────────
print("\n[1/5] Loading latest WFP data...")
df = pd.read_csv("D:/ML/WFP/wfp_food_prices_phl_latest.csv")
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]
df["region"] = df["admin1"]

print(f"   Records: {len(df):,}")
print(f"   Date range: {df['date'].min().date()} — {df['date'].max().date()}")
print(f"   Commodities: {df['commodity'].nunique()}, Regions: {df['region'].nunique()}")

# ─── 2. Feature Engineering ─────────────────────────────────
print("\n[2/5] Engineering features...")

def build_features(data):
    data = data.sort_values("date").copy()
    data["year_num"] = data["year"] - 2000
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    data["price_lag1"] = data["price"].shift(1)
    data["price_lag3"] = data["price"].shift(3)
    data["price_lag6"] = data["price"].shift(6)
    data["price_lag12"] = data["price"].shift(12)
    data["price_ma3"] = data["price"].rolling(3, min_periods=1).mean()
    data["price_ma6"] = data["price"].rolling(6, min_periods=1).mean()
    data["price_ma12"] = data["price"].rolling(12, min_periods=1).mean()
    data["price_diff1"] = data["price"].diff(1)
    data["price_diff12"] = data["price"].diff(12)
    return data

pieces = []
for _, group in df.groupby(["commodity", "region", "pricetype"], group_keys=False):
    pieces.append(build_features(group))
df_feat = pd.concat(pieces, ignore_index=True)

feature_cols = [
    "year_num", "month_sin", "month_cos",
    "price_lag1", "price_lag3", "price_lag6", "price_lag12",
    "price_ma3", "price_ma6", "price_ma12",
    "price_diff1", "price_diff12",
]

le_region = LabelEncoder()
df_feat["region_enc"] = le_region.fit_transform(df_feat["region"])
feature_cols.append("region_enc")

le_pt = LabelEncoder()
df_feat["pt_enc"] = le_pt.fit_transform(df_feat["pricetype"])
feature_cols.append("pt_enc")

df_feat = df_feat.dropna(subset=feature_cols)

print(f"   Features: {len(feature_cols)}, Usable rows: {len(df_feat):,}")

# ─── 3. Train / Validate Split ──────────────────────────────
train_df = df_feat[df_feat["year"] <= 2023]
val_df = df_feat[df_feat["year"] >= 2024]
all_commodities = df_feat["commodity"].unique()

print(f"   Train: {len(train_df):,} (up to 2023)")
print(f"   Validation: {len(val_df):,} (2024 — Jan 2026)")
print(f"   Commodities: {len(all_commodities)}")

# ─── 4. Train all models per commodity ──────────────────────
print("\n[3/5] Training 5 models per commodity...")

# Structure: {model_name: {commodity: model_object}}
trained_models = {name: {} for name in MODEL_DEFS}
# Structure: {model_name: {commodity: {mape, mae, bias, n_val, val_pred, val_actual, ...}}}
model_results = {name: {} for name in MODEL_DEFS}
# For Ridge/KNN we need scalers per commodity
scalers = {}

for i, comm in enumerate(all_commodities):
    comm_train = train_df[train_df["commodity"] == comm]
    comm_val = val_df[val_df["commodity"] == comm]

    if len(comm_train) < 20:
        continue

    X_train = comm_train[feature_cols].values
    y_train = comm_train["price"].values

    # Fit scaler (needed for Ridge and KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scalers[comm] = scaler

    X_val = comm_val[feature_cols].values if len(comm_val) > 0 else None
    y_val = comm_val["price"].values if len(comm_val) > 0 else None
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None

    for model_name, model_factory in MODEL_DEFS.items():
        model = model_factory()

        # Ridge and KNN need scaled features
        needs_scaling = model_name in ("Ridge Regression", "KNN (k=10)")
        Xt = X_train_scaled if needs_scaling else X_train

        model.fit(Xt, y_train)
        trained_models[model_name][comm] = model

        if X_val is not None and len(y_val) > 0:
            Xv = X_val_scaled if needs_scaling else X_val
            y_pred = np.maximum(model.predict(Xv), 0)

            mape = mean_absolute_percentage_error(y_val, y_pred) * 100
            mae = mean_absolute_error(y_val, y_pred)
            bias = np.mean((y_pred - y_val) / y_val) * 100

            model_results[model_name][comm] = {
                "mape": round(mape, 1),
                "mae": round(mae, 2),
                "bias": round(bias, 1),
                "n_val": len(comm_val),
                "val_pred": y_pred.tolist(),
                "val_actual": y_val.tolist(),
                "val_dates": comm_val["date"].dt.strftime("%Y-%m").tolist(),
                "val_regions": comm_val["region"].tolist(),
                "val_pricetypes": comm_val["pricetype"].tolist(),
            }

    if (i + 1) % 20 == 0:
        print(f"   ...{i+1}/{len(all_commodities)} commodities done")

print(f"   Done — {sum(len(v) for v in trained_models.values())} total model instances")

# ─── 5. Overall metrics per model ───────────────────────────
print("\n[4/5] Computing overall metrics...")

overall_metrics = {}
for model_name in MODEL_DEFS:
    all_actual, all_pred = [], []
    for comm, res in model_results[model_name].items():
        all_actual.extend(res["val_actual"])
        all_pred.extend(res["val_pred"])
    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)
    mape = mean_absolute_percentage_error(all_actual, all_pred) * 100
    mae = mean_absolute_error(all_actual, all_pred)
    bias = np.mean((all_pred - all_actual) / all_actual) * 100
    r2 = 1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - np.mean(all_actual)) ** 2)
    overall_metrics[model_name] = {
        "mape": round(mape, 1),
        "mae": round(mae, 2),
        "bias": round(bias, 1),
        "r2": round(r2, 4),
        "n_val": len(all_actual),
    }

print(f"\n   {'Model':<22s} {'MAPE':>8s} {'MAE':>10s} {'Bias':>8s} {'R²':>8s}")
print(f"   {'─'*58}")
for name in MODEL_DEFS:
    m = overall_metrics[name]
    print(f"   {name:<22s} {m['mape']:>7.1f}% {('PHP '+str(m['mae'])):>10s} {m['bias']:>+7.1f}% {m['r2']:>8.4f}")

# ─── 6. Generate forecasts per model ────────────────────────
print("\n[5/5] Generating forecasts (Feb 2026 — Dec 2027) for all models...")

# {model_name: [forecast_rows]}
all_forecasts = {name: [] for name in MODEL_DEFS}

for model_name in MODEL_DEFS:
    needs_scaling = model_name in ("Ridge Regression", "KNN (k=10)")

    for comm in trained_models[model_name]:
        comm_data = df_feat[df_feat["commodity"] == comm].copy()

        for pt in comm_data["pricetype"].unique():
            for region in comm_data["region"].unique():
                series = comm_data[
                    (comm_data["pricetype"] == pt) & (comm_data["region"] == region)
                ]
                if len(series) < 12:
                    continue
                series = series.sort_values("date")

                region_enc = le_region.transform([region])[0] if region in le_region.classes_ else 0
                pt_enc = le_pt.transform([pt])[0] if pt in le_pt.classes_ else 0
                price_history = list(series["price"].values)
                model = trained_models[model_name][comm]
                scaler = scalers.get(comm)

                for year in (2026, 2027):
                    for month in range(1, 13):
                        if year == 2026 and month <= 1:
                            continue
                        year_num = year - 2000
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
                            year_num,
                            math.sin(2 * math.pi * month / 12),
                            math.cos(2 * math.pi * month / 12),
                            lag1, lag3, lag6, lag12,
                            ma3, ma6, ma12,
                            diff1, diff12,
                            region_enc, pt_enc,
                        ]])
                        if needs_scaling and scaler is not None:
                            features = scaler.transform(features)

                        pred = max(0, float(model.predict(features)[0]))
                        price_history.append(pred)

                        all_forecasts[model_name].append({
                            "year": year, "month": month,
                            "region": region, "commodity": comm,
                            "pricetype": pt, "price": round(pred, 2),
                        })

    print(f"   {model_name}: {len(all_forecasts[model_name]):,} forecast rows")

# ─── Build JSON for dashboard ────────────────────────────────
print("\n   Building dashboard JSON...")

# 1) Historical trends (actual data)
hist_trends = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
for _, r in df.iterrows():
    key = f"{r['commodity']}|{r['pricetype']}"
    dk = f"{r['year']}-{r['month']:02d}"
    hist_trends[key][dk]["sum"] += r["price"]
    hist_trends[key][dk]["count"] += 1

# 2) Per-model validation trends
per_model_trends = {}
for model_name in MODEL_DEFS:
    mt = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
    for comm, res in model_results[model_name].items():
        for i in range(len(res["val_actual"])):
            key = f"{comm}|{res['val_pricetypes'][i]}"
            dk = res["val_dates"][i]
            mt[key][dk]["sum"] += res["val_pred"][i]
            mt[key][dk]["count"] += 1
    # Add forecasts
    for row in all_forecasts[model_name]:
        key = f"{row['commodity']}|{row['pricetype']}"
        dk = f"{row['year']}-{row['month']:02d}"
        mt[key][dk]["sum"] += row["price"]
        mt[key][dk]["count"] += 1
    per_model_trends[model_name] = mt

# Combine into trends_json: {series_key: {date: {actual, model1_pred, model2_pred, ...}}}
all_series_keys = set(hist_trends.keys())
for mt in per_model_trends.values():
    all_series_keys.update(mt.keys())

trends_json = {}
for skey in all_series_keys:
    series = {}
    for dk, v in hist_trends.get(skey, {}).items():
        series[dk] = {"actual": round(v["sum"] / v["count"], 2)}
    for model_name, mt in per_model_trends.items():
        for dk, v in mt.get(skey, {}).items():
            if dk not in series:
                series[dk] = {}
            series[dk][model_name] = round(v["sum"] / v["count"], 2)
    trends_json[skey] = series

# 3) Per-commodity comparison across all models
comm_comparison = {}
all_comm_set = set()
for model_name in MODEL_DEFS:
    for comm in model_results[model_name]:
        all_comm_set.add(comm)
for comm in sorted(all_comm_set):
    entry = {"commodity": comm}
    for model_name in MODEL_DEFS:
        res = model_results[model_name].get(comm)
        if res:
            entry[model_name] = {"mape": res["mape"], "mae": res["mae"], "n_val": res["n_val"]}
        else:
            entry[model_name] = None
    comm_comparison[comm] = entry

# 4) Forecasts per model (aggregated by commodity)
forecasts_json = {}
for model_name in MODEL_DEFS:
    fc_agg = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
    for row in all_forecasts[model_name]:
        dk = f"{row['year']}-{row['month']:02d}"
        fc_agg[row["commodity"]][dk]["sum"] += row["price"]
        fc_agg[row["commodity"]][dk]["count"] += 1
    forecasts_json[model_name] = {
        comm: {dk: round(v["sum"] / v["count"], 2) for dk, v in dates.items()}
        for comm, dates in fc_agg.items()
    }

# Assemble final JSON
dashboard_data = {
    "models": list(MODEL_DEFS.keys()),
    "modelColors": MODEL_COLORS,
    "modelDescriptions": MODEL_DESCRIPTIONS,
    "overall": overall_metrics,
    "trends": trends_json,
    "commComparison": comm_comparison,
    "forecasts": forecasts_json,
    "meta": {
        "commodities": sorted(df["commodity"].unique().tolist()),
        "pricetypes": sorted(df["pricetype"].unique().tolist()),
        "regions": sorted(df["region"].unique().tolist()),
        "trainPeriod": "2000–2023",
        "valPeriod": "2024 – Jan 2026",
        "forecastPeriod": "Feb 2026 – Dec 2027",
        "nTrain": len(train_df),
        "nVal": len(val_df),
    },
}

with open("D:/ML/Website/model_comparison.json", "w") as f:
    json.dump(dashboard_data, f, separators=(",", ":"))

print(f"\n{'='*65}")
print(f"  Saved: D:/ML/Website/model_comparison.json")
best = min(overall_metrics.items(), key=lambda x: x[1]["mape"])
print(f"  Best model: {best[0]} (MAPE {best[1]['mape']}%)")
print(f"{'='*65}")
