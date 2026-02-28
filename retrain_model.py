"""
Multi-model food price forecasting pipeline.

Trains 5 different ML model families on WFP Philippines price data.
For each family, 5 hyperparameter variants are evaluated on the held-out
2024–Jan 2026 validation set and the best-performing variant is selected
per commodity. Forecasts are generated through Dec 2027. A single JSON
file is written for consumption by comparison.html.

Model families (5 variants each = 25 models total per commodity):
  1. Gradient Boosting  — sequential ensemble of weak learners; great at trends
  2. Extra Trees        — extremely randomised trees; low variance, fast training
  3. Random Forest      — bagged decision trees; robust to outliers
  4. KNN (k varies)     — instance-based; captures local price similarity
  5. Ridge Regression   — regularised linear model; interpretable, stable trends
"""

import json
import math
import os
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

# ─── Hyperparameter variants (5 per model family) ───────────
# For each family, all 5 variants are trained; the one with the lowest
# per-commodity validation MAPE is retained for predictions / forecasts.
MODEL_VARIANTS = {
    "Gradient Boosting": [
        GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.10, subsample=0.9,  min_samples_leaf=5, random_state=42),
        GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,  min_samples_leaf=5, random_state=42),
        GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,  min_samples_leaf=3, random_state=42),
        GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.02, subsample=0.7,  min_samples_leaf=5, random_state=42),
        GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.8,  min_samples_leaf=5, random_state=42),
    ],
    "Extra Trees": [
        ExtraTreesRegressor(n_estimators=50,  max_depth=6,    min_samples_leaf=3, random_state=42, n_jobs=-1),
        ExtraTreesRegressor(n_estimators=100, max_depth=8,    min_samples_leaf=5, random_state=42, n_jobs=-1),
        ExtraTreesRegressor(n_estimators=200, max_depth=10,   min_samples_leaf=5, random_state=42, n_jobs=-1),
        ExtraTreesRegressor(n_estimators=300, max_depth=12,   min_samples_leaf=3, random_state=42, n_jobs=-1),
        ExtraTreesRegressor(n_estimators=200, max_depth=None, min_samples_leaf=5, random_state=42, n_jobs=-1),
    ],
    "Random Forest": [
        RandomForestRegressor(n_estimators=50,  max_depth=6,    min_samples_leaf=5, random_state=42, n_jobs=-1),
        RandomForestRegressor(n_estimators=100, max_depth=8,    min_samples_leaf=5, random_state=42, n_jobs=-1),
        RandomForestRegressor(n_estimators=200, max_depth=10,   min_samples_leaf=3, random_state=42, n_jobs=-1),
        RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=5, random_state=42, n_jobs=-1),
        RandomForestRegressor(n_estimators=500, max_depth=8,    min_samples_leaf=3, random_state=42, n_jobs=-1),
    ],
    "KNN (k=10)": [
        KNeighborsRegressor(n_neighbors=5,  weights="uniform",  n_jobs=-1),
        KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1),
        KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1),
        KNeighborsRegressor(n_neighbors=20, weights="uniform",  n_jobs=-1),
        KNeighborsRegressor(n_neighbors=30, weights="distance", n_jobs=-1),
    ],
    "Ridge Regression": [
        Ridge(alpha=0.01),
        Ridge(alpha=0.10),
        Ridge(alpha=1.00),
        Ridge(alpha=10.0),
        Ridge(alpha=100.0),
    ],
}

# Parameter grids stored for dashboard display
VARIANT_SEARCH = {
    "Gradient Boosting": {
        "parameter_grid": [
            {"n_estimators": 100,  "learning_rate": 0.10, "max_depth": 3, "subsample": 0.9},
            {"n_estimators": 200,  "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8},
            {"n_estimators": 300,  "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8},
            {"n_estimators": 200,  "learning_rate": 0.02, "max_depth": 4, "subsample": 0.7},
            {"n_estimators": 500,  "learning_rate": 0.01, "max_depth": 3, "subsample": 0.8},
        ],
        "selection_metric": "Validation MAPE",
    },
    "Extra Trees": {
        "parameter_grid": [
            {"n_estimators": 50,   "max_depth": 6,    "min_samples_leaf": 3},
            {"n_estimators": 100,  "max_depth": 8,    "min_samples_leaf": 5},
            {"n_estimators": 200,  "max_depth": 10,   "min_samples_leaf": 5},
            {"n_estimators": 300,  "max_depth": 12,   "min_samples_leaf": 3},
            {"n_estimators": 200,  "max_depth": None, "min_samples_leaf": 5},
        ],
        "selection_metric": "Validation MAPE",
    },
    "Random Forest": {
        "parameter_grid": [
            {"n_estimators": 50,   "max_depth": 6,    "min_samples_leaf": 5},
            {"n_estimators": 100,  "max_depth": 8,    "min_samples_leaf": 5},
            {"n_estimators": 200,  "max_depth": 10,   "min_samples_leaf": 3},
            {"n_estimators": 300,  "max_depth": None, "min_samples_leaf": 5},
            {"n_estimators": 500,  "max_depth": 8,    "min_samples_leaf": 3},
        ],
        "selection_metric": "Validation MAPE",
    },
    "KNN (k=10)": {
        "parameter_grid": [
            {"n_neighbors": 5,   "weights": "uniform"},
            {"n_neighbors": 10,  "weights": "distance"},
            {"n_neighbors": 15,  "weights": "distance"},
            {"n_neighbors": 20,  "weights": "uniform"},
            {"n_neighbors": 30,  "weights": "distance"},
        ],
        "selection_metric": "Validation MAPE",
    },
    "Ridge Regression": {
        "parameter_grid": [
            {"alpha": 0.01},
            {"alpha": 0.10},
            {"alpha": 1.00},
            {"alpha": 10.0},
            {"alpha": 100.0},
        ],
        "selection_metric": "Validation MAPE",
    },
}

MODEL_NAMES = list(MODEL_VARIANTS.keys())

# Legacy MODEL_DEFS: maps each model name to a callable that returns its
# second (index 1) variant as a reasonable single-model default.
MODEL_DEFS = {}
for _name in MODEL_NAMES:
    _variants = MODEL_VARIANTS[_name]
    _default_idx = 1 if len(_variants) > 1 else 0
    MODEL_DEFS[_name] = lambda _v=_variants[_default_idx]: _v

MODEL_COLORS = {
    "Gradient Boosting": "#22c55e",
    "Extra Trees":       "#22d3ee",
    "Random Forest":     "#6366f1",
    "KNN (k=10)":        "#ec4899",
    "Ridge Regression":  "#f59e0b",
}

MODEL_DESCRIPTIONS = {
    "Gradient Boosting": (
        "Builds an ensemble of shallow decision trees sequentially. Each new tree "
        "focuses on correcting the residual errors of all previous trees, guided by "
        "gradient descent in function space. Best suited for structured tabular data "
        "with complex feature interactions. 5 hyperparameter variants were tested "
        "(varying n_estimators 100–500, learning rate 0.01–0.1, depth 3–6) and the "
        "best was selected per commodity on validation MAPE."
    ),
    "Extra Trees": (
        "Extremely Randomized Trees grows many decision trees, but unlike Random Forest "
        "it picks split thresholds completely at random rather than searching for the "
        "optimal cut. This extreme randomisation reduces variance significantly and makes "
        "training much faster. 5 variants tested across n_estimators 50–400 and depths "
        "6–None; the best per-commodity variant was retained."
    ),
    "Random Forest": (
        "Trains a large collection of decision trees on random bootstrap samples of the "
        "data, then averages their predictions. The bagging (bootstrap aggregating) "
        "strategy decorrelates the trees and reduces overfitting. Robust to noisy data "
        "and outliers. 5 variants tested with n_estimators 50–500 and varying max_depth "
        "and min_samples_leaf; best chosen on validation MAPE."
    ),
    "KNN (k=10)": (
        "K-Nearest Neighbours predicts a price by finding the k most similar historical "
        "data points in feature space and computing their distance-weighted average. "
        "Captures highly local patterns and requires no explicit training phase. "
        "5 variants with k in {5, 10, 15, 20, 30} and uniform/distance weighting were "
        "compared; the best was selected per commodity."
    ),
    "Ridge Regression": (
        "A regularised linear model that adds an L2 penalty to the ordinary least-squares "
        "objective, shrinking coefficients to prevent overfitting. Best suited for "
        "commodities with stable, near-linear price trends. Struggles with abrupt price "
        "shocks and non-linear seasonality, but is fully interpretable. 5 alpha values "
        "(0.01, 0.1, 1.0, 10.0, 100.0) were cross-validated; the best was kept."
    ),
}

print("=" * 65)
print("  Multi-Model Food Price Forecasting Pipeline")
print("  (5 hyperparameter variants per model, best selected)")
print("=" * 65)

# ─── Configurable paths ─────────────────────────────────────
DATA_PATH = os.environ.get("WFP_DATA_PATH", "wfp_food_prices_phl_latest.csv")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "model_comparison.json")

# ─── 1. Load WFP data ───────────────────────────────────────
print("\n[1/5] Loading latest WFP data...")
df = pd.read_csv(DATA_PATH)
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

# ─── 4. Train 5 variants per model, select best per commodity ─
print(f"\n[3/5] Training {len(MODEL_NAMES)} models × 5 variants per commodity...")
print(f"      Selecting best variant per commodity by validation MAPE.")

# Structure: {model_name: {commodity: best_model_object}}
trained_models = {name: {} for name in MODEL_NAMES}
# Structure: {model_name: {commodity: metrics_dict}}
model_results = {name: {} for name in MODEL_NAMES}
# Best variant index chosen per (model, commodity)
best_variant_idx = {name: {} for name in MODEL_NAMES}
# For Ridge/KNN we need scalers per commodity
scalers = {}

NEEDS_SCALING = {"Ridge Regression", "KNN (k=10)"}

for i, comm in enumerate(all_commodities):
    comm_train = train_df[train_df["commodity"] == comm]
    comm_val = val_df[val_df["commodity"] == comm]

    if len(comm_train) < 20:
        continue

    X_train = comm_train[feature_cols].values
    y_train = comm_train["price"].values

    # Fit scaler once per commodity (shared by Ridge and KNN variants)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scalers[comm] = scaler

    has_val = len(comm_val) > 0
    X_val = comm_val[feature_cols].values if has_val else None
    y_val = comm_val["price"].values if has_val else None
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None

    for model_name, variants in MODEL_VARIANTS.items():
        needs_scaling = model_name in NEEDS_SCALING
        Xt = X_train_scaled if needs_scaling else X_train
        Xv = X_val_scaled if needs_scaling else X_val

        best_model, best_mape, best_vi, best_pred = None, float("inf"), 0, None

        for v_idx, variant in enumerate(variants):
            try:
                variant.fit(Xt, y_train)
                if has_val and y_val is not None:
                    y_pred = np.maximum(variant.predict(Xv), 0)
                    v_mape = mean_absolute_percentage_error(y_val, y_pred) * 100
                    if v_mape < best_mape:
                        best_mape = v_mape
                        best_model = variant
                        best_vi = v_idx
                        best_pred = y_pred
                else:
                    # No validation data — keep first variant
                    if best_model is None:
                        best_model = variant
                        best_vi = v_idx
            except Exception:
                continue

        if best_model is None:
            continue

        trained_models[model_name][comm] = best_model
        best_variant_idx[model_name][comm] = best_vi

        # Record validation results for the winning variant
        if has_val and best_pred is not None and len(y_val) > 0:
            mape = mean_absolute_percentage_error(y_val, best_pred) * 100
            mae = mean_absolute_error(y_val, best_pred)
            bias = np.mean((best_pred - y_val) / y_val) * 100

            model_results[model_name][comm] = {
                "mape": round(mape, 1),
                "mae": round(mae, 2),
                "bias": round(bias, 1),
                "n_val": len(comm_val),
                "best_variant": best_vi,
                "val_pred": best_pred.tolist(),
                "val_actual": y_val.tolist(),
                "val_dates": comm_val["date"].dt.strftime("%Y-%m").tolist(),
                "val_regions": comm_val["region"].tolist(),
                "val_pricetypes": comm_val["pricetype"].tolist(),
            }

    if (i + 1) % 20 == 0:
        print(f"   ...{i+1}/{len(all_commodities)} commodities done")

total_inst = sum(len(v) for v in trained_models.values())
print(f"   Done — {total_inst} best models selected ({total_inst * 5} variants evaluated)")

# ─── 5. Overall metrics per model ───────────────────────────
print("\n[4/5] Computing overall metrics...")

overall_metrics = {}
for model_name in MODEL_NAMES:
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
for name in MODEL_NAMES:
    m = overall_metrics[name]
    print(f"   {name:<22s} {m['mape']:>7.1f}% {('PHP '+str(m['mae'])):>10s} {m['bias']:>+7.1f}% {m['r2']:>8.4f}")

# ─── 6. Generate forecasts per model ────────────────────────
print("\n[5/5] Generating forecasts (Feb 2026 — Dec 2027) for all models...")

# {model_name: [forecast_rows]}
all_forecasts = {name: [] for name in MODEL_NAMES}

for model_name in MODEL_NAMES:
    needs_scaling = model_name in NEEDS_SCALING

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
for model_name in MODEL_NAMES:
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
    # Skip date entries with no data and skip entirely empty series
    series = {dk: dv for dk, dv in series.items() if dv}
    if series:
        trends_json[skey] = series

# 3) Per-commodity comparison across all models (dict keyed by commodity)
comm_comparison = {}
all_comm_set = set()
for model_name in MODEL_NAMES:
    for comm in model_results[model_name]:
        all_comm_set.add(comm)
for comm in sorted(all_comm_set):
    entry = {"commodity": comm}
    for model_name in MODEL_NAMES:
        res = model_results[model_name].get(comm)
        if res:
            entry[model_name] = {
                "mape": res["mape"], "mae": res["mae"], "n_val": res["n_val"],
                "best_variant": res.get("best_variant", 0),
            }
        else:
            entry[model_name] = None
    comm_comparison[comm] = entry

# 4) Forecasts per model (aggregated by commodity)
forecasts_json = {}
for model_name in MODEL_NAMES:
    fc_agg = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
    for row in all_forecasts[model_name]:
        dk = f"{row['year']}-{row['month']:02d}"
        fc_agg[row["commodity"]][dk]["sum"] += row["price"]
        fc_agg[row["commodity"]][dk]["count"] += 1
    forecasts_json[model_name] = {
        comm: {dk: round(v["sum"] / v["count"], 2) for dk, v in dates.items()}
        for comm, dates in fc_agg.items()
    }

# 5) Annotate VARIANT_SEARCH with best indices per model (averaged over commodities)
vs_annotated = {}
for model_name in MODEL_NAMES:
    vs = dict(VARIANT_SEARCH[model_name])
    # Compute most-common best variant index across commodities
    counts = {}
    for comm, idx in best_variant_idx[model_name].items():
        counts[idx] = counts.get(idx, 0) + 1
    if counts:
        best = max(counts, key=lambda k: counts[k])
    else:
        best = 1
    vs["best_variant"] = best
    vs_annotated[model_name] = vs

# Assemble final JSON
dashboard_data = {
    "models": MODEL_NAMES,
    "modelColors": MODEL_COLORS,
    "modelDescriptions": MODEL_DESCRIPTIONS,
    "variantSearch": vs_annotated,
    "overall": overall_metrics,
    "trends": trends_json,
    "commComparison": comm_comparison,
    "forecasts": forecasts_json,
    "meta": {
        "commodities": sorted(df["commodity"].unique().tolist()),
        "pricetypes": sorted(df["pricetype"].unique().tolist()),
        "regions": sorted(df["region"].unique().tolist()),
        "trainPeriod": "2000 – 2023",
        "valPeriod": "2024 – Jan 2026",
        "forecastPeriod": "Feb 2026 – Dec 2027",
        "nTrain": len(train_df),
        "nVal": len(val_df),
        "nDataPoints": len(val_df),
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(dashboard_data, f, separators=(",", ":"))

print(f"\n{'='*65}")
print(f"  Saved: {OUTPUT_PATH}")
best = min(overall_metrics.items(), key=lambda x: x[1]["mape"])
print(f"  Best model: {best[0]} (MAPE {best[1]['mape']}%)")
print(f"{'='*65}")
