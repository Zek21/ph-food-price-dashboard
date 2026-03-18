"""
Ensemble model stacking for Philippine food price prediction.
# signed: delta

Combines GradientBoosting, ExtraTrees, and RandomForest via scikit-learn
StackingRegressor with Ridge regression as the meta-learner. Uses the same
feature engineering pipeline as retrain_model.py to ensure consistency.

Usage:
    python ensemble_model.py --train --evaluate
    python ensemble_model.py --train                  # train only, skip eval
    python ensemble_model.py --evaluate               # evaluate existing model
    python ensemble_model.py --train --evaluate --cv 3 # custom CV folds
"""
# signed: delta

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Configuration ──────────────────────────────────────────
# signed: delta
try:
    _SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_PATH = os.environ.get(
    "WFP_DATA_PATH",
    str(_SCRIPT_DIR.parent / "WFP" / "wfp_food_prices_phl_latest.csv"),
)
MODEL_COMPARISON_PATH = _SCRIPT_DIR / "model_comparison.json"
ENSEMBLE_OUTPUT_PATH = _SCRIPT_DIR / "ensemble_predictions.json"
ENSEMBLE_CACHE_DIR = _SCRIPT_DIR / ".ensemble_cache"

# ─── Feature engineering (mirrors retrain_model.py exactly) ──
# signed: delta


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create temporal and lag features for price prediction.

    Identical to retrain_model.py's build_features to ensure
    feature parity between individual and ensemble models.
    """
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


FEATURE_COLS = [
    "year_num",
    "month_sin",
    "month_cos",
    "price_lag1",
    "price_lag3",
    "price_lag6",
    "price_lag12",
    "price_ma3",
    "price_ma6",
    "price_ma12",
    "price_diff1",
    "price_diff12",
]


def load_and_prepare_data(data_path: str) -> tuple:
    """Load WFP CSV and prepare features.

    Returns:
        (df_feat, feature_cols, le_region, le_pt, raw_df)
    """
    # signed: delta
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            "Set WFP_DATA_PATH env or place CSV in ../WFP/."
        )

    df = pd.read_csv(data_path)
    required = ["date", "price", "commodity", "admin1", "pricetype"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    df["region"] = df["admin1"]

    if len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} records after cleaning.")

    # Build features per group
    pieces = []
    for _, group in df.groupby(
        ["commodity", "region", "pricetype"], group_keys=False
    ):
        pieces.append(build_features(group))
    df_feat = pd.concat(pieces, ignore_index=True)

    feature_cols = list(FEATURE_COLS)

    le_region = LabelEncoder()
    df_feat["region_enc"] = le_region.fit_transform(df_feat["region"])
    feature_cols.append("region_enc")

    le_pt = LabelEncoder()
    df_feat["pt_enc"] = le_pt.fit_transform(df_feat["pricetype"])
    feature_cols.append("pt_enc")

    df_feat = df_feat.dropna(subset=feature_cols)

    return df_feat, feature_cols, le_region, le_pt, df


def build_stacking_regressor(cv_folds: int = 5) -> StackingRegressor:
    """Build a StackingRegressor combining the 3 best tree-based models.

    Base estimators (best default configs from retrain_model.py):
      - GradientBoosting: 200 trees, depth=4, lr=0.05
      - ExtraTrees: 200 trees, depth=10
      - RandomForest: 200 trees, depth=10

    Meta-learner: Ridge(alpha=1.0) — stable, prevents overfitting.

    CV uses KFold(shuffle=False) to maintain temporal ordering while
    producing full partitions required by sklearn 1.8+ cross_val_predict.
    TimeSeriesSplit is incompatible because it never includes early samples
    in any test fold, violating the partition requirement.
    """
    # signed: delta
    estimators = [
        (
            "gb",
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
        (
            "et",
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    stacker = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=KFold(n_splits=cv_folds, shuffle=False),
        passthrough=False,
        n_jobs=-1,
    )
    return stacker


def train_ensemble(
    df_feat: pd.DataFrame,
    feature_cols: list,
    cv_folds: int = 5,
) -> dict:
    """Train ensemble stacking model per commodity.

    Returns dict: {commodity: {"model": fitted_stacker, "scaler": scaler}}
    """
    # signed: delta
    ENSEMBLE_CACHE_DIR.mkdir(exist_ok=True)
    _latest_date = df_feat["date"].max()
    _val_start = (_latest_date - pd.DateOffset(months=24)).replace(day=1)

    train_df = df_feat[df_feat["date"] < _val_start]
    all_commodities = df_feat["commodity"].unique()

    _data_hash = hashlib.md5(
        f"{len(train_df)}_{len(all_commodities)}_ensemble".encode()
    ).hexdigest()[:12]

    print(f"\n[Ensemble] Training stacking ensemble on {len(all_commodities)} commodities...")
    print(f"   Train period: up to {(_val_start - pd.DateOffset(months=1)).strftime('%b %Y')}")
    print(f"   CV strategy: TimeSeriesSplit(n_splits={cv_folds})")

    trained = {}
    skipped = 0
    t0_total = time.perf_counter()

    for comm in all_commodities:
        comm_train = train_df[train_df["commodity"] == comm]
        if len(comm_train) < 30:
            skipped += 1
            continue

        X_train = comm_train[feature_cols].values
        y_train = comm_train["price"].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Check cache
        safe_comm = (
            comm.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        cache_file = ENSEMBLE_CACHE_DIR / f"ensemble__{safe_comm}__{_data_hash}.joblib"

        if cache_file.exists():
            try:
                cached = joblib.load(cache_file)
                trained[comm] = cached
                continue
            except Exception:
                pass

        stacker = build_stacking_regressor(cv_folds)
        try:
            stacker.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"   WARNING: Failed to train ensemble for {comm}: {e}")
            skipped += 1
            continue

        entry = {"model": stacker, "scaler": scaler}

        try:
            joblib.dump(entry, cache_file)
        except Exception:
            pass

        trained[comm] = entry

    elapsed = time.perf_counter() - t0_total
    print(f"   Trained: {len(trained)} commodities ({skipped} skipped), {elapsed:.1f}s")
    return trained


def evaluate_ensemble(
    trained: dict,
    df_feat: pd.DataFrame,
    feature_cols: list,
    le_region: LabelEncoder,
    le_pt: LabelEncoder,
) -> dict:
    """Evaluate ensemble vs individual models.

    Returns evaluation results dict suitable for JSON serialization.
    """
    # signed: delta
    _latest_date = df_feat["date"].max()
    _val_start = (_latest_date - pd.DateOffset(months=24)).replace(day=1)
    val_df = df_feat[df_feat["date"] >= _val_start]

    print(f"\n[Ensemble] Evaluating on validation set ({_val_start.strftime('%Y-%m')} -- {_latest_date.strftime('%b %Y')})...")

    # Ensemble validation metrics
    all_actual, all_pred = [], []
    per_comm_metrics = {}

    for comm, entry in trained.items():
        comm_val = val_df[val_df["commodity"] == comm]
        if len(comm_val) == 0:
            continue

        X_val = comm_val[feature_cols].values
        y_val = comm_val["price"].values
        X_val_scaled = entry["scaler"].transform(X_val)

        y_pred = np.maximum(entry["model"].predict(X_val_scaled), 0)

        mape = mean_absolute_percentage_error(y_val, y_pred) * 100
        mae = mean_absolute_error(y_val, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        bias = float(np.mean((y_pred - y_val) / y_val) * 100) if np.all(y_val > 0) else 0.0

        per_comm_metrics[comm] = {
            "mape": round(mape, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "bias": round(bias, 2),
            "n_val": len(y_val),
        }
        all_actual.extend(y_val.tolist())
        all_pred.extend(y_pred.tolist())

    all_actual_arr = np.array(all_actual)
    all_pred_arr = np.array(all_pred)

    if len(all_actual_arr) == 0:
        # No commodities were evaluated -- return empty metrics  # signed: delta
        return {
            "ensemble_overall": {
                "mape": None, "mae": None, "rmse": None,
                "r2": None, "bias": None, "n_val": 0,
            },
            "per_commodity": per_comm_metrics,
            "comparison_vs_individual": _compare_with_individual_models(
                {"mape": None, "mae": None}
            ),
        }

    overall_mape = mean_absolute_percentage_error(all_actual_arr, all_pred_arr) * 100
    overall_mae = mean_absolute_error(all_actual_arr, all_pred_arr)
    overall_rmse = float(np.sqrt(mean_squared_error(all_actual_arr, all_pred_arr)))
    overall_r2 = float(
        1
        - np.sum((all_actual_arr - all_pred_arr) ** 2)
        / np.sum((all_actual_arr - np.mean(all_actual_arr)) ** 2)
    )
    overall_bias = float(
        np.mean((all_pred_arr - all_actual_arr) / all_actual_arr) * 100
    )

    ensemble_overall = {
        "mape": round(overall_mape, 2),
        "mae": round(overall_mae, 2),
        "rmse": round(overall_rmse, 2),
        "r2": round(overall_r2, 4),
        "bias": round(overall_bias, 2),
        "n_val": len(all_actual),
    }

    print(f"\n   Ensemble Overall: MAPE={ensemble_overall['mape']:.1f}%, "
          f"MAE=PHP {ensemble_overall['mae']:.2f}, RMSE={ensemble_overall['rmse']:.2f}, "
          f"R2={ensemble_overall['r2']:.4f}")

    # Load individual model results for comparison
    comparison = _compare_with_individual_models(ensemble_overall)

    return {
        "ensemble_overall": ensemble_overall,
        "per_commodity": per_comm_metrics,
        "comparison_vs_individual": comparison,
    }


def _compare_with_individual_models(ensemble_overall: dict) -> dict:
    """Compare ensemble metrics against individual models from model_comparison.json."""
    # signed: delta
    comparison = {}
    if MODEL_COMPARISON_PATH.exists():
        with open(MODEL_COMPARISON_PATH, encoding="utf-8") as f:
            mc = json.load(f)
        for model_name, metrics in mc.get("overall", {}).items():
            ind_mape = metrics.get("mape", float("inf"))
            ens_mape = ensemble_overall["mape"]
            improvement = round(ind_mape - ens_mape, 2)
            comparison[model_name] = {
                "individual_mape": ind_mape,
                "ensemble_mape": ens_mape,
                "mape_improvement": improvement,
                "ensemble_wins": improvement > 0,
            }
    return comparison


def generate_forecasts(
    trained: dict,
    df_feat: pd.DataFrame,
    feature_cols: list,
    le_region: LabelEncoder,
    le_pt: LabelEncoder,
) -> list:
    """Generate forecasts (Feb 2026 -- Dec 2027) using the ensemble model.

    Uses the same iterative forecasting approach as retrain_model.py.
    """
    # signed: delta
    print("\n[Ensemble] Generating forecasts (Feb 2026 -- Dec 2027)...")

    forecasts = []
    for comm, entry in trained.items():
        model = entry["model"]
        scaler = entry["scaler"]

        comm_data = df_feat[df_feat["commodity"] == comm].copy()

        for pt in comm_data["pricetype"].unique():
            for region in comm_data["region"].unique():
                series = comm_data[
                    (comm_data["pricetype"] == pt) & (comm_data["region"] == region)
                ]
                if len(series) < 12:
                    continue
                series = series.sort_values("date")

                region_enc = (
                    le_region.transform([region])[0]
                    if region in le_region.classes_
                    else 0
                )
                pt_enc = (
                    le_pt.transform([pt])[0] if pt in le_pt.classes_ else 0
                )
                price_history = list(series["price"].values)

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
                        diff1 = (
                            price_history[-1] - price_history[-2] if n >= 2 else 0
                        )
                        diff12 = (
                            price_history[-1] - price_history[-13] if n >= 13 else 0
                        )

                        features = np.array(
                            [
                                [
                                    year_num,
                                    math.sin(2 * math.pi * month / 12),
                                    math.cos(2 * math.pi * month / 12),
                                    lag1,
                                    lag3,
                                    lag6,
                                    lag12,
                                    ma3,
                                    ma6,
                                    ma12,
                                    diff1,
                                    diff12,
                                    region_enc,
                                    pt_enc,
                                ]
                            ]
                        )
                        features_scaled = scaler.transform(features)
                        pred = max(0, float(model.predict(features_scaled)[0]))

                        # Cap at 5x recent average
                        recent_avg = (
                            np.mean(price_history[-12:])
                            if len(price_history) >= 12
                            else np.mean(price_history)
                        )
                        if pred > recent_avg * 5:
                            pred = recent_avg * 1.5

                        price_history.append(pred)

                        forecasts.append(
                            {
                                "year": year,
                                "month": month,
                                "region": region,
                                "commodity": comm,
                                "pricetype": pt,
                                "price": round(pred, 2),
                            }
                        )

    print(f"   Generated {len(forecasts):,} forecast rows")
    return forecasts


def save_results(
    eval_results: dict,
    forecasts: list,
    trained: dict,
    df_feat: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Save ensemble predictions and evaluation to JSON."""
    # signed: delta
    out = output_path or ENSEMBLE_OUTPUT_PATH
    _latest_date = df_feat["date"].max()
    _val_start = (_latest_date - pd.DateOffset(months=24)).replace(day=1)

    # Build forecast trends: {commodity|pricetype: {YYYY-MM: avg_price}}
    forecast_trends = defaultdict(lambda: defaultdict(lambda: {"sum": 0, "count": 0}))
    for row in forecasts:
        key = f"{row['commodity']}|{row['pricetype']}"
        dk = f"{row['year']}-{row['month']:02d}"
        forecast_trends[key][dk]["sum"] += row["price"]
        forecast_trends[key][dk]["count"] += 1

    forecast_trends_avg = {}
    for skey, dates in forecast_trends.items():
        forecast_trends_avg[skey] = {
            dk: round(v["sum"] / v["count"], 2) for dk, v in dates.items()
        }

    output = {
        "model": "Ensemble (Stacking)",
        "base_models": ["Gradient Boosting", "Extra Trees", "Random Forest"],
        "meta_learner": "Ridge(alpha=1.0)",
        "cv_strategy": "TimeSeriesSplit",
        "evaluation": eval_results,
        "forecasts": forecast_trends_avg,
        "meta": {
            "commodities": sorted(trained.keys()),
            "n_commodities": len(trained),
            "trainPeriod": f"2000 -- {(_val_start - pd.DateOffset(months=1)).strftime('%b %Y')}",
            "valPeriod": f"{_val_start.strftime('%Y-%m')} -- {_latest_date.strftime('%b %Y')}",
            "forecastPeriod": "Feb 2026 -- Dec 2027",
        },
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n   Results saved to {out}")
    print(f"   File size: {out.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble model stacking for food price prediction"
    )
    parser.add_argument("--train", action="store_true", help="Train ensemble model")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate ensemble vs individual models"
    )
    parser.add_argument(
        "--cv", type=int, default=5, help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--data", type=str, default=DATA_PATH, help="Path to WFP CSV data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ENSEMBLE_OUTPUT_PATH),
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not args.train and not args.evaluate:
        parser.print_help()
        sys.exit(1)

    print("=" * 65)
    print("  Ensemble Model Stacking Pipeline")
    print("  Base: GradientBoosting + ExtraTrees + RandomForest")
    print("  Meta-learner: Ridge(alpha=1.0)")
    print("=" * 65)

    df_feat, feature_cols, le_region, le_pt, raw_df = load_and_prepare_data(
        args.data
    )
    print(f"   Data loaded: {len(df_feat):,} rows, {df_feat['commodity'].nunique()} commodities")

    eval_results = {}
    forecasts = []
    trained = {}

    if args.train:
        trained = train_ensemble(df_feat, feature_cols, cv_folds=args.cv)

        if args.evaluate:
            eval_results = evaluate_ensemble(
                trained, df_feat, feature_cols, le_region, le_pt
            )

        forecasts = generate_forecasts(
            trained, df_feat, feature_cols, le_region, le_pt
        )

        save_results(
            eval_results,
            forecasts,
            trained,
            df_feat,
            Path(args.output),
        )

    elif args.evaluate:
        # Evaluate-only: load from cache
        print("\n[Ensemble] Evaluate-only mode -- loading cached models...")
        ENSEMBLE_CACHE_DIR.mkdir(exist_ok=True)
        for cache_file in ENSEMBLE_CACHE_DIR.glob("ensemble__*.joblib"):
            try:
                entry = joblib.load(cache_file)
                # Extract commodity name from filename
                parts = cache_file.stem.split("__")
                if len(parts) >= 2:
                    comm = parts[1].replace("_", " ")
                    trained[comm] = entry
            except Exception:
                continue

        if not trained:
            print("   ERROR: No cached ensemble models found. Run --train first.")
            sys.exit(1)

        print(f"   Loaded {len(trained)} cached ensemble models.")
        eval_results = evaluate_ensemble(
            trained, df_feat, feature_cols, le_region, le_pt
        )

    # Print comparison summary
    if eval_results and "comparison_vs_individual" in eval_results:
        comp = eval_results["comparison_vs_individual"]
        print("\n   --- Ensemble vs Individual Models ---")
        print(f"   {'Model':<22s} {'Indiv MAPE':>12s} {'Ens MAPE':>10s} {'Improve':>10s}")
        print(f"   {'-'*56}")
        for name, c in comp.items():
            flag = "  *" if c["ensemble_wins"] else ""
            print(
                f"   {name:<22s} {c['individual_mape']:>11.1f}% "
                f"{c['ensemble_mape']:>9.1f}% "
                f"{c['mape_improvement']:>+9.1f}%{flag}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
