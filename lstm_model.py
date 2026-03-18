"""
LSTM food price forecasting for WFP Philippines data.
# signed: alpha

Trains per-commodity LSTM models on sliding windows of 12 months.
Uses the same WFP data and train/validation split as retrain_model.py.
Outputs predictions in a format compatible with the comparison dashboard.

Architecture:
  Input (seq_len=12, features=6) ->
  LSTM(input=6, hidden=128, layers=2, dropout=0.2, batch_first=True) ->
  FC(128, 64) -> ReLU -> Dropout(0.2) -> FC(64, 1)

Features per timestep:
  price (normalized), month_sin, month_cos, year_normalized,
  region_encoded, pricetype_encoded

Usage:
  python lstm_model.py --train --epochs 50
  python lstm_model.py --predict
  python lstm_model.py --predict --commodity "Rice (Regular Milled)" --region "National Capital region"
  python lstm_model.py --train --predict --epochs 100 --lr 0.001
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────
try:
    _SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
except NameError:
    _SCRIPT_DIR = Path.cwd()

DATA_PATH = os.environ.get(
    "WFP_DATA_PATH",
    str(_SCRIPT_DIR / "wfp_food_prices_phl_latest.csv"),
)
# Fallback to WFP folder if not in Website/
if not os.path.exists(DATA_PATH):
    _alt = Path(r"D:\ML\WFP\wfp_food_prices_phl_latest.csv")
    if _alt.exists():
        DATA_PATH = str(_alt)

MODEL_DIR = _SCRIPT_DIR / ".lstm_models"
OUTPUT_PATH = str(_SCRIPT_DIR / "lstm_predictions.json")

# ─── Hyperparameters ────────────────────────────────────────
SEQ_LEN = 12          # 12-month sliding window
HIDDEN_SIZE = 128     # LSTM hidden units
NUM_LAYERS = 2        # Stacked LSTM layers
DROPOUT = 0.2         # Dropout between LSTM layers and in FC head
BATCH_SIZE = 64
DEFAULT_EPOCHS = 50
DEFAULT_LR = 0.001
MIN_SERIES_LEN = 24   # Minimum months of data to train on a series

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature columns fed into LSTM at each timestep
FEATURE_NAMES = [
    "price_norm", "month_sin", "month_cos",
    "year_norm", "region_enc", "pt_enc",
]
N_FEATURES = len(FEATURE_NAMES)


# ─── LSTM Model ─────────────────────────────────────────────
class PriceLSTM(nn.Module):
    """2-layer LSTM with FC head for single-step price prediction."""

    def __init__(self, input_size=N_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use only the last timestep's hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)  # (batch,)


# ─── Data Loading & Preprocessing ───────────────────────────
def load_wfp_data():
    """Load and clean WFP Philippines price data."""
    print(f"  Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}\n"
            "Set WFP_DATA_PATH env var or place wfp_food_prices_phl_latest.csv in Website/."
        )
    df = pd.read_csv(DATA_PATH)
    required = ["date", "price", "commodity", "admin1", "pricetype"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0].copy()
    df["region"] = df["admin1"]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} -- {df['date'].max().date()}")
    print(f"  Commodities: {df['commodity'].nunique()}, Regions: {df['region'].nunique()}")
    return df


def build_monthly_series(df):
    """Aggregate daily prices to monthly means per (commodity, region, pricetype).

    Returns a DataFrame with one row per month per series, sorted by date.
    """
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M")
    agg = (
        df.groupby(["commodity", "region", "pricetype", "ym"])
        .agg(price=("price", "mean"))
        .reset_index()
    )
    agg["date"] = agg["ym"].dt.to_timestamp()
    agg["year"] = agg["date"].dt.year
    agg["month"] = agg["date"].dt.month
    agg = agg.sort_values(["commodity", "region", "pricetype", "date"])
    return agg.drop(columns=["ym"])


def encode_and_engineer(monthly_df):
    """Add encoded and cyclical features to monthly data."""
    df = monthly_df.copy()

    # Encode categoricals
    le_region = LabelEncoder()
    df["region_enc"] = le_region.fit_transform(df["region"])
    le_pt = LabelEncoder()
    df["pt_enc"] = le_pt.fit_transform(df["pricetype"])

    # Cyclical month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Year normalized (0-1 range over data span)
    year_min, year_max = df["year"].min(), df["year"].max()
    span = max(year_max - year_min, 1)
    df["year_norm"] = (df["year"] - year_min) / span

    return df, le_region, le_pt


def create_sequences(series_df, scaler, seq_len=SEQ_LEN):
    """Create sliding-window sequences from a single time series.

    Args:
        series_df: DataFrame for one (commodity, region, pricetype) series,
                   sorted by date, with feature columns present.
        scaler: fitted StandardScaler for price normalization.
        seq_len: number of timesteps per input window.

    Returns:
        X: np.ndarray of shape (n_windows, seq_len, n_features)
        y: np.ndarray of shape (n_windows,) — next month's price (original scale)
        dates: list of date strings for each target
    """
    prices = series_df["price"].values
    price_norm = scaler.transform(prices.reshape(-1, 1)).ravel()
    series_df = series_df.copy()
    series_df["price_norm"] = price_norm

    features = series_df[FEATURE_NAMES].values.astype(np.float32)
    dates = series_df["date"].dt.strftime("%Y-%m").tolist()

    X_list, y_list, d_list = [], [], []
    for i in range(len(features) - seq_len):
        X_list.append(features[i : i + seq_len])
        y_list.append(prices[i + seq_len])  # Target: raw price at next step
        d_list.append(dates[i + seq_len] if (i + seq_len) < len(dates) else "")

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(y_list), d_list


# ─── Training ───────────────────────────────────────────────
def train_lstm(
    df, epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, val_months=24, verbose=True,
):
    """Train per-commodity LSTM models.

    Args:
        df: raw WFP DataFrame.
        epochs: training epochs per commodity.
        lr: learning rate.
        val_months: months reserved for validation from the end of data.
        verbose: print progress.

    Returns:
        results dict with metrics and trained models info.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 65)
    print("  LSTM Price Forecasting Pipeline")
    print(f"  Architecture: LSTM({N_FEATURES}, {HIDDEN_SIZE}, layers={NUM_LAYERS}, dropout={DROPOUT})")
    print(f"  Sequence length: {SEQ_LEN} months | Device: {DEVICE}")
    print("=" * 65)

    # 1. Build monthly series
    print("\n[1/4] Building monthly series...")
    monthly = build_monthly_series(df)
    monthly, le_region, le_pt = encode_and_engineer(monthly)

    # 2. Train/val split (same logic as retrain_model.py)
    latest = monthly["date"].max()
    val_start = (latest - pd.DateOffset(months=val_months)).replace(day=1)
    print(f"  Train: up to {(val_start - pd.DateOffset(months=1)).strftime('%b %Y')}")
    print(f"  Validation: {val_start.strftime('%Y-%m')} -- {latest.strftime('%b %Y')}")

    # Group by series
    series_groups = monthly.groupby(["commodity", "region", "pricetype"])
    all_commodities = monthly["commodity"].unique()
    print(f"  Series: {len(series_groups):,} | Commodities: {len(all_commodities)}")

    # 3. Train per commodity (aggregate all regions/pricetypes)
    print(f"\n[2/4] Training LSTM models ({epochs} epochs each)...")
    commodity_groups = monthly.groupby("commodity")

    results = {}
    total_t0 = time.perf_counter()
    trained_count = 0

    for ci, (comm, comm_df) in enumerate(commodity_groups):
        comm_df = comm_df.sort_values("date").reset_index(drop=True)
        if len(comm_df) < MIN_SERIES_LEN + SEQ_LEN:
            continue

        # Price scaler per commodity
        scaler = StandardScaler()
        scaler.fit(comm_df["price"].values.reshape(-1, 1))

        # Split into train/val
        train_part = comm_df[comm_df["date"] < val_start]
        val_part = comm_df[comm_df["date"] >= val_start]

        # Create sequences
        X_train, y_train, _ = create_sequences(train_part, scaler)
        X_val, y_val, val_dates = create_sequences(comm_df, scaler)

        # Filter val sequences: only those whose target date >= val_start
        if len(X_val) > 0 and len(val_dates) > 0:
            val_start_str = val_start.strftime("%Y-%m")
            val_mask = [d >= val_start_str for d in val_dates]
            X_val = X_val[val_mask]
            y_val = y_val[val_mask]
            val_dates = [d for d, m in zip(val_dates, val_mask) if m]

        if len(X_train) < BATCH_SIZE // 2:
            continue

        # Tensors
        X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        # Model
        model = PriceLSTM().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )
        criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers vs MSE

        # Training loop
        t0 = time.perf_counter()
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(X_train)
            scheduler.step(epoch_loss)

            # Early stopping on val loss
            if len(X_val) > 0:
                model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                    v_pred = model(Xv).cpu().numpy()
                    v_loss = np.mean((v_pred - y_val) ** 2)
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break

        elapsed = time.perf_counter() - t0

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # Save model
        model_path = MODEL_DIR / f"lstm_{comm.replace(' ', '_').replace('/', '_')}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "commodity": comm,
            "epochs_trained": epoch + 1,
            "seq_len": SEQ_LEN,
        }, model_path)

        # Validation metrics
        metrics = {"train_time_s": round(elapsed, 2), "epochs": epoch + 1}
        if len(X_val) > 0 and len(y_val) > 0:
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                val_pred = np.maximum(model(Xv).cpu().numpy(), 0)

            mape = mean_absolute_percentage_error(y_val, val_pred) * 100
            mae = mean_absolute_error(y_val, val_pred)
            rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            bias = np.mean((val_pred - y_val) / np.where(y_val > 0, y_val, 1)) * 100

            metrics.update({
                "mape": round(mape, 1),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "bias": round(bias, 1),
                "n_val": len(y_val),
                "val_pred": val_pred.tolist(),
                "val_actual": y_val.tolist(),
                "val_dates": val_dates,
            })

        results[comm] = metrics
        trained_count += 1
        if verbose and (trained_count % 10 == 0 or trained_count <= 3):
            mape_str = f"MAPE {metrics.get('mape', '?')}%" if "mape" in metrics else "no val"
            print(f"  [{trained_count:3d}] {comm[:40]:<40s} {mape_str} ({elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n  Trained {trained_count} commodity models in {total_elapsed:.1f}s")

    # Overall metrics
    all_actual, all_pred = [], []
    for comm, m in results.items():
        if "val_actual" in m:
            all_actual.extend(m["val_actual"])
            all_pred.extend(m["val_pred"])
    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)

    if len(all_actual) > 0:
        overall = {
            "mape": round(mean_absolute_percentage_error(all_actual, all_pred) * 100, 1),
            "mae": round(mean_absolute_error(all_actual, all_pred), 2),
            "rmse": round(float(np.sqrt(np.mean((all_actual - all_pred) ** 2))), 2),
            "bias": round(float(np.mean((all_pred - all_actual) / np.where(all_actual > 0, all_actual, 1)) * 100), 1),
            "r2": round(float(1 - np.sum((all_actual - all_pred) ** 2) / np.sum((all_actual - np.mean(all_actual)) ** 2)), 4),
            "n_val": len(all_actual),
        }
        print(f"\n  Overall:  MAPE {overall['mape']}%  MAE PHP {overall['mae']}"
              f"  RMSE PHP {overall['rmse']}  R² {overall['r2']}")
    else:
        overall = {}

    return {
        "results": results,
        "overall": overall,
        "le_region": le_region,
        "le_pt": le_pt,
        "monthly": monthly,
        "val_start": val_start,
        "train_time_s": round(total_elapsed, 1),
    }


# ─── Forecasting ────────────────────────────────────────────
def generate_forecasts(train_output):
    """Generate forecasts Feb 2026 -- Dec 2027 using trained LSTM models."""
    print("\n[3/4] Generating LSTM forecasts (Feb 2026 -- Dec 2027)...")
    monthly = train_output["monthly"]
    le_region = train_output["le_region"]
    le_pt = train_output["le_pt"]
    results = train_output["results"]
    year_min = monthly["year"].min()
    year_span = max(monthly["year"].max() - year_min, 1)

    forecasts = {}
    fc_count = 0

    for comm, metrics in results.items():
        model_path = MODEL_DIR / f"lstm_{comm.replace(' ', '_').replace('/', '_')}.pt"
        if not model_path.exists():
            continue

        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model = PriceLSTM()
        model.load_state_dict(checkpoint["model_state"])
        model.to(DEVICE)
        model.eval()

        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint["scaler_mean"])
        scaler.scale_ = np.array(checkpoint["scaler_scale"])
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = 1

        # Get last SEQ_LEN months of data for this commodity
        comm_data = monthly[monthly["commodity"] == comm].sort_values("date")
        if len(comm_data) < SEQ_LEN:
            continue

        # Aggregate across regions/pricetypes: use mode for categoricals
        # For forecasting, use the most common region/pt encoding
        region_enc_mode = comm_data["region_enc"].mode().iloc[0]
        pt_enc_mode = comm_data["pt_enc"].mode().iloc[0]

        # Build the seed window from last SEQ_LEN monthly prices
        # Average across regions for the commodity-level forecast
        monthly_avg = (
            comm_data.groupby("date")
            .agg(price=("price", "mean"),
                 month_sin=("month_sin", "first"),
                 month_cos=("month_cos", "first"),
                 year_norm=("year_norm", "first"))
            .reset_index()
            .sort_values("date")
        )
        if len(monthly_avg) < SEQ_LEN:
            continue

        tail = monthly_avg.tail(SEQ_LEN).copy()
        price_seq = tail["price"].values.copy()
        norm_seq = scaler.transform(price_seq.reshape(-1, 1)).ravel()

        comm_forecasts = {}

        for year in (2026, 2027):
            for month in range(1, 13):
                if year == 2026 and month <= 1:
                    continue
                # Build feature vector for current window
                window = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
                for t in range(SEQ_LEN):
                    window[t, 0] = norm_seq[-(SEQ_LEN - t)]
                    # Recalculate cyclical features for shifted window
                    m_idx = (month - SEQ_LEN + t) % 12
                    if m_idx <= 0:
                        m_idx += 12
                    window[t, 1] = math.sin(2 * math.pi * m_idx / 12)
                    window[t, 2] = math.cos(2 * math.pi * m_idx / 12)
                    window[t, 3] = (year - year_min) / year_span
                    window[t, 4] = region_enc_mode
                    window[t, 5] = pt_enc_mode

                # Predict
                with torch.no_grad():
                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    pred_price = max(0.0, float(model(x).cpu().item()))

                # Cap unrealistic forecasts at 5x recent average
                recent_avg = np.mean(price_seq[-12:])
                if pred_price > recent_avg * 5:
                    pred_price = recent_avg * 1.5

                # Slide window: append new price, drop oldest
                price_seq = np.append(price_seq, pred_price)
                norm_seq = np.append(norm_seq, scaler.transform([[pred_price]])[0, 0])

                date_key = f"{year}-{month:02d}"
                comm_forecasts[date_key] = round(pred_price, 2)
                fc_count += 1

        forecasts[comm] = comm_forecasts

    print(f"  Generated {fc_count:,} forecast points for {len(forecasts)} commodities")
    return forecasts


# ─── JSON Output ────────────────────────────────────────────
def build_output_json(train_output, forecasts):
    """Build dashboard-compatible JSON output."""
    print("\n[4/4] Building output JSON...")
    results = train_output["results"]
    overall = train_output["overall"]
    monthly = train_output["monthly"]

    # Per-commodity metrics for commComparison
    comm_comparison = {}
    for comm, m in results.items():
        if "mape" in m:
            comm_comparison[comm] = {
                "mape": m["mape"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "n_val": m["n_val"],
            }

    # Validation trends: {series_key: {date: predicted_price}}
    val_trends = {}
    for comm, m in results.items():
        if "val_pred" in m and "val_dates" in m:
            key = comm
            val_trends[key] = {}
            for i, d in enumerate(m["val_dates"]):
                val_trends[key][d] = round(m["val_pred"][i], 2)

    output = {
        "model": "LSTM",
        "modelDescription": (
            "Long Short-Term Memory (LSTM) recurrent neural network. Uses a "
            "2-layer stacked LSTM with 128 hidden units processing 12-month "
            "sliding windows of price sequences. Features include normalized "
            "price, cyclical month encoding, year trend, and categorical "
            "encodings for region and price type. Trained per-commodity with "
            "AdamW optimizer, Huber loss (robust to outliers), learning rate "
            "scheduling, gradient clipping, and early stopping. Captures "
            "temporal dependencies and seasonality patterns that tree-based "
            "models may miss."
        ),
        "modelColor": "#e11d48",  # Rose-600 — distinct from existing 5 model colors
        "architecture": {
            "type": "LSTM",
            "input_size": N_FEATURES,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "seq_len": SEQ_LEN,
            "optimizer": "AdamW",
            "loss": "HuberLoss",
            "features": FEATURE_NAMES,
        },
        "overall": overall,
        "commComparison": comm_comparison,
        "valTrends": val_trends,
        "forecasts": forecasts,
        "meta": {
            "commodities": sorted(list(results.keys())),
            "trainTime_s": train_output["train_time_s"],
            "device": str(DEVICE),
            "pytorch_version": torch.__version__,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"  Saved: {OUTPUT_PATH} ({size_kb:.0f} KB)")
    return output


# ─── Single Commodity Prediction CLI ────────────────────────
def predict_single(commodity, region=None):
    """Load a trained model and predict for a single commodity."""
    model_path = MODEL_DIR / f"lstm_{commodity.replace(' ', '_').replace('/', '_')}.pt"
    if not model_path.exists():
        print(f"No trained model found for '{commodity}'. Run --train first.")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model = PriceLSTM()
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = 1

    # Load data for this commodity
    df = load_wfp_data()
    monthly = build_monthly_series(df)
    monthly, le_region, le_pt = encode_and_engineer(monthly)

    comm_data = monthly[monthly["commodity"] == commodity]
    if region:
        comm_data = comm_data[comm_data["region"] == region]
    if len(comm_data) < SEQ_LEN:
        print(f"Insufficient data for '{commodity}' (need {SEQ_LEN}+ months, have {len(comm_data)})")
        sys.exit(1)

    comm_data = comm_data.sort_values("date")
    region_enc = comm_data["region_enc"].mode().iloc[0]
    pt_enc = comm_data["pt_enc"].mode().iloc[0]
    year_min = monthly["year"].min()
    year_span = max(monthly["year"].max() - year_min, 1)

    # Average across regions
    avg = (
        comm_data.groupby("date")
        .agg(price=("price", "mean"),
             month_sin=("month_sin", "first"),
             month_cos=("month_cos", "first"),
             year_norm=("year_norm", "first"))
        .reset_index()
        .sort_values("date")
    )
    tail = avg.tail(SEQ_LEN)
    last_date = tail["date"].max()
    price_seq = tail["price"].values.copy()
    norm_seq = scaler.transform(price_seq.reshape(-1, 1)).ravel()

    print(f"\n  Commodity: {commodity}")
    if region:
        print(f"  Region: {region}")
    print(f"  Last data: {last_date.strftime('%Y-%m')}")
    print(f"  Predicting 24 months ahead...\n")
    print(f"  {'Date':<10s}  {'Price (PHP)':>12s}")
    print(f"  {'─' * 24}")

    for i in range(24):
        next_date = last_date + pd.DateOffset(months=i + 1)
        month = next_date.month
        year = next_date.year

        window = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
        for t in range(SEQ_LEN):
            window[t, 0] = norm_seq[-(SEQ_LEN - t)]
            m_idx = (month - SEQ_LEN + t) % 12
            if m_idx <= 0:
                m_idx += 12
            window[t, 1] = math.sin(2 * math.pi * m_idx / 12)
            window[t, 2] = math.cos(2 * math.pi * m_idx / 12)
            window[t, 3] = (year - year_min) / year_span
            window[t, 4] = region_enc
            window[t, 5] = pt_enc

        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = max(0.0, float(model(x).cpu().item()))

        recent_avg = np.mean(price_seq[-12:])
        if pred > recent_avg * 5:
            pred = recent_avg * 1.5

        price_seq = np.append(price_seq, pred)
        norm_seq = np.append(norm_seq, scaler.transform([[pred]])[0, 0])

        print(f"  {next_date.strftime('%Y-%m'):<10s}  PHP {pred:>8.2f}")


# ─── CLI ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LSTM food price forecasting for WFP Philippines data",
    )
    parser.add_argument("--train", action="store_true", help="Train LSTM models")
    parser.add_argument("--predict", action="store_true", help="Generate forecasts")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--commodity", type=str, default=None, help="Predict for a specific commodity")
    parser.add_argument("--region", type=str, default=None, help="Filter by region (used with --commodity)")

    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.print_help()
        print("\nExamples:")
        print("  python lstm_model.py --train --epochs 100 --lr 0.001")
        print("  python lstm_model.py --predict")
        print("  python lstm_model.py --predict --commodity 'Rice (Regular Milled)'")
        print("  python lstm_model.py --train --predict --epochs 50")
        sys.exit(0)

    df = load_wfp_data()
    train_output = None

    if args.train:
        train_output = train_lstm(df, epochs=args.epochs, lr=args.lr)

    if args.predict:
        if args.commodity:
            predict_single(args.commodity, args.region)
        else:
            if train_output is None:
                # Need to train first to get models + metadata
                print("  No training output available. Training with defaults...")
                train_output = train_lstm(df, epochs=args.epochs, lr=args.lr)
            forecasts = generate_forecasts(train_output)
            output = build_output_json(train_output, forecasts)

            # Print summary
            if output.get("overall"):
                o = output["overall"]
                print(f"\n{'=' * 65}")
                print(f"  LSTM Forecasting Complete")
                print(f"  Overall MAPE: {o.get('mape', '?')}%  |  R²: {o.get('r2', '?')}")
                print(f"  Commodities: {len(output['commComparison'])}")
                print(f"  Forecast points: {sum(len(v) for v in output['forecasts'].values()):,}")
                print(f"{'=' * 65}")

    print("\nDone.")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:  # signed: delta
        print("ERROR: PyTorch is required but not installed.")
        print("Install with: pip install torch")
        sys.exit(1)
    main()
