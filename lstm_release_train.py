"""Leakage-safe LSTM retraining for the publication-gated DirectML forecast path.

The final out-of-time months are never used for fitting, target scaling, model
selection, or early stopping. Each commodity is reduced to one chronological
monthly national-average series, matching gpu_forecast_driver.py and preventing
windows from crossing region/price-type series boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from lstm_architecture import BATCH_SIZE, DEFAULT_LR, FEATURE_NAMES, N_FEATURES, PriceLSTM, SEQ_LEN

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "WFP" / "wfp_food_prices_phl_latest.csv"
DEFAULT_MODEL_DIR = ROOT / ".lstm_models_release_20260911"
DEFAULT_RECEIPT = ROOT / "gpu_driver_evidence" / "rerun_20260911" / "lstm_retrain_receipt.json"
TRAINING_CONTRACT = "commodity-monthly-series_train-only-scaler_scaled-target_oot-v1"
SEED = 20260911


class MeanScale:
    """Small StandardScaler-compatible 1-D scaler without a sklearn dependency."""

    def __init__(self) -> None:
        self.mean_ = np.asarray([0.0], dtype=np.float64)
        self.scale_ = np.asarray([1.0], dtype=np.float64)

    def fit(self, values) -> "MeanScale":
        a = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(a) == 0 or not np.isfinite(a).all():
            raise ValueError("Scaler fit requires finite non-empty values")
        scale = float(np.std(a, ddof=0))
        self.mean_ = np.asarray([float(np.mean(a))], dtype=np.float64)
        self.scale_ = np.asarray([scale if scale > 1e-12 else 1.0], dtype=np.float64)
        return self

    def transform(self, values) -> np.ndarray:
        a = np.asarray(values, dtype=np.float64)
        return (a - self.mean_[0]) / self.scale_[0]

    def inverse_transform(self, values) -> np.ndarray:
        a = np.asarray(values, dtype=np.float64)
        return a * self.scale_[0] + self.mean_[0]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    y = np.asarray(actual, dtype=np.float64).reshape(-1)
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        return {}
    nonzero = np.abs(y) > 1e-12
    return {
        "mape": round(float(np.mean(np.abs((y[nonzero] - p[nonzero]) / y[nonzero])) * 100), 4),
        "mae": round(float(np.mean(np.abs(y - p))), 4),
        "rmse": round(float(np.sqrt(np.mean((y - p) ** 2))), 4),
        "n": int(len(y)),
    }


def load_and_engineer(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(data_path)
    required = {"date", "price", "commodity", "admin1", "pricetype", "unit", "currency"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"WFP data is missing release-required columns: {missing}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw.dropna(subset=list(required)).copy()
    raw = raw[raw["price"] > 0].copy()
    raw["month_period"] = raw["date"].dt.to_period("M")

    currencies = sorted(raw["currency"].astype(str).unique().tolist())
    if currencies != ["PHP"]:
        raise ValueError(f"Unexpected source currency contract: {currencies}")
    unit_counts = raw.groupby("commodity", observed=True)["unit"].nunique()
    mixed_units = sorted(unit_counts[unit_counts != 1].index.astype(str).tolist())
    if mixed_units:
        raise ValueError(f"Commodities with mixed units cannot be release-trained: {mixed_units}")

    monthly = (
        raw.groupby(["commodity", "admin1", "pricetype", "month_period"], observed=True)
        .agg(price=("price", "mean"))
        .reset_index()
    )
    monthly["date"] = monthly["month_period"].dt.to_timestamp()
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    region_map = {name: i for i, name in enumerate(sorted(monthly["admin1"].unique()))}
    pt_map = {name: i for i, name in enumerate(sorted(monthly["pricetype"].unique()))}
    monthly["region_enc"] = monthly["admin1"].map(region_map).astype(float)
    monthly["pt_enc"] = monthly["pricetype"].map(pt_map).astype(float)
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)
    year_min = int(monthly["year"].min())
    year_span = max(int(monthly["year"].max() - year_min), 1)
    monthly["year_norm"] = (monthly["year"] - year_min) / year_span

    summary = {
        "path": str(data_path),
        "sha256": sha256_file(data_path),
        "rows": int(len(raw)),
        "min_date": raw["date"].min().date().isoformat(),
        "max_date": raw["date"].max().date().isoformat(),
        "commodities": int(raw["commodity"].nunique()),
        "regions": int(raw["admin1"].nunique()),
        "price_types": int(raw["pricetype"].nunique()),
        "currency_values": currencies,
        "unit_values": sorted(raw["unit"].astype(str).unique().tolist()),
        "mixed_unit_commodity_count": len(mixed_units),
        "year_min": year_min,
        "year_span": year_span,
    }
    return raw, monthly, summary


def commodity_monthly_series(monthly: pd.DataFrame, commodity: str, cutoff: pd.Period) -> pd.DataFrame:
    rows = monthly[
        (monthly["commodity"] == commodity)
        & (monthly["date"].dt.to_period("M") <= cutoff)
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    region_mode = float(rows["region_enc"].mode().iloc[0])
    pt_mode = float(rows["pt_enc"].mode().iloc[0])
    result = (
        rows.groupby("date", observed=True)
        .agg(
            price=("price", "mean"),
            month_sin=("month_sin", "first"),
            month_cos=("month_cos", "first"),
            year_norm=("year_norm", "first"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    result["region_enc"] = region_mode
    result["pt_enc"] = pt_mode
    return result


def create_sequences(
    series: pd.DataFrame,
    scaler: MeanScale,
    *,
    target_start: pd.Period | None = None,
    target_end: pd.Period | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if series.empty:
        return np.empty((0, SEQ_LEN, N_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    frame = series.sort_values("date").copy()
    normalized = scaler.transform(frame["price"].to_numpy(dtype=np.float64)).reshape(-1)
    frame["price_norm"] = normalized
    features = frame[FEATURE_NAMES].to_numpy(dtype=np.float32)
    periods = frame["date"].dt.to_period("M").tolist()
    X, y, dates = [], [], []
    for i in range(len(frame) - SEQ_LEN):
        target_index = i + SEQ_LEN
        period = periods[target_index]
        if target_start is not None and period < target_start:
            continue
        if target_end is not None and period > target_end:
            continue
        X.append(features[i:target_index])
        y.append(float(normalized[target_index]))
        dates.append(str(period))
    if not X:
        return np.empty((0, SEQ_LEN, N_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.float32), dates


def train_release_models(
    *,
    data_path: Path = DEFAULT_DATA,
    epochs: int = 50,
    lr: float = DEFAULT_LR,
    publication_cutoff: str | None = None,
    internal_val_months: int = 6,
    model_dir: Path = DEFAULT_MODEL_DIR,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    raw, monthly, data_summary = load_and_engineer(data_path)
    latest_period = raw["date"].max().to_period("M")
    cutoff = pd.Period(publication_cutoff, freq="M") if publication_cutoff else latest_period - 3
    if cutoff >= latest_period:
        raise ValueError(f"publication cutoff {cutoff} must precede dataset maximum {latest_period}")
    if internal_val_months < 2:
        raise ValueError("internal_val_months must be >= 2")
    internal_start = cutoff - (internal_val_months - 1)
    train_end = internal_start - 1

    model_dir.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    total_started = time.perf_counter()
    commodities = sorted(monthly["commodity"].unique().tolist())
    for index, commodity in enumerate(commodities):
        series = commodity_monthly_series(monthly, commodity, cutoff)
        if len(series) < SEQ_LEN + internal_val_months + 12:
            continue
        fit_rows = series[series["date"].dt.to_period("M") <= train_end]
        if len(fit_rows) < SEQ_LEN + 12:
            continue
        scaler = MeanScale().fit(fit_rows["price"].to_numpy(dtype=np.float64))
        X_train, y_train, train_dates = create_sequences(series, scaler, target_end=train_end)
        X_val, y_val, val_dates = create_sequences(series, scaler, target_start=internal_start, target_end=cutoff)
        if len(X_train) < 24 or len(X_val) < 2:
            continue

        commodity_seed = SEED + index
        torch.manual_seed(commodity_seed)
        generator = torch.Generator().manual_seed(commodity_seed)
        model = PriceLSTM().to("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.HuberLoss(delta=0.5)
        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        loader = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_ds)), shuffle=True, generator=generator)
        Xv = torch.tensor(X_val, dtype=torch.float32)

        best_state = None
        best_val = float("inf")
        patience = 0
        epochs_trained = 0
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(xb)
                loss = criterion(prediction, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_scaled = model(Xv).numpy()
            val_loss = float(np.mean((val_scaled - y_val) ** 2))
            epochs_trained = epoch + 1
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 8:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_scaled = model(Xv).numpy()
        val_pred = scaler.inverse_transform(val_scaled)
        val_actual = scaler.inverse_transform(y_val)
        naive_value = float(fit_rows.iloc[-1]["price"])
        naive_pred = np.full_like(val_actual, naive_value, dtype=np.float64)

        source_rows = raw[raw["commodity"] == commodity]
        unit = str(source_rows["unit"].iloc[0])
        currency = str(source_rows["currency"].iloc[0])
        model_path = model_dir / f"lstm_{commodity.replace(' ', '_').replace('/', '_')}.pt"
        checkpoint = {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "commodity": commodity,
            "epochs_trained": epochs_trained,
            "seq_len": SEQ_LEN,
            "target_scaled": True,
            "training_cutoff": str(cutoff),
            "internal_validation_start": str(internal_start),
            "internal_validation_end": str(cutoff),
            "scaler_fit_end": str(train_end),
            "training_contract": TRAINING_CONTRACT,
            "series_contract": "one commodity-level chronological monthly series",
            "unit": unit,
            "currency": currency,
            "seed": commodity_seed,
        }
        torch.save(checkpoint, model_path)
        results.append({
            "commodity": commodity,
            "unit": unit,
            "currency": currency,
            "checkpoint": str(model_path),
            "checkpoint_sha256": sha256_file(model_path),
            "train_sequence_count": int(len(X_train)),
            "internal_validation_count": int(len(X_val)),
            "train_target_start": train_dates[0],
            "train_target_end": train_dates[-1],
            "internal_validation_start": val_dates[0],
            "internal_validation_end": val_dates[-1],
            "epochs_trained": epochs_trained,
            "train_time_s": round(time.perf_counter() - started, 3),
            "internal_model": regression_metrics(val_actual, val_pred),
            "internal_naive_persistence": regression_metrics(val_actual, naive_pred),
        })

    receipt = {
        "schema": "ph-food-price-lstm-release-training-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_contract": TRAINING_CONTRACT,
        "seed": SEED,
        "training_device": "cpu",
        "data": data_summary,
        "partition": {
            "scaler_and_fit_targets_end": str(train_end),
            "internal_validation_start": str(internal_start),
            "internal_validation_end": str(cutoff),
            "publication_test_start": str(cutoff + 1),
            "publication_test_end": str(latest_period),
            "publication_test_used_for_training_or_early_stopping": False,
        },
        "target_scaled": True,
        "series_contract": "one chronological commodity-level monthly series; no cross-series windows",
        "model_count": len(results),
        "model_dir": str(model_dir),
        "wall_time_s": round(time.perf_counter() - total_started, 3),
        "models": results,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--publication-cutoff", default=None)
    parser.add_argument("--internal-val-months", type=int, default=6)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    receipt = train_release_models(
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        publication_cutoff=args.publication_cutoff,
        internal_val_months=args.internal_val_months,
        model_dir=args.model_dir,
        receipt_path=args.receipt,
    )
    print(json.dumps({
        "training_contract": receipt["training_contract"],
        "model_count": receipt["model_count"],
        "partition": receipt["partition"],
        "data": receipt["data"],
        "receipt": str(args.receipt.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
