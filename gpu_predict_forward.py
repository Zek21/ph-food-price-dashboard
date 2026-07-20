#!/usr/bin/env python
"""Forward food-price forecasts from the GPU-trained model (publication-eligible).

The delta-MLP passed the out-of-time naive-baseline gate (see
gpu_trained_backtest.json), so unlike the withheld LSTM its forecasts may be
published. This retrains on ALL data through the dataset maximum and rolls each
commodity forward, emitting a dashboard-style predictions artifact plus a hashed
receipt that carries the passing gate disposition.

Run: .venv-torch-dml\\Scripts\\python.exe gpu_predict_forward.py
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import torch

from gpu_train_forecaster import (DATA, EVID, PRED_ARTIFACT, ROOT, SEQ, DeltaNet,
                                  build_training, load_monthly, regression_metrics)

FORWARD_MONTHS = 18  # Jul 2026 .. Dec 2027


def _sha256(path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    import torch_directml as dml
    dev = dml.device()
    dev_name = dml.device_name(0).replace("\x00", "").strip()

    monthly = load_monthly()
    data_max = monthly["date"].max().to_period("M")
    # Train on EVERYTHING up to the dataset maximum (best forward model).
    Xseq, Xmon, Xcid, Ydz, scalers, cid, commodities = build_training(monthly, data_max)
    print(f"forward-train: windows={len(Xseq)} series={len(scalers)} data_max={data_max} dev={dev_name}")

    torch.manual_seed(0)
    model = DeltaNet(len(commodities)).to(dev)
    Xseq, Xmon, Xcid, Ydz = Xseq.to(dev), Xmon.to(dev), Xcid.to(dev), Ydz.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.MSELoss()
    n, bs, epochs = len(Xseq), 4096, 140
    param_dev = str(next(model.parameters()).device)
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xseq[idx], Xmon[idx], Xcid[idx]), Ydz[idx])
            loss.backward(); opt.step()
    train_sec = time.perf_counter() - t0

    horizon = [data_max + k for k in range(1, FORWARD_MONTHS + 1)]
    forecasts: dict[str, dict] = {}
    model.eval()
    with torch.no_grad():
        for c in scalers:
            mu, sd = scalers[c]
            s = monthly[monthly["commodity"] == c]
            seed = s[s["date"].dt.to_period("M") <= data_max].tail(SEQ)
            if len(seed) < SEQ:
                continue
            z = list((seed["price"].to_numpy(np.float64) - mu) / sd)
            series = {}
            for h in horizon:
                mon = [math.sin(2 * math.pi * h.month / 12), math.cos(2 * math.pi * h.month / 12)]
                dz = float(model(
                    torch.tensor([z[-SEQ:]], dtype=torch.float32, device=dev),
                    torch.tensor([mon], dtype=torch.float32, device=dev),
                    torch.tensor([cid[c]], dtype=torch.long, device=dev)).cpu())
                z.append(z[-1] + dz)
                series[str(h)] = round(z[-1] * sd + mu, 4)
            forecasts[c] = series

    # carry the passing gate disposition from the backtest receipt
    backtest = {}
    bt_path = EVID / "gpu_trained_backtest.json"
    if bt_path.exists():
        backtest = json.loads(bt_path.read_text(encoding="utf-8"))

    predictions = {
        "schema": "ph-food-price-gpu-forward-predictions-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": "GPU-trained delta-MLP (torch-directml)",
            "device": dev_name, "trained_on": param_dev,
            "trained_on_gpu": param_dev == "privateuseone:0",
            "train_seconds": round(train_sec, 2), "epochs": epochs,
            "architecture": "per-series z-score + 12-month window + commodity "
                            "embedding -> 2x128 ReLU MLP, predicts z-space delta",
        },
        "data": {"source": str(DATA.name), "sha256": _sha256(DATA),
                 "max_month": str(data_max)},
        "validation": {
            "method": "out-of-time recursive backtest vs naive persistence",
            "cutoff": backtest.get("backtest", {}).get("training_cutoff"),
            "horizon": backtest.get("backtest", {}).get("horizon"),
            "gpu_delta_mlp": backtest.get("backtest", {}).get("gpu_delta_mlp"),
            "naive_persistence": backtest.get("backtest", {}).get("naive_persistence"),
            "disposition": backtest.get("publication_gate", {}).get("status"),
            "beats_naive": backtest.get("publication_gate", {}).get("beats_naive"),
        },
        "forecast_horizon": [str(horizon[0]), str(horizon[-1])],
        "commodities": len(forecasts),
        "forecasts": forecasts,
        "disclaimer": "Experimental research forecasts, not financial advice. "
                      "Model passed the out-of-time naive-baseline gate on "
                      f"{backtest.get('backtest', {}).get('horizon')}; future "
                      "accuracy is not guaranteed.",
    }
    out = ROOT / "gpu_forward_predictions.json"
    out.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    # a small human-facing sample for the blog
    sample_keys = [c for c in forecasts if "rice" in c.lower()][:1] + list(forecasts)[:4]
    print(f"\n  wrote {len(forecasts)} commodity forecasts -> {out.name}")
    print(f"  disposition carried: {predictions['validation']['disposition']}  "
          f"(trained_on_gpu={predictions['model']['trained_on_gpu']}, {train_sec:.1f}s)")
    for c in dict.fromkeys(sample_keys):
        first = list(forecasts[c])[0]; last = list(forecasts[c])[-1]
        print(f"    {c[:38]:<38} {first}: {forecasts[c][first]:>8}  ->  {last}: {forecasts[c][last]:>8}")


if __name__ == "__main__":
    main()
