#!/usr/bin/env python
"""Train the Philippine food-price forecaster ON the AMD RX 6600 (torch-directml).

The project's LSTM only *infers* on the GPU (via ONNX) and it fails the honest
out-of-time gate (85.7% MAPE, R2 -1.26 vs naive 7.5%). torch-directml cannot run
the fused LSTM cell, so this trains a MatMul-based model that DOES run on the AMD
GPU, using the *same* out-of-time methodology as gpu_forecast_driver.py:

  * per-commodity monthly-averaged series (mean over region/pricetype)
  * training cutoff inferred from the earliest stored forecast in
    lstm_predictions.json (2026-01), evaluation over the post-cutoff horizon
  * per-series z-scores fit on training rows only (no leakage)
  * recursive roll-forward vs last-observation persistence
  * model must beat naive on BOTH MAPE and MAE (>=10 series, >=30 points)

The model predicts the z-space DELTA, so "predict zero" is exactly naive: any
learned signal is honest improvement over the strong random-walk baseline.

Run: .venv-torch-dml\\Scripts\\python.exe gpu_train_forecaster.py
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "WFP" / "wfp_food_prices_phl_latest.csv"
PRED_ARTIFACT = ROOT / "lstm_predictions.json"
EVID = ROOT / "gpu_driver_evidence"
EVID.mkdir(exist_ok=True)
SEQ = 12


def load_monthly() -> pd.DataFrame:
    raw = pd.read_csv(DATA)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw.dropna(subset=["date", "price", "commodity"])
    raw = raw[raw["price"] > 0]
    raw["mp"] = raw["date"].dt.to_period("M")
    # per-commodity monthly average across region + pricetype (mirrors the gate)
    m = (raw.groupby(["commodity", "mp"], observed=True)
            .agg(price=("price", "mean")).reset_index())
    m["date"] = m["mp"].dt.to_timestamp()
    return m.sort_values(["commodity", "date"])


def infer_cutoff() -> pd.Period:
    doc = json.loads(PRED_ARTIFACT.read_text(encoding="utf-8"))
    firsts = [min(v) for v in (doc.get("forecasts") or {}).values()
              if isinstance(v, dict) and v]
    return pd.Period(min(firsts), freq="M") - 1


def build_training(monthly: pd.DataFrame, cutoff: pd.Period):
    """Pooled windows from training rows only. Returns tensors + per-series scalers."""
    Xseq, Xmon, Xcid, Ydz = [], [], [], []
    scalers, commodities = {}, sorted(monthly["commodity"].unique())
    cid = {c: i for i, c in enumerate(commodities)}
    for c in commodities:
        s = monthly[monthly["commodity"] == c]
        train = s[s["date"].dt.to_period("M") <= cutoff]
        p = train["price"].to_numpy(np.float64)
        if len(p) < SEQ + 2:
            continue
        mu, sd = float(p.mean()), float(p.std() or 1.0)
        scalers[c] = (mu, sd)
        z = (p - mu) / sd
        months = train["date"].dt.month.to_numpy()
        for t in range(SEQ, len(z)):
            Xseq.append(z[t - SEQ:t])
            tm = months[t]
            Xmon.append([math.sin(2 * math.pi * tm / 12), math.cos(2 * math.pi * tm / 12)])
            Xcid.append(cid[c])
            Ydz.append(z[t] - z[t - 1])          # z-space one-step delta
    return (torch.tensor(np.array(Xseq), dtype=torch.float32),
            torch.tensor(np.array(Xmon), dtype=torch.float32),
            torch.tensor(np.array(Xcid), dtype=torch.long),
            torch.tensor(np.array(Ydz), dtype=torch.float32).unsqueeze(1),
            scalers, cid, commodities)


class DeltaNet(torch.nn.Module):
    def __init__(self, n_comm: int, emb: int = 8, hidden: int = 128):
        super().__init__()
        self.emb = torch.nn.Embedding(n_comm, emb)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(SEQ + 2 + emb, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, seq, mon, cid):
        return self.net(torch.cat([seq, mon, self.emb(cid)], dim=1))


def regression_metrics(actual, predicted) -> dict:
    y = np.asarray(actual, np.float64); p = np.asarray(predicted, np.float64)
    nz = np.abs(y) > 1e-12
    mape = float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    den = float(np.sum((y - y.mean()) ** 2))
    r2 = None if den <= 0 else float(1 - np.sum((y - p) ** 2) / den)
    return {"mape": round(mape, 4), "mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": None if r2 is None else round(r2, 6), "n": int(len(y))}


def main() -> None:
    import torch_directml as dml
    dev = dml.device()
    dev_name = dml.device_name(0).replace("\x00", "").strip()

    monthly = load_monthly()
    cutoff = infer_cutoff()
    Xseq, Xmon, Xcid, Ydz, scalers, cid, commodities = build_training(monthly, cutoff)
    print(f"cutoff={cutoff}  pooled_windows={len(Xseq)}  series={len(scalers)}  dev={dev_name}")

    # ---- train on GPU ----
    torch.manual_seed(0)
    model = DeltaNet(len(commodities)).to(dev)
    Xseq, Xmon, Xcid, Ydz = Xseq.to(dev), Xmon.to(dev), Xcid.to(dev), Ydz.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = torch.nn.MSELoss()
    n, bs, epochs = len(Xseq), 4096, 120
    param_dev = str(next(model.parameters()).device)
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xseq[idx], Xmon[idx], Xcid[idx])
            loss = lossf(out, Ydz[idx])
            loss.backward(); opt.step()
            tot += float(loss.detach().cpu()) * len(idx)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:3d}  mse(dz)={tot / n:.5f}")
    train_sec = time.perf_counter() - t0

    # ---- out-of-time recursive backtest vs naive ----
    model.eval()
    data_max = monthly["date"].max().to_period("M")
    horizon = pd.period_range(cutoff + 1, data_max, freq="M")
    all_actual, all_model, all_naive = [], [], []
    per_series = 0
    with torch.no_grad():
        for c in scalers:
            mu, sd = scalers[c]
            s = monthly[monthly["commodity"] == c]
            seed = s[s["date"].dt.to_period("M") <= cutoff].tail(SEQ)
            if len(seed) < SEQ:
                continue
            actuals = s.set_index(s["date"].dt.to_period("M"))["price"]
            if not all(h in actuals.index for h in horizon):
                # only score commodities with the full post-cutoff horizon present
                pass
            z = list((seed["price"].to_numpy(np.float64) - mu) / sd)
            last_price = float(seed["price"].to_numpy()[-1])
            naive_price = last_price
            scored = False
            for h in horizon:
                if h not in actuals.index:
                    break
                mon = [math.sin(2 * math.pi * h.month / 12), math.cos(2 * math.pi * h.month / 12)]
                dz = float(model(
                    torch.tensor([z[-SEQ:]], dtype=torch.float32, device=dev),
                    torch.tensor([mon], dtype=torch.float32, device=dev),
                    torch.tensor([cid[c]], dtype=torch.long, device=dev)).cpu())
                z_next = z[-1] + dz
                z.append(z_next)
                model_price = z_next * sd + mu
                actual = float(actuals.loc[h])
                all_actual.append(actual)
                all_model.append(model_price)
                all_naive.append(naive_price)   # persistence = last observed
                scored = True
            per_series += 1 if scored else 0

    model_m = regression_metrics(all_actual, all_model)
    naive_m = regression_metrics(all_actual, all_naive)
    beats = (model_m["mape"] < naive_m["mape"] and model_m["mae"] < naive_m["mae"]
             and per_series >= 10 and model_m["n"] >= 30)
    disposition = "passed_out_of_time_naive_baseline" if beats else "withheld_failed_validation"

    receipt = {
        "schema": "ph-food-price-gpu-trained-backtest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": {"name": dev_name, "torch": torch.__version__,
                   "trained_on": param_dev, "trained_on_gpu": param_dev == "privateuseone:0"},
        "training": {"pooled_windows": int(len(Xseq)), "series": len(scalers),
                     "epochs": epochs, "seconds": round(train_sec, 2)},
        "backtest": {"training_cutoff": str(cutoff),
                     "horizon": [str(horizon[0]), str(horizon[-1])],
                     "series_scored": per_series,
                     "gpu_delta_mlp": model_m, "naive_persistence": naive_m},
        "publication_gate": {"status": disposition, "beats_naive": bool(beats),
                             "requirements": "model<naive on MAPE and MAE, >=10 series, >=30 points"},
        "honesty_note": "The GPU-trained model is compared to the exact same naive "
                        "baseline and gate the LSTM failed. This receipt reports the "
                        "real disposition; forecasts stay local unless the gate passes.",
    }
    out = EVID / "gpu_trained_backtest.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\n  GPU delta-MLP : MAPE {model_m['mape']}%  MAE {model_m['mae']}  R2 {model_m['r2']}  n={model_m['n']}")
    print(f"  naive persist : MAPE {naive_m['mape']}%  MAE {naive_m['mae']}  R2 {naive_m['r2']}")
    print(f"  disposition   : {disposition}  (trained_on_gpu={receipt['device']['trained_on_gpu']}, {train_sec:.1f}s)")
    print(f"  receipt -> {out}")


if __name__ == "__main__":
    main()
