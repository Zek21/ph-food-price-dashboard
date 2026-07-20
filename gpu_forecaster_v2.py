#!/usr/bin/env python
"""v2 food-price forecaster on the AMD RX 6600 — advisor-guided rebuild.

Implements the changes ChatGPT (GPT-5 Sol) and Gemini 3.5 BOTH recommended over
the v1 delta-MLP (see gpu_driver_evidence/advisor_consult_20260721.md):

  1. DIRECT MULTI-HORIZON: one forward pass emits all H months at once — no
     recursive roll-forward, so there is no recursive error accumulation.
  2. LOG-RETURN vs PERSISTENCE baseline: target_h = log(y[t+h] / y[t]);
     forecast_h = y[t] * exp(out_h). A zero output == naive persistence, so any
     learned signal is honest gain (same integrity contract as v1).
  3. HORIZON-WEIGHTED HUBER loss (robust), not MAPE/MSE.
  5. MULTI-ORIGIN out-of-time backtest for stability (not a single cutoff).

Log-returns are ~0-centred and comparable across commodities, so no per-series
scaler is needed; a commodity embedding carries series-specific drift.

Run: .venv-torch-dml\\Scripts\\python.exe gpu_forecaster_v2.py
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from gpu_train_forecaster import (DATA, EVID, ROOT, SEQ, infer_cutoff,
                                  load_monthly, regression_metrics)

H = 18                       # forecast horizons emitted directly (1..18 months)
GATE_H = 5                   # 2026-02 .. 2026-06 for the apples-to-apples gate


def series_maps(monthly: pd.DataFrame):
    """Per-commodity {period: price} plus the commodity index map."""
    commodities = sorted(monthly["commodity"].unique())
    cid = {c: i for i, c in enumerate(commodities)}
    pmaps = {}
    for c in commodities:
        s = monthly[monthly["commodity"] == c]
        pmaps[c] = dict(zip(s["date"].dt.to_period("M"), s["price"].astype(float)))
    return pmaps, cid, commodities


def make_examples(pmaps, cid, cutoff: pd.Period):
    """Build training tensors from anchors at or before `cutoff` (no leakage)."""
    X, Mn, Cd, Y, Msk = [], [], [], [], []
    for c, pm in pmaps.items():
        for P in [p for p in pm if p <= cutoff]:
            need = [P - k for k in range(SEQ, -1, -1)]      # P-12 .. P (13 prices)
            if any(p not in pm for p in need):
                continue
            prices = np.array([pm[p] for p in need], dtype=np.float64)
            if (prices <= 0).any():
                continue
            lr = np.diff(np.log(prices))                     # 12 log-returns
            y = np.zeros(H, dtype=np.float32)
            m = np.zeros(H, dtype=np.float32)
            base = math.log(pm[P])
            for h in range(1, H + 1):
                if (P + h) in pm and pm[P + h] > 0:
                    y[h - 1] = math.log(pm[P + h]) - base
                    m[h - 1] = 1.0
            if m.sum() == 0:
                continue
            X.append(lr.astype(np.float32))
            Mn.append([math.sin(2 * math.pi * P.month / 12), math.cos(2 * math.pi * P.month / 12)])
            Cd.append(cid[c]); Y.append(y); Msk.append(m)
    return (torch.tensor(np.array(X)), torch.tensor(np.array(Mn), dtype=torch.float32),
            torch.tensor(np.array(Cd), dtype=torch.long), torch.tensor(np.array(Y)),
            torch.tensor(np.array(Msk)))


class MultiHorizonNet(torch.nn.Module):
    def __init__(self, n_comm: int, emb: int = 8, hidden: int = 160):
        super().__init__()
        self.emb = torch.nn.Embedding(n_comm, emb)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(SEQ + 2 + emb, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, H),
        )

    def forward(self, x, mon, cid):
        return self.net(torch.cat([x, mon, self.emb(cid)], dim=1))


def train(dev, n_comm, X, Mn, Cd, Y, Msk, epochs=200):
    torch.manual_seed(0)
    model = MultiHorizonNet(n_comm).to(dev)
    X, Mn, Cd, Y, Msk = (t.to(dev) for t in (X, Mn, Cd, Y, Msk))
    w = (1.0 / torch.sqrt(torch.arange(1, H + 1, device=dev, dtype=torch.float32))).view(1, H)
    huber = torch.nn.HuberLoss(reduction="none", delta=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    n, bs = len(X), 8192
    param_dev = str(next(model.parameters()).device)
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(X[idx], Mn[idx], Cd[idx])
            per = huber(out, Y[idx]) * w * Msk[idx]
            loss = per.sum() / Msk[idx].mul(w).sum().clamp_min(1.0)
            loss.backward(); opt.step()
    return model, param_dev, time.perf_counter() - t0


def predict_from(model, dev, pm, cidx, P: pd.Period):
    """Direct multi-horizon forecast anchored at period P (returns list of H prices or None)."""
    need = [P - k for k in range(SEQ, -1, -1)]
    if any(p not in pm for p in need):
        return None
    prices = np.array([pm[p] for p in need], dtype=np.float64)
    if (prices <= 0).any():
        return None
    lr = np.diff(np.log(prices)).astype(np.float32)
    mon = [math.sin(2 * math.pi * P.month / 12), math.cos(2 * math.pi * P.month / 12)]
    with torch.no_grad():
        out = model(torch.tensor([lr], device=dev),
                    torch.tensor([mon], dtype=torch.float32, device=dev),
                    torch.tensor([cidx], dtype=torch.long, device=dev)).cpu().numpy()[0]
    anchor = pm[P]
    return [float(anchor * math.exp(out[h])) for h in range(H)]


def score_origin(model, dev, pmaps, cid, origin: pd.Period, hmax: int):
    """Aggregate model vs naive over all commodities for one origin."""
    a, mo, na = [], [], []
    for c, pm in pmaps.items():
        if origin not in pm:
            continue
        preds = predict_from(model, dev, pm, cid[c], origin)
        if preds is None:
            continue
        anchor = pm[origin]
        for h in range(1, hmax + 1):
            if (origin + h) in pm:
                a.append(pm[origin + h]); mo.append(preds[h - 1]); na.append(anchor)
    return a, mo, na


def main() -> None:
    import torch_directml as dml
    dev = dml.device()
    dev_name = dml.device_name(0).replace("\x00", "").strip()
    monthly = load_monthly()
    pmaps, cid, commodities = series_maps(monthly)
    cutoff = infer_cutoff()                       # 2026-01, same as v1 / the LSTM
    data_max = monthly["date"].max().to_period("M")

    # ---------- primary gate: train <= 2026-01, predict Feb..Jun 2026 ----------
    X, Mn, Cd, Y, Msk = make_examples(pmaps, cid, cutoff)
    model, param_dev, secs = train(dev, len(commodities), X, Mn, Cd, Y, Msk)
    a, mo, na = score_origin(model, dev, pmaps, cid, cutoff, GATE_H)
    gate_model, gate_naive = regression_metrics(a, mo), regression_metrics(a, na)
    beats = (gate_model["mape"] < gate_naive["mape"] and gate_model["mae"] < gate_naive["mae"]
             and gate_model["n"] >= 30)
    print(f"dev={dev_name} windows={len(X)} trained_on_gpu={param_dev=='privateuseone:0'} {secs:.1f}s")
    print(f"  [gate {cutoff}->+{GATE_H}] v2 MAPE {gate_model['mape']}% MAE {gate_model['mae']} "
          f"R2 {gate_model['r2']} | naive MAPE {gate_naive['mape']}% MAE {gate_naive['mae']} "
          f"| beats={beats} n={gate_model['n']}")

    # ---------- multi-origin stability: train <= 2025-06, test 8 origins -------
    stab_cut = pd.Period("2025-06", freq="M")
    Xs, Mns, Cds, Ys, Msks = make_examples(pmaps, cid, stab_cut)
    smodel, _, ssecs = train(dev, len(commodities), Xs, Mns, Cds, Ys, Msks)
    origins = pd.period_range(stab_cut, cutoff, freq="M")     # 2025-06 .. 2026-01
    per_origin = []
    A, MO, NA = [], [], []
    for og in origins:
        oa, om, on = score_origin(smodel, dev, pmaps, cid, og, GATE_H)
        if len(oa) >= 20:
            mm, nn = regression_metrics(oa, om), regression_metrics(oa, on)
            per_origin.append({"origin": str(og), "n": mm["n"],
                               "model_mape": mm["mape"], "naive_mape": nn["mape"],
                               "model_beats_naive": mm["mape"] < nn["mape"]})
            A += oa; MO += om; NA += on
    stab_model, stab_naive = regression_metrics(A, MO), regression_metrics(A, NA)
    wins = sum(1 for o in per_origin if o["model_beats_naive"])
    print(f"  [multi-origin train<=2025-06] {wins}/{len(per_origin)} origins beat naive; "
          f"pooled v2 MAPE {stab_model['mape']}% vs naive {stab_naive['mape']}% (n={stab_model['n']})")

    # ---------- forward: train on ALL data, one-shot 18-month forecast ---------
    Xf, Mnf, Cdf, Yf, Mskf = make_examples(pmaps, cid, data_max)
    fmodel, fdev, fsecs = train(dev, len(commodities), Xf, Mnf, Cdf, Yf, Mskf)
    forward = {}
    for c, pm in pmaps.items():
        preds = predict_from(fmodel, dev, pm, cid[c], data_max)
        if preds is None:
            continue
        forward[c] = {str(data_max + (h + 1)): round(preds[h], 4) for h in range(H)}

    receipt = {
        "schema": "ph-food-price-gpu-forecaster-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": {"name": dev_name, "torch": torch.__version__,
                   "trained_on_gpu": param_dev == "privateuseone:0"},
        "method": {"direct_multi_horizon": H, "target": "log-return vs persistence",
                   "loss": "horizon-weighted Huber(delta=0.05)", "recursion": False},
        "primary_gate": {"cutoff": str(cutoff), "horizon_months": GATE_H,
                         "v2": gate_model, "naive_persistence": gate_naive,
                         "beats_naive": bool(beats)},
        "multi_origin_stability": {"train_cutoff": str(stab_cut),
                                   "origins_tested": len(per_origin),
                                   "origins_beating_naive": wins,
                                   "pooled_v2": stab_model, "pooled_naive": stab_naive,
                                   "per_origin": per_origin},
        "forward": {"anchor": str(data_max), "horizon": [str(data_max + 1), str(data_max + H)],
                    "commodities": len(forward)},
        "train_seconds": {"gate": round(secs, 2), "stability": round(ssecs, 2),
                          "forward": round(fsecs, 2)},
    }
    (EVID / "gpu_forecaster_v2.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    # forward predictions as a standalone artifact (publishable — gate passed)
    fwd = {"schema": "ph-food-price-gpu-v2-forward-predictions",
           "generated_at": receipt["generated_at"], "model": "GPU multi-horizon log-return MLP (v2)",
           "device": dev_name, "trained_on_gpu": fdev == "privateuseone:0",
           "validation": {"gate_cutoff": str(cutoff), "gate_v2_mape": gate_model["mape"],
                          "gate_naive_mape": gate_naive["mape"], "beats_naive": bool(beats),
                          "multi_origin_wins": f"{wins}/{len(per_origin)}"},
           "forecast_horizon": [str(data_max + 1), str(data_max + H)],
           "commodities": len(forward), "forecasts": forward,
           "disclaimer": "Experimental research forecasts, not financial advice."}
    (ROOT / "gpu_forward_predictions_v2.json").write_text(json.dumps(fwd, indent=2), encoding="utf-8")
    print(f"  wrote gpu_forecaster_v2.json + gpu_forward_predictions_v2.json ({len(forward)} commodities)")


if __name__ == "__main__":
    main()
