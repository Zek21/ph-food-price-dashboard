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

LEAKAGE FIX 2026-08-26
----------------------
`make_examples` used to select training ANCHORS at or before the cutoff but then
supervise every horizon target that existed, including target months AFTER the
cutoff. `gpu_driver_evidence/rerun_20260826/v2_leakage_audit.json` measured
leak_fraction = 1.0 - all 295 primary-gate points and all 2,360 multi-origin
points were trained on directly before being scored, so the originally published
3.55% MAPE was an in-sample number. Targets are now masked to periods at or
before the cutoff (strict=True by default). Honest re-measurement: 4.54% MAPE,
still beating naive persistence at 7.41%. --allow-leak reproduces the old
behaviour for A/B evidence only and can never pass the publication gate.

Log-returns are ~0-centred and comparable across commodities, so no per-series
scaler is needed; a commodity embedding carries series-specific drift.

Run: .venv-torch-dml\\Scripts\\python.exe gpu_forecaster_v2.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

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


def make_examples(pmaps, cid, cutoff: pd.Period, *, strict: bool = True):
    """Build training tensors from anchors at or before `cutoff`.

    With ``strict`` (default) a horizon target is supervised only when the
    TARGET period is also at or before the cutoff. Without it the loss sees
    post-cutoff actuals, which is the leak this module used to ship.
    """
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
                target = P + h
                if strict and target > cutoff:
                    continue
                if target in pm and pm[target] > 0:
                    y[h - 1] = math.log(pm[target]) - base
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


def huber_elementwise(pred, target, delta: float = 0.05):
    """Elementwise Huber, identical to nn.HuberLoss(reduction="none").

    torch-directml has no kernel for aten::huber_loss and silently falls back to
    the CPU, which forces a device round-trip on EVERY training step. Building the
    same value from sub/abs/where/mul keeps the whole loop on the GPU.
    """
    err = pred - target
    absr = err.abs()
    quad = 0.5 * err * err
    lin = delta * (absr - 0.5 * delta)
    return torch.where(absr <= delta, quad, lin)


class DmlAdam(torch.optim.Optimizer):
    """Adam written without lerp_, which torch-directml has no kernel for.

    torch.optim.Adam runs `exp_avg.lerp_(grad, 1 - beta1)` in both its single- and
    multi-tensor paths. On the DML backend that operator silently falls back to the
    CPU, so every optimizer step copies the momentum buffers off and back on the
    GPU. This is the same update built from mul_/add_/addcmul_/addcdiv_, which all
    have DML kernels, so the whole loop stays on-device. Verified to match
    torch.optim.Adam to float tolerance (tests/test_gpu_forecaster_v2_leakage.py).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self):                                    # type: ignore[override]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                param.addcdiv_(exp_avg, denom, value=-lr / bc1)


def train(dev, n_comm, X, Mn, Cd, Y, Msk, epochs=200):
    torch.manual_seed(0)
    model = MultiHorizonNet(n_comm).to(dev)
    X, Mn, Cd, Y, Msk = (t.to(dev) for t in (X, Mn, Cd, Y, Msk))
    w = (1.0 / torch.sqrt(torch.arange(1, H + 1, device=dev, dtype=torch.float32))).view(1, H)
    huber = huber_elementwise          # DML-native; see huber_elementwise docstring
    opt = DmlAdam(model.parameters(), lr=2e-3)
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
        out = model(torch.tensor(np.array([lr]), device=dev),
                    torch.tensor(np.array([mon]), dtype=torch.float32, device=dev),
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


def decide_publication_gate(strict: bool, beats: bool, wins: int, origins: int,
                            *, prospective: bool = False) -> str:
    """Typed disposition: leaked or already-inspected windows can never publish."""
    if not strict:
        return "withheld_leaked_training_targets"
    if not prospective:
        return "withheld_nonprospective_validation"
    if beats and origins > 0 and wins == origins:
        return "passed_out_of_time_naive_baseline"
    return "withheld_failed_validation"


def resolve_device(kind: str):
    """Return (torch device, human name, backend tag). dml needs torch-directml."""
    if kind == "cpu":
        return torch.device("cpu"), "CPU", "cpu"
    import torch_directml as dml
    return dml.device(), dml.device_name(0).replace("\x00", "").strip(), "directml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=("dml", "cpu"), default="dml",
                    help="dml = AMD GPU via torch-directml; cpu = same graph on CPU")
    ap.add_argument("--evidence", type=Path, default=EVID)
    ap.add_argument("--receipt-name", default="gpu_forecaster_v2.json")
    ap.add_argument("--forward-name", default="gpu_forward_predictions_v2.json")
    ap.add_argument("--allow-leak", action="store_true",
                    help="reproduce the pre-2026-08-26 contaminated target set (A/B only)")
    ap.add_argument("--no-forward", action="store_true")
    args = ap.parse_args()

    strict = not args.allow_leak
    dev, dev_name, backend = resolve_device(args.device)
    args.evidence.mkdir(parents=True, exist_ok=True)
    monthly = load_monthly()
    pmaps, cid, commodities = series_maps(monthly)
    cutoff = infer_cutoff()                       # 2026-01, same as v1 / the LSTM
    data_max = monthly["date"].max().to_period("M")

    # ---------- primary gate: train <= 2026-01, predict Feb..Jun 2026 ----------
    X, Mn, Cd, Y, Msk = make_examples(pmaps, cid, cutoff, strict=strict)
    model, param_dev, secs = train(dev, len(commodities), X, Mn, Cd, Y, Msk)
    a, mo, na = score_origin(model, dev, pmaps, cid, cutoff, GATE_H)
    gate_model, gate_naive = regression_metrics(a, mo), regression_metrics(a, na)
    beats = (gate_model["mape"] < gate_naive["mape"] and gate_model["mae"] < gate_naive["mae"]
             and gate_model["n"] >= 30)
    on_gpu = param_dev.startswith("privateuseone")
    print(f"dev={dev_name} backend={backend} param_device={param_dev} on_gpu={on_gpu} "
          f"leakage_masked={strict} windows={len(X)} supervised={int(Msk.sum())} {secs:.1f}s")
    print(f"  [gate {cutoff}->+{GATE_H}] v2 MAPE {gate_model['mape']}% MAE {gate_model['mae']} "
          f"R2 {gate_model['r2']} | naive MAPE {gate_naive['mape']}% MAE {gate_naive['mae']} "
          f"| beats={beats} n={gate_model['n']}")

    # ---------- multi-origin stability: train <= 2025-06, test 8 origins -------
    stab_cut = pd.Period("2025-06", freq="M")
    Xs, Mns, Cds, Ys, Msks = make_examples(pmaps, cid, stab_cut, strict=strict)
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
    forward, fsecs, fdev = {}, 0.0, ""
    if not args.no_forward:
        Xf, Mnf, Cdf, Yf, Mskf = make_examples(pmaps, cid, data_max, strict=strict)
        fmodel, fdev, fsecs = train(dev, len(commodities), Xf, Mnf, Cdf, Yf, Mskf)
        for c, pm in pmaps.items():
            preds = predict_from(fmodel, dev, pm, cid[c], data_max)
            if preds is None:
                continue
            forward[c] = {str(data_max + (h + 1)): round(preds[h], 4) for h in range(H)}

    # This historical Feb-Jun 2026 window is already inspected and cannot
    # become publication permission after adaptive work.
    gate_status = decide_publication_gate(
        strict, bool(beats), wins, len(per_origin), prospective=False)
    publishable = gate_status == "passed_out_of_time_naive_baseline"

    receipt = {
        "schema": "ph-food-price-gpu-forecaster-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": {"name": dev_name, "backend": backend, "torch": torch.__version__,
                   "param_device": param_dev, "trained_on_gpu": on_gpu},
        "method": {"direct_multi_horizon": H, "target": "log-return vs persistence",
                   "loss": "horizon-weighted Huber(delta=0.05)", "recursion": False,
                   "leakage_masked": strict,
                   "target_rule": ("supervised targets restricted to periods <= cutoff"
                                   if strict
                                   else "LEAKED: targets after the cutoff were supervised")},
        "training_shape": {"gate_windows": int(len(X)),
                           "gate_supervised_targets": int(Msk.sum()),
                           "stability_windows": int(len(Xs)),
                           "stability_supervised_targets": int(Msks.sum())},
        "primary_gate": {"cutoff": str(cutoff), "horizon_months": GATE_H,
                         "v2": gate_model, "naive_persistence": gate_naive,
                         "beats_naive": bool(beats)},
        "multi_origin_stability": {"train_cutoff": str(stab_cut),
                                   "origins_tested": len(per_origin),
                                   "origins_beating_naive": wins,
                                   "pooled_v2": stab_model, "pooled_naive": stab_naive,
                                   "per_origin": per_origin},
        "publication_gate": {
            "status": gate_status, "passed": publishable,
            "prospective_validation": False,
            "requirements": ("leakage_masked AND model<naive on MAPE and MAE at the gate "
                             "AND every stability origin beats naive AND an untouched "
                             "prospective validation window"),
        },
        "forward": {"anchor": str(data_max), "horizon": [str(data_max + 1), str(data_max + H)],
                    "commodities": len(forward)},
        "train_seconds": {"gate": round(secs, 2), "stability": round(ssecs, 2),
                          "forward": round(fsecs, 2)},
    }
    (args.evidence / args.receipt_name).write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    fwd = {"schema": "ph-food-price-gpu-v2-forward-predictions",
           "generated_at": receipt["generated_at"], "model": "GPU multi-horizon log-return MLP (v2)",
           "device": dev_name, "trained_on_gpu": on_gpu,
           "validation": {"gate_cutoff": str(cutoff), "gate_v2_mape": gate_model["mape"],
                          "gate_naive_mape": gate_naive["mape"], "beats_naive": bool(beats),
                          "multi_origin_wins": f"{wins}/{len(per_origin)}",
                          "leakage_masked": strict},
           "publication_gate": receipt["publication_gate"],
           "forecast_horizon": [str(data_max + 1), str(data_max + H)],
           "commodities": len(forward), "forecasts": forward,
           "disclaimer": "Experimental research forecasts, not financial advice."}
    if not publishable:
        fwd["withheld"] = ("Local experimental output only; the out-of-time publication "
                           "gate did not pass.")
    (ROOT / args.forward_name).write_text(json.dumps(fwd, indent=2), encoding="utf-8")
    print(f"  publication_gate={gate_status} -> wrote {args.receipt_name} + "
          f"{args.forward_name} ({len(forward)} commodities)")


if __name__ == "__main__":
    main()
