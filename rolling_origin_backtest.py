"""Retrain and score the release pipeline at several historical forecast origins.

The release gate has been deciding on one window of 177 points (2026-04..2026-06),
where a 0.2 MAE gap carries a 95% CI of [-0.48, +1.01] -- it cannot distinguish a
real improvement from chance in either direction, and the ticket has cycled on
coin-flip verdicts while waiting for WFP to publish data past 2026-06-15.

Waiting is not the only way to get an out-of-time window.  `lstm_release_train.py`
takes `--publication-cutoff`, and a full 59-model retrain costs about 33 seconds, so
the pipeline can be rebuilt at earlier origins and scored on data that origin's
models never saw.  That buys statistical power from local data already on disk.

Each origin is an independent test: train with everything at or before the cutoff,
export to ONNX, then score the recursive forecast against naive persistence over the
months that follow.  Note that the evaluation window runs from the cutoff to the end
of the dataset, so earlier origins are scored over LONGER horizons and recursive
error compounds -- per-origin verdicts are directly valid, but the horizon must be
reported alongside any cross-origin comparison rather than pooled blindly.

Writes one receipt per origin plus a summary.  Trains and scores locally; publishes
nothing and promotes no prediction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import gpu_forecast_driver as driver

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "WFP" / "wfp_food_prices_phl_latest.csv"
DEFAULT_OUT = ROOT / "gpu_driver_evidence" / "rolling_origin_20260912"
DEFAULT_ORIGINS = ("2026-03", "2025-12", "2025-09", "2025-06", "2025-03", "2024-12")


def _receipt_label(path: Path) -> str:
    """ROOT-relative label when possible, absolute otherwise. Never raises."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def run_origin(origin: str, *, data: Path, out_dir: Path, epochs: int) -> dict:
    tag = origin.replace("-", "")
    checkpoints = ROOT / f".lstm_models_oot_{tag}"
    onnx_dir = ROOT / f".onnx_models_oot_{tag}"
    train_receipt = out_dir / f"train_{tag}.json"
    validation = out_dir / f"validation_{tag}.json"

    started = time.perf_counter()
    if not train_receipt.is_file():
        completed = subprocess.run(
            [sys.executable, str(ROOT / "lstm_release_train.py"),
             "--publication-cutoff", origin, "--epochs", str(epochs),
             "--model-dir", str(checkpoints), "--receipt", str(train_receipt)],
            capture_output=True, text=True, cwd=str(ROOT))
        if completed.returncode != 0:
            return {"origin": origin, "ok": False,
                    "error": (completed.stderr or completed.stdout or "")[-800:]}
    train_seconds = time.perf_counter() - started

    driver.export_models(checkpoints, onnx_dir)
    result = driver.validate_models(data, onnx_dir, validation)
    gate = result["publication_gate"]
    significance = gate.get("paired_significance") or {}
    return {
        "origin": origin,
        "ok": True,
        "training_cutoff_proof": result["cutoff_proof"]["training_cutoff"],
        "window": f"{result['validation_start']}..{result['validation_end']}",
        "horizon_months": _months_between(result["validation_start"], result["validation_end"]),
        "points": result["model"]["n"],
        "commodities": result["eligible_model_count"],
        "model": result["model"],
        "naive_persistence": result["naive_persistence"],
        "mae_gap": round(result["model"]["mae"] - result["naive_persistence"]["mae"], 4),
        "mape_gap": round(result["model"]["mape"] - result["naive_persistence"]["mape"], 4),
        "mae_verdict": (significance.get("mae") or {}).get("verdict"),
        "mape_verdict": (significance.get("mape") or {}).get("verdict"),
        "mae_ci95": (significance.get("mae") or {}).get("ci95"),
        "mape_ci95": (significance.get("mape") or {}).get("ci95"),
        "commodities_model_better_mae": (significance.get("mae") or {}).get("model_better_commodities"),
        "commodities_model_worse_mae": (significance.get("mae") or {}).get("model_worse_commodities"),
        "blow_ups": len((gate.get("per_commodity_guard") or {}).get("flagged") or []),
        "gate_passed": gate["passed"],
        "train_seconds": round(train_seconds, 1),
        # relative_to() raises when the out dir is not under ROOT, which a RELATIVE
        # --out also triggers (a relative path is never "in the subpath of" an
        # absolute ROOT). Crashed the whole run after the first origin on 2026-09-14.
        # Resolve first, and fall back to the absolute path for an out dir outside ROOT.
        "validation_receipt": _receipt_label(validation),
    }


def _months_between(start: str, end: str) -> int:
    (year_a, month_a), (year_b, month_b) = (
        tuple(int(part) for part in value.split("-")[:2]) for value in (start, end))
    return (year_b - year_a) * 12 + (month_b - month_a) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--origins", nargs="*", default=list(DEFAULT_ORIGINS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for origin in args.origins:
        row = run_origin(origin, data=args.data, out_dir=args.out, epochs=args.epochs)
        rows.append(row)
        if row.get("ok"):
            print(f"{origin}  n={row['points']:<5} horizon={row['horizon_months']:>2}mo  "
                  f"MAE gap {row['mae_gap']:+.4f} [{row['mae_verdict']}]  "
                  f"MAPE gap {row['mape_gap']:+.4f}  blow-ups {row['blow_ups']}  "
                  f"gate={row['gate_passed']}", flush=True)
        else:
            print(f"{origin}  FAILED: {row.get('error', '')[:200]}", flush=True)

    scored = [row for row in rows if row.get("ok")]
    summary = {
        "schema": "ph-food-price-rolling-origin-backtest-v1",
        "task": "AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS",
        "origins_requested": args.origins,
        "origins_scored": len(scored),
        "total_points": sum(row["points"] for row in scored),
        "origins_where_model_beat_persistence_on_mae": [
            row["origin"] for row in scored if row["mae_gap"] < 0],
        "origins_where_model_beat_persistence_on_mape": [
            row["origin"] for row in scored if row["mape_gap"] < 0],
        "origins_significantly_better_on_mae": [
            row["origin"] for row in scored if row["mae_verdict"] == "model_significantly_better"],
        "origins_significantly_worse_on_mae": [
            row["origin"] for row in scored if row["mae_verdict"] == "model_significantly_worse"],
        "origins_passing_the_gate": [row["origin"] for row in scored if row["gate_passed"]],
        "horizon_caveat": ("evaluation runs cutoff..dataset end, so earlier origins are "
                            "scored over longer horizons; compare per-origin, do not pool blindly"),
        "origins": rows,
    }
    (args.out / "rolling_origin_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "origins"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
