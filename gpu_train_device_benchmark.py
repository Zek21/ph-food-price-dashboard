#!/usr/bin/env python
"""CPU vs AMD-DirectML TRAINING benchmark for the v2 multi-horizon forecaster.

The repo already proves CPU-vs-GPU for warmed ONNX *inference*
(gpu_forecast_driver.py benchmark). That receipt says CPU wins. This one closes
the other half of the honest comparison: it times the identical v2 training run
(same examples, same seed, same epochs) on torch CPU and on torch-directml, with
repeats, and records the measured medians instead of an assumption.

Run: .venv-torch-dml/Scripts/python.exe gpu_train_device_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from gpu_train_forecaster import EVID, SEQ, infer_cutoff, load_monthly, regression_metrics
from gpu_forecaster_v2 import (GATE_H, H, make_examples, resolve_device,
                               score_origin, series_maps, train)


def timed(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        value = fn()
        out.append(time.perf_counter() - t0)
    return out, value


def _alternate(args) -> None:
    """Interleave single-rep cpu/dml measurements, flipping order every round."""
    import statistics as _st
    here = os.path.dirname(os.path.abspath(__file__))
    samples = {"cpu": [], "dml": []}
    rows = {}
    for rnd in range(args.alternate):
        order = ["cpu", "dml"] if rnd % 2 == 0 else ["dml", "cpu"]
        for kind in order:
            cmd = [sys.executable, os.path.abspath(__file__), "--only", kind,
                   "--reps", "1", "--epochs", str(args.epochs)]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=here)
            line = next((l for l in proc.stdout.splitlines() if l.startswith("@@ROW@@")), None)
            if line is None:
                print(f"round {rnd} {kind}: FAILED "
                      f"{(proc.stderr.strip().splitlines() or [chr(63)])[-1][:120]}")
                continue
            row = json.loads(line[len("@@ROW@@"):])
            samples[kind].append(row["train_seconds"][0])
            rows[kind] = row
        print(f"round {rnd} order={order} "
              f"cpu={samples['cpu'][-1] if samples['cpu'] else None} "
              f"dml={samples['dml'][-1] if samples['dml'] else None}")
    for kind in ("cpu", "dml"):
        if rows.get(kind):
            rows[kind]["train_seconds"] = [round(v, 4) for v in samples[kind]]
            rows[kind]["train_median_s"] = round(_st.median(samples[kind]), 4)
            rows[kind]["train_min_s"] = round(min(samples[kind]), 4)
            rows[kind]["reps"] = len(samples[kind])
            rows[kind]["sampling"] = f"interleaved, {args.alternate} rounds, order flipped"
    args.name = args.name.replace(".json", "_alternating.json")
    _emit(args, rows, [], None, "interleaved")


def _emit(args, rows, X, Msk, cutoff) -> None:
    verdict = None
    if rows.get("cpu", {}).get("available") and rows.get("dml", {}).get("available"):
        c, g = rows["cpu"]["train_median_s"], rows["dml"]["train_median_s"]
        verdict = {
            "gpu_train_speedup_vs_cpu": round(c / g, 3),
            "faster_for_training": "directml_gpu" if g < c else "cpu",
            "cpu_train_median_s": c,
            "gpu_train_median_s": g,
            "score_faster": ("cpu" if rows["cpu"]["score_seconds"] <= rows["dml"]["score_seconds"]
                             else "directml_gpu"),
            "cpu_score_s": rows["cpu"]["score_seconds"],
            "gpu_score_s": rows["dml"]["score_seconds"],
        }

    receipt = {
        "schema": "ph-food-price-train-device-benchmark-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {"python": platform.python_version(), "torch": torch.__version__,
                 "machine": platform.machine(), "system": platform.system()},
        "workload": {"model": "v2 multi-horizon log-return MLP", "sequence_len": SEQ,
                     "horizons": H, "gate_horizon": GATE_H, "cutoff": str(cutoff),
                     "windows": int(len(X)) if len(X) else None,
                     "supervised_targets": int(Msk.sum()) if Msk is not None else None,
                     "batch_size": 8192, "optimizer": "Adam(lr=2e-3)",
                     "leakage_masked": True},
        "devices": rows,
        "verdict": verdict,
        "claim_boundary": ("Measures end-to-end training wall time for THIS small MLP on THIS "
                           "host. It does not generalise to larger models, other GPUs, or the "
                           "ONNX LSTM inference benchmark, which is measured separately."),
    }
    (args.evidence / args.name).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("verdict:", json.dumps(verdict))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--evidence", type=Path, default=EVID)
    ap.add_argument("--name", default="train_device_benchmark.json")
    ap.add_argument("--alternate", type=int, default=0,
                    help="rounds of INTERLEAVED cpu/dml single-rep runs; order flips "
                         "every round so machine drift cannot favour one device")
    ap.add_argument("--only", choices=("cpu", "dml"),
                    help="worker mode: measure ONE device and print JSON on stdout")
    args = ap.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)

    monthly = load_monthly()
    pmaps, cid, commodities = series_maps(monthly)
    cutoff = infer_cutoff()
    X, Mn, Cd, Y, Msk = make_examples(pmaps, cid, cutoff, strict=True)
    print(f"examples: windows={len(X)} supervised_targets={int(Msk.sum())} cutoff={cutoff}")

    kinds = [args.only] if args.only else ["cpu", "dml"]
    if args.alternate and not args.only:
        _alternate(args)
        return
    if not args.only:
        # torch-directml crashes autograd when a CPU graph is built before the DML
        # device exists in the same process (engine.cpp device_ready_queues_ assert),
        # so each device is measured in its own interpreter.
        rows = {}
        for kind in kinds:
            cmd = [sys.executable, os.path.abspath(__file__), "--only", kind,
                   "--reps", str(args.reps), "--epochs", str(args.epochs)]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=os.path.dirname(os.path.abspath(__file__)))
            marker = "@@ROW@@"
            line = next((l for l in proc.stdout.splitlines() if l.startswith(marker)), None)
            if line is None:
                rows[kind] = {"available": False,
                              "error": (proc.stderr.strip().splitlines() or ["no output"])[-1]}
            else:
                rows[kind] = json.loads(line[len(marker):])
            print(f"{kind}: {rows[kind].get('train_median_s', rows[kind].get('error'))}")
        _emit(args, rows, X, Msk, cutoff)
        return

    rows = {}
    for kind in kinds:
        try:
            dev, dev_name, backend = resolve_device(kind)
        except Exception as exc:                       # torch-directml absent
            rows[kind] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        # warm-up (driver/kernel compile) is excluded from the timings
        train(dev, len(commodities), X, Mn, Cd, Y, Msk, epochs=2)
        state = {}

        def run():
            model, param_dev, secs = train(dev, len(commodities), X, Mn, Cd, Y, Msk,
                                           epochs=args.epochs)
            state["model"], state["param_dev"], state["inner"] = model, param_dev, secs
            return model

        durations, model = timed(run, args.reps)
        t0 = time.perf_counter()
        a, mo, na = score_origin(model, dev, pmaps, cid, cutoff, GATE_H)
        score_secs = time.perf_counter() - t0
        metrics = regression_metrics(a, mo)
        naive = regression_metrics(a, na)
        rows[kind] = {
            "available": True,
            "device_name": dev_name,
            "backend": backend,
            "param_device": state["param_dev"],
            "epochs": args.epochs,
            "reps": args.reps,
            "train_seconds": [round(d, 4) for d in durations],
            "train_median_s": round(statistics.median(durations), 4),
            "train_min_s": round(min(durations), 4),
            "score_seconds": round(score_secs, 4),
            "gate_mape": metrics["mape"],
            "gate_mae": metrics["mae"],
            "naive_mape": naive["mape"],
            "naive_mae": naive["mae"],
            "n": metrics["n"],
        }
        print(f"{kind:>4}: dev={dev_name} param_device={state['param_dev']} "
              f"train_median={rows[kind]['train_median_s']}s score={score_secs:.2f}s "
              f"gate_mape={metrics['mape']}")

    if args.only:
        print("@@ROW@@" + json.dumps(rows[args.only]))
        return

    _emit(args, rows, X, Msk, cutoff)


if __name__ == "__main__":
    main()
