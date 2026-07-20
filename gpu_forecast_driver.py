"""DirectML inference driver for the Philippine food-price LSTM models.

This is an ML execution driver, not a Windows display or kernel driver.  It
exports the project's PyTorch checkpoints to ONNX, runs the tensor graph through
ONNX Runtime's DirectML execution provider, benchmarks GPU versus CPU with real
model inputs, and generates forecasts from the current WFP dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "WFP" / "wfp_food_prices_phl_latest.csv"
DEFAULT_CHECKPOINTS = ROOT / ".lstm_models"
DEFAULT_MODELS = ROOT / ".onnx_models"
DEFAULT_EVIDENCE = ROOT / "gpu_driver_evidence"
FEATURE_COUNT = 6
SEQUENCE_LENGTH = 12
DML_PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _safe_name(commodity: str) -> str:
    return commodity.replace(" ", "_").replace("/", "_")


def _ort():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is missing. Install requirements-gpu.txt in an isolated environment."
        ) from exc
    return ort


def probe() -> dict:
    """Return the runtime's observed provider truth without assuming GPU use."""
    ort = _ort()
    providers = ort.get_available_providers()
    return {
        "python": os.sys.version.split()[0],
        "onnxruntime": ort.__version__,
        "available_providers": providers,
        "directml_available": DML_PROVIDER in providers,
        "device_id": int(os.environ.get("GPU_DML_DEVICE_ID", "0")),
        "truth_note": (
            "DirectML availability proves a usable provider. Per-node profiling is required "
            "before claiming DirectML device placement for a specific model graph."
        ),
    }


def _session(model_path: Path, *, prefer_gpu: bool, profile_prefix: Path | None = None):
    ort = _ort()
    options = ort.SessionOptions()
    if prefer_gpu:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        if profile_prefix is not None:
            profile_prefix.parent.mkdir(parents=True, exist_ok=True)
            options.enable_profiling = True
            options.profile_file_prefix = str(profile_prefix)
        device_id = str(int(os.environ.get("GPU_DML_DEVICE_ID", "0")))
        providers = [(DML_PROVIDER, {"device_id": device_id}), CPU_PROVIDER]
    else:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [CPU_PROVIDER]
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def export_models(checkpoint_dir: Path, model_dir: Path) -> dict:
    """Export every project LSTM checkpoint to a portable ONNX graph."""
    try:
        import onnx
        import torch
        from lstm_model import PriceLSTM
    except ImportError as exc:
        raise RuntimeError("Export requires PyTorch and ONNX in the current environment") from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    failures = []
    for checkpoint_path in sorted(checkpoint_dir.glob("lstm_*.pt")):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            commodity = checkpoint.get("commodity") or checkpoint_path.stem.removeprefix("lstm_")
            model = PriceLSTM()
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            output_path = model_dir / f"lstm_{_safe_name(commodity)}.onnx"
            dummy = torch.zeros((1, SEQUENCE_LENGTH, FEATURE_COUNT), dtype=torch.float32)
            torch.onnx.export(
                model,
                dummy,
                output_path,
                input_names=["features"],
                output_names=["price"],
                dynamic_axes={"features": {0: "batch"}, "price": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
            graph = onnx.load(output_path)
            metadata = {
                "commodity": str(commodity),
                "scaler_mean": json.dumps(checkpoint["scaler_mean"]),
                "scaler_scale": json.dumps(checkpoint["scaler_scale"]),
                "sequence_length": str(checkpoint.get("seq_len", SEQUENCE_LENGTH)),
                "source_checkpoint_sha256": _sha256(checkpoint_path),
                "source_checkpoint_mtime_utc": datetime.fromtimestamp(
                    checkpoint_path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            del graph.metadata_props[:]
            for key, value in metadata.items():
                item = graph.metadata_props.add()
                item.key = key
                item.value = value
            onnx.save(graph, output_path)
            exported.append(
                {
                    "commodity": commodity,
                    "path": str(output_path),
                    "bytes": output_path.stat().st_size,
                    "sha256": _sha256(output_path),
                }
            )
        except Exception as exc:  # keep a complete export receipt
            failures.append({"checkpoint": str(checkpoint_path), "error": str(exc)})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exported_count": len(exported),
        "failure_count": len(failures),
        "exported": exported,
        "failures": failures,
    }


def _load_monthly(data_path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(data_path)
    required = {"date", "price", "commodity", "admin1", "pricetype"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"WFP data is missing required columns: {missing}")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw.dropna(subset=["date", "price", "commodity", "admin1", "pricetype"])
    raw = raw[raw["price"] > 0].copy()
    raw["month_period"] = raw["date"].dt.to_period("M")
    monthly = (
        raw.groupby(["commodity", "admin1", "pricetype", "month_period"], observed=True)
        .agg(price=("price", "mean"))
        .reset_index()
    )
    monthly["date"] = monthly["month_period"].dt.to_timestamp()
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    regions = {name: index for index, name in enumerate(sorted(monthly["admin1"].unique()))}
    price_types = {name: index for index, name in enumerate(sorted(monthly["pricetype"].unique()))}
    monthly["region_enc"] = monthly["admin1"].map(regions)
    monthly["pt_enc"] = monthly["pricetype"].map(price_types)
    summary = {
        "rows": int(len(raw)),
        "commodities": int(raw["commodity"].nunique()),
        "regions": int(raw["admin1"].nunique()),
        "price_types": int(raw["pricetype"].nunique()),
        "min_date": raw["date"].min().date().isoformat(),
        "max_date": raw["date"].max().date().isoformat(),
        "sha256": _sha256(data_path),
    }
    return monthly, summary


def _seed_for_commodity(monthly: pd.DataFrame, commodity: str, scaler_mean: float,
                        scaler_scale: float,
                        required_latest: pd.Timestamp | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rows = monthly[monthly["commodity"] == commodity].copy()
    if rows.empty:
        raise ValueError(f"No current WFP rows for commodity: {commodity}")
    region_enc = float(rows["region_enc"].mode().iloc[0])
    pt_enc = float(rows["pt_enc"].mode().iloc[0])
    averages = rows.groupby("date", observed=True).agg(price=("price", "mean")).reset_index()
    averages = averages.sort_values("date")
    observed_latest = averages["date"].max()
    if required_latest is not None and observed_latest != required_latest:
        raise ValueError(
            f"Latest observation for {commodity} is {observed_latest.strftime('%Y-%m')}; "
            f"current dataset maximum is {required_latest.strftime('%Y-%m')}"
        )
    if len(averages) < SEQUENCE_LENGTH:
        raise ValueError(f"Need {SEQUENCE_LENGTH} monthly observations for {commodity}")
    prices = averages.tail(SEQUENCE_LENGTH)["price"].to_numpy(dtype=np.float64)
    normalized = (prices - scaler_mean) / scaler_scale
    return prices, normalized, {
        "region_enc": region_enc,
        "pt_enc": pt_enc,
        "last_date": averages["date"].max(),
        "year_min": int(monthly["year"].min()),
        "year_span": max(int(monthly["year"].max() - monthly["year"].min()), 1),
    }


def _window(normalized: np.ndarray, next_date: pd.Timestamp, context: dict) -> np.ndarray:
    window = np.zeros((SEQUENCE_LENGTH, FEATURE_COUNT), dtype=np.float32)
    for index in range(SEQUENCE_LENGTH):
        shifted = next_date - pd.DateOffset(months=SEQUENCE_LENGTH - 1 - index)
        window[index] = (
            normalized[-(SEQUENCE_LENGTH - index)],
            math.sin(2 * math.pi * shifted.month / 12),
            math.cos(2 * math.pi * shifted.month / 12),
            (shifted.year - context["year_min"]) / context["year_span"],
            context["region_enc"],
            context["pt_enc"],
        )
    return window


def _metadata(session) -> dict:
    return session.get_modelmeta().custom_metadata_map


def generate_predictions(data_path: Path, model_dir: Path, output_path: Path,
                         *, horizon: int = 18, prefer_gpu: bool = True) -> dict:
    monthly, data_summary = _load_monthly(data_path)
    current_latest = monthly["date"].max()
    available = probe()
    if prefer_gpu and not available["directml_available"]:
        raise RuntimeError("DirectML was requested but DmlExecutionProvider is unavailable")
    forecasts = {}
    model_receipts = []
    skipped_models = []
    for model_path in sorted(model_dir.glob("lstm_*.onnx")):
        session = _session(model_path, prefer_gpu=prefer_gpu)
        meta = _metadata(session)
        commodity = meta["commodity"]
        scaler_mean = float(json.loads(meta["scaler_mean"])[0])
        scaler_scale = float(json.loads(meta["scaler_scale"])[0])
        try:
            prices, normalized, context = _seed_for_commodity(
                monthly, commodity, scaler_mean, scaler_scale, current_latest
            )
        except ValueError as exc:
            status = (
                "skipped_stale_current_history"
                if "current dataset maximum" in str(exc)
                else "skipped_insufficient_current_history"
            )
            skipped_models.append(
                {
                    "commodity": commodity,
                    "model_sha256": _sha256(model_path),
                    "status": status,
                    "reason": str(exc),
                }
            )
            continue
        commodity_forecasts = {}
        for step in range(1, horizon + 1):
            next_date = context["last_date"] + pd.DateOffset(months=step)
            model_input = _window(normalized, next_date, context)[None, :, :]
            prediction = float(session.run(None, {session.get_inputs()[0].name: model_input})[0][0])
            prediction = max(0.0, prediction)
            recent_average = float(np.mean(prices[-SEQUENCE_LENGTH:]))
            if prediction > recent_average * 5:
                prediction = recent_average * 1.5
            prices = np.append(prices, prediction)
            normalized = np.append(normalized, (prediction - scaler_mean) / scaler_scale)
            commodity_forecasts[next_date.strftime("%Y-%m")] = round(prediction, 2)
        forecasts[commodity] = commodity_forecasts
        model_receipts.append(
            {
                "commodity": commodity,
                "model_sha256": _sha256(model_path),
                "providers": session.get_providers(),
                "source_checkpoint_sha256": meta.get("source_checkpoint_sha256"),
                "source_checkpoint_mtime_utc": meta.get("source_checkpoint_mtime_utc"),
            }
        )
    result = {
        "schema": "ph-food-price-directml-predictions-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "requested": "directml" if prefer_gpu else "cpu",
            "runtime_probe": available,
            "native_gpu_claim": (
                "supported_by_provider_binding; see benchmark profile for per-node placement proof"
                if prefer_gpu else "not_requested"
            ),
        },
        "data": data_summary,
        "method": {
            "model": "project LSTM checkpoints exported to ONNX",
            "horizon_months": horizon,
            "forecast_start": min(next(iter(values)) for values in forecasts.values()),
            "forecast_end": max(next(reversed(values)) for values in forecasts.values()),
            "eligibility": (
                "A model is forecast only when its most recent observation equals the "
                "dataset-wide latest month. Stale series are recorded, not extrapolated."
            ),
            "limitation": (
                "Inference uses the latest WFP observations with previously trained checkpoints. "
                "Forecasts are experimental decision support, not financial advice."
            ),
        },
        "model_count": len(model_receipts),
        "skipped_model_count": len(skipped_models),
        "forecast_point_count": sum(len(values) for values in forecasts.values()),
        "models": model_receipts,
        "skipped_models": skipped_models,
        "forecasts": forecasts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _profile_placement(session, model_input: np.ndarray) -> dict:
    input_name = session.get_inputs()[0].name
    session.run(None, {input_name: model_input})
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    node_events = [event for event in events if event.get("cat") == "Node"]
    by_provider: dict[str, int] = {}
    ops: dict[str, int] = {}
    nodes: list[dict] = []
    for event in node_events:
        args = event.get("args") or {}
        provider = args.get("provider", "unknown")
        operation = args.get("op_name", event.get("name", "unknown"))
        by_provider[provider] = by_provider.get(provider, 0) + 1
        ops[operation] = ops.get(operation, 0) + 1
        nodes.append(
            {
                "name": event.get("name", ""),
                "op_name": operation,
                "provider": provider,
                "duration_us": event.get("dur"),
            }
        )
    profile_path.unlink(missing_ok=True)
    return {
        "node_events": len(node_events),
        "by_provider": by_provider,
        "operations": ops,
        "nodes": nodes,
    }


def benchmark(model_path: Path, data_path: Path, output_path: Path, *,
              batch_sizes: tuple[int, ...] = (1, 8, 32, 128), iterations: int = 100) -> dict:
    runtime = probe()
    if not runtime["directml_available"]:
        raise RuntimeError("DmlExecutionProvider is unavailable; DirectML benchmark cannot run")
    monthly, data_summary = _load_monthly(data_path)
    profile_session = _session(
        model_path,
        prefer_gpu=True,
        profile_prefix=output_path.parent / "directml_profile",
    )
    meta = _metadata(profile_session)
    scaler_mean = float(json.loads(meta["scaler_mean"])[0])
    scaler_scale = float(json.loads(meta["scaler_scale"])[0])
    _, normalized, context = _seed_for_commodity(
        monthly, meta["commodity"], scaler_mean, scaler_scale
    )
    next_date = context["last_date"] + pd.DateOffset(months=1)
    base = _window(normalized, next_date, context)[None, :, :]
    placement = _profile_placement(profile_session, base)
    native_gpu_ops = placement["by_provider"].get(DML_PROVIDER, 0)
    if native_gpu_ops <= 0:
        raise RuntimeError("DirectML session bound, but profiling found zero DirectML node events")

    gpu = _session(model_path, prefer_gpu=True)
    cpu = _session(model_path, prefer_gpu=False)
    input_name = gpu.get_inputs()[0].name
    rows = []
    for batch_size in batch_sizes:
        model_input = np.repeat(base, batch_size, axis=0)
        for session in (gpu, cpu):
            for _ in range(10):
                session.run(None, {input_name: model_input})
        timings = {}
        outputs = {}
        for label, session in (("gpu_directml", gpu), ("cpu", cpu)):
            samples = []
            output = None
            for _ in range(iterations):
                started = time.perf_counter_ns()
                output = session.run(None, {input_name: model_input})[0]
                samples.append((time.perf_counter_ns() - started) / 1_000_000)
            timings[label] = {
                "median_ms": round(statistics.median(samples), 4),
                "p95_ms": round(_percentile(samples, 0.95), 4),
                "throughput_items_s": round(batch_size / (statistics.median(samples) / 1000), 2),
            }
            outputs[label] = output
        speedup = timings["cpu"]["median_ms"] / timings["gpu_directml"]["median_ms"]
        rows.append(
            {
                "batch_size": batch_size,
                **timings,
                "gpu_speedup_vs_cpu": round(speedup, 3),
                "max_abs_output_difference": round(
                    float(np.max(np.abs(outputs["gpu_directml"] - outputs["cpu"]))), 8
                ),
            }
        )
    result = {
        "schema": "ph-food-price-directml-benchmark-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "commodity": meta["commodity"],
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
            "input_shape": ["batch", SEQUENCE_LENGTH, FEATURE_COUNT],
        },
        "data": data_summary,
        "runtime": runtime,
        "sessions": {"gpu": gpu.get_providers(), "cpu": cpu.get_providers()},
        "profile": placement,
        "native_gpu_verified": native_gpu_ops > 0,
        "iterations_per_batch": iterations,
        "warmup_iterations": 10,
        "results": rows,
        "claim_boundary": (
            "This measures warmed ONNX LSTM inference on one real project graph. It does not "
            "measure training, end-to-end CSV processing, every Python operation, or other GPUs."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "export", "benchmark", "predict", "run-all"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--horizon", type=int, default=18)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Use CPU for prediction comparison")
    args = parser.parse_args()

    args.evidence.mkdir(parents=True, exist_ok=True)
    if args.command == "probe":
        result = probe()
    elif args.command == "export":
        result = export_models(args.checkpoints, args.models)
        _write_json(args.evidence / "export_manifest.json", result)
    elif args.command == "benchmark":
        model_path = args.model or next(args.models.glob("lstm_Rice_*regular*_milled*.onnx"))
        result = benchmark(
            model_path, args.data, args.evidence / "benchmark_current.json",
            iterations=args.iterations,
        )
    elif args.command == "predict":
        result = generate_predictions(
            args.data,
            args.models,
            args.evidence / ("predictions_cpu.json" if args.cpu else "predictions_current.json"),
            horizon=args.horizon,
            prefer_gpu=not args.cpu,
        )
    else:
        export_result = export_models(args.checkpoints, args.models)
        _write_json(args.evidence / "export_manifest.json", export_result)
        if export_result["failure_count"]:
            raise RuntimeError(f"ONNX export failed for {export_result['failure_count']} checkpoints")
        model_path = next(args.models.glob("lstm_Rice_*regular*_milled*.onnx"))
        benchmark_result = benchmark(
            model_path, args.data, args.evidence / "benchmark_current.json",
            iterations=args.iterations,
        )
        prediction_result = generate_predictions(
            args.data, args.models, args.evidence / "predictions_current.json",
            horizon=args.horizon, prefer_gpu=True,
        )
        result = {
            "exported_models": export_result["exported_count"],
            "native_gpu_verified": benchmark_result["native_gpu_verified"],
            "prediction_models": prediction_result["model_count"],
            "skipped_prediction_models": prediction_result["skipped_model_count"],
            "forecast_points": prediction_result["forecast_point_count"],
            "evidence_dir": str(args.evidence.resolve()),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
