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


def _directml_unavailable_message() -> str:
    expected = ROOT / ".venv-directml" / "Scripts" / "python.exe"
    detail = "DirectML was requested but DmlExecutionProvider is unavailable."
    if expected.exists() and Path(os.sys.executable).resolve() != expected.resolve():
        return (
            f"{detail} Current interpreter: {Path(os.sys.executable).resolve()}. "
            f"Use the project-pinned interpreter: {expected.resolve()}"
        )
    return f"{detail} Install requirements-gpu.txt in the active isolated environment."


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
                         *, horizon: int = 18, prefer_gpu: bool = True,
                         validation: dict | None = None) -> dict:
    monthly, data_summary = _load_monthly(data_path)
    current_latest = monthly["date"].max()
    available = probe()
    if prefer_gpu and not available["directml_available"]:
        raise RuntimeError(_directml_unavailable_message())
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
        "publication_gate": (
            validation.get("publication_gate", {})
            if validation is not None
            else {
                "status": "withheld_validation_missing",
                "passed": False,
                "reason": "No out-of-time naive-baseline validation receipt was supplied.",
            }
        ),
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


def _regression_metrics(actual: list[float], predicted: list[float]) -> dict:
    """Return transparent aggregate regression metrics for one proof population."""
    if not actual or len(actual) != len(predicted):
        return {}
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    nonzero = np.abs(y) > 1e-12
    mape = float(np.mean(np.abs((y[nonzero] - p[nonzero]) / y[nonzero])) * 100)
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = None if denominator <= 0 else float(1 - np.sum((y - p) ** 2) / denominator)
    return {
        "mape": round(mape, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": None if r2 is None else round(r2, 6),
        "n": int(len(y)),
    }


def _infer_training_cutoff(prediction_artifact: Path) -> dict:
    """Infer the last model-known month from the first dated historical forecast.

    The legacy training artifact was produced with forecasts beginning one month
    after its data cutoff.  The artifact hash and timestamps make this inference
    auditable instead of relying on a filename or remembered date.
    """
    document = json.loads(prediction_artifact.read_text(encoding="utf-8"))
    forecasts = document.get("forecasts") or {}
    first_dates = []
    for values in forecasts.values():
        if isinstance(values, dict) and values:
            first_dates.append(min(values))
    if not first_dates:
        raise ValueError("Prediction artifact has no dated forecasts")
    forecast_start = min(first_dates)
    cutoff = pd.Period(forecast_start, freq="M") - 1
    return {
        "prediction_artifact": str(prediction_artifact),
        "prediction_artifact_sha256": _sha256(prediction_artifact),
        "prediction_artifact_mtime_utc": datetime.fromtimestamp(
            prediction_artifact.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "forecast_start": forecast_start,
        "training_cutoff": str(cutoff),
        "inference": "training cutoff is the month immediately before the earliest stored forecast",
    }


def _publication_gate(model_metrics: dict, baseline_metrics: dict,
                      *, model_count: int, minimum_models: int = 10,
                      minimum_points: int = 30) -> dict:
    reasons = []
    if model_count < minimum_models:
        reasons.append(f"eligible_models={model_count} below minimum={minimum_models}")
    if int(model_metrics.get("n", 0)) < minimum_points:
        reasons.append(
            f"validation_points={model_metrics.get('n', 0)} below minimum={minimum_points}"
        )
    if not model_metrics or not baseline_metrics:
        reasons.append("model or baseline metrics are missing")
    else:
        if model_metrics["mape"] >= baseline_metrics["mape"]:
            reasons.append(
                f"model MAPE {model_metrics['mape']} did not beat naive MAPE {baseline_metrics['mape']}"
            )
        if model_metrics["mae"] >= baseline_metrics["mae"]:
            reasons.append(
                f"model MAE {model_metrics['mae']} did not beat naive MAE {baseline_metrics['mae']}"
            )
    passed = not reasons
    return {
        "status": "passed_out_of_time_naive_baseline" if passed else "withheld_failed_validation",
        "passed": passed,
        "requirements": {
            "minimum_models": minimum_models,
            "minimum_points": minimum_points,
            "model_mape_lt_naive": True,
            "model_mae_lt_naive": True,
        },
        "reasons": reasons,
    }


def validate_models(data_path: Path, model_dir: Path, output_path: Path,
                    *, prediction_artifact: Path | None = None) -> dict:
    """Backtest the exported graphs after their historical training cutoff.

    Each commodity is seeded only with observations at or before the inferred
    cutoff.  Model forecasts and a persistence forecast are then rolled forward
    without peeking at later actuals.  This receipt gates publication; it does
    not change or delete generated local predictions.
    """
    prediction_artifact = prediction_artifact or ROOT / "lstm_predictions.json"
    cutoff_proof = _infer_training_cutoff(prediction_artifact)
    cutoff_period = pd.Period(cutoff_proof["training_cutoff"], freq="M")
    monthly, data_summary = _load_monthly(data_path)
    data_max_period = monthly["date"].max().to_period("M")
    if data_max_period <= cutoff_period:
        raise ValueError(
            f"Dataset maximum {data_max_period} is not after training cutoff {cutoff_period}"
        )

    year_min = int(monthly["year"].min())
    year_span = max(int(monthly["year"].max() - monthly["year"].min()), 1)
    all_actual: list[float] = []
    all_model: list[float] = []
    all_naive: list[float] = []
    model_receipts = []
    skipped = []

    for model_path in sorted(model_dir.glob("lstm_*.onnx")):
        session = _session(model_path, prefer_gpu=False)
        meta = _metadata(session)
        commodity = meta.get("commodity")
        if not commodity:
            skipped.append({"model": str(model_path), "reason": "commodity metadata missing"})
            continue
        scaler_mean = float(json.loads(meta["scaler_mean"])[0])
        scaler_scale = float(json.loads(meta["scaler_scale"])[0])
        rows = monthly[monthly["commodity"] == commodity].copy()
        averages = (
            rows.groupby("date", observed=True).agg(price=("price", "mean")).reset_index()
            .sort_values("date")
        )
        averages["period"] = averages["date"].dt.to_period("M")
        before = averages[averages["period"] <= cutoff_period]
        after = averages[
            (averages["period"] > cutoff_period) & (averages["period"] <= data_max_period)
        ]
        required_seed = pd.period_range(end=cutoff_period, periods=SEQUENCE_LENGTH, freq="M")
        seed_rows = before[before["period"].isin(required_seed)].sort_values("period")
        if list(seed_rows["period"]) != list(required_seed):
            skipped.append({
                "commodity": commodity,
                "model_sha256": _sha256(model_path),
                "reason": f"missing consecutive {SEQUENCE_LENGTH}-month seed ending {cutoff_period}",
            })
            continue
        expected_after = pd.period_range(
            start=cutoff_period + 1, end=data_max_period, freq="M"
        )
        after = after[after["period"].isin(expected_after)].sort_values("period")
        if list(after["period"]) != list(expected_after):
            skipped.append({
                "commodity": commodity,
                "model_sha256": _sha256(model_path),
                "reason": f"missing one or more out-of-time months through {data_max_period}",
            })
            continue

        known_rows = rows[rows["date"].dt.to_period("M") <= cutoff_period]
        region_enc = float(known_rows["region_enc"].mode().iloc[0])
        pt_enc = float(known_rows["pt_enc"].mode().iloc[0])
        prices = seed_rows["price"].to_numpy(dtype=np.float64)
        normalized = (prices - scaler_mean) / scaler_scale
        naive_value = float(prices[-1])
        commodity_actual: list[float] = []
        commodity_model: list[float] = []
        commodity_naive: list[float] = []
        target_scaled = str(meta.get("target_scaled", "false")).lower() == "true"
        input_name = session.get_inputs()[0].name

        for row in after.itertuples(index=False):
            target_date = row.period.to_timestamp()
            context = {
                "year_min": year_min,
                "year_span": year_span,
                "region_enc": region_enc,
                "pt_enc": pt_enc,
            }
            model_input = _window(normalized, target_date, context)[None, :, :]
            raw_prediction = float(session.run(None, {input_name: model_input})[0][0])
            prediction = raw_prediction * scaler_scale + scaler_mean if target_scaled else raw_prediction
            prediction = max(0.0, prediction)
            recent_average = float(np.mean(prices[-SEQUENCE_LENGTH:]))
            if prediction > recent_average * 5:
                prediction = recent_average * 1.5
            actual = float(row.price)
            commodity_actual.append(actual)
            commodity_model.append(prediction)
            commodity_naive.append(naive_value)
            prices = np.append(prices, prediction)
            normalized = np.append(normalized, (prediction - scaler_mean) / scaler_scale)

        model_receipts.append({
            "commodity": commodity,
            "model_sha256": _sha256(model_path),
            "source_checkpoint_sha256": meta.get("source_checkpoint_sha256"),
            "providers": session.get_providers(),
            "target_scaled": target_scaled,
            "model": _regression_metrics(commodity_actual, commodity_model),
            "naive_persistence": _regression_metrics(commodity_actual, commodity_naive),
        })
        all_actual.extend(commodity_actual)
        all_model.extend(commodity_model)
        all_naive.extend(commodity_naive)

    model_metrics = _regression_metrics(all_actual, all_model)
    naive_metrics = _regression_metrics(all_actual, all_naive)
    gate = _publication_gate(model_metrics, naive_metrics, model_count=len(model_receipts))
    result = {
        "schema": "ph-food-price-out-of-time-naive-baseline-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": data_summary,
        "cutoff_proof": cutoff_proof,
        "validation_start": str(cutoff_period + 1),
        "validation_end": str(data_max_period),
        "execution_provider": CPU_PROVIDER,
        "eligible_model_count": len(model_receipts),
        "skipped_model_count": len(skipped),
        "model": model_metrics,
        "naive_persistence": naive_metrics,
        "publication_gate": gate,
        "models": model_receipts,
        "skipped_models": skipped,
        "claim_boundary": (
            "This is a recursive out-of-time backtest from the historical forecast origin. "
            "It compares the exported graphs with a last-observation persistence forecast; "
            "it does not prove future accuracy or suitability for financial decisions."
        ),
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
    parser.add_argument(
        "command", choices=("probe", "export", "benchmark", "validate", "predict", "run-all")
    )
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
    elif args.command == "validate":
        result = validate_models(
            args.data, args.models, args.evidence / "validation_current.json"
        )
    elif args.command == "predict":
        validation_result = validate_models(
            args.data, args.models, args.evidence / "validation_current.json"
        )
        result = generate_predictions(
            args.data,
            args.models,
            args.evidence / ("predictions_cpu.json" if args.cpu else "predictions_current.json"),
            horizon=args.horizon,
            prefer_gpu=not args.cpu,
            validation=validation_result,
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
        validation_result = validate_models(
            args.data, args.models, args.evidence / "validation_current.json"
        )
        prediction_result = generate_predictions(
            args.data, args.models, args.evidence / "predictions_current.json",
            horizon=args.horizon, prefer_gpu=True, validation=validation_result,
        )
        result = {
            "exported_models": export_result["exported_count"],
            "native_gpu_verified": benchmark_result["native_gpu_verified"],
            "prediction_models": prediction_result["model_count"],
            "skipped_prediction_models": prediction_result["skipped_model_count"],
            "forecast_points": prediction_result["forecast_point_count"],
            "prediction_publication_gate": validation_result["publication_gate"],
            "evidence_dir": str(args.evidence.resolve()),
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
