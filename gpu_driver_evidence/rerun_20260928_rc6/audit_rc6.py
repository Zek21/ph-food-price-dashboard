"""Ticket-local rc6 audits; prediction values are written only outside the repo."""
import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(__file__).resolve().parent
PRIVATE = Path(r"D:\ML\AR20260720_private_20260928_rc6\predictions_current.json")
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(name, value):
    (EVIDENCE / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def predictions():
    import gpu_forecast_driver as driver
    if Path(sys.executable).resolve() != Path(r"D:\ML\Website\.venv-directml\Scripts\python.exe").resolve():
        raise RuntimeError("Use the registered DirectML interpreter")
    previous = sha(PRIVATE) if PRIVATE.exists() else None
    validation = read(EVIDENCE / "validation_current.json")
    data = Path(r"D:\ML\WFP\wfp_food_prices_phl_latest.csv")
    models = Path(r"D:\ML\Website\.onnx_models_oot_202603")
    assert sha(data) == validation["data"]["sha256"]
    for item in validation["models"]:
        assert sha(models / ("lstm_" + driver._safe_name(item["commodity"]) + ".onnx")) == item["model_sha256"]
    driver.generate_predictions(data, models, PRIVATE, horizon=18, prefer_gpu=True, validation=validation)
    result = read(PRIVATE)
    counts = [len(values) for values in result["forecasts"].values()]
    assert len(counts) == result["model_count"] == len(result["models"]) == 59
    assert sum(counts) == result["forecast_point_count"] == 1062
    assert all(n == 18 for n in counts)
    assert result["publication_gate"] == validation["publication_gate"]
    assert result["publication_gate"]["passed"] is False
    receipt = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "private_path": str(PRIVATE), "previous_private_sha256": previous,
        "private_sha256": sha(PRIVATE), "validation_sha256": sha(EVIDENCE / "validation_current.json"),
        "model_count": len(counts), "forecast_point_count": sum(counts),
        "points_per_model": sorted(set(counts)), "skipped_model_count": result["skipped_model_count"],
        "publication": "WITHHELD", "contains_forward_values": False,
        "publication_gate": result["publication_gate"], "execution": result["execution"],
    }
    write("private_prediction_audit.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("model_count", "forecast_point_count", "publication", "private_sha256")}))


def diagnostic():
    source = ROOT / "gpu_driver_evidence/rolling_origin_20260912"
    rows = read(source / "ar1_baseline.json")
    assert len(rows) == len({(r["origin"], r["commodity"]) for r in rows}) == 354
    origins = sorted({r["origin"] for r in rows})
    assert len(origins) == 6
    receipts = {o: read(source / ("validation_" + o.replace("-", "") + ".json")) for o in origins}
    pairs = {(o, r["commodity"]): r for o, v in receipts.items() for r in v["models"]}
    assert set(pairs) == {(r["origin"], r["commodity"]) for r in rows}
    diffs, counts, within = [], 0, 0
    for row in rows:
        v = pairs[row["origin"], row["commodity"]]
        counts += row["n"] == v["naive_persistence"]["n"] == v["model"]["n"]
        delta = abs(row["naive_mae"] - v["naive_persistence"]["mae"])
        diffs.append(delta)
        within += delta <= abs(row["naive_mae"]) * .01
        assert abs(row["lstm_mae"] - v["model"]["mae"]) < 1e-8
        assert abs(row["lstm_mape"] - v["model"]["mape"]) < 1e-8
    n = sum(r["n"] for r in rows)
    assert n == 3717 and counts == within == 354
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Recomputed from the carried-forward six-origin receipts; no retraining or origin reselection.",
        "pairs": len(rows), "origins": origins, "points": n,
        "point_count_matches": counts, "within_1pct_naive_mae": within,
        "median_abs_naive_mae_diff": statistics.median(diffs), "max_abs_naive_mae_diff": max(diffs),
        "pooled": {key: sum(r[key]*r["n"] for r in rows)/n for key in ("naive_mae", "ar1_mae", "lstm_mae", "naive_mape", "ar1_mape", "lstm_mape")},
        "source_hashes": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in [source / "ar1_baseline.json"] + [source / ("validation_" + o.replace("-", "") + ".json") for o in origins]},
        "verdict": "aligned", "publication_gate": False,
        "claim_boundary": "Supportive only: overlapping horizons; current validation_current.json remains the publication gate.",
    }
    write("harness_alignment.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["predictions", "diagnostic"])
    args = parser.parse_args()
    {"predictions": predictions, "diagnostic": diagnostic}[args.action]()
