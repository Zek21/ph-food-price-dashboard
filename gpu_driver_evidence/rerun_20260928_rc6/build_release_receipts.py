"""Build the rc6 public documentation and value-free release allowlist."""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from audit_rc6 import ROOT, EVIDENCE, read, sha, write


# Fixed to the existing c3d9d00 release policy. Campaign sources and new local
# evidence must never become release assets through a directory sweep.
RELEASE_FILES = [
    ".gitignore",
    "GPU_DRIVER.md",
    "GPU_DRIVER_RC6_REPRODUCIBILITY.md",
    "INVESTOR_CAMPAIGN_RC6.md",
    "RELEASE_NOTES_RC6.md",
    "gpu_driver_evidence/release_summary_rc6.json",
    "gpu_driver_evidence/rerun_20260928_rc6/audit_rc6.py",
    "gpu_driver_evidence/rerun_20260928_rc6/benchmark_current.json",
    "gpu_driver_evidence/rerun_20260928_rc6/benchmark_stderr.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/benchmark_stdout.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/build_release_receipts.py",
    "gpu_driver_evidence/rerun_20260928_rc6/github_identity_before.json",
    "gpu_driver_evidence/rerun_20260928_rc6/hardware.json",
    "gpu_driver_evidence/rerun_20260928_rc6/harness_alignment.json",
    "gpu_driver_evidence/rerun_20260928_rc6/legacy_forecast_sanitization.json",
    "gpu_driver_evidence/rerun_20260928_rc6/payload_audit.json",
    "gpu_driver_evidence/rerun_20260928_rc6/prediction_audit_stderr.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/prediction_audit_stdout.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/private_prediction_audit.json",
    "gpu_driver_evidence/rerun_20260928_rc6/release_truth_manifest.json",
    "gpu_driver_evidence/rerun_20260928_rc6/rollback_identity_20260928T062736583.json",
    "gpu_driver_evidence/rerun_20260928_rc6/test_worktree_path_before.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/tests_broad_final.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/tests_broad_initial.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/tests_focused.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/tests_focused_final.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/validation_current.json",
    "gpu_driver_evidence/rerun_20260928_rc6/validation_stderr.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/validation_stdout.txt",
    "gpu_driver_evidence/rolling_origin_20260912/ar1_baseline.json",
    "gpu_driver_evidence/rolling_origin_20260912/validation_202412.json",
    "gpu_driver_evidence/rolling_origin_20260912/validation_202503.json",
    "gpu_driver_evidence/rolling_origin_20260912/validation_202506.json",
    "gpu_driver_evidence/rolling_origin_20260912/validation_202509.json",
    "gpu_driver_evidence/rolling_origin_20260912/validation_202512.json",
    "gpu_driver_evidence/rolling_origin_20260912/validation_202603.json",
    "gpu_forecast_driver.py",
    "tests/test_daily_update.py",
    "tests/test_gpu_forecast_driver.py",
    "tests/test_gpu_gate_paired_significance.py",
    "tests/test_gpu_validation_gate.py",
    "SHA256SUMS-rc6.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/release_payload.json",
    "gpu_driver_evidence/rerun_20260928_rc6/codex_doer_report.txt",
    "gpu_driver_evidence/rerun_20260928_rc6/publication_after.json"
]


def refresh_campaign_checksums():
    """Keep engineering receipts and payload membership unchanged for copy edits."""
    manifest = read(EVIDENCE / "release_truth_manifest.json")
    for name, expected in manifest["receipt_sha256"].items():
        assert sha(ROOT / name) == expected, f"Receipt changed: {name}"
    payload = read(EVIDENCE / "release_payload.json")
    assert payload["files"] == RELEASE_FILES, "Release scope changed; review explicitly"
    assert manifest["publication_gate"]["status"] == "withheld_failed_validation"
    assert not manifest["publication_gate"]["passed"]
    paths = [name for name in RELEASE_FILES if name not in
             ("SHA256SUMS-rc6.txt", "gpu_driver_evidence/rerun_20260928_rc6/release_payload.json")]
    for name in paths:
        if name.endswith(".json"):
            doc = read(ROOT / name)
            assert not (isinstance(doc, dict) and doc.get("forecasts")), name
    (ROOT / "SHA256SUMS-rc6.txt").write_text(
        "".join(f"{sha(ROOT / name)}  {name}\n" for name in paths), encoding="utf-8")
    print(json.dumps({"mode": "campaign-only", "payload_files": len(RELEASE_FILES),
                      "checksum_entries": len(paths), "engineering_receipts_unchanged": True}))


if "--campaign-only" in sys.argv:
    refresh_campaign_checksums()
    raise SystemExit(0)


def test_result(name):
    text = (EVIDENCE / name).read_text(encoding="utf-8-sig")
    result = [line for line in text.splitlines() if re.search(r"\d+ passed", line)][-1]
    assert "failed" not in result and "error" not in result, result
    return result


b = read(EVIDENCE / "benchmark_current.json")
v = read(EVIDENCE / "validation_current.json")
p = read(EVIDENCE / "private_prediction_audit.json")
h = read(EVIDENCE / "harness_alignment.json")
focused = test_result("tests_focused_final.txt")
broad = test_result("tests_broad_final.txt")
assert b["native_gpu_verified"] and not v["publication_gate"]["passed"]
assert all(r["gpu_directml"]["median_ms"] > r["cpu"]["median_ms"] for r in b["results"])
assert p["model_count"] == 59 and p["forecast_point_count"] == 1062
latency = b["results"][0]
gpu, cpu = latency["gpu_directml"]["median_ms"], latency["cpu"]["median_ms"]
prefix = "gpu_driver_evidence/rerun_20260928_rc6/"
allowed = [
    "Native DirectML placement for the tested ONNX LSTM graph on ZEKE AMD Radeon RX 6600.",
    "Publication requires a predeclared untouched holdout; the current April-June 2026 window is already inspected.",
    "Both profiled LSTM operators on DmlExecutionProvider; 15/18 node events on DirectML and 3 on CPU.",
    "The tested compact graph is slower on DirectML than CPU at all four measured batch sizes.",
    "The 59-model / 1,062-point current forward set is WITHHELD and private.",
    "AR(1) is worse than persistence; LSTM slightly better on pooled MAE/MAPE in the supportive six-origin diagnostic only.",
]
prohibited = ["GPU-only execution", "GPU speedup for this tested graph", "PyTorch GPU training from this proof", "validated forward prices", "investor guidance from withheld values", "calling the pooled diagnostic the publication gate", "treating the inspected April-June 2026 window as untouched publication evidence"]
scope = ("The current 59-model rerun values are absent from the public payload and Git index. "
         "Legacy GPU/LSTM forward sections were removed from the new branch tip. "
         "Existing published Git history/tag archives and unrelated classical dashboard forecasts are not rewritten; "
         "this is not a claim that the entire historical repository contains no forecasts.")
notes = f"""# GPU Driver v1.0.0-rc6 - fresh ZEKE rerun, forecasts withheld

Ticket: AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS. Fresh run: 2026-09-28 Asia/Manila.

- Per-node proof: both LSTM operators on `DmlExecutionProvider`; 15/18 node events on DirectML, 3 on CPU.
- The tested compact graph was slower on DirectML at all measured batch sizes. Batch-1 medians: {gpu:.4f} ms DirectML versus {cpu:.4f} ms CPU.
- Authoritative April-June 2026 validation: 59 models, 177 points; MAPE 3.9927% versus persistence 4.3018%, MAE 5.1355 versus 5.1679. Both paired 95% intervals cross zero; one commodity breaches the unchanged 3x MAE guard.
- Current forward set: 59 models / 1,062 points, regenerated only in the private directory and **WITHHELD**.
- Publication provenance guard: the April-June 2026 window is already inspected. It cannot become publication permission after adaptive tuning; a future pass requires a predeclared untouched holdout.
- Supportive six-origin diagnostic: pooled MAE persistence 10.3057, AR(1) 10.6965, LSTM 10.1478. All 354 point counts match; all naive-MAE pairs agree within 1%. This is not the publication gate.
- Focused tests: {focused}. Full repo suite: {broad}. Skips identify absent historical receipts, not passing checks. Initial failures are preserved.

No GPU-only, GPU speedup, PyTorch GPU-training, validated-price, or investor-guidance claim is made.

The already-published rc6 tag remains at `1a5130fb8b127b43ac9e46255947f4b4ddca084d`. Fresh docs and receipts are supplied as release assets and in the `release/gpu-driver-v1.0.0-rc6` branch; the original tag source archive is historical. {scope}

Use `GPU_DRIVER_RC6_REPRODUCIBILITY.md`, `release_truth_manifest.json`, and `SHA256SUMS-rc6.txt` in the attached audit bundle.
"""
(ROOT / "RELEASE_NOTES_RC6.md").write_text(notes, encoding="utf-8")
repro = f"""# GPU Driver rc6 reproducibility

Ticket: AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS. Run date: 2026-09-28 Asia/Manila.

## Inputs and runtime

ZEKE; AMD Radeon RX 6600; display driver 32.0.21030.2001; Python 3.13.7; ONNX Runtime 1.24.4. `hardware.json` and `benchmark_current.json` record the observations. Provider availability alone is not placement proof.

Registered DirectML Python: `D:\\ML\\Website\\.venv-directml\\Scripts\\python.exe` (registered in the repository's 2026-07-20 interpreter proposal). Registered test Python: `D:\\ML\\env\\Scripts\\python.exe` (carried-forward rc6 test command). Commands run from `D:\\ML\\Website-rc6`; inputs in the existing Website checkout are read only, with bytecode writes disabled.

Dataset: `D:\\ML\\WFP\\wfp_food_prices_phl_latest.csv`; maximum date 2026-06-15; SHA-256 `{v['data']['sha256']}`. Models: `D:\\ML\\Website\\.onnx_models_oot_202603`; 59 model/checkpoint hashes and the explicit 2026-03 training cutoff are in the validation receipt. No retraining or model selection was performed.

## Reproduce

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& D:\\ML\\Website\\.venv-directml\\Scripts\\python.exe gpu_forecast_driver.py benchmark --data D:\\ML\\WFP\\wfp_food_prices_phl_latest.csv --model 'D:\\ML\\Website\\.onnx_models_oot_202603\\lstm_Rice_(regular,_milled).onnx' --evidence gpu_driver_evidence/rerun_20260928_rc6 --iterations 100
& D:\\ML\\Website\\.venv-directml\\Scripts\\python.exe gpu_forecast_driver.py validate --data D:\\ML\\WFP\\wfp_food_prices_phl_latest.csv --models D:\\ML\\Website\\.onnx_models_oot_202603 --evidence gpu_driver_evidence/rerun_20260928_rc6
& D:\\ML\\Website\\.venv-directml\\Scripts\\python.exe gpu_driver_evidence/rerun_20260928_rc6/audit_rc6.py predictions
& D:\\ML\\env\\Scripts\\python.exe gpu_driver_evidence/rerun_20260928_rc6/audit_rc6.py diagnostic
```

The prediction audit calls the real driver's `generate_predictions` with the freshly validated data/model hashes and writes values only to `D:\\ML\\AR20260720_private_20260928_rc6\\predictions_current.json`. It independently reopens that file, checks 59 models and 18 points each (1,062 total), and publishes only counts, hashes and gate metadata. Its private SHA-256 is `{p['private_sha256']}`. Do not run `predict --evidence` against a public repo directory: that CLI writes forward values there.

## Measured truth

The profile lists individual node names, operations and providers: 2/2 LSTM events on DirectML; 15/18 total events on DirectML; 3 CPU fallback events. Batch-1 median: {gpu:.4f} ms DirectML / {cpu:.4f} ms CPU. CPU won at batches 1, 8, 32 and 128 (100 measured iterations each, 10 warmups). This proves native inference placement for the tested graph only; it does not prove GPU-only execution, PyTorch GPU training, or a speedup.

The authoritative validation uses recursive out-of-time April-June 2026 predictions, 59 models / 177 points, and metadata declaring a train-only scaler and 2026-03 cutoff. MAPE: 3.9927% versus 4.3018%; MAE: 5.1355 versus 5.1679. Paired MAPE CI [-0.758292, +0.099301]; MAE CI [-0.684032, +0.651491]. Fish (threadfin bream) is 16.174x persistence MAE. The metric gate remains withheld_failed_validation; forward forecasts remain **WITHHELD**.

The April-June 2026 window has already been inspected during repeated hardening. The executable publication path now records prospective_validation=false and can never grant publication from that reused window even if later adaptive tuning makes the metrics look better. A future publication pass requires a predeclared untouched validation window; the CLI exposes no bypass flag.

The diagnostic audit re-computes the carried-forward six-origin receipts, with source hashes: 354 commodity-origin rows / 3,717 points, 354/354 count matches and naive-MAE pairs within 1%. Pooled MAE: persistence 10.3057, AR(1) 10.6965, LSTM 10.1478. Pooled MAPE: 8.1550%, 8.8216%, 8.1006%, respectively. The LSTM's small pooled advantage is supportive only; horizons overlap and this diagnostic is not the publication gate.

## Verification

Focused suite including release-training, current rc6 gate and updater tests: {focused}.
Full suite: {broad}.

```powershell
$env:PYTHONIOENCODING='utf-8'
& D:\\ML\\env\\Scripts\\python.exe -m pytest -q -ra -p no:cacheprovider --basetemp D:\\ML\\Website-rc6\\.pytest_rc6_broad_final_20260928_0640 tests
```

Use a fresh task-local `--basetemp` for each rerun. The first focused run hit an inaccessible shared pytest temporary directory; the first full run found the hard-coded checkout-name assertion. Both failure logs are retained. The assertion now checks the script's actual directory. A new rc6 receipt test recomputes the exact unchanged gate; 24 tests still skip unavailable older receipts. A PyTorch scalar-conversion warning remains in an optimizer unit test and is not GPU-training evidence.

## Publication and hashes

{scope}

The existing public rc6 tag is preserved at `1a5130fb8b127b43ac9e46255947f4b4ddca084d`; the refreshed audit is attached to that release and committed on its existing branch. `release_payload.json` explicitly lists the public files. `SHA256SUMS-rc6.txt` hashes their local bytes, excluding itself and the allowlist to avoid self-reference. The bundle contains docs, code, tests and audit receipts, never the private prediction artifact. Receipt hashes and claim boundaries are in the fresh `release_truth_manifest.json`.
"""
(ROOT / "GPU_DRIVER_RC6_REPRODUCIBILITY.md").write_text(repro, encoding="utf-8")
# INVESTOR_CAMPAIGN_RC6.md is an editorial source; never overwrite it here.
old = (ROOT / "GPU_DRIVER.md").read_text(encoding="utf-8")
marker = "## Historical verification update"
(ROOT / "GPU_DRIVER.md").write_text("# AMD DirectML food-price inference driver\n\n" + notes.split("\n", 1)[1] + "\n" + marker + old.split(marker, 1)[1], encoding="utf-8")
receipts = ["benchmark_current.json", "validation_current.json", "harness_alignment.json", "private_prediction_audit.json", "hardware.json", "tests_focused_final.txt", "tests_broad_final.txt", "legacy_forecast_sanitization.json"]
manifest = {
    "schema": "ar20260720.rc6.truth.v2", "ticket": "AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS",
    "generated_at": datetime.now(timezone.utc).isoformat(), "host": "ZEKE", "gpu": "AMD Radeon RX 6600",
    "base_commit": "1a5130fb8b127b43ac9e46255947f4b4ddca084d", "branch": "release/gpu-driver-v1.0.0-rc6",
    "tag_policy": "Preserve existing published rc6 tag; refresh release assets and link the new audit commit.",
    "native_gpu": b["native_gpu_verification"], "profile_by_provider": b["profile"]["by_provider"],
    "benchmark_results": b["results"], "validation_model": v["model"], "validation_persistence": v["naive_persistence"],
    "publication_gate": v["publication_gate"], "predictions": p, "supportive_diagnostic": h,
    "tests": {"focused": focused, "broad": broad},
    "receipt_sha256": {prefix + name: sha(EVIDENCE / name) for name in receipts},
    "scope_limit": scope, "allowed_claims": allowed, "prohibited_claims": prohibited,
}
write("release_truth_manifest.json", manifest)
(ROOT / "gpu_driver_evidence/release_summary_rc6.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
paths = [name for name in RELEASE_FILES if name not in
         ("SHA256SUMS-rc6.txt", prefix + "release_payload.json")]
for name in paths:
    if name.endswith(".json"):
        doc = read(ROOT / name)
        assert not (isinstance(doc, dict) and doc.get("forecasts")), name
assert not subprocess.check_output(["git", "ls-files", "*predictions_current.json"], cwd=ROOT, text=True).strip()
(ROOT / "SHA256SUMS-rc6.txt").write_text("".join(f"{sha(ROOT / name)}  {name}\n" for name in paths), encoding="utf-8")
write("release_payload.json", {"contains_current_forward_values": False, "scope_limit": scope, "files": RELEASE_FILES})
print(json.dumps({"focused": focused, "broad": broad, "payload_files": len(paths) + 2, "publication_gate": v["publication_gate"]["status"]}, indent=2))
