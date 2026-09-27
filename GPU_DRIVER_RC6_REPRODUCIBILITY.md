# GPU Driver rc6 reproducibility

Ticket: AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS. Run date: 2026-09-28 Asia/Manila.

## Inputs and runtime

ZEKE; AMD Radeon RX 6600; display driver 32.0.21030.2001; Python 3.13.7; ONNX Runtime 1.24.4. `hardware.json` and `benchmark_current.json` record the observations. Provider availability alone is not placement proof.

Registered DirectML Python: `D:\ML\Website\.venv-directml\Scripts\python.exe` (registered in the repository's 2026-07-20 interpreter proposal). Registered test Python: `D:\ML\env\Scripts\python.exe` (carried-forward rc6 test command). Commands run from `D:\ML\Website-rc6`; inputs in the existing Website checkout are read only, with bytecode writes disabled.

Dataset: `D:\ML\WFP\wfp_food_prices_phl_latest.csv`; maximum date 2026-06-15; SHA-256 `9623508dfa1e33c6ac6bda2ceeafca1562679b1e49258df977cb3cd40c147124`. Models: `D:\ML\Website\.onnx_models_oot_202603`; 59 model/checkpoint hashes and the explicit 2026-03 training cutoff are in the validation receipt. No retraining or model selection was performed.

## Reproduce

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& D:\ML\Website\.venv-directml\Scripts\python.exe gpu_forecast_driver.py benchmark --data D:\ML\WFP\wfp_food_prices_phl_latest.csv --model 'D:\ML\Website\.onnx_models_oot_202603\lstm_Rice_(regular,_milled).onnx' --evidence gpu_driver_evidence/rerun_20260928_rc6 --iterations 100
& D:\ML\Website\.venv-directml\Scripts\python.exe gpu_forecast_driver.py validate --data D:\ML\WFP\wfp_food_prices_phl_latest.csv --models D:\ML\Website\.onnx_models_oot_202603 --evidence gpu_driver_evidence/rerun_20260928_rc6
& D:\ML\Website\.venv-directml\Scripts\python.exe gpu_driver_evidence/rerun_20260928_rc6/audit_rc6.py predictions
& D:\ML\env\Scripts\python.exe gpu_driver_evidence/rerun_20260928_rc6/audit_rc6.py diagnostic
```

The prediction audit calls the real driver's `generate_predictions` with the freshly validated data/model hashes and writes values only to `D:\ML\AR20260720_private_20260928_rc6\predictions_current.json`. It independently reopens that file, checks 59 models and 18 points each (1,062 total), and publishes only counts, hashes and gate metadata. Its private SHA-256 is `09444f44624016ad5410a979803a9bf7d332fdec49f92cd95a6967dcf0c26b58`. Do not run `predict --evidence` against a public repo directory: that CLI writes forward values there.

## Measured truth

The profile lists individual node names, operations and providers: 2/2 LSTM events on DirectML; 15/18 total events on DirectML; 3 CPU fallback events. Batch-1 median: 1.0352 ms DirectML / 0.1740 ms CPU. CPU won at batches 1, 8, 32 and 128 (100 measured iterations each, 10 warmups). This proves native inference placement for the tested graph only; it does not prove GPU-only execution, PyTorch GPU training, or a speedup.

The authoritative validation uses recursive out-of-time April-June 2026 predictions, 59 models / 177 points, and metadata declaring a train-only scaler and 2026-03 cutoff. MAPE: 3.9927% versus 4.3018%; MAE: 5.1355 versus 5.1679. Paired MAPE CI [-0.758292, +0.099301]; MAE CI [-0.684032, +0.651491]. Fish (threadfin bream) is 16.174x persistence MAE. The unchanged gate is `withheld_failed_validation`; forward forecasts remain **WITHHELD**.

The diagnostic audit re-computes the carried-forward six-origin receipts, with source hashes: 354 commodity-origin rows / 3,717 points, 354/354 count matches and naive-MAE pairs within 1%. Pooled MAE: persistence 10.3057, AR(1) 10.6965, LSTM 10.1478. Pooled MAPE: 8.1550%, 8.8216%, 8.1006%, respectively. The LSTM's small pooled advantage is supportive only; horizons overlap and this diagnostic is not the publication gate.

## Verification

Focused suite including release-training, current rc6 gate and updater tests: 85 passed, 24 skipped, 1 warning in 10.25s.
Full suite: 228 passed, 24 skipped, 1 warning in 153.56s (0:02:33).

```powershell
$env:PYTHONIOENCODING='utf-8'
& D:\ML\env\Scripts\python.exe -m pytest -q -ra -p no:cacheprovider --basetemp D:\ML\Website-rc6\.pytest_rc6_broad_final_20260928_0640 tests
```

Use a fresh task-local `--basetemp` for each rerun. The first focused run hit an inaccessible shared pytest temporary directory; the first full run found the hard-coded checkout-name assertion. Both failure logs are retained. The assertion now checks the script's actual directory. A new rc6 receipt test recomputes the exact unchanged gate; 24 tests still skip unavailable older receipts. A PyTorch scalar-conversion warning remains in an optimizer unit test and is not GPU-training evidence.

## Publication and hashes

The current 59-model rerun values are absent from the public payload and Git index. Legacy GPU/LSTM forward sections were removed from the new branch tip. Existing published Git history/tag archives and unrelated classical dashboard forecasts are not rewritten; this is not a claim that the entire historical repository contains no forecasts.

The existing public rc6 tag is preserved at `1a5130fb8b127b43ac9e46255947f4b4ddca084d`; the refreshed audit is attached to that release and committed on its existing branch. `release_payload.json` explicitly lists the public files. `SHA256SUMS-rc6.txt` hashes their local bytes, excluding itself and the allowlist to avoid self-reference. The bundle contains docs, code, tests and audit receipts, never the private prediction artifact. Receipt hashes and claim boundaries are in the fresh `release_truth_manifest.json`.
