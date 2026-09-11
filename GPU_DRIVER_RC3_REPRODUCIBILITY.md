# GPU Driver v1.0.0-rc3 reproducibility

This prerelease repairs the LSTM release path and records a failed prediction-publication gate. It does **not** release forecast values.

## Evidence boundary

- Target machine: ZEKE, AMD Radeon RX 6600.
- DirectML proof: ONNX Runtime per-node profiling on the freshly retrained Rice LSTM graph.
- Training cutoff: 2026-03.
- Scaler/model-fit targets end: 2025-09.
- Internal model-selection window: 2025-10 through 2026-03.
- Untouched publication-test window used once for this candidate: 2026-04 through 2026-06.
- Dataset: WFP/HDX Philippines food prices, SHA-256 `9623508dfa1e33c6ac6bda2ceeafca1562679b1e49258df977cb3cd40c147124`.
- Dataset-specific HDX license metadata: `cc-by-igo`, Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO), legal code http://creativecommons.org/licenses/by/3.0/igo/legalcode.

## Reproduce the repaired training/export path

Use Python 3.12 on Windows:

```powershell
py -3.12 -m venv .venv-release-training
.\.venv-release-training\Scripts\python.exe -m pip install -r requirements-release-training.txt
.\.venv-release-training\Scripts\python.exe -m pytest tests\test_lstm_release_train.py -q
.\.venv-release-training\Scripts\python.exe lstm_release_train.py --epochs 50 --publication-cutoff 2026-03 --internal-val-months 6 --model-dir .lstm_models_release --receipt gpu_driver_evidence\lstm_retrain_receipt.json
.\.venv-release-training\Scripts\python.exe gpu_forecast_driver.py export --checkpoints .lstm_models_release --models .onnx_models_release --evidence gpu_driver_evidence
```

## Reproduce DirectML validation and benchmark

Use a separate Python 3.13 environment on Windows so `onnxruntime-directml` does not collide with another `onnxruntime` package:

```powershell
py -3.13 -m venv .venv-release-directml
.\.venv-release-directml\Scripts\python.exe -m pip install -r requirements-release-inference.txt
.\.venv-release-directml\Scripts\python.exe -m pytest tests\test_gpu_window_alignment.py tests\test_gpu_forecast_driver.py tests\test_gpu_validation_gate.py -q
.\.venv-release-directml\Scripts\python.exe gpu_forecast_driver.py validate --models .onnx_models_release --evidence gpu_driver_evidence
.\.venv-release-directml\Scripts\python.exe gpu_forecast_driver.py benchmark --models .onnx_models_release --iterations 50 --evidence gpu_driver_evidence
```

The source CSV must be available at `D:\ML\WFP\wfp_food_prices_phl_latest.csv` or the driver/trainer data path must be supplied explicitly.

## Current measured result

The corrected models improve substantially over rc2 but still fail the required out-of-time persistence baseline on the 177 April-June 2026 observations:

- LSTM MAPE: 7.2264%
- persistence MAPE: 4.3018%
- LSTM MAE: 8.5919
- persistence MAE: 5.1679
- publication gate: `withheld_failed_validation`

The 1,062 July 2026-December 2027 experimental forecast points generated locally are therefore withheld and are not release assets.

## Native GPU result

The freshly retrained Rice ONNX graph has SHA-256 `51c943ebc7fd79160433931653ad9d7af0f87eac34f1ecf4294b808351009ab2`. Per-node profiling recorded 18 node events: 15 on `DmlExecutionProvider` and 3 CPU fallback events; both LSTM nodes executed with DirectML. GPU and CPU outputs differed by at most `2.4e-07` in the benchmark.

CPU median latency was lower at every measured batch size, so this release makes **no GPU speedup claim**. The evidence proves native DirectML graph execution on the tested RX 6600, not acceleration, training-on-GPU, or performance on other hardware.
