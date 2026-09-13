# GPU Driver v1.0.0-rc4 reproducibility

This prerelease refreshes native AMD GPU placement proof from a clean release worktree and hardens the forecast publication gate. It does **not** release forward forecast values.

## Evidence boundary

- Target machine: ZEKE, AMD Radeon RX 6600.
- Source base: `gpu-driver-v1.0.0-rc3` (`739394c45b4d7b4e0edeea05d8df462e989c5b89`) plus the rc4 gate/decoder/test changes in this tag.
- Dataset: WFP/HDX Philippines food prices, SHA-256 `9623508dfa1e33c6ac6bda2ceeafca1562679b1e49258df977cb3cd40c147124`.
- Release ONNX graph used for placement proof: Rice (regular, milled), SHA-256 `51c943ebc7fd79160433931653ad9d7af0f87eac34f1ecf4294b808351009ab2`.
- Training cutoff: 2026-03; publication-test window: 2026-04 through 2026-06.

## Reproduce DirectML proof and the publication gate

Use the pinned release environments described in `requirements-release-training.txt` and `requirements-release-inference.txt`. The source CSV must be available at `D:\ML\WFP\wfp_food_prices_phl_latest.csv` or supplied explicitly.

```powershell
py -3.13 -m venv .venv-release-directml
.\.venv-release-directml\Scripts\python.exe -m pip install -r requirements-release-inference.txt
.\.venv-release-directml\Scripts\python.exe -m pytest tests\test_lstm_release_train.py tests\test_gpu_window_alignment.py tests\test_gpu_forecast_driver.py tests\test_gpu_validation_gate.py tests\test_gpu_gate_paired_significance.py -q
.\.venv-release-directml\Scripts\python.exe gpu_forecast_driver.py benchmark --models .onnx_models_release --model ".onnx_models_release\lstm_Rice_(regular,_milled).onnx" --evidence gpu_driver_evidence\rerun_20260913_resp10 --iterations 30
.\.venv-release-directml\Scripts\python.exe gpu_forecast_driver.py validate --models .onnx_models_release --evidence gpu_driver_evidence\rerun_20260913_resp10
.\.venv-release-directml\Scripts\python.exe gpu_forecast_driver.py predict --models .onnx_models_release --evidence gpu_driver_evidence\rerun_20260913_resp10 --horizon 18
```

On ZEKE the combined release/driver test selection completed with `27 passed, 7 skipped`.

## Native GPU result

ONNX Runtime 1.24.4 exposed `DmlExecutionProvider` and `CPUExecutionProvider`. Per-node profiling recorded 18 node events: 15 on DirectML and 3 on CPU. Both LSTM nodes (`/lstm/LSTM_kernel_time` and `/lstm/LSTM_1_kernel_time`) executed on `DmlExecutionProvider`, so native GPU placement for this exact graph is verified.

The GPU is **not faster** for this compact graph. In this rerun, DirectML median latency was slower than CPU at batch sizes 1, 8, 32 and 128. The claim is device placement, not acceleration, GPU-only execution, training-on-GPU, or performance on other hardware.

Fresh benchmark receipt: `gpu_driver_evidence/rerun_20260913_resp10/benchmark_current.json`, SHA-256 `f95349984f3c63d490f3fcdbf00266c8762554b308bb69ff9869bf836111ed94`.

## Forecast-skill result

The 59-model LSTM set fails the untouched April-June 2026 persistence gate on 177 points:

- LSTM MAPE: 7.2264%; persistence MAPE: 4.3018%.
- LSTM MAE: 8.5919; persistence MAE: 5.1679.
- Paired commodity-cluster bootstrap: MAPE difference +2.924646, 95% CI [+1.432562, +4.435314]; MAE difference +3.423924, 95% CI [+1.594583, +5.296231]. Positive means the model is worse.
- 45 of 59 commodities fail to beat persistence; 15 exceed the 3x persistence-MAE guard, with the worst at 34.002x.
- Publication gate: `withheld_failed_validation`.

Fresh validation receipt: `gpu_driver_evidence/rerun_20260913_resp10/validation_current.json`, SHA-256 `6f45db54702d5705a68db791a65623fa5602a0a49202636636cfa9602c0fb278`.

The DirectML run generated 1,062 local July 2026-December 2027 points, but the gate above failed. `predictions_current.json` is therefore retained only as local diagnostic evidence and is **not** a release asset or an investor/public prediction claim. Local receipt SHA-256: `c9864c5c89c894b5d8cf5c4b7747a9bd73197a34bf3406935afa116b27b36ef8`.
