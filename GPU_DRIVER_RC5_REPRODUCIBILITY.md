# GPU Driver v1.0.0-rc5 reproducibility

This prerelease hardens the native-GPU proof gate introduced in rc4. The model set, dataset, and publication decision are unchanged: the exact ONNX LSTM graph is demonstrably placed on DirectML for its LSTM operators, while the forward forecasts remain withheld because the 59-model set does not beat persistence out of time.

## What changed from rc4

rc4 correctly recorded per-node placement showing both LSTM operators on DirectML, but the code-level `native_gpu_verified` boolean only required at least one DirectML node event. rc5 closes that proof gap: a benchmark is verified only when at least one LSTM node is present in the profile and every profiled LSTM node is placed on `DmlExecutionProvider`.

Source base: `gpu-driver-v1.0.0-rc4` / commit `9eeda76268e78366dff15b2de90ef0c12d01791c`.

## Verification on ZEKE

Target machine: ZEKE, AMD Radeon RX 6600.

Release-oriented test selection:

```powershell
D:\ML\Website\.venv-torch-dml\Scripts\python.exe -m pytest tests\test_lstm_release_train.py tests\test_gpu_window_alignment.py tests\test_gpu_forecast_driver.py tests\test_gpu_validation_gate.py tests\test_gpu_gate_paired_significance.py -q
```

Result: `29 passed, 7 skipped`.

DirectML-driver focused tests:

```powershell
D:\ML\Website\.venv-directml\Scripts\python.exe -m pytest tests\test_gpu_forecast_driver.py -q
```

Result: `11 passed`.

## Strict native GPU placement proof

The benchmark was rerun against the release Rice (regular, milled) ONNX graph, SHA-256 `51c943ebc7fd79160433931653ad9d7af0f87eac34f1ecf4294b808351009ab2`, using ONNX Runtime 1.24.4 with `DmlExecutionProvider` and `CPUExecutionProvider` available.

Strict result:

- 18 profiled node events total.
- 15 node events on DirectML and 3 on CPU.
- 2 LSTM node events profiled.
- 2 of 2 LSTM node events placed on `DmlExecutionProvider`.
- `native_gpu_verified: true` under the rc5 strict LSTM-placement criterion.

Fresh strict benchmark receipt: `gpu_driver_evidence/rerun_20260913_resp10_strict/benchmark_current.json`, SHA-256 `fc60f74a9414213a1dcb50125c93494db913554a53a7c02ebd11fdc17e4ced1c`.

This proves native DirectML execution for the LSTM operators in this exact inference graph. It does not prove GPU-only execution, PyTorch LSTM training on GPU, end-to-end Python execution on GPU, or acceleration. In this benchmark CPU latency remained lower at every tested batch size.

## Current prediction gate

The 59-model validation was rerun over April-June 2026, 177 out-of-time points. The result is unchanged and remains unsuitable for publication as a predictive claim:

- LSTM MAPE: 7.2264%; persistence MAPE: 4.3018%.
- LSTM MAE: 8.5919; persistence MAE: 5.1679.
- MAPE paired difference: +2.924646, 95% CI [+1.432562, +4.435314].
- MAE paired difference: +3.423924, 95% CI [+1.594583, +5.296231].
- 45/59 commodities fail to beat persistence.
- 15 commodities exceed the 3x persistence-MAE guard; worst ratio 34.002x.
- Publication status: `withheld_failed_validation`.

Fresh validation receipt: `gpu_driver_evidence/rerun_20260913_resp10_strict/validation_current.json`, SHA-256 `e674a6a7c244389bb8a76ea87d6f5afb98e19bb3c348f89d98ca645f005af334`.

The DirectML prediction pass produced 1,062 local diagnostic forecast points for July 2026-December 2027, but they remain withheld and are not release assets or investor/public prediction claims. Local-only receipt SHA-256: `3eece032ed656199750dc15c7da45b70988f695297e2701603a81d1d95d56e2f`.

## Claim boundary

Public material may truthfully say that the release LSTM inference graph has native DirectML placement evidence on ZEKE's AMD Radeon RX 6600, including both profiled LSTM operators. It must also say that this compact graph was slower than CPU in the measured benchmark and that the current forward forecasts are withheld because the model failed the persistence baseline gate.
