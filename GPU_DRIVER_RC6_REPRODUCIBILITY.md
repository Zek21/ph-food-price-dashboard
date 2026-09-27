# GPU Driver v1.0.0-rc6 reproducibility

rc6 publishes the current leakage-safe driver code and receipts while preserving a strict public-claim boundary: native DirectML placement is verified for the tested ONNX LSTM graph, but the 59-model forward forecast set remains withheld.

## Machine and runtime

- Host: ZEKE
- GPU: AMD Radeon RX 6600
- Windows display-driver version: 32.0.21030.2001
- Python: 3.13.7
- ONNX Runtime: 1.24.4
- Available providers: DmlExecutionProvider, CPUExecutionProvider
- Dataset: `D:\ML\WFP\wfp_food_prices_phl_latest.csv`
- Dataset SHA-256: `9623508dfa1e33c6ac6bda2ceeafca1562679b1e49258df977cb3cd40c147124`
- Dataset maximum date: 2026-06-15

## Strict native-GPU proof

The benchmark criterion is code-enforced: at least one `LSTM` node must be present in the ONNX Runtime profile and every profiled `LSTM` node must run on `DmlExecutionProvider`. The fresh run records 18 node events: 15 on DirectML and 3 on CPU; 2/2 LSTM events are on DirectML.

The tested graph is not faster on this GPU. Batch-1 median was 1.0109 ms DirectML versus 0.1671 ms CPU. CPU remained faster at every measured batch size. This release does not claim GPU-only execution, PyTorch LSTM training on GPU, or end-to-end Python execution on GPU.

Reproduce the strict benchmark on the same local model/data inputs:

```powershell
.\.venv-directml\Scripts\python.exe gpu_forecast_driver.py benchmark `
  --data D:\ML\WFP\wfp_food_prices_phl_latest.csv `
  --model "D:\ML\Website\.onnx_models_oot_202603\lstm_Rice_(regular,_milled).onnx" `
  --evidence gpu_driver_evidence\rerun_20260928_rc6 `
  --iterations 100
```

## Current prediction gate

The 59-model out-of-time gate spans April?June 2026 (177 points). Model MAPE/MAE are 3.9927% / 5.1355 versus persistence 4.3018% / 5.1679. Point estimates favor the model slightly, but the paired MAPE CI is [-0.758292, +0.099301] and the MAE CI is [-0.684032, +0.651491], so neither excludes zero. Fish (threadfin bream) is also a 16.174x persistence-MAE regression. Publication status is therefore **WITHHELD**.

Reproduce validation and the local-only prediction artifact:

```powershell
.\.venv-directml\Scripts\python.exe gpu_forecast_driver.py validate --data D:\ML\WFP\wfp_food_prices_phl_latest.csv --models D:\ML\Website\.onnx_models_oot_202603 --evidence gpu_driver_evidence\rerun_20260928_rc6
.\.venv-directml\Scripts\python.exe gpu_forecast_driver.py predict  --data D:\ML\WFP\wfp_food_prices_phl_latest.csv --models D:\ML\Website\.onnx_models_oot_202603 --evidence gpu_driver_evidence\rerun_20260928_rc6 --horizon 18
```

The second command regenerated 1,062 forward points from 59 models, but the values are not published in rc6 because the gate failed.

## AR(1) diagnostic and harness alignment

The six-origin pooled diagnostic contains 354 commodity-origin rows / 3,717 points. N-weighted MAE: persistence 10.3057, AR(1) 10.6965, LSTM 10.1478. N-weighted MAPE: persistence 8.1550%, AR(1) 8.8216%, LSTM 8.1006%. Thus AR(1) is worse than persistence while the LSTM is slightly better on the pooled diagnostic. This is supportive only because the horizons overlap and are not the publication gate.

The fresh alignment audit reproduces 354/354 point counts and 354/354 naive-MAE pairs within 1%; median absolute naive-MAE difference is 0.00002530.

## Tests

Focused release suite after the rc6 gate fix:

```powershell
D:\ML\env\Scripts\python.exe -m pytest -q tests\test_gpu_forecast_driver.py tests\test_gpu_validation_gate.py tests\test_gpu_gate_paired_significance.py tests\test_gpu_window_alignment.py tests\test_gpu_forecaster_v2_leakage.py tests\test_per_commodity_eligibility.py tests\test_per_commodity_fallback_selector.py tests\test_regional_aggregation_raises_snr.py tests\test_rolling_origin_backtest.py
```

Result: **46 passed, 24 skipped, 1 warning**. The release-training tests separately pass **4/4**. A broader pre-fix worktree run reached **224 passed, 24 skipped** with one path-name-only failure (`Website-rc6` vs hard-coded `Website`), not a GPU/forecast failure.

## Receipt hashes

- strict benchmark: `47d82e96c15c8c6c988d2ea668de1c4b9d310db81283571c925ed40740a20e76`
- validation: `2ebe70e2e3a4b47fa5f8b158200bd48a3ad79070b437ba15d859cc98459e89a1`
- local-only prediction receipt: `57f481099962ea8db826fdae259674cd58afca544afca5d98e57fbe8b444c0b9`
- harness alignment: `9ed4073c766b23fe86f63282a2c97001238b2d2bad2ccfee0b35d803e7a4fab3`
- AR(1) baseline: `24abc102b170ca1c9f1ddd9ad8a8b65fefb3b0e5561f6f67ba788fd6fd9f2153`

The prediction receipt hash is provided for audit continuity; the rc6 public release intentionally does not ship the forecast-value file.
