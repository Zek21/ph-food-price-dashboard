# Investor campaign claims — GPU Driver v1.0.0-rc5

Use this wording as the evidence boundary for the Skynet Blog, companion video, and social posts. It is intentionally narrower than older campaign material.

## Approved core claim

On ZEKE's AMD Radeon RX 6600, the current Philippine food-price LSTM inference graph has native DirectML execution evidence. ONNX Runtime 1.24.4 profiled 18 graph node events: 15 on `DmlExecutionProvider` and 3 on CPU. Both profiled LSTM operators (2/2) ran on DirectML. GPU Driver v1.0.0-rc5 now requires this LSTM-operator placement, rather than merely detecting any DirectML node, before `native_gpu_verified` can be true.

Public release: https://github.com/Zek21/ph-food-price-dashboard/releases/tag/gpu-driver-v1.0.0-rc5

## Performance boundary

Do **not** describe this LSTM inference graph as GPU-accelerated relative to CPU. In the fresh rc5 benchmark, CPU median latency was lower at every tested batch size. The truthful story is verified GPU placement and reproducible heterogeneous execution, not a speedup claim.

The benchmark proves execution placement for this exact ONNX graph on this host. It does not prove GPU-only execution, Python/CSV processing on GPU, PyTorch LSTM training on GPU, or performance on other GPUs.

## Prediction boundary

The current 59-model LSTM release fails its untouched April-June 2026 persistence gate over 177 out-of-time points:

- LSTM MAPE 7.2264% vs persistence 4.3018%.
- LSTM MAE 8.5919 vs persistence 5.1679.
- MAPE paired difference +2.924646, 95% CI [+1.432562, +4.435314].
- MAE paired difference +3.423924, 95% CI [+1.594583, +5.296231].
- 45/59 commodities fail to beat persistence.
- 15 commodities exceed the 3x persistence-MAE guard.

Therefore the current LSTM forward predictions remain `withheld_failed_validation`. The local DirectML run produced 1,062 diagnostic forecast points, but none of those values should be used as investor/public prediction claims or release assets.

This does not invalidate separately versioned v2-forecaster experiments. Any v2 claim must remain explicitly labeled as a different model and supported by its own leakage-audited receipts; it must never be attributed to the current rc5 LSTM release.

## Blog wording

A defensible update is:

> September 13, 2026: the current LSTM release now has stricter GPU proof. On ZEKE's AMD Radeon RX 6600, both profiled LSTM operators ran through ONNX Runtime's DirectML provider (15 of 18 total graph node events were on DirectML). This proves native GPU placement for the exact inference graph, not GPU-only execution or a speed advantage; CPU remained faster for this compact graph. Forecast skill is a separate gate, and the current 59-model LSTM did not beat naive persistence on the untouched April-June 2026 test window, so its forward forecasts remain withheld.

## Video wording

Show the benchmark receipt and release page while saying:

> We verified the actual LSTM operators, not just that a GPU provider exists. Both LSTM nodes were placed on DirectML on the RX 6600. The CPU was still faster for this small graph, and the model failed the forecast-skill gate, so we are publishing the execution proof but withholding its forward predictions.

## Social wording

> New reproducible GPU proof on an AMD RX 6600: both LSTM operators in our release inference graph were profiled on DirectML. We also tightened the driver so a random GPU node can no longer make the native-GPU gate pass. Important boundary: CPU latency was lower for this compact graph, and the current 59-model LSTM failed its persistence baseline, so its forward forecasts remain withheld. Evidence + hashes: GPU Driver v1.0.0-rc5.

## Evidence

- Strict benchmark: `gpu_driver_evidence/rerun_20260913_resp10_strict/benchmark_current.json`, SHA-256 `fc60f74a9414213a1dcb50125c93494db913554a53a7c02ebd11fdc17e4ced1c`.
- Validation: `gpu_driver_evidence/rerun_20260913_resp10_strict/validation_current.json`, SHA-256 `e674a6a7c244389bb8a76ea87d6f5afb98e19bb3c348f89d98ca645f005af334`.
- Local-only withheld prediction receipt: SHA-256 `3eece032ed656199750dc15c7da45b70988f695297e2701603a81d1d95d56e2f`.
- Release-oriented tests: 29 passed, 7 skipped.
- Focused DirectML-driver tests: 11 passed.
