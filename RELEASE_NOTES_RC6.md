# GPU Driver v1.0.0-rc6 ? strict DirectML proof, forecasts withheld

This prerelease closes a proof-gate regression and publishes the current leakage-safe code and audit receipts.

- Strict device gate: 2/2 profiled LSTM operators on `DmlExecutionProvider`; 15/18 total node events on DirectML.
- No speedup claim: batch-1 median was 1.0109 ms DirectML vs 0.1671 ms CPU; CPU won every measured batch size.
- Current 59-model validation: MAPE 3.9927% vs 4.3018%, MAE 5.1355 vs 5.1679. Paired confidence intervals include zero, and one commodity breaches the 3x MAE guard.
- Prediction artifact regenerated: 1,062 points / 59 models, **WITHHELD** from public release.
- Six-origin diagnostic: persistence MAE 10.3057, AR(1) 10.6965, LSTM 10.1478; supportive only, not the publication gate.
- Harness alignment: 354/354 point-count matches, 100% of naive-MAE pairs within 1%.
- Focused post-fix suite: 46 passed, 24 skipped.

See `GPU_DRIVER_RC6_REPRODUCIBILITY.md` and the hashed receipts. Forecast values are deliberately omitted from the public release.
