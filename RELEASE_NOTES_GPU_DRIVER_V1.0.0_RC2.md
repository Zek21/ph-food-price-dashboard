# GPU Driver v1.0.0-rc2

This prerelease hardens the Python DirectML inference driver with an executable
out-of-time naive-baseline publication gate.

## Added

- `gpu_forecast_driver.py validate`
- SHA-256-bound inference of the historical training cutoff from
  `lstm_predictions.json`
- Recursive post-cutoff evaluation of every eligible exported ONNX graph
- Aggregate and per-commodity model-versus-persistence MAPE, MAE, RMSE, and R²
- A typed prediction `publication_gate`
- A fail-closed error that identifies the project-pinned DirectML interpreter
  when another active Python environment exposes only CPU/Azure providers

## Current measured result

The current exported checkpoints failed the new gate on observations after the
historical forecast origin:

- evaluation window: February–June 2026
- eligible models: 59
- observations: 295
- LSTM MAPE: 85.6945%
- naive-persistence MAPE: 7.5142%
- LSTM MAE: 141.9462
- naive-persistence MAE: 8.4031
- publication status: `withheld_failed_validation`

The DirectML rerun still produced 1,062 local experimental values from 59
current models and recorded 13 stale/incomplete model series. Those values are
not release assets and are not public forecast evidence.

## Unchanged execution evidence

The previously released node-profile and latency benchmark remains applicable
to the same exported Rice LSTM graph:

- 15 of 18 profiled node events used `DmlExecutionProvider`
- both LSTM nodes used DirectML
- three nodes used CPU fallback
- GPU/CPU maximum absolute output difference: `0.00000381`
- CPU median latency remained lower at every measured batch

DirectML placement is not called acceleration. This is an ML execution driver,
not a Windows display or kernel driver.

## Release assets

- `benchmark_current.json`
- `benchmark_latency.svg`
- `export_manifest.json`
- `validation_current.json`

Predictions and model bundles are intentionally excluded.
