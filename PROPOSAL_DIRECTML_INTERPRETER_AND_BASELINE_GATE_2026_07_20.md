# Proposal: DirectML Interpreter Pinning and Forecast Baseline Gate

Date: 2026-07-20
Owner: Codex / Skynet
Incident receipt: guarded bus fingerprint `cf08085c8a7c62e8`

## Failure

The first prediction rerun after adding the out-of-time validation command used
the inherited `D:\Prospects\env\Scripts\python.exe`. That environment contained
ONNX Runtime 1.24.3 with Azure and CPU providers only. The DirectML driver
correctly failed before writing predictions, but its original error did not say
that the project already had a working pinned DirectML interpreter.

No GitHub, Blog, or social mutation occurred during the failed command. The
existing prediction receipt was not overwritten.

## Root cause

PowerShell inherited `VIRTUAL_ENV=D:\Prospects\env`, so bare `python` resolved
outside the ML project even though `D:\ML\Website\.venv-directml` existed. Python
version equality was not enough: both environments used Python 3.13.7, but they
contained different ONNX Runtime distributions and execution providers.

The broader model-quality path also lacked an executable baseline gate. The
project had a poor historical validation metric in JSON and prose warning that
predictions were experimental, but `predict` could still generate a local file
without attaching the validation decision to that file.

## Repairs

1. Added `gpu_forecast_driver.py validate`.
2. The validator hashes and reads the historical `lstm_predictions.json`, takes
   the month before its earliest stored forecast as the auditable training
   cutoff, and evaluates only later observations.
3. Each eligible commodity is seeded with 12 consecutive pre-cutoff months.
4. The model and a last-observation persistence forecast are rolled recursively
   over the same February–June 2026 horizon without using later actuals as
   inputs.
5. Publication requires the model to beat persistence on aggregate MAPE and
   MAE, with explicit minimum model and sample counts.
6. `predict` and `run-all` automatically execute validation and embed the typed
   `publication_gate` in the prediction receipt.
7. If DirectML is unavailable while the project-pinned interpreter exists, the
   exception now names both the current interpreter and the exact pinned path.
8. Documentation and regression tests encode the interpreter and validation
   contracts.

## Measured result

The gate evaluated 59 exported models over 295 post-cutoff observations:

- LSTM: 85.6945% MAPE, 141.9462 MAE, R² -1.259310
- naive persistence: 7.5142% MAPE, 8.4031 MAE, R² 0.987981
- decision: `withheld_failed_validation`

The corrected DirectML prediction rerun then used:

`D:\ML\Website\.venv-directml\Scripts\python.exe gpu_forecast_driver.py predict`

Its runtime probe recorded ONNX Runtime 1.24.4 with
`DmlExecutionProvider` and `CPUExecutionProvider`. It generated 1,062 local
experimental values and embedded the failed publication gate. The values remain
local and are not forecast claims.

## Acceptance boundary

- The driver can execute the tested ONNX graph through DirectML; the existing
  node profile remains the proof for actual node placement.
- The forecasts are not publishable merely because their tensor graph ran on a
  GPU.
- The baseline receipt does not prove future accuracy; it proves that these
  checkpoints performed far worse than a simple forecast on the observed
  post-cutoff period.
- A later retrain must create a new out-of-time origin and beat the same explicit
  baseline gate before prediction assets or forecast claims can be released.

## Regression coverage

- exact regression-metric calculation
- both-MAPE-and-MAE baseline requirement
- cutoff inference from the earliest stored forecast
- explicit withheld status when validation is missing
- exact project-pinned interpreter in the DirectML-unavailable error

## Permanent next step

Repair the LSTM training contract before retraining: sequences must never cross
region/price-type series, scalers must fit training rows only, target scaling
must be explicit in checkpoint/ONNX metadata, and final model selection must not
consume the out-of-time publication test. New models remain local until they
pass the same deterministic gate.
