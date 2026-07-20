# GPU Driver v1.0.0 release candidate — DirectML GPU placement, measured without hype

This release adds a Python ML execution driver for the Philippine Food Price
Dashboard's LSTM models on AMD GPUs through ONNX Runtime DirectML.

## What shipped

- exports 72 existing PyTorch LSTM checkpoints to portable ONNX graphs;
- verifies actual graph placement with ONNX Runtime node profiling;
- benchmarks GPU and CPU sessions on the same real `Rice (regular, milled)` graph;
- generated a local experimental run of 1,062 forecast points for 59 current
  commodity models from July 2026 through December 2027;
- gives 13 stale commodity series typed skip receipts instead of projecting
  pre-2023 observations as current;
- records the WFP source hash, model hashes, runtime versions, provider lists,
  numerical parity, and claim limits in release JSON.

## Tested host

- AMD Radeon RX 6600, DirectML device 0
- Python 3.13.7
- ONNX Runtime DirectML 1.24.4
- WFP snapshot: 236,252 rows, 73 commodities, 17 regions, 3 price types
- source maximum: June 15, 2026
- source SHA-256: `9623508dfa1e33c6ac6bda2ceeafca1562679b1e49258df977cb3cd40c147124`

## Benchmark result

DirectML device placement is verified: 15 of 18 profiled node events—including
both LSTM events—ran on `DmlExecutionProvider`; Gather, Unsqueeze, and Concat
used the CPU fallback. DirectML-session and CPU-session outputs agreed within
`3.81e-6` maximum absolute difference.

The DirectML path on this RX 6600 did **not** win this compact recurrent workload. These are one-host
warmed medians; the near-constant DirectML time at batches 8–128 likely reflects
a synchronized dispatch floor rather than useful scaling:

| Batch | DirectML median | CPU median | CPU advantage |
|---:|---:|---:|---:|
| 1 | 0.9808 ms | 0.1685 ms | 5.82× |
| 8 | 21.3591 ms | 0.3564 ms | 59.93× |
| 32 | 21.6465 ms | 0.5867 ms | 36.90× |
| 128 | 22.2508 ms | 3.2285 ms | 6.89× |

That negative result is part of the release, not hidden. DirectML provides a
cross-vendor GPU path on supported Windows systems; measurement decides whether a production
workload should actually use it. Larger dense/vision graphs may behave
differently and require their own benchmark.

## Predictions and model bundle are withheld from the public release

The local prediction artifact is reproducible, but it is not evidence of
forecasting skill. The existing checkpoint validation artifact reports 81.1%
MAPE and R² -1.1107, most new series are persistence-like, and no new
naive-persistence baseline has passed. The autoregressive driver also applies a
safety clamp when a prediction exceeds five times the recent mean. Raw WFP units
have not been verified as a single comparable `PHP/kg` unit across commodities.

Accordingly, the 1,062 values are retained as local experimental output and must
not be charted as validated price forecasts. The 72-model ONNX bundle is also
withheld until model-weight redistribution and WFP attribution terms are
confirmed. The engineering release candidate contains driver source, tests, the
negative latency benchmark, the chart, and hash manifests—not a predictive-
accuracy claim.

## Claim boundary

This benchmark measures warmed synchronous inference for one repository LSTM graph on one
host. It does not measure training, end-to-end CSV preparation, every Python
operation, other GPUs, or universal DirectML performance. The local forecasts
use current WFP observations with previously trained checkpoints and are
unvalidated experimental output—not financial advice.

The separate classical-model comparison output is excluded from this release's
accuracy claims because an audit found target leakage in legacy rolling/difference
features. No `0.0% MAPE` or `R²=1.0` claim is credited here.
