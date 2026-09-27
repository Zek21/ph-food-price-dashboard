# AMD DirectML food-price inference driver


Ticket: AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS. Fresh run: 2026-09-28 Asia/Manila.

- Per-node proof: both LSTM operators on `DmlExecutionProvider`; 15/18 node events on DirectML, 3 on CPU.
- The tested compact graph was slower on DirectML at all measured batch sizes. Batch-1 medians: 1.0352 ms DirectML versus 0.1740 ms CPU.
- Authoritative April-June 2026 validation: 59 models, 177 points; MAPE 3.9927% versus persistence 4.3018%, MAE 5.1355 versus 5.1679. Both paired 95% intervals cross zero; one commodity breaches the unchanged 3x MAE guard.
- Current forward set: 59 models / 1,062 points, regenerated only in the private directory and **WITHHELD**.
- Supportive six-origin diagnostic: pooled MAE persistence 10.3057, AR(1) 10.6965, LSTM 10.1478. All 354 point counts match; all naive-MAE pairs agree within 1%. This is not the publication gate.
- Focused tests: 85 passed, 24 skipped, 1 warning in 10.25s. Full repo suite: 228 passed, 24 skipped, 1 warning in 153.56s (0:02:33). Skips identify absent historical receipts, not passing checks. Initial failures are preserved.

No GPU-only, GPU speedup, PyTorch GPU-training, validated-price, or investor-guidance claim is made.

The already-published rc6 tag remains at `1a5130fb8b127b43ac9e46255947f4b4ddca084d`. Fresh docs and receipts are supplied as release assets and in the `release/gpu-driver-v1.0.0-rc6` branch; the original tag source archive is historical. The current 59-model rerun values are absent from the public payload and Git index. Legacy GPU/LSTM forward sections were removed from the new branch tip. Existing published Git history/tag archives and unrelated classical dashboard forecasts are not rewritten; this is not a claim that the entire historical repository contains no forecasts.

Use `GPU_DRIVER_RC6_REPRODUCIBILITY.md`, `release_truth_manifest.json`, and `SHA256SUMS-rc6.txt` in the attached audit bundle.

## Historical verification update ? 2026-09-13 (superseded by rc6 above)

Fresh ZEKE receipts under `gpu_driver_evidence/rerun_20260913_native/` tighten the claim boundary. The release Rice LSTM ONNX graph **does execute on the RX 6600 through DirectML**: profiling placed 15 of 18 node events on `DmlExecutionProvider`, including both LSTM nodes; three shape/input nodes fell back to CPU. The receipt SHA-256 is `1cd76b9dc251c579200071abcd9fdeeaa78557d5ae58a8d0ed95e5dc44415134`. This is native GPU placement for the ONNX inference graph, not a GPU-only or speedup claim; CPU latency was lower at every measured batch size.

The current 59-model release set **fails** the April?June 2026 out-of-time persistence gate: model MAPE/MAE `7.2264% / 8.5919` versus naive `4.3018% / 5.1679`, with paired 95% CIs entirely on the worse side (`MAPE +1.432562..+4.435314`, `MAE +1.594583..+5.296231`). Therefore release-model predictions remain withheld and must not be used in investor/public performance claims. Validation receipt SHA-256: `55afd761a8f7dca623ce18975a445ddf113f3eb6bee884a421b495f373c6c204`.

A separate current `torch-directml` probe confirms ordinary tensor/matmul placement on `privateuseone:0`, but `torch.nn.LSTM` fails because `aten::_thnn_fused_lstm_cell` falls back to CPU and is unsupported in this runtime. Production LSTM GPU evidence therefore comes from ONNX Runtime DirectML profiling, not PyTorch LSTM training.

`gpu_forecast_driver.py` runs the dashboard's real LSTM tensor graphs on an AMD
GPU through ONNX Runtime DirectML. It is a Python ML execution driver—not a
Windows display driver, kernel driver, CUDA replacement, or a way to move plain
Python code onto a GPU.

The driver has four proof-producing stages:

1. `probe` reports the providers actually visible to ONNX Runtime.
2. `export` converts the project's PyTorch checkpoints to portable ONNX graphs.
3. `benchmark` compares warmed DirectML and CPU inference on the same graph and
   records per-node provider placement plus numerical output differences.
4. `predict` uses the current WFP CSV and the exported models to create a dated,
   hashed forecast artifact.

## Install and run

```powershell
python -m venv .venv-directml
.\.venv-directml\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-directml\Scripts\python.exe -m pip install torch
.\.venv-directml\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv-directml\Scripts\python.exe gpu_forecast_driver.py run-all
```

The interpreter path is part of the execution contract. An inherited or active
virtual environment may contain ordinary `onnxruntime` with only CPU/Azure
providers. The driver now fails before prediction and names the pinned
`.venv-directml` interpreter when that mismatch is detected.

## Out-of-time publication gate

Run the validation gate independently with:

```powershell
.\.venv-directml\Scripts\python.exe gpu_forecast_driver.py validate
```

The gate infers the historical training cutoff from the SHA-256-bound legacy
forecast artifact, rolls each eligible exported graph forward only after that
cutoff, and compares it with a last-observation persistence forecast. Public
prediction use requires the model to beat the naive baseline on both aggregate
MAPE and MAE with the minimum model/sample counts recorded in the receipt.

`predict` and `run-all` execute this gate automatically. They may still write
local experimental predictions for diagnosis, but every prediction receipt
contains a typed `publication_gate`. A failed or missing validation gate means
the values remain withheld.

Generated ONNX graphs stay under `.onnx_models/`. Reproducible JSON receipts are
written to `gpu_driver_evidence/`. Do not publish model bundles or prediction
receipts until their licensing, units, and validation gates pass.

## Measured device split (rerun 2026-08-26)

Re-measured on the same host and the same dataset
(`wfp_food_prices_phl_latest.csv`, sha256 `9623508d...c0147124`). Receipts under
`gpu_driver_evidence/rerun_20260826/`.

| Workload | CPU | RX 6600 (DirectML) | Winner |
|----------|----:|-------------------:|:------:|
| ONNX LSTM warmed inference, batch 1 | **0.168 ms** | 1.00 ms | CPU 6.0x |
| ONNX LSTM warmed inference, batch 8 | **0.315 ms** | 21.67 ms | CPU 68.7x |
| ONNX LSTM warmed inference, batch 32 | **0.932 ms** | 21.52 ms | CPU 23.1x |
| ONNX LSTM warmed inference, batch 128 | **7.46 ms** | 23.15 ms | CPU 3.1x |
| v2 forecaster training, 200 epochs | 4.57 s | **3.21 s** | GPU 1.42x |
| 4096x4096 FP32 matmul | 406.9 ms | **48.5 ms** | GPU 8.4x (2,833 GFLOPS) |

DirectML placement is real (15 of 18 profiled LSTM node events land on
`DmlExecutionProvider`, max output difference 3.81e-06 vs CPU) and it is still the
slower device for this project's inference. The GPU only wins once the work is
ML-sized: the training loop and large matmuls. `gpu_train_device_benchmark.py`
produces the training half of that comparison; each device runs in its own
interpreter because torch-directml trips an autograd `device_ready_queues_`
assertion if a CPU graph is built before the DML device exists.

## Prediction leakage correction (2026-08-26)

`gpu_forecaster_v2.py` supervised horizon targets past the training cutoff. The
audit `gpu_driver_evidence/rerun_20260826/v2_leakage_audit.json` measured
`leak_fraction = 1.0` on both the primary gate (295 points) and the multi-origin
backtest (2,360 points). Targets are now masked at the cutoff. The honest
out-of-time result is **4.54% MAPE / 5.62 MAE** against naive persistence at
**7.41% / 8.14**, with 8 of 8 stability origins beating naive (pooled 4.81% vs
6.70%), so the gate still passes - on smaller numbers than were first published.
`tests/test_gpu_forecaster_v2_leakage.py` locks the fix.

The exported ONNX LSTM graphs remain **withheld**: rolled forward from the same
2026-01 cutoff they score 85.69% MAPE / 141.95 MAE versus naive 7.51% / 8.40.

## Truth boundary

The benchmark covers warmed inference for this repository's LSTM graph. It does
not benchmark model training, end-to-end CSV preparation, all Python workloads,
or GPUs other than the tested host. A DirectML provider listing alone is not
treated as device-placement proof: the benchmark must also find model-node events
assigned to `DmlExecutionProvider` in the ONNX Runtime profile.

The benchmark schema retains the field name `native_gpu_verified` for receipt
compatibility. In this project it means that profiling found at least one node
event assigned to `DmlExecutionProvider`; it does not mean GPU-only execution,
full-graph placement, or a latency advantage.

Forecasts are unvalidated experimental output and not financial advice. A local
July 2026 run was withheld from the release candidate after the older checkpoint
validation reported 81.1% MAPE and R² -1.1107 and the new trajectories appeared
mostly persistence-like.
