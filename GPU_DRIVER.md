# AMD DirectML food-price inference driver

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

Generated ONNX graphs stay under `.onnx_models/`. Reproducible JSON receipts are
written to `gpu_driver_evidence/`. Do not publish model bundles or prediction
receipts until their licensing, units, and validation gates pass.

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
