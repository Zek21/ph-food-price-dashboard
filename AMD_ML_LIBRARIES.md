# Real machine learning on an AMD Radeon RX 6600 (Windows, DirectML)

The goal here is **general machine learning — training and inference — not LLM /
ollama inference.** These results prove that mainstream Python ML frameworks run
on a consumer AMD RX 6600 (gfx1032, 8 GB) on Windows, and that a model *trained
on that GPU* now passes the same honest out-of-time gate the project's LSTM
failed.

Every number below comes from a reproducible receipt in `gpu_driver_evidence/`,
produced on this host (Windows 11, AMD Radeon RX 6600, driver 32.0.21030.2001).

## Why two virtual environments

`torch-directml` (the DirectML backend for PyTorch) and the TensorFlow DirectML
plugin do **not** ship wheels for Python 3.13, which is the only interpreter
installed system-wide. Standalone interpreters were fetched with `uv` (no admin):

| venv | Python | Purpose |
|------|--------|---------|
| `.venv-directml` | 3.13 | ONNX Runtime DirectML, scikit-learn, skl2onnx, LightGBM, XGBoost |
| `.venv-torch-dml` | 3.12 | `torch==2.4.1` + `torch-directml==0.2.5.dev240914` |
| `.venv-tf-dml` | 3.10 | `tensorflow-cpu==2.10.1` + `tensorflow-directml-plugin` (needs numpy<2) |

## AMD RX 6600 ML-library support matrix

Receipt: `gpu_driver_evidence/amd_rx6600_ml_matrix.json` — each row is a live
probe, not a claim. ONNX GPU placement is confirmed by profiling node providers,
never by a provider *listing* alone.

| Library | GPU on RX 6600 | Mode | Evidence |
|---------|:--------------:|------|----------|
| **PyTorch** (torch-directml) | ✅ yes | train + infer | matmul + MLP training on `privateuseone:0` |
| **TensorFlow 2.10** (DirectML plugin) | ✅ yes | train + infer | Keras `fit()` on `/GPU:0` |
| **ONNX Runtime** (DirectML EP) | ✅ yes | infer | 15/18 LSTM nodes on `DmlExecutionProvider` |
| **torchvision / CNN** (Conv2d) | ✅ yes | train + infer | conv output on GPU |
| **scikit-learn MLP → ONNX → DirectML** | ✅ yes | infer | Gemm/MatMul nodes on `DmlExecutionProvider` |
| scikit-learn RandomForest → ONNX | ⚠️ CPU-op | infer | `TreeEnsemble` not implemented on DirectML |
| torch `nn.LSTM` | ❌ no | — | `aten::_thnn_fused_lstm_cell` not on DML backend |
| LightGBM | ❌ no | CPU | pip wheel built without `-DUSE_GPU=1` (needs OpenCL build) |
| XGBoost | ❌ no | CPU | GPU is CUDA/NVIDIA-only; prints "No visible GPU", uses CPU |
| scikit-learn (native) | ❌ no | CPU | no GPU backend — export to ONNX for AMD GPU inference |

**Takeaway:** three major frameworks (PyTorch, TensorFlow, ONNX Runtime) plus
classic scikit-learn models (via ONNX export) all accelerate on the RX 6600
through DirectML. The gradient-boosting libraries (XGBoost/LightGBM) remain CPU
because their GPU paths are CUDA/OpenCL, not DirectML.

## PyTorch training + inference benchmark

Receipt: `gpu_driver_evidence/pytorch_directml_proof.json`. Square FP32 matmul,
warmed, GPU vs. CPU on the same host:

| Matmul | GPU (DirectML) | GPU GFLOPS | CPU | GPU speedup |
|-------:|---------------:|-----------:|----:|:-----------:|
| 512²   | 0.53 ms | 508 | 0.93 ms | **1.75×** |
| 1024²  | 1.25 ms | 1714 | 6.58 ms | **5.26×** |
| 2048²  | 7.19 ms | 2389 | 57.86 ms | **8.04×** |
| 4096²  | 50.72 ms | 2710 | 388.39 ms | **7.66×** |

A real MLP training loop ran end-to-end on the GPU (`privateuseone:0`): loss
0.5394 → 0.0003 over 300 steps, **1.84× faster than CPU**. Parameters and
gradients were verified on the DirectML device.

> Honest nuance: for the project's *tiny* ONNX LSTM (a 12×6 graph), DirectML
> inference is *slower* than CPU — a ~21 ms fixed dispatch cost dominates a
> microsecond-scale graph (see `benchmark_current.json`). The GPU wins once the
> work is ML-sized (≥1024² matmul, real training), which is exactly what
> `torch-directml` unlocks that ONNX-inference-only did not.

## Doing better: a GPU-trained forecaster that beats naive

Receipt: `gpu_driver_evidence/gpu_trained_backtest.json`. `torch-directml` cannot
run the fused LSTM cell, so `gpu_train_forecaster.py` trains a MatMul-based model
(per-series z-score, pooled windows, commodity embedding) that predicts the
**z-space monthly delta** — so "predict zero" is *exactly* naive persistence and
any learned signal is honest improvement. It is judged with the **same**
out-of-time methodology as the LSTM: cutoff 2026-01, horizon 2026-02…06,
per-commodity recursive roll-forward, aggregate metrics, model must beat naive on
**both** MAPE and MAE.

| Model (295 out-of-time points) | MAPE | MAE | R² | Gate |
|--------------------------------|-----:|----:|---:|:----:|
| Old LSTM (ONNX export) | 85.69% | 141.95 | **−1.259** | ❌ withheld |
| Naive persistence (baseline) | 7.41% | 8.14 | 0.9877 | — |
| v1 GPU delta-MLP | ≈5.8% | ≈7.0 | +0.9916 | ✅ passed |
| **v2 GPU multi-horizon** | **≈3.5%** | **≈4.5** | **+0.9970** | ✅ **passed** |

The models **trained on the RX 6600 in ~3–4 seconds** (`trained_on_gpu=true`).

### v2 — advisor-guided rebuild (2026-07-21)

Both free CDP advisors — **ChatGPT (GPT-5 Sol)** and **Gemini 3.5** — were asked
how to improve the driver and **converged** on the same plan (receipt:
`advisor_consult_20260721.md`). v2 (`gpu_forecaster_v2.py`) implements it:

- **Direct multi-horizon** (one forward pass emits 18 months) — *no recursive
  roll-forward*, so no recursive error accumulation.
- **Log-return target vs persistence** (`forecast = last · exp(residual)`; zero
  output == naive, preserving the honest-gain contract).
- **Horizon-weighted Huber loss** (robust), not MAPE/MSE.
- **Multi-origin backtest** for stability.

Result — receipt `gpu_forecaster_v2.json`:

- Primary gate (cutoff 2026-01, Feb–Jun): **MAPE 3.5%** / MAE 4.5 / R² 0.9970 vs
  naive 7.41% / 8.14 / 0.9877 — roughly **half** the v1 error.
- **Multi-origin: 8 / 8 out-of-time origins beat naive** (pooled 4.65% vs 6.70%
  MAPE, n=2360) — the win is stable across origins, not a single-cutoff fluke.

Deferred next step (advisor rec #4): fold in the project's ENSO/USD-PHP/FAO
exogenous features once `exogenous_data.json` is fetched.

## Reproduce

```powershell
# PyTorch on the AMD GPU (train + benchmark)
.\.venv-torch-dml\Scripts\python.exe gpu_pytorch_proof.py

# Library support matrix (run in each venv; results merge)
.\.venv-torch-dml\Scripts\python.exe ml_amd_matrix.py
.\.venv-directml\Scripts\python.exe   ml_amd_matrix.py
.\.venv-tf-dml\Scripts\python.exe      ml_amd_matrix.py

# GPU-trained forecaster + out-of-time gate vs naive
.\.venv-torch-dml\Scripts\python.exe gpu_train_forecaster.py

# Charts
.\.venv-directml\Scripts\python.exe render_amd_ml_charts.py
```

## Truth boundary

- These are warmed benchmarks and one project's data on one host (AMD RX 6600).
  They do not generalize to every workload or GPU.
- The GPU-trained forecaster passes the out-of-time naive gate on the 2026-02…06
  horizon; that is evidence of skill on that window, not a guarantee of future
  accuracy. Forecasts remain experimental and are not financial advice.
- "Trained on GPU" means model parameters and gradients lived on the DirectML
  device (`privateuseone:0`) during training, verified in the receipt.
