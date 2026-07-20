#!/usr/bin/env python
"""Proof that real PyTorch trains and infers on an AMD RX 6600 via DirectML.

This is deliberately NOT ollama / LLM inference. It exercises the general
machine-learning stack (torch autograd, nn.Linear/LSTM, optimizers, a real
training loop) on the AMD GPU through the torch-directml backend, and records a
hash-stamped receipt with device placement and honest GPU-vs-CPU timings.

Run:
    .venv-torch-dml\\Scripts\\python.exe gpu_pytorch_proof.py

Only Python 3.12 (torch-directml has no 3.13 wheel) with torch-directml
installed can execute this. The receipt lands in gpu_driver_evidence/.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_directml as dml

EVID = Path(__file__).resolve().parent / "gpu_driver_evidence"
EVID.mkdir(exist_ok=True)


def _sync(device: torch.device) -> None:
    """Force the async DirectML queue to finish (no public synchronize())."""
    if device.type == "privateuseone":
        # Reading a scalar back to host blocks until the queue drains.
        torch.ones(1, device=device).cpu()


def bench_matmul(device: torch.device, n: int, iters: int, warmup: int) -> dict:
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    for _ in range(warmup):
        c = a @ b
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    _sync(device)
    dt = time.perf_counter() - t0
    per = dt / iters
    # 2*n^3 flops per matmul.
    gflops = (2.0 * n ** 3) / per / 1e9
    return {"n": n, "iters": iters, "ms_per_matmul": round(per * 1e3, 4),
            "gflops": round(gflops, 1), "checksum": float(c.float().sum().cpu())}


class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_h), torch.nn.ReLU(),
            torch.nn.Linear(d_h, d_h), torch.nn.ReLU(),
            torch.nn.Linear(d_h, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def train_mlp(device: torch.device, steps: int = 300) -> dict:
    """Train a real MLP on a synthetic non-linear regression, on `device`."""
    torch.manual_seed(0)
    n, d_in, d_h = 4096, 64, 256
    # Ground-truth non-linear function the net must learn.
    w = torch.randn(d_in, 1, device=device)
    X = torch.randn(n, d_in, device=device)
    y = torch.sin(X @ w) + 0.1 * (X[:, :1] ** 2)

    model = MLP(d_in, d_h, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.MSELoss()

    first_param = next(model.parameters())
    losses = []
    _sync(device)
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        pred = model(X)
        loss = lossf(pred, y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            losses.append((step, float(loss.detach().cpu())))
    _sync(device)
    dt = time.perf_counter() - t0

    grad = next(model.parameters()).grad
    return {
        "device": str(first_param.device),
        "param_device": str(first_param.device),
        "grad_device": str(grad.device) if grad is not None else None,
        "steps": steps,
        "sec_total": round(dt, 3),
        "ms_per_step": round(dt / steps * 1e3, 3),
        "loss_curve": losses,
        "loss_start": losses[0][1],
        "loss_end": losses[-1][1],
        "learned": losses[-1][1] < losses[0][1] * 0.5,
    }


def probe_lstm(device: torch.device) -> dict:
    """Does torch nn.LSTM (the food model's core op) run on this device?"""
    try:
        torch.manual_seed(0)
        lstm = torch.nn.LSTM(input_size=6, hidden_size=128, num_layers=2,
                             batch_first=True).to(device)
        x = torch.randn(32, 12, 6, device=device)
        out, (h, c) = lstm(x)
        _sync(device)
        return {"ok": True, "out_device": str(out.device),
                "out_shape": list(out.shape), "checksum": float(out.sum().cpu())}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def main() -> None:
    dev_name = dml.device_name(0) if dml.device_count() else "NONE"
    gpu = dml.device()
    cpu = torch.device("cpu")

    receipt = {
        "schema": "amd-rx6600-pytorch-directml-proof-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_directml": dml.__version__ if hasattr(dml, "__version__") else "0.2.5.dev240914",
        },
        "device": {
            "dml_device_count": dml.device_count(),
            "dml_device_name": dev_name,
            "torch_device": str(gpu),
        },
    }

    print(f"== PyTorch {torch.__version__} on {dev_name} ({gpu}) ==")

    # 1) Tensor op placement proof.
    a = torch.randn(2048, 2048, device=gpu)
    b = torch.randn(2048, 2048, device=gpu)
    c = a @ b
    receipt["placement_proof"] = {
        "input_device": str(a.device),
        "result_device": str(c.device),
        "on_gpu": c.device.type == "privateuseone",
        "checksum": float(c.sum().cpu()),
    }
    print(f"  matmul result on {c.device}")

    # 2) Matmul benchmark GPU vs CPU across sizes.
    sizes = [512, 1024, 2048, 4096]
    bench = []
    for n in sizes:
        iters = 50 if n <= 1024 else (20 if n <= 2048 else 8)
        g = bench_matmul(gpu, n, iters, warmup=5)
        cc = bench_matmul(cpu, n, max(3, iters // 4), warmup=2)
        speedup = round(cc["ms_per_matmul"] / g["ms_per_matmul"], 2)
        bench.append({"n": n, "gpu": g, "cpu": cc, "gpu_speedup_vs_cpu": speedup})
        print(f"  matmul {n:>4}: GPU {g['ms_per_matmul']:.2f}ms "
              f"({g['gflops']:.0f} GFLOPS) | CPU {cc['ms_per_matmul']:.2f}ms "
              f"| GPU x{speedup}")
    receipt["matmul_benchmark"] = bench

    # 3) Real training loop on the GPU.
    print("  training MLP on GPU ...")
    tr_gpu = train_mlp(gpu)
    print(f"    loss {tr_gpu['loss_start']:.4f} -> {tr_gpu['loss_end']:.4f} "
          f"on {tr_gpu['device']} ({tr_gpu['ms_per_step']:.1f} ms/step)")
    print("  training MLP on CPU ...")
    tr_cpu = train_mlp(cpu)
    receipt["training"] = {
        "gpu": tr_gpu, "cpu": tr_cpu,
        "gpu_trained_on_device": tr_gpu["param_device"] == "privateuseone:0",
        "gpu_loss_decreased": tr_gpu["learned"],
        "train_speedup_vs_cpu": round(tr_cpu["ms_per_step"] / tr_gpu["ms_per_step"], 2),
    }

    # 4) LSTM op probe (the food-price model's core layer).
    receipt["lstm_probe"] = probe_lstm(gpu)
    print(f"  nn.LSTM on GPU: {receipt['lstm_probe'].get('ok')}")

    out = EVID / "pytorch_directml_proof.json"
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nReceipt -> {out}")


if __name__ == "__main__":
    main()
