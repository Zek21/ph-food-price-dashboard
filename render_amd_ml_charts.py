#!/usr/bin/env python
"""Render blog charts from the AMD RX 6600 receipts (pure SVG, no dependencies)."""
from __future__ import annotations
import json
from pathlib import Path

E = Path(__file__).resolve().parent / "gpu_driver_evidence"


def bar_chart(path: Path, title: str, labels, values, colors, unit: str,
              note: str = "", w: int = 720, h: int = 400) -> None:
    pad_l, pad_b, pad_t = 70, 70, 70
    plot_h = h - pad_b - pad_t
    plot_w = w - pad_l - 40
    vmax = max(values) * 1.15 or 1
    bw = plot_w / (len(values) * 1.6)
    gap = bw * 0.6
    bars = []
    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        x = pad_l + gap / 2 + i * (bw + gap)
        bh = (val / vmax) * plot_h
        y = pad_t + plot_h - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
            f'rx="4" fill="{col}"/>'
            f'<text x="{x + bw/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
            f'font-size="15" font-weight="600" fill="#e6edf3">{val:g}{unit}</text>'
            f'<text x="{x + bw/2:.1f}" y="{pad_t + plot_h + 22:.1f}" text-anchor="middle" '
            f'font-size="13" fill="#9fb0c0">{lab}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'font-family="Segoe UI,Arial,sans-serif">'
        f'<rect width="{w}" height="{h}" fill="#0d1117"/>'
        f'<text x="{w/2}" y="34" text-anchor="middle" font-size="19" '
        f'font-weight="700" fill="#e6edf3">{title}</text>'
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{w - 40}" '
        f'y2="{pad_t + plot_h}" stroke="#30363d"/>'
        + "".join(bars)
        + (f'<text x="{w/2}" y="{h - 20}" text-anchor="middle" font-size="12" '
           f'fill="#7d8ea0">{note}</text>' if note else "")
        + "</svg>"
    )
    path.write_text(svg, encoding="utf-8")
    print("wrote", path)


def main() -> None:
    pt = json.loads((E / "pytorch_directml_proof.json").read_text())
    bt = json.loads((E / "gpu_trained_backtest.json").read_text())

    # 1) PyTorch matmul speedup (GPU vs CPU) by size.
    sizes = [r["n"] for r in pt["matmul_benchmark"]]
    speed = [r["gpu_speedup_vs_cpu"] for r in pt["matmul_benchmark"]]
    bar_chart(
        E / "pytorch_gpu_speedup.svg",
        "PyTorch on AMD RX 6600 (torch-directml): GPU speedup vs CPU",
        [f"{s}x{s}" for s in sizes], speed,
        ["#f78166", "#3fb950", "#3fb950", "#3fb950"], "x",
        note="Square matmul, warmed; RX 6600 reaches ~2710 GFLOPS FP32. Higher is better.",
    )

    # 2) Out-of-time forecast MAPE: broken LSTM vs GPU-trained MLP vs naive.
    g = bt["backtest"]["gpu_delta_mlp"]["mape"]
    naive = bt["backtest"]["naive_persistence"]["mape"]
    bar_chart(
        E / "forecast_mape_comparison.svg",
        "Out-of-time forecast error (MAPE %, n=295) - lower is better",
        ["Old LSTM\n(ONNX)", "Naive\npersistence", "GPU-trained\ndelta-MLP"],
        [85.6945, round(naive, 2), round(g, 2)],
        ["#f85149", "#8b949e", "#3fb950"], "%",
        note="Same 2026-02..06 horizon and naive baseline the LSTM failed; "
             "GPU-trained model passes the gate.",
    )


if __name__ == "__main__":
    main()
