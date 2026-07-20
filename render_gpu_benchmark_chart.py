"""Render the current DirectML benchmark receipt as an honest standalone SVG."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "gpu_driver_evidence" / "benchmark_current.json"
OUTPUT = ROOT / "gpu_driver_evidence" / "benchmark_latency.svg"


def main() -> None:
    benchmark = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = benchmark["results"]
    width, height = 1400, 900
    left, top, chart_w, chart_h = 150, 230, 1100, 470
    max_ms = max(max(row["gpu_directml"]["median_ms"], row["cpu"]["median_ms"]) for row in rows)
    scale = chart_h / max_ms
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1400" height="900" fill="#07111f"/>',
        '<text x="80" y="85" fill="#f8fafc" font-family="Arial" font-size="46" font-weight="700">Native AMD GPU worked. CPU won this graph.</text>',
        '<text x="80" y="135" fill="#94a3b8" font-family="Arial" font-size="25">Rice LSTM · warmed median latency · RX 6600 DirectML vs CPU · lower is better</text>',
        '<rect x="80" y="165" width="24" height="24" rx="4" fill="#ec4899"/><text x="118" y="185" fill="#cbd5e1" font-family="Arial" font-size="22">DirectML</text>',
        '<rect x="280" y="165" width="24" height="24" rx="4" fill="#22d3ee"/><text x="318" y="185" fill="#cbd5e1" font-family="Arial" font-size="22">CPU</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#475569" stroke-width="2"/>',
    ]
    group_w = chart_w / len(rows)
    for index, row in enumerate(rows):
        center = left + group_w * (index + 0.5)
        gpu_ms = row["gpu_directml"]["median_ms"]
        cpu_ms = row["cpu"]["median_ms"]
        for offset, value, color in ((-42, gpu_ms, "#ec4899"), (18, cpu_ms, "#22d3ee")):
            bar_h = max(3, value * scale)
            x = center + offset
            y = top + chart_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="48" height="{bar_h:.1f}" rx="7" fill="{color}"/>')
            parts.append(f'<text x="{x + 24:.1f}" y="{max(top - 8, y - 12):.1f}" text-anchor="middle" fill="#e2e8f0" font-family="Arial" font-size="18">{value:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + chart_h + 42}" text-anchor="middle" fill="#f8fafc" font-family="Arial" font-size="24" font-weight="700">batch {row["batch_size"]}</text>')
        cpu_advantage = gpu_ms / cpu_ms
        parts.append(f'<text x="{center:.1f}" y="{top + chart_h + 75}" text-anchor="middle" fill="#67e8f9" font-family="Arial" font-size="19">CPU {cpu_advantage:.2f}× faster</text>')
    parts.extend(
        [
            '<text x="80" y="805" fill="#a7f3d0" font-family="Arial" font-size="24" font-weight="700">Profile proof: 15 DirectML nodes, including 2 LSTM ops · max CPU/GPU difference 3.81e-6</text>',
            '<text x="80" y="850" fill="#94a3b8" font-family="Arial" font-size="20">Truth boundary: one warmed project graph on one host. This is not a universal GPU benchmark.</text>',
            '</svg>',
        ]
    )
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
