"""Locks the 2026-08-26 leakage fix in gpu_forecaster_v2.make_examples.

Before the fix, make_examples picked ANCHORS at or before the cutoff but then
supervised every horizon target that existed in the series, including target
months AFTER the cutoff. A numeric audit measured leak_fraction = 1.0: all 295
primary-gate points and all 2,360 multi-origin points were trained on directly
before being scored, so the published "3.54% MAPE beats naive" figure was an
in-sample number. With masking the same gate measures ~4.54% MAPE, which still
beats naive persistence (7.41%).

Requires the torch-directml environment:
    .venv-torch-dml/Scripts/python.exe -m pytest tests/test_gpu_forecaster_v2_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
from gpu_forecaster_v2 import H, make_examples  # noqa: E402


def _toy():
    """Two commodities, 40 consecutive months of strictly positive prices."""
    periods = pd.period_range("2020-01", periods=40, freq="M")
    pmaps = {
        "AAA": {p: 100.0 + i for i, p in enumerate(periods)},
        "BBB": {p: 50.0 + 0.5 * i for i, p in enumerate(periods)},
    }
    return pmaps, {"AAA": 0, "BBB": 1}, periods


def _supervised_targets(pmaps, cid, cutoff, strict):
    """Recover the (commodity, target period) pairs the mask actually supervises."""
    pairs = set()
    for commodity, pm in pmaps.items():
        single = {commodity: pm}
        X, _, _, _, Msk = make_examples(single, {commodity: cid[commodity]}, cutoff,
                                        strict=strict)
        anchors = [p for p in sorted(pm) if p <= cutoff]
        eligible = [p for p in anchors if all((p - k) in pm for k in range(13))]
        rows = []
        for anchor in eligible:
            has_target = any(
                (anchor + h) in pm and (not strict or (anchor + h) <= cutoff)
                for h in range(1, H + 1)
            )
            if has_target:
                rows.append(anchor)
        assert len(rows) == len(X)
        for row, anchor in enumerate(rows):
            for h in range(1, H + 1):
                if float(Msk[row][h - 1]) == 1.0:
                    pairs.add((commodity, anchor + h))
    return pairs


def test_strict_mode_never_supervises_a_target_after_the_cutoff():
    pmaps, cid, _ = _toy()
    cutoff = pd.Period("2021-06", freq="M")
    pairs = _supervised_targets(pmaps, cid, cutoff, strict=True)
    assert pairs, "strict mode still needs training signal"
    late = sorted(str(p) for _, p in pairs if p > cutoff)
    assert late == [], f"strict mode leaked post-cutoff targets: {late}"


def test_leaky_mode_reproduces_the_original_contamination():
    pmaps, cid, _ = _toy()
    cutoff = pd.Period("2021-06", freq="M")
    pairs = _supervised_targets(pmaps, cid, cutoff, strict=False)
    late = [p for _, p in pairs if p > cutoff]
    assert late, "the --allow-leak A/B path must still reproduce the leak"
    assert max(late) == cutoff + H


def test_scored_gate_window_is_disjoint_from_strict_training_targets():
    """The exact pairs the gate scores must not appear in the training targets."""
    pmaps, cid, periods = _toy()
    cutoff = pd.Period("2021-06", freq="M")
    gate_h = 5
    scored = {(c, cutoff + h) for c in pmaps for h in range(1, gate_h + 1)
              if (cutoff + h) in pmaps[c]}
    assert len(scored) == 2 * gate_h
    trained = _supervised_targets(pmaps, cid, cutoff, strict=True)
    assert scored.isdisjoint(trained)
    leaked = _supervised_targets(pmaps, cid, cutoff, strict=False)
    assert scored.issubset(leaked), "regression guard: the old path really did leak"


def test_default_is_strict():
    pmaps, cid, _ = _toy()
    cutoff = pd.Period("2021-06", freq="M")
    _, _, _, _, default_mask = make_examples(pmaps, cid, cutoff)
    _, _, _, _, strict_mask = make_examples(pmaps, cid, cutoff, strict=True)
    assert float(default_mask.sum()) == float(strict_mask.sum())
    _, _, _, _, leak_mask = make_examples(pmaps, cid, cutoff, strict=False)
    assert float(leak_mask.sum()) > float(strict_mask.sum())


def test_publication_gate_never_passes_a_leaked_run():
    from gpu_forecaster_v2 import decide_publication_gate

    assert decide_publication_gate(False, True, 8, 8) == "withheld_leaked_training_targets"
    assert decide_publication_gate(True, True, 8, 8) == "passed_out_of_time_naive_baseline"
    assert decide_publication_gate(True, False, 8, 8) == "withheld_failed_validation"
    assert decide_publication_gate(True, True, 7, 8) == "withheld_failed_validation"
    assert decide_publication_gate(True, True, 0, 0) == "withheld_failed_validation"


# --- DirectML CPU-fallback removal (2026-08-26) -----------------------------
# torch-directml has no kernel for aten::huber_loss or aten::lerp.Scalar_out, so
# nn.HuberLoss and torch.optim.Adam silently ran those steps on the CPU, copying
# tensors off and back on the GPU every iteration. Both were replaced with
# on-device equivalents; these tests hold the equivalents numerically honest.

def test_huber_elementwise_matches_torch_huber_loss():
    from gpu_forecaster_v2 import huber_elementwise

    torch.manual_seed(0)
    pred = torch.randn(400, 18)
    target = torch.randn(400, 18)
    for delta in (0.05, 0.5, 2.0):
        reference = torch.nn.HuberLoss(reduction="none", delta=delta)(pred, target)
        assert torch.equal(reference, huber_elementwise(pred, target, delta))


def test_dml_adam_matches_torch_adam():
    import copy

    from gpu_forecaster_v2 import DmlAdam

    torch.manual_seed(0)
    reference_model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(),
                                          torch.nn.Linear(32, 4))
    ported_model = copy.deepcopy(reference_model)
    reference_opt = torch.optim.Adam(reference_model.parameters(), lr=2e-3)
    ported_opt = DmlAdam(ported_model.parameters(), lr=2e-3)
    x, y = torch.randn(256, 16), torch.randn(256, 4)

    for step in range(1, 21):
        for model, opt in ((reference_model, reference_opt), (ported_model, ported_opt)):
            opt.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
            opt.step()
        drift = max(float((a - b).abs().max())
                    for a, b in zip(reference_model.parameters(), ported_model.parameters()))
        if step == 1:
            assert drift == 0.0, "the first Adam step must be bit-identical"
        assert drift < 1e-6, f"step {step} drifted {drift}"


def test_v2_training_loop_uses_the_on_device_ops():
    """Guard against a refactor silently reintroducing the CPU-fallback ops."""
    source = (ROOT / "gpu_forecaster_v2.py").read_text(encoding="utf-8")
    body = source[source.index("def train(dev, n_comm"):source.index("def predict_from(")]
    assert "huber_elementwise" in body
    assert "DmlAdam(" in body
    assert "nn.HuberLoss" not in body
    assert "torch.optim.Adam(" not in body
