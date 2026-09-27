"""The publication gate must not decide a release on sampling noise.

Found live on 2026-09-12 against gpu_driver_evidence/rerun_20260912: the gate
blocked the release because aggregate model MAE 5.3757 was worse than naive
persistence 5.1679.  That reason reads like a capability statement, but the gap is
0.2078 on a window of 177 points across 59 commodities, and a commodity-clustered
bootstrap puts the 95% CI at [-0.48, +1.01].  The interval straddles zero, so the
aggregate verdict is not distinguishable from chance -- and symmetrically, the
same run's MAPE "win" of 0.1471 has CI [-0.62, +0.38] and is not real either.

The model in fact beat persistence on 34 of 59 commodities on BOTH metrics while
the aggregate MAE said it lost, because the five largest absolute differences sum
to +29.63 against a total gap of +12.26 -- two and a half times the entire
deficit, concentrated in three blow-ups (Beans green fresh 21.42 vs 7.97, Shrimp
tiger 21.11 vs 9.50, Fish threadfin bream 5.78 vs 0.36, a 16x regression).

So the gate had two holes at once: it could block on noise, and an aggregate mean
could equally have PASSED while hiding a commodity where the model was many times
worse than doing nothing.  Both new requirements make the gate strictly harder to
clear; the withhold decision on this run is unchanged, only its stated reason is.

These tests read the committed evidence receipt and call pure functions.  No model
is trained, no ONNX session is created, and no artifact is rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gpu_forecast_driver as driver

ROOT = Path(__file__).resolve().parent.parent
LIVE_VALIDATION = ROOT / "gpu_driver_evidence" / "rerun_20260912" / "validation_current.json"


def _live_models() -> list[dict]:
    if not LIVE_VALIDATION.is_file():
        pytest.skip("rerun_20260912 validation receipt not present")
    return json.loads(LIVE_VALIDATION.read_text(encoding="utf-8"))["models"]


def _receipts(diffs, *, points: int = 3, naive_mae: float = 10.0):
    """Per-commodity receipts with an explicit model-minus-naive MAE difference."""
    return [
        {
            "commodity": f"Commodity {index}",
            "model": {"mape": 5.0 + diff, "mae": naive_mae + diff, "n": points},
            "naive_persistence": {"mape": 5.0, "mae": naive_mae, "n": points},
        }
        for index, diff in enumerate(diffs)
    ]


# --- the aggregate identity the paired test relies on -------------------------


def test_weighted_paired_difference_reproduces_the_aggregate_gap():
    """The bootstrap must measure the same quantity the aggregate gate reads.

    Aggregate MAE is the point-count-weighted mean of the per-commodity MAEs, so
    weighting the paired differences by n reproduces the aggregate difference
    exactly.  Without this correspondence the significance test would be
    answering a different question from the one the gate asks.
    """
    document = json.loads(LIVE_VALIDATION.read_text(encoding="utf-8")) \
        if LIVE_VALIDATION.is_file() else pytest.skip("receipt not present")
    models = document["models"]
    for metric in ("mape", "mae"):
        diagnostics = driver._paired_metric_diagnostics(models, metric=metric)
        aggregate_gap = document["model"][metric] - document["naive_persistence"][metric]
        assert diagnostics["weighted_mean_difference"] == pytest.approx(
            aggregate_gap, abs=5e-4), (
            f"{metric}: paired weighted difference "
            f"{diagnostics['weighted_mean_difference']} does not reconstruct the "
            f"aggregate gap {aggregate_gap}")


# --- the live finding --------------------------------------------------------


def test_live_run_mae_verdict_is_noise_not_a_capability_statement():
    diagnostics = driver._paired_metric_diagnostics(_live_models(), metric="mae")
    low, high = diagnostics["ci95"]
    assert low < 0 < high, (
        f"the 2026-09-12 MAE gap was reported as a failure, but its CI is "
        f"[{low}, {high}]")
    assert diagnostics["verdict"] == "indistinguishable_from_persistence"


def test_live_run_mape_win_is_also_not_significant():
    """The favourable metric gets the same scrutiny as the unfavourable one."""
    diagnostics = driver._paired_metric_diagnostics(_live_models(), metric="mape")
    low, high = diagnostics["ci95"]
    assert low < 0 < high
    assert diagnostics["weighted_mean_difference"] < 0, "MAPE favoured the model"
    assert diagnostics["verdict"] == "indistinguishable_from_persistence"


def test_live_run_majority_of_commodities_favoured_the_model():
    """Guards the specific contradiction: aggregate loss, per-commodity majority win."""
    mae = driver._paired_metric_diagnostics(_live_models(), metric="mae")
    assert mae["model_better_commodities"] > mae["model_worse_commodities"]
    assert mae["weighted_mean_difference"] > 0, (
        "aggregate MAE favoured persistence on this run")


def test_live_run_has_a_commodity_many_times_worse_than_doing_nothing():
    outliers = driver._commodity_regression_outliers(_live_models())
    assert outliers["flagged"], "expected at least one blow-up commodity"
    assert outliers["flagged"][0]["ratio"] > 3.0


def test_live_run_still_fails_the_gate_after_strengthening():
    """The strengthened gate must not flip a withheld run into a publishable one."""
    document = json.loads(LIVE_VALIDATION.read_text(encoding="utf-8")) \
        if LIVE_VALIDATION.is_file() else pytest.skip("receipt not present")
    gate = driver._publication_gate(
        document["model"], document["naive_persistence"],
        model_count=len(document["models"]), per_commodity=document["models"])
    assert gate["passed"] is False
    assert gate["status"] == "withheld_failed_validation"
    assert any("does not exclude zero" in reason for reason in gate["reasons"])


# --- the gate contract ------------------------------------------------------


def test_gate_fails_closed_without_per_commodity_receipts():
    """Absent evidence is not passing evidence."""
    gate = driver._publication_gate(
        {"mape": 8.0, "mae": 3.0, "n": 40}, {"mape": 10.0, "mae": 4.0, "n": 40},
        model_count=12)
    assert gate["passed"] is False
    assert any("per-commodity receipts were not supplied" in reason
               for reason in gate["reasons"])


def test_aggregate_win_cannot_pass_on_a_noisy_paired_difference():
    """A tiny mean advantage swamped by variance must not clear the gate."""
    diffs = [-0.1 if index % 2 else 12.0 for index in range(20)]
    diffs[0] = -240.0  # drags the weighted mean negative while variance stays huge
    receipts = _receipts(diffs, naive_mae=300.0)
    gate = driver._publication_gate(
        {"mape": 8.0, "mae": 3.0, "n": 60}, {"mape": 10.0, "mae": 4.0, "n": 60},
        model_count=20, per_commodity=receipts)
    assert gate["passed"] is False
    assert any("does not exclude zero" in reason for reason in gate["reasons"])


def test_consistent_improvement_still_passes():
    """Strengthening must not make a genuinely better model unpublishable."""
    receipts = _receipts([-1.0] * 15)
    gate = driver._publication_gate(
        {"mape": 8.0, "mae": 3.0, "n": 45}, {"mape": 10.0, "mae": 4.0, "n": 45},
        model_count=15, per_commodity=receipts)
    assert gate["passed"] is True, gate["reasons"]
    assert gate["paired_significance"]["mae"]["verdict"] == "model_significantly_better"


def test_one_blown_up_commodity_blocks_an_otherwise_passing_aggregate():
    receipts = _receipts([-1.0] * 15)
    receipts.append({
        "commodity": "Blow-up",
        "model": {"mape": 4.0, "mae": 40.0, "n": 3},
        "naive_persistence": {"mape": 5.0, "mae": 2.0, "n": 3},
    })
    gate = driver._publication_gate(
        {"mape": 8.0, "mae": 3.0, "n": 48}, {"mape": 10.0, "mae": 4.0, "n": 48},
        model_count=16, per_commodity=receipts)
    assert gate["passed"] is False
    assert any("persistence MAE cap" in reason for reason in gate["reasons"])
    assert gate["per_commodity_guard"]["flagged"][0]["commodity"] == "Blow-up"


def test_zero_persistence_error_is_excluded_not_reported_as_infinite():
    """A degenerate denominator must not manufacture a blow-up."""
    receipts = _receipts([-1.0] * 12)
    receipts.append({
        "commodity": "Perfect persistence",
        "model": {"mape": 0.1, "mae": 0.5, "n": 3},
        "naive_persistence": {"mape": 0.0, "mae": 0.0, "n": 3},
    })
    outliers = driver._commodity_regression_outliers(receipts)
    assert outliers["excluded_zero_persistence_error"] == 1
    assert not outliers["flagged"]


def test_losing_on_most_commodities_blocks_even_with_a_favourable_aggregate():
    """One enormous win cannot buy a release the model loses everywhere else."""
    diffs = [1.0] * 14 + [-200.0]
    gate = driver._publication_gate(
        {"mape": 8.0, "mae": 3.0, "n": 45}, {"mape": 10.0, "mae": 4.0, "n": 45},
        model_count=15, per_commodity=_receipts(diffs, naive_mae=400.0))
    assert gate["passed"] is False
    assert any("failed to beat persistence on" in reason for reason in gate["reasons"])


# --- determinism ------------------------------------------------------------


def test_bootstrap_is_deterministic_across_calls():
    """Two reads of one receipt must produce identical intervals, or the published
    confidence interval would depend on when the gate happened to run."""
    models = _live_models()
    first = driver._paired_metric_diagnostics(models, metric="mae")
    second = driver._paired_metric_diagnostics(models, metric="mae")
    assert first["ci95"] == second["ci95"]
    assert first["bootstrap_seed"] == driver.BOOTSTRAP_SEED


def test_single_commodity_cannot_be_declared_significant():
    diagnostics = driver._paired_metric_diagnostics(_receipts([-5.0]), metric="mae")
    assert diagnostics["evaluated"] is False
    assert "fewer than two" in diagnostics["reason"]


def test_missing_point_counts_are_skipped_rather_than_weighted_as_one():
    receipts = _receipts([-1.0] * 5)
    receipts[0]["model"]["n"] = 0
    diagnostics = driver._paired_metric_diagnostics(receipts, metric="mae")
    assert diagnostics["commodities"] == 4
