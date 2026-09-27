"""Per-commodity persistence fallback: the ceiling is real, the selector is not.

The release is blocked because the model is indistinguishable from naive persistence
on the only out-of-time window available, with one commodity 16x worse than doing
nothing.  The obvious repair is a per-commodity fallback: where the model cannot beat
persistence, use persistence.  Tested on 2026-09-12 against the committed receipts,
the family has a real ceiling and no usable selector.

CEILING (oracle, needs the answers, so an upper bound only): falling back on the 25
of 59 commodities the model loses on takes the MAE gap from +0.2077 to -0.7058 with a
paired 95% CI of [-1.0247, -0.4198] and zero blow-ups -- it clears the strengthened
gate decisively.  So the family is worth pursuing.

SELECTOR (what can actually be built): the internal validation window
2025-10..2026-03 is disjoint from the publication test 2026-04..2026-06 and was not
used for fitting or early stopping, so selecting on it is protocol-clean.  It catches
**0 of the 3 holdout blow-ups** and moves the aggregate MAE gap by **exactly
0.0000**, because the two commodities it selects are precisely the two that tied.

WHY it fails, and why this is structural: all three blow-ups BEAT persistence on the
internal window -- Beans (green, fresh) 31.35 vs 34.95, Shrimp (tiger) 7.27 vs 9.85,
Fish (threadfin bream) 2.42 vs 3.49 -- and then broke on the holdout at 2.7x, 2.2x
and 16.1x.  Historical per-commodity skill does not predict future blow-up, so a
skill ranking is the wrong instrument; what is needed is an instability/regime
detector.

Screening seven pre-holdout signals (internal MAE ratio, internal RMSE ratio,
internal tail shape, price level, history length, residual scale, internal MAPE) at
top-10 and top-25, the best captures 28.5% of the oracle gain at 52% precision
against a 39% base rate -- and a 2,000-draw random control captures 10.0% on average
with a p95 of 44.7%, so the best signal sits INSIDE the random distribution.  Several
signals are worse than doing nothing.

The screen was scored against a window that has already been inspected, so it is a
feasibility screen rather than a validated result.  That only matters in the
permissive direction, and the conclusion here is negative: nothing is licensed.

These tests read committed receipts and call pure functions.  No model is trained, no
prediction is published, no artifact is rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import gpu_forecast_driver as driver

ROOT = Path(__file__).resolve().parent.parent
RETRAIN = ROOT / "gpu_driver_evidence" / "rerun_20260911_residual_cal" / "lstm_retrain_receipt.json"
VALIDATION = ROOT / "gpu_driver_evidence" / "rerun_20260912" / "validation_current.json"
BLOW_UPS = ("Beans (green, fresh)", "Shrimp (tiger)", "Fish (threadfin bream)")


def _receipts() -> tuple[dict, dict]:
    if not (RETRAIN.is_file() and VALIDATION.is_file()):
        pytest.skip("release receipts not present")
    return (json.loads(RETRAIN.read_text(encoding="utf-8")),
            json.loads(VALIDATION.read_text(encoding="utf-8")))


def _weighted_gap(holdout: list[dict], fallback: set[str], metric: str = "mae") -> float:
    """Aggregate model-minus-naive gap when `fallback` commodities use persistence."""
    weights = np.array([row["model"]["n"] for row in holdout], dtype=float)
    model = np.array([
        row["naive_persistence"][metric] if row["commodity"] in fallback
        else row["model"][metric] for row in holdout])
    naive = np.array([row["naive_persistence"][metric] for row in holdout])
    return float(((model - naive) * weights).sum() / weights.sum())


def _substituted(holdout: list[dict], fallback: set[str]) -> list[dict]:
    return [{"commodity": row["commodity"],
             "naive_persistence": row["naive_persistence"],
             "model": dict(row["naive_persistence"]) if row["commodity"] in fallback
                      else dict(row["model"])}
            for row in holdout]


# --- the selection window must be clean -------------------------------------


def test_internal_window_is_disjoint_from_the_publication_test():
    """Selecting on the internal window must not peek at the holdout."""
    retrain, _ = _receipts()
    partition = retrain["partition"]
    assert partition["internal_validation_end"] < partition["publication_test_start"]
    assert partition["publication_test_used_for_training_or_early_stopping"] is False


# --- the ceiling is real ----------------------------------------------------


def test_oracle_fallback_would_clear_the_strengthened_gate():
    """Upper bound: worth pursuing a selector at all."""
    _, validation = _receipts()
    holdout = validation["models"]
    oracle = {row["commodity"] for row in holdout
              if row["model"]["mae"] >= row["naive_persistence"]["mae"]}
    assert _weighted_gap(holdout, oracle) < -0.5
    diagnostics = driver._paired_metric_diagnostics(
        _substituted(holdout, oracle), metric="mae")
    assert diagnostics["verdict"] == "model_significantly_better"
    assert diagnostics["ci95"][1] < 0
    assert not driver._commodity_regression_outliers(
        _substituted(holdout, oracle))["flagged"]


def test_a_fallback_commodity_can_only_tie_never_win():
    """The structural bound on this whole family of fixes.

    Substituting persistence makes the commodity contribute exactly the persistence
    error, so the aggregate deficit can shrink toward zero but any actual WIN has to
    come from the commodities left on the model.
    """
    _, validation = _receipts()
    holdout = validation["models"]
    every = {row["commodity"] for row in holdout}
    assert _weighted_gap(holdout, every) == pytest.approx(0.0, abs=1e-9)


# --- the selector is not ----------------------------------------------------


def test_internal_validation_selector_catches_none_of_the_blow_ups():
    retrain, validation = _receipts()
    internal = {row["commodity"]: row for row in retrain["models"]}
    selected = {c for c, row in internal.items()
                if row["internal_model"]["mae"] >= row["internal_naive_persistence"]["mae"]}
    caught = [c for c in BLOW_UPS if c in selected]
    assert not caught, f"internal selection unexpectedly caught {caught}"


def test_each_blow_up_actually_beat_persistence_on_the_internal_window():
    """Why a skill ranking cannot work: the failures were internal successes."""
    retrain, _ = _receipts()
    internal = {row["commodity"]: row for row in retrain["models"]}
    for commodity in BLOW_UPS:
        row = internal[commodity]
        assert row["internal_model"]["mae"] < row["internal_naive_persistence"]["mae"], (
            f"{commodity} no longer beats persistence internally; the reasoning in "
            "this module's docstring needs rechecking")


def test_internal_validation_selector_does_not_move_the_aggregate_gap():
    retrain, validation = _receipts()
    holdout = validation["models"]
    internal = {row["commodity"]: row for row in retrain["models"]}
    selected = {c for c, row in internal.items()
                if row["internal_model"]["mae"] >= row["internal_naive_persistence"]["mae"]}
    assert _weighted_gap(holdout, selected) == pytest.approx(
        _weighted_gap(holdout, set()), abs=1e-9)


def test_the_selected_commodities_are_exactly_the_ones_that_tied():
    """Explains the 0.0000: the substitution is a no-op on this window."""
    retrain, validation = _receipts()
    holdout = validation["models"]
    internal = {row["commodity"]: row for row in retrain["models"]}
    selected = {c for c, row in internal.items()
                if row["internal_model"]["mae"] >= row["internal_naive_persistence"]["mae"]}
    tied = {row["commodity"] for row in holdout
            if row["model"]["mae"] == row["naive_persistence"]["mae"]}
    assert selected <= tied, f"selected but not tied: {sorted(selected - tied)}"


def test_no_screened_selector_survives_a_random_control():
    """The finding that stops this being re-proposed.

    If a future signal genuinely beats chance, this test fails and demands that the
    candidate be predeclared and scored on a fresh holdout rather than adopted here.
    """
    retrain, validation = _receipts()
    holdout = validation["models"]
    internal = {row["commodity"]: row for row in retrain["models"]}
    baseline = _weighted_gap(holdout, set())
    oracle = _weighted_gap(holdout, {
        row["commodity"] for row in holdout
        if row["model"]["mae"] >= row["naive_persistence"]["mae"]})

    def signals(row: dict) -> dict[str, float]:
        model, naive = row["internal_model"], row["internal_naive_persistence"]
        return {
            "internal_mae_ratio": model["mae"] / naive["mae"] if naive["mae"] > 1e-12 else 0.0,
            "internal_rmse_ratio": model["rmse"] / naive["rmse"] if naive["rmse"] > 1e-12 else 0.0,
            "internal_tail_shape": model["rmse"] / model["mae"] if model["mae"] > 1e-12 else 0.0,
            "internal_level": naive["mae"],
            "short_history": -float(row["train_sequence_count"]),
            "residual_scale": float(row.get("residual_scale") or 1.0),
            "internal_mape": model["mape"],
        }

    captured = []
    for name in signals(retrain["models"][0]):
        ranked = sorted(internal, key=lambda c: -signals(internal[c])[name])
        for top_k in (10, 25):
            gap = _weighted_gap(holdout, set(ranked[:top_k]))
            captured.append(((baseline - gap) / (baseline - oracle), name, top_k))
    best_share, best_name, best_k = max(captured)

    rng = np.random.default_rng(20260912)
    commodities = list(internal)
    control = np.array([
        (baseline - _weighted_gap(holdout, set(rng.choice(commodities, size=best_k, replace=False))))
        / (baseline - oracle) for _ in range(2000)])
    assert best_share <= np.percentile(control, 95), (
        f"{best_name} top-{best_k} captured {best_share:.1%} of the oracle gain, above "
        f"the random p95 of {np.percentile(control, 95):.1%}. Predeclare it and score "
        "it on a fresh holdout; do not adopt it from this already-inspected window.")
