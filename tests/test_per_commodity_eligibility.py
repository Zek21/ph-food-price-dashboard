"""The eligibility rule must stay frozen, and stay strict.

per_commodity_eligibility.py exists because the model is a coin flip in aggregate
(179/354 commodity-origin MAE wins, 50.6%, gate cleared at 1 of 6 origins) yet skill is
concentrated: selecting on the four oldest origins and measuring on the two the
selection never saw gave 14/18 = 77.8% (p=0.0154) against 50/100 = 50.0% (p=0.5398).

Two things these tests protect.

1. THE RULE STAYS FROZEN. Retuning MIN_WINS_REQUIRED, REQUIRED_METRICS or
   MIN_SELECTION_ORIGINS against the same six origins would restore exactly the
   selection bias the holdout was built to exclude. A test failure here is the point:
   it forces a deliberate decision instead of a quiet drift.
2. THE RULE STAYS STRICT. "Wins on MAE" is not the rule -- BOTH metrics at EVERY
   selection origin are required. A prior selector that ranked commodities by the
   internal validation window of a single training run caught none of the blow-ups
   (tests/test_per_commodity_fallback_selector.py: "the ceiling is real, the selector
   is not"). What makes this one transfer is agreement across INDEPENDENT retrains, so
   any loosening removes the property that distinguishes it.

Synthetic receipts drive the unit tests; the integration test skips when the real
backtest output is absent. Nothing trains, nothing publishes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

WEBSITE = Path(__file__).resolve().parent.parent
if str(WEBSITE) not in sys.path:
    sys.path.insert(0, str(WEBSITE))

import per_commodity_eligibility as pce  # noqa: E402

REAL = WEBSITE / "gpu_driver_evidence" / "rolling_origin_20260914_full"


def _receipt(rows: dict[str, tuple[float, float, float, float]]) -> dict:
    """rows: commodity -> (model_mae, naive_mae, model_mape, naive_mape)."""
    return {"models": [
        {"commodity": c,
         "model": {"mae": mm, "mape": mp},
         "naive_persistence": {"mae": nm, "mape": np_}}
        for c, (mm, nm, mp, np_) in rows.items()]}


@pytest.fixture
def synthetic(tmp_path):
    """Winner wins everywhere; SplitMetric wins MAE but loses MAPE; Loser loses."""
    for tag in ("202401", "202402", "202403", "202404", "202405"):
        rows = {
            "Winner": (1.0, 2.0, 1.0, 2.0),
            "SplitMetric": (1.0, 2.0, 3.0, 2.0),
            "Loser": (2.0, 1.0, 2.0, 1.0),
        }
        (tmp_path / f"validation_{tag}.json").write_text(json.dumps(_receipt(rows)), encoding="utf-8")
    return tmp_path


# --- the rule stays frozen ---------------------------------------------------


def test_the_frozen_constants_are_unchanged():
    assert pce.MIN_WINS_REQUIRED == "all"
    assert pce.REQUIRED_METRICS == ("mae", "mape")
    assert pce.MIN_SELECTION_ORIGINS == 4, (
        "loosening the origin floor makes chance agreement easy; if this is intentional, "
        "the holdout must be re-run prospectively rather than re-tuned")


def test_the_frozen_selection_record_is_intact():
    assert len(pce.FROZEN_SELECTION_20260914) == 9
    assert set(pce.FROZEN_HOLDOUT_CONFIRMED_20260914) <= set(pce.FROZEN_SELECTION_20260914)
    for name in ("Anchovies", "Cabbage", "Coconut", "Fish (tilapia)", "Meat (beef)"):
        assert name in pce.FROZEN_HOLDOUT_CONFIRMED_20260914


# --- the rule stays strict ---------------------------------------------------


def test_both_metrics_are_required_not_either(synthetic):
    receipts = pce.load_receipts(synthetic)
    selected = pce.eligible_commodities(receipts, ["202401", "202402", "202403", "202404"])
    assert selected == ["Winner"]
    assert "SplitMetric" not in selected, (
        "a commodity winning MAE while losing MAPE was admitted; the rule requires BOTH")


def test_every_selection_origin_must_be_won(synthetic):
    """One loss anywhere in the selection window disqualifies."""
    rows = {"Winner": (2.0, 1.0, 2.0, 1.0)}  # Winner loses at this origin only
    (synthetic / "validation_202404.json").write_text(json.dumps(_receipt(rows)), encoding="utf-8")
    receipts = pce.load_receipts(synthetic)
    assert pce.eligible_commodities(receipts, ["202401", "202402", "202403", "202404"]) == []


def test_too_few_selection_origins_is_refused(synthetic):
    receipts = pce.load_receipts(synthetic)
    with pytest.raises(ValueError, match="selection origins"):
        pce.eligible_commodities(receipts, ["202401", "202402"])


def test_a_commodity_absent_from_one_origin_cannot_qualify(synthetic):
    """Consistency cannot be demonstrated where there is no observation."""
    rows = {"Loser": (2.0, 1.0, 2.0, 1.0)}  # Winner missing entirely here
    (synthetic / "validation_202403.json").write_text(json.dumps(_receipt(rows)), encoding="utf-8")
    receipts = pce.load_receipts(synthetic)
    assert pce.eligible_commodities(receipts, ["202401", "202402", "202403", "202404"]) == []


def test_a_missing_selection_origin_raises_rather_than_silently_shrinking(synthetic):
    receipts = pce.load_receipts(synthetic)
    with pytest.raises(ValueError, match="absent from receipts"):
        pce.eligible_commodities(receipts, ["202401", "202402", "202403", "209901"])


# --- statistics --------------------------------------------------------------


def test_binomial_tail_is_exact():
    assert pce.binomial_tail(18, 18) == pytest.approx(0.5 ** 18)
    assert pce.binomial_tail(0, 18) == pytest.approx(1.0)
    assert pce.binomial_tail(14, 18) == pytest.approx(0.0154, abs=5e-4)


def test_report_carries_its_own_truth_limits(synthetic):
    receipts = pce.load_receipts(synthetic)
    report = pce.holdout_report(receipts, ["202401", "202402", "202403", "202404"], ["202405"])
    joined = " ".join(report["truth_limits"]).lower()
    assert "owner-gated" in joined
    assert "does not beat persistence in aggregate" in joined
    assert "frozen" in joined


# --- integration against the real backtest -----------------------------------


def test_the_real_receipts_reproduce_the_recorded_holdout():
    if not REAL.is_dir():
        pytest.skip(f"{REAL} not present")
    receipts = pce.load_receipts(REAL)
    if len(receipts) < 6:
        pytest.skip(f"expected 6 origins, found {sorted(receipts)}")
    report = pce.holdout_report(receipts, ["202412", "202503", "202506", "202509"],
                                ["202512", "202603"])
    assert tuple(report["selected"]) == pce.FROZEN_SELECTION_20260914
    assert (report["selected_wins"], report["selected_trials"]) == (14, 18)
    assert report["selected_p_value"] == pytest.approx(0.0154, abs=5e-4)
    assert (report["unselected_wins"], report["unselected_trials"]) == (50, 100)
    assert report["transfers"] is True, "the selection no longer transfers; re-derive before relying on it"
