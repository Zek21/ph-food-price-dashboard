"""Rolling-origin backtesting, and what it says about the withheld release.

The gate had been deciding on ONE window of 177 points, where the MAE gap's 95% CI
was [-0.48, +1.01] -- no power to decide either way -- and the ticket was parked
waiting for WFP to publish actuals past 2026-06-15.

Waiting was never the only route.  `lstm_release_train.py` accepts
`--publication-cutoff` and a 59-model retrain costs ~33s, so the pipeline can be
rebuilt at earlier origins and scored on months those models never saw.  Run on
2026-09-12 over six origins (2026-03 back to 2024-12), that takes the evaluation
surface from 177 points to **3,717** using only data already on disk.

What it shows: the model sits at rough parity with naive persistence.  It is
favourable at 4 of 6 origins on MAE and 4 of 6 on MAPE, mean gap -0.1764, but the
sign test gives p=0.344 -- and that p is optimistic, because the origins share data
and their evaluation windows nest, so they are not independent draws.

Only 2025-09 is individually significant (95% CI [-1.4772, -0.1802]).  Testing six
origins, that is what chance produces.  Under Bonferroni the two-sided 99.17% CI is
[-1.7188, +0.0247] and includes zero; one-sided it is marginal (bootstrap
P(gap>0)=0.0053 against the 0.00417 a two-sided correction needs).  It cannot carry a
release.

One further fact worth keeping: retraining at the SAME 2026-03 cutoff produced an MAE
gap of -0.0324 (model slightly ahead) where the shipped residual-calibrated run
produced +0.2077 (model behind).  Same cutoff, opposite sign -- direct evidence that
the original blocking verdict was run-to-run noise, which is exactly what the paired
significance test concluded from the other direction.

Blow-ups are chronic rather than a one-window artefact: present at 4 of 6 origins,
worst 3 at the 18-month horizon.  The per-commodity guard is load-bearing.

These tests read committed receipts.  No model is trained here and nothing is
published; the harness itself is `rolling_origin_backtest.py`.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "gpu_driver_evidence" / "rolling_origin_20260912" / "rolling_origin_summary.json"
SINGLE_WINDOW_POINTS = 177


def _summary() -> dict:
    if not SUMMARY.is_file():
        pytest.skip("rolling-origin summary not present; run rolling_origin_backtest.py")
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_the_harness_multiplies_the_evaluation_surface():
    """The point of the exercise: power without waiting for new data."""
    summary = _summary()
    assert summary["origins_scored"] >= 6
    assert summary["total_points"] > 10 * SINGLE_WINDOW_POINTS
    for origin in summary["origins"]:
        assert origin["ok"], origin
        assert origin["points"] >= SINGLE_WINDOW_POINTS


def test_each_origin_is_scored_after_its_own_training_cutoff():
    """Guards the whole exercise: an origin scored on data it trained on is worthless."""
    for origin in _summary()["origins"]:
        cutoff = origin["training_cutoff_proof"]
        window_start = origin["window"].split("..")[0]
        assert cutoff == origin["origin"], (
            f"{origin['origin']}: ONNX metadata reports cutoff {cutoff}")
        assert window_start > cutoff, (
            f"{origin['origin']}: evaluation starts at {window_start}, not after {cutoff}")


def test_no_origin_establishes_a_win_after_correcting_for_six_tests():
    """The release bar, honestly applied across the whole surface."""
    summary = _summary()
    significant = summary["origins_significantly_better_on_mae"]
    assert len(significant) <= 1, (
        f"more origins are significant than chance explains: {significant}; re-examine "
        "whether a release is now defensible")
    assert not summary["origins_significantly_worse_on_mae"]


def test_the_cross_origin_sign_test_is_not_significant():
    summary = _summary()
    gaps = [origin["mae_gap"] for origin in summary["origins"]]
    favourable = sum(gap < 0 for gap in gaps)
    total = len(gaps)
    one_sided = sum(comb(total, k) for k in range(favourable, total + 1)) / 2 ** total
    assert one_sided > 0.05, (
        f"{favourable}/{total} origins favour the model at p={one_sided:.3f}; if this "
        "ever drops below 0.05, note the origins are NOT independent (their windows "
        "nest) before treating it as a result")
    assert np.mean(gaps) < 0, "the mean gap no longer even points the model's way"


def test_retraining_the_same_cutoff_can_flip_the_verdict_sign():
    """Why a single-window gate could never be trusted.

    The shipped residual-calibrated run at cutoff 2026-03 reported an MAE gap of
    +0.2077 and blocked the release on it.  A fresh retrain at the same cutoff lands
    on the other side of zero.
    """
    summary = _summary()
    fresh = next((o for o in summary["origins"] if o["origin"] == "2026-03"), None)
    if fresh is None:
        pytest.skip("2026-03 origin not in the summary")
    shipped_gap = 0.2077
    assert fresh["mae_gap"] * shipped_gap < 0, (
        f"expected an opposite sign to the shipped {shipped_gap:+}, got "
        f"{fresh['mae_gap']:+}")
    assert abs(fresh["mae_gap"]) < 0.2, "the fresh run is not close to parity either"


def test_blow_ups_persist_across_origins():
    """The per-commodity guard is load-bearing, not tuned to one window."""
    summary = _summary()
    with_blow_ups = [o["origin"] for o in summary["origins"] if o["blow_ups"]]
    assert len(with_blow_ups) >= 3, (
        f"only {with_blow_ups} show blow-ups; if they have genuinely gone away the "
        "guard's threshold should be revisited rather than left unexercised")


def test_the_model_wins_a_majority_of_commodities_at_every_origin():
    """The consistent finding the aggregate keeps hiding."""
    for origin in _summary()["origins"]:
        better = origin["commodities_model_better_mae"]
        worse = origin["commodities_model_worse_mae"]
        assert better >= worse, (
            f"{origin['origin']}: model better on {better} commodities, worse on {worse}")


def test_the_horizon_caveat_is_recorded_with_the_numbers():
    """Earlier origins are scored over longer horizons; pooling them would mislead."""
    summary = _summary()
    assert "horizon" in summary["horizon_caveat"].lower()
    horizons = {origin["origin"]: origin["horizon_months"] for origin in summary["origins"]}
    assert len(set(horizons.values())) > 1, "horizons are uniform; caveat may be stale"
