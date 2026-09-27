"""Averaging across regions RAISES signal-to-noise; disaggregating would lower it.

AR-20260720 left an open question: the trainer collapses every (admin1, pricetype)
series for a commodity into one monthly mean, so does that averaging destroy
forecastable regional structure? It was worth asking because the model reaches only
parity with persistence, and an autocorrelation artifact pointed at that seam.

Measured on 2026-09-13 against wfp_food_prices_phl_latest.csv, the answer is the
reverse of the hypothesis, and two candidate mechanisms were tested before the third
survived:

  commodity-mean series   lag-1 acf +0.148 (mean over 61 commodities)
  underlying series       lag-1 acf +0.035 -- aggregation RAISES persistence, on 79%

  REFUTED -- compositional drift. The obvious suspect was the reporting panel
  changing month to month (median churn 42%, and 52 of 65 commodities never hold a
  constant panel), which would move the mean by WHO reported rather than by price. A
  balanced panel restricted to always-reporting series carries the same persistence
  (+0.167 vs +0.157 naive), and churn does not predict the gap: spearman rho=+0.17,
  p=0.31. Composition is not the source.

  CONFIRMED -- cross-sectional noise cancellation. 42% of the 448 individual balanced
  series have NEGATIVE lag-1 acf, the signature of measurement noise bouncing rather
  than of a price process. Averaging k series shrinks that idiosyncratic variance
  ~1/k while preserving the common component, so measured persistence must climb with
  k -- and it does, strictly monotonically: +0.043 at k=1 to +0.153 at k=8.

The consequence for the release gate is a route closed, not a defect fixed. The
trainer already feeds the best-conditioned target available; a per-region or
otherwise disaggregated model would train against series that are 42% noise-dominated
and near-white. The model's parity with persistence is therefore NOT an aggregation
artifact, and a retrain cycle spent on disaggregation would be spent against the
gradient.

These tests read the local CSV and compute statistics. They train nothing, write
nothing, and touch no live service.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WEBSITE = Path(__file__).resolve().parent.parent
if str(WEBSITE) not in sys.path:
    sys.path.insert(0, str(WEBSITE))

PRICES = Path("D:/ML/WFP/wfp_food_prices_phl_latest.csv")

# Drawn against the measurements above with margin, so ordinary data refreshes do not
# flip them but a mechanism change does.
MIN_NEGATIVE_ACF_FRACTION = 0.25  # measured 0.42 -- the noise signature
MIN_AGGREGATION_GAIN = 0.04       # measured +0.110 from k=1 to k=8
MIN_PANEL_SERIES = 8
SEED = 20260912


def _acf1(prices) -> float | None:
    """Lag-1 autocorrelation of log returns. None when the series is too short or flat."""
    values = np.asarray(prices, dtype=float)
    if len(values) < 24:
        return None
    returns = np.diff(np.log(np.clip(values, 1e-9, None)))
    centred = returns - returns.mean()
    denominator = (centred ** 2).sum()
    if denominator <= 0:
        return None
    return float((centred[:-1] * centred[1:]).sum() / denominator)


@pytest.fixture(scope="module")
def balanced_panels():
    """Per commodity, the wide matrix of series reporting in EVERY month.

    Holding the panel fixed is what separates a noise-cancellation effect from a
    composition effect -- with these columns, membership cannot vary by month.
    """
    if not PRICES.is_file():
        pytest.skip(f"price data not present at {PRICES}")
    import pandas as pd

    import lstm_release_train as rel

    _, monthly, _ = rel.load_and_engineer(PRICES)
    cut = pd.Period("2026-03", freq="M")

    panels = {}
    for commodity in sorted(monthly["commodity"].unique()):
        subset = monthly[(monthly["commodity"] == commodity)
                         & (monthly["date"].dt.to_period("M") <= cut)].copy()
        if subset.empty:
            continue
        subset["m"] = subset["date"].dt.to_period("M")
        subset["key"] = subset["admin1"].astype(str) + "|" + subset["pricetype"].astype(str)
        months = sorted(subset["m"].unique())
        if len(months) < 30:
            continue
        wide = subset.pivot_table(index="m", columns="key", values="price",
                                  observed=True).reindex(months)
        full = wide.dropna(axis=1)
        if full.shape[1] >= MIN_PANEL_SERIES:
            panels[commodity] = full
    if not panels:
        pytest.skip("no commodity has a balanced panel wide enough to test")
    return panels


def _mean_acf_at(panels, k: int, draws: int) -> float:
    rng = np.random.default_rng(SEED)
    values = []
    for full in panels.values():
        if full.shape[1] < k:
            continue
        for _ in range(draws):
            picked = rng.choice(full.shape[1], size=k, replace=False)
            acf = _acf1(full.iloc[:, picked].mean(axis=1).to_numpy())
            if acf is not None:
                values.append(acf)
    assert values, f"no usable draws at k={k}"
    return float(np.mean(values))


def test_the_panels_are_wide_enough_for_the_comparison(balanced_panels):
    """Without this the monotonicity test would be vacuous."""
    assert len(balanced_panels) >= 5
    assert all(p.shape[1] >= MIN_PANEL_SERIES for p in balanced_panels.values())


def test_individual_series_are_noise_dominated(balanced_panels):
    """The premise. Negative lag-1 acf means a reading tends to be corrected next
    month -- measurement noise, not a price process. If this fraction collapses, the
    noise-cancellation explanation below loses its mechanism and needs rechecking."""
    acfs = [a for full in balanced_panels.values()
            for a in (_acf1(full[col].to_numpy()) for col in full.columns)
            if a is not None]
    assert len(acfs) >= 100, f"only {len(acfs)} series; too few to characterise"
    negative = float(np.mean(np.asarray(acfs) < 0))
    assert negative >= MIN_NEGATIVE_ACF_FRACTION, (
        f"only {negative:.0%} of individual series have negative lag-1 acf "
        f"(was 42% on 2026-09-13); the noise signature that makes averaging helpful "
        "may no longer hold")


def test_aggregation_raises_persistence_monotonically(balanced_panels):
    """The invariant, and the falsifiable prediction of noise cancellation.

    Composition is held fixed by construction here, so a rise with k can only come
    from averaging away idiosyncratic variance.
    """
    curve = {k: _mean_acf_at(balanced_panels, k, draws=10) for k in (1, 2, 4, 8)}
    ordered = [curve[k] for k in sorted(curve)]
    assert all(b > a for a, b in zip(ordered, ordered[1:])), (
        f"acf1 no longer rises monotonically with panel size: {curve}")
    gain = ordered[-1] - ordered[0]
    assert gain >= MIN_AGGREGATION_GAIN, (
        f"averaging 8 series gains only {gain:+.4f} lag-1 acf over a single series "
        f"(was +0.110); {curve}")


def test_disaggregation_is_the_worse_conditioned_target(balanced_panels):
    """Guards the conclusion a future session is most likely to reverse: that the
    trainer should model regions separately to 'recover' structure."""
    single = _mean_acf_at(balanced_panels, 1, draws=10)
    pooled = _mean_acf_at(balanced_panels, MIN_PANEL_SERIES, draws=10)
    assert pooled > single, (
        "a single regional series is now at least as persistent as the pooled mean; "
        "the case against disaggregated modelling rests on this and would need "
        f"re-deriving (single={single:+.4f}, pooled={pooled:+.4f})")
