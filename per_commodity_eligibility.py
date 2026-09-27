#!/usr/bin/env python3
"""Per-commodity publication eligibility, with the rule FROZEN before the next origin.

Why this exists. The rolling-origin backtest (6 genuinely retrained origins, 3,717
out-of-time points) shows the model is a coin flip IN AGGREGATE -- 179 of 354
commodity-origin MAE comparisons, 50.6%, with the win-count distribution centred near
3/6 -- and it clears the publication gate at only 1 of 6 origins. So an aggregate
release is indefensible and all 1,062 predictions are withheld.

But skill is not absent, it is CONCENTRATED. Selecting commodities on the four OLDEST
origins alone and measuring on the two origins the selection never saw:

    selected     14/18 MAE wins = 77.8%   one-sided binomial p = 0.0154
    everyone else 50/100        = 50.0%   p = 0.5398 (exact chance)

The selection transfers. That makes a per-commodity eligibility filter defensible where
an aggregate release is not.

HOW THIS DIFFERS FROM THE SELECTOR ALREADY REFUTED. tests/test_per_commodity_fallback_
selector.py concluded "the ceiling is real, the selector is not" -- but that selector
ranked commodities by the INTERNAL VALIDATION WINDOW OF A SINGLE TRAINING RUN, and it
caught none of the blow-ups. This one requires AGREEMENT ACROSS INDEPENDENT RETRAINS at
different cutoffs, which is a different and much harder signal to satisfy by chance.
The objectives differ too: that work substituted persistence where the model was bad;
this one only withholds where the model is not reliably good. Publishing less is the
safer direction.

THE RULE IS FROZEN DELIBERATELY. Tuning MIN_CONSECUTIVE_WINS or the metric set against
the same six origins would re-introduce exactly the selection bias the holdout was
designed to exclude. The honest next test is prospective: apply this rule unchanged to
the NEXT origin as data arrives, and compare against FROZEN_SELECTION_20260914.

Read-only analysis over validation receipts. Trains nothing, publishes nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# --- FROZEN RULE (do not tune against the origins that produced it) ----------
MIN_WINS_REQUIRED = "all"          # a commodity must win EVERY selection origin
REQUIRED_METRICS = ("mae", "mape")  # on BOTH metrics, not either
MIN_SELECTION_ORIGINS = 4           # fewer origins makes chance agreement too easy

# The selection this rule produced on origins 2024-12..2025-09, recorded so a future
# prospective run can be compared rather than re-derived (and re-tuned).
FROZEN_SELECTION_20260914 = (
    "Anchovies", "Cabbage", "Coconut", "Eggs (duck)", "Fish (milkfish)",
    "Fish (tilapia)", "Meat (beef)", "Pineapples", "Squashes",
)
# Subset that additionally won BOTH held-out origins (2025-12, 2026-03).
FROZEN_HOLDOUT_CONFIRMED_20260914 = (
    "Anchovies", "Cabbage", "Coconut", "Fish (tilapia)", "Meat (beef)",
)


def load_receipts(directory: str | Path) -> dict[str, dict[str, tuple[dict, dict]]]:
    """{origin_tag: {commodity: (model_metrics, naive_metrics)}} from validation_*.json."""
    out: dict[str, dict[str, tuple[dict, dict]]] = {}
    for path in sorted(Path(directory).glob("validation_*.json")):
        tag = path.stem.replace("validation_", "")
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            continue
        rows = {}
        for entry in payload.get("models") or []:
            commodity = entry.get("commodity")
            model, naive = entry.get("model"), entry.get("naive_persistence")
            if commodity and isinstance(model, dict) and isinstance(naive, dict):
                rows[commodity] = (model, naive)
        if rows:
            out[tag] = rows
    return out


def _beats(pair: tuple[dict, dict], metric: str) -> bool:
    model, naive = pair
    return float(model[metric]) < float(naive[metric])


def eligible_commodities(receipts: dict, selection_origins: list[str]) -> list[str]:
    """Commodities winning EVERY selection origin on EVERY required metric.

    Requires MIN_SELECTION_ORIGINS or more, and only considers commodities scored at
    every one of them -- a commodity missing from an origin cannot be shown consistent.
    """
    if len(selection_origins) < MIN_SELECTION_ORIGINS:
        raise ValueError(
            f"need >= {MIN_SELECTION_ORIGINS} selection origins, got {len(selection_origins)}; "
            "fewer makes agreement by chance too easy to reach")
    missing = [o for o in selection_origins if o not in receipts]
    if missing:
        raise ValueError(f"selection origins absent from receipts: {missing}")
    common = set.intersection(*(set(receipts[o]) for o in selection_origins))
    return sorted(
        c for c in common
        if all(_beats(receipts[o][c], m) for o in selection_origins for m in REQUIRED_METRICS)
    )


def binomial_tail(k: int, n: int, p: float = 0.5) -> float:
    """One-sided P(X >= k). Exact, so a small holdout is not over-read."""
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def holdout_report(receipts: dict, selection_origins: list[str],
                   holdout_origins: list[str], metric: str = "mae") -> dict:
    """Score the frozen rule on origins it never saw."""
    selected = eligible_commodities(receipts, selection_origins)
    common = set.intersection(*(set(receipts[o]) for o in selection_origins + holdout_origins))
    rest = sorted(common - set(selected))

    def tally(group: list[str]) -> tuple[int, int]:
        wins = sum(1 for c in group for o in holdout_origins if _beats(receipts[o][c], metric))
        return wins, len(group) * len(holdout_origins)

    sel_w, sel_n = tally([c for c in selected if c in common])
    rest_w, rest_n = tally(rest)
    return {
        "rule": {"min_wins": MIN_WINS_REQUIRED, "metrics": list(REQUIRED_METRICS),
                 "min_selection_origins": MIN_SELECTION_ORIGINS},
        "selection_origins": selection_origins, "holdout_origins": holdout_origins,
        "selected": selected,
        "selected_wins": sel_w, "selected_trials": sel_n,
        "selected_rate": round(sel_w / sel_n, 4) if sel_n else None,
        "selected_p_value": round(binomial_tail(sel_w, sel_n), 4) if sel_n else None,
        "unselected_wins": rest_w, "unselected_trials": rest_n,
        "unselected_rate": round(rest_w / rest_n, 4) if rest_n else None,
        "unselected_p_value": round(binomial_tail(rest_w, rest_n), 4) if rest_n else None,
        "transfers": bool(sel_n and rest_n and (sel_w / sel_n) > (rest_w / rest_n)),
        "truth_limits": [
            "Eligibility is not a publication decision; releasing anything is owner-gated.",
            "The rule is frozen. Retuning it against these origins would restore the "
            "selection bias the holdout exists to exclude.",
            "The model does NOT beat persistence in aggregate; it clears the gate at 1 of 6 origins.",
        ],
    }
