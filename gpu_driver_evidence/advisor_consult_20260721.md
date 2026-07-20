# CDP advisor consult — improving the AMD RX 6600 ML driver (2026-07-21)

Consulted **both** free CDP lanes on the SOCIALS browser (`skynet_ai_convene_cdp.py
--target both`): **ChatGPT (GPT-5, "Sol", High reasoning)** and **Gemini 3.5
(WVSU school account, authuser 3)**. Screenshots:
`data/ai_convene_cdp/20260721_004446_{chatgpt,gemini}_4_final.png` (ScreenMemory).

Both models **converged independently** on the same high-ROI changes:

| # | Recommendation | Both agreed |
|---|----------------|:-----------:|
| 1 | **Direct multi-horizon** output (emit h=1..H at once) instead of recursive one-step roll-forward — removes recursive error accumulation | ✅ ChatGPT + Gemini |
| 2 | **Log-return residual vs a baseline**: `target_h = log(y[t+h]) − log(baseline[t,h])`; `forecast_h = baseline · exp(residual_h)`. Zero output still = baseline, preserving the honest-gain design | ✅ ChatGPT + Gemini |
| 3 | **Robust loss — horizon-weighted Huber**, not MAPE/MSE | ✅ ChatGPT + Gemini |
| 4 | **Add lagged exogenous features** (ENSO ONI, USD/PHP, FAO FPI) already in the project | ✅ ChatGPT + Gemini |
| 5 | **Prove across many forecast origins**, not a single cutoff (stability) | ✅ ChatGPT |

ChatGPT's framing: *"the pooled R² is not the main evidence of improvement… the
engineering priority should be reducing recursive error accumulation and proving
gains across many historical forecast origins."*

## What v2 implements (`gpu_forecaster_v2.py`)

- (1) **Direct multi-horizon** MLP: one forward pass emits all H months — **zero
  recursion**.
- (2) **Log-return target vs persistence baseline**: `target_h = log(y[t+h]/y[t])`,
  `forecast_h = y[t]·exp(out_h)`; a zero output reproduces naive persistence
  exactly (honest-gain preserved, same as v1's delta design).
- (3) **Horizon-weighted Huber loss** on the log-return residuals.
- (5) **Multi-origin backtest** over the last N monthly cutoffs, aggregated, so the
  win over naive is shown to be stable rather than a single-origin fluke.
- Trains on the **AMD RX 6600** via torch-directml.

Deferred (documented next step, per rec #4): wire the ENSO/USD-PHP/FAO exogenous
features once `exogenous_data.json` is fetched — kept out of v2 to avoid a
network dependency and to keep the receipt self-contained.
