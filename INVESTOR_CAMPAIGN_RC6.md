# RC6 investor-facing campaign claim sheet

## Core message

We verified native DirectML execution of the tested ONNX LSTM graph on an AMD Radeon RX 6600 and tightened the code gate so it only passes when every profiled LSTM operator is actually on DirectML. The same evidence also shows that this compact graph is slower on DirectML than CPU, so no speedup claim is made.

The forecasting result is intentionally conservative. The current 59-model run has slightly lower aggregate point estimates than persistence, but the paired confidence intervals include zero and one commodity violates the 3x regression guard. The 1,062 forward values were regenerated for audit but remain withheld from investor/public prediction claims.

A separate six-origin diagnostic reproduced the persistence harness across all 354 commodity-origin pairs. In that diagnostic AR(1) was worse than persistence (MAE 10.6965 vs 10.3057) while the LSTM was slightly better (10.1478). That result is supportive context, not a publication gate.

## Video narration

This release separates hardware proof from forecasting proof. On ZEKE's AMD Radeon RX 6600, ONNX Runtime profiling places 15 of 18 node events on DirectML, including both LSTM operators. But the CPU is still faster for this compact model. On prediction quality, the model's latest point estimates edge persistence, yet the uncertainty interval crosses zero and a per-commodity safety guard fails. So the 1,062 forward predictions stay private. The engineering result is not a forecast promise; it is a stricter, reproducible GPU proof and a validation system that refuses to publish when the evidence is not strong enough.

## Social copy

Skynet GPU release rc6: native DirectML placement is verified on an AMD RX 6600 for the tested ONNX LSTM graph ? 2/2 profiled LSTM operators and 15/18 total node events on DirectML. The compact graph is still faster on CPU. We also regenerated 1,062 forward predictions, but they remain withheld because the publication gate did not clear statistical and per-commodity safeguards. Reproducibility and receipts are in the GitHub release.

## Prohibited claims

Do not claim GPU-only execution, GPU speedup for this LSTM graph, validated forward prices, investor guidance from the withheld values, or that the pooled AR(1)/LSTM diagnostic is the publication gate.
