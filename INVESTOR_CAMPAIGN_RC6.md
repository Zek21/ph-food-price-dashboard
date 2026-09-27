# RC6 investor-facing claim sheet

Native DirectML placement is verified on ZEKE's AMD Radeon RX 6600 for the tested ONNX LSTM graph: both LSTM operators and 15/18 node events were on DirectML; three events used CPU fallback. The compact graph was slower on DirectML: batch-1 medians 1.0352 ms versus 0.1740 ms CPU. This is inference placement evidence, not a GPU-only, speedup, or PyTorch GPU-training claim.

The authoritative 59-model validation has MAPE 3.9927% versus persistence 4.3018% and MAE 5.1355 versus 5.1679, but both paired confidence intervals cross zero and one commodity violates the unchanged 3x regression guard. All 1,062 current forward points remain private and **WITHHELD**. Do not present them as validated prices or use them for investor guidance.

The six-origin diagnostic reproduces all 354 commodity-origin count pairs and all naive-MAE pairs within 1%. Pooled MAE is persistence 10.3057, AR(1) 10.6965 and LSTM 10.1478; pooled MAPE is 8.1550%, 8.8216% and 8.1006%. AR(1) is worse and LSTM slightly better here. This supportive diagnostic is not the publication gate.

Suggested copy: "Our rc6 rerun verifies native DirectML placement for the tested ONNX LSTM graph on an RX 6600. CPU remains faster for this compact graph. Forecast validation fails the uncertainty and commodity-regression safeguards, so the 59-model, 1,062-point forward set stays private. The release shares reproducibility evidence, not a price forecast."

The published rc6 tag is historical; use the refreshed release audit assets and branch commit. The current 59-model rerun values are absent from the public payload and Git index. Legacy GPU/LSTM forward sections were removed from the new branch tip. Existing published Git history/tag archives and unrelated classical dashboard forecasts are not rewritten; this is not a claim that the entire historical repository contains no forecasts.
