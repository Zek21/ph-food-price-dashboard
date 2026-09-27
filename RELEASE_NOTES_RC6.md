# GPU Driver v1.0.0-rc6 - fresh ZEKE rerun, forecasts withheld

Ticket: AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS. Fresh run: 2026-09-28 Asia/Manila.

- Per-node proof: both LSTM operators on `DmlExecutionProvider`; 15/18 node events on DirectML, 3 on CPU.
- The tested compact graph was slower on DirectML at all measured batch sizes. Batch-1 medians: 1.0352 ms DirectML versus 0.1740 ms CPU.
- Authoritative April-June 2026 validation: 59 models, 177 points; MAPE 3.9927% versus persistence 4.3018%, MAE 5.1355 versus 5.1679. Both paired 95% intervals cross zero; one commodity breaches the unchanged 3x MAE guard.
- Current forward set: 59 models / 1,062 points, regenerated only in the private directory and **WITHHELD**.
- Supportive six-origin diagnostic: pooled MAE persistence 10.3057, AR(1) 10.6965, LSTM 10.1478. All 354 point counts match; all naive-MAE pairs agree within 1%. This is not the publication gate.
- Focused tests: 85 passed, 24 skipped, 1 warning in 10.25s. Full repo suite: 228 passed, 24 skipped, 1 warning in 153.56s (0:02:33). Skips identify absent historical receipts, not passing checks. Initial failures are preserved.

No GPU-only, GPU speedup, PyTorch GPU-training, validated-price, or investor-guidance claim is made.

The already-published rc6 tag remains at `1a5130fb8b127b43ac9e46255947f4b4ddca084d`. Fresh docs and receipts are supplied as release assets and in the `release/gpu-driver-v1.0.0-rc6` branch; the original tag source archive is historical. The current 59-model rerun values are absent from the public payload and Git index. Legacy GPU/LSTM forward sections were removed from the new branch tip. Existing published Git history/tag archives and unrelated classical dashboard forecasts are not rewritten; this is not a claim that the entire historical repository contains no forecasts.

Use `GPU_DRIVER_RC6_REPRODUCIBILITY.md`, `release_truth_manifest.json`, and `SHA256SUMS-rc6.txt` in the attached audit bundle.
