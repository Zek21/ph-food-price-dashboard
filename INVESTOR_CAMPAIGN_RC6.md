# RC6 campaign packet: DirectML inference verified, forecasts withheld

Ticket: AR-20260720-ML-NATIVE-GPU-DRIVER-RELEASE-PREDICTIONS. Truth source: the 2026-09-28 rc6 receipts in `gpu_driver_evidence/rerun_20260928_rc6/`. This is a draft campaign packet for an article, video and social post; it is not proof that any of those has been published. The release shares reproducibility evidence. All current forward forecasts remain **WITHHELD** and private.

## Exact investor-facing claim sheet

Use the following wording with its qualification intact. Metrics below are benchmark timings and historical validation errors, never forward prices.

| Topic | Approved wording | Evidence |
| --- | --- | --- |
| Runtime | “The rc6 rerun used ZEKE's AMD Radeon RX 6600, ONNX Runtime 1.24.4 and Python 3.13.7.” | `hardware.json`, `benchmark_current.json` |
| Placement | “Both profiled LSTM operators (2/2) ran on DmlExecutionProvider. Of 18 total node events, 15 used DirectML and 3 used CPU fallback.” | `benchmark_current.json` |
| Speed | “CPU was faster at every measured batch for this compact graph. Batch-1 median latency was 1.0352 ms on DirectML versus 0.1740 ms on CPU.” | `benchmark_current.json` |
| Authoritative gate | “The current April–June 2026 validation covers 59 models and 177 points. LSTM MAPE was 3.9927% versus 4.3018% for persistence; MAE was 5.1355 versus 5.1679. These lower aggregate errors did not satisfy the publication gate.” | `validation_current.json` |
| Uncertainty and regression | “Paired 95% model-minus-persistence MAPE CI: [-0.758292, +0.099301]; MAE CI: [-0.684032, +0.651491]. Both cross zero. Fish (threadfin bream) MAE was 16.174x persistence, breaching the commodity regression safeguard.” | `validation_current.json` |
| Forecast disposition | “The authoritative disposition is withheld_failed_validation. The current forward artifact covers 59 models and 1,062 points. Forecasts are withheld and remain private; they are not validated prices or investor guidance.” | `validation_current.json`, value-free `private_prediction_audit.json` |
| Supportive diagnostic | “The six-origin diagnostic has 354/354 aligned pairs. Pooled MAE: persistence 10.3057, AR(1) 10.6965, LSTM 10.1478. Pooled MAPE: 8.1550%, 8.8216%, 8.1006%, respectively. AR(1) is worse and LSTM slightly better on these pooled measures. This supportive diagnostic is not the publication gate.” | `harness_alignment.json` |

Reusable pitch: “We verified native DirectML placement for a tested ONNX LSTM graph on an RX 6600 and measured the limits honestly: CPU was faster for this compact graph. The current forecast gate fails, so the forward set stays private. The engineering evidence is reproducible; it does not establish forecast reliability or investment returns.”

## Video narration and storyboard: target 55 seconds

Read at a measured pace and time the recorded take to 45–60 seconds. Use only the supplied text cards and a neutral hardware visual. Keep **FORECASTS WITHHELD** visible throughout. No price curves, forecast screenshots, training animations or speedup graphics.

| Time | Visual / on-screen text | Exact narration |
| --- | --- | --- |
| 0–10 s | RX 6600 visual; “Native DirectML inference verified” | “Can this AMD Radeon RX 6600 run our ONNX LSTM graph through DirectML? The rc6 profile says yes: both LSTM operators ran on DirectML.” |
| 10–20 s | “15/18 DirectML · 3 CPU fallback” | “Across the full profile, fifteen of eighteen node events used DirectML, with three on CPU. That is mixed execution.” |
| 20–33 s | “Batch 1 median: DirectML 1.0352 ms / CPU 0.1740 ms”; “CPU faster at every measured batch” | “CPU was faster at every measured batch. At batch one, DirectML took about one millisecond; CPU took less than two tenths.” |
| 33–46 s | “59 models · 177 validation points”; “withheld_failed_validation” | “Forecast validation still fails. Lower aggregate errors do not overcome confidence intervals crossing zero and a commodity regression safeguard breach.” |
| 46–55 s | “59 models · 1,062 forward points · PRIVATE”; “FORECASTS WITHHELD” | “Forecasts are withheld. All current forward points stay private. We are sharing reproducible engineering evidence, not validated prices or investor guidance.” |

The six-origin diagnostic is omitted from this short narration to keep the authoritative gate clear. Do not splice its pooled results into the validation card. The script and storyboard are drafts; no rendered video or timed recording is claimed.

## LinkedIn / social post draft

Native GPU execution is worth measuring. So are its limits.

Our rc6 profile on ZEKE's AMD Radeon RX 6600 placed both ONNX LSTM operators on DirectML: 15/18 total node events used DirectML, with 3 on CPU fallback. CPU was faster at every measured batch for this compact graph; batch-1 medians were 1.0352 ms DirectML versus 0.1740 ms CPU.

The authoritative validation gate still fails. Both paired error confidence intervals cross zero, and a commodity breaches the regression safeguard. **Forecasts are withheld:** the current 59-model, 1,062-point forward set stays private.

The useful result is reproducible inference evidence with explicit limits. It is not a claim of validated prices or investor guidance.

Article and evidence: [ARTICLE_URL]

#DirectML #MachineLearning #Reproducibility

Draft only. Replace `[ARTICLE_URL]` after the corrected article is published and independently verified. Do not post this placeholder.

## Prohibited-claim checklist

- [ ] No GPU-only execution claim; retain the 3 CPU fallback events.
- [ ] No GPU speedup claim for this graph; retain the CPU-faster finding.
- [ ] No PyTorch/TensorFlow GPU-training claim, GPU-trained delta-MLP narrative, training speedup or inherited historical training chart.
- [ ] No passed-gate claim, statistically established improvement, guaranteed prices, validated forward prices, investment returns or investor guidance.
- [ ] No substitution of the supportive six-origin diagnostic for the authoritative April–June 2026 gate.
- [ ] No current private forecast values, tables, screenshots, curves, exports or attachments in articles, video frames, captions, repository files or release assets.
- [ ] No claim that historical Git/tag archives or the entire repository are forecast-free.
- [ ] No claim of a published article, social post or completed video without direct read-after-write proof.

## Publication prerequisites

1. Check the exact claim sheet against the current hashed benchmark, validation, private audit and diagnostic receipts. Keep `withheld_failed_validation` and **FORECASTS WITHHELD** prominent. Engineering test success never changes the forecast gate.
2. Use the repaired `blog_amd_gpu_section.html` as the campaign source. It remains tracked, outside the existing rc6 release payload. Inspect the complete destination article, title, metadata, captions and images for stale training, speedup and passed-gate claims before publication.
3. Independently scan the Git index, staged changes and every public payload/ZIP member against the private current artifact without printing or copying its values. Verify checksums and the remote asset digests after refresh.
4. For a later video, produce a 45–60 second recording, inspect every frame and its captions, confirm the audible withheld statement, and verify the final export. This packet is a script, not a completed video.
5. Publish and independently read back the corrected article only in a separately authorized publishing step; then replace `[ARTICLE_URL]` with its verified permalink. Social posting is a separate step and is not requested here.
6. Identify the current rc6 branch commit and refreshed audit assets. Preserve the existing published tag. The current 59-model rerun values are absent from the public payload and Git index. Legacy GPU/LSTM forward sections were removed from the current branch tip. Existing history/tag archives and unrelated classical forecasts remain; do not generalize the current-value scan to all historical data.

Campaign readiness means the claims match the evidence. It does not authorize release of the private 1,062-point forward artifact or relax the forecast gate.
