# Colab USGS + Korean weak-label execution record — 2026-10-15

The second weak-label experiment added 149 crops from black-ink AMS Korea
1:50,000 sheets to the 144 USGS crops and trained with the `ink_recolor` and
`label_gaps` augmentations. All three seeds finished on a Colab T4, one FP32
ONNX bundle was exported, and local CPU and real-QGIS checks passed. Every
label is machine generated: 0 crops are human approved, and Calibration and
Holdout were never opened. The comparisons below use the same validation crops
that selected the seed and a machine reference; they show direction, not
contour accuracy or general Korean-map performance.

## Training

| Recorded item | Evidence |
| --- | --- |
| Colab runtime | `2026.07`, Python `3.12.13`, Tesla T4, torch `2.14.1` |
| Frozen cohort | 16 USGS sheets / 144 crops and 16 AMS Korea L751 sheets / 149 crops |
| Training access | train 162 (72 USGS, 90 Korean), validation 48 (24 + 24) |
| Unopened | calibration 36 (24 + 12), holdout 47 (24 + 23) |
| Augmentation | `ink_recolor=true`, `label_gaps=true` |
| Review scope | `automated_weak_labels`; `human_review_required=true`; 0 human-approved samples |
| Training code | [`ea3a2d1`](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/commit/ea3a2d1a4525b0f0cd10c0746882d429686f1c4b) |
| Dataset manifest SHA-256 | `b2fd2a087f75de4e5e1d31b44f13dfdb1a8af3963e18e68f81f9cb4a91e32249` |

Korean validation crops come from kr-09 (Namwon) and kr-10 (Mupungjang).

| Seed | Stop | Epochs | Optimizer steps | Best weak-label validation loss |
| ---: | --- | ---: | ---: | ---: |
| 3407 | max epochs | 60 | 652 | 0.8657 (epoch 60) |
| **3408** | max epochs | 60 | 655 | **0.8603** (epoch 59) |
| 3409 | max epochs | 60 | 654 | 0.9563 (epoch 58) |

Seed 3409 started higher (validation 2.10 after epoch 1) and fluctuated more.
Validation loss alone selected seed 3408 (`calibration_accessed=false`,
`holdout_accessed=false`), checkpoint SHA-256
`525c99a23988d80390b9363831f35e2219052770f1cc37637bea52cc748130f6`. These losses
are not comparable with the first experiment because the validation set changed.

## ONNX bundle

`ai-weak-contour-guide-korea-v1` stores `model.onnx` (49,848,570 bytes, SHA-256
`ae5fbf01e8fb073d686f81ec4f2f23367bd66386eddbfde8b314777e02e70398`) with a
manifest that binds the checkpoint, dataset, code, environments and
`review_level=automated_weak_labels`, `research_ready=false`,
`publication_ranking_eligible=false`, `GPL-2.0-only`. PyTorch and ONNX Runtime
CPU probabilities differed by at most `2.9e-6` (tolerance `1e-4`). The bundle is
kept outside git.

## Local checks (M4 Mac, CPU only)

| Check | Result | Target |
| --- | --- | --- |
| Bundle inspection and load | ready, 0.14 s | — |
| Prepared tile inference | p50 55 ms, p95 56 ms | p95 ≤ 1 s |
| Peak RSS growth (standalone) | 253 MiB | ≤ 512 MiB |
| Model size | 47.5 MiB | ≤ 64 MiB |

Pixel agreement with weak labels on the 48 validation crops (threshold 0.5):

| Crops | USGS-only model precision / recall | Korea model precision / recall |
| --- | --- | --- |
| Korean (24) | 0.21 / 0.94 | 0.75 / 0.79 |
| USGS (24) | 0.70 / 0.99 | 0.76 / 0.92 |

The USGS-only model marked much of the Korean non-contour ink as contour. These
crops selected the new seed, so this table favours the new model.

One synthetic curve, drawn in three colours, measures the contour-support and
avoidance scores on the line:

| Line colour | USGS-only support / avoid | Korea model support / avoid |
| --- | --- | --- |
| blue | 0.007 / 0.997 | 0.876 / 0.058 |
| brown | 0.992 / 0.169 | 0.984 / 0.006 |
| black | 0.145 / 0.906 | 0.892 / 0.054 |

In QGIS 3.44.8, `scripts/qgis_clean_profile_smoke.py --learned-bundle` passed
with this bundle: the guided trace followed the synthetic blue line with a
0.19 px median error (USGS-only model: 37.7 px), prepared tile p95 55 ms, peak
RSS growth 435 MiB, preview coordinates saved unchanged, one Undo, and no
inference at 0% assist.

## Same-click QGIS comparison

Ten validation crops were traced in QGIS with the same two clicks for Ink, the
USGS-only guide and the new guide. The reference follows weak-label centerline
pixels between the clicks. The two Ink runs repeated within 1.41 px.

| Crop | Ink median / p95 px | USGS-only guide | Korea guide |
| --- | --- | --- | --- |
| kr-09-crop-08 | 3.61 / 16.31 | 6.80 / 18.83 | 3.44 / 14.29 |
| kr-09-crop-03 | 0.62 / 7.56 | 1.00 / 7.55 | 0.37 / 4.26 |
| kr-09-crop-04 | 0.60 / 7.55 | 0.99 / 6.13 | 0.37 / 4.27 |
| kr-10-crop-04 | 0.21 / 0.60 | 6.37 / 19.03 | 0.21 / 0.56 |
| kr-10-crop-08 | 51.22 / 63.99 | 14.07 / 26.99 | 11.10 / 26.29 |
| kr-10-crop-09 | 13.68 / 18.73 | 4.03 / 13.12 | 6.12 / 18.20 |
| sheet-07-crop-12 | 0.41 / 7.27 | 0.34 / 15.74 | 0.50 / 7.16 |
| sheet-08-crop-03 | 0.36 / 14.05 | 0.37 / 5.43 | 0.50 / 19.41 |
| sheet-08-crop-04 | 9.95 / 28.91 | 0.27 / 3.05 | 0.26 / 3.63 |
| sheet-08-crop-09 | 0.20 / 0.50 | 0.20 / 0.50 | 0.20 / 0.52 |

kr-09-crop-03 and kr-09-crop-04 overlap in the source sheet and trace the same
contour segment, so the Korean rows hold five independent cases. Averaged over
the six Korean rows, the median distance was Ink 11.66 px, USGS-only guide
5.54 px and Korea guide 3.60 px; reference coverage within 2 px was 0.41, 0.29
and 0.53.

On kr-10-crop-04 the USGS-only guide left a clean black contour that Ink traced
exactly; the new guide keeps Ink's path. On kr-10-crop-08 Ink followed a
straight black grid line and the USGS-only guide cut straight across the
contours; the new guide follows contour shape but still crosses to a
neighbouring contour, so that case remains a failure. On kr-10-crop-09 it was
better than Ink and worse than the USGS-only guide. On USGS crops it kept the
sheet-08-crop-04 correction, but sheet-08-crop-03 regressed at p95 (19.4 px
against Ink 14.1 px and USGS-only 5.4 px).

## Remaining work

These results come from weak labels, one selection-biased validation set and
five independent Korean cases. Human review of Korean and USGS crops,
Calibration of the support weight and avoidance cost, a reviewed Holdout
comparison against Ink and Smart Recovery, and blue-contour Korean sheets
(the cohort is black ink only; blue is covered only through `ink_recolor`)
remain open. The guide stays an experimental, default-OFF import.

See the [Korean weak-label notebook](colab_train_korea_weak.ipynb) and the
[first execution record](COLAB_WEAK_RUN_2026-10-14.md).
