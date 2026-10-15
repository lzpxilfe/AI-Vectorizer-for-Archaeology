# Colab USGS + Korean weak-label execution record — 2026-10-15

The second weak-label experiment added 149 crops from black-ink AMS Korea
1:50,000 sheets to the 144 USGS crops and trained with the `ink_recolor` and
`label_gaps` augmentations. All three seeds finished on a Colab T4, one FP32
ONNX bundle was exported, and local CPU and real-QGIS checks passed. Every
label is machine generated: 0 crops are human approved, and Calibration and
Holdout were never opened. The same-click QGIS comparison in this record used
an invalid machine reference and is withdrawn; see that section.

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

## Same-click QGIS comparison (withdrawn)

Ten validation crops were traced in QGIS with the same two clicks for Ink, the
USGS-only guide and the new guide, and scored against a reference built from
weak-label centerline pixels. An inspection at native resolution showed that
this reference is not valid. The builder joins label fragments with a dilation
and follows the longest path, so on dense Korean sheets the path hops between
neighbouring contours. On kr-10-crop-08 and kr-10-crop-09 both clicks also sit
on straight grid lines, so no single contour connects them. Distances to such a
reference reward any line that stays near it, including one that cuts across
contours, and the averages first reported here (Ink 11.7 px, USGS-only 5.5 px,
Korea guide 3.6 px) are withdrawn.

Of the six Korean rows, only kr-10-crop-04 has a reference that follows one
contour from click to click. There Ink and the Korea guide both follow that
contour (median 0.21 px), while the USGS-only guide cut across it (6.37 px). On
kr-10-crop-08 every method failed: Ink followed the grid, the USGS-only guide
cut straight across, and the Korea guide also crossed contours diagonally. The
USGS rows were not re-inspected and are not used as evidence either.

The comparison therefore shows only that the Korea guide no longer breaks a
clean black contour that Ink already traces. It does not show better tracing on
dense Korean contours. Grid lines remain a separate problem: Ink snaps to them
when a click lands on or near one.

## Remaining work

These results come from weak labels and one selection-biased validation set,
and no valid Korean tracing comparison exists yet. A comparison needs click
pairs verified to lie on one contour, away from grid lines. Human review of Korean and USGS crops,
Calibration of the support weight and avoidance cost, a reviewed Holdout
comparison against Ink and Smart Recovery, and blue-contour Korean sheets
(the cohort is black ink only; blue is covered only through `ink_recolor`)
remain open. The guide stays an experimental, default-OFF import.

See the [Korean weak-label notebook](colab_train_korea_weak.ipynb) and the
[first execution record](COLAB_WEAK_RUN_2026-10-14.md).
