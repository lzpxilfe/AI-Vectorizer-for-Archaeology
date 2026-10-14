# Colab weak-label execution record — 2026-10-14

The separate USGS weak-label experiment finished all three seeds on a Colab T4,
exported one FP32 ONNX bundle, and passed local CPU and real-QGIS lifecycle
checks. Every label is machine generated: 0 crops are human approved, and
Calibration and Holdout were never opened. Nothing below is a contour accuracy
result, an improvement claim over Ink, or evidence for Korean maps.

## Training

| Recorded item | Evidence |
| --- | --- |
| Colab runtime | `2026.07`, Python `3.12.13`, Tesla T4 |
| Frozen cohort | 16 sheets / 144 native 512×512 RGB crops |
| Training access | 72 train and 24 validation crops |
| Review scope | `machine_labeled`; `human_review_required=true`; 0 human-approved samples |
| Training code | [`b9c3b9c`](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/commit/b9c3b9c8b3c80f483442a4febfdc2a694e86d0a2) |
| Dataset manifest SHA-256 | `143182ed4adc7f5e096aaff2f68e4c8afec35d2cc4b7bf3ff95f3da25a655346` |

| Seed | Stop | Epochs | Optimizer steps | Best weak-label validation loss |
| ---: | --- | ---: | ---: | ---: |
| 3407 | early stopping | 54 | 261 | 1.0792 |
| **3408** | max epochs | 60 | 289 | **1.0018** |
| 3409 | early stopping | 27 | 135 | 1.3651 |

Selection used validation loss alone (`calibration_accessed=false`,
`holdout_accessed=false`) and chose seed 3408, checkpoint SHA-256
`04710799ec9dbc9300eef00d91ac4b3a9da1dbf34ead20fb10bcc25403086735`.

## ONNX bundle

`ai-weak-contour-guide-v1` stores `model.onnx` (49,848,570 bytes, SHA-256
`1a1ac337c76057dda3e85c7d3c38933b41b2225fe3e6900ebd0e211f362e07ce`) with a
manifest that binds the checkpoint, dataset, code, environments, and
`review_level=automated_weak_labels`, `research_ready=false`,
`publication_ranking_eligible=false`, `GPL-2.0-only`. PyTorch and ONNX Runtime
CPU probabilities differed by at most `1.7e-6` (tolerance `1e-4`). The bundle
is kept outside git.

## Local checks (M4 Mac, CPU only)

| Check | Result | Target |
| --- | --- | --- |
| Bundle inspection and load | ready, 0.13 s | — |
| Prepared tile inference | p50 61 ms, p95 69 ms | p95 ≤ 1 s |
| Peak RSS growth (standalone) | 254 MiB | ≤ 512 MiB |
| Model size | 47.5 MiB | ≤ 64 MiB |
| Validation pixel agreement with weak labels | precision 0.70, recall 0.99 | not an accuracy metric |

In QGIS 3.44.8 (Python 3.12, NumPy 1.26.4, onnxruntime 1.31.0 on a separate
dependency path), `scripts/qgis_clean_profile_smoke.py --learned-bundle`
passed the lifecycle checks: CPU-only loading, prepared tile p95 71 ms, peak
RSS growth 439 MiB, preview coordinates saved unchanged, one Undo, and no
inference at 0% assist.

## Observed limitations

The model learned USGS colour conventions. On one synthetic line drawn in three
colours, brown received contour support 0.99, while blue and black received
avoidance 0.997 and 0.906. In QGIS the guided trace left the synthetic blue line
by a median 37.7 px (Ink: 0.24 px); the smoke now records this as a warning.
Maps whose contours are black or blue, including many Korean maps, therefore
need different training data before this model can help them.

Five validation crops were traced in QGIS with the same clicks for Ink and the
learned guide and compared with a machine reference that follows weak-label
centerline pixels between the clicks:

| Crop | Ink median / p95 px | Learned median / p95 px |
| --- | --- | --- |
| sheet-07-crop-12 | 0.39 / 7.27 | 0.34 / 15.74 |
| sheet-08-crop-03 | 0.36 / 14.05 | 0.37 / 5.43 |
| sheet-08-crop-04 | 9.95 / 28.91 | 0.27 / 3.05 |
| sheet-08-crop-09 | 0.20 / 0.50 | 0.20 / 0.50 |
| sheet-08-crop-11 | 0.34 / 10.20 | 0.34 / 9.81 |

Ink jumped to a neighbouring contour on sheet-08-crop-04, while the learned
guide stayed on the clicked one. On sheet-07-crop-12 the machine reference
itself switches between two nested contours, so its worse p95 is not
interpretable. Five cases against a machine reference only indicate direction.
Dense sheets also showed diffuse contour support and weak text avoidance.

## Remaining work

Human review of the pilot crops, Calibration of the support weight and
avoidance cost, a reviewed Holdout comparison against Ink and Smart Recovery,
and training data with black and blue contours remain open. The guide stays an
experimental, default-OFF import.

See the [weak-label notebook](colab_train_weak_labels.ipynb) and
[experiment guide](README.weak-labels.ko.md) for the explicit experimental scope.
