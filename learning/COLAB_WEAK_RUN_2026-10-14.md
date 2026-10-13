# Colab weak-label execution record — 2026-10-14

Actual GPU execution was observed for the separate USGS weak-label experiment.
This record captures progress during the run, not a completed model or a contour
accuracy result.

| Recorded item | Evidence |
| --- | --- |
| Colab runtime | `2026.07` |
| Python | `3.12.13` |
| GPU | Tesla T4 |
| Frozen cohort | 16 sheets / 144 native 512×512 RGB crops |
| Training access | 72 train and 24 validation crops |
| Observed progress | Seed 3407 completed at least 6 epochs |
| Epoch 6 validation loss | `1.517157`, measured against weak labels |
| Persistent checkpoints | `latest.pt` and `best.pt` saved to Drive |
| Review scope | `machine_labeled`; `human_review_required=true`; 0 human-approved samples |
| Remaining work | Complete all three seeds, select by validation, and export ONNX |

A downloaded seed 3407 `best.pt` snapshot was independently loaded with
`learning.train.load_checkpoint`: epoch 2, 10 optimizer steps, finite model
tensors, FP32 model tensor storage of 49,887,376 bytes, and 92 AdamW state
entries. Its recorded code, dataset, and weak-label flags matched this run.
This earlier snapshot verifies a usable persisted checkpoint; the live run had
already advanced to at least epoch 6.

Checkpoint SHA-256:
`497109711ae71220cfbbc6ab723788747d53d7f915042e671606faa25fba6044`.

The checkpoint records Torch `2.14.1`, torchvision `0.29.1`,
Segmentation Models PyTorch `0.5.0`, NumPy `2.5.4`, Pillow `12.3.0`, and device
`cuda`, in addition to the Python version above.

Training implementation: [`b9c3b9c8b3c80f483442a4febfdc2a694e86d0a2`](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/commit/b9c3b9c8b3c80f483442a4febfdc2a694e86d0a2).

Frozen dataset manifest SHA-256:
`143182ed4adc7f5e096aaff2f68e4c8afec35d2cc4b7bf3ff95f3da25a655346`.

Calibration and holdout data are excluded from the training loaders. The human
reviewed pilot and its holdout evaluation remain pending. Weak-label validation
loss measures agreement with generated labels; it does not establish actual
contour accuracy, an improvement over Ink, or Korean-map generalization.

See the [weak-label notebook](colab_train_weak_labels.ipynb) and
[experiment guide](README.weak-labels.ko.md) for the explicit experimental scope.
