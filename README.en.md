# 🏛️ ArchaeoTrace

[한국어](README.md)

A QGIS plugin for tracing elevation contours on scanned historical maps with a
human in the loop, and for building a DEM and hillshade from the traced contours.
Everything runs on your own computer; map data is never sent to a remote server.

![QGIS 3.22-4.99](https://img.shields.io/badge/QGIS-3.22--4.99-3c8c3c.svg)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)
![Experimental](https://img.shields.io/badge/status-experimental-f28c28.svg)
![Local first](https://img.shields.io/badge/processing-local--first-2f855a.svg)
![License GPLv2](https://img.shields.io/badge/license-GPLv2-d64541.svg)

## 💡 Why

Hand-digitising contours from old map scans is slow, repetitive and easy to get
subtly wrong. Fully automatic extraction, on the other hand, often trips over
elevation labels, grid lines and faded ink, and finding where it went wrong becomes
a job of its own.

ArchaeoTrace sits in between. The plugin proposes the path to follow as a green
preview; you check it and click to accept. **Only the segments you accept become
geometry.** Edge maps and AI masks are never saved as results.

## ✨ Features

- **Ink Centerline (default)** follows the centre of printed lines at several
  stroke widths. Works without extra packages.
- **Assist strength** from `0%` (exactly where your cursor is) to `100%` (exactly
  the proposed path).
- **Handling historical-map clutter** bridges contours broken by elevation labels
  (`G`), discourages following grid lines, lets you mark text or stains to avoid
  (`Alt`+click), and lets you pick a contour colour (`C`).
- **Optional AI assistance (off by default)** Smart Recovery uses EfficientSAM-Ti
  only on uncertain Ink segments, and a learned contour guide runs a local ONNX
  model. Both fall back to the Ink path on any failure.
- **Terrain reconstruction** turns contour elevations and spot heights into a
  linear-TIN DEM and hillshade.
- **Native QGIS editing** respects the edit buffer, single-step Undo and
  `Save Layer Edits`.
- **Korean and English UI**

## ⚙️ How it works

```text
Scanned map → line evidence → proposed path (green preview) → you click to accept
            → saved to a QGIS layer → elevations → DEM · hillshade
```

Optional AI assistance only improves the proposed path, and is discarded when it
fails safety checks. See [Features and architecture](docs/FEATURES_AND_ARCHITECTURE.md)
for the design and module map.

## 📦 Installation

1. In QGIS, open `Plugins > Manage and Install Plugins` and enable **Show also
   experimental plugins** in Settings.
2. Search for `ArchaeoTrace` and install it, or use `Install from ZIP` with a
   release ZIP.
3. That is all the default modes (Freehand, Ink Centerline) need. For AI assistance
   or OpenCV-based modes, follow the optional steps in the
   [user guide](ai_vectorizer/README.md).

## 🚀 Quick start

1. Choose a georeferenced map raster and a line layer to write to.
2. Choose `Ink Centerline` and an assist strength.
3. Click along a contour, checking and accepting the green preview. Press `Enter`
   or right-click to finish the line.
4. Enter the elevation and run `Save Layer Edits`.
5. Use `Step 4 > Build DEM…` to create the DEM and hillshade.

Shortcuts and troubleshooting are in the [user guide](ai_vectorizer/README.md).

## 🌍 Used in

- [Geoawesome, "Digitising Historical Maps: Testing AI-Assisted Contour Tracing in QGIS"](https://geoawesome.com/digitising-historical-maps-testing-ai-assisted-contour-tracing-in-qgis/)
  (October 2026) An external case study that digitised a 1929-39 WIG 1:25,000
  sheet and a 1960s-80s topographic map of a small Polish sand mine to rebuild the
  terrain before, during and after extraction, and compared it with a present-day
  LiDAR DEM.

If you use ArchaeoTrace in research or practice, please open an issue so it can be
listed here.

## 🧪 What the Colab experiments established

Three-seed training, ONNX verification and actual QGIS comparison are complete.
The new general-text model did not improve the saved routes in 13 positive cases
among 17 development diagnostics, so the existing Korea model is retained.
Human-approved annotations remain zero; formal Calibration/Holdout evaluation is pending.

- [Training/QGIS record](learning/COLAB_KOREA_TEXT_WEAK_RUN_2026-10-19.md): conditions, hashes, 17 cases and the model decision.
- [Failure diagnosis](learning/GENERAL_TEXT_FAILURE_DIAGNOSIS_2026-10-19.md): six development pairs and opt-in logs locate the missing O word group. A subsequent [detector experiment](learning/WORD_GROUP_EXPERIMENT_2026-11-11.md) did not improve saved routes and was not adopted.
- [Real-map pilot](docs/REAL_MAP_PILOT_2026-10-19.md): the Ink kernel saved three open fragments without a learned model. This is a small draft with no elevations or geographic coordinates. These detailed records are in Korean.

## 🚧 Status and limitations

- The plugin is experimental. The latest release in the official QGIS plugin
  repository is `0.1.6` (published 2026-09-09). Current source is an **unpublished
  `0.1.7` experimental candidate**, addressing active-trace Undo, manual preview/click
  consistency, existing elevations on resume, DEM input validation and small-screen UI.
  Learned guidance stays default OFF. See the [CHANGELOG](CHANGELOG.md) and
  [final candidate verification record](docs/RELEASE_REVIEW_2026-11-12.md).
- Accuracy against human-reviewed reference lines and time savings for users are
  **still being evaluated**. Progress is recorded in
  [benchmarks](benchmarks/README.md) and [learning](learning/README.md).
- A generated DEM is a **terrain hypothesis to review**, not ground truth. It is
  sensitive to sparse or wrong contours and does not yet include topology QA or
  uncertainty layers.
- Always inspect proposed lines before accepting them.

## 🔒 Data and privacy

Maps, traced geometry and inference results never leave your computer, and there is
no telemetry. Network access reads a WMS/WMTS map you selected or downloads an
AI model you requested. Downloaded models are verified against a fixed size and SHA-256. See
[SECURITY](SECURITY.md) for details.

## 📚 Further reading

| Document | Contents |
| --- | --- |
| [User guide](ai_vectorizer/README.md) | Optional installs, modes, shortcuts, troubleshooting |
| [Features and architecture](docs/FEATURES_AND_ARCHITECTURE.md) | Implementation flow, safety boundaries, module map |
| [Roadmap](ROADMAP.md) | Next steps and deliberate non-goals |
| [Learning pilot](learning/README.md) | Data, training, evaluation and run records for the learned guide |
| [Benchmarks](benchmarks/README.md) | Reproducible evaluation harness and evidence format |
| [Contributing](CONTRIBUTING.md) | Development setup, tests, release procedure |

## 🤝 Citation, contributing and license

If you use ArchaeoTrace in research, teaching or applied work, please cite it with
[CITATION.cff](CITATION.cff). Bug reports, translations, documentation and
redistributable sample maps are welcome; start with [CONTRIBUTING](CONTRIBUTING.md).

GNU General Public License v2.0 · [LICENSE](LICENSE)
