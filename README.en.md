# ArchaeoTrace

[한국어](README.md)

A QGIS plugin for tracing contours on scanned historical maps and building a DEM
and hillshade from their elevations. Check the green preview, click to accept a
segment, and draw manually where needed. Maps and tracing results are processed
on your computer.

QGIS 3.22–4.99 · Experimental · GPL v2

## Installation and language

In QGIS, open `Plugins > Manage and Install Plugins`, enable experimental plugins,
and search for ArchaeoTrace. To install a candidate ZIP, use `Install from ZIP`.
Experimental `0.1.7` has been submitted to the QGIS plugin repository and is
awaiting approval. The latest approved version remains `0.1.6`.
[Submission status](https://plugins.qgis.org/plugins/ai_vectorizer/version/0.1.7/) ·
[Final package and validation](docs/QGIS_SUBMISSION_0.1.7.md)

Use `Language` at the top of the panel to choose `English` or `한국어`.
On first use, the plugin follows the QGIS language: Korean for Korean locales,
English for other locales. It remembers any language you choose.

Freehand and the default Ink Centerline mode work without additional packages.
See the [English user guide](ai_vectorizer/README.en.md) for optional models and
installation instructions. A [Korean guide](ai_vectorizer/README.md) is also
available. Both guides are included in the ZIP.

## Trace a contour

1. Select a georeferenced map raster and an output line layer.
2. Choose Ink Centerline, set the assist strength, and start tracing.
3. Check the green preview, then click to accept each segment.
4. Press `Enter` or right-click to finish the contour and enter its elevation.
5. Use QGIS `Save Layer Edits` to write the changes to the file.

You can extend a contour from an existing endpoint. During a trace, `Ctrl+Z`
undoes the last checkpoint. After finishing, it undoes the QGIS edit operation.
`Shift`- or `Ctrl`-click draws a straight segment to the cursor.

## Tools

| Tool | Use |
| --- | --- |
| Ink Centerline | Propose a path along printed lines |
| Freehand and assist strength | Draw manually or blend the cursor and proposed path |
| Gap bridge `G` | Preview and accept a curve across a break |
| Ink colour `C` · avoid regions `Alt`+click | Select the line colour or mark text and stains to avoid |
| Smart Recovery · learned contour guide | Optional local model assistance; off by default |
| DEM and hillshade | Build a linear-TIN surface from contours and spot heights |

When building a DEM, check that you selected the elevation field. Fields with
names other than `elevation` require an explicit choice. Save input edits first
and use a projected CRS with metre units.

## Used in

[A Geoawesome case study](https://geoawesome.com/digitising-historical-maps-testing-ai-assisted-contour-tracing-in-qgis/)
used ArchaeoTrace to digitise 1929–39 WIG maps and 1960s–80s topographic maps of a
Polish sand mine, then compared the reconstructed terrain with a present-day
LiDAR DEM (October 2026). This is an external account of a specific workflow,
not a general accuracy assessment.

## Validation and limitations

The 0.1.7 review addressed Undo during contour extension, manual preview and save
consistency, existing elevations, DEM input checks, small-screen layouts and
QGIS 4 layer selection. The [review record](docs/RELEASE_REVIEW_2026-11-12.en.md)
identifies the tested candidate and the scope of the checks.

Three-seed Colab training, ONNX validation and QGIS comparisons are complete.
The new general-text model did not improve saved routes in the 13 positive cases
among 17 diagnostics, so the existing Korea model is retained. A later word-group
detector change also produced no route improvement and was not adopted.

- [Training and model comparison](learning/COLAB_KOREA_TEXT_WEAK_RUN_2026-10-19.md)
- [Text-gap diagnosis](learning/GENERAL_TEXT_FAILURE_DIAGNOSIS_2026-10-19.md) · [Detector experiment](learning/WORD_GROUP_EXPERIMENT_2026-11-11.md)
- [Real-map pilot](docs/REAL_MAP_PILOT_2026-10-19.md): three open fragments, before assigning elevations or geographic coordinates

These experiment records are in Korean. Accuracy against human-reviewed
reference lines and time savings have not yet been established. Check generated
DEMs against the source map: they depend on the contours supplied. Topology
checks and uncertainty layers are not yet provided.

## Data and development

Maps and tracing results are not sent to a remote inference service, and the
plugin collects no usage telemetry. Network access is used to read a WMS/WMTS
source you select or download a model you request. Models are checked against a
fixed size and SHA-256. See [Security](SECURITY.md).

[Architecture](docs/FEATURES_AND_ARCHITECTURE.md) · [Roadmap](ROADMAP.md) ·
[Benchmarks](benchmarks/README.md) · [Training records](learning/README.md) ·
[Contributing](CONTRIBUTING.md) · [Citation](CITATION.cff) · [Licence](LICENSE)
