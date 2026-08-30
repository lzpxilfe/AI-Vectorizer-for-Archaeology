# 🏛️ ArchaeoTrace

[한국어](README.md)

ArchaeoTrace is a local-first, open-source QGIS plugin for human-reviewed
tracing of elevation contours on historical maps and for building DEM and
hillshade outputs from saved elevations. Its purpose is to make the complete
workflow inspectable and repeatable without sending map data to a remote
inference service.

## Current status

- The plugin metadata is an experimental `0.1.5` release candidate.
- New tracing work on `main` remains `Unreleased`. Development does not bump
  metadata, retag, or replace the existing `0.1.5` release ZIP.
- `0.1.5` is being prepared as the next release after `0.1.4` in the
  [QGIS plugin repository](https://plugins.qgis.org/plugins/ai_vectorizer/).
  The `0.1.5–0.1.7` values in Git history and the unpublished `0.1.8`
  worktree were development metadata, not QGIS repository releases. History is
  preserved rather than rewritten.
- The metadata target is QGIS `3.22–4.99`, and the source contract is Python
  `3.8+`. Local checks cover Python 3.8/3.10/3.12 and macOS QGIS 3.44.8. The
  QGIS 3.22/3.44/4.2 package matrix is configured but has not yet run remotely
  for this candidate. See the
  [release-readiness record](docs/RELEASE_READINESS_0.1.5.md) for exact evidence.
- If you manually installed an unpublished `0.1.6–0.1.8` source or ZIP, QGIS
  may not treat `0.1.5` as an update. Remove that plugin and install the verified
  `0.1.5` ZIP. Verified model files are kept outside the plugin directory in the
  QGIS profile.

## Features

- Freehand tracing with no additional model
- Default multi-scale Ink Centerline evidence for dark and colored printed strokes
- Optional Smart Recovery that uses a verified EfficientSAM-Ti model only as a
  corridor prior; it is experimental, default OFF, and keeps the Ink route on failure
- Bounded, direction-aware Live-Wire with literal 0–100% coordinate blending
- LSD, HED, MobileSAM, SAM ViT-B, and Legacy Canny preserved under
  `Advanced / Legacy methods`
- A green preview of the path that the next click will accept
- New features and contour extensions through the QGIS edit buffer and Undo
- Elevation attributes and optional spot heights
- Experimental linear-TIN DEM and GDAL hillshade generation
- Korean and English UI

The source-of-truth description of implemented, experimental, and planned
behavior is [Features and architecture](docs/FEATURES_AND_ARCHITECTURE.md).

## How it works

```text
Raster
  → source-grid multi-scale Ink score and tangent/coherence
  → bounded Live-Wire Ink champion
  → optional EfficientSAM corridor challenger on uncertain segments
  → challenger only when every safety check passes
  → human-reviewed green preview and anchors
  → QGIS edit buffer and Undo
  → elevation contours and optional spot heights
  → linear-TIN DEM and hillshade
```

An edge map or model mask is never saved as an archaeological fact. Only
segments accepted by the user become candidate geometry, and the final commit
remains an explicit QGIS `Save Layer Edits` action.

## Modes and optional dependencies

| Mode | Role | Additional runtime |
| --- | --- | --- |
| Freehand | Direct user input | No additional pip package or model |
| Ink Centerline | Multi-scale/color line evidence and bounded Live-Wire | None; SciPy/scikit-image optional |
| Smart Recovery | EfficientSAM corridor challenger for low-confidence Ink segments | ONNX Runtime and an explicitly installed, fixed-hash model |
| LSD | OpenCV line segments and shared Live-Wire | Advanced/legacy; OpenCV; SciPy optional |
| HED | Caffe HED edge map and shared Live-Wire | Advanced/legacy; OpenCV, a 56.1 MiB model; SciPy optional |
| MobileSAM | Point mask, edge/skeleton guidance, and A* | Advanced/legacy; OpenCV, PyTorch, backend, 38.8 MiB weights |
| SAM ViT-B | Point mask, edge/skeleton guidance, and A* | Advanced/legacy; OpenCV, PyTorch, backend, 357.7 MiB checkpoint |
| Legacy Canny | Gradient edge and Live-Wire | Advanced/legacy; OpenCV and SciPy optional |

NumPy from the QGIS Python environment is a common plugin prerequisite. Without
SciPy, every non-SAM edge mode (Ink/LSD/HED/Canny) falls back from
direction-aware Live-Wire to a bounded NumPy nearby-edge snap. A bounded search
can still follow adjacent contours, text, or symbols;
review the green path and correct it with anchors or Freehand. Smart Recovery
never saves the SAM mask as a line or binary-ORs it with Ink; it accepts a
challenger only after endpoint, detour, strong-Ink retention, and branch-switch
checks. Assist at `0%` skips model and evidence work.

The declared OpenCV range is 4.8–4.11:

```bash
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12"
```

The [plugin user guide](ai_vectorizer/README.md), which is included in the ZIP,
contains backend-specific installation commands, model hashes, manual paths,
shortcuts, and troubleshooting guidance.

## Quick start

1. Use `Plugins > Manage and Install Plugins > Install from ZIP` and enable
   ArchaeoTrace.
2. Select a georeferenced raster and a 2D line output layer.
3. Choose the default Ink Centerline or Freehand and an assist strength. Enable
   Smart Recovery only when wanted.
4. Inspect the green preview and the `Ink`, `Recovering`, `Enhanced`, or
   `Ink fallback` state. Older methods are under `Advanced / Legacy methods`.
5. Accept anchors with clicks; use `Enter` or right-click to add the result to
   the edit buffer. Closing near the first point allows elevation entry.
6. Run `Save Layer Edits`, then use `Step 4 > Build DEM…` to review the grid and
   output paths.

## Data and safety boundary

- Raster crops, vector geometry, and local inference results are not uploaded
  to a remote inference service. There is no default telemetry.
- Network access occurs only when the user explicitly installs a Recovery model
  or downloads HED/SAM models. Recovery never auto-downloads; the
  content-addressed size, SHA-256, and ONNX session are prepared in a background
  task while Ink remains available. Recovery runs only on native Byte rasters;
  wider integer rasters retain Ink v2. SAM
  Check/Status verifies a valid local checkpoint offline and queries its pinned
  source only when the file is missing. The EfficientSAM benchmark uses network
  access only for an explicit `model fetch`.
- A SAM status report can include the current working directory, QGIS/Python
  environment values, and model paths, and it is also copied to the clipboard.
  Review and redact it before sharing.
- Model artifacts use pinned sources, exact sizes and SHA-256 values, staged
  writes, `fsync`, atomic publication, and rollback. Symlinked storage targets
  are rejected.
- Trace changes remain in the QGIS edit buffer until the user saves them.

See [SECURITY.md](SECURITY.md) for vulnerability reporting and diagnostic-data
redaction.

## Terrain limitations

The DEM path requires saved 2D contours, finite numeric elevations, at least two
distinct elevation values, non-collinear vertices, and a projected metre CRS.
The default grid limit is 25 million cells. Staged outputs are validated before
the DEM and hillshade pair is published.

Linear TIN is sensitive to sparse, incorrect, and out-of-range contours.
Version `0.1.5` does not yet produce topology QA, uncertainty/NoData layers, or
a provenance sidecar. Treat the output as a terrain hypothesis for review, not
as archaeological ground truth.

## Repository guide

- [Features and architecture](docs/FEATURES_AND_ARCHITECTURE.md): implemented
  flows, safety boundaries, and module map
- [Roadmap](ROADMAP.md): implemented work, next stages, and deliberate non-goals
- [Open-source development plan](docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md): principles
  and delivery gates
- [Release readiness](docs/RELEASE_READINESS_0.1.5.md): verification evidence and
  residual risk
- [Contributing](CONTRIBUTING.md): development setup, test tiers, documentation,
  translation, and dataset contributions
- [Security](SECURITY.md): private reporting and safe diagnostics

The harness registers independent Ink v1, Ink v2, EfficientSAM, and product-like
Recovery method IDs. A rights-and-annotation validator and an 8-sheet/48-crop
template are included, but the redistributable source maps and independent
review are not yet populated, so `publication_ranking_eligible=false`. It does
not establish historical-map accuracy or superiority over another tool. See
[benchmarks/README.md](benchmarks/README.md) for its evidence format.

## Versioning, citation, and license

`ai_vectorizer/metadata.txt` is the release-version source of truth. Normal
development stays under `Unreleased`; metadata changes once when release
preparation begins. See [CHANGELOG.md](CHANGELOG.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

Validate current source through an isolated package; this does not touch the
frozen `dist/ai_vectorizer-0.1.5.zip` or production release tree:

```bash
current_source_dir="$(mktemp -d)"
current_source_zip="$current_source_dir/ai_vectorizer-unreleased.zip"
python3 scripts/package_release.py --output "$current_source_zip"
python3 scripts/package_release.py --check --output "$current_source_zip"
```

The metadata-derived production ZIP cannot be replaced when the current-source
hash differs from its recorded frozen SHA-256. A deliberate release requires
the exact metadata version through `--approve-release-overwrite VERSION`.

Use [CITATION.cff](CITATION.cff) for citation metadata.

GNU General Public License v2.0. [LICENSE](LICENSE)
