# 🏛️ ArchaeoTrace

[한국어](README.md)

ArchaeoTrace is a local-first, open-source QGIS plugin for human-reviewed
tracing of elevation contours on historical maps and for building DEM and
hillshade outputs from saved elevations. Its purpose is to make the complete
workflow inspectable and repeatable without sending map data to a remote
inference service.

## Current status

- The [QGIS plugin repository](https://plugins.qgis.org/plugins/ai_vectorizer/version/0.1.5/)
  published experimental `0.1.5` on 2026-08-26. Current source is the
  experimental `0.1.6` candidate with Ink v2, Smart Recovery, and an optional
  learned contour guide. It has not been published as a QGIS version, GitHub
  Release, or tag.
- The official QGIS `0.1.5` ZIP is 1,483,635 bytes with SHA-256
  `24f1def6acd63d483ea6bf7c20b944f56507ead52190667ec4f35562fca6c964`.
  The repository's `dist/ai_vectorizer-0.1.5.zip` is an earlier local candidate
  from commit `89b9f20` (SHA-256 `d2925198…`), not the
  official download.
- Ink v2, Smart Recovery, and later fixes on `main` are now grouped into the
  `0.1.6` candidate. Further development does not bump metadata, retag, or republish a same-version QGIS
  artifact. The `0.1.5–0.1.7` values in Git history and the unpublished `0.1.8`
  worktree did not represent additional QGIS releases; history is preserved.
- The metadata target is QGIS `3.22–4.99`, and the source contract is Python
  `3.8+`. Local checks cover Python 3.8/3.10/3.12 and macOS QGIS 3.44.8.
  Exact candidate commit `f48111a` passed QGIS 3.22.16/3.44.13/4.2.1 packaged
  import and runtime-safety checks plus deterministic Linux/Windows ZIP builds
  in [CI run 33344520364](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/33344520364).
  The commit, ZIP, and CI run are bound in the
  [0.1.6 release-readiness record](docs/RELEASE_READINESS_0.1.6.md).
- Implementation snapshot `3bf2b5a2c9e793f54286bfc633351bcb4aa2a095` on `main`
  passed all 12 jobs in [CI run 38437848923](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/38437848923).
  This is source validation, not a new release or evidence of trained-model accuracy.
- If you manually installed an older unpublished `0.1.7–0.1.8` source or ZIP,
  QGIS may not treat `0.1.6` as an update. Remove that plugin before installing
  the `0.1.6` ZIP. Verified model files remain outside the plugin directory.

## Features

- 🧪 [Learned contour pilot](learning/README.md): 144 prepared USGS crops, QGIS annotation
  tools, a resumable Colab notebook, CPU evaluation, and a default-OFF local ONNX
  guide. Human-approved annotations remain at zero; a separate weak-label
  experiment has completed three-seed training, ONNX export, and QGIS checks on
  USGS and Korean historical-map crops.
- Freehand tracing with no additional model
- Default multi-scale Ink Centerline evidence for dark and colored printed strokes
- Automatic dark-grid detection in native Byte Ink raises the cost of following
  a grid while allowing contours to cross it.
- Optional Smart Recovery that uses a verified EfficientSAM-Ti model only as a
  corridor prior; it is experimental, default OFF, and keeps the Ink route on failure
- Bounded, direction-aware Live-Wire with literal 0–100% coordinate blending
- Session-only manual avoid regions for text, symbols, or stains in Ink tracing
- Optional, session-only ink color sampling with `C`; `Shift+C` clears it
- An explicit, reviewed `Label-gap bridge` preview for black/brown contours
  actually interrupted by numeric labels
- LSD, HED, MobileSAM, SAM ViT-B, and Legacy Canny preserved under
  `Advanced / Legacy methods`
- A green preview of the path that the next click will accept
- New features and contour extensions through the QGIS edit buffer and Undo
- Elevation attributes and optional spot heights
- Experimental linear-TIN DEM and GDAL hillshade generation
- Korean and English UI

The existing tracing and DEM architecture is described in
[Features and architecture](docs/FEATURES_AND_ARCHITECTURE.md). The learned
pilot's data, training, and evaluation contract is in the [learning guide](learning/README.md).

## How it works

```text
Raster
  → source-grid multi-scale Ink score and tangent/coherence
  → optional local learned contour support and noncontour avoidance
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
| Learned contour guide | Experimental contour support and noncontour avoidance for Ink | SciPy, ONNX, ONNX Runtime, and an explicitly imported local model bundle |
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

## Experimental: ink color sampling

After starting an Ink trace, hover over a colored contour away from crossings,
text, and paper, then press `C`. `Sample ink color` also uses the last map
cursor position; the status shows the selected RGB. `Shift+C` or `Clear color`
removes it. Sampling requires ready native Byte RGB Ink v2 evidence, SciPy,
and assist above `0%`. Gray ink, paper, and ambiguous nearby colors are refused.

The optional sample adds finite soft cost to differently colored ink. It is
combined with avoid regions using the per-pixel maximum and shared by Ink and
Recovery. Same-color text and symbols remain ambiguous, so review the green
route. The color is reapplied to a new cache after panning and cleared when the
trace finishes, is cancelled, or resets. It is neither saved nor uploaded.
Historical-map accuracy and user-time improvement have not been established.

## Experimental: learned contour guide

This guide is **default OFF**. Open **Advanced: legacy tracing models** and
import a local directory containing `model.onnx`
and its hash/provenance `manifest.json` with **Import local model**, then enable
the checkbox. QGIS runs background CPU inference; PyTorch and the training stack
stay outside QGIS. Install the optional runtime using
[`requirements-learned-guide.txt`](requirements-learned-guide.txt) in the selected
QGIS Python environment. It requires Python 3.10+; older environments retain Ink.

The guide needs SciPy, Ink mode, assist above `0%`, and an aligned local
three-band Byte RGB GeoTIFF, PNG, or JPEG. It reads native 512×512 source-anchored
tiles and uses their central 384×384 cores with 64px halos. This fixes the context
for each source pixel during panning; it does not promise whole-image-equivalent
predictions. Missing or corrupt models, missing runtimes, invalid outputs, and
unsupported rasters fall back to Ink. Assist at `0%` skips inference.

Contour support joins the existing line evidence, while finite noncontour
avoidance costs combine with manual avoid regions and color hints and also reach
Smart Recovery. Late results cannot change an offered preview or a pending click.
Review the green path before accepting it; the usual edit-buffer, save, and Undo
workflow still applies.

The separate pilot freezes **16 USGS sheets / 144 native crops** before cropping:
72 train, 24 validation, 24 calibration, and 24 holdout. The lossless crops are
staged and verified against their declared source pixels, but **all labels remain
drafts without human approval**. Use **Annotate training crop** and **Review /
export labels** to correct the Ink draft, mark every visible contour in reviewed
areas, label other ink, and mask uncertain areas with `ignore`. A single reviewer
is recorded as an experimental pilot, not independent publication review.

The [Colab notebook](learning/colab_train.ipynb) uses a pinned public Git commit and
a separate Python 3.12 environment. It trains an ImageNet ResNet18 encoder with a
small U-Net decoder for `contour_support` and `other_ink_risk`, saves resumable
checkpoints to Drive, and selects one of three seeds using validation alone.

Before human review, a separate [weak-label notebook](learning/colab_train_weak_labels.ipynb)
can train a distinct experimental dataset with explicit `--allow-weak-labels`.
Annotations remain `machine_labeled`; checkpoints and ONNX provenance retain
`human_review_required=true`. Validation measures agreement with weak labels,
not actual contour accuracy. These models cannot enter human-reviewed pilot
Calibration/Holdout evaluation or be sealed as passing pilot models.

On 2026-10-14, all three seeds trained on the frozen 144-crop dataset on Colab
runtime `2026.07` (Python 3.12.13, Tesla T4), using 72 train and 24 validation
crops. Validation loss alone selected seed 3408 (60 epochs, weak-label loss
1.0018), exported as a 47.5 MiB FP32 ONNX model whose probabilities differ from
PyTorch by at most `1.7e-6`. On an M4 Mac CPU, prepared tile inference p95 was
69 ms with 254 MiB of extra RAM, and QGIS 3.44.8 saved preview coordinates
unchanged, undid in one step, and ran no inference at 0% assist.

The model learned USGS brown-contour conventions. It treats the same line drawn
in black or blue as something to avoid, so use the Korean retrained model below
for maps with black or blue contours, including many Korean maps. In a five-crop QGIS comparison on
validation sheets, it corrected one case where Ink jumped to a neighbouring
contour; the others were similar or not interpretable. That comparison uses a
machine reference and only shows direction. Human review, Calibration, and
Holdout evaluation remain pending; the [execution record](learning/COLAB_WEAK_RUN_2026-10-14.md)
holds the hashes and measurements.

On 2026-10-15 a second weak-label experiment added 149 crops from 16 black-ink
AMS Korea 1:50,000 (Series L751) sheets to the 144 USGS crops. Korean labels also
come from stroke geometry without human review: long, thin, curved black strokes
are contour candidates; blue overprint, straight grid lines, glyphs, and filled
symbols are other ink; unresolved pixels are excluded from the loss. Two
augmentations were enabled. `ink_recolor` repaints contour ink black, grey, blue,
or brown so colour alone cannot identify a contour, and `label_gaps` writes
elevation-style numbers across contours and cuts the printed line while the
target stays continuous, teaching the model to treat the number as other ink and
join the cut line.

The [Korean notebook](learning/colab_train_korea_weak.ipynb) trained all three
seeds for 60 epochs on a Colab T4 with 162 train and 48 validation crops.
Validation loss alone selected seed 3408 (0.860), exported as the 47.5 MiB FP32
`ai-weak-contour-guide-korea-v1`; Calibration and Holdout stayed unopened. On an
M4 Mac CPU, prepared tile p95 was 56 ms with 253 MiB of extra RAM.

The new model supports the synthetic blue and black lines that the first model
avoided (support 0.88 and 0.89), and its QGIS deviation from the synthetic blue
line fell from 37.7 px to 0.19 px. There is not yet evidence that it traces real
Korean contours better. The first same-click comparison was withdrawn because its
machine reference hopped between neighbouring contours and some clicks sat on grid
lines. In the one case whose reference follows a single contour, the new guide drew
the same line as Ink while the USGS-only guide cut across it; on a dense sheet with
grid lines all three methods failed. Click pairs verified on one contour, reviewed
answers, blue-contour sheets, and Calibration/Holdout evaluation remain pending,
and the model stays a default-OFF experiment. See the
[Korean execution record](learning/COLAB_KOREA_WEAK_RUN_2026-10-15.md).

For the human-reviewed pilot, calibration freezes path costs before the holdout
comparison. Human-reviewed Colab training, reviewed holdout results, and
Korean-map validation are still pending.
Synthetic diagnostic bundles only verify the software path and carry
`research_ready=false`; they are not domain-trained contour models. See the
[learning guide](learning/README.md) for annotation, training, export, and
evaluation commands. Elevation-number and legend reading with Unsloth remains
a later stage.

## Grid avoidance and contour continuity

Native Byte Ink caches detect long, dark, low-chroma grid candidates when at
least two separated horizontal and two vertical strokes are present. Small scan
rotation is supported. Live-Wire raises the finite cost of following the grid;
Recovery also receives a weaker finite cost. Raster pixels stay intact and
transverse contour crossings remain possible. Saturated blue grids, large
rotation, and short or broken grids can be missed; thick grid shoulders can
still attract the route. Classification depends on
the available cache context and is not guaranteed to stay identical after panning.

A trial guard that preferred Ink over learned routes regressed a real case where
Ink was wrong and the model was correct. That guard and the trial visible-ink
support restriction are **disabled by default**. Learned guidance remains
default OFF, and switching to neighboring contours is still unresolved. Explicit
reconstruction across numeric labels uses the reviewed `Label-gap bridge`
preview below.

On four validation cases inspected at enlarged native resolution before model
outputs, grid avoidance preserved the existing successes and failures. Synthetic
grid improvements are not evidence of higher accuracy on real maps. See the
[grid and continuity check](learning/GRID_CONTINUITY_CHECK_2026-10-15.md) for
settings, regressions and failures.

## Experimental: label-gap bridge

This is a reviewed, explicit action for a black or brown contour that a numeric
label has **actually interrupted**. It is not automatic reconnection.

```text
Confirm the pre-gap anchor → hover the continuation → G → review green preview → click that endpoint
```

| Evidence at the explicit endpoints | Result |
| --- | --- |
| Outward Ink support on both sides agrees with the gap | A bounded, assist-weighted curve preview |
| A glyph corrupts the 3px local direction field | Re-check only the outward 12px centerline support |
| No intended contour, only a parallel line/glyph, or diffuse direction | **Keep the existing Ink preview; create no bridge** |

Click the displayed endpoint to accept. `Esc` or a click elsewhere cancels only
the proposal, while hovering preserves it. There is no OCR, model, or automatic
endpoint selection. The behavior is product-path tested on synthetic fixtures;
it is not evidence of historical-map accuracy or user-time improvement.

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
   Smart Recovery or an imported learned guide only when wanted.
4. Inspect the green preview and the `Ink`, `Recovering`, `Enhanced`, or
   `Ink fallback` state. Older methods are under `Advanced / Legacy methods`.
5. In Ink tracing, Alt+click two corners to mark text/noise to avoid. The orange
   region is a soft local cost hint, not a hard wall; `Alt+Backspace` removes the
   latest region and `Alt+Shift+Backspace` clears all regions for that trace.
6. Accept anchors with clicks; use `Enter` or right-click to add the result to
   the edit buffer. Closing near the first point allows elevation entry.
7. Run `Save Layer Edits`, then use `Step 4 > Build DEM…` to review the grid and
   output paths.

## Data and safety boundary

- Raster crops, vector geometry, and local inference results are not uploaded
  to a remote inference service. There is no default telemetry.
- Network access occurs only when the user explicitly installs a Recovery model
  or downloads HED/SAM models. Recovery never auto-downloads; the
  content-addressed size, SHA-256, and ONNX session are prepared in a background
  task while Ink remains available. A corrupt regular cache object is
  quarantined and re-fetched only through the explicit `Repair Recovery Model`
  action; unsafe objects are never changed automatically. Recovery runs only on native Byte rasters;
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
- Learned-guide import uses a local verified bundle and never downloads a model.
  Dataset preparation and Colab encoder initialization are explicit, separate
  research commands; they are not QGIS inference uploads.
- Trace changes remain in the QGIS edit buffer until the user saves them.

See [SECURITY.md](SECURITY.md) for vulnerability reporting and diagnostic-data
redaction.

## Terrain limitations

The DEM path requires saved 2D contours, finite numeric elevations, at least two
distinct elevation values, non-collinear vertices, and a projected metre CRS.
The default grid limit is 25 million cells. Staged outputs are validated before
the DEM and hillshade pair is published.

Linear TIN is sensitive to sparse, incorrect, and out-of-range contours.
The `0.1.6` candidate does not yet produce topology QA, uncertainty/NoData layers, or
a provenance sidecar. Treat the output as a terrain hypothesis for review, not
as archaeological ground truth.

## Repository guide

- [Features and architecture](docs/FEATURES_AND_ARCHITECTURE.md): implemented
  flows, safety boundaries, and module map
- [Roadmap](ROADMAP.md): implemented work, next stages, and deliberate non-goals
- [Open-source development plan](docs/OPEN_SOURCE_DEVELOPMENT_PLAN.md): principles
  and delivery gates
- [0.1.6 release readiness](docs/RELEASE_READINESS_0.1.6.md): current candidate evidence and
  residual risk
- [0.1.5 historical readiness](docs/RELEASE_READINESS_0.1.5.md): published and
  frozen artifact identities
- [Contributing](CONTRIBUTING.md): development setup, test tiers, documentation,
  translation, and dataset contributions
- [Security](SECURITY.md): private reporting and safe diagnostics
- [Learning guide](learning/README.md): the separate USGS pilot, annotation
  review, Colab training/resume, ONNX export, and calibrated CPU evaluation

Routine development uses `main`, the repository's only named branch. Do not
create feature/release branches or pull requests automatically. If isolation is
needed, use a detached worktree, integrate the completed work into `main`, and
remove the temporary checkout; see [AGENTS.md](AGENTS.md).

The harness registers independent Ink v1, Ink v2, EfficientSAM, and product-like
Recovery method IDs. The first USGS HTMC sheet and six lossless crops are now
staged in the rights-and-annotation-validating 8-sheet/48-crop plan. The
remaining seven sheets and all independent review are still incomplete, so
`publication_ranking_eligible=false`. It does
not establish historical-map accuracy or superiority over another tool. See
[benchmarks/README.md](benchmarks/README.md) for its evidence format.

## Versioning, citation, and license

`ai_vectorizer/metadata.txt` is the release-version source of truth. Normal
development stays under `Unreleased`; metadata changes once when release
preparation begins. See [CHANGELOG.md](CHANGELOG.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

Validate current source through an isolated package; this does not touch the
repository-local `0.1.5` or `0.1.6` candidate ZIP/release trees:

```bash
current_source_dir="$(mktemp -d)"
current_source_zip="$current_source_dir/ai_vectorizer-unreleased.zip"
python3 scripts/package_release.py --output "$current_source_zip"
python3 scripts/package_release.py --check --output "$current_source_zip"
```

The metadata-derived repository-local candidate ZIP cannot be replaced when
the current-source hash differs from its recorded frozen SHA-256. That guard
hash is not the identity of the official QGIS download. A deliberate new
release requires the exact metadata version through
`--approve-release-overwrite VERSION`.

Use [CITATION.cff](CITATION.cff) for citation metadata.

GNU General Public License v2.0. [LICENSE](LICENSE)
