# ArchaeoTrace user guide

[한국어](README.md)

ArchaeoTrace is a QGIS plugin for tracing contours from scanned maps and building
a DEM and hillshade from their elevations. It proposes a green path for you to
check and accept. Maps, geometry and model inference are processed locally.

This guide accompanies the unpublished `0.1.7` experimental candidate. Its
metadata targets QGIS `3.22–4.99` and Python `3.8+` source compatibility; optional
packages have their own Python requirements. See the
[release review](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/docs/LANGUAGE_REVIEW_2026-11-12.md)
for the exact artifact and tested environments. Experimental features still need
human review; the plugin has no established accuracy or time-saving guarantee.

## Install and choose a language

1. In QGIS, open `Plugins > Manage and Install Plugins > Install from ZIP` and
   select the plugin ZIP. Restart QGIS and enable **ArchaeoTrace**.
2. Open the ArchaeoTrace panel. At the top, choose **English** in the **Language**
   selector. The choice is saved in your QGIS settings.
3. Start with **Ink Centerline**, the default mode. It and **Freehand (No Assist)**
   work with QGIS's NumPy, without downloading a model or installing extra packages.

Official QGIS repository releases can also be installed through the plugin
manager after enabling **Show also experimental plugins**. They may differ from
this source candidate. If you previously installed a development ZIP labelled
`0.1.7` or `0.1.8`, check the new ZIP's metadata and hash: a version number alone
does not identify those builds. Model caches are outside the plugin directory
and are retained when the plugin is reinstalled.

## Trace and save a contour

1. Load a georeferenced raster in QGIS, then select it under **Input Map**.
2. Under **Output File**, create a Shapefile or select an existing editable 2D
   line layer, such as a GeoPackage layer. Elevation is stored in an attribute,
   rather than in Z/M coordinates. **Start Tracing** becomes available when both
   the raster and output layer are ready.
3. Leave **Ink Centerline** selected and choose an assist strength. `100%` uses
   the proposed path; `1–99%` blends it with your cursor path. `0%` uses the exact
   cursor position and skips edge detection and model inference. **Freehand**
   also gives you manual control.
4. Press **Start Tracing**, then click a visible contour to place the first
   anchor. Move along the line, check the green preview and click to accept each
   segment. Each click establishes a checkpoint. You can also drag to draw.
   If the plugin is still preparing a clicked segment, wait before clicking again.
5. Press **Enter** or right-click to finish an open line, or click near its
   starting point to close it. Enter its elevation in metres. Finishing uses
   the accepted segments; an unaccepted hover preview is not added to the line.
6. Use QGIS **Save Layer Edits** to write the result to the file. Finishing a
   line only adds it to the layer's edit buffer.

To extend an existing contour, start near one of its endpoints and use the snap
marker to select that end. Finish the extension as above. The elevation dialog
starts with the existing finite elevation; keep it or enter a deliberate change.
Cancelling the trace leaves the existing feature intact.

To record a spot height, double-click the starting point before drawing a longer
line, then enter the elevation. The point goes into **Spot Heights**. Save its
edits as well; if it is a temporary layer, export it to a file for later use.

### Undo and keyboard controls

These shortcuts apply while the map canvas has focus.

| Control | Action |
| --- | --- |
| `Ctrl+Z` / `Cmd+Z`, or plain `Backspace`, while tracing | Return to the previous checkpoint. Stops at the first anchor and keeps the existing feature when extending a contour. |
| `Ctrl+Z` / `Cmd+Z` after finishing | Use QGIS Undo to reverse the complete addition or extension in the edit buffer. |
| `Shift`+click or `Ctrl`+click | Accept a straight segment to the exact cursor position. |
| `Enter` or right-click | Finish the current line and enter its elevation. |
| `Esc` or `Delete` | Cancel the current trace. If a label-gap proposal is showing, `Esc` cancels that proposal first. |
| `Alt`+click two corners | Mark a rectangle of text or noise for Ink to avoid. This is two separate clicks, not a double-click. |
| `Alt+Backspace` | Remove the most recent avoidance rectangle. |
| `Alt+Shift+Backspace` | Remove all avoidance rectangles. |
| `C` | Sample a coloured contour under the cursor for the current Ink trace. |
| `Shift+C` | Clear the sampled colour. |
| `G` | Propose a curve across a gap from the last anchor to the visible endpoint under the cursor. |

Undo is not a substitute for saving. Check the layer's edit state before closing
QGIS, and use **Save Layer Edits** when you want the changes written to disk.

## Keep the preview on the intended line

Ink follows visible line evidence and can still choose a nearby contour, text
stroke or junction. Add closer checkpoints, lower the assist strength, use a
manual straight segment, or switch to Freehand when the preview is wrong.
Without SciPy, non-SAM modes use a local NumPy edge snap instead of the full
direction-aware Live-Wire path search.

Ink can discourage following long dark grid lines while allowing contours to
cross them. It can miss broken, rotated or saturated blue grids, and a thick
grid's edges may still attract the path. Check again after panning, since the
available image context changes.

### Colour and avoidance

For a clearly coloured contour, start an Ink trace, hover over the line and press
`C`. The sample button uses the last map cursor position. Grey ink, paper and
ambiguous samples are refused. This feature needs native Byte RGB raster
evidence, SciPy and assist above `0%`. The colour remains active after panning
but is cleared when the trace finishes or is cancelled. It is not saved or
uploaded. Text printed in the same colour can still confuse the route.

Use `Alt`+click on opposite corners to mark text, symbols or stains for avoidance.
These hints influence the proposed path; they do not remove pixels from your map.

### Gaps under labels

Ink can propose connections across some numeric and text gaps during ordinary
tracing. This requires SciPy and assist above `0%`, but no OCR or downloaded model.
It checks visible line ends and nearby competing lines; joined glyphs and
ambiguous junctions may still prevent a useful connection.

For an explicit **Label-gap bridge**, place an anchor on the last visible stroke
before the gap. Hover over its first visible continuation and press `G`. Review
the green curve, then click that same endpoint to accept it. `Esc` or a click at
another target cancels the proposal. If the endpoints or curve do not pass the
checks, the plugin keeps the ordinary Ink path. A curve through hidden ink is an
inference for you to review, not a recovered ground-truth line.

## Build a DEM and hillshade

1. Save edits to your contour layer and any spot-height layer.
2. Open **Terrain Reconstruction > Build DEM…** and choose the contour layer.
3. Select its numeric elevation field. Only a field named `elevation` is selected
   automatically. Choose custom fields such as `height_m` explicitly and check
   that an identifier such as `id` has not been selected as height.
4. If needed, choose a spot-height point layer and its numeric elevation field.
   **None** excludes spot heights. Contours and points must use the same CRS and
   elevation units.
5. Check the cell size and output paths, then build the GeoTIFF DEM and hillshade.
   Existing output files require confirmation before replacement.

Inputs need a projected CRS in metres, nonempty geometry, finite XY coordinates
and numeric elevations. The combined inputs must contain at least two distinct
elevations and three non-collinear vertices. Invalid coordinates are rejected
before interpolation. Temporary spot layers are supported, but saving them to a
file makes the work reproducible.

The workflow uses QGIS linear TIN interpolation and GDAL hillshade. QGIS
Processing must provide `qgis:tininterpolation`, `gdal:translate`,
`native:rasterlayerstatistics` and `gdal:hillshade`. The default output limit is
25 million cells.

Inspect the result against the source map. Sparse or incorrect contours and
areas outside their coverage limit what the DEM can represent. It is a terrain
hypothesis; topology checks and uncertainty layers are not yet included.

## Optional modes and packages

**Smart Recovery** and **Learned contour guide** are experimental and off by
default. The legacy modes are under **Advanced / Tracing modes**. Model weights
are not included in the plugin ZIP.

| Option | Purpose and requirements |
| --- | --- |
| Smart Recovery | Uses a local EfficientSAM-Ti model to reconsider uncertain Ink segments. Requires ONNX Runtime and about 39.4 MiB of model files. Keeps Ink when the model is unavailable, fails or offers no acceptable improvement. |
| Learned contour guide | Uses a local ONNX bundle to support contours and, when configured, discourage text or roads. Requires SciPy, ONNX validation and CPU ONNX Runtime. |
| LSD | Line-segment detection with OpenCV. |
| HED | Learned edge detection with OpenCV and about 56.1 MiB of weights. |
| MobileSAM | Point-prompt segmentation with OpenCV, PyTorch, MobileSAM and about 38.8 MiB of weights. |
| SAM (ViT-B) | Point-prompt segmentation with OpenCV, PyTorch, `segment_anything` and about 357.7 MiB of weights. |
| Legacy Canny | Edge detection with NumPy; OpenCV and SciPy are optional. |

Install packages into **QGIS's Python**, not an unrelated system Python. Replace
`<QGIS_PYTHON>` below with its executable path. Use a supported Python environment
for optional packages, preferably Python 3.11 or later. Python 3.8 source
compatibility does not mean a current optional dependency stack can be installed
on it. Back up the profile and check pip's proposed changes before changing the
Python environment shared by QGIS.

```bash
# Optional Live-Wire and image-processing support
<QGIS_PYTHON> -m pip install scipy scikit-image

# LSD, HED and SAM modes
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12"

# Smart Recovery
<QGIS_PYTHON> -m pip install "onnxruntime>=1.17,<2"

# Learned guide: versions tested together on Python 3.12
<QGIS_PYTHON> -m pip install "scipy>=1.10" "onnx==1.23.2" "onnxruntime==1.31.0"

# MobileSAM, including download checks
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12" requests torch torchvision \
  "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2"

# SAM (ViT-B), including download checks
<QGIS_PYTHON> -m pip install "opencv-python-headless>=4.8,<4.12" requests torch torchvision \
  "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"
```

Install only the lines needed for your chosen mode, then restart QGIS. Restricting
OpenCV's version does not by itself ensure compatibility with QGIS's NumPy.
Separate training packages are not needed for tracing; training belongs in its
own environment, not QGIS's Python.

### Set up a model

- **Smart Recovery:** enable it, then explicitly press **Install Recovery Model**.
  Downloads are checked against fixed sizes and SHA-256 hashes before use. A
  damaged installed model offers **Repair Recovery Model**. Ink remains usable
  while model files and sessions are being prepared. Status shows **Ink**,
  **Recovering**, **Enhanced** or **Ink fallback**. Recovery needs native Byte
  raster evidence.
- **Learned contour guide:** use **Import local model** to choose a folder
  containing `model.onnx` and `manifest.json`, then enable **Learned contour guide**.
  The plugin validates the bundle and copies it to a separate profile cache;
  it does not download a model. Use a local, north-up, three-band Byte RGB
  GeoTIFF, PNG or JPEG. Unsupported rasters or invalid models retain Ink.
  Results that arrive late become available at the next anchor and do not
  replace a preview or pending click. At `0%`, no learned inference runs.
- **HED:** select HED and use **Download HED**.
- **MobileSAM / SAM:** select the mode, use **Verify Selected SAM Model**, then
  its download button if required. An existing local checkpoint is checked
  offline; only a missing file causes a check of the fixed download source.

Some learned bundles provide only contour support because no evaluated
text-avoidance setting is attached. Their status states this limitation. The
existing Korea model is retained: separate general-text training and QGIS
comparisons did not show improved saved routes. Human-reviewed accuracy and
formal Calibration/Holdout evaluation remain pending. Details are in the
[learning records](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/learning/README.md).

## Annotate a training crop

1. Under **Advanced**, choose **Annotate training crop**, then a prepared
   `512×512` Byte RGB PNG and a working folder.
2. Correct the Ink draft in `contours`. Draw every visible contour centreline
   within the area you review. Mark non-contour lines in `other_ink` and regions
   in `other_ink_regions`. Use `reviewed` polygons for inspected areas and
   `ignore` polygons for uncertain or hidden pixels. Do not invent a line across
   a missing section and label it as observed truth. Pixels outside `reviewed`
   or inside `ignore` are excluded from training loss.
3. Use **Save Layer Edits** on every annotation layer. For evaluation, explicitly
   select one open contour whose endpoints fit within the 320-pixel Live-Wire
   window.
4. Choose **Review / export labels**, supply a new JSON path and your reviewer
   identifier, and confirm all visible contours only if you have checked them.
   **No** exports an unapproved draft. Set standard/hard difficulty for the
   evaluation contour. One review is recorded as `single_reviewer_pilot`; it is
   not independent validation.
5. Attach the JSON to a new versioned data manifest using the repository's
   separate
   [learning CLI](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/learning/README.md).
   Training code and Colab notebooks are not in this ZIP. The 144 USGS pilot
   crops remain drafts awaiting human review.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Start Tracing is unavailable | Select a readable raster and an editable output line layer. On a small screen, scroll the panel to reach the controls. |
| Nothing appears in Trace Preview | Confirm that the selected raster is visible at the current extent and scale. Transparent and NoData pixels are background, with no line to follow. |
| WMS/WMTS differs from a local raster | Services without a fixed source pixel grid use Ink from a bounded viewport image. Native-grid learned guidance and Recovery are unavailable. Authentication, server responses and project settings need separate checks. |
| The preview follows text or a nearby line | Add checkpoints, mark avoidance rectangles, try a colour sample, or use a manual segment/Freehand. Always review an inferred gap curve. |
| `No module named 'cv2'` | Install OpenCV into the Python used by QGIS. Ink and Freehand do not need it. |
| HED is invalid or fails to load | Download the model again through the plugin. |
| SAM verification or download fails | Check network access and `requests` in QGIS Python. Use **SAM Status Report** for diagnostics. HED downloads do not depend on `requests`. |
| A learned model is inactive | Check the bundle, runtime and supported raster format. A status of **using Ink** means the ordinary Ink path is still available. |
| A completed line is missing from the file | Finishing the trace is separate from **Save Layer Edits**. Check the layer's pending edits. |
| DEM creation is refused | Save edits; check the numeric fields, finite coordinates, metre CRS, required height variation and QGIS Processing algorithms. |

For a bug report, include the OS, QGIS/Python/plugin versions, steps to reproduce
and expected versus actual behaviour. Use a small redistributable example;
do not attach sensitive or restricted maps.

## Privacy, citation and license

The plugin has no telemetry and does not upload maps, traced geometry or inference
results. Network access is used to read a WMS/WMTS map you select, download a model
you request, or check a SAM download source when its local checkpoint is missing.

**SAM Status Report** can include your working directory, QGIS/Python environment
values, model paths and system information, and is also copied to the clipboard.
Read it before sharing. Remove usernames, project or network paths, and tokens
or keys in environment values. Report security or data-loss problems through the
private contact in the
[security policy](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/SECURITY.md).

Use
[CITATION.cff](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/blob/main/CITATION.cff)
when citing the project. ArchaeoTrace is licensed under the GNU General Public
License v2.0; see the bundled [LICENSE](LICENSE). Optional models and dependencies
retain their own licenses.
