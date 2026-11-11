# 0.1.7 workflow and code review

[한국어](RELEASE_REVIEW_2026-11-12.md)

For the subsequent bilingual build, see the [language update](LANGUAGE_REVIEW_2026-11-12.md).

2026-11-12 · Unpublished experimental candidate

The map-reading, tracing, extension, elevation, DEM and saving workflows were
reviewed, and reproduced problems were fixed. Local validation, map regressions
and all 12 remote CI jobs passed for the **v3 artifact identified below**. That
artifact was judged suitable as an experimental `0.1.7` installation and upgrade
candidate. No official QGIS upload, tag or GitHub Release was created.

This is the record of that completed review. Its results do not automatically
apply to subsequent language or documentation changes; a later candidate needs
its own artifact identity and validation record.

The changes preserve editing intent and input values and improve access to the
controls. The existing Korea model remains in place, and learned guidance remains
off by default. This review makes no claim of improved general-text model accuracy
or reduced working time.

## Reproduced problems and fixes

| Workflow | Observed problem | Fix |
| --- | --- | --- |
| Shift/Ctrl manual click | A deep U-shaped case showed a straight preview but accepted a 259-point curve. | Click and close operations use the manual path and invalidate late Recovery/SAM results. |
| Extend an existing contour | Extending a 125 m contour and accepting the default elevation changed it to 0 m. | The existing finite elevation is used as the initial value, preserving its precision and range as well as deliberate edits. |
| Custom DEM field | With `id,height_m`, IDs 1 and 2 were selected automatically instead of heights 100 and 120 m. | Fields other than `elevation` require explicit selection; an explicit choice is retained for the same layer. |
| Invalid DEM coordinates | Some vertices from a line containing NaN XY coordinates entered the TIN and changed the terrain. | Contours and spot heights with nonfinite XY coordinates are rejected before generation. |
| Small screen and large font | A 700 px window was forced to a minimum height of about 1,179 px; English text overflowed horizontally on Linux. | Vertical scrolling, wrapping and button layout were improved. Default/Fusion 18 px fonts and model-installation screens were checked. |
| Readiness and instructions | Tracing could start without inputs, and the instructions could suggest that Enter saved the file. | A map and output line layer are required before starting. Korean and English instructions distinguish finishing a line from saving layer edits. |
| First layer selection in QGIS 4 | The visible selection was correct, but the internal output layer and Start button state were not updated. | Layer/model selection changes reconcile state, retain the same trace where appropriate, end it on a changed or removed layer, and block delayed processing after cleanup. |

The full checks also cover the earlier fixes for active-trace Ctrl+Z, initial
spot-height fields in the DEM dialog, and transparent/NoData raster reading.
The [first candidate record](RELEASE_READINESS_0.1.7.md) is retained separately.

## Reviewed artifact

| Item | Record |
| --- | --- |
| Installation candidate | `ai_vectorizer-0.1.7-review-v3-2026-11-12.zip` |
| Product implementation commit | `048605891e661d3c6b01a8876d17341f5cca62ec` |
| Validation commit, including tests | `9c85d8a98e3ce09ff3d9c24db41e1f5c917b0350` |
| SHA-256 | `7829bc9cba216b5105f4a504cae0e61cdfc1bcd630725e20b25cb3969d74b075` |
| Size / contents | 2,107,839 bytes / 52 entries, including 46 Python files |
| Metadata | `0.1.7`, `experimental=True`, QGIS `3.22–4.99` |
| Publication status | Unpublished; model weights and original maps are not included in the ZIP. |

The original frozen ZIP, SHA-256
`a3f4eebf07b51febf3a6f9eeddc2213764c9fda407617611f1811bfbae72ba28`, and intermediate
review artifacts were not overwritten. The v3 file above was the installation
candidate produced by this review; it does not include later language changes.

## Validation

| Check | Result |
| --- | --- |
| General Python suite | 1,246 passed, 202 skipped, 171 subtests passed. |
| Actual QGIS 3.44.8 | 220 tests across 15 modules; no failures, errors or skips. |
| Isolated installation | New ZIP import and seven scenarios through the actual plugin loader passed; temporary profiles were removed. |
| Existing map regressions | 23 cases × three methods = 69 comparisons with identical saved coordinates and input clicks; 138/138 Undo checks passed. |
| UI dimensions | Korean/English × Advanced collapsed/expanded, in a 1024×700 window and 360 px dock: no horizontal scrolling; Start, DEM and Help were accessible. |
| Fonts and optional-installation screens | Buttons and commands were accessible with Fusion 18 px, missing HED models/runtime, and MobileSAM/SAM installation states. |
| Deterministic packaging | Build/check passed and every file in the ZIP matched the source used for the review. |
| Remote CI | 12/12 passed: Python 3.11/3.12, Python 3.8 source contract, CPU learning contract, security/dependencies, identical Linux/Windows ZIPs, and QGIS 3.22.16/3.44.13/4.2.1. See [run 43058158450](https://github.com/lzpxilfe/AI-Vectorizer-for-Archaeology/actions/runs/43058158450). |

The [machine-readable validation summary](evidence/release-review-2026-11-12-validation.json)
records ZIP identity, test-source and raw-log hashes, per-case comparison hashes,
UI measurements and the exact CI run.

The map comparisons used the first frozen candidate and the final reviewed
product with the same inputs, clicks and CPU conditions. In
`qgis_grid_comparison.py`, the `usgs` key means the **existing Korea model**, and
`korea` means the new general-text model. The comparisons confirm path preservation
before and after the product changes within each method. They do not demonstrate
better model performance. Hidden curves and formal Calibration/Holdout results
were not scored.

Initial CI found Linux width and QGIS 4 selection-state problems; both were fixed
in the product. The final Qt6 failure was corrected in a new test by replacing an
integer mock mouse button with Qt's actual `NoButton` type. All CI jobs, including
QGIS 4, then passed. That test-only correction did not change the ZIP bytes.

## Scope and limits

- **Tracing:** preview/click agreement, manual and asynchronous paths,
  extensions, elevations, Undo, tool shutdown and layer removal.
- **Data:** NoData/alpha and WMS-like provider boundaries; DEM fields, XY, CRS,
  saved edits and output replacement; model hashes, offline validation and
  caches. Annotation workflows retain human review and explicit saving.
- **UI:** Korean/English, empty projects, small windows, enlarged fonts,
  advanced model-installation states, Start/DEM/Help access, and close/reopen.
  macOS QGIS restart and scroll-panel loading were checked. Narrow-window and
  lower-control access were measured with actual QGIS Qt widget tests.
- The reporter's complete WMS service and DEM project were not reproduced.
  Manual Windows desktop QA, varied HiDPI displays and accuracy against a
  human-reviewed Holdout remain unverified.
- Passing review and regression checks does not establish the absence of all
  defects or the accuracy of reconstructed terrain.
