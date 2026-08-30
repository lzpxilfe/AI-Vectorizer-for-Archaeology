# ArchaeoTrace contour benchmark

This directory contains a dependency-free evaluator plus optional isolated CPU
workers for the M1 model benchmark. The evaluator can score imported traces;
the worker path executes Canny/LSD, Ink Live-Wire, the pinned EfficientSAM-Ti
split ONNX adapter, or the conditional Ink-v2 recovery route. It calls the
same QGIS-independent trace kernels as `SmartTraceTool` and emits an
`archaeotrace-centerline/1` artifact.

That boundary matters: edge maps, region masks, and final ordered lines are not
interchangeable. Scoring begins only after every method crosses the shared
final-line boundary in [`ADAPTER_CONTRACT.md`](ADAPTER_CONTRACT.md).

## Evaluator smoke check

From the repository root:

```bash
python3 -m benchmarks validate benchmarks/data/synthetic-smoke/manifest.json
python3 -m benchmarks evaluate benchmarks/data/synthetic-smoke/manifest.json \
  --output work/benchmark-results
```

The evaluator writes an immutable run directory containing a strict JSON
report, a sample-level CSV, a method-summary CSV, and a commit file with all
three hashes. `benchmark_latest.json` is the single atomically replaced pointer
to the active run, so an interrupted write cannot mix report generations. It
needs only Python's standard library.

## Isolated Canny/LSD product-path smoke check

The workers need NumPy, OpenCV, and scikit-image. A disposable environment
keeps benchmark dependencies separate from QGIS's Python:

```bash
python3 -m venv work/benchmark-runtime
work/benchmark-runtime/bin/python -m pip install \
  'opencv-python-headless>=4.8,<4.12' 'scikit-image>=0.21.0'

work/benchmark-runtime/bin/python -m benchmarks generate \
  benchmarks/data/runtime-template/manifest.json \
  --output work/runtime-smoke \
  --python-executable work/benchmark-runtime/bin/python

python3 -m benchmarks validate work/runtime-smoke/manifest.json
python3 -m benchmarks evaluate work/runtime-smoke/manifest.json \
  --output work/runtime-smoke-report --require-eligible
```

`runtime-template` contains schema-valid placeholders only so inputs can be
validated before generation. Do not evaluate the template as a measured run.
`generate` never edits it: verified inputs go to a private staging directory,
one fresh process runs per sample/method pair, and input/configuration/source
hashes, CPU state, repetition counts, and artifact metadata are checked against
the request. The finished manifest is reloaded and validated, then the dataset
is published with an atomic no-replace rename.

The current measured method ids describe the real hybrid implementations:

- `canny-adaptive-v1`: adaptive dark-line threshold + Gaussian/Canny + closing;
- `lsd-adaptive-v1`: OpenCV LSD + adaptive dark-line threshold + closing.

Both fix `edge_weight=0.5`, require scikit-image skeletonization, use one CPU
thread, and disable OpenCL. Dependency/backend failures and explicit fallbacks
remain structured execution records instead of becoming silent successes.

## Ink Live-Wire and conditional recovery methods

The registered comparison ids are deliberately separate contracts:

- `ink-livewire-v1` calls the legacy `EdgeDetector.detect_edges()` Ink path;
- `ink-livewire-v2` calls `detect_ink_evidence()`, consumes
  `LineEvidence.centerline`, and passes the full evidence object to bounded
  Live-Wire. It is the Smart Recovery **OFF** control;
- `efficientsam-ti-onnx-v1` remains the pure segmentation comparison;
- `ink-v2-effsam-recovery-v1` first builds the identical Ink-v2 champion. It
  runs the EfficientSAM encoder/decoder only when the product recovery gate
  marks that route low quality, then uses `build_corridor_cost_map()` and
  `arbitrate_routes()` to accept or reject the challenger.

Both Ink adapters freeze the 320-pixel Live-Wire window, 6-pixel target snap,
five-point endpoint-restored smoothing, and an explicit edge strength. Worker
request schema `archaeotrace-worker-request/2` adds optional `previous_xy` so
directional history can be measured. The worker still accepts schema v1
requests without that field, and their canonical prompt hashes remain byte-for-
byte compatible. Prompt hash provenance follows the carrying protocol rather
than field population: request /2 always hashes `archaeotrace-trace-prompt/2`,
even when `previous_xy` is omitted, while request /1 keeps prompt /1.

The hybrid recovery adapter derives its model prompt through the same
QGIS-free helper used by the plugin: anchor and target are the only positives,
with up to four clipped 10-pixel perpendicular negatives. `previous_xy` is
Live-Wire direction only, and `positive_xy`/`negative_xy` do not alter the
recovery model tensor. Runtime evidence and the centerline artifact bind the
SHA-256 of that actual derived tensor.

Recovery thresholds are not implied defaults. Every request contains all
`RecoveryConfig` fields in `recovery_thresholds`, the provisional policy id
`smart-recovery-gate-v1-provisional`, and the configuration SHA-256. Runtime
and artifact evidence separately record gate metrics/trigger, champion and
optional challenger route hashes, selected route, fallback or rejection
reason, and conditional segmentation evidence. An untriggered record is
invalid if it contains decoder evidence. These thresholds remain provisional
until calibration is completed on the public dataset below.

## Isolated EfficientSAM-Ti ONNX smoke check

EfficientSAM is a measured candidate, not the product default. Its official
encoder and decoder are not bundled. The `model fetch` command is the only
benchmark operation allowed to use the network; it downloads the immutable
upstream URLs into SHA-256 content-addressed objects and refuses redirects,
wrong lengths, wrong hashes, symlinks, and replacement of an existing object.
`status`, `verify`, and `generate` are offline-only.

```bash
python3 -m venv work/efficientsam-runtime
work/efficientsam-runtime/bin/python -m pip install \
  'onnxruntime>=1.17.0' 'opencv-python-headless>=4.8,<4.12' \
  'scikit-image>=0.21.0'

work/efficientsam-runtime/bin/python -m benchmarks model status \
  --model-cache work/model-cache
work/efficientsam-runtime/bin/python -m benchmarks model fetch \
  --model-cache work/model-cache
work/efficientsam-runtime/bin/python -m benchmarks model verify \
  --model-cache work/model-cache

work/efficientsam-runtime/bin/python -m benchmarks generate \
  benchmarks/data/efficientsam-runtime-template/manifest.json \
  --output work/efficientsam-runtime-smoke \
  --python-executable work/efficientsam-runtime/bin/python \
  --model-cache work/model-cache
python3 -m benchmarks evaluate \
  work/efficientsam-runtime-smoke/manifest.json \
  --output work/efficientsam-runtime-report --require-eligible
```

Generation re-hashes both model objects before creating staging and every
fresh worker reads verified bytes again before constructing ONNX Runtime
sessions. Both sessions must report only `CPUExecutionProvider`; intra-op and
inter-op threads are fixed to one with sequential execution and explicit
`ORT_ENABLE_ALL` graph optimization. These are read back independently from
the constructed encoder and decoder sessions rather than copied from the
requested settings. OpenCV's effective thread count and OpenCL state are also
read back before SAM mask postprocessing is allowed. The encoder is run once during per-image
preparation and reused across the warm-up and measured decoder/trace
repetitions. The EfficientSAM edge map is filled lazily by the first warm-up
and then reused. Runtime evidence records the bundle fingerprint, both artifact
hashes, upstream commit, provider lists, source hashes, versions, phase timing,
repeated output hashes, and peak RSS without recording the cache path in the
reproducibility identity.

Every generated artifact also binds the complete semantic prompt and the exact
float32 EfficientSAM point/label tensor ordering with separate SHA-256 values.
For every warm-up and measured prediction, runtime evidence records the three
predicted IoUs, selected maximum-IoU index, decoder time, and hashes of the IoU
tensor, selected logits, raw binary mask, and accepted postprocessed mask. The
artifact embeds the stable evidence from the first measured prediction; timing
is kept outside the artifact so repeated output hashes test model/result
determinism rather than clock equality. Private worker request/result files are
removed before publication, so an absolute local model-cache path cannot leak
into a generated dataset.

Latency is reported in three phases. `model_load_wall_ns` covers backend and
model construction. Per-image evidence covers decode, colour conversion, and
reusable setup such as the EfficientSAM encoder. The legacy
`wall_ns_samples`/`cpu_ns_samples` fields—and their clearer `prompt_*` report
aliases—cover only a warmed `predict(...)` call plus canonical centerline JSON
serialization. They exclude model/image setup, warm-up, artifact publication,
and file rehashing, so they are not full image-to-vector latency.

The prompt boundary is shared, but the implementations do different work
inside it: Canny/LSD recompute their edge and cost maps for every measured
prompt, while EfficientSAM reuses both its encoder embedding and the edge cache
filled during its first warm-up. When the worker supplies image-load and
warm-up evidence, reports also provide explicitly named image-first-prompt and
cold-worker-first-prompt estimates. Compare these phases separately; the
legacy `estimated_dataset_pass_ns` is retained for compatibility and means a
warm-prompt-only pass, with `estimated_warm_prompt_pass_ns` as its unambiguous
alias.

`efficientsam-runtime-template` is a 1024×1024 outlined-ellipse connectivity
fixture with start/end positive points and one explicit negative guide point.
It tests the real model-to-mask-to-centerline path only. Its score is not
historical-map accuracy evidence and cannot make the run publication-eligible.

## Canonical centerline artifact

Coordinates use source-image pixels, with origin at the top left and `(x, y)`
meaning `(column, row)`. Integer coordinates are pixel centers. Lines are
clipped to the declared canvas, rounded half-up, and rasterized with inclusive
one-pixel Bresenham segments without antialiasing.

```json
{
  "schema_version": "archaeotrace-centerline/1",
  "coordinate_space": "pixel_xy",
  "image_size": {"width": 512, "height": 512},
  "paths": [
    {
      "id": "target-contour",
      "closed": false,
      "points": [[12, 80], [31, 79], [54, 83]]
    }
  ]
}
```

Reference paths remain ordered so the evaluator can distinguish an endpoint
omission from an internal break. A prediction may contain multiple paths when
the trace is fragmented.

## Benchmark manifest guarantees

Schema `archaeotrace-contour-benchmark/1` requires:

- a fixed canvas of at most 1024 × 1024 pixels and a lossless PNG/PNM image
  whose dimensions match it;
- SHA-256 checksums for every image, reference, and successful prediction;
- dataset, sample, and method source/license metadata;
- the strata `map_type`, `print_state`, and `scan_quality`;
- start/end and optional point prompts in source-pixel coordinates;
- prompt SHA-256 evidence for generated results, plus exact point/label tensor
  and first-measured segmentation evidence for the EfficientSAM adapter;
- CPU device/provider type, actual backend, fallback/error status, timing
  repetitions, and peak-RSS fields for every method result;
- a structured CPU `provider_kind` with an exact provider name (`opencv` →
  `OpenCV CPU`, `onnxruntime` → `CPUExecutionProvider`, and so on);
- one warm-up, at least three matched wall/CPU/output-hash repetitions, and an
  explicit deterministic-output result;
- one shared Python/platform/CPU/thread fingerprint for every timing record;
- the same method set for every sample.

Absolute paths, parent-directory traversal, symlink escapes, checksum changes,
dimension mismatches, duplicate JSON keys, non-finite numbers, control
characters, and oversized inputs fail validation. Centerline and manifest JSON
is hashed from the same bounded bytes that are parsed; artifacts are rehashed
again immediately before scoring.

## Metrics

The primary score is macro `F1@3px`; JSON also records F1 at 1, 2, 3, and 5
pixels. The evaluator reports exact centerline Dice, bidirectional exact
Euclidean mean and nearest-rank p95 distances, connected components, endpoint
and branch zones, unmatched predicted and reference branch zones,
ordered-path coverage, fragments, and internal breaks. Reporting both sides
distinguishes a spurious junction from a junction the tracer failed to recover.

Both-empty traces score 1 for overlap but have null distance. A method failure
has null metrics and reduces completion rate and failure-adjusted macro F1.
JSON output never contains NaN or Infinity.

Schema v1 limits tolerances to 8 pixels, ordered rasterized path samples to
100,000 per artifact, and the whole manifest to 256 sample-method
evaluations. PNG files are checked through CRCs, complete chunks, bounded IDAT
decompression, and scanline filters; PNM headers and complete sample payloads
are also validated.

`eligible=true` means a result is complete, uses the requested backend, and
repeats deterministically under the manifest's shared local timing environment.
Imported data remains manifest-attested. The isolated generator instead copies
evidence emitted by the worker that performed the prediction: provider,
package/thread state, source-file and input/configuration hashes, timing, RSS,
and repeated output hashes. It also requires the published artifact to equal
the first measured output and to bind the same input, prompt, configuration,
backend, smoothing profile, and—in the SAM path—selected logits/masks/IoUs in
its metadata.

Neither 9×9 smoke fixture is an accuracy benchmark. They prove only that
validation, execution, tracing, and reporting are connected. Reports keep
`publication_ranking_eligible=false` until a licensed, stratified historical-map
dataset and review protocol exist.

## Public 8-sheet / 48-crop plan

[`data/public-8x6-template/dataset-plan.json`](data/public-8x6-template/dataset-plan.json)
is an incrementally materialized, rights-safe plan. It reserves eight
independent source sheets and six ordered-reference crops per sheet, split at
sheet level into four calibration and four frozen holdout sheets. Each split
contains two
`usgs_htmc` and two `korea_rights_cleared` sources. The eight fixed failure
strata each occur three times per split (six times overall): clean dark curve,
thick/scale variation, faded/broken line, coloured line, text/number crossing,
dense parallel lines, stain/fold/bleed, and straight/grid distractors.

Validate the structural template offline:

```bash
python3 -c "from benchmarks.public_dataset import validate_public_dataset_plan as v; print(v('benchmarks/data/public-8x6-template/dataset-plan.json'))"
```

The companion
[`public-dataset-plan-v1.schema.json`](schemas/public-dataset-plan-v1.schema.json)
documents the JSON shape. `validate_public_dataset_plan()` additionally checks
the distribution and leakage constraints JSON Schema cannot express. It never
downloads anything. Calibration sheet `cal-01` is the first locally staged
source: the official USGS HTMC East Denver, Colorado 1890 GeoTIFF, its local
rights/provenance snapshot, six exact lossless source crops, prompt-v2 records,
and ordered-reference drafts are all hash-bound in the plan. The reference
drafts deliberately remain `unreviewed`/`pending`; the other seven source slots
are unresolved. The whole plan therefore still reports `materialized=false`
and `publication_ranking_eligible=false`.

With Pillow installed, verify that every staged PNG is pixel-for-pixel equal
to its declared GeoTIFF rectangle and that each draft reference runs from the
recorded prompt start to end:

```bash
python3 -m benchmarks.public_assets benchmarks/data/public-8x6-template/dataset-plan.json
```

This pass is also offline. It is separate from the dependency-free structural
validator so a clean QGIS installation does not need a TIFF codec merely to
read the plan.

Before a materialized plan is accepted, every source must have an open/public-
domain rights status, publisher/date or sheet id, source and rights-statement
URLs, a locally captured rights-text snapshot and hash, a source-raster hash,
and a structured immutable `provenance_id` whose authority/namespace/value is
globally unique across both splits. Every crop must have source
`x,y,width,height`, the matching explicit `source_tile_origin_xy=[x,y]`, hashed
local image and exactly one open, in-bounds ordered reference, a bounded prompt
explicitly marked `schema_version=archaeotrace-trace-prompt/2`, and completed
approval from named, distinct reviewer and adjudicator identities.
Ink v2/Recovery worker evidence binds that origin together with the image hash
so an identical PNG cannot silently move to a different normalization tile.
Run both the asset verifier above and the structural validator with
`require_materialized=True` for that gate. Calibration can tune the provisional
recovery thresholds; the frozen holdout cannot be used for tuning.

Do not fill a source slot with rights-unclear material or the restricted-
distribution Library of Congress L851 collection. A public URL by itself is
not redistribution permission; retain the item-level rights evidence and its
local snapshot.

## Adding real cases

1. Use lossless, legally redistributable crops. Keep source URL and license per
   sample.
2. Annotate the target contour as an ordered final line, not a thick region
   mask.
3. Record prompts before running any method so every adapter receives the same
   user intent. `start_xy` and `end_xy` are always positive SAM points;
   `positive_xy` contains only additional guides and must not repeat them.
4. Split cases by map type, print condition, and scan quality before selecting
   models.
5. Generate predictions with `python3 -m benchmarks generate`; do not paste
   timing/provider values into a manifest by hand.
6. Add all file hashes to the manifest and run `validate` before evaluation.

PNG crops under `benchmarks/data/` are explicitly allowed by `.gitignore`; the
repository-wide PNG ignore remains in place for unrelated local images.
