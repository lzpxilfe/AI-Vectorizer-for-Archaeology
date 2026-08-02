# ArchaeoTrace contour benchmark

This directory contains a dependency-free evaluator plus optional isolated CPU
workers for the M1 model benchmark. The evaluator can score imported traces;
the worker path executes the product's Canny/LSD detector or the pinned
EfficientSAM-Ti split ONNX adapter, calls the same QGIS-independent trace
kernels as `SmartTraceTool`, and emits an `archaeotrace-centerline/1` artifact.

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
  'opencv-python-headless>=4.8.0' 'scikit-image>=0.21.0'

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
  'onnxruntime>=1.17.0' 'opencv-python-headless>=4.8.0' \
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
and branch zones, ordered-path coverage, fragments, and internal breaks.

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
