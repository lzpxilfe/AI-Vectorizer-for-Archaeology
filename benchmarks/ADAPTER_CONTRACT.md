# Benchmark adapter contract

## Comparison boundary

Every method must return the same final representation before it is scored:

```text
image + trace prompt
  -> detector / segmentation backend
  -> cost map and tracing
  -> final ordered centerline
  -> canonical archaeotrace-centerline/1 artifact
```

Raw Canny/LSD edge maps, SAM region masks, probability-like scores, and final
traced lines are not interchangeable. The benchmark therefore scores only the
ordered centerline that would be saved by the product. A method that silently
falls back to another backend must record:

- `status: fallback`
- the requested and actual backend identifiers
- a non-empty `fallback_reason`

Failed predictions remain in the report with null metrics. They are not removed
from a method's completion rate or failure-adjusted aggregate.

The interactive A* policy now lives in the QGIS-independent
`ai_vectorizer/core/trace_kernel.py`, and `SmartTraceTool` delegates to it. The
worker calls strict `trace_path()`; the QGIS compatibility wrapper retains the
historical rounding/clamping and partial-path behavior. The legacy
`core/path_finder.py` and `Vectorizer.mask_to_line()` remain non-equivalent and
must not be used by benchmark adapters.

The versioned final-line profile is `smart-trace-v1-historical`:

```text
A* extension (start excluded, target included)
  -> centered five-point moving average
  -> three open-path Chaikin passes
  -> prepend the untouched segment start
```

This preserves current UI behavior, including movement of both extension
endpoints on paths longer than five pixels. Endpoint preservation would be a
product behavior change and needs a new profile plus side-by-side results.

## Worker protocol

Real adapters run in one fresh worker process per sample/method pair so native
memory and model state do not leak across comparisons. The current request and
result schemas are `archaeotrace-worker-request/1` and
`archaeotrace-worker-result/1`; the conceptual boundary is:

```text
load(artifact_manifest, device="cpu") -> backend_info
predict(rgb_uint8_hwc, trace_prompt) -> prediction
```

`prediction` must contain:

- a canonical ordered-centerline artifact;
- `status`, `requested_backend`, `actual_backend`, and `fallback_reason`;
- model and runtime versions, exact actual execution provider, structured
  `provider_kind` and `provider_device_type: cpu`, thread settings, and
  source-file, input, configuration, and output hashes;
- a canonical hash of the complete prompt; model-backed adapters must also bind
  the exact backend tensor representation and stable first-measured inference
  evidence to the artifact;
- model/backend-load time, per-image preparation time, and warmed prompt-stage
  time as distinct latency phases;
- one warm-up followed by at least three `perf_counter_ns()` and
  `process_time_ns()` samples;
- worker peak RSS in bytes.

The benchmark device is explicitly CPU. An adapter may not auto-select CUDA or
another accelerator. Canny/LSD workers also disable OpenCL, request OpenCV's
single-thread mode, and record the actual state. The number of repeated output
hashes must equal the number of measured timing repetitions; differing hashes mark the result
non-deterministic. Python, platform, CPU, and thread settings must be identical
across the run before timings are pooled.

For the split ONNX adapter, requested settings are insufficient evidence. Both
sessions must independently return the expected provider and session options
through ONNX Runtime readback, and OpenCV must report its effective thread and
OpenCL state. Each EfficientSAM invocation records the three IoU scores,
maximum-IoU selection index, and SHA-256 of the IoU tensor, selected logits,
thresholded mask, and accepted postprocessed mask. The first measured stable
record is embedded in the canonical centerline artifact; per-call decoder
timings remain in runtime evidence.

The three latency phases have deliberately different meanings:

1. `model_load_wall_ns` covers backend import and adapter/model construction,
   including ONNX session creation where applicable.
2. Per-image load/preparation evidence covers lossless image decode, colour
   conversion, and reusable image state such as the EfficientSAM encoder
   embedding. Lazy cache work performed by the first warm-up is recorded with
   the warm-up evidence rather than hidden in a measured repetition.
3. `wall_ns_samples` and `cpu_ns_samples` cover a warmed
   `predict(...)` call plus canonical centerline serialization. They exclude
   model load, per-image preparation, warm-up, artifact publication, and file
   rehashing. Reports expose the less ambiguous `prompt_*` aliases for these
   legacy fields.

These phases describe the observed adapter, not identical internal operations.
Canny/LSD currently recompute edge and cost maps inside every measured
`predict(...)`, so that work is prompt-stage latency. EfficientSAM reuses its
per-image encoder embedding, and its first-warm-up edge cache is outside the
measured prompt repetitions. Full first-prompt estimates therefore require the
image-load and warm-up evidence; `wall_ns_samples` alone must not be presented
as end-to-end image-to-vector latency.

Schema v1 uses exact CPU provider pairs: `onnxruntime` /
`CPUExecutionProvider`, `opencv` / `OpenCV CPU`, `pytorch` / `PyTorch CPU`,
`python` / `Python CPU`, or the synthetic fixture pair. This is a validation
contract, not runtime attestation; the worker must obtain provider identity from
the library itself.

Imported precomputed manifests remain self-attested. `benchmarks.generate`
copies worker-emitted evidence without editing it, binds it back to the exact
request, verifies that the artifact is the first measured output, and
atomically publishes only a complete validated dataset without replacing an
existing destination. Hand-edited records are still not proof that a backend
ran.

## Implemented EfficientSAM-Ti ONNX contract

The initial candidate is the official EfficientSAM repository at commit
[`d525f622e6f640acf5a0fc37c7ca1f243da5bde0`](https://github.com/yformer/EfficientSAM/tree/d525f622e6f640acf5a0fc37c7ca1f243da5bde0),
which declares [Apache-2.0](https://github.com/yformer/EfficientSAM/blob/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/LICENSE).
The repository contains official Ti ONNX binaries, but its
[`export_to_onnx.py`](https://github.com/yformer/EfficientSAM/blob/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/export_to_onnx.py)
does not provide a reproducible exporter. Accordingly, the benchmark pins the
published binary itself and never describes its export as reproducible.

Pinned Ti artifacts:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| encoder ONNX | 24,799,761 | `84ed466ffcc5c1f8d08409bc34a23bb364ab2c15e402cb12d4335a42be0e0951` |
| decoder ONNX | 16,565,728 | `a62f8fa5ea080447c0689418d69e58f1e83e0b7adf9c142e2bd9bcc8045c0b11` |
| combined ONNX (equivalence fixture only) | 41,365,520 | `143c3198a7b2a15f23c21cdb723432fb3fbcdbabbdad3483cf3babd8b95c1397` |

The canonical split-bundle specification fingerprint is
`f9d4b88041640ca39ca9b484629eb9476fabcd1a15f0cc0b71ab435e12602b8c`.
It hashes the bundle id/version, exact URLs, sizes and artifact hashes, source
repository/commit, and license identity. The combined artifact is documented
but is neither fetched nor required by the runtime bundle.

The adapter will follow the repository's
[`EfficientSAM_onnx_example.py`](https://github.com/yformer/EfficientSAM/blob/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/EfficientSAM_onnx_example.py)
and [`onnx_models.py`](https://github.com/yformer/EfficientSAM/blob/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/onnx_models.py):

- use `CPUExecutionProvider` and record the actual ONNX Runtime provider;
- fix and record one intra-op thread, one inter-op thread, sequential execution,
  and `ORT_ENABLE_ALL` graph optimization, then read back and compare all four
  settings independently for encoder and decoder;
- use 1024 × 1024 RGB crops for the first controlled comparison;
- pass RGB `uint8 HWC` as `float32 NCHW / 255` without duplicate ImageNet
  normalization outside the model;
- use source-pixel `(x, y)` prompts and reject more than six points, counting
  the always-positive start and end; explicit labels are limited to positive
  `1` and negative `0`, while bounding-box labels `2`/`3` are rejected;
- hash both the normalized semantic prompt and the exact float32 point/label
  tensors, including shape and the adapter's start/positive/end/negative order;
- validate all ONNX input/output names, ranks, and dtypes at load time;
- retain raw logits and predicted-IoU values, select the candidate with maximum
  predicted IoU, threshold the selected logit at zero, and retain per-run
  hashes through the first published measured artifact;
- call sigmoid output a non-calibrated guide score, not a probability;
- reuse the split encoder state across repeated prompts; use combined ONNX only
  as an equivalence fixture.

The decoder output contract requests only the stable `output_masks` and
`iou_predictions` names. The pinned binary also exposes an exporter-generated
third output named `onnx::Shape_1830`; the adapter validates its presence but
does not depend on its low-resolution logits. No model or ONNX Runtime package
is bundled.

### Model lifecycle and trace boundary

`python3 -m benchmarks model fetch --model-cache PATH` is the sole networked
operation. It uses fixed HTTPS URLs, identity encoding, exact content length,
a `size + 1` read ceiling, streaming SHA-256, same-directory `0600` temporary
files, `fsync`, and atomic no-replace publication. Cache objects live below
`objects/sha256/<prefix>/<full-digest>`. Offline inspection and reads reject
missing, corrupt, symlink, or non-regular cache components and pass verified
bytes—not mutable paths—to ONNX Runtime.

The selected boolean mask still is not a prediction artifact. It crosses the
shared `ai_vectorizer.core.sam_trace_kernel` boundary:

```text
selected logit >= 0
  -> first close + size/area guard
  -> second close + Canny-assisted cost map + skeleton
  -> skeleton endpoint snap (whole-mask fallback)
  -> strict shared A*
  -> moving average + three open Chaikin passes + raw start prepend
  -> canonical ordered centerline
```

`SmartTraceTool` delegates the same mask postprocessing, cost, snap, and strict
trace stages to this QGIS-independent module. The benchmark adapter applies the
same final-line smoothing profile in one call; the UI retains its historical
moving-average/canvas-Chaikin stage locations, avoiding double smoothing.

The EfficientSAM paper reports speed on its own hardware and natural-image
benchmarks; it does not establish accuracy for historical-map contours. Only
our CPU benchmark may justify a later default-model decision.
