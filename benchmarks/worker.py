"""Isolated, CPU-only contour benchmark worker.

The module intentionally imports only the Python standard library at import
time.  NumPy, OpenCV, :class:`EdgeDetector`, and the shared trace kernel are
loaded inside the worker process.  This keeps ordinary benchmark validation
usable on machines which do not have the optional detector dependencies.

The worker owns the evidence written into a manifest prediction record:
backend selection, provider identity, repeated output hashes, timings, and
peak RSS are all observed in the process which performed the prediction.  A
parent process should insert ``result["prediction"]`` into the matching sample
and method in a benchmark manifest without editing that evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import struct
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from . import evidence as benchmark_evidence
from . import geometry as benchmark_geometry
from . import manifest as benchmark_manifest
from .geometry import ARTIFACT_SCHEMA_VERSION, COORDINATE_SPACE, MAX_POINTS
from .manifest import (
    MAX_CANVAS_PIXELS,
    MAX_TIMING_REPETITIONS,
    image_dimensions,
)


WORKER_REQUEST_SCHEMA_VERSION_V1 = "archaeotrace-worker-request/1"
WORKER_REQUEST_SCHEMA_VERSION = "archaeotrace-worker-request/2"
WORKER_REQUEST_SCHEMA_VERSIONS = frozenset(
    {WORKER_REQUEST_SCHEMA_VERSION_V1, WORKER_REQUEST_SCHEMA_VERSION}
)
WORKER_RESULT_SCHEMA_VERSION = "archaeotrace-worker-result/1"
WORKER_ADAPTER_VERSION = "archaeotrace-worker/2"
OPENCV_ADAPTER_VERSION = "opencv-edge-worker/1"
EFFICIENTSAM_ADAPTER_VERSION = "efficientsam-ti-onnx-worker/1"
INK_LIVEWIRE_ADAPTER_VERSION = "ink-livewire-worker/2"
RECOVERY_ADAPTER_VERSION = "ink-v2-effsam-recovery-worker/1"
PRODUCT_SMOOTHING_PROFILE = "smart-trace-v1-historical"
INK_LIVEWIRE_SMOOTHING_PROFILE = "smart-trace-livewire-v1"
RECOVERY_EVIDENCE_SCHEMA_VERSION = "archaeotrace-ink-effsam-recovery/1"
PRODUCT_CACHE_MAX_DIMENSION = 1_000
EFFICIENTSAM_INPUT_DIMENSION = 1_024
METHOD_EDGE_BACKENDS = {
    "canny-adaptive-v1": "canny",
    "lsd-adaptive-v1": "lsd",
}
INK_LIVEWIRE_V1_BACKEND = "ink-livewire-v1"
INK_LIVEWIRE_V2_BACKEND = "ink-livewire-v2"
METHOD_INK_BACKENDS = frozenset(
    {INK_LIVEWIRE_V1_BACKEND, INK_LIVEWIRE_V2_BACKEND}
)
EFFICIENTSAM_BACKEND = "efficientsam-ti-onnx-v1"
INK_V2_EFFICIENTSAM_RECOVERY_BACKEND = "ink-v2-effsam-recovery-v1"
METHOD_RECOVERY_BACKENDS = frozenset({INK_V2_EFFICIENTSAM_RECOVERY_BACKEND})
METHOD_SOURCE_GRID_BACKENDS = frozenset(
    {INK_LIVEWIRE_V2_BACKEND, *METHOD_RECOVERY_BACKENDS}
)
METHOD_SAM_BACKENDS = frozenset(
    {EFFICIENTSAM_BACKEND, *METHOD_RECOVERY_BACKENDS}
)
EFFICIENTSAM_ORT_SESSION_OPTIONS = {
    "intra_op_num_threads": 1,
    "inter_op_num_threads": 1,
    "execution_mode": "ORT_SEQUENTIAL",
    "graph_optimization_level": "ORT_ENABLE_ALL",
}
EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION = (
    "archaeotrace-efficientsam-prediction-evidence/1"
)
LATENCY_SCOPE = "warmed_predict_plus_canonical_artifact_v1"
SUPPORTED_BACKENDS = frozenset(
    (*METHOD_EDGE_BACKENDS, *METHOD_INK_BACKENDS, *METHOD_SAM_BACKENDS)
)
MAX_REQUEST_BYTES = 1024 * 1024
MAX_ERROR_LENGTH = 2_000
SHA256_LENGTH = 64
MAX_SOURCE_TILE_ORIGIN = (1 << 63) - 1
CPU_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class WorkerError(RuntimeError):
    """Base class for worker protocol and execution failures."""


class WorkerRequestError(WorkerError):
    """Raised when a worker request is invalid or unsafe."""


class WorkerDependencyError(WorkerError):
    """Raised when a lazy runtime dependency is unavailable."""


class BackendUnavailableError(WorkerError):
    """Raised when the requested detector cannot run in this OpenCV build."""


class PredictionError(WorkerError):
    """Raised when a detector/kernel prediction cannot become a centerline."""


@dataclass(frozen=True)
class TracePrompt:
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    positive_xy: tuple[tuple[float, float], ...] = ()
    negative_xy: tuple[tuple[float, float], ...] = ()
    previous_xy: tuple[float, float] | None = None
    schema_version: str = benchmark_evidence.PROMPT_EVIDENCE_SCHEMA_VERSION


@dataclass(frozen=True)
class WorkerRequest:
    request_id: str
    requested_backend: str
    fallback_backend: str | None
    device: str
    root: Path
    image_path: Path
    image_sha256: str
    width: int
    height: int
    artifact_path: Path
    artifact_manifest_path: str
    prompt: TracePrompt
    configuration: dict[str, Any]
    warmup_runs: int
    measurement_runs: int
    threads: int
    model_cache: Path | None = None
    schema_version: str = WORKER_REQUEST_SCHEMA_VERSION
    source_tile_origin_xy: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class BackendInfo:
    """Runtime identity reported by a loaded detector pipeline."""

    actual_backend: str
    provider_kind: str
    actual_provider: str
    provider_device_type: str
    adapter_version: str
    package_versions: dict[str, str | None]
    thread_settings: dict[str, Any]
    provider_verified: bool = True
    source_files_sha256: dict[str, str] = field(default_factory=dict)
    model_artifacts_sha256: dict[str, str] = field(default_factory=dict)
    model_bundle_id: str | None = None
    model_bundle_sha256: str | None = None
    runtime_details: dict[str, Any] = field(default_factory=dict)


class LoadedPipeline(Protocol):
    """Minimal injected boundary used by real and fake worker pipelines."""

    info: BackendInfo

    def load_image(self, path: Path, width: int, height: int) -> Any:
        """Decode a validated lossless benchmark image."""

    def predict(
        self,
        image: Any,
        prompt: TracePrompt,
        configuration: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]:
        """Return ordered ``(x, y)`` points from the shared trace kernel."""


PipelineLoader = Callable[[str, int], LoadedPipeline]


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkerRequestError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise WorkerRequestError(f"Non-standard JSON number is not allowed: {value}.")


def _safe_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {text}"[:MAX_ERROR_LENGTH]


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_sha256(points: Sequence[Sequence[float]]) -> str:
    """Hash a route independently of artifact metadata and timings."""

    normalized = [[float(point[0]), float(point[1])] for point in points]
    raw = json.dumps(
        normalized,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerRequestError(f"{label} must be a non-empty identifier.")
    if not all(character.isascii() and (character.isalnum() or character in "._-") for character in value):
        raise WorkerRequestError(f"{label} contains unsupported characters.")
    if not value[0].isalnum() or value.lower() != value:
        raise WorkerRequestError(f"{label} must start with a lowercase letter or digit.")
    return value


def _backend_family(backend: str) -> str:
    if backend in METHOD_EDGE_BACKENDS:
        return "edge"
    if backend in METHOD_INK_BACKENDS:
        return "ink"
    if backend in METHOD_SAM_BACKENDS:
        return "sam"
    raise WorkerRequestError(f"Unsupported backend: {backend!r}.")


def _provider_contract(backend: str) -> tuple[str, str]:
    if _backend_family(backend) == "sam":
        return "onnxruntime", "CPUExecutionProvider"
    if _backend_family(backend) == "ink":
        return "python", "Python CPU"
    return "opencv", "OpenCV CPU"


def _model_cache_path(value: Any, *, required: bool) -> Path | None:
    if value is None:
        if required:
            raise WorkerRequestError(
                "EfficientSAM requests require an absolute model_cache path."
            )
        return None
    if not isinstance(value, str) or not value:
        raise WorkerRequestError("model_cache must be a non-empty absolute path.")
    path = Path(value)
    if not path.is_absolute():
        raise WorkerRequestError("model_cache must be an absolute path.")
    return Path(os.path.abspath(os.fspath(path)))


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise WorkerRequestError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _point(value: Any, width: int, height: int, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise WorkerRequestError(f"{label} must be an [x, y] pair.")
    coordinates: list[float] = []
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise WorkerRequestError(f"{label}[{index}] must be numeric.")
        number = float(coordinate)
        if not math.isfinite(number):
            raise WorkerRequestError(f"{label}[{index}] must be finite.")
        coordinates.append(number)
    if not (0 <= coordinates[0] <= width - 1 and 0 <= coordinates[1] <= height - 1):
        raise WorkerRequestError(f"{label} lies outside the image.")
    return coordinates[0], coordinates[1]


def _source_tile_origin(value: Any) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise WorkerRequestError("source_tile_origin_xy must be an [x, y] pair.")
    return tuple(
        _integer(
            coordinate,
            f"source_tile_origin_xy[{index}]",
            0,
            MAX_SOURCE_TILE_ORIGIN,
        )
        for index, coordinate in enumerate(value)
    )


def _safe_input_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise WorkerRequestError(f"{label} must be a non-empty relative path.")
    raw = Path(value)
    if raw.is_absolute():
        raise WorkerRequestError(f"{label} must be relative to the request file.")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkerRequestError(f"{label} escapes the request directory.") from exc
    if not resolved.is_file():
        raise WorkerRequestError(f"{label} does not exist: {resolved}")
    return resolved


def _safe_output_path(root: Path, value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, str) or not value:
        raise WorkerRequestError(f"{label} must be a non-empty relative path.")
    raw = Path(value)
    if raw.is_absolute():
        raise WorkerRequestError(f"{label} must be relative to the request file.")
    if (
        not raw.parts
        or raw.parts[0] != "predictions"
        or raw.suffix.lower() != ".json"
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise WorkerRequestError(
            f"{label} must be a normalized JSON path below predictions/."
        )
    candidate = root / raw
    current = root
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            raise WorkerRequestError(f"{label} must not contain symbolic links.")
    if candidate.exists():
        raise WorkerRequestError(f"{label} already exists and will not be overwritten.")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkerRequestError(f"{label} escapes the request directory.") from exc
    return resolved, raw.as_posix()


def _validate_json_value(value: Any, label: str, depth: int = 0) -> None:
    if depth > 32:
        raise WorkerRequestError(f"{label} nesting is too deep.")
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkerRequestError(f"{label} must contain finite numbers.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkerRequestError(f"{label} keys must be strings.")
            _validate_json_value(item, f"{label}.{key}", depth + 1)
        return
    raise WorkerRequestError(f"{label} contains an unsupported JSON value.")


def load_worker_request(path: str | Path) -> WorkerRequest:
    """Load a bounded, checksummed worker request rooted at its directory."""

    request_path = Path(path).resolve()
    try:
        with request_path.open("rb") as handle:
            raw = handle.read(MAX_REQUEST_BYTES + 1)
    except OSError as exc:
        raise WorkerRequestError(f"Could not read worker request: {exc}") from exc
    if len(raw) > MAX_REQUEST_BYTES:
        raise WorkerRequestError(f"Worker request exceeds {MAX_REQUEST_BYTES} bytes.")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise WorkerRequestError(f"Invalid worker request JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise WorkerRequestError("Worker request root must be an object.")
    _validate_json_value(payload, "request")
    request_schema_version = payload.get("schema_version")
    if request_schema_version not in WORKER_REQUEST_SCHEMA_VERSIONS:
        raise WorkerRequestError("Unsupported worker request schema.")

    requested_backend = _identifier(payload.get("requested_backend"), "requested_backend")
    if requested_backend not in SUPPORTED_BACKENDS:
        raise WorkerRequestError(f"requested_backend must be one of {sorted(SUPPORTED_BACKENDS)}.")
    fallback_value = payload.get("fallback_backend")
    fallback_backend = None
    if fallback_value is not None:
        if requested_backend in METHOD_RECOVERY_BACKENDS:
            raise WorkerRequestError(
                "Smart Recovery does not permit an external fallback_backend; "
                "model or inference failures must keep its Ink champion."
            )
        fallback_backend = _identifier(fallback_value, "fallback_backend")
        if fallback_backend not in SUPPORTED_BACKENDS or fallback_backend == requested_backend:
            raise WorkerRequestError("fallback_backend must be a distinct supported backend.")
        if _backend_family(fallback_backend) != _backend_family(requested_backend):
            raise WorkerRequestError(
                "fallback_backend must use the same detector family as requested_backend."
            )
    source_origin_value = payload.get("source_tile_origin_xy")
    if request_schema_version == WORKER_REQUEST_SCHEMA_VERSION_V1:
        if source_origin_value is not None:
            raise WorkerRequestError(
                "worker request schema v1 does not support source_tile_origin_xy."
            )
        source_tile_origin_xy = (0, 0)
    elif requested_backend in METHOD_SOURCE_GRID_BACKENDS:
        if source_origin_value is None:
            raise WorkerRequestError(
                "Ink v2 and Smart Recovery requests require "
                "source_tile_origin_xy."
            )
        source_tile_origin_xy = _source_tile_origin(source_origin_value)
    else:
        if source_origin_value is not None:
            raise WorkerRequestError(
                "source_tile_origin_xy is only valid for Ink v2 and "
                "Smart Recovery requests."
            )
        source_tile_origin_xy = (0, 0)
    device = payload.get("device")
    if device != "cpu":
        raise WorkerRequestError("Worker requests must set device='cpu'.")

    image = payload.get("image")
    if not isinstance(image, dict):
        raise WorkerRequestError("image must be an object.")
    maximum_dimension = (
        EFFICIENTSAM_INPUT_DIMENSION
        if requested_backend == EFFICIENTSAM_BACKEND
        else PRODUCT_CACHE_MAX_DIMENSION
    )
    width = _integer(image.get("width"), "image.width", 1, maximum_dimension)
    height = _integer(image.get("height"), "image.height", 1, maximum_dimension)
    if width * height > MAX_CANVAS_PIXELS:
        raise WorkerRequestError(f"image exceeds {MAX_CANVAS_PIXELS} pixels.")
    image_digest = image.get("sha256")
    if not _is_sha256(image_digest):
        raise WorkerRequestError("image.sha256 must be a lowercase SHA-256 digest.")

    root = request_path.parent.resolve()
    image_path = _safe_input_path(root, image.get("path"), "image.path")
    if image_dimensions(image_path) != (width, height):
        raise WorkerRequestError("image dimensions do not match the request.")
    actual_digest = _sha256_file(image_path)
    if actual_digest != image_digest:
        raise WorkerRequestError(
            f"image checksum mismatch: expected {image_digest}, got {actual_digest}."
        )
    artifact_path, artifact_manifest_path = _safe_output_path(
        root,
        payload.get("artifact"),
        "artifact",
    )
    if artifact_path in {request_path, image_path}:
        raise WorkerRequestError("artifact must not overwrite the request or input image.")
    prompt_payload = payload.get("prompt")
    if not isinstance(prompt_payload, dict):
        raise WorkerRequestError("prompt must be an object.")
    positive_values = prompt_payload.get("positive_xy", [])
    negative_values = prompt_payload.get("negative_xy", [])
    if not isinstance(positive_values, list) or not isinstance(negative_values, list):
        raise WorkerRequestError("positive_xy and negative_xy must be arrays.")
    guide_count = len(positive_values) + len(negative_values)
    if guide_count > 6:
        raise WorkerRequestError("A trace prompt may contain at most six guide points.")
    if _backend_family(requested_backend) == "sam" and guide_count + 2 > 6:
        raise WorkerRequestError(
            "EfficientSAM allows at most four guide points in addition to start/end."
        )
    previous_value = prompt_payload.get("previous_xy")
    if (
        request_schema_version == WORKER_REQUEST_SCHEMA_VERSION_V1
        and previous_value is not None
    ):
        raise WorkerRequestError(
            "worker request schema v1 does not support prompt.previous_xy."
        )
    prompt = TracePrompt(
        start_xy=_point(prompt_payload.get("start_xy"), width, height, "prompt.start_xy"),
        end_xy=_point(prompt_payload.get("end_xy"), width, height, "prompt.end_xy"),
        positive_xy=tuple(
            _point(value, width, height, f"prompt.positive_xy[{index}]")
            for index, value in enumerate(positive_values)
        ),
        negative_xy=tuple(
            _point(value, width, height, f"prompt.negative_xy[{index}]")
            for index, value in enumerate(negative_values)
        ),
        previous_xy=(
            _point(previous_value, width, height, "prompt.previous_xy")
            if previous_value is not None
            else None
        ),
        schema_version=(
            benchmark_evidence.PROMPT_EVIDENCE_SCHEMA_VERSION_V1
            if request_schema_version == WORKER_REQUEST_SCHEMA_VERSION_V1
            else benchmark_evidence.PROMPT_EVIDENCE_SCHEMA_VERSION
        ),
    )
    if _backend_family(requested_backend) == "sam":
        sam_points = (
            prompt.start_xy,
            *prompt.positive_xy,
            prompt.end_xy,
            *prompt.negative_xy,
        )
        if len(set(sam_points)) != len(sam_points):
            raise WorkerRequestError(
                "EfficientSAM guide points must not duplicate start, end, or each other."
            )
    configuration = payload.get("configuration", {})
    if not isinstance(configuration, dict):
        raise WorkerRequestError("configuration must be an object.")
    _validate_backend_configuration(requested_backend, configuration)
    model_cache = _model_cache_path(
        payload.get("model_cache"),
        required=_backend_family(requested_backend) == "sam",
    )
    threads = _integer(payload.get("threads", 1), "threads", 1, 1)

    return WorkerRequest(
        request_id=_identifier(payload.get("request_id"), "request_id"),
        requested_backend=requested_backend,
        fallback_backend=fallback_backend,
        device="cpu",
        root=root,
        image_path=image_path,
        image_sha256=image_digest,
        width=width,
        height=height,
        artifact_path=artifact_path,
        artifact_manifest_path=artifact_manifest_path,
        prompt=prompt,
        configuration=dict(configuration),
        warmup_runs=_integer(payload.get("warmup_runs", 1), "warmup_runs", 1, 100),
        measurement_runs=_integer(
            payload.get("measurement_runs", 3),
            "measurement_runs",
            3,
            MAX_TIMING_REPETITIONS,
        ),
        threads=threads,
        model_cache=model_cache,
        schema_version=request_schema_version,
        source_tile_origin_xy=source_tile_origin_xy,
    )


def _configuration_sha256(configuration: Mapping[str, Any]) -> str:
    raw = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(raw)


def _edge_weight(configuration: Mapping[str, Any]) -> float:
    value = configuration.get("edge_weight", 0.5)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerRequestError("configuration.edge_weight must be numeric.")
    edge_weight = float(value)
    if not math.isfinite(edge_weight) or edge_weight != 0.5:
        raise WorkerRequestError(
            "M1.1 Canny/LSD comparisons require configuration.edge_weight=0.5."
        )
    return edge_weight


def _ink_livewire_contract(configuration: Mapping[str, Any]) -> float:
    """Validate the frozen product-like Live-Wire comparison contract."""

    value = configuration.get("edge_weight")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerRequestError(
            "Ink Live-Wire requires an explicit numeric configuration.edge_weight."
        )
    strength = float(value)
    if not math.isfinite(strength) or not 0.0 < strength <= 1.0:
        raise WorkerRequestError(
            "Ink Live-Wire edge_weight must be finite and in (0, 1]."
        )
    expected = {
        "livewire_window_px": 320,
        "target_snap_radius_px": 6,
        "smoothing_window_px": 5,
        "smoothing_profile": INK_LIVEWIRE_SMOOTHING_PROFILE,
    }
    for key, expected_value in expected.items():
        if configuration.get(key) != expected_value:
            raise WorkerRequestError(
                f"Ink Live-Wire requires configuration.{key}={expected_value!r}."
            )
    return strength


def _recovery_contract(configuration: Mapping[str, Any]) -> Any:
    """Return the explicit, hashed provisional product recovery policy.

    Calibration runs may change threshold values, but they cannot silently
    change them: every field is serialized in ``recovery_thresholds`` and its
    canonical SHA-256 must be supplied beside the provisional policy id.
    """

    try:
        from ai_vectorizer.core.smart_recovery import (
            RECOVERY_POLICY_ID,
            RecoveryConfig,
        )
    except Exception as exc:
        raise WorkerDependencyError(
            "the product Smart Recovery policy is unavailable"
        ) from exc

    if configuration.get("recovery_policy_id") != RECOVERY_POLICY_ID:
        raise WorkerRequestError(
            f"Recovery requires recovery_policy_id={RECOVERY_POLICY_ID!r}."
        )
    if configuration.get("recovery_provisional") is not True:
        raise WorkerRequestError(
            "Recovery thresholds must be marked recovery_provisional=true."
        )
    raw_thresholds = configuration.get("recovery_thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise WorkerRequestError(
            "Recovery requires an explicit configuration.recovery_thresholds object."
        )
    expected_keys = set(asdict(RecoveryConfig()))
    if set(raw_thresholds) != expected_keys:
        raise WorkerRequestError(
            "configuration.recovery_thresholds must contain every RecoveryConfig "
            "field exactly once."
        )
    try:
        config = RecoveryConfig(**dict(raw_thresholds)).validate()
    except (TypeError, ValueError) as exc:
        raise WorkerRequestError(
            f"configuration.recovery_thresholds is invalid: {exc}"
        ) from exc
    if config.policy_id != RECOVERY_POLICY_ID:
        raise WorkerRequestError(
            "configuration.recovery_thresholds.policy_id must match "
            "recovery_policy_id."
        )
    supplied_sha256 = configuration.get("recovery_configuration_sha256")
    if supplied_sha256 != config.sha256:
        raise WorkerRequestError(
            "configuration.recovery_configuration_sha256 does not match the "
            "canonical recovery thresholds."
        )
    return config


def _canny_thresholds(configuration: Mapping[str, Any]) -> tuple[int, int]:
    low = configuration.get("canny_low_threshold", 30)
    high = configuration.get("canny_high_threshold", 100)
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, int)
        or not isinstance(high, int)
        or not 0 <= low < high <= 255
    ):
        raise WorkerRequestError(
            "Canny thresholds must be integers satisfying 0 <= low < high <= 255."
        )
    return low, high


def _smoothing_profile(configuration: Mapping[str, Any]) -> str:
    value = configuration.get("smoothing_profile", PRODUCT_SMOOTHING_PROFILE)
    if value != PRODUCT_SMOOTHING_PROFILE:
        raise WorkerRequestError(
            "M1.1 workers require smoothing_profile="
            f"{PRODUCT_SMOOTHING_PROFILE!r}."
        )
    return PRODUCT_SMOOTHING_PROFILE


def _sam_model_contract(
    configuration: Mapping[str, Any],
    *,
    require_canny: bool = True,
) -> tuple[str, str]:
    try:
        from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT
        from ai_vectorizer.core.model_store import bundle_fingerprint
    except Exception as exc:
        raise WorkerDependencyError(
            "the pinned EfficientSAM model specification is unavailable"
        ) from exc

    bundle_id = str(EFFICIENTSAM_TI_SPLIT.id)
    fingerprint = bundle_fingerprint(EFFICIENTSAM_TI_SPLIT)
    if configuration.get("model_bundle_id") != bundle_id:
        raise WorkerRequestError(
            f"EfficientSAM requires configuration.model_bundle_id={bundle_id!r}."
        )
    if configuration.get("model_bundle_sha256") != fingerprint:
        raise WorkerRequestError(
            "EfficientSAM configuration.model_bundle_sha256 does not match the "
            "pinned split-model specification."
        )
    threshold = configuration.get("mask_logit_threshold", 0.0)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise WorkerRequestError("configuration.mask_logit_threshold must be numeric.")
    if not math.isfinite(float(threshold)) or float(threshold) != 0.0:
        raise WorkerRequestError(
            "EfficientSAM requires configuration.mask_logit_threshold=0.0."
        )
    if require_canny:
        _canny_thresholds(configuration)
        _smoothing_profile(configuration)
    return bundle_id, fingerprint


def _validate_backend_configuration(
    backend: str,
    configuration: Mapping[str, Any],
) -> None:
    if backend == EFFICIENTSAM_BACKEND:
        _sam_model_contract(configuration, require_canny=True)
        return
    if backend in METHOD_RECOVERY_BACKENDS:
        _sam_model_contract(configuration, require_canny=False)
        _ink_livewire_contract(configuration)
        _recovery_contract(configuration)
        return
    if backend in METHOD_INK_BACKENDS:
        _ink_livewire_contract(configuration)
        return
    _edge_weight(configuration)
    _canny_thresholds(configuration)
    _smoothing_profile(configuration)


_SAM_PREDICTION_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "selected_mask_index",
        "iou_predictions",
        "iou_predictions_sha256",
        "selected_logits_sha256",
        "selected_binary_mask_sha256",
        "accepted_mask_sha256",
        "decoder_wall_ns",
    }
)


def _validated_sam_prediction_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SAM_PREDICTION_EVIDENCE_KEYS:
        raise PredictionError(
            "EfficientSAM prediction evidence has an unsupported structure"
        )
    if value.get("schema_version") != EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION:
        raise PredictionError(
            "EfficientSAM prediction evidence has an unsupported schema"
        )
    selected_index = value.get("selected_mask_index")
    if (
        isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not 0 <= selected_index < 3
    ):
        raise PredictionError(
            "EfficientSAM prediction evidence has an invalid selected mask index"
        )
    raw_ious = value.get("iou_predictions")
    if not isinstance(raw_ious, (list, tuple)) or len(raw_ious) != 3:
        raise PredictionError(
            "EfficientSAM prediction evidence must contain three IoU predictions"
        )
    ious: list[float] = []
    for raw_iou in raw_ious:
        if isinstance(raw_iou, bool) or not isinstance(raw_iou, (int, float)):
            raise PredictionError(
                "EfficientSAM prediction IoU evidence must be numeric"
            )
        iou = float(raw_iou)
        if not math.isfinite(iou):
            raise PredictionError(
                "EfficientSAM prediction IoU evidence must be finite"
            )
        try:
            iou = struct.unpack("<f", struct.pack("<f", iou))[0]
        except (OverflowError, struct.error) as exc:
            raise PredictionError(
                "EfficientSAM prediction IoUs cannot be represented as float32"
            ) from exc
        if not math.isfinite(iou):
            raise PredictionError(
                "EfficientSAM prediction IoUs must remain finite as float32"
            )
        ious.append(iou)
    expected_selected_index = max(range(len(ious)), key=ious.__getitem__)
    if selected_index != expected_selected_index:
        raise PredictionError(
            "EfficientSAM selected mask index disagrees with maximum predicted IoU"
        )
    for key in (
        "iou_predictions_sha256",
        "selected_logits_sha256",
        "selected_binary_mask_sha256",
        "accepted_mask_sha256",
    ):
        if not _is_sha256(value.get(key)):
            raise PredictionError(f"EfficientSAM prediction evidence {key} is invalid")
    try:
        expected_iou_sha256 = _sha256_bytes(struct.pack("<3f", *ious))
    except (OverflowError, struct.error) as exc:
        raise PredictionError(
            "EfficientSAM prediction IoUs cannot be represented as float32"
        ) from exc
    if value["iou_predictions_sha256"] != expected_iou_sha256:
        raise PredictionError(
            "EfficientSAM IoU tensor hash disagrees with its float32 values"
        )
    decoder_wall_ns = value.get("decoder_wall_ns")
    if (
        isinstance(decoder_wall_ns, bool)
        or not isinstance(decoder_wall_ns, int)
        or decoder_wall_ns < 0
    ):
        raise PredictionError(
            "EfficientSAM decoder timing evidence must be a non-negative integer"
        )
    return {
        "schema_version": EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION,
        "selected_mask_index": selected_index,
        "iou_predictions": ious,
        "iou_predictions_sha256": value["iou_predictions_sha256"],
        "selected_logits_sha256": value["selected_logits_sha256"],
        "selected_binary_mask_sha256": value["selected_binary_mask_sha256"],
        "accepted_mask_sha256": value["accepted_mask_sha256"],
        "decoder_wall_ns": decoder_wall_ns,
    }


def _stable_sam_prediction_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: ([*item] if key == "iou_predictions" else item)
        for key, item in value.items()
        if key != "decoder_wall_ns"
    }


_ROUTE_QUALITY_KEYS = frozenset(
    {
        "support_q10",
        "mean_support",
        "longest_unsupported_run",
        "mean_coherence",
        "detour_ratio",
        "branch_density",
        "endpoint_error",
        "sample_count",
    }
)
_RECOVERY_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "configuration_sha256",
        "provisional",
        "gate",
        "champion_sha256",
        "challenger_sha256",
        "selected_route",
        "selection",
        "fallback_reason",
        "rejection_reason",
        "segmentation_evidence",
    }
)


def _validated_route_quality(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ROUTE_QUALITY_KEYS:
        raise PredictionError(f"{label} has an unsupported structure")
    result: dict[str, Any] = {}
    for key in _ROUTE_QUALITY_KEYS - {"longest_unsupported_run", "sample_count"}:
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise PredictionError(f"{label}.{key} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or number < 0.0:
            raise PredictionError(f"{label}.{key} must be finite and non-negative")
        result[key] = number
    for key in ("longest_unsupported_run", "sample_count"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise PredictionError(f"{label}.{key} must be a non-negative integer")
        result[key] = raw
    return {key: result[key] for key in sorted(result)}


def _optional_reason(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_ERROR_LENGTH:
        raise PredictionError(f"{label} must be null or a bounded non-empty string")
    return " ".join(value.split())


def _validated_recovery_prediction_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECOVERY_EVIDENCE_KEYS:
        raise PredictionError("Smart Recovery evidence has an unsupported structure")
    if value.get("schema_version") != RECOVERY_EVIDENCE_SCHEMA_VERSION:
        raise PredictionError("Smart Recovery evidence has an unsupported schema")
    if value.get("provisional") is not True:
        raise PredictionError("Smart Recovery evidence must remain provisional")
    try:
        from ai_vectorizer.core.smart_recovery import RECOVERY_POLICY_ID
    except Exception as exc:
        raise PredictionError("Smart Recovery policy identity is unavailable") from exc
    if value.get("policy_id") != RECOVERY_POLICY_ID:
        raise PredictionError("Smart Recovery evidence changed policy_id")
    if not _is_sha256(value.get("configuration_sha256")):
        raise PredictionError("Smart Recovery evidence has an invalid configuration hash")
    if not _is_sha256(value.get("champion_sha256")):
        raise PredictionError("Smart Recovery evidence has an invalid champion hash")

    gate = value.get("gate")
    if not isinstance(gate, Mapping) or set(gate) != {"trigger", "reason", "metrics"}:
        raise PredictionError("Smart Recovery gate evidence has an unsupported structure")
    if not isinstance(gate.get("trigger"), bool):
        raise PredictionError("Smart Recovery gate trigger must be boolean")
    gate_reason = _optional_reason(gate.get("reason"), "recovery.gate.reason")
    if gate_reason is None:
        raise PredictionError("Smart Recovery gate reason cannot be null")
    gate_metrics = _validated_route_quality(
        gate.get("metrics"), "recovery.gate.metrics"
    )

    challenger_sha256 = value.get("challenger_sha256")
    if challenger_sha256 is not None and not _is_sha256(challenger_sha256):
        raise PredictionError("Smart Recovery challenger hash is invalid")
    selected_route = value.get("selected_route")
    if selected_route not in {"champion", "challenger"}:
        raise PredictionError("Smart Recovery selected_route is invalid")
    selection_value = value.get("selection")
    selection = None
    if selection_value is not None:
        selection_keys = {
            "reason",
            "champion_metrics",
            "challenger_metrics",
            "strong_ink_retention",
            "route_separation_p95",
        }
        if not isinstance(selection_value, Mapping) or set(selection_value) != selection_keys:
            raise PredictionError("Smart Recovery selection has an unsupported structure")
        selection_reason = _optional_reason(
            selection_value.get("reason"), "recovery.selection.reason"
        )
        if selection_reason is None:
            raise PredictionError("Smart Recovery selection reason cannot be null")
        strong_retention = selection_value.get("strong_ink_retention")
        if (
            isinstance(strong_retention, bool)
            or not isinstance(strong_retention, (int, float))
            or not math.isfinite(float(strong_retention))
            or not 0.0 <= float(strong_retention) <= 1.0
        ):
            raise PredictionError("Smart Recovery strong-ink retention is invalid")
        separation = selection_value.get("route_separation_p95")
        if separation is not None and (
            isinstance(separation, bool)
            or not isinstance(separation, (int, float))
            or not math.isfinite(float(separation))
            or float(separation) < 0.0
        ):
            raise PredictionError("Smart Recovery route separation is invalid")
        challenger_metrics = selection_value.get("challenger_metrics")
        selection = {
            "reason": selection_reason,
            "champion_metrics": _validated_route_quality(
                selection_value.get("champion_metrics"),
                "recovery.selection.champion_metrics",
            ),
            "challenger_metrics": (
                _validated_route_quality(
                    challenger_metrics,
                    "recovery.selection.challenger_metrics",
                )
                if challenger_metrics is not None
                else None
            ),
            "strong_ink_retention": float(strong_retention),
            "route_separation_p95": (
                float(separation) if separation is not None else None
            ),
        }

    segmentation_value = value.get("segmentation_evidence")
    segmentation = (
        _validated_sam_prediction_evidence(segmentation_value)
        if segmentation_value is not None
        else None
    )
    trigger = bool(gate["trigger"])
    fallback_reason = _optional_reason(
        value.get("fallback_reason"), "recovery.fallback_reason"
    )
    rejection_reason = _optional_reason(
        value.get("rejection_reason"), "recovery.rejection_reason"
    )
    if not trigger and any(
        item is not None
        for item in (challenger_sha256, selection, fallback_reason, segmentation)
    ):
        raise PredictionError("an untriggered recovery must not run a challenger")
    if selected_route == "challenger" and (
        not trigger
        or challenger_sha256 is None
        or segmentation is None
        or selection is None
        or selection["reason"] != "accepted"
        or fallback_reason is not None
        or rejection_reason is not None
    ):
        raise PredictionError("accepted recovery evidence is inconsistent")
    if trigger and selected_route == "champion" and rejection_reason is None and fallback_reason is None:
        raise PredictionError("rejected recovery evidence needs a rejection or fallback reason")

    return {
        "schema_version": RECOVERY_EVIDENCE_SCHEMA_VERSION,
        "policy_id": RECOVERY_POLICY_ID,
        "configuration_sha256": value["configuration_sha256"],
        "provisional": True,
        "gate": {
            "trigger": trigger,
            "reason": gate_reason,
            "metrics": gate_metrics,
        },
        "champion_sha256": value["champion_sha256"],
        "challenger_sha256": challenger_sha256,
        "selected_route": selected_route,
        "selection": selection,
        "fallback_reason": fallback_reason,
        "rejection_reason": rejection_reason,
        "segmentation_evidence": segmentation,
    }


def _stable_recovery_prediction_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    gate = dict(result["gate"])
    gate["metrics"] = dict(gate["metrics"])
    result["gate"] = gate
    if result["selection"] is not None:
        selection = dict(result["selection"])
        selection["champion_metrics"] = dict(selection["champion_metrics"])
        if selection["challenger_metrics"] is not None:
            selection["challenger_metrics"] = dict(selection["challenger_metrics"])
        result["selection"] = selection
    if result["segmentation_evidence"] is not None:
        result["segmentation_evidence"] = _stable_sam_prediction_evidence(
            result["segmentation_evidence"]
        )
    return result


def _capture_prediction_evidence(
    pipeline: LoadedPipeline,
    info: BackendInfo,
) -> dict[str, Any] | None:
    if info.actual_backend not in METHOD_SAM_BACKENDS:
        return None
    getter = getattr(pipeline, "prediction_evidence", None)
    if not callable(getter):
        raise PredictionError(
            "EfficientSAM pipeline did not expose prediction evidence"
        )
    is_recovery = info.actual_backend in METHOD_RECOVERY_BACKENDS
    evidence = (
        _validated_recovery_prediction_evidence(getter())
        if is_recovery
        else _validated_sam_prediction_evidence(getter())
    )
    sample_key = (
        "recovery_prediction_samples" if is_recovery else "sam_prediction_samples"
    )
    samples = info.runtime_details.setdefault(sample_key, [])
    if not isinstance(samples, list):
        raise PredictionError(
            "model pipeline changed prediction sample evidence storage"
        )
    if is_recovery:
        samples.append(_stable_recovery_prediction_evidence(evidence) | {
            "segmentation_evidence": (
                dict(evidence["segmentation_evidence"])
                if evidence["segmentation_evidence"] is not None
                else None
            )
        })
    else:
        samples.append(
            {
                key: ([*item] if key == "iou_predictions" else item)
                for key, item in evidence.items()
            }
        )
    return evidence


def _canonical_artifact_bytes(
    request: WorkerRequest,
    actual_backend: str,
    points: Sequence[Sequence[float]],
    *,
    prediction_evidence: Mapping[str, Any] | None = None,
) -> bytes:
    normalized: list[list[float]] = []
    if isinstance(points, (str, bytes)):
        raise PredictionError("trace kernel points must be a sequence of [x, y] pairs")
    for index, point in enumerate(points):
        if index >= MAX_POINTS:
            raise PredictionError(
                f"trace kernel returned more than {MAX_POINTS} ordered points"
            )
        if isinstance(point, (str, bytes)) or len(point) != 2:
            raise PredictionError(f"trace point {index} must be an [x, y] pair")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise PredictionError(f"trace point {index} is not numeric") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise PredictionError(f"trace point {index} is not finite")
        if not (0 <= x <= request.width - 1 and 0 <= y <= request.height - 1):
            raise PredictionError(f"trace point {index} lies outside the image")
        normalized.append([x, y])
    if len(normalized) < 2:
        raise PredictionError("trace kernel must return at least two ordered points")

    ink_or_recovery = (
        actual_backend in METHOD_INK_BACKENDS
        or actual_backend in METHOD_RECOVERY_BACKENDS
    )
    metadata = {
        "actual_backend": actual_backend,
        "configuration_sha256": _configuration_sha256(request.configuration),
        "input_sha256": request.image_sha256,
        "prompt_sha256": benchmark_evidence.prompt_sha256(request.prompt),
        "requested_backend": request.requested_backend,
        "smoothing": (
            INK_LIVEWIRE_SMOOTHING_PROFILE
            if ink_or_recovery
            else PRODUCT_SMOOTHING_PROFILE
        ),
        "trace_kernel": (
            "ai_vectorizer.core.livewire"
            if actual_backend in METHOD_INK_BACKENDS
            else "ai_vectorizer.core.trace_kernel"
        ),
    }
    if request.requested_backend in METHOD_SOURCE_GRID_BACKENDS:
        metadata.update(
            source_tile_origin_xy=list(request.source_tile_origin_xy),
            source_grid_input_sha256=(
                benchmark_evidence.source_grid_input_sha256(
                    request.image_sha256,
                    request.source_tile_origin_xy,
                )
            ),
        )
    if actual_backend == EFFICIENTSAM_BACKEND:
        from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT

        bundle_id, fingerprint = _sam_model_contract(
            request.configuration, require_canny=True
        )
        if prediction_evidence is None:
            raise PredictionError(
                "EfficientSAM canonical artifacts require prediction evidence"
            )
        validated_evidence = _validated_sam_prediction_evidence(prediction_evidence)
        metadata.update(
            mask_trace_kernel="ai_vectorizer.core.sam_trace_kernel",
            model_bundle_id=bundle_id,
            model_bundle_sha256=fingerprint,
            model_source_commit=EFFICIENTSAM_TI_SPLIT.source_commit,
            sam_prompt_tensor_sha256=(
                benchmark_evidence.sam_prompt_tensor_sha256(request.prompt)
            ),
            segmentation_evidence=_stable_sam_prediction_evidence(
                validated_evidence
            ),
        )
    elif actual_backend in METHOD_RECOVERY_BACKENDS:
        from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT

        bundle_id, fingerprint = _sam_model_contract(
            request.configuration, require_canny=False
        )
        recovery_config = _recovery_contract(request.configuration)
        if prediction_evidence is None:
            raise PredictionError(
                "Smart Recovery canonical artifacts require recovery evidence"
            )
        recovery_evidence = _validated_recovery_prediction_evidence(
            prediction_evidence
        )
        if recovery_evidence["configuration_sha256"] != recovery_config.sha256:
            raise PredictionError(
                "Smart Recovery artifact evidence changed the configured policy hash"
            )
        metadata.update(
            livewire_kernel="ai_vectorizer.core.livewire",
            mask_trace_kernel="ai_vectorizer.core.sam_trace_kernel",
            recovery_kernel="ai_vectorizer.core.smart_recovery",
            model_bundle_id=bundle_id,
            model_bundle_sha256=fingerprint,
            model_source_commit=EFFICIENTSAM_TI_SPLIT.source_commit,
            sam_prompt_tensor_sha256=(
                benchmark_evidence.recovery_prompt_tensor_sha256(
                    request.prompt,
                    width=request.width,
                    height=request.height,
                )
            ),
            recovery_evidence=_stable_recovery_prediction_evidence(
                recovery_evidence
            ),
        )
    elif prediction_evidence is not None:
        raise PredictionError(
            "non-SAM canonical artifacts must not include segmentation evidence"
        )
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "coordinate_space": COORDINATE_SPACE,
        "image_size": {"width": request.width, "height": request.height},
        "paths": [{"id": "trace-0", "closed": False, "points": normalized}],
        "metadata": metadata,
    }
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _peak_rss_bytes() -> int:
    try:
        import resource
    except ImportError:  # pragma: no cover - exercised by Windows workers.
        if sys.platform != "win32":
            raise WorkerDependencyError("peak RSS measurement is unavailable")
        # PROCESS_MEMORY_COUNTERS.WorkingSetSize is the current RSS. A Windows
        # subprocess remains one job per sample, so this is a conservative
        # fallback when the Unix ru_maxrss high-water mark is unavailable.
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        ):
            raise WorkerDependencyError("Windows peak RSS measurement failed")
        return int(counters.PeakWorkingSetSize)

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the BSDs used by CI report KiB.
    return raw if sys.platform == "darwin" else raw * 1024


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_write_no_replace(path: Path, raw: bytes) -> None:
    """Atomically publish ``raw`` only when ``path`` is still absent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-linking a same-directory temporary file is atomic and fails if
        # another process created the requested artifact after validation.
        os.link(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass


def _discard_published_artifact(path: Path) -> None:
    """Remove an unclaimed output after publication verification fails."""

    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise PredictionError(
            "could not remove an unverified published artifact"
        ) from exc


def _base_runtime(request: WorkerRequest, info: BackendInfo | None) -> dict[str, Any]:
    expected_kind, expected_provider = _provider_contract(request.requested_backend)
    package_versions = dict(info.package_versions) if info else {
        "numpy": None,
        "onnxruntime" if expected_kind == "onnxruntime" else "opencv": None,
    }
    if info:
        thread_settings = dict(info.thread_settings)
    elif expected_kind == "onnxruntime":
        thread_settings = {
            "threads": request.threads,
            "onnx_intra_op_num_threads": 1,
            "onnx_inter_op_num_threads": 1,
            "onnx_execution_mode": "ORT_SEQUENTIAL",
            "onnx_graph_optimization_level": "ORT_ENABLE_ALL",
            **{variable: os.environ.get(variable) for variable in CPU_THREAD_VARIABLES},
        }
    else:
        thread_settings = {
            "threads": request.threads,
            "opencv_set_num_threads": 0,
            "opencl": False,
            **{variable: os.environ.get(variable) for variable in CPU_THREAD_VARIABLES},
        }
    thread_settings.setdefault("threads", request.threads)
    runtime = {
        "adapter_version": info.adapter_version if info else WORKER_ADAPTER_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine() or "unknown",
        "actual_provider": info.actual_provider if info else expected_provider,
        "provider_kind": info.provider_kind if info else expected_kind,
        "provider_device_type": info.provider_device_type if info else "cpu",
        "provider_verified": info.provider_verified if info else False,
        "package_versions": package_versions,
        "source_files_sha256": (
            dict(info.source_files_sha256)
            if info
            else {"benchmarks/worker.py": _sha256_file(Path(__file__))}
        ),
        "thread_settings": thread_settings,
        "input_sha256": request.image_sha256,
        "configuration_sha256": _configuration_sha256(request.configuration),
        "prompt_sha256": benchmark_evidence.prompt_sha256(request.prompt),
        "deterministic": None,
        "output_sha256_samples": [],
    }
    if request.requested_backend in METHOD_SOURCE_GRID_BACKENDS:
        runtime.update(
            source_tile_origin_xy=list(request.source_tile_origin_xy),
            source_grid_input_sha256=(
                benchmark_evidence.source_grid_input_sha256(
                    request.image_sha256,
                    request.source_tile_origin_xy,
                )
            ),
        )
    if info and info.model_artifacts_sha256:
        runtime["model_artifacts_sha256"] = dict(info.model_artifacts_sha256)
    if info and info.model_bundle_id is not None:
        runtime["model_bundle_id"] = info.model_bundle_id
    if info and info.model_bundle_sha256 is not None:
        runtime["model_bundle_sha256"] = info.model_bundle_sha256
    if info and info.runtime_details:
        runtime.update(dict(info.runtime_details))
    if info is None and request.requested_backend in METHOD_SAM_BACKENDS:
        from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT

        bundle_id, fingerprint = _sam_model_contract(
            request.configuration,
            require_canny=request.requested_backend == EFFICIENTSAM_BACKEND,
        )
        runtime.update(
            model_bundle_id=bundle_id,
            model_bundle_sha256=fingerprint,
            model_source_commit=EFFICIENTSAM_TI_SPLIT.source_commit,
            model_artifacts_sha256={
                artifact.id: artifact.sha256
                for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
            },
        )
    if request.requested_backend == EFFICIENTSAM_BACKEND:
        runtime["sam_prompt_tensor_sha256"] = (
            benchmark_evidence.sam_prompt_tensor_sha256(request.prompt)
        )
    elif request.requested_backend in METHOD_RECOVERY_BACKENDS:
        runtime["sam_prompt_tensor_sha256"] = (
            benchmark_evidence.recovery_prompt_tensor_sha256(
                request.prompt,
                width=request.width,
                height=request.height,
            )
        )
    if request.requested_backend in METHOD_RECOVERY_BACKENDS:
        recovery_config = _recovery_contract(request.configuration)
        runtime.update(
            recovery_policy_id=recovery_config.policy_id,
            recovery_configuration_sha256=recovery_config.sha256,
            recovery_provisional=True,
        )
    return runtime


def _pipeline_attestation_error(
    request: WorkerRequest,
    info: BackendInfo,
) -> str | None:
    if (
        info.actual_backend not in SUPPORTED_BACKENDS
        or info.actual_backend
        not in {request.requested_backend, request.fallback_backend}
    ):
        return "pipeline selected an unrequested backend"

    expected_kind, expected_provider = _provider_contract(info.actual_backend)
    if (
        info.provider_kind != expected_kind
        or info.actual_provider != expected_provider
        or info.provider_device_type.lower() != "cpu"
        or info.provider_verified is not True
    ):
        return f"pipeline did not attest the required {expected_provider} CPU provider"
    if (
        isinstance(info.thread_settings.get("threads"), bool)
        or not isinstance(info.thread_settings.get("threads"), int)
        or info.thread_settings["threads"] != request.threads
    ):
        return "pipeline did not attest the requested CPU thread count"

    if expected_kind == "opencv":
        if (
            info.thread_settings.get("opencl") is not False
            or info.thread_settings.get("opencv_set_num_threads") != 0
        ):
            return "pipeline did not attest disabled OpenCV threading/OpenCL"
        return None

    if expected_kind == "python":
        if (
            info.thread_settings.get("opencl") is not False
            or info.thread_settings.get("opencv_set_num_threads") != 0
        ):
            return "pipeline did not attest deterministic CPU Live-Wire execution"
        return None

    required_ort_threads = {
        "onnx_intra_op_num_threads": 1,
        "onnx_inter_op_num_threads": 1,
        "onnx_execution_mode": "ORT_SEQUENTIAL",
        "onnx_graph_optimization_level": "ORT_ENABLE_ALL",
    }
    if any(
        info.thread_settings.get(key) != value
        for key, value in required_ort_threads.items()
    ):
        return "pipeline did not attest deterministic single-thread ONNX execution"
    if (
        info.thread_settings.get("opencv_set_num_threads") != 0
        or info.thread_settings.get("opencv_effective_num_threads")
        != request.threads
        or info.thread_settings.get("opencl") is not False
    ):
        return "pipeline did not attest deterministic CPU OpenCV execution"
    if info.runtime_details.get("onnx_providers") != {
        "encoder": ["CPUExecutionProvider"],
        "decoder": ["CPUExecutionProvider"],
    }:
        return "pipeline raw ONNX provider evidence did not attest CPU-only sessions"
    if info.runtime_details.get("onnx_session_options") != {
        "encoder": dict(EFFICIENTSAM_ORT_SESSION_OPTIONS),
        "decoder": dict(EFFICIENTSAM_ORT_SESSION_OPTIONS),
    }:
        return "pipeline raw ONNX session evidence did not attest deterministic options"
    bundle_id, fingerprint = _sam_model_contract(
        request.configuration,
        require_canny=info.actual_backend == EFFICIENTSAM_BACKEND,
    )
    from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT

    expected_artifacts = {
        artifact.id: artifact.sha256
        for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
    }
    if (
        info.model_bundle_id != bundle_id
        or info.model_bundle_sha256 != fingerprint
        or info.model_artifacts_sha256 != expected_artifacts
        or info.runtime_details.get("model_source_commit")
        != EFFICIENTSAM_TI_SPLIT.source_commit
    ):
        return "pipeline did not attest the pinned EfficientSAM split-model hashes"
    return None


def _failure_result(
    request: WorkerRequest,
    exc: BaseException,
    *,
    info: BackendInfo | None,
    actual_backend: str | None,
    model_load_wall_ns: int,
    image_decode_wall_ns: int | None,
    wall_samples: Sequence[int] = (),
    cpu_samples: Sequence[int] = (),
    output_hashes: Sequence[str] = (),
    warmup_wall_samples: Sequence[int] = (),
    warmup_runs_completed: int = 0,
    prior_backend_failure: str | None = None,
) -> dict[str, Any]:
    runtime = _base_runtime(request, info)
    runtime["latency_scope"] = LATENCY_SCOPE
    runtime["image_load_wall_ns"] = image_decode_wall_ns
    runtime["image_decode_wall_ns"] = image_decode_wall_ns
    runtime["warmup_wall_ns_samples"] = list(warmup_wall_samples)
    runtime["output_sha256_samples"] = list(output_hashes)
    runtime["deterministic"] = (
        len(set(output_hashes)) <= 1 if output_hashes else None
    )
    error = _safe_error(exc)
    if prior_backend_failure:
        error = (
            f"requested backend failed ({prior_backend_failure}); "
            f"fallback execution failed ({error})"
        )[:MAX_ERROR_LENGTH]
    return {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "prediction": {
            "execution": {
                "status": "failed",
                "requested_backend": request.requested_backend,
                "actual_backend": actual_backend,
                "fallback_reason": None,
                "error": error,
                "device": "cpu",
                "runtime": runtime,
                "timing": {
                    "warmup_runs": warmup_runs_completed,
                    "wall_ns_samples": list(wall_samples),
                    "cpu_ns_samples": list(cpu_samples),
                    "model_load_wall_ns": model_load_wall_ns,
                    "peak_rss_bytes": _peak_rss_bytes(),
                },
            }
        },
    }


def run_worker(
    request: WorkerRequest,
    *,
    pipeline_loader: PipelineLoader | None = None,
    perf_counter_ns: Callable[[], int] = time.perf_counter_ns,
    process_time_ns: Callable[[], int] = time.process_time_ns,
) -> dict[str, Any]:
    """Execute one detector/sample in the current (normally fresh) process."""

    if request.device != "cpu":
        raise WorkerRequestError("Only CPU worker requests are supported.")
    if (
        request.requested_backend in METHOD_RECOVERY_BACKENDS
        and request.fallback_backend is not None
    ):
        raise WorkerRequestError(
            "Smart Recovery does not permit an external fallback_backend; "
            "model or inference failures must keep its Ink champion."
        )
    _validate_backend_configuration(request.requested_backend, request.configuration)
    if pipeline_loader is None:
        def loader(backend: str, threads: int) -> LoadedPipeline:
            return _load_pipeline(
                backend,
                threads,
                model_cache=request.model_cache,
            )
    else:
        loader = pipeline_loader
    for variable in CPU_THREAD_VARIABLES:
        os.environ[variable] = str(request.threads)
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

    load_started = perf_counter_ns()
    fallback_reason = None
    pipeline: LoadedPipeline | None = None
    load_error: BaseException | None = None
    try:
        pipeline = loader(request.requested_backend, request.threads)
    except Exception as exc:  # Dependency/backend errors are result evidence.
        load_error = exc
        if request.fallback_backend is not None:
            fallback_reason = _safe_error(exc)
            try:
                pipeline = loader(request.fallback_backend, request.threads)
            except Exception as fallback_exc:
                load_error = WorkerDependencyError(
                    f"requested backend failed ({_safe_error(exc)}); "
                    f"fallback failed ({_safe_error(fallback_exc)})"
                )
    model_load_wall_ns = max(0, perf_counter_ns() - load_started)
    if pipeline is None:
        assert load_error is not None
        return _failure_result(
            request,
            load_error,
            info=None,
            actual_backend=None,
            model_load_wall_ns=model_load_wall_ns,
            image_decode_wall_ns=None,
        )

    info = pipeline.info
    attestation_error = _pipeline_attestation_error(request, info)
    if attestation_error is not None:
        return _failure_result(
            request,
            PredictionError(attestation_error),
            info=info,
            actual_backend=info.actual_backend,
            model_load_wall_ns=model_load_wall_ns,
            image_decode_wall_ns=None,
            prior_backend_failure=fallback_reason,
        )

    decode_started = perf_counter_ns()
    try:
        image = pipeline.load_image(request.image_path, request.width, request.height)
        if info.actual_backend in METHOD_SOURCE_GRID_BACKENDS:
            if not isinstance(image, dict):
                raise PredictionError(
                    "Ink v2 source-grid input must use the product image record"
                )
            image["source_tile_origin_xy"] = tuple(
                request.source_tile_origin_xy
            )
    except Exception as exc:
        return _failure_result(
            request,
            exc,
            info=info,
            actual_backend=info.actual_backend,
            model_load_wall_ns=model_load_wall_ns,
            image_decode_wall_ns=max(0, perf_counter_ns() - decode_started),
            prior_backend_failure=fallback_reason,
        )
    image_decode_wall_ns = max(0, perf_counter_ns() - decode_started)

    warmup_runs_completed = 0
    warmup_wall_samples: list[int] = []
    try:
        for _ in range(request.warmup_runs):
            warmup_started = perf_counter_ns()
            warmup_points = pipeline.predict(image, request.prompt, request.configuration)
            warmup_evidence = _capture_prediction_evidence(pipeline, info)
            _canonical_artifact_bytes(
                request,
                info.actual_backend,
                warmup_points,
                prediction_evidence=warmup_evidence,
            )
            warmup_wall_samples.append(
                max(0, perf_counter_ns() - warmup_started)
            )
            warmup_runs_completed += 1
    except Exception as exc:
        return _failure_result(
            request,
            exc,
            info=info,
            actual_backend=info.actual_backend,
            model_load_wall_ns=model_load_wall_ns,
            image_decode_wall_ns=image_decode_wall_ns,
            warmup_wall_samples=warmup_wall_samples,
            warmup_runs_completed=warmup_runs_completed,
            prior_backend_failure=fallback_reason,
        )

    wall_samples: list[int] = []
    cpu_samples: list[int] = []
    output_hashes: list[str] = []
    first_output: bytes | None = None
    try:
        for _ in range(request.measurement_runs):
            wall_started = perf_counter_ns()
            cpu_started = process_time_ns()
            points = pipeline.predict(image, request.prompt, request.configuration)
            prediction_evidence = _capture_prediction_evidence(pipeline, info)
            output = _canonical_artifact_bytes(
                request,
                info.actual_backend,
                points,
                prediction_evidence=prediction_evidence,
            )
            cpu_samples.append(max(0, process_time_ns() - cpu_started))
            wall_samples.append(max(0, perf_counter_ns() - wall_started))
            output_hashes.append(_sha256_bytes(output))
            if first_output is None:
                first_output = output
    except Exception as exc:
        return _failure_result(
            request,
            exc,
            info=info,
            actual_backend=info.actual_backend,
            model_load_wall_ns=model_load_wall_ns,
            image_decode_wall_ns=image_decode_wall_ns,
            wall_samples=wall_samples,
            cpu_samples=cpu_samples,
            output_hashes=output_hashes,
            warmup_wall_samples=warmup_wall_samples,
            warmup_runs_completed=warmup_runs_completed,
            prior_backend_failure=fallback_reason,
        )

    assert first_output is not None
    try:
        _atomic_write_no_replace(request.artifact_path, first_output)
    except Exception as exc:
        return _failure_result(
            request,
            exc,
            info=info,
            actual_backend=info.actual_backend,
            model_load_wall_ns=model_load_wall_ns,
            image_decode_wall_ns=image_decode_wall_ns,
            wall_samples=wall_samples,
            cpu_samples=cpu_samples,
            output_hashes=output_hashes,
            warmup_wall_samples=warmup_wall_samples,
            warmup_runs_completed=warmup_runs_completed,
            prior_backend_failure=fallback_reason,
        )
    try:
        published_hash = _sha256_file(request.artifact_path)
    except Exception as exc:
        try:
            _discard_published_artifact(request.artifact_path)
        except Exception as cleanup_exc:
            exc = PredictionError(
                f"{_safe_error(exc)}; {_safe_error(cleanup_exc)}"
            )
        return _failure_result(
            request,
            exc,
            info=info,
            actual_backend=info.actual_backend,
            model_load_wall_ns=model_load_wall_ns,
            image_decode_wall_ns=image_decode_wall_ns,
            wall_samples=wall_samples,
            cpu_samples=cpu_samples,
            output_hashes=output_hashes,
            warmup_wall_samples=warmup_wall_samples,
            warmup_runs_completed=warmup_runs_completed,
            prior_backend_failure=fallback_reason,
        )
    if published_hash != output_hashes[0]:
        try:
            _discard_published_artifact(request.artifact_path)
        except Exception as cleanup_exc:
            publication_error: BaseException = PredictionError(
                "published artifact hash changed during atomic write; "
                f"{_safe_error(cleanup_exc)}"
            )
        else:
            publication_error = PredictionError(
                "published artifact hash changed during atomic write"
            )
        return _failure_result(
            request,
            publication_error,
            info=info,
            actual_backend=info.actual_backend,
            model_load_wall_ns=model_load_wall_ns,
            image_decode_wall_ns=image_decode_wall_ns,
            wall_samples=wall_samples,
            cpu_samples=cpu_samples,
            output_hashes=output_hashes,
            warmup_wall_samples=warmup_wall_samples,
            warmup_runs_completed=warmup_runs_completed,
            prior_backend_failure=fallback_reason,
        )

    runtime = _base_runtime(request, info)
    runtime.update(
        latency_scope=LATENCY_SCOPE,
        image_load_wall_ns=image_decode_wall_ns,
        image_decode_wall_ns=image_decode_wall_ns,
        warmup_wall_ns_samples=warmup_wall_samples,
        deterministic=len(set(output_hashes)) == 1,
        output_sha256_samples=output_hashes,
    )
    status = "fallback" if info.actual_backend != request.requested_backend else "ok"
    if status == "fallback" and fallback_reason is None:
        fallback_reason = "worker selected the configured fallback backend"
    prediction = {
        "artifact": request.artifact_manifest_path,
        "sha256": published_hash,
        "execution": {
            "status": status,
            "requested_backend": request.requested_backend,
            "actual_backend": info.actual_backend,
            "fallback_reason": fallback_reason if status == "fallback" else None,
            "error": None,
            "device": "cpu",
            "runtime": runtime,
            "timing": {
                "warmup_runs": request.warmup_runs,
                "wall_ns_samples": wall_samples,
                "cpu_ns_samples": cpu_samples,
                "model_load_wall_ns": model_load_wall_ns,
                "peak_rss_bytes": _peak_rss_bytes(),
            },
        },
    }
    return {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "prediction": prediction,
    }


class _OpenCVTracePipeline:
    """Real adapter joining ``EdgeDetector`` to the shared trace kernel."""

    def __init__(self, backend: str, threads: int, np_module: Any, cv2_module: Any):
        import ai_vectorizer as ai_vectorizer_package
        from ai_vectorizer import core as core_package
        from ai_vectorizer.core import dependencies as dependencies_module
        from ai_vectorizer.core import edge_detector as edge_detector_module
        from ai_vectorizer.core import efficientsam_spec as efficientsam_spec_module
        import benchmarks as benchmarks_package

        try:
            from ai_vectorizer.core import trace_kernel
        except Exception as exc:
            raise WorkerDependencyError(
                "shared ai_vectorizer.core.trace_kernel is unavailable"
            ) from exc

        self._np = np_module
        self._cv2 = cv2_module
        self._trace_kernel = trace_kernel
        if edge_detector_module._skimage_skeletonize is None:
            raise WorkerDependencyError(
                "controlled Canny/LSD benchmarks require scikit-image skeletonize"
            )
        EdgeDetector = edge_detector_module.EdgeDetector
        edge_backend = METHOD_EDGE_BACKENDS[backend]
        self._detector = EdgeDetector(method=edge_backend)
        if edge_backend == "lsd" and getattr(self._detector, "lsd", None) is None:
            raise BackendUnavailableError("OpenCV LineSegmentDetector is unavailable")

        package_versions: dict[str, str | None] = {
            "numpy": str(getattr(np_module, "__version__", "unknown")),
            "opencv": str(getattr(cv2_module, "__version__", "unknown")),
        }
        for distribution in (
            "opencv-python-headless",
            "opencv-python",
            "opencv-contrib-python-headless",
            "opencv-contrib-python",
        ):
            try:
                package_versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
        try:
            package_versions["scikit-image"] = importlib.metadata.version("scikit-image")
        except importlib.metadata.PackageNotFoundError:
            package_versions["scikit-image"] = None
        actual_threads = (
            int(cv2_module.getNumThreads())
            if hasattr(cv2_module, "getNumThreads")
            else threads
        )
        if actual_threads < 1:
            raise WorkerDependencyError(
                "OpenCV did not report a positive effective CPU thread count"
            )
        self.info = BackendInfo(
            actual_backend=backend,
            provider_kind="opencv",
            actual_provider="OpenCV CPU",
            provider_device_type="cpu",
            adapter_version=OPENCV_ADAPTER_VERSION,
            package_versions=package_versions,
            thread_settings={
                "threads": actual_threads,
                "opencv_set_num_threads": 0,
                "opencl": bool(
                    getattr(cv2_module, "ocl", None)
                    and hasattr(cv2_module.ocl, "useOpenCL")
                    and cv2_module.ocl.useOpenCL()
                ),
                **{variable: os.environ.get(variable) for variable in CPU_THREAD_VARIABLES},
            },
            provider_verified=True,
            source_files_sha256={
                "benchmarks/__init__.py": _sha256_file(
                    Path(benchmarks_package.__file__)
                ),
                "benchmarks/worker.py": _sha256_file(Path(__file__)),
                "benchmarks/evidence.py": _sha256_file(
                    Path(benchmark_evidence.__file__)
                ),
                "benchmarks/geometry.py": _sha256_file(
                    Path(benchmark_geometry.__file__)
                ),
                "benchmarks/manifest.py": _sha256_file(
                    Path(benchmark_manifest.__file__)
                ),
                "ai_vectorizer/__init__.py": _sha256_file(
                    Path(ai_vectorizer_package.__file__)
                ),
                "ai_vectorizer/core/__init__.py": _sha256_file(
                    Path(core_package.__file__)
                ),
                "ai_vectorizer/core/dependencies.py": _sha256_file(
                    Path(dependencies_module.__file__)
                ),
                "ai_vectorizer/core/edge_detector.py": _sha256_file(
                    Path(edge_detector_module.__file__)
                ),
                "ai_vectorizer/core/efficientsam_spec.py": _sha256_file(
                    Path(efficientsam_spec_module.__file__)
                ),
                "ai_vectorizer/core/trace_kernel.py": _sha256_file(
                    Path(trace_kernel.__file__)
                ),
            },
        )

    def load_image(self, path: Path, width: int, height: int) -> Any:
        image = self._cv2.imread(str(path), self._cv2.IMREAD_UNCHANGED)
        if image is None:
            raise PredictionError(f"OpenCV could not decode {path.name}")
        if image.dtype != self._np.uint8:
            raise PredictionError("benchmark detector input must be uint8")
        if tuple(image.shape[:2]) != (height, width):
            raise PredictionError("decoded image dimensions changed after validation")
        if image.ndim == 3 and image.shape[2] == 3:
            image = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = self._cv2.cvtColor(image, self._cv2.COLOR_BGRA2RGBA)
        elif image.ndim == 3 and image.shape[2] in (1, 2):
            # Grayscale(+alpha) PNGs are valid benchmark inputs. EdgeDetector
            # consumes the luminance plane and does not use alpha.
            image = image[..., 0]
        elif image.ndim not in (2, 3):
            raise PredictionError("benchmark image must be gray, RGB, or RGBA")
        elif image.ndim == 3:
            raise PredictionError("benchmark image has an unsupported channel count")
        return self._np.ascontiguousarray(image)

    def predict(
        self,
        image: Any,
        prompt: TracePrompt,
        configuration: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]:
        low, high = _canny_thresholds(configuration)
        edge_weight = _edge_weight(configuration)
        edges = self._detector.detect_edges(
            image,
            low_threshold=low,
            high_threshold=high,
        )
        cost_map = self._detector.get_edge_cost_map(edges, edge_weight=edge_weight)

        # trace_path is the strict, QGIS-free product kernel.  Its raw A* path
        # excludes product smoothing; centerline_points applies the historical
        # moving-average + open Chaikin profile and restores the segment start.
        start_xy = (int(prompt.start_xy[0]), int(prompt.start_xy[1]))
        end_xy = (int(prompt.end_xy[0]), int(prompt.end_xy[1]))
        height, width = cost_map.shape
        for label, (x, y) in (("start", start_xy), ("end", end_xy)):
            if not (0 <= x < int(width) and 0 <= y < int(height)):
                raise PredictionError(f"truncated {label} prompt lies outside the cost map")
        trace_config = self._trace_kernel.TraceConfig(
            validate_all_costs=False,
            validate_accessed_costs=False,
        )
        result = self._trace_kernel.trace_path(
            cost_map,
            start_xy,
            end_xy,
            allow_partial=False,
            config=trace_config,
        )
        if result.status != "complete":
            raise PredictionError("shared trace kernel did not reach the requested endpoint")
        return self._trace_kernel.centerline_points(
            result,
            segment_start_xy=prompt.start_xy,
            segment_target_xy=prompt.end_xy,
        )


class _InkLiveWirePipeline:
    """Product-like Ink Centerline + bounded Live-Wire benchmark adapter."""

    def __init__(
        self,
        backend: str,
        threads: int,
        np_module: Any,
        cv2_module: Any,
        *,
        opencv_effective_num_threads: int,
        opencv_opencl_enabled: bool,
    ):
        import ai_vectorizer as ai_vectorizer_package
        from ai_vectorizer import core as core_package
        from ai_vectorizer.core import edge_detector as edge_detector_module
        from ai_vectorizer.core import line_evidence as line_evidence_module
        from ai_vectorizer.core import livewire as livewire_module
        from ai_vectorizer.core import trace_kernel
        import benchmarks as benchmarks_package

        if backend not in METHOD_INK_BACKENDS:
            raise BackendUnavailableError(f"unsupported Ink Live-Wire backend: {backend}")
        if threads != 1:
            raise WorkerRequestError("Ink Live-Wire benchmark workers require threads=1")
        if not livewire_module.is_livewire_available():
            raise WorkerDependencyError("Ink Live-Wire requires a working SciPy runtime")

        self._backend = backend
        self._np = np_module
        self._cv2 = cv2_module
        self._edge_detector_module = edge_detector_module
        self._livewire = livewire_module
        self._trace_kernel = trace_kernel
        self._detector = edge_detector_module.EdgeDetector(method="ink")

        source_modules = (
            benchmarks_package,
            benchmark_evidence,
            benchmark_geometry,
            benchmark_manifest,
            ai_vectorizer_package,
            core_package,
            edge_detector_module,
            line_evidence_module,
            livewire_module,
            trace_kernel,
        )
        repository_root = Path(__file__).resolve().parents[1]
        source_hashes = {"benchmarks/worker.py": _sha256_file(Path(__file__))}
        for module in source_modules:
            module_path = Path(module.__file__).resolve()
            try:
                label = module_path.relative_to(repository_root).as_posix()
            except ValueError:
                label = module_path.name
            source_hashes[label] = _sha256_file(module_path)

        self.info = BackendInfo(
            actual_backend=backend,
            provider_kind="python",
            actual_provider="Python CPU",
            provider_device_type="cpu",
            adapter_version=INK_LIVEWIRE_ADAPTER_VERSION,
            package_versions={
                "numpy": str(getattr(np_module, "__version__", "unknown")),
                "opencv": str(getattr(cv2_module, "__version__", "unknown")),
                "scipy": _distribution_version("scipy"),
            },
            thread_settings={
                "threads": threads,
                "opencv_set_num_threads": 0,
                "opencv_effective_num_threads": opencv_effective_num_threads,
                "opencl": opencv_opencl_enabled,
                **{variable: os.environ.get(variable) for variable in CPU_THREAD_VARIABLES},
            },
            provider_verified=True,
            source_files_sha256=source_hashes,
            runtime_details={
                "livewire_window_px": 320,
                "target_snap_radius_px": 6,
                "smoothing_window_px": 5,
                "ink_evidence_api": (
                    "detect_ink_evidence"
                    if backend == INK_LIVEWIRE_V2_BACKEND
                    else "detect_edges"
                ),
            },
        )

    def load_image(self, path: Path, width: int, height: int) -> Any:
        decode_started = time.perf_counter_ns()
        image = self._cv2.imread(str(path), self._cv2.IMREAD_UNCHANGED)
        if image is None:
            raise PredictionError(f"OpenCV could not decode {path.name}")
        if image.dtype != self._np.uint8 or tuple(image.shape[:2]) != (height, width):
            raise PredictionError("Ink Live-Wire input must decode as sized uint8")
        if image.ndim == 2:
            rgb = image
        elif image.ndim == 3 and image.shape[2] == 1:
            rgb = image[..., 0]
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGRA2RGBA)
        else:
            raise PredictionError("Ink Live-Wire input must be gray, RGB, or RGBA")
        self.info.runtime_details["image_file_decode_ns"] = max(
            0, time.perf_counter_ns() - decode_started
        )
        return {
            "rgb": self._np.ascontiguousarray(rgb),
            "edges": None,
            "evidence": None,
        }

    def _prepare_ink(self, image: Any) -> tuple[Any, Any | None]:
        if image["edges"] is not None:
            return image["edges"], image["evidence"]
        started = time.perf_counter_ns()
        if self._backend == INK_LIVEWIRE_V1_BACKEND:
            # This is intentionally the legacy API. It is the OFF/control
            # boundary for proving that v2 is not silently v1 evidence.
            edges = self._detector.detect_edges(image["rgb"])
            evidence = None
        else:
            evidence = self._detector.detect_ink_evidence(
                image["rgb"],
                tile_origin=image.get("source_tile_origin_xy", (0, 0)),
            )
            centerline = self._np.asarray(evidence.centerline)
            if tuple(centerline.shape) != tuple(image["rgb"].shape[:2]):
                raise PredictionError("LineEvidence.centerline changed image dimensions")
            if centerline.dtype == self._np.bool_:
                edges = centerline.astype(self._np.uint8) * 255
            else:
                edges = self._np.where(centerline > 0, 255, 0).astype(self._np.uint8)
        image["edges"] = self._np.ascontiguousarray(edges)
        image["evidence"] = evidence
        self.info.runtime_details["ink_cache_fill_wall_ns"] = max(
            0, time.perf_counter_ns() - started
        )
        return image["edges"], image["evidence"]

    def _champion_path(
        self,
        image: Any,
        prompt: TracePrompt,
        configuration: Mapping[str, Any],
    ) -> tuple[list[tuple[float, float]], Any | None, Any]:
        strength = _ink_livewire_contract(configuration)
        edges, evidence = self._prepare_ink(image)
        root = (int(prompt.start_xy[0]), int(prompt.start_xy[1]))
        incoming_direction = None
        if prompt.previous_xy is not None:
            previous = (
                int(prompt.previous_xy[0]),
                int(prompt.previous_xy[1]),
            )
            incoming_direction = (
                root[0] - previous[0],
                root[1] - previous[1],
            )
            if incoming_direction == (0, 0):
                incoming_direction = None
        config = self._livewire.LiveWireConfig(
            max_window_size=320,
            target_snap_radius=6,
        )
        tree = self._livewire.build_livewire_tree(
            image["rgb"],
            edges,
            root,
            strength=strength,
            incoming_direction=incoming_direction,
            evidence=evidence if self._backend == INK_LIVEWIRE_V2_BACKEND else None,
            config=config,
        )
        path = list(tree.trace(prompt.end_xy))
        if len(path) > 5:
            smoothed = list(self._trace_kernel.smooth_pixel_path(path, window_size=5))
            smoothed[0] = path[0]
            smoothed[-1] = path[-1]
            path = smoothed
        if len(path) < 2:
            raise PredictionError("Ink Live-Wire did not return an ordered segment")
        path[0] = (float(prompt.start_xy[0]), float(prompt.start_xy[1]))
        # Recovery must remain inside this exact bounded tree window.  Return
        # the tree rather than reconstructing a second, subtly different crop
        # from the prompt coordinates in the challenger adapter.
        return path, evidence, tree

    def predict(
        self,
        image: Any,
        prompt: TracePrompt,
        configuration: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]:
        path, _evidence, _tree = self._champion_path(
            image,
            prompt,
            configuration,
        )
        return path


class _EfficientSAMTracePipeline:
    """Official split EfficientSAM-Ti ONNX plus the product SAM trace policy."""

    def __init__(
        self,
        backend: str,
        threads: int,
        model_cache: Path,
        np_module: Any,
        cv2_module: Any,
        *,
        opencv_effective_num_threads: int,
        opencv_opencl_enabled: bool,
    ):
        import ai_vectorizer as ai_vectorizer_package
        from ai_vectorizer import core as core_package
        from ai_vectorizer.core import dependencies as dependencies_module
        from ai_vectorizer.core import edge_detector as edge_detector_module
        from ai_vectorizer.core import efficientsam_onnx
        from ai_vectorizer.core import efficientsam_spec
        from ai_vectorizer.core import model_store
        from ai_vectorizer.core import sam_trace_kernel
        from ai_vectorizer.core import trace_kernel
        import benchmarks as benchmarks_package

        if backend != EFFICIENTSAM_BACKEND:
            raise BackendUnavailableError(f"unsupported EfficientSAM backend: {backend}")
        if threads != 1:
            raise WorkerRequestError("EfficientSAM benchmark workers require threads=1")
        if edge_detector_module._skimage_skeletonize is None:
            raise WorkerDependencyError(
                "controlled EfficientSAM tracing requires scikit-image skeletonize"
            )

        self._np = np_module
        self._cv2 = cv2_module
        self._edge_detector_module = edge_detector_module
        self._sam_trace_kernel = sam_trace_kernel
        self._sam_trace_config = sam_trace_kernel.SamTraceConfig(
            max_dimension=EFFICIENTSAM_INPUT_DIMENSION
        )
        self._trace_kernel = trace_kernel
        self._edge_detector = edge_detector_module.EdgeDetector(method="canny")
        self._latest_prediction_evidence: dict[str, Any] | None = None

        bundle = model_store.resolve_bundle(
            model_cache,
            efficientsam_spec.EFFICIENTSAM_TI_SPLIT,
        )
        encoder_bytes = bundle.read_bytes("encoder")
        decoder_bytes = bundle.read_bytes("decoder")
        self._engine = efficientsam_onnx.EfficientSAMOnnxEngine(
            encoder_bytes,
            decoder_bytes,
            threads=threads,
        )
        engine_metadata = dict(self._engine.metadata)
        providers = engine_metadata.get("providers", {})
        provider_verified = (
            isinstance(providers, dict)
            and providers.get("encoder") == ["CPUExecutionProvider"]
            and providers.get("decoder") == ["CPUExecutionProvider"]
        )
        session_options = engine_metadata.get("session_options", {})
        if not isinstance(session_options, dict):
            session_options = {}
        session_options_by_session = engine_metadata.get(
            "session_options_by_session"
        )

        artifact_hashes = {
            artifact.id: artifact.sha256
            for artifact in efficientsam_spec.EFFICIENTSAM_TI_SPLIT.artifacts
        }
        source_modules = (
            benchmarks_package,
            benchmark_evidence,
            benchmark_geometry,
            benchmark_manifest,
            ai_vectorizer_package,
            core_package,
            dependencies_module,
            edge_detector_module,
            efficientsam_onnx,
            efficientsam_spec,
            model_store,
            sam_trace_kernel,
            trace_kernel,
        )
        repository_root = Path(__file__).resolve().parents[1]
        source_hashes = {"benchmarks/worker.py": _sha256_file(Path(__file__))}
        for module in source_modules:
            module_path = Path(module.__file__).resolve()
            try:
                label = module_path.relative_to(repository_root).as_posix()
            except ValueError:
                label = module_path.name
            source_hashes[label] = _sha256_file(module_path)

        runtime_details = {
            "onnx_providers": providers,
            "onnx_session_options": session_options_by_session,
            "encoder_reused_across_predictions": True,
            "model_source_commit": efficientsam_spec.EFFICIENTSAM_TI_SPLIT.source_commit,
        }
        initialization_ms = engine_metadata.get("timing_ms", {}).get(
            "session_initialization"
        ) if isinstance(engine_metadata.get("timing_ms"), dict) else None
        if isinstance(initialization_ms, (int, float)):
            runtime_details["session_initialization_ns"] = max(
                0, int(float(initialization_ms) * 1_000_000)
            )

        self.info = BackendInfo(
            actual_backend=backend,
            provider_kind="onnxruntime",
            actual_provider="CPUExecutionProvider",
            provider_device_type="cpu",
            adapter_version=EFFICIENTSAM_ADAPTER_VERSION,
            package_versions={
                "numpy": str(getattr(np_module, "__version__", "unknown")),
                "opencv": str(getattr(cv2_module, "__version__", "unknown")),
                "onnxruntime": str(engine_metadata.get("onnxruntime_version", "unknown")),
                "scikit-image": _distribution_version("scikit-image"),
            },
            thread_settings={
                "threads": threads,
                "onnx_intra_op_num_threads": session_options.get(
                    "intra_op_num_threads"
                ),
                "onnx_inter_op_num_threads": session_options.get(
                    "inter_op_num_threads"
                ),
                "onnx_execution_mode": session_options.get("execution_mode"),
                "onnx_graph_optimization_level": session_options.get(
                    "graph_optimization_level"
                ),
                "opencv_set_num_threads": 0,
                "opencv_effective_num_threads": opencv_effective_num_threads,
                "opencl": opencv_opencl_enabled,
                **{variable: os.environ.get(variable) for variable in CPU_THREAD_VARIABLES},
            },
            provider_verified=provider_verified,
            source_files_sha256=source_hashes,
            model_artifacts_sha256=artifact_hashes,
            model_bundle_id=efficientsam_spec.EFFICIENTSAM_TI_SPLIT.id,
            model_bundle_sha256=model_store.bundle_fingerprint(
                efficientsam_spec.EFFICIENTSAM_TI_SPLIT
            ),
            runtime_details=runtime_details,
        )

    def load_image(self, path: Path, width: int, height: int) -> Any:
        if (width, height) != (1024, 1024):
            raise PredictionError(
                "M1.2 EfficientSAM benchmark inputs must be exactly 1024x1024"
            )
        decode_started = time.perf_counter_ns()
        image = self._cv2.imread(str(path), self._cv2.IMREAD_UNCHANGED)
        if image is None:
            raise PredictionError(f"OpenCV could not decode {path.name}")
        if image.dtype != self._np.uint8 or tuple(image.shape[:2]) != (height, width):
            raise PredictionError("EfficientSAM input must decode as 1024x1024 uint8")
        if image.ndim == 2:
            rgb = self._cv2.cvtColor(image, self._cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            rgb = self._cv2.cvtColor(image[..., 0], self._cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGRA2RGB)
        else:
            raise PredictionError("EfficientSAM input must be gray, RGB, or RGBA")
        rgb = self._np.ascontiguousarray(rgb)
        decoded_ns = max(0, time.perf_counter_ns() - decode_started)

        encode_started = time.perf_counter_ns()
        encoding = self._engine.encode(rgb)
        encoded_ns = max(0, time.perf_counter_ns() - encode_started)
        self.info.runtime_details.update(
            image_file_decode_ns=decoded_ns,
            image_encode_wall_ns=encoded_ns,
        )
        return {"rgb": rgb, "encoding": encoding, "edges": None}

    def predict(
        self,
        image: Any,
        prompt: TracePrompt,
        configuration: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]:
        low, high = _canny_thresholds(configuration)
        if image["edges"] is None:
            edge_started = time.perf_counter_ns()
            image["edges"] = self._edge_detector.detect_edges(
                image["rgb"],
                low_threshold=low,
                high_threshold=high,
            )
            self.info.runtime_details["edge_cache_fill_wall_ns"] = max(
                0,
                time.perf_counter_ns() - edge_started,
            )
        positive_points = [prompt.start_xy, *prompt.positive_xy, prompt.end_xy]
        points = [*positive_points, *prompt.negative_xy]
        labels = [1] * len(positive_points) + [0] * len(prompt.negative_xy)
        if not 2 <= len(points) <= 6:
            raise PredictionError("EfficientSAM requires two to six prompt points")

        prediction = self._engine.predict(image["encoding"], points, labels)
        processed = self._sam_trace_kernel.postprocess_mask(
            prediction.mask,
            cv2_module=self._cv2,
            np_module=self._np,
            config=self._sam_trace_config,
        )
        if processed is None:
            raise PredictionError(
                "EfficientSAM mask failed the product area/size acceptance policy"
            )
        metadata = prediction.metadata
        timing = metadata.get("timing_ms") if isinstance(metadata, Mapping) else None
        decoder_ms = timing.get("decoder") if isinstance(timing, Mapping) else None
        if (
            isinstance(decoder_ms, bool)
            or not isinstance(decoder_ms, (int, float))
            or not math.isfinite(float(decoder_ms))
            or float(decoder_ms) < 0.0
        ):
            raise PredictionError(
                "EfficientSAM decoder did not expose finite timing evidence"
            )

        float32_dtype = self._np.dtype("<f4")
        iou_array = self._np.ascontiguousarray(
            self._np.asarray(prediction.iou_predictions, dtype=float32_dtype)
        )
        if tuple(iou_array.shape) != (3,) or not bool(
            self._np.isfinite(iou_array).all()
        ):
            raise PredictionError(
                "EfficientSAM decoder returned invalid IoU prediction evidence"
            )
        selected_index = int(prediction.selected_index)
        if selected_index != int(self._np.argmax(iou_array)):
            raise PredictionError(
                "EfficientSAM selected mask disagrees with maximum predicted IoU"
            )
        selected_logits = self._np.ascontiguousarray(
            self._np.asarray(prediction.selected_logits, dtype=float32_dtype)
        )
        selected_mask = self._np.ascontiguousarray(
            self._np.asarray(prediction.mask, dtype=self._np.uint8)
        )
        accepted_mask = self._np.ascontiguousarray(
            self._np.asarray(processed, dtype=self._np.uint8)
        )
        expected_shape = (EFFICIENTSAM_INPUT_DIMENSION, EFFICIENTSAM_INPUT_DIMENSION)
        if any(
            tuple(array.shape) != expected_shape
            for array in (selected_logits, selected_mask, accepted_mask)
        ):
            raise PredictionError(
                "EfficientSAM mask evidence changed the fixed 1024x1024 contract"
            )
        if not bool(self._np.isfinite(selected_logits).all()):
            raise PredictionError(
                "EfficientSAM selected logits evidence must be finite"
            )
        self._latest_prediction_evidence = {
            "schema_version": EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION,
            "selected_mask_index": selected_index,
            "iou_predictions": [float(value) for value in iou_array.tolist()],
            "iou_predictions_sha256": _sha256_bytes(iou_array.tobytes(order="C")),
            "selected_logits_sha256": _sha256_bytes(
                selected_logits.tobytes(order="C")
            ),
            "selected_binary_mask_sha256": _sha256_bytes(
                selected_mask.tobytes(order="C")
            ),
            "accepted_mask_sha256": _sha256_bytes(
                accepted_mask.tobytes(order="C")
            ),
            "decoder_wall_ns": max(0, int(float(decoder_ms) * 1_000_000)),
        }

        return self._sam_trace_kernel.trace_mask_centerline(
            processed,
            image["edges"],
            prompt.start_xy,
            prompt.end_xy,
            trace_kernel=self._trace_kernel,
            cv2_module=self._cv2,
            np_module=self._np,
            thin_binary_mask=self._edge_detector_module.EdgeDetector.thin_binary_mask,
            config=self._sam_trace_config,
        )

    def prediction_evidence(self) -> Mapping[str, Any]:
        if self._latest_prediction_evidence is None:
            raise PredictionError(
                "EfficientSAM prediction evidence was requested before inference"
            )
        return {
            key: ([*value] if key == "iou_predictions" else value)
            for key, value in self._latest_prediction_evidence.items()
        }


class _InkV2EfficientSAMRecoveryPipeline(_InkLiveWirePipeline):
    """Conditional Ink-v2 champion with a pinned EfficientSAM challenger."""

    def __init__(
        self,
        backend: str,
        threads: int,
        model_cache: Path,
        np_module: Any,
        cv2_module: Any,
        *,
        opencv_effective_num_threads: int,
        opencv_opencl_enabled: bool,
    ):
        if backend != INK_V2_EFFICIENTSAM_RECOVERY_BACKEND:
            raise BackendUnavailableError(f"unsupported recovery backend: {backend}")
        super().__init__(
            INK_LIVEWIRE_V2_BACKEND,
            threads,
            np_module,
            cv2_module,
            opencv_effective_num_threads=opencv_effective_num_threads,
            opencv_opencl_enabled=opencv_opencl_enabled,
        )

        from ai_vectorizer.core import efficientsam_onnx
        from ai_vectorizer.core import efficientsam_spec
        from ai_vectorizer.core import model_store
        from ai_vectorizer.core import recovery_prompts
        from ai_vectorizer.core import sam_trace_kernel
        from ai_vectorizer.core import smart_recovery
        from ai_vectorizer.core import trace_kernel

        bundle = model_store.resolve_bundle(
            model_cache,
            efficientsam_spec.EFFICIENTSAM_TI_SPLIT,
        )
        self._engine = efficientsam_onnx.EfficientSAMOnnxEngine(
            bundle.read_bytes("encoder"),
            bundle.read_bytes("decoder"),
            threads=threads,
        )
        self._recovery_prompts = recovery_prompts
        self._sam_trace_kernel = sam_trace_kernel
        self._smart_recovery = smart_recovery
        self._recovery_trace_kernel = trace_kernel
        self._latest_prediction_evidence: dict[str, Any] | None = None

        engine_metadata = dict(self._engine.metadata)
        providers = engine_metadata.get("providers", {})
        provider_verified = (
            isinstance(providers, dict)
            and providers.get("encoder") == ["CPUExecutionProvider"]
            and providers.get("decoder") == ["CPUExecutionProvider"]
        )
        session_options = engine_metadata.get("session_options", {})
        if not isinstance(session_options, dict):
            session_options = {}
        artifact_hashes = {
            artifact.id: artifact.sha256
            for artifact in efficientsam_spec.EFFICIENTSAM_TI_SPLIT.artifacts
        }
        source_hashes = dict(self.info.source_files_sha256)
        repository_root = Path(__file__).resolve().parents[1]
        for module in (
            efficientsam_onnx,
            efficientsam_spec,
            model_store,
            recovery_prompts,
            sam_trace_kernel,
            smart_recovery,
            trace_kernel,
        ):
            module_path = Path(module.__file__).resolve()
            try:
                label = module_path.relative_to(repository_root).as_posix()
            except ValueError:
                label = module_path.name
            source_hashes[label] = _sha256_file(module_path)

        runtime_details = dict(self.info.runtime_details)
        runtime_details.update(
            onnx_providers=providers,
            onnx_session_options=engine_metadata.get("session_options_by_session"),
            encoder_reused_across_predictions=True,
            encoder_execution="conditional_after_recovery_gate",
            model_source_commit=efficientsam_spec.EFFICIENTSAM_TI_SPLIT.source_commit,
            recovery_policy_id=smart_recovery.RECOVERY_POLICY_ID,
            recovery_configuration_sha256=(
                smart_recovery.DEFAULT_RECOVERY_CONFIG.sha256
            ),
            recovery_provisional=True,
        )
        initialization_ms = (
            engine_metadata.get("timing_ms", {}).get("session_initialization")
            if isinstance(engine_metadata.get("timing_ms"), dict)
            else None
        )
        if isinstance(initialization_ms, (int, float)):
            runtime_details["session_initialization_ns"] = max(
                0, int(float(initialization_ms) * 1_000_000)
            )

        self.info = BackendInfo(
            actual_backend=backend,
            provider_kind="onnxruntime",
            actual_provider="CPUExecutionProvider",
            provider_device_type="cpu",
            adapter_version=RECOVERY_ADAPTER_VERSION,
            package_versions={
                **dict(self.info.package_versions),
                "onnxruntime": str(
                    engine_metadata.get("onnxruntime_version", "unknown")
                ),
            },
            thread_settings={
                "threads": threads,
                "onnx_intra_op_num_threads": session_options.get(
                    "intra_op_num_threads"
                ),
                "onnx_inter_op_num_threads": session_options.get(
                    "inter_op_num_threads"
                ),
                "onnx_execution_mode": session_options.get("execution_mode"),
                "onnx_graph_optimization_level": session_options.get(
                    "graph_optimization_level"
                ),
                "opencv_set_num_threads": 0,
                "opencv_effective_num_threads": opencv_effective_num_threads,
                "opencl": opencv_opencl_enabled,
                **{variable: os.environ.get(variable) for variable in CPU_THREAD_VARIABLES},
            },
            provider_verified=provider_verified,
            source_files_sha256=source_hashes,
            model_artifacts_sha256=artifact_hashes,
            model_bundle_id=efficientsam_spec.EFFICIENTSAM_TI_SPLIT.id,
            model_bundle_sha256=model_store.bundle_fingerprint(
                efficientsam_spec.EFFICIENTSAM_TI_SPLIT
            ),
            runtime_details=runtime_details,
        )

    def load_image(self, path: Path, width: int, height: int) -> Any:
        image = super().load_image(path, width, height)
        image["encoding"] = None
        return image

    def _model_rgb(self, image: Any) -> Any:
        rgb = self._np.asarray(image["rgb"])
        if rgb.ndim == 2:
            rgb = self._np.repeat(rgb[..., None], 3, axis=2)
        elif rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise PredictionError("Smart Recovery model input must be RGB")
        return self._np.ascontiguousarray(rgb, dtype=self._np.uint8)

    def _segmentation_evidence(self, prediction: Any) -> dict[str, Any]:
        metadata = prediction.metadata
        timing = metadata.get("timing_ms") if isinstance(metadata, Mapping) else None
        decoder_ms = timing.get("decoder") if isinstance(timing, Mapping) else None
        if (
            isinstance(decoder_ms, bool)
            or not isinstance(decoder_ms, (int, float))
            or not math.isfinite(float(decoder_ms))
            or float(decoder_ms) < 0.0
        ):
            raise PredictionError("EfficientSAM decoder omitted finite timing evidence")
        float32_dtype = self._np.dtype("<f4")
        iou_array = self._np.ascontiguousarray(
            self._np.asarray(prediction.iou_predictions, dtype=float32_dtype)
        )
        selected_logits = self._np.ascontiguousarray(
            self._np.asarray(prediction.selected_logits, dtype=float32_dtype)
        )
        selected_mask = self._np.ascontiguousarray(
            self._np.asarray(prediction.mask, dtype=self._np.uint8)
        )
        if tuple(iou_array.shape) != (3,) or not bool(self._np.isfinite(iou_array).all()):
            raise PredictionError("EfficientSAM returned invalid IoU evidence")
        if not bool(self._np.isfinite(selected_logits).all()):
            raise PredictionError("EfficientSAM returned non-finite logits")
        if selected_logits.shape != selected_mask.shape:
            raise PredictionError("EfficientSAM mask/logit evidence dimensions disagree")
        selected_index = int(prediction.selected_index)
        if selected_index != int(self._np.argmax(iou_array)):
            raise PredictionError("EfficientSAM selected mask disagrees with predicted IoU")
        return {
            "schema_version": EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION,
            "selected_mask_index": selected_index,
            "iou_predictions": [float(value) for value in iou_array.tolist()],
            "iou_predictions_sha256": _sha256_bytes(iou_array.tobytes(order="C")),
            "selected_logits_sha256": _sha256_bytes(
                selected_logits.tobytes(order="C")
            ),
            "selected_binary_mask_sha256": _sha256_bytes(
                selected_mask.tobytes(order="C")
            ),
            "accepted_mask_sha256": _sha256_bytes(
                selected_mask.tobytes(order="C")
            ),
            "decoder_wall_ns": max(0, int(float(decoder_ms) * 1_000_000)),
        }

    def _bounded_recovery_inputs(
        self,
        evidence: Any,
        corridor: Any,
        champion_tree: Any,
        prompt: TracePrompt,
    ) -> tuple[Any, Any, tuple[float, float], tuple[float, float], tuple[int, int]]:
        """Crop recovery to the champion Live-Wire tree's exact pixel window."""

        from ai_vectorizer.core.line_evidence import LineEvidence

        try:
            origin_x, origin_y = (int(value) for value in champion_tree.origin)
            height, width = (int(value) for value in champion_tree.shape)
        except (AttributeError, TypeError, ValueError) as exc:
            raise PredictionError(
                "Smart Recovery champion omitted its bounded Live-Wire window"
            ) from exc
        if (
            height < 1
            or width < 1
            or height > 320
            or width > 320
            or origin_x < 0
            or origin_y < 0
        ):
            raise PredictionError("Smart Recovery champion window is outside the 320px contract")

        score_shape = tuple(self._np.asarray(evidence.center_score).shape)
        corridor_array = self._np.asarray(corridor, dtype=self._np.float32)
        if len(score_shape) != 2 or tuple(corridor_array.shape) != score_shape:
            raise PredictionError(
                "Smart Recovery corridor must match the full Ink evidence grid"
            )
        end_x = origin_x + width
        end_y = origin_y + height
        if end_x > score_shape[1] or end_y > score_shape[0]:
            raise PredictionError("Smart Recovery champion window leaves the evidence grid")

        local_start = (
            float(prompt.start_xy[0]) - origin_x,
            float(prompt.start_xy[1]) - origin_y,
        )
        local_end = (
            float(prompt.end_xy[0]) - origin_x,
            float(prompt.end_xy[1]) - origin_y,
        )
        for label, (x, y) in (("start", local_start), ("end", local_end)):
            if not (0.0 <= x < width and 0.0 <= y < height):
                raise PredictionError(
                    f"Smart Recovery {label} lies outside the champion Live-Wire window"
                )

        window = (slice(origin_y, end_y), slice(origin_x, end_x))
        bounded_evidence = LineEvidence(
            center_score=evidence.center_score[window],
            centerline=evidence.centerline[window],
            tangent_x=evidence.tangent_x[window],
            tangent_y=evidence.tangent_y[window],
            coherence=evidence.coherence[window],
            scale_px=evidence.scale_px[window],
        )
        bounded_corridor = self._np.ascontiguousarray(corridor_array[window])
        return (
            bounded_evidence,
            bounded_corridor,
            local_start,
            local_end,
            (origin_x, origin_y),
        )

    def predict(
        self,
        image: Any,
        prompt: TracePrompt,
        configuration: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]:
        recovery_config = _recovery_contract(configuration)
        champion, evidence, champion_tree = self._champion_path(
            image,
            prompt,
            configuration,
        )
        if evidence is None:
            raise PredictionError("Smart Recovery requires explicit LineEvidence")
        gate = self._smart_recovery.recovery_gate(
            champion,
            evidence,
            expected_start=prompt.start_xy,
            expected_end=prompt.end_xy,
            config=recovery_config,
        )
        record: dict[str, Any] = {
            "schema_version": RECOVERY_EVIDENCE_SCHEMA_VERSION,
            "policy_id": gate.policy_id,
            "configuration_sha256": gate.configuration_sha256,
            "provisional": True,
            "gate": {
                "trigger": bool(gate.trigger),
                "reason": gate.reason,
                "metrics": gate.quality.as_dict(),
            },
            "champion_sha256": _route_sha256(champion),
            "challenger_sha256": None,
            "selected_route": "champion",
            "selection": None,
            "fallback_reason": None,
            "rejection_reason": "recovery_not_triggered",
            "segmentation_evidence": None,
        }
        if not gate.trigger:
            self._latest_prediction_evidence = record
            return champion

        # A model call is forbidden before this point. The encoder is cached
        # only after the product Ink quality gate explicitly permits recovery.
        record["rejection_reason"] = None
        try:
            if image["encoding"] is None:
                encode_started = time.perf_counter_ns()
                image["encoding"] = self._engine.encode(self._model_rgb(image))
                self.info.runtime_details["conditional_image_encode_wall_ns"] = max(
                    0, time.perf_counter_ns() - encode_started
                )
            recovery_prompts = self._recovery_prompts.build_recovery_prompt_tensors(
                prompt.start_xy,
                prompt.end_xy,
                width=int(image["rgb"].shape[1]),
                height=int(image["rgb"].shape[0]),
            )
            points, labels = recovery_prompts.as_numpy(self._np)
            prediction = self._engine.predict(image["encoding"], points, labels)
            segmentation_evidence = self._segmentation_evidence(prediction)
            corridor = self._np.asarray(prediction.mask, dtype=self._np.float32)
            (
                bounded_evidence,
                bounded_corridor,
                local_start,
                local_end,
                (origin_x, origin_y),
            ) = self._bounded_recovery_inputs(
                evidence,
                corridor,
                champion_tree,
                prompt,
            )
            cost_map = self._smart_recovery.build_corridor_cost_map(
                bounded_evidence,
                bounded_corridor,
                config=recovery_config,
            )
            window_height, window_width = cost_map.shape
            trace_config = self._recovery_trace_kernel.TraceConfig(
                max_width=int(window_width),
                max_height=int(window_height),
                max_cells=int(window_width) * int(window_height),
                validate_all_costs=False,
                validate_accessed_costs=False,
            )
            trace = self._recovery_trace_kernel.trace_path(
                cost_map,
                local_start,
                local_end,
                allow_partial=False,
                config=trace_config,
            )
            if trace.status != "complete":
                raise PredictionError("Smart Recovery challenger did not reach the endpoint")
            challenger = [
                (float(x) + origin_x, float(y) + origin_y)
                for x, y in trace.points_xy
            ]
            # Product recovery displays a five-point-smoothed route.  Apply
            # that transformation before the safety arbiter and use this one
            # exact route for evidence, hashing, and output; a post-arbitration
            # geometry change could otherwise bypass endpoint/branch checks.
            if len(challenger) > 5:
                challenger = list(
                    self._recovery_trace_kernel.smooth_pixel_path(
                        challenger,
                        window_size=5,
                    )
                )
            challenger[0] = (
                float(prompt.start_xy[0]),
                float(prompt.start_xy[1]),
            )
            challenger[-1] = (
                float(prompt.end_xy[0]),
                float(prompt.end_xy[1]),
            )
            selection = self._smart_recovery.arbitrate_routes(
                champion,
                challenger,
                evidence,
                expected_start=prompt.start_xy,
                expected_end=prompt.end_xy,
                config=recovery_config,
            )
            separation = float(selection.route_separation_p95)
            record.update(
                challenger_sha256=_route_sha256(challenger),
                selected_route=selection.selected,
                selection={
                    "reason": selection.reason,
                    "champion_metrics": selection.champion_quality.as_dict(),
                    "challenger_metrics": (
                        selection.challenger_quality.as_dict()
                        if selection.challenger_quality is not None
                        else None
                    ),
                    "strong_ink_retention": float(selection.strong_ink_retention),
                    "route_separation_p95": (
                        separation if math.isfinite(separation) else None
                    ),
                },
                rejection_reason=(
                    None if selection.accepted else selection.reason
                ),
                segmentation_evidence=segmentation_evidence,
            )
            self._latest_prediction_evidence = record
            return challenger if selection.accepted else champion
        except Exception as exc:
            record.update(
                selected_route="champion",
                fallback_reason=_safe_error(exc),
                rejection_reason=None,
            )
            self._latest_prediction_evidence = record
            return champion

    def prediction_evidence(self) -> Mapping[str, Any]:
        if self._latest_prediction_evidence is None:
            raise PredictionError(
                "Smart Recovery evidence was requested before prediction"
            )
        return dict(self._latest_prediction_evidence)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_opencv_pipeline(backend: str, threads: int) -> LoadedPipeline:
    """Lazy-load the real OpenCV adapter without importing QGIS."""

    try:
        import numpy as np
    except Exception as exc:
        raise WorkerDependencyError(
            "Canny/LSD benchmark workers require NumPy in the worker environment"
        ) from exc
    try:
        import cv2
    except Exception as exc:
        raise WorkerDependencyError(
            "Canny/LSD benchmark workers require OpenCV (cv2) in the worker environment"
        ) from exc

    if backend not in METHOD_EDGE_BACKENDS:
        raise BackendUnavailableError(f"unsupported detector backend: {backend}")
    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(0)
    ocl = getattr(cv2, "ocl", None)
    if ocl is not None and hasattr(ocl, "setUseOpenCL"):
        ocl.setUseOpenCL(False)
    if ocl is not None and hasattr(ocl, "useOpenCL") and ocl.useOpenCL():
        raise WorkerDependencyError("OpenCV OpenCL could not be disabled for the CPU benchmark")
    return _OpenCVTracePipeline(backend, threads, np, cv2)


def _configure_efficientsam_opencv(
    cv2_module: Any,
    threads: int,
) -> tuple[int, bool]:
    """Apply and read back the deterministic OpenCV state used by SAM tracing."""

    set_num_threads = getattr(cv2_module, "setNumThreads", None)
    get_num_threads = getattr(cv2_module, "getNumThreads", None)
    if not callable(set_num_threads) or not callable(get_num_threads):
        raise WorkerDependencyError(
            "EfficientSAM tracing requires observable OpenCV thread controls"
        )
    try:
        set_num_threads(0)
        effective_threads = get_num_threads()
    except Exception as exc:
        raise WorkerDependencyError(
            "OpenCV thread state could not be configured and read back"
        ) from exc
    if (
        isinstance(effective_threads, bool)
        or not isinstance(effective_threads, int)
        or effective_threads != threads
    ):
        raise WorkerDependencyError(
            "OpenCV thread readback did not attest the requested CPU thread count"
        )

    ocl = getattr(cv2_module, "ocl", None)
    set_use_opencl = getattr(ocl, "setUseOpenCL", None)
    use_opencl = getattr(ocl, "useOpenCL", None)
    if not callable(set_use_opencl) or not callable(use_opencl):
        raise WorkerDependencyError(
            "EfficientSAM tracing requires observable OpenCV OpenCL controls"
        )
    try:
        set_use_opencl(False)
        opencl_enabled = bool(use_opencl())
    except Exception as exc:
        raise WorkerDependencyError(
            "OpenCV OpenCL state could not be disabled and read back"
        ) from exc
    if opencl_enabled:
        raise WorkerDependencyError("OpenCV OpenCL could not be disabled")
    return int(effective_threads), opencl_enabled


def _load_efficientsam_pipeline(
    backend: str,
    threads: int,
    model_cache: Path | None,
) -> LoadedPipeline:
    """Lazy-load the pinned ONNX adapter without importing QGIS."""

    if backend not in METHOD_SAM_BACKENDS:
        raise BackendUnavailableError(f"unsupported EfficientSAM backend: {backend}")
    if model_cache is None:
        raise WorkerRequestError("EfficientSAM requires a verified model cache")
    try:
        import numpy as np
    except Exception as exc:
        raise WorkerDependencyError(
            "EfficientSAM benchmark workers require NumPy"
        ) from exc
    try:
        import cv2
    except Exception as exc:
        raise WorkerDependencyError(
            "EfficientSAM tracing requires OpenCV (cv2)"
        ) from exc

    opencv_effective_num_threads, opencv_opencl_enabled = (
        _configure_efficientsam_opencv(cv2, threads)
    )
    pipeline_class = (
        _InkV2EfficientSAMRecoveryPipeline
        if backend in METHOD_RECOVERY_BACKENDS
        else _EfficientSAMTracePipeline
    )
    return pipeline_class(
        backend,
        threads,
        model_cache,
        np,
        cv2,
        opencv_effective_num_threads=opencv_effective_num_threads,
        opencv_opencl_enabled=opencv_opencl_enabled,
    )


def _load_ink_pipeline(backend: str, threads: int) -> LoadedPipeline:
    """Lazy-load the CPU Ink + Live-Wire adapter without importing QGIS."""

    if backend not in METHOD_INK_BACKENDS:
        raise BackendUnavailableError(f"unsupported Ink backend: {backend}")
    try:
        import numpy as np
    except Exception as exc:
        raise WorkerDependencyError("Ink Live-Wire workers require NumPy") from exc
    try:
        import cv2
    except Exception as exc:
        raise WorkerDependencyError("Ink Live-Wire workers require OpenCV") from exc
    effective_threads, opencl_enabled = _configure_efficientsam_opencv(cv2, threads)
    return _InkLiveWirePipeline(
        backend,
        threads,
        np,
        cv2,
        opencv_effective_num_threads=effective_threads,
        opencv_opencl_enabled=opencl_enabled,
    )


def _load_pipeline(
    backend: str,
    threads: int,
    *,
    model_cache: Path | None,
) -> LoadedPipeline:
    if backend in METHOD_EDGE_BACKENDS:
        return _load_opencv_pipeline(backend, threads)
    if backend in METHOD_INK_BACKENDS:
        return _load_ink_pipeline(backend, threads)
    if backend in METHOD_SAM_BACKENDS:
        return _load_efficientsam_pipeline(backend, threads, model_cache)
    raise BackendUnavailableError(f"unsupported benchmark backend: {backend}")


def write_worker_result(path: str | Path, result: Mapping[str, Any]) -> Path:
    """Atomically persist a JSON worker result for manifest assembly."""

    target = Path(path).resolve()
    raw = (
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(target, raw)
    return target


def run_request_file(
    request_path: str | Path,
    result_path: str | Path,
    *,
    pipeline_loader: PipelineLoader | None = None,
) -> dict[str, Any]:
    """Load, execute, and atomically record one worker request."""

    request_file = Path(request_path).resolve()
    result_file = Path(result_path).resolve()
    request = load_worker_request(request_file)
    if result_file in {request_file, request.image_path, request.artifact_path}:
        raise WorkerRequestError(
            "worker result must not overwrite the request, input image, or artifact"
        )
    result = run_worker(request, pipeline_loader=pipeline_loader)
    write_worker_result(result_file, result)
    return result


def run_isolated_worker(
    request_path: str | Path,
    result_path: str | Path,
    *,
    python_executable: str | None = None,
    timeout_seconds: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    """Launch one fresh Python process for one method/sample prediction."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    return subprocess.run(
        [
            python_executable or sys.executable,
            "-m",
            "benchmarks.worker",
            str(Path(request_path).resolve()),
            str(Path(result_path).resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        cwd=Path(__file__).resolve().parents[1],
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("usage: python3 -m benchmarks.worker REQUEST.json RESULT.json", file=sys.stderr)
        return 2
    try:
        result = run_request_file(arguments[0], arguments[1])
    except (WorkerError, OSError, ValueError) as exc:
        print(_safe_error(exc), file=sys.stderr)
        return 2
    return 0 if result["prediction"]["execution"]["status"] != "failed" else 3


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess.
    raise SystemExit(main())


__all__ = [
    "BackendInfo",
    "BackendUnavailableError",
    "EFFICIENTSAM_BACKEND",
    "INK_LIVEWIRE_V1_BACKEND",
    "INK_LIVEWIRE_V2_BACKEND",
    "INK_V2_EFFICIENTSAM_RECOVERY_BACKEND",
    "LoadedPipeline",
    "METHOD_EDGE_BACKENDS",
    "METHOD_INK_BACKENDS",
    "METHOD_RECOVERY_BACKENDS",
    "METHOD_SAM_BACKENDS",
    "METHOD_SOURCE_GRID_BACKENDS",
    "PredictionError",
    "PRODUCT_SMOOTHING_PROFILE",
    "TracePrompt",
    "WORKER_REQUEST_SCHEMA_VERSION",
    "WORKER_REQUEST_SCHEMA_VERSION_V1",
    "WORKER_REQUEST_SCHEMA_VERSIONS",
    "WORKER_RESULT_SCHEMA_VERSION",
    "SUPPORTED_BACKENDS",
    "WorkerDependencyError",
    "WorkerError",
    "WorkerRequest",
    "WorkerRequestError",
    "load_worker_request",
    "main",
    "run_isolated_worker",
    "run_request_file",
    "run_worker",
    "write_worker_result",
]
