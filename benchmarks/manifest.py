"""Strict, checksummed benchmark manifest loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any
import zlib

from ai_vectorizer.core.efficientsam_spec import (
    EFFICIENTSAM_TI_SPLIT,
    bundle_fingerprint,
)

from .evidence import (
    PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
    PROMPT_EVIDENCE_SCHEMA_VERSIONS,
    prompt_sha256,
    recovery_prompt_tensor_sha256,
    sam_prompt_tensor_sha256,
    source_grid_input_sha256,
)
from .geometry import MAX_ARTIFACT_BYTES, load_centerline_artifact


MANIFEST_SCHEMA_VERSION = "archaeotrace-contour-benchmark/1"
METHOD_KIND_PRECOMPUTED = "precomputed_centerline"
REQUIRED_STRATA = ("map_type", "print_state", "scan_quality")
IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_CANVAS_DIMENSION = 1024
MAX_CANVAS_PIXELS = 1024 * 1024
MAX_METHODS = 32
MAX_SAMPLES = 1_000
MAX_EVALUATIONS = 256
MAX_TOLERANCES = 16
MAX_TOLERANCE_PX = 8.0
MAX_TIMING_REPETITIONS = 100
MAX_COUNTER_VALUE = (1 << 63) - 1
MAX_IMAGE_HEADER_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
CPU_PROVIDER_NAMES = {
    "synthetic": "synthetic CPU",
    "opencv": "OpenCV CPU",
    "onnxruntime": "CPUExecutionProvider",
    "python": "Python CPU",
    "pytorch": "PyTorch CPU",
}
SHARED_THREAD_SETTING_KEYS = (
    "threads",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
EFFICIENTSAM_METHOD_ID = "efficientsam-ti-onnx-v1"
RECOVERY_METHOD_ID = "ink-v2-effsam-recovery-v1"
INK_V2_METHOD_ID = "ink-livewire-v2"
SOURCE_GRID_METHOD_IDS = frozenset({INK_V2_METHOD_ID, RECOVERY_METHOD_ID})
EFFICIENTSAM_INPUT_DIMENSION = 1024
EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION = (
    "archaeotrace-efficientsam-prediction-evidence/1"
)
WORKER_LATENCY_SCOPE = "warmed_predict_plus_canonical_artifact_v1"
PRODUCT_SMOOTHING_PROFILE = "smart-trace-v1-historical"
EFFICIENTSAM_BUNDLE_SHA256 = bundle_fingerprint(EFFICIENTSAM_TI_SPLIT)
EFFICIENTSAM_ARTIFACTS_SHA256 = {
    artifact.id: artifact.sha256 for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
}
EFFICIENTSAM_ORT_SESSION_OPTIONS = {
    "intra_op_num_threads": 1,
    "inter_op_num_threads": 1,
    "execution_mode": "ORT_SEQUENTIAL",
    "graph_optimization_level": "ORT_ENABLE_ALL",
}
EFFICIENTSAM_ARTIFACT_METADATA_KEYS = frozenset(
    {
        "actual_backend",
        "configuration_sha256",
        "input_sha256",
        "mask_trace_kernel",
        "model_bundle_id",
        "model_bundle_sha256",
        "model_source_commit",
        "prompt_sha256",
        "requested_backend",
        "sam_prompt_tensor_sha256",
        "segmentation_evidence",
        "smoothing",
        "trace_kernel",
    }
)
RECOVERY_ARTIFACT_METADATA_KEYS = frozenset(
    {
        "actual_backend",
        "configuration_sha256",
        "input_sha256",
        "livewire_kernel",
        "mask_trace_kernel",
        "model_bundle_id",
        "model_bundle_sha256",
        "model_source_commit",
        "prompt_sha256",
        "recovery_evidence",
        "recovery_kernel",
        "requested_backend",
        "sam_prompt_tensor_sha256",
        "source_grid_input_sha256",
        "source_tile_origin_xy",
        "smoothing",
        "trace_kernel",
    }
)
INK_V2_ARTIFACT_METADATA_KEYS = frozenset(
    {
        "actual_backend",
        "configuration_sha256",
        "input_sha256",
        "prompt_sha256",
        "requested_backend",
        "smoothing",
        "source_grid_input_sha256",
        "source_tile_origin_xy",
        "trace_kernel",
    }
)
SAM_PREDICTION_EVIDENCE_KEYS = frozenset(
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


class ManifestError(ValueError):
    """Raised when a benchmark manifest is incomplete or unsafe."""


@dataclass(frozen=True)
class MetricConfig:
    primary_tolerance_px: float
    tolerances_px: tuple[float, ...]
    branch_tolerance_px: float
    connectivity: int
    diagonal_rule: str
    percentile: str


@dataclass(frozen=True)
class DatasetInfo:
    identifier: str
    version: str
    description: str
    license: str
    source: str


@dataclass(frozen=True)
class MethodSpec:
    identifier: str
    label: str
    kind: str
    source: str
    version: str
    license: str
    model_sha256: str | None
    configuration: dict[str, Any]


@dataclass(frozen=True)
class TimingRecord:
    warmup_runs: int
    wall_ns_samples: tuple[int, ...]
    cpu_ns_samples: tuple[int, ...]
    model_load_wall_ns: int | None
    peak_rss_bytes: int | None


@dataclass(frozen=True)
class ExecutionRecord:
    status: str
    requested_backend: str
    actual_backend: str | None
    fallback_reason: str | None
    error: str | None
    device: str
    runtime: dict[str, Any]
    timing: TimingRecord


@dataclass(frozen=True)
class PredictionSpec:
    artifact_path: Path | None
    artifact_sha256: str | None
    execution: ExecutionRecord


@dataclass(frozen=True)
class PromptSpec:
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    positive_xy: tuple[tuple[float, float], ...]
    negative_xy: tuple[tuple[float, float], ...]
    previous_xy: tuple[float, float] | None = None
    schema_version: str = PROMPT_EVIDENCE_SCHEMA_VERSION_V1


@dataclass(frozen=True)
class SampleSpec:
    identifier: str
    width: int
    height: int
    image_path: Path
    image_sha256: str
    reference_path: Path
    reference_sha256: str
    prompt: PromptSpec
    strata: dict[str, str]
    source: dict[str, str]
    predictions: dict[str, PredictionSpec]
    source_tile_origin_xy: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class BenchmarkManifest:
    path: Path
    sha256: str
    dataset: DatasetInfo
    metric_config: MetricConfig
    methods: tuple[MethodSpec, ...]
    samples: tuple[SampleSpec, ...]


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ManifestError(
        f"Non-finite or non-standard JSON number is not allowed: {value}."
    )


def _json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ManifestError(f"JSON number must be finite: {value}.")
    return number


def _json_integer(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise ManifestError("JSON integer exceeds the signed 64-bit limit.")
    number = int(value)
    if abs(number) > MAX_COUNTER_VALUE:
        raise ManifestError(
            f"JSON integer exceeds the signed 64-bit limit: {value[:32]}."
        )
    return number


def _validate_json_numbers(value: Any, label: str) -> None:
    """Reject exponent-overflow floats in arbitrary configuration/runtime data."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError(f"{label} must contain only finite numbers.")
        return
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ManifestError(f"{label} must not contain Unicode surrogate code points.")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise ManifestError(f"{label} must not contain ASCII control characters.")
        return
    if value is None or isinstance(value, (int, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_numbers(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_numbers(key, f"{label} key")
            _validate_json_numbers(item, f"{label}.{key}")
        return
    raise ManifestError(f"{label} contains an unsupported JSON value.")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object.")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be an array.")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string.")
    text = value.strip()
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in text):
        raise ManifestError(f"{label} must not contain ASCII control characters.")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in text):
        raise ManifestError(f"{label} must not contain Unicode surrogate code points.")
    return text


def _identifier(value: Any, label: str) -> str:
    identifier = _text(value, label)
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ManifestError(
            f"{label} must use lowercase letters, digits, dots, underscores, or hyphens."
        )
    return identifier


def _positive_integer(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_COUNTER_VALUE
    ):
        raise ManifestError(f"{label} must be a positive integer.")
    return value


def _nonnegative_integer(value: Any, label: str, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_COUNTER_VALUE
    ):
        raise ManifestError(f"{label} must be a non-negative integer.")
    return value


def _source_tile_origin(value: Any, label: str) -> tuple[int, int]:
    values = _array(value, label)
    if len(values) != 2:
        raise ManifestError(f"{label} must be an [x, y] pair.")
    return (
        _nonnegative_integer(values[0], f"{label}[0]"),
        _nonnegative_integer(values[1], f"{label}[1]"),
    )


def _positive_number(value: Any, label: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{label} must be a number.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ManifestError(f"{label} must be a finite number.") from exc
    minimum_ok = number >= 0 if allow_zero else number > 0
    if not math.isfinite(number) or not minimum_ok:
        relation = "non-negative" if allow_zero else "greater than zero"
        raise ManifestError(f"{label} must be finite and {relation}.")
    return number


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label).lower()
    if not SHA256_PATTERN.fullmatch(digest):
        raise ManifestError(f"{label} must be a 64-character SHA-256 digest.")
    return digest


def _configuration_sha256(configuration: dict[str, Any]) -> str:
    """Hash method configuration exactly as the isolated worker does."""

    try:
        raw = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ManifestError(f"Could not hash method configuration: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _safe_file(root: Path, value: Any, label: str) -> Path:
    raw_path = Path(_text(value, label))
    if raw_path.is_absolute():
        raise ManifestError(f"{label} must be relative to the manifest.")
    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"{label} escapes the manifest directory.") from exc
    if not resolved.is_file():
        raise ManifestError(f"{label} does not exist: {resolved}")
    return resolved


def _verified_file(root: Path, path_value: Any, digest_value: Any, label: str) -> tuple[Path, str]:
    path = _safe_file(root, path_value, f"{label}.path")
    expected = _sha256(digest_value, f"{label}.sha256")
    raw = _read_limited_file(path, MAX_ARTIFACT_BYTES, label)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ManifestError(
            f"{label} checksum mismatch: expected {expected}, got {actual}."
        )
    return path, actual


def _read_limited_file(path: Path, limit: int, label: str) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ManifestError(f"{label} exceeds {limit} bytes.")
    return raw


def _point(value: Any, width: int, height: int, label: str) -> tuple[float, float]:
    pair = _array(value, label)
    if len(pair) != 2:
        raise ManifestError(f"{label} must be an [x, y] pair.")
    x = _positive_number(pair[0], f"{label}[0]", allow_zero=True)
    y = _positive_number(pair[1], f"{label}[1]", allow_zero=True)
    if x > width - 1 or y > height - 1:
        raise ManifestError(f"{label} is outside the declared canvas.")
    return x, y


_PNM_WHITESPACE = b" \t\r\n\v\f"


def _pnm_token(
    raw: bytes,
    offset: int,
    label: str,
    header: bool = False,
) -> tuple[bytes | None, int]:
    while offset < len(raw):
        if raw[offset] in _PNM_WHITESPACE:
            offset += 1
            continue
        if raw[offset] == ord("#"):
            newline = raw.find(b"\n", offset + 1)
            offset = len(raw) if newline < 0 else newline + 1
            continue
        break
    if offset >= len(raw):
        return None, offset
    start = offset
    while offset < len(raw):
        if raw[offset] in _PNM_WHITESPACE or raw[offset] == ord("#"):
            break
        offset += 1
    if header and offset > MAX_IMAGE_HEADER_BYTES:
        raise ManifestError(
            f"PNM header exceeds {MAX_IMAGE_HEADER_BYTES} bytes: {label}"
        )
    return raw[start:offset], offset


def _pnm_integer(token: bytes | None, label: str) -> int:
    if token is None or not token or len(token) > 10 or not token.isdigit():
        raise ManifestError(f"Invalid PNM integer for {label}.")
    return int(token)


def _validate_pnm(raw: bytes, suffix: str, label: str) -> tuple[int, int]:
    magic, offset = _pnm_token(raw, 0, label, header=True)
    if magic not in {b"P1", b"P2", b"P3", b"P4", b"P5", b"P6"}:
        raise ManifestError(f"Unsupported PNM header: {label}")
    expected_magic = {
        ".pbm": {b"P1", b"P4"},
        ".pgm": {b"P2", b"P5"},
        ".ppm": {b"P3", b"P6"},
        ".pnm": {b"P1", b"P2", b"P3", b"P4", b"P5", b"P6"},
    }[suffix]
    if magic not in expected_magic:
        raise ManifestError(f"PNM magic {magic!r} does not match {suffix}: {label}")

    width_token, offset = _pnm_token(raw, offset, label, header=True)
    height_token, offset = _pnm_token(raw, offset, label, header=True)
    width = _pnm_integer(width_token, f"{label} width")
    height = _pnm_integer(height_token, f"{label} height")
    if (
        width <= 0
        or height <= 0
        or width > MAX_CANVAS_DIMENSION
        or height > MAX_CANVAS_DIMENSION
        or width * height > MAX_CANVAS_PIXELS
    ):
        raise ManifestError(f"PNM dimensions are outside schema-v1 limits: {label}")

    max_value = 1
    if magic not in {b"P1", b"P4"}:
        max_token, offset = _pnm_token(raw, offset, label, header=True)
        max_value = _pnm_integer(max_token, f"{label} maxval")
        if not 1 <= max_value <= 65535:
            raise ManifestError(f"PNM maxval must be between 1 and 65535: {label}")

    channels = 3 if magic in {b"P3", b"P6"} else 1
    sample_count = width * height * channels
    if magic in {b"P1", b"P2", b"P3"}:
        count = 0
        while True:
            token, offset = _pnm_token(raw, offset, label)
            if token is None:
                break
            value = _pnm_integer(token, f"{label} sample")
            if value > max_value:
                raise ManifestError(f"PNM sample exceeds maxval: {label}")
            count += 1
            if count > sample_count:
                raise ManifestError(f"PNM contains too many samples: {label}")
        if count != sample_count:
            raise ManifestError(
                f"PNM contains {count} samples; expected {sample_count}: {label}"
            )
        return width, height

    if offset >= len(raw) or raw[offset] not in _PNM_WHITESPACE:
        raise ManifestError(f"Binary PNM header lacks a raster delimiter: {label}")
    if raw[offset : offset + 2] == b"\r\n":
        offset += 2
    else:
        offset += 1
    payload = raw[offset:]
    if magic == b"P4":
        expected_bytes = ((width + 7) // 8) * height
    else:
        bytes_per_sample = 1 if max_value < 256 else 2
        expected_bytes = sample_count * bytes_per_sample
    if len(payload) != expected_bytes:
        raise ManifestError(
            f"Binary PNM has {len(payload)} raster bytes; expected {expected_bytes}: {label}"
        )
    if magic != b"P4" and max_value < (255 if max_value < 256 else 65535):
        step = 1 if max_value < 256 else 2
        for index in range(0, len(payload), step):
            value = (
                payload[index]
                if step == 1
                else int.from_bytes(payload[index : index + 2], "big")
            )
            if value > max_value:
                raise ManifestError(f"Binary PNM sample exceeds maxval: {label}")
    return width, height


def _validate_png(raw: bytes, label: str) -> tuple[int, int]:
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ManifestError(f"Invalid PNG signature: {label}")
    offset = 8
    width = height = bit_depth = color_type = None
    palette = None
    idat_parts = []
    seen_ihdr = False
    seen_idat = False
    idat_ended = False
    seen_iend = False
    while offset < len(raw):
        if len(raw) - offset < 12:
            raise ManifestError(f"Truncated PNG chunk: {label}")
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(raw):
            raise ManifestError(f"Truncated PNG chunk payload: {label}")
        if len(chunk_type) != 4 or not all(
            ord("A") <= byte <= ord("Z") or ord("a") <= byte <= ord("z")
            for byte in chunk_type
        ):
            raise ManifestError(f"Invalid PNG chunk type: {label}")
        data = raw[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", raw[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ManifestError(f"PNG chunk CRC mismatch: {label}")
        if not seen_ihdr and chunk_type != b"IHDR":
            raise ManifestError(f"PNG IHDR must be the first chunk: {label}")

        if chunk_type == b"IHDR":
            if seen_ihdr or length != 13:
                raise ManifestError(f"Invalid or duplicate PNG IHDR: {label}")
            seen_ihdr = True
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            valid_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if bit_depth not in valid_depths.get(color_type, set()):
                raise ManifestError(f"Unsupported PNG color type/bit depth: {label}")
            if compression != 0 or filtering != 0 or interlace != 0:
                raise ManifestError(
                    f"PNG must use standard compression/filtering and no interlace: {label}"
                )
            if (
                width <= 0
                or height <= 0
                or width > MAX_CANVAS_DIMENSION
                or height > MAX_CANVAS_DIMENSION
                or width * height > MAX_CANVAS_PIXELS
            ):
                raise ManifestError(f"PNG dimensions are outside schema-v1 limits: {label}")
        elif chunk_type == b"PLTE":
            if palette is not None or seen_idat or length == 0 or length % 3 or length > 768:
                raise ManifestError(f"Invalid PNG palette: {label}")
            palette = data
        elif chunk_type == b"IDAT":
            if idat_ended:
                raise ManifestError(f"PNG IDAT chunks must be consecutive: {label}")
            seen_idat = True
            idat_parts.append(data)
        elif chunk_type == b"IEND":
            if length != 0 or seen_iend:
                raise ManifestError(f"Invalid PNG IEND: {label}")
            seen_iend = True
            offset = chunk_end
            break
        elif chunk_type[0] & 0x20 == 0:
            raise ManifestError(f"Unknown critical PNG chunk {chunk_type!r}: {label}")

        if seen_idat and chunk_type != b"IDAT":
            idat_ended = True
        offset = chunk_end

    if not seen_ihdr or not seen_idat or not seen_iend or offset != len(raw):
        raise ManifestError(f"PNG is missing required chunks or has trailing bytes: {label}")
    if color_type == 3:
        if palette is None or len(palette) // 3 > 1 << bit_depth:
            raise ManifestError(f"Indexed PNG requires a valid palette: {label}")
    elif color_type in {0, 4} and palette is not None:
        raise ManifestError(f"Grayscale PNG may not contain PLTE: {label}")

    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    row_bytes = (width * channels * bit_depth + 7) // 8
    expected_decoded = (row_bytes + 1) * height
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(b"".join(idat_parts), expected_decoded + 1)
        if decoder.unconsumed_tail or len(decoded) > expected_decoded:
            raise ManifestError(f"PNG decompressed payload exceeds its canvas: {label}")
        decoded += decoder.flush()
    except zlib.error as exc:
        raise ManifestError(f"PNG IDAT data is invalid: {label}") from exc
    if (
        len(decoded) != expected_decoded
        or not decoder.eof
        or decoder.unused_data
    ):
        raise ManifestError(f"PNG decompressed payload size is invalid: {label}")
    if any(decoded[row * (row_bytes + 1)] > 4 for row in range(height)):
        raise ManifestError(f"PNG contains an invalid row filter: {label}")
    return width, height


def _image_dimensions_bytes(raw: bytes, suffix: str, label: str) -> tuple[int, int]:
    if suffix in {".pbm", ".pgm", ".ppm", ".pnm"}:
        return _validate_pnm(raw, suffix, label)
    if suffix == ".png":
        return _validate_png(raw, label)

    raise ManifestError(
        f"Benchmark schema v1 accepts only lossless PNG or PNM images, not {suffix}."
    )


def image_dimensions(path: Path) -> tuple[int, int]:
    """Read bounded lossless-image dimensions without mandatory dependencies."""

    raw = _read_limited_file(path, MAX_IMAGE_BYTES, f"image {path}")
    return _image_dimensions_bytes(raw, path.suffix.lower(), str(path))


def _verified_image(
    root: Path,
    path_value: Any,
    digest_value: Any,
    label: str,
) -> tuple[Path, str, tuple[int, int]]:
    path = _safe_file(root, path_value, f"{label}.path")
    expected = _sha256(digest_value, f"{label}.sha256")
    raw = _read_limited_file(path, MAX_IMAGE_BYTES, label)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ManifestError(
            f"{label} checksum mismatch: expected {expected}, got {actual}."
        )
    dimensions = _image_dimensions_bytes(raw, path.suffix.lower(), str(path))
    return path, actual, dimensions


def _metric_config(payload: Any) -> MetricConfig:
    config = _object(payload, "metric_config")
    raw_tolerances = _array(config.get("tolerances_px"), "metric_config.tolerances_px")
    if len(raw_tolerances) > MAX_TOLERANCES:
        raise ManifestError(
            f"metric_config.tolerances_px may contain at most {MAX_TOLERANCES} values."
        )
    tolerances = tuple(
        sorted(
            {
                _positive_number(value, f"metric_config.tolerances_px[{index}]", allow_zero=True)
                for index, value in enumerate(raw_tolerances)
            }
        )
    )
    if not tolerances:
        raise ManifestError("metric_config.tolerances_px must not be empty.")
    if any(tolerance > MAX_TOLERANCE_PX for tolerance in tolerances):
        raise ManifestError(
            f"metric_config tolerances may not exceed {MAX_TOLERANCE_PX:g}px."
        )
    primary = _positive_number(
        config.get("primary_tolerance_px"),
        "metric_config.primary_tolerance_px",
        allow_zero=True,
    )
    if primary not in tolerances:
        raise ManifestError("primary_tolerance_px must be listed in tolerances_px.")
    branch_tolerance = _positive_number(
        config.get("branch_tolerance_px", primary),
        "metric_config.branch_tolerance_px",
        allow_zero=True,
    )
    if branch_tolerance != primary:
        raise ManifestError(
            "Benchmark schema v1 requires branch_tolerance_px to equal "
            "primary_tolerance_px."
        )
    connectivity = config.get("connectivity", 8)
    if connectivity != 8:
        raise ManifestError("Benchmark schema v1 requires 8-connectivity.")
    diagonal_rule = _text(config.get("diagonal_rule", "no_corner_cut"), "metric_config.diagonal_rule")
    if diagonal_rule != "no_corner_cut":
        raise ManifestError("Benchmark schema v1 requires diagonal_rule=no_corner_cut.")
    percentile = _text(config.get("percentile", "nearest_rank"), "metric_config.percentile")
    if percentile != "nearest_rank":
        raise ManifestError("Benchmark schema v1 requires percentile=nearest_rank.")
    return MetricConfig(primary, tolerances, branch_tolerance, 8, diagonal_rule, percentile)


def _dataset(payload: Any) -> DatasetInfo:
    dataset = _object(payload, "dataset")
    return DatasetInfo(
        identifier=_identifier(dataset.get("id"), "dataset.id"),
        version=_text(dataset.get("version"), "dataset.version"),
        description=_text(dataset.get("description"), "dataset.description"),
        license=_text(dataset.get("license"), "dataset.license"),
        source=_text(dataset.get("source"), "dataset.source"),
    )


def _method(payload: Any, index: int) -> MethodSpec:
    label = f"methods[{index}]"
    method = _object(payload, label)
    kind = _text(method.get("kind"), f"{label}.kind")
    if kind != METHOD_KIND_PRECOMPUTED:
        raise ManifestError(
            f"{label}.kind must be {METHOD_KIND_PRECOMPUTED!r} in schema v1."
        )
    model_sha = method.get("model_sha256")
    if model_sha is not None:
        model_sha = _sha256(model_sha, f"{label}.model_sha256")
    configuration = method.get("configuration", {})
    if not isinstance(configuration, dict):
        raise ManifestError(f"{label}.configuration must be an object.")
    return MethodSpec(
        identifier=_identifier(method.get("id"), f"{label}.id"),
        label=_text(method.get("label"), f"{label}.label"),
        kind=kind,
        source=_text(method.get("source"), f"{label}.source"),
        version=_text(method.get("version"), f"{label}.version"),
        license=_text(method.get("license"), f"{label}.license"),
        model_sha256=model_sha,
        configuration=dict(configuration),
    )


def _validate_efficientsam_method_contract(method: MethodSpec) -> None:
    """Mirror the isolated worker's accepted EfficientSAM configuration."""

    configuration = method.configuration
    threshold = configuration.get("mask_logit_threshold", 0.0)
    try:
        threshold_number = float(threshold)
    except (OverflowError, TypeError, ValueError):
        threshold_number = math.nan
    low = configuration.get("canny_low_threshold", 30)
    high = configuration.get("canny_high_threshold", 100)
    valid_canny = (
        not isinstance(low, bool)
        and not isinstance(high, bool)
        and isinstance(low, int)
        and isinstance(high, int)
        and 0 <= low < high <= 255
    )
    if (
        method.model_sha256 != EFFICIENTSAM_BUNDLE_SHA256
        or configuration.get("model_bundle_id") != EFFICIENTSAM_TI_SPLIT.id
        or configuration.get("model_bundle_sha256")
        != EFFICIENTSAM_BUNDLE_SHA256
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold_number)
        or threshold_number != 0.0
        or not valid_canny
        or configuration.get(
            "smoothing_profile",
            PRODUCT_SMOOTHING_PROFILE,
        )
        != PRODUCT_SMOOTHING_PROFILE
    ):
        raise ManifestError(
            f"Method {EFFICIENTSAM_METHOD_ID!r} must match the pinned worker "
            "model, zero mask threshold, valid Canny thresholds, and product "
            "smoothing contract."
        )


def _timing(payload: Any, label: str, successful: bool) -> TimingRecord:
    timing = _object(payload, label)
    warmup_runs = _nonnegative_integer(timing.get("warmup_runs"), f"{label}.warmup_runs")
    wall_values = _array(timing.get("wall_ns_samples", []), f"{label}.wall_ns_samples")
    cpu_values = _array(timing.get("cpu_ns_samples", []), f"{label}.cpu_ns_samples")
    if len(wall_values) > MAX_TIMING_REPETITIONS or len(cpu_values) > MAX_TIMING_REPETITIONS:
        raise ManifestError(
            f"{label} may contain at most {MAX_TIMING_REPETITIONS} repetitions."
        )
    wall = tuple(
        _nonnegative_integer(value, f"{label}.wall_ns_samples[{index}]")
        for index, value in enumerate(wall_values)
    )
    cpu = tuple(
        _nonnegative_integer(value, f"{label}.cpu_ns_samples[{index}]")
        for index, value in enumerate(cpu_values)
    )
    if successful and (not wall or len(wall) != len(cpu)):
        raise ManifestError(
            f"{label} needs equal, non-empty wall_ns_samples and cpu_ns_samples."
        )
    if successful and len(wall) < 3:
        raise ManifestError(f"{label} needs at least three measured repetitions.")
    if successful and warmup_runs < 1:
        raise ManifestError(f"{label}.warmup_runs must be at least one.")
    if not successful and len(wall) != len(cpu):
        raise ManifestError(f"{label} wall and CPU sample counts must match.")
    record = TimingRecord(
        warmup_runs=warmup_runs,
        wall_ns_samples=wall,
        cpu_ns_samples=cpu,
        model_load_wall_ns=_nonnegative_integer(
            timing.get("model_load_wall_ns"),
            f"{label}.model_load_wall_ns",
            optional=True,
        ),
        peak_rss_bytes=_nonnegative_integer(
            timing.get("peak_rss_bytes"),
            f"{label}.peak_rss_bytes",
            optional=True,
        ),
    )
    if successful and record.model_load_wall_ns is None:
        raise ManifestError(f"{label}.model_load_wall_ns is required.")
    if successful and record.peak_rss_bytes is None:
        raise ManifestError(f"{label}.peak_rss_bytes is required.")
    return record


def _runtime_record(
    payload: Any,
    label: str,
    successful: bool,
    artifact_sha256: str | None,
    timing: TimingRecord,
) -> dict[str, Any]:
    runtime = _object(payload, label)
    for key in (
        "adapter_version",
        "python_version",
        "platform",
        "cpu",
        "actual_provider",
    ):
        _text(runtime.get(key), f"{label}.{key}")
    provider_kind = _identifier(
        runtime.get("provider_kind"),
        f"{label}.provider_kind",
    )
    if provider_kind not in CPU_PROVIDER_NAMES:
        raise ManifestError(
            f"{label}.provider_kind must be one of {sorted(CPU_PROVIDER_NAMES)}."
        )
    provider_device_type = _text(
        runtime.get("provider_device_type"),
        f"{label}.provider_device_type",
    ).lower()
    if provider_device_type != "cpu":
        raise ManifestError(f"{label}.provider_device_type must be 'cpu'.")
    actual_provider = str(runtime["actual_provider"])
    expected_provider = CPU_PROVIDER_NAMES[provider_kind]
    if actual_provider != expected_provider:
        raise ManifestError(
            f"{label}.actual_provider must equal {expected_provider!r} for "
            f"provider_kind={provider_kind!r}."
        )
    if provider_kind == "onnxruntime":
        providers = _object(
            runtime.get("onnx_providers"),
            f"{label}.onnx_providers",
        )
        expected_sessions = {"encoder", "decoder"}
        if set(providers) != expected_sessions or any(
            providers.get(session) != ["CPUExecutionProvider"]
            for session in expected_sessions
        ):
            raise ManifestError(
                f"{label}.onnx_providers must attest CPUExecutionProvider "
                "for encoder and decoder only."
            )
    package_versions = _object(runtime.get("package_versions"), f"{label}.package_versions")
    for key, value in package_versions.items():
        _text(key, f"{label}.package_versions key")
        if value is not None:
            _text(value, f"{label}.package_versions.{key}")
    thread_settings = _object(runtime.get("thread_settings"), f"{label}.thread_settings")
    if not thread_settings:
        raise ManifestError(f"{label}.thread_settings must not be empty.")
    _positive_integer(thread_settings.get("threads"), f"{label}.thread_settings.threads")
    for key, value in thread_settings.items():
        _text(key, f"{label}.thread_settings key")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ManifestError(
                f"{label}.thread_settings.{key} must be a scalar or null."
            )
    deterministic = runtime.get("deterministic")
    if successful and not isinstance(deterministic, bool):
        raise ManifestError(f"{label}.deterministic must be boolean.")
    if not successful and deterministic is not None and not isinstance(deterministic, bool):
        raise ManifestError(f"{label}.deterministic must be boolean or null.")
    raw_hashes = _array(
        runtime.get("output_sha256_samples", []),
        f"{label}.output_sha256_samples",
    )
    output_hashes = [
        _sha256(value, f"{label}.output_sha256_samples[{index}]")
        for index, value in enumerate(raw_hashes)
    ]
    if successful and len(output_hashes) < 3:
        raise ManifestError(f"{label} needs at least three repeated output hashes.")
    if output_hashes and len(output_hashes) != len(timing.wall_ns_samples):
        raise ManifestError(
            f"{label} output hash count must match measured timing repetitions."
        )
    if successful and artifact_sha256 not in output_hashes:
        raise ManifestError(f"{label} output hashes do not include the published artifact.")
    observed_deterministic = len(set(output_hashes)) <= 1 if output_hashes else None
    if deterministic is not None and observed_deterministic is not None and deterministic != observed_deterministic:
        raise ManifestError(
            f"{label}.deterministic disagrees with repeated output hashes."
        )
    return dict(runtime)


def _prediction(root: Path, payload: Any, method_id: str, label: str) -> PredictionSpec:
    prediction = _object(payload, label)
    execution = _object(prediction.get("execution"), f"{label}.execution")
    status = _text(execution.get("status"), f"{label}.execution.status")
    if status not in {"ok", "fallback", "failed"}:
        raise ManifestError(f"{label}.execution.status must be ok, fallback, or failed.")
    successful = status in {"ok", "fallback"}
    requested = _identifier(
        execution.get("requested_backend"),
        f"{label}.execution.requested_backend",
    )
    if requested != method_id:
        raise ManifestError(f"{label} requested_backend must match method id {method_id!r}.")
    actual_value = execution.get("actual_backend")
    actual = _identifier(actual_value, f"{label}.execution.actual_backend") if actual_value else None
    fallback_value = execution.get("fallback_reason")
    fallback = _text(fallback_value, f"{label}.execution.fallback_reason") if fallback_value else None
    error_value = execution.get("error")
    error = _text(error_value, f"{label}.execution.error") if error_value else None
    device = _text(execution.get("device"), f"{label}.execution.device").lower()
    if device != "cpu":
        raise ManifestError(f"{label} must record device='cpu' for the M1 benchmark.")
    if successful and actual is None:
        raise ManifestError(f"{label} needs actual_backend for a successful result.")
    if status == "ok":
        if actual != requested:
            raise ManifestError(
                f"{label} status=ok requires actual_backend to equal requested_backend."
            )
        if fallback or error:
            raise ManifestError(
                f"{label} status=ok cannot include fallback_reason or error."
            )
    elif status == "fallback":
        if actual == requested:
            raise ManifestError(
                f"{label} status=fallback requires a different actual_backend."
            )
        if not fallback or error:
            raise ManifestError(
                f"{label} status=fallback needs fallback_reason and no error."
            )
    elif not error or fallback:
        raise ManifestError(
            f"{label} status=failed needs error and no fallback_reason."
        )

    timing = _timing(execution.get("timing"), f"{label}.execution.timing", successful)

    if successful:
        artifact, digest = _verified_file(
            root,
            prediction.get("artifact"),
            prediction.get("sha256"),
            f"{label}.artifact",
        )
    else:
        if prediction.get("artifact") is not None or prediction.get("sha256") is not None:
            raise ManifestError(f"{label} failed results must not claim an artifact.")
        artifact, digest = None, None
    runtime = _runtime_record(
        execution.get("runtime"),
        f"{label}.execution.runtime",
        successful,
        digest,
        timing,
    )
    execution_record = ExecutionRecord(
        status=status,
        requested_backend=requested,
        actual_backend=actual,
        fallback_reason=fallback,
        error=error,
        device=device,
        runtime=runtime,
        timing=timing,
    )
    return PredictionSpec(artifact, digest, execution_record)


def _prompt(payload: Any, width: int, height: int, label: str) -> PromptSpec:
    prompt = _object(payload, label)
    schema_version = prompt.get(
        "schema_version",
        PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
    )
    if schema_version not in PROMPT_EVIDENCE_SCHEMA_VERSIONS:
        raise ManifestError(f"{label}.schema_version is unsupported.")
    positive_values = _array(prompt.get("positive_xy", []), f"{label}.positive_xy")
    negative_values = _array(prompt.get("negative_xy", []), f"{label}.negative_xy")
    positive = tuple(
        _point(value, width, height, f"{label}.positive_xy[{index}]")
        for index, value in enumerate(positive_values)
    )
    negative = tuple(
        _point(value, width, height, f"{label}.negative_xy[{index}]")
        for index, value in enumerate(negative_values)
    )
    if len(positive) + len(negative) > 6:
        raise ManifestError(f"{label} may contain at most six SAM prompt points.")
    previous_value = prompt.get("previous_xy")
    if (
        schema_version == PROMPT_EVIDENCE_SCHEMA_VERSION_V1
        and previous_value is not None
    ):
        raise ManifestError(
            f"{label} schema v1 does not support previous_xy."
        )
    return PromptSpec(
        start_xy=_point(prompt.get("start_xy"), width, height, f"{label}.start_xy"),
        end_xy=_point(prompt.get("end_xy"), width, height, f"{label}.end_xy"),
        positive_xy=positive,
        negative_xy=negative,
        previous_xy=(
            _point(previous_value, width, height, f"{label}.previous_xy")
            if previous_value is not None
            else None
        ),
        schema_version=schema_version,
    )


def _sam_prediction_evidence(payload: Any, label: str) -> dict[str, Any]:
    evidence = _object(payload, label)
    if set(evidence) != SAM_PREDICTION_EVIDENCE_KEYS:
        raise ManifestError(f"{label} has an unsupported structure.")
    if (
        evidence.get("schema_version")
        != EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION
    ):
        raise ManifestError(f"{label} has an unsupported schema.")
    selected_index = _nonnegative_integer(
        evidence.get("selected_mask_index"),
        f"{label}.selected_mask_index",
    )
    if selected_index >= 3:
        raise ManifestError(f"{label}.selected_mask_index must be below three.")
    raw_ious = _array(evidence.get("iou_predictions"), f"{label}.iou_predictions")
    if len(raw_ious) != 3:
        raise ManifestError(f"{label}.iou_predictions must contain three values.")
    ious = []
    for index, raw_iou in enumerate(raw_ious):
        if isinstance(raw_iou, bool) or not isinstance(raw_iou, (int, float)):
            raise ManifestError(
                f"{label}.iou_predictions[{index}] must be numeric."
            )
        iou = float(raw_iou)
        if not math.isfinite(iou):
            raise ManifestError(
                f"{label}.iou_predictions[{index}] must be finite."
            )
        try:
            iou = struct.unpack("<f", struct.pack("<f", iou))[0]
        except (OverflowError, struct.error) as exc:
            raise ManifestError(
                f"{label}.iou_predictions[{index}] cannot be represented as "
                "float32."
            ) from exc
        if not math.isfinite(iou):
            raise ManifestError(
                f"{label}.iou_predictions[{index}] must remain finite as "
                "float32."
            )
        ious.append(iou)
    if selected_index != max(range(len(ious)), key=ious.__getitem__):
        raise ManifestError(
            f"{label}.selected_mask_index disagrees with maximum predicted IoU."
        )
    for key in (
        "iou_predictions_sha256",
        "selected_logits_sha256",
        "selected_binary_mask_sha256",
        "accepted_mask_sha256",
    ):
        _sha256(evidence.get(key), f"{label}.{key}")
    try:
        expected_iou_sha256 = hashlib.sha256(struct.pack("<3f", *ious)).hexdigest()
    except (OverflowError, struct.error) as exc:
        raise ManifestError(
            f"{label}.iou_predictions cannot be represented as float32."
        ) from exc
    if evidence["iou_predictions_sha256"] != expected_iou_sha256:
        raise ManifestError(
            f"{label}.iou_predictions_sha256 disagrees with its float32 values."
        )
    decoder_wall_ns = _nonnegative_integer(
        evidence.get("decoder_wall_ns"),
        f"{label}.decoder_wall_ns",
    )
    return {
        "schema_version": EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION,
        "selected_mask_index": selected_index,
        "iou_predictions": ious,
        "iou_predictions_sha256": evidence["iou_predictions_sha256"],
        "selected_logits_sha256": evidence["selected_logits_sha256"],
        "selected_binary_mask_sha256": evidence[
            "selected_binary_mask_sha256"
        ],
        "accepted_mask_sha256": evidence["accepted_mask_sha256"],
        "decoder_wall_ns": decoder_wall_ns,
    }


def _stable_sam_prediction_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: ([*value] if key == "iou_predictions" else value)
        for key, value in evidence.items()
        if key != "decoder_wall_ns"
    }


def _sample(
    root: Path,
    payload: Any,
    index: int,
    methods_by_id: dict[str, MethodSpec],
) -> SampleSpec:
    label = f"samples[{index}]"
    method_ids = tuple(methods_by_id)
    sample = _object(payload, label)
    width = _positive_integer(sample.get("width"), f"{label}.width")
    height = _positive_integer(sample.get("height"), f"{label}.height")
    if width > MAX_CANVAS_DIMENSION or height > MAX_CANVAS_DIMENSION:
        raise ManifestError(
            f"{label} dimensions may not exceed {MAX_CANVAS_DIMENSION} pixels."
        )
    if width * height > MAX_CANVAS_PIXELS:
        raise ManifestError(f"{label} canvas exceeds {MAX_CANVAS_PIXELS} pixels.")
    if EFFICIENTSAM_METHOD_ID in method_ids and (
        width != EFFICIENTSAM_INPUT_DIMENSION
        or height != EFFICIENTSAM_INPUT_DIMENSION
    ):
        raise ManifestError(
            f"{label} EfficientSAM inputs must be exactly "
            f"{EFFICIENTSAM_INPUT_DIMENSION}x{EFFICIENTSAM_INPUT_DIMENSION}."
        )
    image, image_sha, actual_dimensions = _verified_image(
        root,
        sample.get("image"),
        sample.get("image_sha256"),
        f"{label}.image",
    )
    if actual_dimensions != (width, height):
        raise ManifestError(
            f"{label} image is {actual_dimensions[0]}x{actual_dimensions[1]}, "
            f"not the declared {width}x{height}."
        )
    reference, reference_sha = _verified_file(
        root,
        sample.get("reference"),
        sample.get("reference_sha256"),
        f"{label}.reference",
    )

    strata_payload = _object(sample.get("strata"), f"{label}.strata")
    strata = {
        key: _text(strata_payload.get(key), f"{label}.strata.{key}")
        for key in REQUIRED_STRATA
    }
    for key, value in strata_payload.items():
        if key not in strata:
            strata[_text(key, f"{label}.strata key")] = _text(
                value,
                f"{label}.strata.{key}",
            )

    source_payload = _object(sample.get("source"), f"{label}.source")
    source = {
        key: _text(source_payload.get(key), f"{label}.source.{key}")
        for key in ("name", "license", "url")
    }
    raw_source_tile_origin = sample.get("source_tile_origin_xy")
    needs_source_grid = bool(set(method_ids) & SOURCE_GRID_METHOD_IDS)
    synthetic_source = source["url"].startswith("generated://")
    if raw_source_tile_origin is None:
        if needs_source_grid and not synthetic_source:
            raise ManifestError(
                f"{label}.source_tile_origin_xy is required for non-synthetic "
                "Ink v2 and Smart Recovery samples."
            )
        source_tile_origin_xy = (0, 0)
    else:
        source_tile_origin_xy = _source_tile_origin(
            raw_source_tile_origin,
            f"{label}.source_tile_origin_xy",
        )

    prompt = _prompt(sample.get("prompt"), width, height, f"{label}.prompt")
    if EFFICIENTSAM_METHOD_ID in method_ids:
        sam_points = (
            prompt.start_xy,
            *prompt.positive_xy,
            prompt.end_xy,
            *prompt.negative_xy,
        )
        if len(sam_points) > 6:
            raise ManifestError(
                f"{label}.prompt may contain at most six EfficientSAM points "
                "including start and end."
            )
        if len(set(sam_points)) != len(sam_points):
            raise ManifestError(
                f"{label}.prompt EfficientSAM points must not repeat start, "
                "end, or another guide."
            )
    raw_predictions = _object(sample.get("predictions"), f"{label}.predictions")
    unknown = sorted(set(raw_predictions) - set(method_ids))
    missing = sorted(set(method_ids) - set(raw_predictions))
    if unknown or missing:
        raise ManifestError(
            f"{label}.predictions mismatch; missing={missing}, unknown={unknown}."
        )
    predictions = {
        method_id: _prediction(
            root,
            raw_predictions[method_id],
            method_id,
            f"{label}.predictions.{method_id}",
        )
        for method_id in method_ids
    }
    expected_prompt_sha256 = prompt_sha256(prompt)
    expected_sam_tensor_sha256 = sam_prompt_tensor_sha256(prompt)
    expected_recovery_tensor_sha256 = (
        recovery_prompt_tensor_sha256(
            prompt,
            width=width,
            height=height,
        )
        if RECOVERY_METHOD_ID in method_ids
        else None
    )
    expected_source_grid_sha256 = source_grid_input_sha256(
        image_sha,
        source_tile_origin_xy,
    )
    for method_id, prediction in predictions.items():
        runtime = prediction.execution.runtime
        if method_id in SOURCE_GRID_METHOD_IDS:
            if (
                runtime.get("source_tile_origin_xy")
                != list(source_tile_origin_xy)
                or runtime.get("source_grid_input_sha256")
                != expected_source_grid_sha256
            ):
                raise ManifestError(
                    f"{label}.predictions.{method_id} is not bound to the "
                    "sample image and source-grid tile origin."
                )
        observed_prompt_sha256 = runtime.get("prompt_sha256")
        if observed_prompt_sha256 is not None:
            _sha256(
                observed_prompt_sha256,
                f"{label}.predictions.{method_id}.execution.runtime.prompt_sha256",
            )
            if observed_prompt_sha256 != expected_prompt_sha256:
                raise ManifestError(
                    f"{label}.predictions.{method_id} prompt_sha256 does not "
                    "match the sample prompt."
                )
        if (
            method_id == INK_V2_METHOD_ID
            and prediction.execution.status in {"ok", "fallback"}
        ):
            if prediction.artifact_path is None:
                raise ManifestError(
                    f"{label}.predictions.{method_id} omitted its artifact."
                )
            try:
                artifact = load_centerline_artifact(prediction.artifact_path)
            except (OSError, ValueError) as exc:
                raise ManifestError(
                    f"{label}.predictions.{method_id} artifact is invalid: {exc}"
                ) from exc
            expected_ink_metadata = {
                "actual_backend": prediction.execution.actual_backend,
                "configuration_sha256": _configuration_sha256(
                    methods_by_id[method_id].configuration
                ),
                "input_sha256": image_sha,
                "prompt_sha256": expected_prompt_sha256,
                "requested_backend": method_id,
                "smoothing": "smart-trace-livewire-v1",
                "source_grid_input_sha256": expected_source_grid_sha256,
                "source_tile_origin_xy": list(source_tile_origin_xy),
                "trace_kernel": "ai_vectorizer.core.livewire",
            }
            if (
                set(artifact.metadata) != INK_V2_ARTIFACT_METADATA_KEYS
                or artifact.metadata != expected_ink_metadata
            ):
                raise ManifestError(
                    f"{label}.predictions.{method_id} artifact is not bound to "
                    "its source-grid tile origin."
                )
        if (
            method_id == EFFICIENTSAM_METHOD_ID
            and prediction.execution.status in {"ok", "fallback"}
        ):
            if runtime.get("provider_kind") != "onnxruntime":
                raise ManifestError(
                    f"{label}.predictions.{method_id} must use the "
                    "onnxruntime CPU provider contract."
                )
            if observed_prompt_sha256 != expected_prompt_sha256:
                raise ManifestError(
                    f"{label}.predictions.{method_id} must bind the complete prompt."
                )
            observed_tensor_sha256 = runtime.get("sam_prompt_tensor_sha256")
            if observed_tensor_sha256 is not None:
                _sha256(
                    observed_tensor_sha256,
                    f"{label}.predictions.{method_id}.execution.runtime."
                    "sam_prompt_tensor_sha256",
                )
            if observed_tensor_sha256 != expected_sam_tensor_sha256:
                raise ManifestError(
                    f"{label}.predictions.{method_id} must bind the exact "
                    "EfficientSAM prompt tensors."
                )
            runtime_label = (
                f"{label}.predictions.{method_id}.execution.runtime"
            )
            method = methods_by_id[method_id]
            expected_configuration_sha256 = _configuration_sha256(
                method.configuration
            )
            if (
                prediction.execution.status != "ok"
                or prediction.execution.actual_backend != method_id
            ):
                raise ManifestError(
                    f"{label}.predictions.{method_id} does not permit a "
                    "fallback backend."
                )
            if runtime.get("provider_verified") is not True:
                raise ManifestError(
                    f"{runtime_label}.provider_verified must attest the "
                    "observed CPU provider."
                )
            if (
                runtime.get("input_sha256") != image_sha
                or runtime.get("configuration_sha256")
                != expected_configuration_sha256
            ):
                raise ManifestError(
                    f"{runtime_label} is not bound to the sample image and "
                    "method configuration."
                )
            expected_session_options = {
                "encoder": dict(EFFICIENTSAM_ORT_SESSION_OPTIONS),
                "decoder": dict(EFFICIENTSAM_ORT_SESSION_OPTIONS),
            }
            if runtime.get("onnx_session_options") != expected_session_options:
                raise ManifestError(
                    f"{runtime_label}.onnx_session_options does not match the "
                    "required encoder and decoder readback."
                )
            if (
                runtime.get("model_bundle_id") != EFFICIENTSAM_TI_SPLIT.id
                or runtime.get("model_bundle_sha256")
                != EFFICIENTSAM_BUNDLE_SHA256
                or runtime.get("model_source_commit")
                != EFFICIENTSAM_TI_SPLIT.source_commit
                or runtime.get("model_artifacts_sha256")
                != EFFICIENTSAM_ARTIFACTS_SHA256
            ):
                raise ManifestError(
                    f"{runtime_label} does not match the pinned EfficientSAM "
                    "bundle, commit, and artifact hashes."
                )
            thread_settings = runtime["thread_settings"]
            expected_thread_settings = {
                "threads": 1,
                "onnx_intra_op_num_threads": 1,
                "onnx_inter_op_num_threads": 1,
                "onnx_execution_mode": "ORT_SEQUENTIAL",
                "onnx_graph_optimization_level": "ORT_ENABLE_ALL",
                "opencv_set_num_threads": 0,
                "opencv_effective_num_threads": 1,
                "opencl": False,
                **{
                    variable: "1"
                    for variable in SHARED_THREAD_SETTING_KEYS
                    if variable != "threads"
                },
            }
            if any(
                thread_settings.get(key) != value
                for key, value in expected_thread_settings.items()
            ):
                raise ManifestError(
                    f"{runtime_label}.thread_settings does not attest the "
                    "single-thread CPU execution contract."
                )
            if runtime.get("latency_scope") != WORKER_LATENCY_SCOPE:
                raise ManifestError(
                    f"{runtime_label}.latency_scope has an unsupported value."
                )
            image_load_wall_ns = _nonnegative_integer(
                runtime.get("image_load_wall_ns"),
                f"{runtime_label}.image_load_wall_ns",
            )
            if runtime.get("image_decode_wall_ns") != image_load_wall_ns:
                raise ManifestError(
                    f"{runtime_label}.image_decode_wall_ns must be the legacy "
                    "alias of image_load_wall_ns."
                )
            warmup_values = _array(
                runtime.get("warmup_wall_ns_samples"),
                f"{runtime_label}.warmup_wall_ns_samples",
            )
            warmup_wall_ns = [
                _nonnegative_integer(
                    value,
                    f"{runtime_label}.warmup_wall_ns_samples[{index}]",
                )
                for index, value in enumerate(warmup_values)
            ]
            timing = prediction.execution.timing
            if len(warmup_wall_ns) != timing.warmup_runs:
                raise ManifestError(
                    f"{runtime_label}.warmup_wall_ns_samples count must match "
                    "completed warm-ups."
                )
            phase_values = {
                key: _nonnegative_integer(
                    runtime.get(key),
                    f"{runtime_label}.{key}",
                )
                for key in (
                    "image_file_decode_ns",
                    "image_encode_wall_ns",
                    "edge_cache_fill_wall_ns",
                    "session_initialization_ns",
                )
            }
            if runtime.get("encoder_reused_across_predictions") is not True:
                raise ManifestError(
                    f"{runtime_label} must attest encoder state reuse."
                )
            if (
                image_load_wall_ns
                < phase_values["image_file_decode_ns"]
                + phase_values["image_encode_wall_ns"]
                or not warmup_wall_ns
                or warmup_wall_ns[0] < phase_values["edge_cache_fill_wall_ns"]
                or timing.model_load_wall_ns is None
                or phase_values["session_initialization_ns"]
                > timing.model_load_wall_ns
            ):
                raise ManifestError(
                    f"{runtime_label} has inconsistent phase timing boundaries."
                )
            raw_sam_samples = _array(
                runtime.get("sam_prediction_samples"),
                f"{runtime_label}.sam_prediction_samples",
            )
            sam_samples = [
                _sam_prediction_evidence(
                    value,
                    f"{runtime_label}.sam_prediction_samples[{index}]",
                )
                for index, value in enumerate(raw_sam_samples)
            ]
            if len(sam_samples) != timing.warmup_runs + len(
                timing.wall_ns_samples
            ):
                raise ManifestError(
                    f"{runtime_label}.sam_prediction_samples count must match "
                    "warm-up and measured repetitions."
                )
            measured_sam_samples = sam_samples[timing.warmup_runs :]
            if any(
                evidence["decoder_wall_ns"] > wall_ns
                for evidence, wall_ns in zip(
                    measured_sam_samples,
                    timing.wall_ns_samples,
                )
            ):
                raise ManifestError(
                    f"{runtime_label} decoder timing exceeds its measured prompt."
                )
            if prediction.artifact_path is None:
                raise ManifestError(
                    f"{label}.predictions.{method_id} omitted its artifact."
                )
            try:
                artifact = load_centerline_artifact(prediction.artifact_path)
            except (OSError, ValueError) as exc:
                raise ManifestError(
                    f"{label}.predictions.{method_id} artifact is invalid: {exc}"
                ) from exc
            expected_artifact_metadata = {
                "actual_backend": prediction.execution.actual_backend,
                "configuration_sha256": expected_configuration_sha256,
                "input_sha256": image_sha,
                "mask_trace_kernel": "ai_vectorizer.core.sam_trace_kernel",
                "model_bundle_id": EFFICIENTSAM_TI_SPLIT.id,
                "model_bundle_sha256": EFFICIENTSAM_BUNDLE_SHA256,
                "model_source_commit": EFFICIENTSAM_TI_SPLIT.source_commit,
                "prompt_sha256": expected_prompt_sha256,
                "requested_backend": method_id,
                "sam_prompt_tensor_sha256": expected_sam_tensor_sha256,
                "segmentation_evidence": _stable_sam_prediction_evidence(
                    sam_samples[timing.warmup_runs]
                ),
                "smoothing": PRODUCT_SMOOTHING_PROFILE,
                "trace_kernel": "ai_vectorizer.core.trace_kernel",
            }
            if (
                set(artifact.metadata) != EFFICIENTSAM_ARTIFACT_METADATA_KEYS
                or artifact.metadata != expected_artifact_metadata
            ):
                raise ManifestError(
                    f"{label}.predictions.{method_id} artifact is not bound to "
                    "its backend, input, configuration, model, prompt, and "
                    "first measured SAM prediction."
                )
        elif (
            method_id == RECOVERY_METHOD_ID
            and prediction.execution.status in {"ok", "fallback"}
        ):
            runtime_label = f"{label}.predictions.{method_id}.execution.runtime"
            observed_tensor_sha256 = runtime.get("sam_prompt_tensor_sha256")
            if observed_tensor_sha256 is not None:
                _sha256(
                    observed_tensor_sha256,
                    f"{runtime_label}.sam_prompt_tensor_sha256",
                )
            if observed_tensor_sha256 != expected_recovery_tensor_sha256:
                raise ManifestError(
                    f"{label}.predictions.{method_id} must bind the exact product "
                    "Smart Recovery prompt tensors."
                )
            method = methods_by_id[method_id]
            expected_configuration_sha256 = _configuration_sha256(
                method.configuration
            )
            if (
                prediction.execution.status != "ok"
                or prediction.execution.actual_backend != method_id
                or runtime.get("provider_kind") != "onnxruntime"
                or runtime.get("actual_provider") != "CPUExecutionProvider"
                or runtime.get("provider_device_type") != "cpu"
                or runtime.get("provider_verified") is not True
            ):
                raise ManifestError(
                    f"{label}.predictions.{method_id} requires the verified "
                    "ONNX CPU backend and permits no backend fallback."
                )
            if (
                runtime.get("input_sha256") != image_sha
                or runtime.get("configuration_sha256")
                != expected_configuration_sha256
                or runtime.get("model_bundle_id") != EFFICIENTSAM_TI_SPLIT.id
                or runtime.get("model_bundle_sha256")
                != EFFICIENTSAM_BUNDLE_SHA256
                or runtime.get("model_source_commit")
                != EFFICIENTSAM_TI_SPLIT.source_commit
                or runtime.get("model_artifacts_sha256")
                != EFFICIENTSAM_ARTIFACTS_SHA256
            ):
                raise ManifestError(
                    f"{runtime_label} is not bound to the sample, configuration, "
                    "and pinned EfficientSAM bundle."
                )
            expected_session_options = {
                "encoder": dict(EFFICIENTSAM_ORT_SESSION_OPTIONS),
                "decoder": dict(EFFICIENTSAM_ORT_SESSION_OPTIONS),
            }
            if runtime.get("onnx_session_options") != expected_session_options:
                raise ManifestError(
                    f"{runtime_label}.onnx_session_options does not match the "
                    "required encoder and decoder readback."
                )
            if prediction.artifact_path is None:
                raise ManifestError(
                    f"{label}.predictions.{method_id} omitted its artifact."
                )
            try:
                artifact = load_centerline_artifact(prediction.artifact_path)
            except (OSError, ValueError) as exc:
                raise ManifestError(
                    f"{label}.predictions.{method_id} artifact is invalid: {exc}"
                ) from exc
            expected_artifact_fields = {
                "actual_backend": method_id,
                "configuration_sha256": expected_configuration_sha256,
                "input_sha256": image_sha,
                "livewire_kernel": "ai_vectorizer.core.livewire",
                "mask_trace_kernel": "ai_vectorizer.core.sam_trace_kernel",
                "model_bundle_id": EFFICIENTSAM_TI_SPLIT.id,
                "model_bundle_sha256": EFFICIENTSAM_BUNDLE_SHA256,
                "model_source_commit": EFFICIENTSAM_TI_SPLIT.source_commit,
                "prompt_sha256": expected_prompt_sha256,
                "recovery_kernel": "ai_vectorizer.core.smart_recovery",
                "requested_backend": method_id,
                "sam_prompt_tensor_sha256": expected_recovery_tensor_sha256,
                "source_grid_input_sha256": expected_source_grid_sha256,
                "source_tile_origin_xy": list(source_tile_origin_xy),
                "smoothing": "smart-trace-livewire-v1",
                "trace_kernel": "ai_vectorizer.core.trace_kernel",
            }
            if (
                set(artifact.metadata) != RECOVERY_ARTIFACT_METADATA_KEYS
                or any(
                    artifact.metadata.get(key) != expected_value
                    for key, expected_value in expected_artifact_fields.items()
                )
                or not isinstance(artifact.metadata.get("recovery_evidence"), dict)
            ):
                raise ManifestError(
                    f"{label}.predictions.{method_id} artifact is not bound to "
                    "its backend, input, configuration, model, and actual "
                    "recovery prompt tensors."
                )
    return SampleSpec(
        identifier=_identifier(sample.get("id"), f"{label}.id"),
        width=width,
        height=height,
        image_path=image,
        image_sha256=image_sha,
        reference_path=reference,
        reference_sha256=reference_sha,
        prompt=prompt,
        strata=strata,
        source=source,
        predictions=predictions,
        source_tile_origin_xy=source_tile_origin_xy,
    )


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Load a benchmark manifest and verify every declared artifact hash."""

    manifest_path = Path(path).resolve()
    try:
        with manifest_path.open("rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ManifestError(f"Manifest exceeds {MAX_MANIFEST_BYTES} bytes.")
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_invalid_json_constant,
            parse_float=_json_float,
            parse_int=_json_integer,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ManifestError(f"Could not read benchmark manifest: {exc}") from exc
    try:
        _validate_json_numbers(payload, "manifest")
    except RecursionError as exc:
        raise ManifestError("Manifest nesting is too deep.") from exc
    root = manifest_path.parent.resolve()
    document = _object(payload, "manifest")
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported benchmark schema: {document.get('schema_version')!r}."
        )

    raw_methods = _array(document.get("methods"), "methods")
    if len(raw_methods) > MAX_METHODS:
        raise ManifestError(f"methods may contain at most {MAX_METHODS} entries.")
    methods = tuple(_method(method, index) for index, method in enumerate(raw_methods))
    if not methods:
        raise ManifestError("methods must not be empty.")
    method_ids = tuple(method.identifier for method in methods)
    if len(set(method_ids)) != len(method_ids):
        raise ManifestError("Method ids must be unique.")
    methods_by_id = {method.identifier: method for method in methods}
    efficient_sam_method = methods_by_id.get(EFFICIENTSAM_METHOD_ID)
    if efficient_sam_method is not None:
        _validate_efficientsam_method_contract(efficient_sam_method)

    raw_samples = _array(document.get("samples"), "samples")
    if len(raw_samples) > MAX_SAMPLES:
        raise ManifestError(f"samples may contain at most {MAX_SAMPLES} entries.")
    if len(raw_samples) * len(raw_methods) > MAX_EVALUATIONS:
        raise ManifestError(
            f"samples x methods may contain at most {MAX_EVALUATIONS} evaluations."
        )
    samples = tuple(
        _sample(root, sample, index, methods_by_id)
        for index, sample in enumerate(raw_samples)
    )
    if not samples:
        raise ManifestError("samples must not be empty.")
    sample_ids = [sample.identifier for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ManifestError("Sample ids must be unique.")

    timing_fingerprints = {
        (
            prediction.execution.runtime["python_version"],
            prediction.execution.runtime["platform"],
            prediction.execution.runtime["cpu"],
            json.dumps(
                {
                    key: prediction.execution.runtime["thread_settings"].get(key)
                    for key in SHARED_THREAD_SETTING_KEYS
                    if key == "threads"
                    or key in prediction.execution.runtime["thread_settings"]
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        for sample in samples
        for prediction in sample.predictions.values()
    }
    if len(timing_fingerprints) != 1:
        raise ManifestError(
            "All prediction timings must share python, platform, CPU, and thread settings."
        )

    for method_id in method_ids:
        fingerprints_by_execution = {}
        for sample in samples:
            execution = sample.predictions[method_id].execution
            runtime = execution.runtime
            group = (execution.status, execution.actual_backend)
            fingerprint = (
                runtime["adapter_version"],
                runtime["provider_kind"],
                runtime["actual_provider"],
                runtime["provider_device_type"],
                json.dumps(
                    runtime["package_versions"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                json.dumps(
                    runtime["thread_settings"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
            fingerprints_by_execution.setdefault(group, set()).add(fingerprint)
        if any(len(fingerprints) != 1 for fingerprints in fingerprints_by_execution.values()):
            raise ManifestError(
                f"Method {method_id!r} must use one adapter/provider/package "
                "fingerprint per execution status and actual backend."
            )

    for method_id, method in methods_by_id.items():
        if method.model_sha256 is None:
            continue
        for sample in samples:
            prediction = sample.predictions[method_id]
            if prediction.execution.status not in {"ok", "fallback"}:
                continue
            observed_model = prediction.execution.runtime.get(
                "model_bundle_sha256"
            )
            if observed_model != method.model_sha256:
                raise ManifestError(
                    f"Method {method_id!r} model_sha256 does not match successful "
                    f"runtime evidence for sample {sample.identifier!r}."
                )

    return BenchmarkManifest(
        path=manifest_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        dataset=_dataset(document.get("dataset")),
        metric_config=_metric_config(document.get("metric_config")),
        methods=methods,
        samples=samples,
    )
