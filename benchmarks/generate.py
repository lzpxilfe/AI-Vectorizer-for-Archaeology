"""Atomic benchmark-dataset generation through isolated CPU workers.

The evaluator consumes immutable, checksummed manifests.  This module bridges
that format to :mod:`benchmarks.worker`: it copies a valid template dataset to
a private staging directory, launches one fresh process for every
sample/method pair, merges only verified worker evidence, validates the whole
dataset, and finally publishes the directory with one rename.

The input template is never edited.  Generation also refuses to replace an
existing output directory, making reruns explicit and recoverable.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
from typing import Any, Mapping, Sequence

from .evidence import prompt_sha256, sam_prompt_tensor_sha256
from .geometry import load_centerline_artifact
from .manifest import BenchmarkManifest, MAX_MANIFEST_BYTES, load_manifest
from .runner import validate_benchmark
from .worker import (
    EFFICIENTSAM_BACKEND,
    EFFICIENTSAM_ORT_SESSION_OPTIONS,
    EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION,
    LATENCY_SCOPE,
    METHOD_EDGE_BACKENDS,
    METHOD_SAM_BACKENDS,
    PRODUCT_SMOOTHING_PROFILE,
    SUPPORTED_BACKENDS,
    WORKER_REQUEST_SCHEMA_VERSION,
    WORKER_RESULT_SCHEMA_VERSION,
)


OUTPUT_MANIFEST_NAME = "manifest.json"
MAX_WORKER_RESULT_BYTES = 4 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 600.0
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_REAL_WORKER_SOURCES = {
    "edge": (
        "benchmarks/__init__.py",
        "benchmarks/evidence.py",
        "benchmarks/geometry.py",
        "benchmarks/manifest.py",
        "benchmarks/worker.py",
        "ai_vectorizer/__init__.py",
        "ai_vectorizer/core/__init__.py",
        "ai_vectorizer/core/dependencies.py",
        "ai_vectorizer/core/edge_detector.py",
        "ai_vectorizer/core/efficientsam_spec.py",
        "ai_vectorizer/core/trace_kernel.py",
    ),
    "sam": (
        "benchmarks/__init__.py",
        "benchmarks/evidence.py",
        "benchmarks/geometry.py",
        "benchmarks/manifest.py",
        "benchmarks/worker.py",
        "ai_vectorizer/__init__.py",
        "ai_vectorizer/core/__init__.py",
        "ai_vectorizer/core/dependencies.py",
        "ai_vectorizer/core/edge_detector.py",
        "ai_vectorizer/core/efficientsam_onnx.py",
        "ai_vectorizer/core/efficientsam_spec.py",
        "ai_vectorizer/core/model_store.py",
        "ai_vectorizer/core/sam_trace_kernel.py",
        "ai_vectorizer/core/trace_kernel.py",
    ),
}


class GenerationError(RuntimeError):
    """Raised when a dataset cannot be generated without losing evidence."""


def _strict_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GenerationError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise GenerationError(f"Non-standard JSON number is not allowed: {value}.")


def _read_json_object(path: Path, limit: int, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise GenerationError(f"Could not read {label}: {exc}") from exc
    if len(raw) > limit:
        raise GenerationError(f"{label} exceeds {limit} bytes.")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_invalid_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise GenerationError(f"Invalid {label} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GenerationError(f"{label} root must be an object.")
    return payload


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise GenerationError(f"Could not serialize generation record: {exc}") from exc


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configuration_sha256(configuration: Mapping[str, Any]) -> str:
    """Hash configuration bytes exactly as the isolated worker does."""

    try:
        raw = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise GenerationError(f"Could not hash worker configuration: {exc}") from exc
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nonnegative_integer(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _worker_source_paths(
    repository_root: Path,
    worker_command: Sequence[str] | None,
    backend: str | None = None,
) -> dict[str, Path]:
    """Resolve the source files whose bytes define one generation run."""

    if worker_command is None:
        family = "sam" if backend in METHOD_SAM_BACKENDS else "edge"
        return {
            label: repository_root / label
            for label in _REAL_WORKER_SOURCES[family]
        }

    source_paths: dict[str, Path] = {}
    for argument in worker_command[1:]:
        raw_path = Path(argument)
        candidate = (
            raw_path if raw_path.is_absolute() else repository_root / raw_path
        ).resolve()
        if not candidate.is_file():
            continue
        try:
            label = candidate.relative_to(repository_root).as_posix()
        except ValueError:
            label = candidate.name
        if label in source_paths and source_paths[label] != candidate:
            raise GenerationError(f"Duplicate worker source label: {label!r}.")
        source_paths[label] = candidate
    if not source_paths:
        raise GenerationError(
            "worker_command must include at least one readable source-file argument."
        )
    return source_paths


def _source_snapshot(source_paths: Mapping[str, Path]) -> dict[str, str]:
    try:
        return {
            label: _sha256_file(source_path)
            for label, source_path in source_paths.items()
        }
    except OSError as exc:
        raise GenerationError(f"Could not hash worker source files: {exc}") from exc


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory only when the destination is absent."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    result: int
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:  # pragma: no cover - supported on target macOS.
            raise GenerationError("Atomic no-replace directory publishing is unavailable.")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        # RENAME_EXCL: fail if the destination appeared after our initial check.
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - old/non-glibc Linux.
            raise GenerationError("Atomic no-replace directory publishing is unavailable.")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)
    elif os.name == "nt":  # Windows rename already refuses an existing target.
        os.rename(source, destination)
        return
    else:  # pragma: no cover - fail closed on unsupported operating systems.
        raise GenerationError("Atomic no-replace directory publishing is unavailable.")

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _relative_path(*parts: str) -> str:
    path = Path(*parts)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise GenerationError(f"Unsafe generated relative path: {path}")
    return path.as_posix()


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    actual = _sha256_file(destination)
    if actual != expected_sha256:
        raise GenerationError(
            f"Copied asset checksum changed: expected {expected_sha256}, got {actual}."
        )


def _request_payload(
    *,
    request_id: str,
    backend: str,
    fallback_backend: str | None,
    sample,
    method,
    image_path: str,
    artifact_path: str,
    warmup_runs: int,
    measurement_runs: int,
    threads: int,
    model_cache: Path | None = None,
) -> dict[str, Any]:
    prompt = sample.prompt
    payload = {
        "schema_version": WORKER_REQUEST_SCHEMA_VERSION,
        "request_id": request_id,
        "requested_backend": backend,
        "fallback_backend": fallback_backend,
        "device": "cpu",
        "image": {
            "path": image_path,
            "sha256": sample.image_sha256,
            "width": sample.width,
            "height": sample.height,
        },
        "artifact": artifact_path,
        "prompt": {
            "start_xy": list(prompt.start_xy),
            "end_xy": list(prompt.end_xy),
            "positive_xy": [list(point) for point in prompt.positive_xy],
            "negative_xy": [list(point) for point in prompt.negative_xy],
        },
        "configuration": copy.deepcopy(method.configuration),
        "warmup_runs": warmup_runs,
        "measurement_runs": measurement_runs,
        "threads": threads,
    }
    if backend in METHOD_SAM_BACKENDS:
        if model_cache is None:
            raise GenerationError("EfficientSAM generation requires --model-cache.")
        payload["model_cache"] = str(model_cache.absolute())
    return payload


def _run_process(
    command_prefix: Sequence[str],
    request_path: Path,
    result_path: Path,
    *,
    repository_root: Path,
    threads: int,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for variable in _THREAD_VARIABLES:
        environment[variable] = str(threads)
    environment["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        return subprocess.run(
            [*command_prefix, str(request_path.resolve()), str(result_path.resolve())],
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GenerationError(
            f"Worker process for {request_path.name} could not complete: {exc}"
        ) from exc


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


def _validated_sam_prediction_sample(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _SAM_PREDICTION_EVIDENCE_KEYS:
        raise GenerationError(f"{label} has an unsupported structure.")
    if value.get("schema_version") != EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION:
        raise GenerationError(f"{label} has an unsupported schema.")
    selected_index = value.get("selected_mask_index")
    if (
        isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not 0 <= selected_index < 3
    ):
        raise GenerationError(f"{label} has an invalid selected mask index.")
    raw_ious = value.get("iou_predictions")
    if not isinstance(raw_ious, list) or len(raw_ious) != 3:
        raise GenerationError(f"{label} must contain three IoU predictions.")
    ious: list[float] = []
    for raw_iou in raw_ious:
        if isinstance(raw_iou, bool) or not isinstance(raw_iou, (int, float)):
            raise GenerationError(f"{label} IoU predictions must be numeric.")
        iou = float(raw_iou)
        if not math.isfinite(iou):
            raise GenerationError(f"{label} IoU predictions must be finite.")
        try:
            iou = struct.unpack("<f", struct.pack("<f", iou))[0]
        except (OverflowError, struct.error) as exc:
            raise GenerationError(
                f"{label} IoU predictions cannot be represented as float32."
            ) from exc
        if not math.isfinite(iou):
            raise GenerationError(
                f"{label} IoU predictions must remain finite as float32."
            )
        ious.append(iou)
    if selected_index != max(range(len(ious)), key=ious.__getitem__):
        raise GenerationError(
            f"{label} selected mask index disagrees with maximum predicted IoU."
        )
    for key in (
        "iou_predictions_sha256",
        "selected_logits_sha256",
        "selected_binary_mask_sha256",
        "accepted_mask_sha256",
    ):
        if not _is_sha256(value.get(key)):
            raise GenerationError(f"{label}.{key} must be a SHA-256 digest.")
    try:
        expected_iou_sha256 = hashlib.sha256(struct.pack("<3f", *ious)).hexdigest()
    except (OverflowError, struct.error) as exc:
        raise GenerationError(
            f"{label} IoU predictions cannot be represented as float32."
        ) from exc
    if value["iou_predictions_sha256"] != expected_iou_sha256:
        raise GenerationError(
            f"{label}.iou_predictions_sha256 disagrees with its float32 values."
        )
    decoder_wall_ns = value.get("decoder_wall_ns")
    if (
        isinstance(decoder_wall_ns, bool)
        or not isinstance(decoder_wall_ns, int)
        or decoder_wall_ns < 0
    ):
        raise GenerationError(
            f"{label}.decoder_wall_ns must be a non-negative integer."
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


def _stable_sam_prediction_sample(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: ([*item] if key == "iou_predictions" else item)
        for key, item in value.items()
        if key != "decoder_wall_ns"
    }


def _verified_prediction(
    result_path: Path,
    *,
    expected_request_id: str,
    expected_backend: str,
    expected_fallback_backend: str | None,
    expected_artifact: str,
    expected_input_sha256: str,
    expected_configuration_sha256: str,
    expected_prompt_sha256: str,
    expected_sam_prompt_tensor_sha256: str | None,
    expected_source_files_sha256: Mapping[str, str],
    expected_model_sha256: str | None,
    expected_model_artifacts_sha256: Mapping[str, str] | None,
    expected_model_bundle_id: str | None,
    expected_model_source_commit: str | None,
    expected_width: int,
    expected_height: int,
    expected_threads: int,
    expected_warmup_runs: int,
    expected_measurement_runs: int,
    stage_root: Path,
    return_code: int,
    stderr: str,
) -> dict[str, Any]:
    if not result_path.is_file():
        detail = " ".join(stderr.split())[:1_000]
        raise GenerationError(
            f"Worker {expected_request_id!r} exited {return_code} without a result"
            + (f": {detail}" if detail else ".")
        )
    result = _read_json_object(
        result_path,
        MAX_WORKER_RESULT_BYTES,
        f"worker result {expected_request_id}",
    )
    if result.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION:
        raise GenerationError(f"Worker {expected_request_id!r} used an unsupported result schema.")
    if result.get("request_id") != expected_request_id:
        raise GenerationError(f"Worker result request_id does not match {expected_request_id!r}.")
    prediction = result.get("prediction")
    if not isinstance(prediction, dict):
        raise GenerationError(f"Worker {expected_request_id!r} omitted prediction evidence.")
    execution = prediction.get("execution")
    if not isinstance(execution, dict):
        raise GenerationError(f"Worker {expected_request_id!r} omitted execution evidence.")
    status = execution.get("status")
    if status not in {"ok", "fallback", "failed"}:
        raise GenerationError(f"Worker {expected_request_id!r} returned an invalid status.")
    if execution.get("requested_backend") != expected_backend:
        raise GenerationError(
            f"Worker {expected_request_id!r} changed requested_backend evidence."
        )
    if execution.get("device") != "cpu":
        raise GenerationError(f"Worker {expected_request_id!r} changed CPU device evidence.")
    successful = status in {"ok", "fallback"}

    actual_backend = execution.get("actual_backend")
    fallback_reason = execution.get("fallback_reason")
    error = execution.get("error")
    if status == "ok":
        if actual_backend != expected_backend or fallback_reason is not None or error is not None:
            raise GenerationError(f"Worker {expected_request_id!r} returned inconsistent ok evidence.")
    elif status == "fallback":
        if (
            expected_fallback_backend is None
            or actual_backend != expected_fallback_backend
            or not isinstance(fallback_reason, str)
            or not fallback_reason.strip()
            or error is not None
        ):
            raise GenerationError(
                f"Worker {expected_request_id!r} returned an unconfigured fallback."
            )
    elif (
        actual_backend not in {None, expected_backend, expected_fallback_backend}
        or fallback_reason is not None
        or not isinstance(error, str)
        or not error.strip()
    ):
        raise GenerationError(f"Worker {expected_request_id!r} returned inconsistent failure evidence.")

    runtime = execution.get("runtime")
    timing = execution.get("timing")
    if not isinstance(runtime, dict) or not isinstance(timing, dict):
        raise GenerationError(f"Worker {expected_request_id!r} omitted runtime or timing evidence.")
    if runtime.get("input_sha256") != expected_input_sha256:
        raise GenerationError(f"Worker {expected_request_id!r} changed input checksum evidence.")
    if runtime.get("configuration_sha256") != expected_configuration_sha256:
        raise GenerationError(
            f"Worker {expected_request_id!r} changed configuration checksum evidence."
        )
    if runtime.get("prompt_sha256") != expected_prompt_sha256:
        raise GenerationError(
            f"Worker {expected_request_id!r} changed prompt checksum evidence."
        )
    provider_backend = actual_backend or expected_backend
    expected_provider_kind = (
        "onnxruntime" if provider_backend in METHOD_SAM_BACKENDS else "opencv"
    )
    expected_provider_name = (
        "CPUExecutionProvider"
        if expected_provider_kind == "onnxruntime"
        else "OpenCV CPU"
    )
    if (
        runtime.get("provider_kind") != expected_provider_kind
        or runtime.get("actual_provider") != expected_provider_name
        or runtime.get("provider_device_type") != "cpu"
        or not isinstance(runtime.get("provider_verified"), bool)
    ):
        raise GenerationError(f"Worker {expected_request_id!r} changed provider evidence.")

    thread_settings = runtime.get("thread_settings")
    common_thread_error = (
        not isinstance(thread_settings, dict)
        or isinstance(thread_settings.get("threads"), bool)
        or thread_settings.get("threads") != expected_threads
        or any(
            thread_settings.get(variable) != str(expected_threads)
            for variable in _THREAD_VARIABLES
        )
    )
    if expected_provider_kind == "onnxruntime":
        provider_thread_error = (
            not isinstance(thread_settings, dict)
            or thread_settings.get("onnx_intra_op_num_threads") != 1
            or thread_settings.get("onnx_inter_op_num_threads") != 1
            or thread_settings.get("onnx_execution_mode") != "ORT_SEQUENTIAL"
            or thread_settings.get("onnx_graph_optimization_level")
            != "ORT_ENABLE_ALL"
            or thread_settings.get("opencv_set_num_threads") != 0
            or thread_settings.get("opencv_effective_num_threads")
            != expected_threads
            or thread_settings.get("opencl") is not False
        )
    else:
        provider_thread_error = (
            not isinstance(thread_settings, dict)
            or thread_settings.get("opencv_set_num_threads") != 0
            or thread_settings.get("opencl") is not False
        )
    if common_thread_error or provider_thread_error:
        raise GenerationError(f"Worker {expected_request_id!r} changed CPU thread evidence.")

    if expected_backend in METHOD_SAM_BACKENDS:
        if (
            not _is_sha256(expected_model_sha256)
            or runtime.get("model_bundle_sha256") != expected_model_sha256
            or runtime.get("model_bundle_id") != expected_model_bundle_id
            or runtime.get("model_source_commit")
            != expected_model_source_commit
            or runtime.get("model_artifacts_sha256")
            != dict(expected_model_artifacts_sha256 or {})
            or runtime.get("sam_prompt_tensor_sha256")
            != expected_sam_prompt_tensor_sha256
        ):
            raise GenerationError(
                f"Worker {expected_request_id!r} changed pinned model evidence."
            )
        if runtime.get("onnx_providers") != {
            "encoder": ["CPUExecutionProvider"],
            "decoder": ["CPUExecutionProvider"],
        }:
            raise GenerationError(
                f"Worker {expected_request_id!r} changed raw ONNX provider evidence."
            )
        if runtime.get("onnx_session_options") != {
            "encoder": dict(EFFICIENTSAM_ORT_SESSION_OPTIONS),
            "decoder": dict(EFFICIENTSAM_ORT_SESSION_OPTIONS),
        }:
            raise GenerationError(
                f"Worker {expected_request_id!r} changed raw ONNX session evidence."
            )
    source_hashes = runtime.get("source_files_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(
            not isinstance(name, str) or not name or not _is_sha256(digest)
            for name, digest in source_hashes.items()
        )
    ):
        raise GenerationError(f"Worker {expected_request_id!r} omitted source-file checksums.")
    expected_sources = dict(expected_source_files_sha256)
    if successful:
        source_evidence_matches = source_hashes == expected_sources
    else:
        source_evidence_matches = (
            set(source_hashes).issubset(expected_sources)
            and all(expected_sources[name] == digest for name, digest in source_hashes.items())
        )
    if not source_evidence_matches:
        raise GenerationError(
            f"Worker {expected_request_id!r} source-file checksums changed during generation."
        )

    warmup_completed = timing.get("warmup_runs")
    wall_samples = timing.get("wall_ns_samples")
    cpu_samples = timing.get("cpu_ns_samples")
    output_hashes = runtime.get("output_sha256_samples")
    if (
        isinstance(warmup_completed, bool)
        or not isinstance(warmup_completed, int)
        or not 0 <= warmup_completed <= expected_warmup_runs
        or not isinstance(wall_samples, list)
        or not isinstance(cpu_samples, list)
        or not isinstance(output_hashes, list)
        or len(wall_samples) != len(cpu_samples)
        or len(wall_samples) != len(output_hashes)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in wall_samples)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cpu_samples)
        or any(not _is_sha256(value) for value in output_hashes)
    ):
        raise GenerationError(f"Worker {expected_request_id!r} returned inconsistent repetition evidence.")
    image_load_wall_ns = runtime.get("image_load_wall_ns")
    legacy_image_decode_wall_ns = runtime.get("image_decode_wall_ns")
    warmup_wall_samples = runtime.get("warmup_wall_ns_samples")
    if (
        runtime.get("latency_scope") != LATENCY_SCOPE
        or not isinstance(warmup_wall_samples, list)
        or len(warmup_wall_samples) != warmup_completed
        or any(
            not _is_nonnegative_integer(value)
            for value in warmup_wall_samples
        )
        or (
            image_load_wall_ns is not None
            and not _is_nonnegative_integer(image_load_wall_ns)
        )
        or legacy_image_decode_wall_ns != image_load_wall_ns
    ):
        raise GenerationError(
            f"Worker {expected_request_id!r} returned inconsistent latency-scope evidence."
        )
    if successful and not _is_nonnegative_integer(image_load_wall_ns):
        raise GenerationError(
            f"Worker {expected_request_id!r} omitted image preparation timing."
        )
    deterministic = runtime.get("deterministic")
    observed_deterministic = len(set(output_hashes)) <= 1 if output_hashes else None
    if deterministic != observed_deterministic:
        raise GenerationError(f"Worker {expected_request_id!r} changed determinism evidence.")

    validated_sam_samples: list[dict[str, Any]] = []
    if expected_backend in METHOD_SAM_BACKENDS:
        raw_sam_samples = runtime.get("sam_prediction_samples")
        if not isinstance(raw_sam_samples, list):
            raise GenerationError(
                f"Worker {expected_request_id!r} omitted repeated SAM prediction evidence."
            )
        validated_sam_samples = [
            _validated_sam_prediction_sample(
                sample,
                f"Worker {expected_request_id!r} SAM prediction sample[{index}]",
            )
            for index, sample in enumerate(raw_sam_samples)
        ]
        if successful and len(validated_sam_samples) != (
            expected_warmup_runs + expected_measurement_runs
        ):
            raise GenerationError(
                f"Worker {expected_request_id!r} returned the wrong number of "
                "SAM prediction samples."
            )
        if not successful and len(validated_sam_samples) > (
            expected_warmup_runs + expected_measurement_runs
        ):
            raise GenerationError(
                f"Worker {expected_request_id!r} exceeded requested SAM predictions."
            )
        measured_sam_samples = validated_sam_samples[
            warmup_completed : warmup_completed + len(wall_samples)
        ]
        if len(measured_sam_samples) != len(wall_samples) or any(
            sample["decoder_wall_ns"] > wall_ns
            for sample, wall_ns in zip(measured_sam_samples, wall_samples)
        ):
            raise GenerationError(
                f"Worker {expected_request_id!r} returned inconsistent SAM phase timing."
            )
        if successful:
            image_file_decode_ns = runtime.get("image_file_decode_ns")
            image_encode_wall_ns = runtime.get("image_encode_wall_ns")
            edge_cache_fill_wall_ns = runtime.get("edge_cache_fill_wall_ns")
            session_initialization_ns = runtime.get("session_initialization_ns")
            model_load_wall_ns = timing.get("model_load_wall_ns")
            if any(
                not _is_nonnegative_integer(value)
                for value in (
                    image_file_decode_ns,
                    image_encode_wall_ns,
                    edge_cache_fill_wall_ns,
                    session_initialization_ns,
                    model_load_wall_ns,
                )
            ):
                raise GenerationError(
                    f"Worker {expected_request_id!r} omitted SAM phase timing evidence."
                )
            if (
                runtime.get("encoder_reused_across_predictions") is not True
                or image_load_wall_ns
                < image_file_decode_ns + image_encode_wall_ns
                or not warmup_wall_samples
                or warmup_wall_samples[0] < edge_cache_fill_wall_ns
                or session_initialization_ns > model_load_wall_ns
            ):
                raise GenerationError(
                    f"Worker {expected_request_id!r} returned inconsistent SAM phase boundaries."
                )

    if successful:
        if return_code != 0:
            raise GenerationError(
                f"Successful worker {expected_request_id!r} exited with code {return_code}."
            )
        if runtime.get("provider_verified") is not True:
            raise GenerationError(f"Worker {expected_request_id!r} did not verify its provider.")
        if warmup_completed != expected_warmup_runs or len(wall_samples) != expected_measurement_runs:
            raise GenerationError(
                f"Worker {expected_request_id!r} did not complete the requested repetitions."
            )
    else:
        if return_code != 3:
            raise GenerationError(
                f"Failed worker {expected_request_id!r} exited with code {return_code}, not 3."
            )
        if len(wall_samples) > expected_measurement_runs:
            raise GenerationError(f"Worker {expected_request_id!r} exceeded requested repetitions.")
        if prediction.get("artifact") is not None or prediction.get("sha256") is not None:
            raise GenerationError(
                f"Failed worker {expected_request_id!r} must not publish an artifact."
            )
        if os.path.lexists(stage_root / expected_artifact):
            raise GenerationError(
                f"Failed worker {expected_request_id!r} left an artifact behind."
            )
        return copy.deepcopy(prediction)

    if prediction.get("artifact") != expected_artifact:
        raise GenerationError(
            f"Worker {expected_request_id!r} changed the canonical artifact path."
        )
    artifact_path_unresolved = stage_root / expected_artifact
    current = stage_root
    for part in Path(expected_artifact).parts:
        current = current / part
        if current.is_symlink():
            raise GenerationError(
                f"Worker {expected_request_id!r} published through a symbolic link."
            )
    artifact_path = artifact_path_unresolved.resolve()
    try:
        artifact_path.relative_to(stage_root.resolve())
    except ValueError as exc:  # Defensive: expected_artifact is generated locally.
        raise GenerationError("Generated artifact path escaped the staging dataset.") from exc
    if not artifact_path.is_file():
        raise GenerationError(f"Worker {expected_request_id!r} did not publish its artifact.")
    actual_digest = _sha256_file(artifact_path)
    if prediction.get("sha256") != actual_digest:
        raise GenerationError(
            f"Worker {expected_request_id!r} artifact checksum does not match its bytes."
        )
    artifact = load_centerline_artifact(artifact_path)
    if artifact.sha256 != actual_digest:
        raise GenerationError(f"Worker {expected_request_id!r} artifact hash changed while loading.")
    if artifact.width != expected_width or artifact.height != expected_height:
        raise GenerationError(f"Worker {expected_request_id!r} changed artifact dimensions.")
    expected_metadata = {
        "actual_backend": actual_backend,
        "configuration_sha256": expected_configuration_sha256,
        "input_sha256": expected_input_sha256,
        "prompt_sha256": expected_prompt_sha256,
        "requested_backend": expected_backend,
        "smoothing": PRODUCT_SMOOTHING_PROFILE,
        "trace_kernel": "ai_vectorizer.core.trace_kernel",
    }
    if expected_backend in METHOD_SAM_BACKENDS:
        expected_metadata.update(
            mask_trace_kernel="ai_vectorizer.core.sam_trace_kernel",
            model_bundle_id=expected_model_bundle_id,
            model_bundle_sha256=expected_model_sha256,
            model_source_commit=expected_model_source_commit,
            sam_prompt_tensor_sha256=expected_sam_prompt_tensor_sha256,
            segmentation_evidence=_stable_sam_prediction_sample(
                validated_sam_samples[expected_warmup_runs]
            ),
        )
    for key, expected_value in expected_metadata.items():
        if artifact.metadata.get(key) != expected_value:
            raise GenerationError(
                f"Worker {expected_request_id!r} artifact metadata changed {key!r}."
            )
    if not output_hashes or output_hashes[0] != actual_digest:
        raise GenerationError(
            f"Worker {expected_request_id!r} published bytes other than its first measured output."
        )
    return copy.deepcopy(prediction)


def _validate_parameters(
    *,
    warmup_runs: int,
    measurement_runs: int,
    threads: int,
    timeout_seconds: float,
) -> None:
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or warmup_runs < 1:
        raise GenerationError("warmup_runs must be a positive integer.")
    if (
        isinstance(measurement_runs, bool)
        or not isinstance(measurement_runs, int)
        or not 3 <= measurement_runs <= 100
    ):
        raise GenerationError("measurement_runs must be between 3 and 100.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads != 1:
        raise GenerationError("M1.1 generation requires threads=1.")
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds):
        raise GenerationError("timeout_seconds must be finite and positive.")
    if timeout_seconds <= 0:
        raise GenerationError("timeout_seconds must be finite and positive.")


def _resolve_required_models(
    methods: Sequence[Any],
    model_cache: str | Path | None,
) -> tuple[Path | None, dict[str, str], str | None, str | None]:
    """Offline-preflight every model-backed method before staging begins."""

    sam_methods = [
        method for method in methods if method.identifier in METHOD_SAM_BACKENDS
    ]
    if not sam_methods:
        return None, {}, None, None
    if model_cache is None:
        raise GenerationError(
            "EfficientSAM templates require --model-cache; run `model fetch` first."
        )
    cache_root = Path(os.path.abspath(os.fspath(model_cache)))
    try:
        from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT
        from ai_vectorizer.core.model_store import bundle_fingerprint, resolve_bundle

        verified = resolve_bundle(cache_root, EFFICIENTSAM_TI_SPLIT)
        fingerprint = bundle_fingerprint(EFFICIENTSAM_TI_SPLIT)
    except Exception as exc:
        raise GenerationError(
            f"Pinned EfficientSAM split model is not ready in {cache_root}: {exc}"
        ) from exc

    artifact_hashes = {
        artifact.id: artifact.sha256
        for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
    }
    # Re-read verified objects into immutable bytes during preflight. This is
    # deliberately redundant with each worker's verification and catches a
    # corrupt or swapped cache before an output parent/staging tree is made.
    for artifact_id in artifact_hashes:
        verified.read_bytes(artifact_id)

    for method in sam_methods:
        if method.model_sha256 != fingerprint:
            raise GenerationError(
                f"Method {method.identifier!r} model_sha256 must equal the pinned "
                "EfficientSAM bundle fingerprint."
            )
        if method.configuration.get("model_bundle_id") != EFFICIENTSAM_TI_SPLIT.id:
            raise GenerationError(
                f"Method {method.identifier!r} has the wrong model_bundle_id."
            )
        if method.configuration.get("model_bundle_sha256") != fingerprint:
            raise GenerationError(
                f"Method {method.identifier!r} has the wrong model_bundle_sha256."
            )
    return (
        cache_root,
        artifact_hashes,
        EFFICIENTSAM_TI_SPLIT.id,
        EFFICIENTSAM_TI_SPLIT.source_commit,
    )


def _validate_model_backed_samples(template: BenchmarkManifest) -> None:
    if not any(
        method.identifier in METHOD_SAM_BACKENDS for method in template.methods
    ):
        return
    for sample in template.samples:
        if (sample.width, sample.height) != (1024, 1024):
            raise GenerationError(
                "M1.2 EfficientSAM samples must be exactly 1024x1024."
            )
        prompt = sample.prompt
        points = (
            prompt.start_xy,
            *prompt.positive_xy,
            prompt.end_xy,
            *prompt.negative_xy,
        )
        if len(points) > 6:
            raise GenerationError(
                "EfficientSAM start/end and guide points may total at most six."
            )
        if len(set(points)) != len(points):
            raise GenerationError(
                "EfficientSAM guide points must not duplicate start, end, or each other."
            )
def generate_benchmark_dataset(
    template_manifest: str | Path,
    output_directory: str | Path,
    *,
    python_executable: str | None = None,
    worker_command: Sequence[str] | None = None,
    warmup_runs: int = 1,
    measurement_runs: int = 3,
    threads: int = 1,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    fallback_backends: Mapping[str, str | None] | None = None,
    model_cache: str | Path | None = None,
) -> BenchmarkManifest:
    """Generate and atomically publish a checksummed benchmark dataset.

    ``template_manifest`` must already be a valid schema-v1 dataset whose
    method ids are worker backends (currently ``canny-adaptive-v1`` and/or
    ``lsd-adaptive-v1``).  Existing prediction records act only as schema-valid
    placeholders and are replaced in the staged copy.  Every worker launch is
    a separate process.

    ``worker_command`` is primarily an integration-test seam.  It must be a
    command prefix accepting ``REQUEST.json RESULT.json`` as its final two
    arguments.  Production callers should leave it unset.
    """

    _validate_parameters(
        warmup_runs=warmup_runs,
        measurement_runs=measurement_runs,
        threads=threads,
        timeout_seconds=timeout_seconds,
    )
    repository_root = Path(__file__).resolve().parents[1]
    command_prefix = tuple(worker_command or (
        python_executable or sys.executable,
        "-m",
        "benchmarks.worker",
    ))
    if not command_prefix or any(
        not isinstance(part, str) or not part for part in command_prefix
    ):
        raise GenerationError("worker_command must contain non-empty string arguments.")
    template_path = Path(template_manifest).resolve()
    output_root = Path(output_directory).absolute()
    if os.path.lexists(output_root):
        raise GenerationError(f"Output directory already exists: {output_root}")
    resolved_output_root = output_root.resolve()
    template_root = template_path.parent.resolve()
    try:
        resolved_output_root.relative_to(template_root)
    except ValueError:
        pass
    else:
        raise GenerationError("Output directory must not be inside the template dataset.")
    try:
        template_root.relative_to(resolved_output_root)
    except ValueError:
        pass
    else:
        raise GenerationError("Output directory must not contain the template dataset.")

    template = validate_benchmark(template_path)
    _validate_model_backed_samples(template)
    method_ids = tuple(method.identifier for method in template.methods)
    unsupported = sorted(set(method_ids) - set(SUPPORTED_BACKENDS))
    if unsupported:
        raise GenerationError(
            "Template methods must be isolated worker backends; unsupported: "
            + ", ".join(unsupported)
        )
    fallback_map = dict(fallback_backends or {})
    unknown_fallback_keys = sorted(set(fallback_map) - set(method_ids))
    if unknown_fallback_keys:
        raise GenerationError(
            "Fallback mappings reference unknown methods: "
            + ", ".join(unknown_fallback_keys)
        )
    for backend, fallback in fallback_map.items():
        if fallback is not None and (
            fallback not in SUPPORTED_BACKENDS
            or fallback == backend
            or (fallback in METHOD_SAM_BACKENDS)
            != (backend in METHOD_SAM_BACKENDS)
        ):
            raise GenerationError(
                f"Invalid same-family fallback backend for {backend!r}: "
                f"{fallback!r}"
            )
    (
        resolved_model_cache,
        model_artifact_hashes,
        model_bundle_id,
        model_source_commit,
    ) = _resolve_required_models(template.methods, model_cache)
    source_paths_by_backend = {
        backend: _worker_source_paths(
            repository_root,
            None if worker_command is None else command_prefix,
            backend,
        )
        for backend in method_ids
    }
    expected_sources_by_backend = {
        backend: _source_snapshot(paths)
        for backend, paths in source_paths_by_backend.items()
    }
    if _sha256_file(template_path) != template.sha256:
        raise GenerationError("Template manifest changed after validation.")
    document = _read_json_object(template_path, MAX_MANIFEST_BYTES, "template manifest")
    if _sha256_file(template_path) != template.sha256:
        raise GenerationError("Template manifest changed while generation was starting.")
    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list):
        raise GenerationError("Template samples must be an array.")
    sample_documents = {
        sample.get("id"): sample
        for sample in raw_samples
        if isinstance(sample, dict) and isinstance(sample.get("id"), str)
    }
    if len(sample_documents) != len(template.samples):
        raise GenerationError("Template sample ids changed between validation and generation.")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        )
    ).resolve()
    published = False

    try:
        methods_by_id = {method.identifier: method for method in template.methods}
        for sample in template.samples:
            raw_sample = sample_documents[sample.identifier]
            image_suffix = sample.image_path.suffix.lower() or ".img"
            image_relative = _relative_path("images", f"{sample.identifier}{image_suffix}")
            reference_relative = _relative_path("references", f"{sample.identifier}.json")
            _copy_verified(
                sample.image_path,
                staging / image_relative,
                sample.image_sha256,
            )
            _copy_verified(
                sample.reference_path,
                staging / reference_relative,
                sample.reference_sha256,
            )
            raw_sample["image"] = image_relative
            raw_sample["image_sha256"] = sample.image_sha256
            raw_sample["reference"] = reference_relative
            raw_sample["reference_sha256"] = sample.reference_sha256
            generated_predictions: dict[str, Any] = {}

            for method_id in method_ids:
                method = methods_by_id[method_id]
                request_id = f"{sample.identifier}--{method_id}"
                artifact_relative = _relative_path(
                    "predictions",
                    method_id,
                    f"{sample.identifier}.json",
                )
                request_relative = _relative_path(f"worker-request--{request_id}.json")
                result_relative = _relative_path(f"worker-result--{request_id}.json")
                request_path = staging / request_relative
                result_path = staging / result_relative
                request = _request_payload(
                    request_id=request_id,
                    backend=method_id,
                    fallback_backend=fallback_map.get(method_id),
                    sample=sample,
                    method=method,
                    image_path=image_relative,
                    artifact_path=artifact_relative,
                    warmup_runs=warmup_runs,
                    measurement_runs=measurement_runs,
                    threads=threads,
                    model_cache=resolved_model_cache,
                )
                _atomic_write(request_path, _json_bytes(request))
                process = _run_process(
                    command_prefix,
                    request_path,
                    result_path,
                    repository_root=repository_root,
                    threads=threads,
                    timeout_seconds=float(timeout_seconds),
                )
                prediction = _verified_prediction(
                    result_path,
                    expected_request_id=request_id,
                    expected_backend=method_id,
                    expected_fallback_backend=fallback_map.get(method_id),
                    expected_artifact=artifact_relative,
                    expected_input_sha256=sample.image_sha256,
                    expected_configuration_sha256=_configuration_sha256(
                        method.configuration
                    ),
                    expected_prompt_sha256=prompt_sha256(sample.prompt),
                    expected_sam_prompt_tensor_sha256=(
                        sam_prompt_tensor_sha256(sample.prompt)
                        if method_id in METHOD_SAM_BACKENDS
                        else None
                    ),
                    expected_source_files_sha256=expected_sources_by_backend[method_id],
                    expected_model_sha256=method.model_sha256,
                    expected_model_artifacts_sha256=(
                        model_artifact_hashes
                        if method_id in METHOD_SAM_BACKENDS
                        else None
                    ),
                    expected_model_bundle_id=(
                        model_bundle_id
                        if method_id in METHOD_SAM_BACKENDS
                        else None
                    ),
                    expected_model_source_commit=(
                        model_source_commit
                        if method_id in METHOD_SAM_BACKENDS
                        else None
                    ),
                    expected_width=sample.width,
                    expected_height=sample.height,
                    expected_threads=threads,
                    expected_warmup_runs=warmup_runs,
                    expected_measurement_runs=measurement_runs,
                    stage_root=staging,
                    return_code=process.returncode,
                    stderr=process.stderr,
                )
                # Request/result files are a private IPC detail. In particular,
                # model-backed requests contain the local cache path, which is
                # neither reproducibility evidence nor safe to publish.
                request_path.unlink()
                result_path.unlink()
                generated_predictions[method_id] = prediction
            raw_sample["predictions"] = generated_predictions

        manifest_path = staging / OUTPUT_MANIFEST_NAME
        _atomic_write(manifest_path, _json_bytes(document))
        # Both calls intentionally reload bytes and hashes instead of trusting
        # the in-memory document assembled above.
        load_manifest(manifest_path)
        validate_benchmark(manifest_path)

        for backend, source_paths in source_paths_by_backend.items():
            if _source_snapshot(source_paths) != expected_sources_by_backend[backend]:
                raise GenerationError("Worker source files changed during generation.")

        _rename_no_replace(staging, output_root)
        published = True
        published_manifest = output_root / OUTPUT_MANIFEST_NAME
        load_manifest(published_manifest)
        return validate_benchmark(published_manifest)
    except GenerationError:
        raise
    except Exception as exc:
        raise GenerationError(f"Benchmark dataset generation failed: {exc}") from exc
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "GenerationError",
    "OUTPUT_MANIFEST_NAME",
    "generate_benchmark_dataset",
]
