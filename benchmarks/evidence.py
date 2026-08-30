"""Canonical hashes that bind benchmark outputs to interactive prompts.

The helpers in this module intentionally depend only on the Python standard
library.  They are shared by the manifest loader, generator, and isolated
worker so that each boundary independently derives the same evidence instead
of trusting a hash reported by another process.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any, Mapping, Sequence

PROMPT_EVIDENCE_SCHEMA_VERSION_V1 = "archaeotrace-trace-prompt/1"
PROMPT_EVIDENCE_SCHEMA_VERSION = "archaeotrace-trace-prompt/2"
PROMPT_EVIDENCE_SCHEMA_VERSIONS = frozenset(
    {PROMPT_EVIDENCE_SCHEMA_VERSION_V1, PROMPT_EVIDENCE_SCHEMA_VERSION}
)
SAM_TENSOR_EVIDENCE_SCHEMA_VERSION = "archaeotrace-efficientsam-prompt-tensors/1"
SOURCE_GRID_INPUT_EVIDENCE_SCHEMA_VERSION = (
    "archaeotrace-source-grid-input/1"
)


def _field(prompt: Any, name: str) -> Any:
    if isinstance(prompt, Mapping):
        if name not in prompt:
            raise ValueError(f"prompt.{name} is required")
        return prompt[name]
    try:
        return getattr(prompt, name)
    except AttributeError as exc:
        raise ValueError(f"prompt.{name} is required") from exc


def _optional_field(prompt: Any, name: str) -> Any:
    """Return an optional prompt field without changing legacy-v1 hashes."""

    if isinstance(prompt, Mapping):
        return prompt.get(name)
    return getattr(prompt, name, None)


def _coordinate(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _point(value: Any, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must be an [x, y] pair")
    return [
        _coordinate(value[0], f"{label}[0]"),
        _coordinate(value[1], f"{label}[1]"),
    ]


def _points(value: Any, label: str) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of [x, y] pairs")
    return [_point(point, f"{label}[{index}]") for index, point in enumerate(value)]


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _prompt_schema_version(prompt: Any, explicit: str | None) -> str:
    """Resolve provenance while retaining the pre-v2 helper's legacy default."""

    if explicit is not None:
        version = explicit
    elif isinstance(prompt, Mapping) and "schema_version" in prompt:
        version = prompt["schema_version"]
    else:
        version = getattr(prompt, "schema_version", None)
    if version is None:
        # Calls made before request/prompt provenance existed inferred v2 only
        # from a populated previous_xy. Keep that compatibility surface, but
        # current worker and manifest objects always carry an explicit version.
        version = (
            PROMPT_EVIDENCE_SCHEMA_VERSION
            if _optional_field(prompt, "previous_xy") is not None
            else PROMPT_EVIDENCE_SCHEMA_VERSION_V1
        )
    if version not in PROMPT_EVIDENCE_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported trace prompt schema: {version!r}")
    return version


def canonical_prompt(
    prompt: Any,
    *,
    schema_version: str | None = None,
) -> dict[str, Any]:
    """Return the normalized semantic prompt document used for hashing.

    ``schema_version`` describes the protocol which carried the prompt, not
    whether its optional ``previous_xy`` value happens to be populated. Thus a
    worker request /2 with that field omitted still hashes as trace-prompt /2.
    """

    previous = _optional_field(prompt, "previous_xy")
    version = _prompt_schema_version(prompt, schema_version)
    if version == PROMPT_EVIDENCE_SCHEMA_VERSION_V1 and previous is not None:
        raise ValueError("trace prompt schema v1 does not support previous_xy")
    result = {
        "schema_version": version,
        "start_xy": _point(_field(prompt, "start_xy"), "prompt.start_xy"),
        "end_xy": _point(_field(prompt, "end_xy"), "prompt.end_xy"),
        "positive_xy": _points(
            _field(prompt, "positive_xy"),
            "prompt.positive_xy",
        ),
        "negative_xy": _points(
            _field(prompt, "negative_xy"),
            "prompt.negative_xy",
        ),
    }
    if previous is not None:
        result["previous_xy"] = _point(previous, "prompt.previous_xy")
    return result


def prompt_sha256(
    prompt: Any,
    *,
    schema_version: str | None = None,
) -> str:
    """Hash the complete, ordered semantic trace prompt."""

    return hashlib.sha256(
        _canonical_json_bytes(
            canonical_prompt(prompt, schema_version=schema_version)
        )
    ).hexdigest()


def canonical_source_grid_input(
    image_sha256: str,
    source_tile_origin_xy: Sequence[int],
) -> dict[str, Any]:
    """Bind immutable image bytes to their source-raster pixel origin.

    Ink v2 fixes its normalization tiles to the source raster grid.  The same
    crop bytes therefore have different semantics at different source
    origins.  This evidence is deliberately separate from the historical
    image and prompt hashes so the frozen Ink v1 contract remains unchanged.
    """

    if (
        not isinstance(image_sha256, str)
        or len(image_sha256) != 64
        or any(character not in "0123456789abcdef" for character in image_sha256)
    ):
        raise ValueError("image_sha256 must be a lowercase SHA-256 digest")
    if (
        isinstance(source_tile_origin_xy, (str, bytes))
        or not isinstance(source_tile_origin_xy, Sequence)
        or len(source_tile_origin_xy) != 2
    ):
        raise ValueError("source_tile_origin_xy must be an [x, y] pair")
    origin = []
    for index, value in enumerate(source_tile_origin_xy):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"source_tile_origin_xy[{index}] must be a non-negative integer"
            )
        origin.append(int(value))
    return {
        "schema_version": SOURCE_GRID_INPUT_EVIDENCE_SCHEMA_VERSION,
        "image_sha256": image_sha256,
        "source_tile_origin_xy": origin,
    }


def source_grid_input_sha256(
    image_sha256: str,
    source_tile_origin_xy: Sequence[int],
) -> str:
    """Hash image identity plus the source-grid origin used by Ink v2."""

    return hashlib.sha256(
        _canonical_json_bytes(
            canonical_source_grid_input(image_sha256, source_tile_origin_xy)
        )
    ).hexdigest()


def _float32(value: float, label: str) -> float:
    try:
        normalized = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{label} cannot be represented as float32") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{label} cannot be represented as finite float32")
    return normalized


def canonical_sam_prompt_tensors(prompt: Any) -> dict[str, Any]:
    """Describe the exact float32 point/label tensors sent to EfficientSAM.

    The adapter orders positive points as ``start, positive guides, end`` and
    appends negative guides.  Shapes, dtypes, ordering, and float32 rounding are
    all part of the hash contract.
    """

    semantic = canonical_prompt(prompt)
    ordered_points = [
        semantic["start_xy"],
        *semantic["positive_xy"],
        semantic["end_xy"],
        *semantic["negative_xy"],
    ]
    positive_count = 2 + len(semantic["positive_xy"])
    labels = [1.0] * positive_count + [0.0] * len(semantic["negative_xy"])
    normalized_points = [
        [
            _float32(point[0], f"batched_point_coords[{index}][0]"),
            _float32(point[1], f"batched_point_coords[{index}][1]"),
        ]
        for index, point in enumerate(ordered_points)
    ]
    point_count = len(normalized_points)
    return {
        "schema_version": SAM_TENSOR_EVIDENCE_SCHEMA_VERSION,
        "batched_point_coords": {
            "dtype": "float32-le",
            "shape": [1, 1, point_count, 2],
            "values": normalized_points,
        },
        "batched_point_labels": {
            "dtype": "float32-le",
            "shape": [1, 1, point_count],
            "values": labels,
        },
    }


def sam_prompt_tensor_sha256(prompt: Any) -> str:
    """Hash the canonical EfficientSAM point-coordinate and label tensors."""

    return hashlib.sha256(
        _canonical_json_bytes(canonical_sam_prompt_tensors(prompt))
    ).hexdigest()


def canonical_recovery_prompt_tensors(
    prompt: Any,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Describe the actual product Smart Recovery model input.

    Semantic guide points and ``previous_xy`` remain bound by
    :func:`prompt_sha256`, but they are intentionally absent from this tensor:
    the product recovery model receives only anchor/end positives plus its
    deterministic perpendicular negatives.
    """

    from ai_vectorizer.core.recovery_prompts import build_recovery_prompt_tensors

    tensors = build_recovery_prompt_tensors(
        _field(prompt, "start_xy"),
        _field(prompt, "end_xy"),
        width=width,
        height=height,
    )
    return tensors.canonical_document()


def recovery_prompt_tensor_sha256(
    prompt: Any,
    *,
    width: int,
    height: int,
) -> str:
    """Hash the exact product Smart Recovery point/label tensors."""

    from ai_vectorizer.core.recovery_prompts import build_recovery_prompt_tensors

    tensors = build_recovery_prompt_tensors(
        _field(prompt, "start_xy"),
        _field(prompt, "end_xy"),
        width=width,
        height=height,
    )
    return tensors.sha256


__all__ = [
    "PROMPT_EVIDENCE_SCHEMA_VERSION",
    "PROMPT_EVIDENCE_SCHEMA_VERSION_V1",
    "PROMPT_EVIDENCE_SCHEMA_VERSIONS",
    "SAM_TENSOR_EVIDENCE_SCHEMA_VERSION",
    "SOURCE_GRID_INPUT_EVIDENCE_SCHEMA_VERSION",
    "canonical_prompt",
    "canonical_recovery_prompt_tensors",
    "canonical_sam_prompt_tensors",
    "canonical_source_grid_input",
    "prompt_sha256",
    "recovery_prompt_tensor_sha256",
    "sam_prompt_tensor_sha256",
    "source_grid_input_sha256",
]
