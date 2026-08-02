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


PROMPT_EVIDENCE_SCHEMA_VERSION = "archaeotrace-trace-prompt/1"
SAM_TENSOR_EVIDENCE_SCHEMA_VERSION = "archaeotrace-efficientsam-prompt-tensors/1"


def _field(prompt: Any, name: str) -> Any:
    if isinstance(prompt, Mapping):
        if name not in prompt:
            raise ValueError(f"prompt.{name} is required")
        return prompt[name]
    try:
        return getattr(prompt, name)
    except AttributeError as exc:
        raise ValueError(f"prompt.{name} is required") from exc


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


def canonical_prompt(prompt: Any) -> dict[str, Any]:
    """Return the normalized semantic prompt document used for hashing."""

    return {
        "schema_version": PROMPT_EVIDENCE_SCHEMA_VERSION,
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


def prompt_sha256(prompt: Any) -> str:
    """Hash the complete, ordered semantic trace prompt."""

    return hashlib.sha256(_canonical_json_bytes(canonical_prompt(prompt))).hexdigest()


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


__all__ = [
    "PROMPT_EVIDENCE_SCHEMA_VERSION",
    "SAM_TENSOR_EVIDENCE_SCHEMA_VERSION",
    "canonical_prompt",
    "canonical_sam_prompt_tensors",
    "prompt_sha256",
    "sam_prompt_tensor_sha256",
]
