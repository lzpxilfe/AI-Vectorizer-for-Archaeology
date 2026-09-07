"""Immutable, model-agnostic soft guidance for a bounded trace.

``TraceGuidance`` is intentionally separate from :mod:`line_evidence`.
Line evidence says where a map line is likely to be; trace guidance says
where a human or an optional future model would prefer a route *not* to go.
Keeping both contracts small and QGIS-free lets the Live-Wire worker receive
an immutable snapshot without turning OCR, semantic segmentation, or a UI
selection into a hard mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class TraceGuidance:
    """An immutable per-pixel soft avoidance score.

    ``avoidance_score`` is a guide rather than a hard exclusion: values in
    the closed interval ``[0, 1]`` add route cost but can never make an Ink
    path unreachable.  This makes a user-marked label, stain, or future OCR
    text-risk prior safe to use around narrow or interrupted map lines.
    """

    SCHEMA_VERSION: ClassVar[str] = "archaeotrace-trace-guidance/1"

    avoidance_score: np.ndarray

    def __post_init__(self) -> None:
        try:
            score = np.array(
                self.avoidance_score,
                dtype=np.float32,
                order="C",
                copy=True,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "avoidance_score must be a finite numeric array"
            ) from exc
        if score.ndim != 2 or min(score.shape, default=0) < 1:
            raise ValueError("avoidance_score must be a non-empty 2D array")
        if not np.isfinite(score).all():
            raise ValueError("avoidance_score must contain only finite values")
        if np.any((score < 0.0) | (score > 1.0)):
            raise ValueError(
                "avoidance_score values must be between zero and one"
            )
        score.setflags(write=False)
        object.__setattr__(self, "avoidance_score", score)

    @property
    def shape(self) -> Tuple[int, int]:
        """The ``(height, width)`` pixel-grid shape."""

        return tuple(int(value) for value in self.avoidance_score.shape)


def guidance_from_boxes(
    shape: Tuple[int, int],
    boxes: Iterable[Tuple[float, float, float, float]],
    *,
    feather_pixels: float = 2.0,
) -> TraceGuidance:
    """Build max-composed soft avoidance from cache-pixel rectangles.

    ``boxes`` contains ``(x0, y0, x1, y1)`` extrema in the target image's
    pixel-coordinate system.  Reversed corners are accepted.  The rectangle
    interior receives score one; outside it, score falls linearly to zero
    across ``feather_pixels``.  Coordinates may be outside the image, which
    lets a map-space user selection safely overlap a changing cache extent.
    """

    height, width = _validate_shape(shape)
    feather = _validate_feather(feather_pixels)
    if isinstance(boxes, (str, bytes)):
        raise ValueError("boxes must be an iterable of four-number rectangles")
    try:
        iterator = iter(boxes)
    except TypeError as exc:
        raise ValueError(
            "boxes must be an iterable of four-number rectangles"
        ) from exc

    score = np.zeros((height, width), dtype=np.float32)
    grid_x = np.arange(width, dtype=np.float32)
    grid_y = np.arange(height, dtype=np.float32)
    for index, box in enumerate(iterator):
        x0, y0, x1, y1 = _validate_box(box, index)
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))

        # Distance to an axis-aligned rectangle is zero inside it.  The
        # outer product keeps the operation NumPy-only (no SciPy runtime is
        # introduced by this optional guide).
        distance_x = np.maximum(
            np.maximum(left - grid_x, 0.0),
            grid_x - right,
        )
        distance_y = np.maximum(
            np.maximum(top - grid_y, 0.0),
            grid_y - bottom,
        )
        if feather <= 0.0:
            current = (
                (grid_y[:, None] >= top)
                & (grid_y[:, None] <= bottom)
                & (grid_x[None, :] >= left)
                & (grid_x[None, :] <= right)
            ).astype(np.float32)
        else:
            distance = np.hypot(distance_y[:, None], distance_x[None, :])
            current = np.clip(1.0 - distance / feather, 0.0, 1.0).astype(
                np.float32
            )
        np.maximum(score, current, out=score)
    return TraceGuidance(score)


def crop_trace_guidance(
    guidance: TraceGuidance,
    bounds: Tuple[int, int, int, int],
) -> TraceGuidance:
    """Return an immutable end-exclusive crop of ``guidance``.

    Bounds follow the same ``(x0, y0, x1, y1)`` contract used by
    :func:`ai_vectorizer.core.line_evidence.crop_line_evidence`.
    """

    if not isinstance(guidance, TraceGuidance):
        raise TypeError("guidance must be a TraceGuidance instance")
    x0, y0, x1, y1 = _validate_bounds(bounds, guidance.shape)
    return TraceGuidance(guidance.avoidance_score[y0:y1, x0:x1])


def _validate_shape(shape) -> Tuple[int, int]:
    if (
        isinstance(shape, (str, bytes))
        or not hasattr(shape, "__len__")
        or len(shape) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in shape
        )
    ):
        raise ValueError("shape must be an integer (height, width) tuple")
    height, width = (int(value) for value in shape)
    if height < 1 or width < 1:
        raise ValueError("shape dimensions must be positive")
    return height, width


def _validate_feather(value) -> float:
    if isinstance(value, bool):
        raise ValueError("feather_pixels must be a finite non-negative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "feather_pixels must be a finite non-negative number"
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("feather_pixels must be a finite non-negative number")
    return result


def _validate_box(box, index: int) -> Tuple[float, float, float, float]:
    if (
        isinstance(box, (str, bytes))
        or not hasattr(box, "__len__")
        or len(box) != 4
    ):
        raise ValueError(f"box {index} must contain four finite coordinates")
    values = []
    for value in box:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"box {index} must contain four finite coordinates")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"box {index} must contain four finite coordinates"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(f"box {index} must contain four finite coordinates")
        values.append(number)
    return tuple(values)


def _validate_bounds(bounds, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    if (
        isinstance(bounds, (str, bytes))
        or not hasattr(bounds, "__len__")
        or len(bounds) != 4
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in bounds
        )
    ):
        raise ValueError("bounds must be an integer (x0, y0, x1, y1) tuple")
    x0, y0, x1, y1 = (int(value) for value in bounds)
    height, width = shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("bounds must be a non-empty rectangle inside guidance")
    return x0, y0, x1, y1


__all__ = [
    "TraceGuidance",
    "crop_trace_guidance",
    "guidance_from_boxes",
]
