"""Explicit, local ink-colour guidance independent of QGIS and detectors.

The user samples observed ink once.  A small immutable colour descriptor then
adds finite route cost to differently coloured ink.  It never learns from a
suggested route, changes detector evidence, or fills missing map ink.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import ClassVar, Optional, Sequence, Tuple

import numpy as np

from .line_evidence import LineEvidence
from .trace_guidance import TraceGuidance


MIN_SAMPLE_CHROMA = 12.0 / 255.0
MIN_GUIDANCE_CHROMA = 2.0 / 255.0
STRONG_INK_SCORE = 0.55
SAMPLE_AMBIGUITY_DISTANCE = 0.35
MATCH_COLOR_DISTANCE = 0.25
MISMATCH_COLOR_DISTANCE = 0.80


@dataclass(frozen=True)
class InkColorProfile:
    """One explicitly sampled chromatic ink colour, with no image references.

    ``sample_rgb`` is useful for showing the selected colour to the user.
    ``chroma_direction`` removes channel mean and normalizes magnitude, so
    darker ink and its antialiasing towards white retain the same style.
    Neutral ink has no identifiable hue and cannot form a colour profile.
    """

    SCHEMA_VERSION: ClassVar[str] = "archaeotrace-ink-color/1"

    sample_rgb: Tuple[int, int, int]
    chroma_direction: Tuple[float, float, float] = field(init=False)

    def __post_init__(self) -> None:
        rgb = self.sample_rgb
        if (
            isinstance(rgb, (str, bytes))
            or not hasattr(rgb, "__len__")
            or len(rgb) != 3
            or any(
                isinstance(channel, (bool, np.bool_))
                or not isinstance(channel, (int, np.integer))
                or not 0 <= channel <= 255
                for channel in rgb
            )
        ):
            raise ValueError("sample_rgb must contain three Byte RGB integers")
        rgb = tuple(int(channel) for channel in rgb)
        values = np.asarray(rgb, dtype=np.float64) / 255.0
        if float(values.max() - values.min()) < MIN_SAMPLE_CHROMA:
            raise ValueError("sample_rgb must have a distinguishable ink colour")
        chroma = values - values.mean()
        direction = chroma / float(np.linalg.norm(chroma))
        object.__setattr__(self, "sample_rgb", rgb)
        object.__setattr__(
            self, "chroma_direction", tuple(float(value) for value in direction)
        )


def sample_ink_color(
    image: np.ndarray,
    evidence: LineEvidence,
    point: Sequence[float],
    *,
    sample_radius: float = 2.0,
) -> Optional[InkColorProfile]:
    """Sample nearby observed chromatic ink, or refuse an ambiguous sample.

    Coordinates are cache pixel centres, in ``(x, y)`` order.  Only native
    Byte RGB is supported: view-dependent stretches and grayscale inputs
    return ``None``.  The nearest centreline or strong continuous score must
    lie inside ``sample_radius``.  Different colours, including neutral ink,
    within 0.75 pixels of that nearest support make the sample ambiguous.
    Blank pixels are never sampled as a style.
    """

    rgb = _byte_rgb(image, evidence)
    x, y = _validate_point(point)
    radius = _validate_radius(sample_radius)
    if rgb is None:
        return None
    height, width = evidence.shape
    if not (0.0 <= x < width and 0.0 <= y < height):
        return None
    left, right = max(0, math.ceil(x - radius)), min(width - 1, math.floor(x + radius))
    top, bottom = max(0, math.ceil(y - radius)), min(height - 1, math.floor(y + radius))
    if left > right or top > bottom:
        return None
    selection = np.s_[top : bottom + 1, left : right + 1]
    supported = evidence.centerline[selection] | (
        evidence.center_score[selection] >= STRONG_INK_SCORE
    )
    candidate_y, candidate_x = np.nonzero(supported)
    if not candidate_x.size:
        return None
    candidate_x = candidate_x + left
    candidate_y = candidate_y + top
    distance = np.hypot(candidate_x - x, candidate_y - y)
    inside = distance <= radius + 1e-6
    if not inside.any():
        return None
    nearest = float(distance[inside].min())
    close = inside & (distance <= nearest + 0.75)
    colors = rgb[candidate_y[close], candidate_x[close]].astype(np.float64) / 255.0
    if np.any(np.ptp(colors, axis=1) < MIN_SAMPLE_CHROMA):
        return None
    chroma = colors - colors.mean(axis=1, keepdims=True)
    directions = chroma / np.linalg.norm(chroma, axis=1, keepdims=True)
    reference_index = int(np.argmin(distance[close]))
    disagreement = np.linalg.norm(directions - directions[reference_index], axis=1)
    if np.any(disagreement > SAMPLE_AMBIGUITY_DISTANCE):
        return None
    # Median resists a single quantized antialias value without making the
    # profile depend on any subsequently accepted route or detector result.
    representative = np.rint(np.median(colors, axis=0) * 255.0).astype(np.uint8)
    return InkColorProfile(tuple(int(channel) for channel in representative))


def guidance_from_ink_color(
    image: np.ndarray,
    evidence: LineEvidence,
    profile: InkColorProfile,
) -> TraceGuidance:
    """Return immutable soft mismatch cost only where ink is observed.

    Matching hues receive zero penalty; different hues and neutral ink
    receive a bounded ``[0, 1]`` penalty.  A whiteward change in intensity
    preserves hue, including ordinary antialiasing.  Gaps with no Ink support
    remain zero.  Same-colour symbols still need cursor steering or manual
    avoidance; this descriptor does not assign semantics to map strokes.
    """

    rgb = _byte_rgb(image, evidence)
    if not isinstance(profile, InkColorProfile):
        raise TypeError("profile must be an InkColorProfile instance")
    if rgb is None:
        raise ValueError("ink colour guidance requires native Byte RGB")
    values = rgb.astype(np.float32) / np.float32(255.0)
    chroma = values - values.mean(axis=2, keepdims=True)
    norm = np.linalg.norm(chroma, axis=2)
    colored = np.ptp(values, axis=2) >= MIN_GUIDANCE_CHROMA
    direction = np.zeros_like(chroma, dtype=np.float32)
    np.divide(chroma, norm[..., None], out=direction, where=norm[..., None] > 1e-8)
    color_distance = np.linalg.norm(
        direction - np.asarray(profile.chroma_direction, dtype=np.float32),
        axis=2,
    )
    mismatch = np.clip(
        (color_distance - MATCH_COLOR_DISTANCE)
        / (MISMATCH_COLOR_DISTANCE - MATCH_COLOR_DISTANCE),
        0.0,
        1.0,
    )
    mismatch[~colored] = 1.0
    observed_ink = np.maximum(
        evidence.center_score, evidence.centerline.astype(np.float32)
    )
    return TraceGuidance(mismatch * observed_ink)


def _byte_rgb(image: np.ndarray, evidence: LineEvidence) -> Optional[np.ndarray]:
    if not isinstance(evidence, LineEvidence):
        raise TypeError("evidence must be a LineEvidence instance")
    values = np.asarray(image)
    if values.ndim not in (2, 3) or values.shape[:2] != evidence.shape:
        raise ValueError("image dimensions must match Ink evidence")
    if values.ndim != 3 or values.shape[2] < 3 or values.dtype != np.uint8:
        return None
    return values[..., :3]


def _validate_point(point: Sequence[float]) -> Tuple[float, float]:
    if (
        isinstance(point, (str, bytes))
        or not hasattr(point, "__len__")
        or len(point) != 2
        or any(isinstance(value, (bool, np.bool_)) for value in point)
    ):
        raise ValueError("point must contain two finite pixel coordinates")
    try:
        coordinates = tuple(float(value) for value in point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("point must contain two finite pixel coordinates") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("point must contain two finite pixel coordinates")
    return coordinates


def _validate_radius(value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("sample_radius must be a finite number between zero and eight")
    try:
        radius = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "sample_radius must be a finite number between zero and eight"
        ) from exc
    if not math.isfinite(radius) or not 0.0 <= radius <= 8.0:
        raise ValueError("sample_radius must be a finite number between zero and eight")
    return radius


__all__ = ["InkColorProfile", "sample_ink_color", "guidance_from_ink_color"]
