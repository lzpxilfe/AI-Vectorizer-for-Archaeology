"""Explicit, geometry-only bridges for contour gaps under printed labels.

This module intentionally does not inspect raster pixels, load a model, or
choose endpoints.  A caller must supply two user-confirmed endpoints and their
local tangents.  It then returns one bounded Hermite segment that can be shown
as a preview and explicitly accepted by the user.  It is therefore suitable
for neutral or dark contours where an automatic colour-based continuation would
be unsafe around letters, grids, or a parallel contour.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple


Point = Tuple[float, float]


class ManualGapBridgeError(ValueError):
    """Raised when an explicit gap bridge cannot meet its safety contract."""


@dataclass(frozen=True)
class ManualGapBridgeConfig:
    """Resource and geometry bounds for an explicitly requested bridge."""

    min_gap_pixels: float = 3.0
    max_gap_pixels: float = 128.0
    max_abs_tangent_slope: float = 0.35
    max_detour_ratio: float = 1.25
    max_vertices: int = 256

    def validate(self) -> "ManualGapBridgeConfig":
        values = (
            self.min_gap_pixels,
            self.max_gap_pixels,
            self.max_abs_tangent_slope,
            self.max_detour_ratio,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ManualGapBridgeError("manual gap bridge settings must be finite")
        if self.min_gap_pixels <= 0.0:
            raise ManualGapBridgeError("manual gap bridge minimum gap must be positive")
        if self.max_gap_pixels < self.min_gap_pixels:
            raise ManualGapBridgeError("manual gap bridge maximum gap is invalid")
        if self.max_abs_tangent_slope < 0.0:
            raise ManualGapBridgeError("manual gap bridge tangent slope is invalid")
        if self.max_detour_ratio < 1.0:
            raise ManualGapBridgeError("manual gap bridge detour ratio is invalid")
        if (
            isinstance(self.max_vertices, bool)
            or not isinstance(self.max_vertices, int)
            or self.max_vertices < 2
        ):
            raise ManualGapBridgeError("manual gap bridge vertex limit is invalid")
        return self


@dataclass(frozen=True)
class ManualGapBridge:
    """A bounded geometry proposal, with exact user-confirmed endpoints."""

    points_xy: Tuple[Point, ...]
    gap_length_pixels: float
    path_length_pixels: float
    source_tangent_slope: float
    target_tangent_slope: float

    @property
    def detour_ratio(self) -> float:
        return self.path_length_pixels / self.gap_length_pixels


DEFAULT_MANUAL_GAP_BRIDGE_CONFIG = ManualGapBridgeConfig().validate()


def _point(value: Sequence[float], name: str) -> Point:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ManualGapBridgeError(f"{name} must contain exactly two coordinates")
    try:
        point = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ManualGapBridgeError(f"{name} must be numeric") from exc
    if not all(math.isfinite(component) for component in point):
        raise ManualGapBridgeError(f"{name} must be finite")
    return point


def _unit_direction(value: Sequence[float], name: str) -> Point:
    x, y = _point(value, name)
    length = math.hypot(x, y)
    if length <= 1e-9:
        raise ManualGapBridgeError(f"{name} must be non-zero")
    return x / length, y / length


def _path_length(points: Sequence[Point]) -> float:
    return sum(
        math.hypot(next_point[0] - point[0], next_point[1] - point[1])
        for point, next_point in zip(points, points[1:])
    )


def _oriented_slope(
    tangent: Point,
    direction: Point,
    normal: Point,
    maximum: float,
) -> float:
    forward = tangent[0] * direction[0] + tangent[1] * direction[1]
    if forward < 0.0:
        tangent = -tangent[0], -tangent[1]
        forward = -forward
    forward = max(forward, 1e-9)
    lateral = tangent[0] * normal[0] + tangent[1] * normal[1]
    return max(-maximum, min(maximum, lateral / forward))


def build_manual_gap_bridge(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    start_tangent_xy: Sequence[float],
    end_tangent_xy: Sequence[float],
    *,
    config: ManualGapBridgeConfig = DEFAULT_MANUAL_GAP_BRIDGE_CONFIG,
) -> ManualGapBridge:
    """Return a previewable bridge between two user-confirmed contour ends.

    Tangents are axial: their sign is normalized toward the requested endpoint
    before fitting the curve.  The method cannot snap to ink or turn a text
    glyph into a line; endpoint selection remains a visible user action.
    """

    if not isinstance(config, ManualGapBridgeConfig):
        raise TypeError("config must be a ManualGapBridgeConfig")
    config.validate()
    start = _point(start_xy, "start_xy")
    end = _point(end_xy, "end_xy")
    delta = end[0] - start[0], end[1] - start[1]
    gap_length = math.hypot(*delta)
    if not config.min_gap_pixels <= gap_length <= config.max_gap_pixels:
        raise ManualGapBridgeError("manual gap length is outside the allowed range")
    direction = delta[0] / gap_length, delta[1] / gap_length
    normal = -direction[1], direction[0]
    start_tangent = _unit_direction(start_tangent_xy, "start_tangent_xy")
    end_tangent = _unit_direction(end_tangent_xy, "end_tangent_xy")
    source_slope = _oriented_slope(
        start_tangent,
        direction,
        normal,
        config.max_abs_tangent_slope,
    )
    target_slope = _oriented_slope(
        end_tangent,
        direction,
        normal,
        config.max_abs_tangent_slope,
    )
    point_count = int(math.ceil(gap_length)) + 1
    if point_count > config.max_vertices:
        raise ManualGapBridgeError("manual gap bridge would exceed its vertex limit")

    points = []
    for index in range(point_count):
        fraction = index / float(point_count - 1)
        h00 = 2.0 * fraction**3 - 3.0 * fraction**2 + 1.0
        h10 = fraction**3 - 2.0 * fraction**2 + fraction
        h01 = -2.0 * fraction**3 + 3.0 * fraction**2
        h11 = fraction**3 - fraction**2
        point = (
            h00 * start[0]
            + h10 * gap_length * (direction[0] + normal[0] * source_slope)
            + h01 * end[0]
            + h11 * gap_length * (direction[0] + normal[0] * target_slope),
            h00 * start[1]
            + h10 * gap_length * (direction[1] + normal[1] * source_slope)
            + h01 * end[1]
            + h11 * gap_length * (direction[1] + normal[1] * target_slope),
        )
        if not points or point != points[-1]:
            points.append(point)
    points[0] = start
    points[-1] = end
    path_length = _path_length(points)
    if path_length > gap_length * config.max_detour_ratio:
        raise ManualGapBridgeError("manual gap bridge exceeds its detour limit")
    return ManualGapBridge(
        points_xy=tuple(points),
        gap_length_pixels=gap_length,
        path_length_pixels=path_length,
        source_tangent_slope=source_slope,
        target_tangent_slope=target_slope,
    )


__all__ = [
    "DEFAULT_MANUAL_GAP_BRIDGE_CONFIG",
    "ManualGapBridge",
    "ManualGapBridgeConfig",
    "ManualGapBridgeError",
    "build_manual_gap_bridge",
]
