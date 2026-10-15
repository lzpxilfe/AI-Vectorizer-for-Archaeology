"""Bounded broad-turn curves between explicitly confirmed gap endpoints.

This geometry-only operation never finds or moves endpoints.  Its signed
directions must point outward from each visible stroke into the hidden
interval.  The returned cubic is a preview inference, not hidden ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence, Tuple


Point = Tuple[float, float]


class ConfirmedGapCurveError(ValueError):
    """An explicit curve cannot meet the bounded geometry contract."""


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfirmedGapCurveError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ConfirmedGapCurveError(f"{name} must be a finite number")
    return value


@dataclass(frozen=True)
class ConfirmedGapCurveConfig:
    min_gap_pixels: float = 3.0
    max_gap_pixels: float = 128.0
    handle_ratio: float = 2.0 / 3.0
    max_detour_ratio: float = 1.8
    max_deviation_ratio: float = 0.75
    min_forward_alignment: float = -0.5
    max_vertices: int = 512

    def validate(self):
        for name in (
            "min_gap_pixels", "max_gap_pixels", "handle_ratio", "max_detour_ratio",
            "max_deviation_ratio", "min_forward_alignment",
        ):
            _number(getattr(self, name), name)
        if not 1.0 <= self.min_gap_pixels <= self.max_gap_pixels <= 128.0:
            raise ConfirmedGapCurveError("confirmed gap length must be between 1 and 128 pixels")
        if not 0.25 <= self.handle_ratio <= 0.80:
            raise ConfirmedGapCurveError("confirmed gap handle ratio is invalid")
        if not 1.0 <= self.max_detour_ratio <= 1.8:
            raise ConfirmedGapCurveError("confirmed gap detour ratio is invalid")
        if not 0.05 <= self.max_deviation_ratio <= 0.75:
            raise ConfirmedGapCurveError("confirmed gap deviation ratio is invalid")
        if not -0.5 <= self.min_forward_alignment <= 0.5:
            raise ConfirmedGapCurveError("confirmed gap direction allowance is invalid")
        if isinstance(self.max_vertices, bool) or not isinstance(self.max_vertices, Integral):
            raise ConfirmedGapCurveError("confirmed gap vertex limit must be an integer")
        if not 16 <= self.max_vertices <= 1024:
            raise ConfirmedGapCurveError("confirmed gap vertex limit is invalid")
        return self


DEFAULT_CONFIRMED_GAP_CURVE_CONFIG = ConfirmedGapCurveConfig().validate()


@dataclass(frozen=True)
class ConfirmedGapCurve:
    """A retained cubic with exact explicit anchors and signed end tangents."""

    points_xy: Tuple[Point, ...]
    gap_length_pixels: float
    path_length_pixels: float
    source_outward: Point
    target_outward: Point

    @property
    def detour_ratio(self):
        return self.path_length_pixels / self.gap_length_pixels


def _point(value, name):
    if isinstance(value, (str, bytes, dict)):
        raise ConfirmedGapCurveError(f"{name} must contain two finite coordinates")
    try:
        if len(value) != 2:
            raise ConfirmedGapCurveError(f"{name} must contain two finite coordinates")
        return (_number(value[0], name), _number(value[1], name))
    except (TypeError, KeyError, IndexError) as exc:
        raise ConfirmedGapCurveError(f"{name} must contain two finite coordinates") from exc


def _unit(value, name):
    x, y = _point(value, name)
    largest = max(abs(x), abs(y))
    if largest < 1e-12:
        raise ConfirmedGapCurveError(f"{name} must be a nonzero signed direction")
    x, y = x / largest, y / largest
    length = math.hypot(x, y)
    return x / length, y / length


def _cross(first, second, third):
    return ((second[0] - first[0]) * (third[1] - first[1])
            - (second[1] - first[1]) * (third[0] - first[0]))


def _self_intersects(points):
    """Conservative segment intersection, including contacts and overlaps."""

    segments = [
        (first, last, min(first[0], last[0]), max(first[0], last[0]),
         min(first[1], last[1]), max(first[1], last[1]))
        for first, last in zip(points, points[1:])
    ]
    epsilon = 1e-10
    for index, (a, b, xmin, xmax, ymin, ymax) in enumerate(segments):
        for c, d, x0, x1, y0, y1 in segments[index + 2:]:
            if x0 > xmax + epsilon or xmin > x1 + epsilon or y0 > ymax + epsilon or ymin > y1 + epsilon:
                continue
            if _cross(a, b, c) * _cross(a, b, d) <= epsilon and _cross(c, d, a) * _cross(c, d, b) <= epsilon:
                return True
    return False


def build_confirmed_gap_curve(
    source_xy: Sequence[float],
    target_xy: Sequence[float],
    source_outward: Sequence[float],
    target_outward: Sequence[float],
    *,
    config: ConfirmedGapCurveConfig = DEFAULT_CONFIRMED_GAP_CURVE_CONFIG,
) -> ConfirmedGapCurve:
    """Construct one exact-anchor, signed-tangent cubic for explicit review.

    Unlike a chordwise Hermite curve, this permits a broad U whose last
    visible source points partly away from the target chord.  It rejects
    directions more than 120 degrees against that chord, excessive sag or
    length, sampled cusps and self intersection.  The caller retains control
    over visible-line evidence, neighboring strokes and user acceptance.
    """

    config.validate()
    source = _point(source_xy, "source_xy")
    target = _point(target_xy, "target_xy")
    first = _unit(source_outward, "source_outward")
    last = _unit(target_outward, "target_outward")
    dx, dy = target[0] - source[0], target[1] - source[1]
    gap = math.hypot(dx, dy)
    if not config.min_gap_pixels <= gap <= config.max_gap_pixels:
        raise ConfirmedGapCurveError("confirmed gap length is outside the bounded range")
    axis = dx / gap, dy / gap
    if (first[0] * axis[0] + first[1] * axis[1] < config.min_forward_alignment - 1e-12
            or -(last[0] * axis[0] + last[1] * axis[1]) < config.min_forward_alignment - 1e-12):
        raise ConfirmedGapCurveError("confirmed gap directions point away from the hidden interval")
    handle = config.handle_ratio * gap
    control1 = source[0] + first[0] * handle, source[1] + first[1] * handle
    control2 = target[0] + last[0] * handle, target[1] + last[1] * handle
    # Bound work using the control-polygon length, an upper bound on curve
    # length; a 128px chord still has at most 512 default preview vertices.
    polygon_length = 2 * handle + math.hypot(control2[0] - control1[0], control2[1] - control1[1])
    count = min(config.max_vertices, max(16, int(math.ceil(polygon_length * 2)) + 1))
    points = []
    for index in range(count):
        t = index / (count - 1)
        s = 1 - t
        x = s ** 3 * source[0] + 3 * s ** 2 * t * control1[0] + 3 * s * t ** 2 * control2[0] + t ** 3 * target[0]
        y = s ** 3 * source[1] + 3 * s ** 2 * t * control1[1] + 3 * s * t ** 2 * control2[1] + t ** 3 * target[1]
        derivative_x = 3 * (s ** 2 * (control1[0] - source[0])
                            + 2 * s * t * (control2[0] - control1[0])
                            + t ** 2 * (target[0] - control2[0]))
        derivative_y = 3 * (s ** 2 * (control1[1] - source[1])
                            + 2 * s * t * (control2[1] - control1[1])
                            + t ** 2 * (target[1] - control2[1]))
        if math.hypot(derivative_x, derivative_y) < gap * 0.05:
            raise ConfirmedGapCurveError("confirmed gap curve contains an unstable cusp")
        deviation = abs((x - source[0]) * axis[1] - (y - source[1]) * axis[0])
        if deviation > gap * config.max_deviation_ratio + 1e-9:
            raise ConfirmedGapCurveError("confirmed gap curve bends too far outside its endpoints")
        points.append((x, y))
    points[0], points[-1] = source, target
    path_length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    if path_length > gap * config.max_detour_ratio + 1e-9:
        raise ConfirmedGapCurveError("confirmed gap curve exceeds its bounded detour allowance")
    if _self_intersects(points):
        raise ConfirmedGapCurveError("confirmed gap curve would intersect itself")
    return ConfirmedGapCurve(tuple(points), gap, path_length, first, last)
