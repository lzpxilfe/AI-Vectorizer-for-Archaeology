"""Explicit, geometry-only bridges for contour gaps under printed labels.

The geometry builder uses two user-confirmed endpoints and their local
tangents to return one bounded Hermite preview.  The optional evidence sampler
reads directions from an existing Ink snapshot, so product and benchmark can
use the same endpoint lookup.  Neither operation chooses or moves endpoints,
loads a model, or commits a line.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Optional, Sequence, Tuple


Point = Tuple[float, float]
_MAX_BRIDGE_VERTICES = 4096
_MAX_TANGENT_RADIUS = 32.0
_MIN_TANGENT_COHERENCE = 0.15
_MIN_TANGENT_SCORE = 0.15
_MAX_TANGENT_DISAGREEMENT_RADIANS = math.radians(35.0)
_DEFAULT_CONTEXT_TANGENT_RADIUS = 12.0
_MIN_CONTEXT_SUPPORT_PIXELS = 5
_MIN_CONTEXT_SPAN_PIXELS = 4.0
_MIN_CONTEXT_AXIS_RATIO = 0.80
_CONTEXT_CONE_TANGENT = math.tan(math.radians(40.0))


class ManualGapBridgeError(ValueError):
    """Raised when an explicit gap bridge cannot meet its safety contract."""


@dataclass(frozen=True)
class ManualGapBridgeConfig:
    """Resource and geometry bounds for an explicitly requested bridge.

    ``min_tangent_alignment`` is the absolute cosine with the endpoint chord;
    its default rejects directions more than 45 degrees away before any slope
    clipping.  ``max_vertices`` has a hard ceiling of 4096 even for custom
    configurations.
    """

    min_gap_pixels: float = 3.0
    max_gap_pixels: float = 128.0
    max_abs_tangent_slope: float = 0.35
    max_detour_ratio: float = 1.25
    max_vertices: int = 256
    min_tangent_alignment: float = math.sqrt(0.5)

    def validate(self) -> "ManualGapBridgeConfig":
        for name in (
            "min_gap_pixels", "max_gap_pixels", "max_abs_tangent_slope",
            "max_detour_ratio", "min_tangent_alignment",
        ):
            _finite_number(getattr(self, name), name)
        if self.min_gap_pixels <= 0.0:
            raise ManualGapBridgeError("manual gap bridge minimum gap must be positive")
        if self.max_gap_pixels < self.min_gap_pixels:
            raise ManualGapBridgeError("manual gap bridge maximum gap is invalid")
        if not 0.0 <= self.max_abs_tangent_slope <= 1.0:
            raise ManualGapBridgeError("manual gap bridge tangent slope is invalid")
        if self.max_detour_ratio < 1.0:
            raise ManualGapBridgeError("manual gap bridge detour ratio is invalid")
        if (
            isinstance(self.max_vertices, bool)
            or not isinstance(self.max_vertices, Integral)
            or not 2 <= self.max_vertices <= _MAX_BRIDGE_VERTICES
        ):
            raise ManualGapBridgeError("manual gap bridge vertex limit is invalid")
        if not 0.5 <= self.min_tangent_alignment <= 1.0:
            raise ManualGapBridgeError("manual gap bridge tangent alignment is invalid")
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


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ManualGapBridgeError(f"{name} must be numeric, not a boolean or text")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ManualGapBridgeError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ManualGapBridgeError(f"{name} must be finite")
    return result


DEFAULT_MANUAL_GAP_BRIDGE_CONFIG = ManualGapBridgeConfig().validate()


def _point(value: Sequence[float], name: str) -> Point:
    try:
        if isinstance(value, (str, bytes, dict)) or len(value) != 2:
            raise ManualGapBridgeError(f"{name} must contain exactly two coordinates")
        return (
            _finite_number(value[0], name),
            _finite_number(value[1], name),
        )
    except (TypeError, KeyError, IndexError) as exc:
        raise ManualGapBridgeError(f"{name} must contain exactly two coordinates") from exc


def _unit_direction(value: Sequence[float], name: str) -> Point:
    x, y = _point(value, name)
    largest = max(abs(x), abs(y))
    if largest <= 1e-9:
        raise ManualGapBridgeError(f"{name} must be non-zero")
    x, y = x / largest, y / largest
    length = math.hypot(x, y)
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
    minimum_alignment: float,
) -> float:
    forward = tangent[0] * direction[0] + tangent[1] * direction[1]
    if forward < 0.0:
        tangent = -tangent[0], -tangent[1]
        forward = -forward
    if forward < minimum_alignment - 1e-12:
        raise ManualGapBridgeError(
            "manual gap bridge endpoint direction is incompatible with the gap"
        )
    lateral = tangent[0] * normal[0] + tangent[1] * normal[1]
    return max(-maximum, min(maximum, lateral / forward))


def sample_manual_gap_tangent(
    evidence,
    pixel_xy: Sequence[float],
    *,
    radius_pixels: float = 3.0,
) -> Optional[Point]:
    """Read a local, unambiguous axial direction near an explicit endpoint.

    Search uses a bounded circle around the exact floating-point endpoint.
    Only centerline pixels with useful score and direction coherence qualify.
    Distance takes priority over coherence, so a stronger nearby glyph cannot
    displace the contour under the cursor.  Competing directions within one
    pixel of the nearest support distance make the request ambiguous and
    return ``None``.  The endpoint is never snapped to the sampled pixels.

    NumPy and ``LineEvidence`` are imported only when this optional sampler is
    used; the geometry-only builder remains usable without either dependency.
    Invalid inputs raise ``ManualGapBridgeError``; absent or ambiguous support
    returns ``None`` for the caller to retain its existing Ink preview.
    """

    point = _point(pixel_xy, "pixel_xy")
    radius = _finite_number(radius_pixels, "radius_pixels")
    if not 0.0 < radius <= _MAX_TANGENT_RADIUS:
        raise ManualGapBridgeError("tangent radius must be between zero and 32 pixels")
    try:
        import numpy as np
        from .line_evidence import LineEvidence
    except ImportError as exc:
        raise ManualGapBridgeError("NumPy is required to sample Ink directions") from exc
    if not isinstance(evidence, LineEvidence):
        raise ManualGapBridgeError("evidence must be a LineEvidence snapshot")
    height, width = evidence.shape
    px, py = point
    if not (0.0 <= px < width and 0.0 <= py < height):
        raise ManualGapBridgeError("manual bridge endpoint leaves the Ink cache")

    x0, x1 = max(0, math.ceil(px - radius)), min(width, math.floor(px + radius) + 1)
    y0, y1 = max(0, math.ceil(py - radius)), min(height, math.floor(py + radius) + 1)
    selection = np.s_[y0:y1, x0:x1]
    ys, xs = np.nonzero(
        evidence.centerline[selection]
        & (evidence.center_score[selection] >= _MIN_TANGENT_SCORE)
        & (evidence.coherence[selection] >= _MIN_TANGENT_COHERENCE)
    )
    candidates = []
    for local_y, local_x in zip(ys.tolist(), xs.tolist()):
        x, y = x0 + local_x, y0 + local_y
        distance = math.hypot(x - px, y - py)
        if distance > radius:
            continue
        tangent = _unit_direction(
            (float(evidence.tangent_x[y, x]), float(evidence.tangent_y[y, x])),
            "sampled tangent",
        )
        candidates.append((distance, y, x, tangent, float(evidence.coherence[y, x])))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    nearest_distance, _, _, reference, _ = candidates[0]
    close = [item for item in candidates if item[0] <= nearest_distance + 1.0]

    angles = []
    sum_x, sum_y = 0.0, 0.0
    for distance, _, _, tangent, coherence in close:
        tx, ty = tangent
        dot = tx * reference[0] + ty * reference[1]
        if dot < 0.0:
            tx, ty = -tx, -ty
        angle = math.atan2(reference[0] * ty - reference[1] * tx,
                           reference[0] * tx + reference[1] * ty)
        angles.append(angle)
        weight = coherence / (1.0 + distance) ** 2
        sum_x += weight * tx
        sum_y += weight * ty
    if max(angles) - min(angles) > _MAX_TANGENT_DISAGREEMENT_RADIANS:
        return None
    return _unit_direction((sum_x, sum_y), "sampled tangent")


def _contextual_endpoint_tangent(
    evidence,
    anchor: Point,
    chord: Point,
    outward: Point,
    *,
    support_radius_pixels: float,
):
    """Fit a one-sided local centreline rather than a glyph-shaped tensor.

    A printed number can overlap the three-pixel neighbourhood of a genuine
    contour endpoint and rotate its gradient tensor.  The anchor pair gives a
    modest, explicit piece of extra information: each contour must continue
    *away* from the blank.  This helper only considers centerline support in
    that one-sided cone.  It deliberately does not search along the chord into
    the blank, where numeral strokes belong, and it rejects diffuse support
    rather than choosing a direction from it.
    """

    import numpy as np

    height, width = evidence.shape
    px, py = anchor
    radius = support_radius_pixels
    x0, x1 = max(0, math.ceil(px - radius)), min(width, math.floor(px + radius) + 1)
    y0, y1 = max(0, math.ceil(py - radius)), min(height, math.floor(py + radius) + 1)
    selection = np.s_[y0:y1, x0:x1]
    ys, xs = np.nonzero(
        evidence.centerline[selection]
        & (evidence.center_score[selection] >= _MIN_TANGENT_SCORE)
    )
    candidates = []
    for local_y, local_x in zip(ys.tolist(), xs.tolist()):
        x, y = x0 + local_x, y0 + local_y
        dx, dy = x - px, y - py
        distance = math.hypot(dx, dy)
        if distance > radius:
            continue
        projection = dx * outward[0] + dy * outward[1]
        if projection < 0.5:
            continue
        lateral = abs(dx * outward[1] - dy * outward[0])
        # A one-pixel allowance preserves discretized shallow curves at the
        # anchor.  Beyond it, the fixed cone blocks a nearby parallel contour.
        if lateral > 1.0 + projection * _CONTEXT_CONE_TANGENT:
            continue
        candidates.append((x, y, distance, projection))
    if len(candidates) < _MIN_CONTEXT_SUPPORT_PIXELS:
        return None

    distances = [candidate[2] for candidate in candidates]
    projections = [candidate[3] for candidate in candidates]
    if min(distances) > 3.0 or max(projections) - min(projections) < _MIN_CONTEXT_SPAN_PIXELS:
        return None

    points = np.asarray([(x, y) for x, y, _distance, _projection in candidates], dtype=np.float64)
    # Nearer centreline support should matter most, but a single endpoint pixel
    # must not dictate an orientation after a glyph has contaminated it.
    weights = np.asarray(
        [1.0 / (1.0 + distance) ** 2 for _x, _y, distance, _projection in candidates],
        dtype=np.float64,
    )
    centre = np.average(points, axis=0, weights=weights)
    offsets = points - centre
    covariance = (offsets * weights[:, None]).T @ offsets / float(weights.sum())
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = float(eigenvalues[-1])
    minor = max(0.0, float(eigenvalues[0]))
    if major <= 1e-9 or major / (major + minor) < _MIN_CONTEXT_AXIS_RATIO:
        return None
    tangent = (float(eigenvectors[0, -1]), float(eigenvectors[1, -1]))
    tangent = _unit_direction(tangent, "contextual sampled tangent")
    if abs(tangent[0] * chord[0] + tangent[1] * chord[1]) < math.sqrt(0.5):
        return None
    return tangent


def sample_manual_gap_bridge_tangents(
    evidence,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    *,
    support_radius_pixels: float = _DEFAULT_CONTEXT_TANGENT_RADIUS,
) -> Optional[Tuple[Point, Point]]:
    """Return two conservative, chord-contextual endpoint directions.

    This is a fallback for an explicitly requested bridge, not an automatic
    endpoint finder.  It uses the already-confirmed endpoint pair only to
    inspect the visible contour leading away from the blank.  Both sides need
    a nearby, elongated one-sided centerline; otherwise it returns ``None``.
    In particular, it cannot turn a numeral within the blank or a parallel
    contour outside the cone into an accepted bridge.
    """

    start = _point(start_xy, "start_xy")
    end = _point(end_xy, "end_xy")
    radius = _finite_number(support_radius_pixels, "support_radius_pixels")
    if not _DEFAULT_CONTEXT_TANGENT_RADIUS <= radius <= _MAX_TANGENT_RADIUS:
        raise ManualGapBridgeError("context tangent radius must be between 12 and 32 pixels")
    try:
        from .line_evidence import LineEvidence
    except ImportError as exc:
        raise ManualGapBridgeError("NumPy is required to sample Ink directions") from exc
    if not isinstance(evidence, LineEvidence):
        raise ManualGapBridgeError("evidence must be a LineEvidence snapshot")
    height, width = evidence.shape
    for point in (start, end):
        if not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
            raise ManualGapBridgeError("manual bridge endpoint leaves the Ink cache")
    chord = _unit_direction((end[0] - start[0], end[1] - start[1]), "manual bridge chord")
    start_tangent = _contextual_endpoint_tangent(
        evidence, start, chord, (-chord[0], -chord[1]), support_radius_pixels=radius,
    )
    end_tangent = _contextual_endpoint_tangent(
        evidence, end, chord, chord, support_radius_pixels=radius,
    )
    if start_tangent is None or end_tangent is None:
        return None
    return start_tangent, end_tangent


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
    before fitting the curve.  Incompatible directions are rejected before
    mildly noisy slopes are bounded for fitting.  The method cannot snap to
    ink or turn a text glyph into a line; endpoint selection remains a visible
    user action.
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
        config.min_tangent_alignment,
    )
    target_slope = _oriented_slope(
        end_tangent,
        direction,
        normal,
        config.max_abs_tangent_slope,
        config.min_tangent_alignment,
    )
    point_count = int(math.ceil(gap_length)) + 1
    if point_count > config.max_vertices:
        raise ManualGapBridgeError("manual gap bridge would exceed its vertex limit")

    points = []
    for index in range(point_count):
        fraction = index / float(point_count - 1)
        h10 = fraction**3 - 2.0 * fraction**2 + fraction
        h11 = fraction**3 - fraction**2
        offset = gap_length * (h10 * source_slope + h11 * target_slope)
        point = (
            start[0] + fraction * delta[0] + normal[0] * offset,
            start[1] + fraction * delta[1] + normal[1] * offset,
        )
        if not all(math.isfinite(component) for component in point):
            raise ManualGapBridgeError("manual gap bridge generated non-finite geometry")
        if not points or point != points[-1]:
            points.append(point)
    points[0] = start
    points[-1] = end
    path_length = _path_length(points)
    if not math.isfinite(path_length) or path_length > gap_length * config.max_detour_ratio:
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
    "sample_manual_gap_tangent",
]
