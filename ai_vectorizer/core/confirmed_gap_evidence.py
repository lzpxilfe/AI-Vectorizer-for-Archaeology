"""Bounded native endpoint directions for an explicitly confirmed gap.

The user's coordinates remain authoritative. Nearby native endpoints only
supply signed directions: they do not move either coordinate or decide that
two contours are the same. The sampler accepts broad turns whose tangents
are almost perpendicular to the endpoint chord, while refusing to borrow
context from a different stroke or a short printed glyph.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional, Sequence, Tuple

import numpy as np

from .line_evidence import LineEvidence


Point = Tuple[float, float]
_RADIUS = 20
_ANCHOR_RADIUS = 3.0
_MIN_CONTEXT = 8.0
_CONTEXT = 12.0


def _point(value, name):
    if isinstance(value, (str, bytes, dict)) or not hasattr(value, "__len__") or len(value) != 2:
        raise ValueError(f"{name} must contain two finite coordinates")
    result = []
    for component in value:
        if isinstance(component, (bool, np.bool_)) or not isinstance(component, Real):
            raise ValueError(f"{name} must contain two finite coordinates")
        component = float(component)
        if not math.isfinite(component):
            raise ValueError(f"{name} must contain two finite coordinates")
        result.append(component)
    return tuple(result)


def _neighbours(mask, pixel):
    """Native adjacency with redundant digital-bend diagonals suppressed."""
    x, y = pixel
    height, width = mask.shape
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height and mask[ny, nx]):
                continue
            if dx and dy and (mask[y, nx] or mask[ny, x]):
                continue
            result.append((nx, ny))
    return result


def _component(mask, seed):
    pending = [seed]
    visited = {seed}
    while pending:
        current = pending.pop()
        for neighbour in _neighbours(mask, current):
            if neighbour not in visited:
                visited.add(neighbour)
                pending.append(neighbour)
    return visited


def _chain(mask, endpoint, anchor, nearest):
    previous = None
    current = endpoint
    points = [endpoint]
    length = 0.0
    # This loop is bounded by native distance, not component size.
    for _ in range(int(math.ceil(_CONTEXT)) + 2):
        neighbours = _neighbours(mask, current)
        expected = 1 if previous is None else 2
        if len(neighbours) != expected:
            if length < _MIN_CONTEXT:
                return None
            break
        following = [point for point in neighbours if point != previous]
        if len(following) != 1:
            return None
        following = following[0]
        length += math.hypot(following[0] - current[0], following[1] - current[1])
        previous, current = current, following
        points.append(current)
        if length >= _CONTEXT:
            break
    if length < _MIN_CONTEXT:
        return None
    points = np.asarray(points, dtype=np.float64)
    # A near endpoint on some other part of the same connected network may
    # not use distant grid/glyph junctions to reach the clicked fragment.
    if float(np.linalg.norm(points[:5] - anchor, axis=1).min()) > nearest + 0.5:
        return None
    travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    chord = float(np.linalg.norm(points[-1] - points[0]))
    if chord < travel[-1] * 0.80:
        return None
    signed = -travel
    design = np.column_stack((signed, signed ** 2))
    coefficients = np.linalg.lstsq(design, points - points[0], rcond=None)[0]
    fitted = design @ coefficients
    if float(np.sqrt(np.mean(np.sum((fitted - (points - points[0])) ** 2, axis=1)))) > 1.25:
        return None
    outward = coefficients[0]
    norm = float(np.linalg.norm(outward))
    if not math.isfinite(norm) or norm < 0.4:
        return None
    outward /= norm
    near_index = max(1, int(np.searchsorted(travel, min(5.0, travel[-1]))))
    near = points[0] - points[near_index]
    near /= np.linalg.norm(near)
    if float(np.dot(outward, near)) < 0.5:
        return None
    return float(outward[0]), float(outward[1])


def _sample(evidence, anchor):
    height, width = evidence.shape
    px, py = anchor
    if not (0 <= px < width and 0 <= py < height):
        raise ValueError("confirmed gap endpoint leaves the Ink cache")
    x0, y0 = max(0, int(math.floor(px)) - _RADIUS), max(0, int(math.floor(py)) - _RADIUS)
    x1, y1 = min(width, int(math.floor(px)) + _RADIUS + 1), min(height, int(math.floor(py)) + _RADIUS + 1)
    mask = np.asarray(
        evidence.centerline[y0:y1, x0:x1]
        & (evidence.center_score[y0:y1, x0:x1] >= 0.15), dtype=bool,
    )
    anchor = np.asarray((px - x0, py - y0), dtype=np.float64)
    ys, xs = np.nonzero(mask)
    distances = np.hypot(xs - anchor[0], ys - anchor[1])
    nearby = np.flatnonzero(distances <= _ANCHOR_RADIUS + 1e-9)
    if not len(nearby):
        return None
    order = nearby[np.argsort(distances[nearby], kind="stable")]
    nearest = float(distances[order[0]])
    seed = int(xs[order[0]]), int(ys[order[0]])
    own = _component(mask, seed)
    # Distance takes priority over a longer, straighter neighbouring stroke.
    # Close competing components make this endpoint ambiguous.
    for index in order:
        if distances[index] > nearest + 0.75:
            break
        if (int(xs[index]), int(ys[index])) not in own:
            return None
    endpoints = []
    for index in order:
        pixel = int(xs[index]), int(ys[index])
        if pixel not in own or len(_neighbours(mask, pixel)) != 1:
            continue
        gx, gy = pixel[0] + x0, pixel[1] + y0
        if gx <= 1 or gy <= 1 or gx >= width - 2 or gy >= height - 2:
            continue
        endpoints.append((float(distances[index]), pixel))
    if not endpoints:
        return None
    endpoints.sort()
    if len(endpoints) > 1 and endpoints[1][0] <= endpoints[0][0] + 0.75:
        return None
    return _chain(mask, endpoints[0][1], anchor, nearest)


def _skips_competing_endpoint(evidence, start, end, outward):
    """Refuse a pair which leaps over a nearer credible terminal fragment.

    This does not select the alternative or prove contour identity. A nearby
    similarly directed native endpoint earlier in the gap corridor makes the
    farther explicit pair uncertain, so it requires further user guidance.
    """
    axis = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    gap = float(np.linalg.norm(axis))
    axis /= gap
    height, width = evidence.shape
    ex, ey = end
    x0, y0 = max(0, int(math.floor(ex)) - 9), max(0, int(math.floor(ey)) - 9)
    x1, y1 = min(width, int(math.floor(ex)) + 10), min(height, int(math.floor(ey)) + 10)
    mask = evidence.centerline[y0:y1, x0:x1] & (evidence.center_score[y0:y1, x0:x1] >= 0.15)
    ys, xs = np.nonzero(mask)
    considered = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        if len(_neighbours(mask, (x, y))) != 1:
            continue
        point = (float(x + x0), float(y + y0))
        delta = np.asarray(point) - np.asarray(start)
        target_distance = math.hypot(point[0] - ex, point[1] - ey)
        along = float(np.dot(delta, axis))
        normal = abs(float(delta[0] * axis[1] - delta[1] * axis[0]))
        if not (
            3.0 < target_distance <= 8.0 and gap * 0.5 <= along <= gap - 3.0
            # Match the fixed native anchor uncertainty. Expanding this
            # corridor with gap length would absorb a nested U endpoint
            # several pixels to the side even though the explicit pair
            # does not leap over that neighbouring contour.
            and normal <= _ANCHOR_RADIUS
        ):
            continue
        considered += 1
        if considered > 16:
            return True
        alternative = _sample(evidence, point)
        if alternative is not None and float(np.dot(alternative, outward)) >= 0.80:
            return True
    return False


def sample_confirmed_gap_tangents(
    evidence: LineEvidence,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    *,
    incoming_direction: Optional[Sequence[float]] = None,
) -> Optional[Tuple[Point, Point]]:
    """Return signed outward endpoint directions, or ``None`` if uncertain.

    Both vectors point from their own visible fragment into the gap. At the
    destination this is the reverse of forward path direction. Tangents can
    be perpendicular to the chord. The optional trusted incoming direction
    is only an agreement check at the source and cannot supply missing ink.
    Work reads fixed local boxes, walks at most twelve native pixels for each
    endpoint and inspects at most sixteen nearby competing terminals per side.
    No detector, model, whole-raster labels or edits run here.
    """
    if not isinstance(evidence, LineEvidence):
        raise ValueError("evidence must be a LineEvidence snapshot")
    start = _point(start_xy, "start_xy")
    end = _point(end_xy, "end_xy")
    if math.hypot(end[0] - start[0], end[1] - start[1]) <= 1e-9:
        raise ValueError("confirmed gap endpoints must differ")
    incoming = None
    if incoming_direction is not None:
        incoming = np.asarray(_point(incoming_direction, "incoming_direction"), dtype=np.float64)
        magnitude = float(np.linalg.norm(incoming))
        if not math.isfinite(magnitude) or magnitude <= 1e-9:
            raise ValueError("incoming_direction must be non-zero")
        incoming /= magnitude
    first = _sample(evidence, start)
    last = _sample(evidence, end)
    if first is None or last is None:
        return None
    if incoming is not None and float(np.dot(first, incoming)) < 0.5:
        return None
    if (_skips_competing_endpoint(evidence, start, end, last)
            or _skips_competing_endpoint(evidence, end, start, first)):
        return None
    return first, last


def _sustained_stroke_at(mask, pixel):
    """Recognize long, nearly straight local arms without semantic labels."""
    origin = np.asarray(pixel, dtype=np.float64)
    arms = []
    for neighbour in _neighbours(mask, pixel):
        pending = [([pixel, neighbour], math.hypot(neighbour[0] - pixel[0], neighbour[1] - pixel[1]))]
        inspected = 0
        best = None
        while pending:
            path, length = pending.pop()
            inspected += 1
            if inspected > 64:
                return True  # Dense branching is an unresolved obstruction.
            points = np.asarray(path, dtype=np.float64) - origin
            chord = float(np.linalg.norm(points[-1]))
            if chord <= 0.0:
                continue
            direction = points[-1] / chord
            normal = np.abs(points[:, 0] * direction[1] - points[:, 1] * direction[0])
            if chord < length * 0.88 or float(np.max(normal)) > 1.25:
                continue
            if length >= 6.0:
                best = direction, length
            if length >= 12.0 or len(path) >= 15:
                continue
            following = [point for point in _neighbours(mask, path[-1]) if point not in path]
            for point in following:
                extra = math.hypot(point[0] - path[-1][0], point[1] - path[-1][1])
                pending.append((path + [point], length + extra))
        if best is not None:
            arms.append(best)
    if any(length >= 12.0 for _direction, length in arms):
        return True
    return any(
        first_length + last_length >= 14.0 and float(np.dot(first, last)) < -0.80
        for index, (first, first_length) in enumerate(arms)
        for last, last_length in arms[index + 1:]
    )


def _compact_straight_fragment(component):
    """A short detector break must not turn a clear line into a glyph."""
    points = np.asarray(tuple(component), dtype=np.float64)
    if len(points) < 8 or float(np.max(np.ptp(points, axis=0))) < 8.0:
        return False
    covariance = np.cov(points.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(eigenvalues[-1] / max(float(eigenvalues.sum()), 1e-9)) >= 0.94


def confirmed_gap_curve_is_clear(evidence: LineEvidence, points_xy) -> bool:
    """Refuse a proposed curve touching a sustained local native stroke.

    Tiny marks and compact nonlinear components may occupy a printed-label
    blank. They are allowed by explicit geometry bounds, never classified as
    text. A compact but elongated line fragment still blocks. Contacts in
    larger or fused components are checked for sustained, nearly straight
    local arms; a connection elsewhere in a numeral's component cannot by
    itself classify that numeral as a contour. Excessive work or branching
    blocks the proposal. The first and last
    three pixels accommodate the exact confirmed anchors and native end caps.
    Component walks stay inside the curve's bounded local box and never
    alter the observed evidence or paint route support.
    """
    if not isinstance(evidence, LineEvidence):
        raise ValueError("evidence must be a LineEvidence snapshot")
    try:
        points = np.asarray(points_xy, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if (points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 1024
            or not np.isfinite(points).all()):
        return False
    height, width = evidence.shape
    if np.any((points[:, 0] < 0) | (points[:, 0] >= width)
              | (points[:, 1] < 0) | (points[:, 1] >= height)):
        return False
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    if float(np.max(maximum - minimum)) > 256.0:
        return False
    x0, y0 = max(0, int(math.floor(minimum[0])) - 20), max(0, int(math.floor(minimum[1])) - 20)
    x1, y1 = min(width, int(math.ceil(maximum[0])) + 21), min(height, int(math.ceil(maximum[1])) + 21)
    if (x1 - x0) * (y1 - y0) > 65536:
        return False
    mask = evidence.centerline[y0:y1, x0:x1] & (evidence.center_score[y0:y1, x0:x1] >= 0.15)
    local = points - np.asarray((x0, y0))
    contacts = set()
    # Half-pixel sampling plus a one-pixel inspection neighbourhood also
    # catches diagonal segment intersections between occupied raster cells.
    sampled = 0
    for first, last in zip(local, local[1:]):
        steps = max(1, int(math.ceil(float(np.linalg.norm(last - first)) * 2)))
        sampled += steps
        if sampled > 2048:
            return False
        for t in np.linspace(0.0, 1.0, steps + 1):
            point = first + t * (last - first)
            if min(float(np.linalg.norm(point - local[0])),
                   float(np.linalg.norm(point - local[-1]))) <= 3.0:
                continue
            cx, cy = np.rint(point).astype(int)
            for y in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                for x in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                    if mask[y, x] and math.hypot(x - point[0], y - point[1]) <= 1.25:
                        contacts.add((x, y))
    known = {}
    for seed in contacts:
        if seed in known:
            if not known[seed] and _sustained_stroke_at(mask, seed):
                return False
            continue
        pending = [seed]
        component = {seed}
        min_x = max_x = seed[0]
        min_y = max_y = seed[1]
        compact = True
        while pending:
            x, y = pending.pop()
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            if (max(max_x - min_x, max_y - min_y) >= 18 or len(component) > 256
                    or x == 0 or y == 0 or x == mask.shape[1] - 1 or y == mask.shape[0] - 1):
                compact = False
                break
            # Component identity uses all native8 edges. Suppressing a
            # redundant triangle here could separate a fused printed glyph.
            for ny in range(max(0, y - 1), min(mask.shape[0], y + 2)):
                for nx in range(max(0, x - 1), min(mask.shape[1], x + 2)):
                    neighbour = nx, ny
                    if mask[ny, nx] and neighbour not in component:
                        component.add(neighbour)
                        pending.append(neighbour)
        for pixel in component:
            known[pixel] = compact
        if compact and _compact_straight_fragment(component):
            return False
        if not compact and _sustained_stroke_at(mask, seed):
            return False
    return True


__all__ = ["sample_confirmed_gap_tangents", "confirmed_gap_curve_is_clear"]
