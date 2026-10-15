"""Bounded native-pixel proposals for compact printed label occlusions.

Two similarly sized, separately printed enclosed counters can identify a
small glyph group without reading its number.  This detector deliberately
does not infer contour identity or manufacture a bridge.  Its immutable
masks only say where later geometry checks may inspect a possible text gap.
Single loops, grid cells and open strokes are insufficient evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import numpy as np

from .observed_connection import ObservedConnectionCancelled


@dataclass(frozen=True)
class LabelOcclusion:
    """One compact proposal; ``mask`` uses end-exclusive bbox coordinates.

    ``score`` is a geometric agreement score, not a probability.  The mask
    contains the convex footprint of compact glyph ink and a two-pixel native
    margin.  It never fills the whole axis-aligned bounding rectangle.
    """

    bbox_xyxy: Tuple[int, int, int, int]
    mask: np.ndarray
    baseline_xy: Tuple[float, float]
    counter_count: int
    score: float
    polygon_xy: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self):
        bounds = self.bbox_xyxy
        if (not isinstance(bounds, tuple) or len(bounds) != 4
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in bounds)):
            raise ValueError("label bounds must be four integers")
        x0, y0, x1, y1 = map(int, bounds)
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("label bounds must be a non-empty native rectangle")
        mask = np.array(self.mask, dtype=bool, order="C", copy=True)
        if mask.shape != (y1 - y0, x1 - x0) or not mask.any():
            raise ValueError("label mask must fill the native bounds")
        axis = np.asarray(self.baseline_xy, dtype=np.float64)
        if axis.shape != (2,) or not np.isfinite(axis).all() or not math.isclose(float(np.linalg.norm(axis)), 1.0, abs_tol=1e-6):
            raise ValueError("label baseline must be a finite unit vector")
        if isinstance(self.counter_count, (bool, np.bool_)) or not isinstance(self.counter_count, (int, np.integer)) or self.counter_count < 2:
            raise ValueError("a label proposal requires at least two counters")
        if not math.isfinite(float(self.score)) or not 0.0 <= self.score <= 1.0:
            raise ValueError("label score must be between zero and one")
        polygon = np.asarray(self.polygon_xy, dtype=np.float64)
        if polygon.size and (polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all()):
            raise ValueError("label polygon must contain finite native points")
        mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "bbox_xyxy", (x0, y0, x1, y1))
        object.__setattr__(self, "baseline_xy", tuple(map(float, axis)))
        object.__setattr__(self, "polygon_xy", tuple(tuple(map(float, point)) for point in polygon))


def _check_cancel(cancel_check):
    if cancel_check is not None and cancel_check():
        raise ObservedConnectionCancelled("label detection cancelled")


def _gray(image):
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("label detection needs native Byte RGB or gray pixels")
    if array.ndim == 2:
        gray = array.astype(np.float32)
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = array.astype(np.float32).mean(axis=2)
    else:
        raise ValueError("label detection needs native Byte RGB or gray pixels")
    if not gray.size or gray.ndim != 2 or gray.size > 1024 * 1024:
        raise ValueError("label detection window must contain 1 to 1048576 pixels")
    return gray


def _convex_footprint(points, bounds):
    """Rasterise a compact monotone hull; coordinates remain native pixels."""
    ordered = sorted(set(map(tuple, points.tolist())))

    def cross(first, middle, last):
        return ((middle[0] - first[0]) * (last[1] - first[1])
                - (middle[1] - first[1]) * (last[0] - first[0]))

    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    polygon = lower[:-1] + upper[:-1]
    x0, y0, x1, y1 = bounds
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
    for first, last in zip(polygon, polygon[1:] + polygon[:1]):
        mask &= ((last[0] - first[0]) * (ys - first[1])
                 - (last[1] - first[1]) * (xs - first[0])) >= 0
    return mask, tuple(tuple(map(float, point)) for point in polygon)


def detect_label_occlusions(
    image: np.ndarray,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[LabelOcclusion, ...]:
    """Return compact glyph proposals, or an empty tuple if none is reliable.

    Work is linear in the bounded input plus at most 512 counter candidates
    and 64 retained regions.  SciPy is optional, as for the tracing graph;
    callers should keep the existing route if it is unavailable.  No OCR,
    image resizing, model or platform-specific executable is required.
    """

    _check_cancel(cancel_check)
    gray = _gray(image)
    from scipy import ndimage as ndi

    paper = float(np.percentile(gray, 80))
    threshold = min(paper * 0.75, max(paper * 0.35, float(np.percentile(gray, 15))))
    ink = gray < threshold
    if not ink.any() or ink.all():
        return ()
    _check_cancel(cancel_check)
    components, _ = ndi.label(ink, structure=np.ones((3, 3), dtype=bool))
    component_boxes = ndi.find_objects(components)
    component_sizes = np.bincount(components.ravel())
    white, _ = ndi.label(~ink)
    height, width = gray.shape
    holes = []
    for index, box in enumerate(ndi.find_objects(white), 1):
        if index % 128 == 0:
            _check_cancel(cancel_check)
        if box is None:
            continue
        rows, cols = box
        h, w = rows.stop - rows.start, cols.stop - cols.start
        if (rows.start == 0 or cols.start == 0 or rows.stop == height or cols.stop == width
                or min(h, w) < 2 or max(h, w) > 14):
            continue
        mask = white[box] == index
        size = int(mask.sum())
        if not 4 <= size <= 80:
            continue
        ys, xs = np.nonzero(mask)
        points = np.column_stack((xs + cols.start, ys + rows.start)).astype(np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
        elongation = float(eigenvalues[-1] / max(eigenvalues[0], 0.25))
        if elongation < 1.5:
            continue
        # A compact white hole must belong to exactly one printed component.
        # Inspect only its fixed local border, never a full-image mask per hole.
        near = np.s_[rows.start - 1:rows.stop + 1, cols.start - 1:cols.stop + 1]
        local = white[near] == index
        border = ndi.binary_dilation(local) & ink[near]
        parents = np.unique(components[near][border])
        parents = parents[parents > 0]
        if len(parents) != 1:
            continue
        holes.append((points.mean(axis=0), eigenvectors[:, -1], size, max(h, w), int(parents[0])))
        if len(holes) > 512:
            return ()

    def component_points(identity):
        rows, cols = component_boxes[identity - 1]
        ys, xs = np.nonzero(components[rows, cols] == identity)
        return np.column_stack((xs + cols.start, ys + rows.start))

    found = []
    used = set()
    for first_index, first in enumerate(holes):
        _check_cancel(cancel_check)
        for last in holes[:first_index]:
            center, major, area, span, parent = first
            other, other_major, other_area, other_span, other_parent = last
            if parent == other_parent:
                continue
            difference = center - other
            distance = float(np.linalg.norm(difference))
            if not 1.1 * max(span, other_span) <= distance <= 2.8 * min(span, other_span):
                continue
            axis = difference / distance
            if (min(area, other_area) / max(area, other_area) < 0.45
                    or abs(float(np.dot(axis, major))) > 0.80
                    or abs(float(np.dot(axis, other_major))) > 0.80
                    or abs(float(np.dot(major, other_major))) < 0.70):
                continue
            parents = {parent, other_parent}
            compact = True
            for identity in parents:
                rows, cols = component_boxes[identity - 1]
                if max(rows.stop - rows.start, cols.stop - cols.start) > 3.0 * distance:
                    compact = False
                    break
            if not compact:
                continue
            # Only whole compact ink components may extend the group, such
            # as a leading open digit.  Long contour or grid components fail
            # these native geometric bounds and are never partly swallowed.
            origin = (center + other) * 0.5
            normal = np.asarray((-axis[1], axis[0]))
            radius = int(math.ceil(distance * 2.2))
            x0, y0 = np.maximum(0, np.floor(origin - radius).astype(int))
            x1, y1 = np.minimum((width, height), np.ceil(origin + radius + 1).astype(int))
            nearby = np.unique(components[y0:y1, x0:x1])
            typical_area = min(component_sizes[parent], component_sizes[other_parent])
            for identity in nearby.tolist():
                if not identity or identity in parents:
                    continue
                rows, cols = component_boxes[identity - 1]
                if (max(rows.stop - rows.start, cols.stop - cols.start) > distance * 1.8
                        or component_sizes[identity] < typical_area * 0.45):
                    continue
                points = component_points(identity)
                offsets = points - origin
                along, across = offsets @ axis, offsets @ normal
                if (np.max(np.abs(along)) <= distance * 1.85
                        and np.max(np.abs(across)) <= distance * 1.40
                        and np.ptp(along) <= distance * 1.6
                        and np.ptp(across) <= distance * 1.6):
                    parents.add(identity)
            key = frozenset(parents)
            if key in used:
                continue
            used.add(key)
            points = np.concatenate([component_points(identity) for identity in sorted(parents)])
            x0, y0 = np.maximum(0, points.min(axis=0) - 2)
            x1, y1 = np.minimum((width, height), points.max(axis=0) + 3)
            local, polygon = _convex_footprint(points, (x0, y0, x1, y1))
            local = ndi.binary_dilation(local, structure=np.ones((3, 3), dtype=bool), iterations=2)
            if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
                axis = -axis
            score = min(area, other_area) / max(area, other_area)
            found.append(LabelOcclusion((int(x0), int(y0), int(x1), int(y1)), local, tuple(axis), 2, score, polygon))
            if len(found) >= 64:
                return tuple(found)
    return tuple(found)


def combined_occlusion_mask(regions, shape):
    """Place immutable local proposal masks in one immutable native mask."""
    if (not isinstance(shape, tuple) or len(shape) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in shape)
            or int(shape[0]) * int(shape[1]) > 1024 * 1024):
        raise ValueError("occlusion shape must be a bounded native image shape")
    result = np.zeros(shape, dtype=bool)
    for region in regions:
        if not isinstance(region, LabelOcclusion):
            raise ValueError("occlusions must be LabelOcclusion snapshots")
        x0, y0, x1, y1 = region.bbox_xyxy
        if x1 > shape[1] or y1 > shape[0]:
            raise ValueError("label occlusion leaves the native image")
        result[y0:y1, x0:x1] |= region.mask
    result.setflags(write=False)
    return result


__all__ = ["LabelOcclusion", "detect_label_occlusions", "combined_occlusion_mask"]
