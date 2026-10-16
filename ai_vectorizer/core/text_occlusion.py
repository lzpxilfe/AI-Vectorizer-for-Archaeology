"""Conservative native-pixel proposals for groups of printed letters.

These snapshots identify possible text ink, never contour identity.  A group
needs several compact, aligned components with independent branched shapes.
Each component keeps its own footprint, and its actual ink is stored apart
from the context margin.  No word rectangle or glyph hole erases visible ink.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import numpy as np

from .label_occlusion import _check_cancel, _convex_footprint, _gray


@dataclass(frozen=True)
class TextOcclusion:
    """An immutable glyph-component proposal supported by a larger group.

    ``glyph_count`` counts independently observed compact ink components;
    it is not an OCR character count. ``mask`` is the local context footprint;
    only ``core_mask`` contains the actual selected component ink.  Neither
    grants permission to create an automatic endpoint in the native graph.
    """

    bbox_xyxy: Tuple[int, int, int, int]
    mask: np.ndarray
    core_mask: np.ndarray
    baseline_xy: Tuple[float, float]
    glyph_count: int
    branched_count: int
    score: float
    polygon_xy: Tuple[Tuple[float, float], ...]
    ink_class: str
    evidence_kind: str = "aligned_components"
    cell_count: int = 0

    def __post_init__(self):
        bounds = self.bbox_xyxy
        if (not isinstance(bounds, tuple) or len(bounds) != 4
                or any(isinstance(v, (bool, np.bool_))
                       or not isinstance(v, (int, np.integer)) for v in bounds)):
            raise ValueError("text bounds must be four integers")
        x0, y0, x1, y1 = map(int, bounds)
        if not (0 <= x0 < x1 and 0 <= y0 < y1) or (x1 - x0) * (y1 - y0) > 1024 * 1024:
            raise ValueError("text bounds must be a bounded native rectangle")
        shape = (y1 - y0, x1 - x0)
        mask = np.array(self.mask, dtype=bool, order="C", copy=True)
        core = np.array(self.core_mask, dtype=bool, order="C", copy=True)
        if (mask.shape != shape or core.shape != shape or not core.any()
                or np.any(core & ~mask)):
            raise ValueError("text ink must be non-empty and inside its local context mask")
        axis = np.asarray(self.baseline_xy, dtype=np.float64)
        if (axis.shape != (2,) or not np.isfinite(axis).all()
                or not math.isclose(float(np.linalg.norm(axis)), 1.0, abs_tol=1e-6)):
            raise ValueError("text baseline must be a finite unit vector")
        for value in (self.glyph_count, self.branched_count, self.cell_count):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError("text component counts must be integers")
        if self.evidence_kind == "aligned_components":
            if (not 3 <= self.glyph_count <= 16
                    or not 2 <= self.branched_count <= self.glyph_count or self.cell_count != 0):
                raise ValueError("aligned text needs three components and two independently branched shapes")
        elif self.evidence_kind == "cells_and_heading":
            if self.glyph_count != 2 or self.branched_count != 1 or self.cell_count != 2:
                raise ValueError("structural text needs a branched two-cell body and a separate heading")
        else:
            raise ValueError("unsupported text evidence kind")
        if not math.isfinite(float(self.score)) or not 0.0 <= self.score <= 1.0:
            raise ValueError("text score must be between zero and one")
        polygon = np.asarray(self.polygon_xy, dtype=np.float64)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all()):
            raise ValueError("text polygon must contain finite native points")
        if self.ink_class not in ("gray", "blue"):
            raise ValueError("text ink class must be gray or blue")
        mask.setflags(write=False)
        core.setflags(write=False)
        object.__setattr__(self, "bbox_xyxy", (x0, y0, x1, y1))
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "core_mask", core)
        object.__setattr__(self, "baseline_xy", tuple(map(float, axis)))
        object.__setattr__(self, "polygon_xy", tuple(tuple(map(float, p)) for p in polygon))


class _TextBudgetExceeded(RuntimeError):
    pass


def _branched_shape(mask, *, include_layout=False):
    # Keep this descriptor identical in QGIS and the isolated Python tests.
    # No image-wide skeleton or optional accelerated thinning is requested.
    from .edge_detector import EdgeDetector

    skeleton = EdgeDetector._thin_binary_mask_numpy(mask)
    count = int(skeleton.sum())
    if count < 6:
        return (False, count, np.empty((0, 2))) if include_layout else (False, count)
    padded = np.pad(skeleton, 1)
    neighbours = (
        padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:],
        padded[2:, 2:], padded[2:, 1:-1], padded[2:, :-2],
        padded[1:-1, :-2], padded[:-2, :-2],
    )
    crossing = sum((~first & last).astype(np.uint8)
                   for first, last in zip(neighbours, neighbours[1:] + neighbours[:1]))
    junctions = skeleton & (crossing >= 3)
    branched = bool(junctions.any())
    if not include_layout:
        return branched, count
    from scipy import ndimage as ndi

    groups, total = ndi.label(junctions, structure=np.ones((3, 3), dtype=bool))
    layout = []
    for identity, box in enumerate(ndi.find_objects(groups, max_label=total), 1):
        ys, xs = np.nonzero(groups[box] == identity)
        layout.append(((float(xs.mean()) + box[1].start) / max(1, mask.shape[1] - 1),
                       (float(ys.mean()) + box[0].start) / max(1, mask.shape[0] - 1)))
    return branched, count, np.asarray(layout, dtype=np.float64).reshape(-1, 2)


def _repeated_branch_layout(members):
    """Decline nearly repeated symbols despite small scan/scale variations."""
    first_strips = members[0]["stroke_layout"]
    strips_repeat = True
    for member in members[1:]:
        for first_axis, other_axis in zip(first_strips, member["stroke_layout"]):
            if len(first_axis) != len(other_axis) or np.max(np.abs(first_axis - other_axis)) > 0.18:
                strips_repeat = False
                break
    if strips_repeat:
        return True
    first = members[0]["branch_layout"]
    if len(first) == 0:
        return False
    for member in members[1:]:
        layout = member["branch_layout"]
        if len(layout) != len(first):
            return False
        distances = np.linalg.norm(first[:, None, :] - layout[None, :, :], axis=2)
        if max(float(distances.min(axis=0).max()), float(distances.min(axis=1).max())) > 0.18:
            return False
    return True


def _stroke_layout(mask):
    # Native row/column occupancy preserves repeated crossbars when a one-pixel
    # shortened trunk makes a skeleton junction disappear. Only the descriptor
    # coordinates are normalized; no source pixel is resampled or relocated.
    result = []
    for axis in (1, 0):
        profile = mask.sum(axis=axis)
        strong = profile >= float(profile.max()) * 0.75
        starts = np.flatnonzero(strong & ~np.r_[False, strong[:-1]])
        stops = np.flatnonzero(strong & ~np.r_[strong[1:], False])
        result.append((starts + stops) * 0.5 / max(1, len(profile) - 1))
    return tuple(result)


def _candidate_pairs(candidates, spend, cancel_check):
    """Spatially bounded hypotheses between independently branched glyphs."""
    cells = {}
    result = []
    for index, candidate in enumerate(candidates):
        if index % 32 == 0:
            _check_cancel(cancel_check)
        if not candidate["branched"]:
            continue
        center = candidate["center"]
        cell = tuple(np.floor(center / 36.0).astype(int))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for other in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    spend(1)
                    distance = float(np.linalg.norm(center - candidates[other]["center"]))
                    if 4.0 <= distance <= 36.0:
                        result.append((index, other))
                        if len(result) > 2048:
                            raise _TextBudgetExceeded()
        cells.setdefault(cell, []).append(index)
    return result


def _find_groups(candidates, image_shape, spend, cancel_check):
    if not candidates:
        return ()
    centers = np.asarray([candidate["center"] for candidate in candidates])
    found, seen = [], set()
    height, width = image_shape
    for first, last in _candidate_pairs(candidates, spend, cancel_check):
        _check_cancel(cancel_check)
        axis = centers[first] - centers[last]
        axis /= np.linalg.norm(axis)
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
            axis = -axis
        normal = np.asarray((-axis[1], axis[0]))
        offsets = centers - centers[first]
        spend(len(candidates))
        along, across = offsets @ axis, offsets @ normal
        near = np.flatnonzero((np.abs(along) <= 160.0) & (np.abs(across) <= 3.0))
        projected = []
        for index in near.tolist():
            candidate = candidates[index]
            points = candidate["points"] - centers[first]
            spend(len(points))
            u, v = points @ axis, points @ normal
            extents = (float(u.min()), float(u.max()), float(v.min()), float(v.max()))
            glyph_width = extents[1] - extents[0] + 1.0
            glyph_height = extents[3] - extents[2] + 1.0
            if 2.0 <= glyph_width <= 25.0 and 5.0 <= glyph_height <= 28.0:
                projected.append((float(along[index]), index, extents, glyph_height))
        projected.sort(key=lambda value: value[0])
        clusters, cluster = [], []
        for value in projected:
            if cluster:
                previous = cluster[-1]
                gap = value[2][0] - previous[2][1] - 1.0
                if (gap < -1.0 or gap > min(value[3], previous[3]) * 1.1
                        or min(value[3], previous[3]) / max(value[3], previous[3]) < 0.55):
                    clusters.append(cluster)
                    cluster = []
            cluster.append(value)
        if cluster:
            clusters.append(cluster)
        for cluster in clusters:
            key = frozenset(value[1] for value in cluster)
            if key in seen or not 3 <= len(key) <= 16:
                continue
            members = [candidates[value[1]] for value in cluster]
            branched = sum(member["branched"] for member in members)
            if branched < 2:
                continue
            # A repeated tree/vegetation symbol does not provide independent
            # alphabetic evidence merely because three marks share a row.
            statistics = np.asarray([[member["area"], member["skeleton_size"]] for member in members])
            if np.all(statistics.min(axis=0) / np.maximum(1, statistics.max(axis=0)) > 0.92):
                continue
            if _repeated_branch_layout(members):
                continue
            group_centers = np.asarray([member["center"] for member in members])
            _values, vectors = np.linalg.eigh(np.cov(group_centers.T))
            baseline = vectors[:, -1]
            if baseline[0] < 0.0:
                baseline = -baseline
            cross = np.asarray((-baseline[1], baseline[0]))
            residual = float(np.max(np.abs((group_centers - group_centers.mean(axis=0)) @ cross)))
            heights = np.asarray([value[3] for value in cluster])
            if residual > max(1.5, float(np.median(heights)) * 0.15) or heights.min() / heights.max() < 0.55:
                continue
            points = np.concatenate([member["points"] for member in members])
            margin = float(heights.max()) + 2.0
            if (np.any(points.min(axis=0) < margin)
                    or np.any(np.asarray((width - 1, height - 1)) - points.max(axis=0) < margin)):
                continue
            seen.add(key)
            found.append((tuple(value[1] for value in cluster), baseline, residual, branched))
            if len(found) > 128:
                raise _TextBudgetExceeded()
    return tuple(found)


def _structural_groups(ink, components, boxes, image_shape, spend, cancel_check):
    """A compact branched two-cell body with a separate printed heading bar.

    This is a different source signature from an aligned alphabetic group.
    A lone B/8, two touching loops, a grid or a decorative line is insufficient.
    The body and bar remain whole observed components; no OCR is performed.
    """
    from scipy import ndimage as ndi

    height, width = image_shape
    white, count = ndi.label(~ink)
    if count > 8192:
        return ()
    holes = {}
    for identity, box in enumerate(ndi.find_objects(white), 1):
        if identity % 64 == 0:
            _check_cancel(cancel_check)
        if box is None:
            continue
        rows, cols = box
        h, w = rows.stop - rows.start, cols.stop - cols.start
        if (rows.start == 0 or cols.start == 0 or rows.stop == height or cols.stop == width
                or min(h, w) < 2 or max(h, w) > 14):
            continue
        local = white[box] == identity
        area = int(local.sum())
        if not 4 <= area <= 80:
            continue
        spend(h * w)
        ys, xs = np.nonzero(local)
        points = np.column_stack((xs + cols.start, ys + rows.start)).astype(np.float64)
        eigenvalues, vectors = np.linalg.eigh(np.cov(points.T))
        if eigenvalues[-1] / max(eigenvalues[0], 0.25) < 1.5:
            continue
        near = np.s_[rows.start - 1:rows.stop + 1, cols.start - 1:cols.stop + 1]
        border = ndi.binary_dilation(white[near] == identity) & ink[near]
        parents = np.unique(components[near][border])
        parents = parents[parents > 0]
        if len(parents) != 1:
            continue
        parent = int(parents[0])
        parent_rows, parent_cols = boxes[parent - 1]
        if max(parent_rows.stop - parent_rows.start, parent_cols.stop - parent_cols.start) > 28:
            continue
        holes.setdefault(parent, []).append((points.mean(axis=0), vectors[:, -1], area))
        if len(holes) > 256:
            raise _TextBudgetExceeded()

    def points_for(identity):
        rows, cols = boxes[identity - 1]
        local = components[rows, cols] == identity
        ys, xs = np.nonzero(local)
        spend(local.size)
        return np.column_stack((xs + cols.start, ys + rows.start)).astype(np.float64), local

    found = []
    for parent, cells in holes.items():
        _check_cancel(cancel_check)
        if len(cells) != 2:
            continue
        first, last = cells
        difference = last[0] - first[0]
        spacing = float(np.linalg.norm(difference))
        if not 3.0 <= spacing <= 14.0:
            continue
        axis = difference / spacing
        if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
            axis = -axis
        if (abs(float(np.dot(axis, first[1]))) > 0.5
                or abs(float(np.dot(axis, last[1]))) > 0.5
                or abs(float(np.dot(first[1], last[1]))) < 0.80
                or min(first[2], last[2]) / max(first[2], last[2]) < 0.40):
            continue
        body, body_mask = points_for(parent)
        if not 50 <= len(body) <= 240 or not _branched_shape(body_mask)[0]:
            continue
        normal = np.asarray((-axis[1], axis[0]))
        center = body.mean(axis=0)
        relative = body - center
        body_u, body_v = relative @ axis, relative @ normal
        body_width = float(np.ptp(body_u)) + 1.0
        body_height = float(np.ptp(body_v)) + 1.0
        if (not 10.0 <= body_width <= 28.0 or not 8.0 <= body_height <= 28.0
                or not 0.65 <= body_width / body_height <= 1.8):
            continue
        bx0, by0 = np.maximum(0, np.floor(body.min(axis=0) - 8).astype(int))
        bx1, by1 = np.minimum((width, height), np.ceil(body.max(axis=0) + 9).astype(int))
        headings = []
        for other in np.unique(components[by0:by1, bx0:bx1]).tolist():
            if not other or other == parent:
                continue
            rows, cols = boxes[other - 1]
            if max(rows.stop - rows.start, cols.stop - cols.start) > 28:
                continue
            bar, bar_mask = points_for(other)
            if not 12 <= len(bar) <= 120 or _branched_shape(bar_mask)[0]:
                continue
            along, across = (bar - center) @ axis, (bar - center) @ normal
            values, vectors = np.linalg.eigh(np.cov(bar.T))
            bar_axis = vectors[:, -1]
            bar_normal = np.asarray((-bar_axis[1], bar_axis[0]))
            span = float(np.ptp(along)) + 1.0
            thickness = float(np.ptp(bar @ bar_normal)) + 1.0
            if not body_width * 0.55 <= span <= body_width * 1.25 or thickness > 4.0:
                continue
            # A detached line must sit outside the body, close to one of its
            # sides, and overlap its width. An arbitrary neighboring stroke
            # or a continuation of the glyph cannot serve as a heading.
            gap = max(float(across.min() - body_v.max()), float(body_v.min() - across.max()))
            if (not 1.0 <= gap <= 7.0 or abs(float(along.mean())) > body_width * 0.35
                    or float(along.min()) < float(body_u.min()) - 3.0
                    or float(along.max()) > float(body_u.max()) + 3.0):
                continue
            agreement = abs(float(np.dot(vectors[:, -1], axis)))
            if values[-1] < max(values[0], 0.25) * 8.0 or agreement < 0.90:
                continue
            all_points = np.concatenate((body, bar))
            margin = max(body_width, body_height) + 2.0
            if (np.any(all_points.min(axis=0) < margin)
                    or np.any(np.asarray((width - 1, height - 1)) - all_points.max(axis=0) < margin)):
                continue
            score = agreement * min(first[2], last[2]) / max(first[2], last[2])
            headings.append((score, all_points))
        # Two plausible headings are an ambiguity, not an enumeration tie.
        if len(headings) == 1:
            score, points = headings[0]
            found.append((points, axis, score))
        if len(found) > 64:
            raise _TextBudgetExceeded()
    return tuple(found)


def _snapshot(points, baseline, score, image_shape, ink_class, *,
              glyph_count, branched_count, evidence_kind="aligned_components", cell_count=0):
    from scipy import ndimage as ndi

    height, width = image_shape
    points = points.astype(int)
    x0, y0 = np.maximum(0, points.min(axis=0) - 2)
    x1, y1 = np.minimum((width, height), points.max(axis=0) + 3)
    mask, polygon = _convex_footprint(points, (x0, y0, x1, y1))
    mask = ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)
    core = np.zeros(mask.shape, dtype=bool)
    core[points[:, 1] - y0, points[:, 0] - x0] = True
    return TextOcclusion(
        (int(x0), int(y0), int(x1), int(y1)), mask, core, tuple(baseline),
        glyph_count, branched_count, score, polygon, ink_class, evidence_kind, cell_count,
    )


def _detect_channel(ink, ink_class, blue, image_shape, cancel_check):
    from scipy import ndimage as ndi

    work = 0

    def spend(amount):
        nonlocal work
        work += amount
        if work > 262144:
            raise _TextBudgetExceeded()

    components, count = ndi.label(ink, structure=np.ones((3, 3), dtype=bool))
    if count > 8192:
        return ()
    boxes = ndi.find_objects(components)
    structural = _structural_groups(ink, components, boxes, image_shape, spend, cancel_check)
    candidates = []
    height, width = image_shape
    for identity, box in enumerate(boxes, 1):
        if identity % 64 == 0:
            _check_cancel(cancel_check)
        if box is None:
            continue
        rows, cols = box
        h, w = rows.stop - rows.start, cols.stop - cols.start
        if (rows.start <= 2 or cols.start <= 2 or rows.stop >= height - 2 or cols.stop >= width - 2
                or min(h, w) < 2 or max(h, w) > 28):
            continue
        local = components[box] == identity
        area = int(local.sum())
        if not 8 <= area <= 240 or not 0.1 <= area / (h * w) <= 0.90:
            continue
        if ink_class == "gray" and blue is not None and float(np.mean(blue[box][local])) >= 0.20:
            continue
        spend(h * w)
        branched, skeleton_size, branch_layout = _branched_shape(local, include_layout=True)
        ys, xs = np.nonzero(local)
        points = np.column_stack((xs + cols.start, ys + rows.start)).astype(np.float64)
        # A one-pixel diagonal may have a two-dimensional bounding box, but
        # still has no footprint. It cannot complete an otherwise weak pair.
        if np.linalg.matrix_rank(points - points[0]) < 2:
            continue
        candidates.append({
            "points": points, "center": points.mean(axis=0), "area": area,
            "branched": branched, "skeleton_size": skeleton_size, "branch_layout": branch_layout,
            "stroke_layout": _stroke_layout(local),
        })
        if len(candidates) > 512:
            return ()
    groups = _find_groups(candidates, image_shape, spend, cancel_check)
    claimed = set()
    found = [_snapshot(points, baseline, score, image_shape, ink_class,
                       glyph_count=2, branched_count=1, evidence_kind="cells_and_heading", cell_count=2)
             for points, baseline, score in structural]
    for members, baseline, residual, branched in sorted(groups, key=lambda value: -len(value[0])):
        _check_cancel(cancel_check)
        # Every output still belongs to a complete independently supported
        # group. Overlapping alternative baseline hypotheses do not add pixels.
        if claimed.intersection(members):
            continue
        if len(found) + len(members) > 64:
            continue
        claimed.update(members)
        for index in members:
            found.append(_snapshot(
                candidates[index]["points"], baseline, 1.0 / (1.0 + residual), image_shape, ink_class,
                glyph_count=len(members), branched_count=branched,
            ))
    return tuple(found)


def detect_text_occlusions(
    image: np.ndarray,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[TextOcclusion, ...]:
    """Return source-only compact letter proposals without OCR or resizing.

    Gray and independently separated blue print use separate complete work
    budgets. A crowded channel declines as a whole, preserving any completed
    other channel. At most 64 component snapshots are returned. SciPy remains
    optional for callers; cancellation propagates through the existing worker
    exception contract. This detector never examines prompts, models or paths.
    """
    _check_cancel(cancel_check)
    gray = _gray(image)
    paper = float(np.percentile(gray, 80))
    threshold = min(paper * 0.75, max(paper * 0.35, float(np.percentile(gray, 15))))
    blue = None
    channels = []
    source = np.asarray(image)
    if source.ndim == 3:
        red, green, channel_blue = source.astype(np.int16).transpose(2, 0, 1)
        blue = ((channel_blue - red >= 18) & (channel_blue - green >= 5)
                & (gray < paper * 0.85))
        if blue.any():
            channels.append(("blue", blue))
    channels.append(("gray", gray < threshold))
    found = []
    for ink_class, ink in channels:
        _check_cancel(cancel_check)
        if not ink.any() or ink.all():
            continue
        try:
            regions = _detect_channel(ink, ink_class, blue, gray.shape, cancel_check)
        except _TextBudgetExceeded:
            continue
        # Do not truncate a supported group at the cross-channel output cap.
        if len(found) + len(regions) <= 64:
            found.extend(regions)
    return tuple(found)


__all__ = ["TextOcclusion", "detect_text_occlusions"]
