"""Sparse, bounded native-arm routes across independently detected labels.

A detector region is an uncertainty mask, never new observed ink. Its pixels
are excluded only from this separate graph. Signed visible arms determine one
retained cubic and all other region arms remain barriers to that continuation.
The original image, LineEvidence and ordinary Live-Wire tree are untouched.
Only genuine endpoints in the original native graph qualify automatically;
cutting a continuous stroke with a label mask cannot create such authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .confirmed_gap_curve import (
    ConfirmedGapCurveConfig, ConfirmedGapCurveError, build_confirmed_gap_curve,
)
from .label_gap import _degree, _neighbours
from .observed_connection import (
    ObservedConnectionCancelled, ObservedConnectionTree, build_observed_connection,
)


@dataclass(frozen=True)
class _Arm:
    pixel: tuple
    outward: tuple
    context: tuple
    width: float
    perimeter: float


@dataclass(frozen=True)
class _Route:
    native_indices: np.ndarray
    lookup: np.ndarray
    predecessors: np.ndarray
    distances: np.ndarray
    prefix: tuple
    prefix_length: float
    target_node: int
    region_index: int
    require_unbranched: bool = False
    native_prefix_count: int = 0


@dataclass(frozen=True)
class OccludedConnectionTree:
    """Precomputed single-curve routes; hover never processes an image."""
    root: tuple
    origin: tuple
    shape: tuple
    mask: np.ndarray
    outside_native: np.ndarray
    routes: tuple
    connectors: tuple
    outside_connection: Optional[ObservedConnectionTree] = None
    outside_degree: Optional[np.ndarray] = None
    text_cores: tuple = ()

    def outside_route(self, target):
        """Preserve a healthy native route, without privileging a junction detour."""
        if self.outside_connection is None:
            return None
        points = self.outside_connection.trace(target)
        if points is not None and tuple(points[0]) != self.root:
            points = [self.root] + points
        if points is not None and self.outside_degree is not None:
            # Native connectivity through distant grid/glyph junctions does
            # not establish contour identity. Prefer an already admissible
            # genuine-endpoint bridge only when its two visible arms remain
            # unbranched and the competing outside route crosses a junction.
            # A continuous unbranched outside U retains its existing priority.
            candidate = self.trace(target)
            selected = None if candidate is None else self._selected_connector(candidate)
            if selected is not None and self._has_native_junction(points) is True:
                _source, _target, _curve, start, end, _region = selected
                arms = [*candidate[:start + 1], *candidate[end - 1:]]
                if self._has_native_junction(arms) is False:
                    return None
        return points

    def _has_native_junction(self, points):
        """Read prepared native degrees only; uncertain geometry keeps priority."""
        values = np.asarray(points, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != 2 or not 2 <= len(values) <= 8192
                or not np.isfinite(values).all()):
            return None
        local = values - np.asarray(self.origin)
        pixels = np.rint(local).astype(np.int64)
        height, width = self.shape
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        if not np.all(valid):
            return None
        valid &= self.outside_native[pixels[:, 1], pixels[:, 0]]
        # Literal root/cursor positions can sit beside the skeleton; all
        # intermediate arm vertices must be original native observations.
        if not np.all(valid[1:-1]):
            return None
        if np.any(self.outside_degree[pixels[valid, 1], pixels[valid, 0]] > 2):
            return True
        # The routing graph retains all eight neighbor edges. A unit
        # diagonal can therefore bypass the shared orthogonal T pixel even
        # though pruned context adjacency correctly gives that pixel degree3.
        # Inspect only those actual edge corners, not a dilated neighborhood.
        delta = np.diff(pixels, axis=0)
        diagonal = (valid[:-1] & valid[1:] & (np.abs(delta[:, 0]) == 1)
                    & (np.abs(delta[:, 1]) == 1))
        for first, last in zip(pixels[:-1][diagonal], pixels[1:][diagonal]):
            for x, y in ((first[0], last[1]), (last[0], first[1])):
                if self.outside_native[y, x] and self.outside_degree[y, x] > 2:
                    return True
        return False

    def intersects(self, path):
        """Whether an existing route enters an independently detected label."""
        points = np.asarray(path, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            return False
        if len(points) > 8192:
            return False
        local = points - np.asarray(self.origin)
        height, width = self.shape
        work = 0
        for first, last in zip(local, local[1:]):
            steps = max(1, int(math.ceil(float(np.linalg.norm(last - first)) * 2)))
            work += steps
            if work > 16384:
                return False
            pixels = np.rint(first + np.linspace(0.0, 1.0, steps + 1)[:, None] * (last - first)).astype(int)
            valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                     & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
            pixels = pixels[valid]
            if len(pixels) and np.any(self.mask[pixels[:, 1], pixels[:, 0]]):
                return True
        return False

    def trace(self, target):
        point = np.asarray(target, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            return None
        local = point - np.asarray(self.origin)
        height, width = self.shape
        if not (0 <= local[0] < width and 0 <= local[1] < height):
            return None
        tx, ty = np.rint(local).astype(int)
        if not (0 <= tx < width and 0 <= ty < height) or self.mask[ty, tx]:
            return None
        chord = float(np.linalg.norm(point - self.root))
        if chord < 1e-9:
            return None
        limit = min(2 * max(self.shape), max(8.0, chord * 6.0))
        yy, xx = np.mgrid[max(0, int(math.floor(local[1] - 2))):min(height, int(math.ceil(local[1] + 2)) + 1),
                          max(0, int(math.floor(local[0] - 2))):min(width, int(math.ceil(local[0] + 2)) + 1)]
        offsets = np.hypot(xx - local[0], yy - local[1])
        native_anchor = self.outside_native[yy, xx] & (offsets <= 2.0 + 1e-6)
        if not np.any(native_anchor):
            return None
        nearest_offset = float(offsets[native_anchor].min())
        closest = native_anchor & (offsets <= nearest_offset + 0.25)
        # Equal nearby arms are an ambiguity, even if a distant intersection
        # makes them one global component. Flood only this tiny anchor box.
        seeds = list(zip(*np.nonzero(closest)))
        pending = [seeds[0]]
        seen = {seeds[0]}
        while pending:
            y, x = pending.pop()
            for ny in range(max(0, y - 1), min(native_anchor.shape[0], y + 2)):
                for nx in range(max(0, x - 1), min(native_anchor.shape[1], x + 2)):
                    if native_anchor[ny, nx] and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        pending.append((ny, nx))
        if any(seed not in seen for seed in seeds):
            return None
        options = []
        for route in self.routes:
            nodes = route.lookup[yy, xx]
            if np.any(nodes[closest] < 0):
                continue
            valid = (nodes >= 0) & (offsets <= 2.0 + 1e-6)
            if not np.any(valid):
                continue
            # Distance authority comes from *all* original outside arms.
            # A route which excludes the clicked neighbouring arm may not
            # recompute a more convenient, farther anchor on its own graph.
            valid &= offsets <= nearest_offset + 0.25
            if not np.any(valid):
                continue
            costs = route.distances[nodes[valid]] + offsets[valid] * 4.0
            index = int(np.argmin(costs))
            if not math.isfinite(float(costs[index])):
                continue
            node = int(nodes[valid][index])
            length = float(offsets[valid][index]) + route.prefix_length
            backwards = []
            previous = point
            for _ in range(min(len(route.native_indices), int(math.ceil(limit)) + 2)):
                native = int(route.native_indices[node])
                current = (float(native % width + self.origin[0]), float(native // width + self.origin[1]))
                if backwards:
                    length += math.hypot(current[0] - previous[0], current[1] - previous[1])
                if length > limit:
                    break
                backwards.append(current)
                previous = current
                if node == route.target_node:
                    if route.require_unbranched:
                        # Printed words can sit over a network joined by a
                        # distant grid or glyph. That global component does
                        # not authorize switching contours after the gap.
                        # Degrees were prepared in the worker, including
                        # diagonal corners that bypass an actual junction.
                        native_prefix = route.prefix[:route.native_prefix_count]
                        if any(
                            len(arm) > 1 and self._has_native_junction(arm) is not False
                            for arm in (native_prefix, backwards)
                        ):
                            break
                    points = [*route.prefix, *backwards[-2::-1], tuple(point)]
                    points = [p for i, p in enumerate(points) if not i or p != points[i - 1]]
                    options.append((float(costs[index]) + route.prefix_length * 2, points))
                    break
                node = int(route.predecessors[node])
                if node < 0 or node >= len(route.native_indices):
                    break
        if not options:
            return None
        options.sort(key=lambda value: value[0])
        # Different inferred continuations to the exact same target must be
        # clearly distinguished, not resolved by their enumeration order.
        if len(options) > 1 and options[1][0] - options[0][0] <= 0.5:
            return None
        return options[0][1]

    def accepts_display(self, displayed):
        """Reject a strength blend that touches a different visible stroke.

        Only immutable geometry and native pixels are read. Glyph pixels in
        the independently detected mask remain uncertain, not obstacles.
        """
        points = np.asarray(displayed, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or not 2 <= len(points) <= 8192
                or not np.isfinite(points).all()):
            return False
        full = self.trace(points[-1])
        if full is None:
            return False
        selected = self._selected_connector(full)
        if selected is None:
            return False
        _source, _target, _curve, start, end, _region = selected
        # Only the original native prefix/suffix and the two genuine ends
        # establish ownership of nearby observed pixels. The virtual curve
        # cannot claim a foreign stroke merely because it passes near it.
        owned = np.concatenate((np.asarray(full[:start + 1]), np.asarray(full[end - 1:])))
        height, width = self.shape
        allowed = set()
        for x, y in owned - np.asarray(self.origin):
            for cy in range(max(0, int(math.floor(y - 1.25))), min(height, int(math.ceil(y + 1.25)) + 1)):
                for cx in range(max(0, int(math.floor(x - 1.25))), min(width, int(math.ceil(x + 1.25)) + 1)):
                    if math.hypot(cx - x, cy - y) <= 1.25:
                        allowed.add((cx, cy))
        local = points - np.asarray(self.origin)
        work = 0
        for first, last in zip(local, local[1:]):
            steps = max(1, int(math.ceil(float(np.linalg.norm(last - first)) * 2)))
            work += steps
            if work > 16384:
                return False
            for x, y in first + np.linspace(0.0, 1.0, steps + 1)[:, None] * (last - first):
                if not (0 <= x < width and 0 <= y < height):
                    return False
                for cy in range(max(0, int(math.floor(y - 1.0))), min(height, int(math.ceil(y + 1.0)) + 1)):
                    for cx in range(max(0, int(math.floor(x - 1.0))), min(width, int(math.ceil(x + 1.0)) + 1)):
                        if (self.outside_native[cy, cx] and (cx, cy) not in allowed
                                and math.hypot(cx - x, cy - y) <= 1.0):
                            return False
        return True

    def _selected_connector(self, inferred):
        """Locate the retained virtual interval in one completed route."""
        points = np.asarray(inferred, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) > 8192:
            return None
        origin = np.asarray(self.origin)
        for source, target, curve, region in self.connectors:
            expected = np.asarray(curve, dtype=np.float64) + origin
            starts = np.flatnonzero(np.linalg.norm(points - expected[0], axis=1) < 1e-7)
            for start in starts:
                end = start + len(expected)
                if end <= len(points) and np.allclose(points[start:end], expected, rtol=0, atol=1e-7):
                    return source, target, curve, int(start), int(end), region
        return None

    def preserves_visible_route(self, ordinary, inferred):
        """Retain an existing route which already follows the selected ends."""
        from .occlusion_route_guard import preserves_visible_route

        selected = self._selected_connector(inferred)
        if selected is not None:
            source, target, _curve, _start, _end, region = selected
            origin = np.asarray(self.origin)
            preserved = preserves_visible_route(
                ordinary, inferred, mask=self.mask, origin=self.origin,
                source_xy=tuple(np.asarray(source) + origin),
                target_xy=tuple(np.asarray(target) + origin),
            )
            if preserved and region < len(self.text_cores) and self.text_cores[region] is not None:
                from .text_route_guard import follows_text_ink

                if follows_text_ink(
                    ordinary, inferred, core=self.text_cores[region], origin=self.origin,
                    source_xy=tuple(np.asarray(source) + origin),
                    target_xy=tuple(np.asarray(target) + origin),
                ):
                    return False
            return preserved
        return False


def _region_mask(region, shape):
    result = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = tuple(int(value) for value in region.bbox_xyxy)
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("occlusion region is outside the image")
    supplied = np.asarray(region.mask, dtype=bool)
    if supplied.shape == shape:
        result[:] = supplied
    elif supplied.shape == (y1 - y0, x1 - x0):
        result[y0:y1, x0:x1] = supplied
    else:
        raise ValueError("occlusion mask dimensions do not match its box")
    return result, (x0, y0, x1, y1)


def _glyph_core(region, proposal):
    """Keep visible strokes in the detector's padding outside glyph ink.

    A proposal's two-pixel collar collects nearby ends; it is not authority
    to erase a continuous contour. The undilated convex glyph footprint is
    the only uncertain area excluded from the native routing graph.
    """
    # General text snapshots distinguish observed glyph ink from their
    # context hull. A contour through a letter hole or inter-letter space
    # remains native evidence, even when it lies inside that hull.
    supplied_core = getattr(region, "core_mask", None)
    if supplied_core is not None:
        x0, y0, x1, y1 = region.bbox_xyxy
        local = np.asarray(supplied_core, dtype=bool)
        if local.shape != (y1 - y0, x1 - x0) or not local.any():
            raise ValueError("text core dimensions do not match its box")
        core = np.zeros(proposal.shape, dtype=bool)
        core[y0:y1, x0:x1] = local
        if np.any(core & ~proposal):
            raise ValueError("text core must stay inside its proposal")
        return core
    polygon = np.asarray(getattr(region, "polygon_xy", ()), dtype=np.float64)
    if polygon.size == 0:
        return proposal.copy()
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("glyph core needs a finite polygon")
    x0, y0, x1, y1 = region.bbox_xyxy
    yy, xx = np.mgrid[y0:y1, x0:x1]
    local = np.ones(xx.shape, dtype=bool)
    for first, last in zip(polygon, np.roll(polygon, -1, axis=0)):
        local &= ((last[0] - first[0]) * (yy - first[1])
                  - (last[1] - first[1]) * (xx - first[0])) >= -1e-8
    core = np.zeros(proposal.shape, dtype=bool)
    core[y0:y1, x0:x1] = local
    return core & proposal


def _perimeter(point, box):
    x, y = point
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    cx, cy = min(x1, max(x0, x)), min(y1, max(y0, y))
    candidates = (
        ((x - cx) ** 2 + (y - y0) ** 2, cx - x0),
        ((x - x1) ** 2 + (y - cy) ** 2, width + cy - y0),
        ((x - cx) ** 2 + (y - y1) ** 2, width + height + x1 - cx),
        ((x - x0) ** 2 + (y - cy) ** 2, 2 * width + height + y1 - cy),
    )
    return min(candidates)[1] % (2 * (width + height))


def _arm(mask, endpoint, width_image, box):
    """Keep sustained native identity while fitting a local endpoint tangent."""
    previous = None
    current = endpoint
    points = [endpoint]
    length = 0.0
    for _ in range(16):
        neighbours = _neighbours(mask, current)
        expected = 1 if previous is None else 2
        if len(neighbours) != expected:
            if length < 8.0:
                return None
            break
        following = [point for point in neighbours if point != previous]
        if len(following) != 1:
            return None
        following = following[0]
        length += math.hypot(following[0] - current[0], following[1] - current[1])
        previous, current = current, following
        points.append(current)
        if length >= 12.0:
            break
    if length < 8.0:
        return None
    context = np.asarray(points, dtype=np.float64)
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(context, axis=0), axis=1))]

    def fit_tangent(fit_context, fit_distances, chord_ratio, residual_limit, agreement):
        if float(np.linalg.norm(fit_context[-1] - fit_context[0])) < fit_distances[-1] * chord_ratio:
            return None
        signed = -fit_distances
        design = np.column_stack((signed, signed ** 2))
        coefficients = np.linalg.lstsq(design, fit_context - fit_context[0], rcond=None)[0]
        residual = design @ coefficients - (fit_context - fit_context[0])
        if float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))) > residual_limit:
            return None
        outward = coefficients[0]
        norm = float(np.linalg.norm(outward))
        if not 0.4 <= norm <= 2.0:
            return None
        outward /= norm
        nearest = max(1, int(np.searchsorted(fit_distances, 4.0)))
        secant = fit_context[0] - fit_context[nearest]
        secant /= np.linalg.norm(secant)
        if float(np.dot(outward, secant)) < agreement:
            return None
        return outward

    # Retain the previous fit exactly whenever its sustained context succeeds.
    # A tight visible bend farther along the same native stroke can invalidate
    # that fit without making its endpoint or its immediate tangent ambiguous.
    outward = fit_tangent(context, distances, 0.70, 1.25, 0.65)
    if outward is None:
        local_fits = []
        for extent in (4.0, 6.0):
            last = int(np.searchsorted(distances, extent)) + 1
            local = fit_tangent(context[:last], distances[:last], 0.75, 0.60, 0.90)
            if local is None:
                return None
            local_fits.append(local)
        # Separate local scales must agree; short glyph fragments, forks and
        # a tangent that changes with the fit window do not gain authority.
        if float(np.dot(local_fits[0], local_fits[1])) < 0.95:
            return None
        outward = local_fits[0]
    stroke_width = 1.0
    if width_image is not None:
        samples = np.asarray(points, dtype=int)[(distances >= 3.0) & (distances <= 10.0)]
        stroke_width = float(np.median(width_image[samples[:, 1], samples[:, 0]]))
        if stroke_width <= 0:
            return None
    return _Arm(endpoint, tuple(outward), tuple(points), stroke_width, _perimeter(endpoint, box))


def _curve_cells(points, shape):
    pixels = np.rint(points).astype(int)
    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < shape[1])
             & (pixels[:, 1] >= 0) & (pixels[:, 1] < shape[0]))
    return pixels[valid]


def _connectors(mask, region_mask, box, width_image, orientation, cancel, *, original_degree=None,
                thin_width_uncertainty=False):
    from scipy.ndimage import distance_transform_edt
    degree = _degree(mask)
    near = distance_transform_edt(~region_mask)
    # Use the narrow dimension so a ninety-degree rotation does not expand
    # the terminal search and introduce a different neighbouring contour.
    collar = min(8.0, max(4.0, min(box[2] - box[0], box[3] - box[1]) * 0.45))
    ys, xs = np.nonzero((degree == 1) & (near <= collar))
    if len(xs) > 64:
        return (), ()
    arms = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        cancel()
        arm = _arm(mask, (x, y), width_image, box)
        # An unusable tangent still denotes a competing observed fragment.
        # Dropping it would let a farther neighbour become the next alleged
        # continuation merely because the correct arm has a short branch.
        arms.append(arm if arm is not None else _Arm(
            (x, y), (), (), 0.0, _perimeter((x, y), box)
        ))
    if len(arms) < 2 or len(arms) > 32:
        return (), tuple(arms)
    arms.sort(key=lambda arm: arm.perimeter)
    axis = np.asarray(orientation, dtype=float)
    if axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 0.5:
        axis = np.asarray((1.0, 0.0))
    axis /= np.linalg.norm(axis)
    extent = abs(axis[0]) * (box[2] - box[0]) + abs(axis[1]) * (box[3] - box[1])
    pairs = []
    votes = [[] for _ in arms]
    for i, first in enumerate(arms):
        for j in range(i + 1, len(arms)):
            cancel()
            last = arms[j]
            if not first.outward or not last.outward:
                continue
            if original_degree is not None and (
                original_degree[first.pixel[1], first.pixel[0]] != 1
                or original_degree[last.pixel[1], last.pixel[0]] != 1
            ):
                continue
            # Non-crossing contours meet a simply connected occlusion boundary
            # in paired order. Skipping an odd number of sustained arms would
            # force a crossing or an unpaired neighbouring contour.
            between = j - i - 1
            perimeter = 2 * ((box[2] - box[0]) + (box[3] - box[1]))
            if last.perimeter - first.perimeter > perimeter * 0.5:
                between = len(arms) - (j - i) - 1
            if between % 2:
                continue
            span = abs(float(np.dot(np.asarray(last.pixel) - first.pixel, axis)))
            if span < max(4.0, extent * 0.60):
                # A contour can cross just one digit near a tight bend.
                # The text baseline is not the contour's direction. Such a
                # short crossing still needs genuine ends and two signed
                # local tangents pointing into the same hidden interval.
                delta = np.asarray(last.pixel, dtype=float) - first.pixel
                length = float(np.linalg.norm(delta))
                if original_degree is None or length < 4.0:
                    continue
                chord = delta / length
                if (np.dot(first.outward, chord) < 0.50
                        or np.dot(last.outward, -chord) < 0.50):
                    continue
            ratio = max(first.width, last.width) / min(first.width, last.width)
            if width_image is not None and ratio > 1.55:
                continue
            try:
                curve = build_confirmed_gap_curve(
                    first.pixel, last.pixel, first.outward, last.outward,
                    config=ConfirmedGapCurveConfig(handle_ratio=0.55),
                )
            except ConfirmedGapCurveError:
                continue
            if curve.gap_length_pixels > 96.0:
                continue
            points = np.asarray(curve.points_xy)
            cells = _curve_cells(points, mask.shape)
            if len(cells) != len(points):
                continue
            # A candidate must actually bridge the detected label, not join
            # two fragments along an unrelated edge of its surrounding box.
            inside = region_mask[cells[:, 1], cells[:, 0]]
            if float(np.mean(inside)) < 0.35:
                continue
            # Outside the label all contacts are observed and must belong to
            # the exact endpoint context. A neighbouring continuous stroke
            # remains a barrier even when its tangents resemble the pair.
            allowed = {p for arm in (first, last) for p in arm.context[:4]}
            blocked = False
            for px, py in points[2:-2]:
                if min(np.linalg.norm(np.asarray((px, py)) - first.pixel), np.linalg.norm(np.asarray((px, py)) - last.pixel)) <= 2:
                    continue
                for y in range(max(0, int(math.floor(py - 1))), min(mask.shape[0], int(math.ceil(py + 1)) + 1)):
                    for x in range(max(0, int(math.floor(px - 1))), min(mask.shape[1], int(math.ceil(px + 1)) + 1)):
                        if mask[y, x] and (x, y) not in allowed and math.hypot(x - px, y - py) <= 1.0:
                            blocked = True
            if blocked:
                continue
            # At native resolution EDT radii 1 and sqrt(2) can describe the
            # same thin stroke at different angles. New text proposals must
            # not rank a distant equal-width neighbor above that uncertainty.
            # The hard width gate, uniqueness margin and crossing rejection
            # stay intact; established numeric ranking remains unchanged.
            width_penalty = ratio - 1
            if thin_width_uncertainty and max(first.width, last.width) <= math.sqrt(2) + 1e-6:
                width_penalty = 0.0
            score = width_penalty * 3 + (curve.detour_ratio - 1) + curve.gap_length_pixels / 128
            idx = len(pairs)
            pairs.append((i, j, curve, score))
            votes[i].append((score, idx))
            votes[j].append((score, idx))
    counts = np.zeros(len(pairs), dtype=np.uint8)
    for options in votes:
        options.sort()
        if options and (len(options) == 1 or options[1][0] - options[0][0] > 0.15):
            counts[options[0][1]] += 1
    selected = tuple((arms[i], arms[j], curve) for (i, j, curve, _), count in zip(pairs, counts) if count == 2)
    occupancy, crossing = {}, set()
    for index, (_first, _last, curve) in enumerate(selected):
        cells = set()
        for a, b in zip(curve.points_xy, curve.points_xy[1:]):
            for x in range(int(math.floor(min(a[0], b[0]))), int(math.floor(max(a[0], b[0]))) + 1):
                for y in range(int(math.floor(min(a[1], b[1]))), int(math.floor(max(a[1], b[1]))) + 1):
                    cells.add((x, y))
        for cell in cells:
            if cell in occupancy:
                crossing.update((index, occupancy[cell]))
            else:
                occupancy[cell] = index
    return tuple(value for index, value in enumerate(selected) if index not in crossing), tuple(arms)


def _widths(image, shape):
    if image is None:
        return None
    from scipy.ndimage import distance_transform_edt, maximum_filter
    source = np.asarray(image)
    if (source.shape[:2] != shape or source.dtype != np.uint8 or source.ndim not in (2, 3)
            or (source.ndim == 3 and source.shape[2] != 3)):
        raise ValueError("occlusion image must be native Byte gray/RGB")
    gray = source.astype(np.float64) if source.ndim == 2 else source[:, :, :3].astype(np.float64).mean(axis=2)
    paper = maximum_filter(gray, size=15, mode="nearest")
    contrast = paper - gray
    positive = contrast[contrast > 0]
    if not len(positive):
        return np.zeros(shape, dtype=np.float64)
    threshold = max(6.0, float(np.percentile(positive, 75)) * 0.50)
    return distance_transform_edt(contrast >= threshold)


def build_occluded_connection(graph, centerline, root, origin, regions, *, sparse, dijkstra, image=None, cancel_check=None):
    """Build separate masked-native prefix/curve/suffix routes in the worker.

    Detector masks must be independent of prompts or desired output paths.
    At most sixteen regions and sixteen retained connectors are considered;
    exceeding a budget declines the optional feature without partial results.
    Both anchors must already terminate in the original centreline. Fused
    glyph/contour continuations are deliberately left to explicit review.
    """
    def cancel():
        if cancel_check is not None and cancel_check():
            raise ObservedConnectionCancelled("Occlusion connection was cancelled")
    cancel()
    native = np.asarray(centerline, dtype=bool)
    if native.ndim != 2 or min(native.shape, default=0) < 1:
        raise ValueError("centerline must be a non-empty 2D array")
    if native.size > 1024 * 1024:
        return None
    regions = tuple(regions)
    if not regions or len(regions) > 16:
        return None
    masks = [_region_mask(region, native.shape) for region in regions]
    union = np.logical_or.reduce([value[0] for value in masks])
    cores = [_glyph_core(region, mask) for region, (mask, _box) in zip(regions, masks)]
    core_union = np.logical_or.reduce(cores)
    root_array = np.asarray(root, dtype=float)
    origin_array = np.asarray(origin, dtype=int)
    if root_array.shape != (2,) or origin_array.shape != (2,) or not np.isfinite(root_array).all():
        raise ValueError("root and origin must have two finite coordinates")
    local_root = np.rint(root_array - origin_array).astype(int)
    if not (0 <= local_root[0] < native.shape[1] and 0 <= local_root[1] < native.shape[0]):
        raise ValueError("root must lie in the centerline window")
    if union[local_root[1], local_root[0]]:
        return None
    outside = native & ~core_union
    cancel()
    # A native detector can place the centre of a broad printed stroke just
    # beyond the ordinary two-pixel anchor. Permit one extra pixel only here,
    # with distance priority and a competing-component ambiguity check.
    ys, xs = np.nonzero(outside)
    offsets = np.hypot(xs - (root_array[0] - origin_array[0]), ys - (root_array[1] - origin_array[1]))
    if not len(offsets) or float(offsets.min()) > 3.0:
        return None
    nearest = np.flatnonzero(offsets <= float(offsets.min()) + 0.25)
    from scipy.ndimage import label
    components, _ = label(outside, structure=np.ones((3, 3), dtype=np.uint8))
    if len(set(components[ys[nearest], xs[nearest]].tolist())) != 1:
        return None
    snapped_root = root
    if float(offsets.min()) > 2.0:
        best = int(np.argmin(offsets))
        snapped_root = (int(xs[best] + origin_array[0]), int(ys[best] + origin_array[1]))
    baseline = build_observed_connection(graph, outside, snapped_root, origin, sparse=sparse, dijkstra=dijkstra, cancel_check=cancel_check)
    if baseline is None:
        return None
    width_image = _widths(image, native.shape)
    original_degree = _degree(native)
    routes, retained = [], []
    height, width = native.shape
    for region_index, (region, (region_mask, box)) in enumerate(zip(regions, masks)):
        cancel()
        general_text = getattr(region, "core_mask", None) is not None
        orientation = getattr(region, "baseline_xy", getattr(region, "orientation_xy", (1.0, 0.0)))
        selected, arms = _connectors(outside, region_mask, box, width_image, orientation, cancel,
                                     original_degree=original_degree,
                                     thin_width_uncertainty=general_text)
        for first, last, curve in selected:
            # Automatic reconstruction requires actual observed terminals.
            # A glyph footprint can overlap an ordinary continuous contour;
            # cutting that contour must not manufacture authority to pair it
            # with the next indexed line. Fused/continuous ambiguous entries
            # remain available to the explicit reviewed-gap workflow.
            if (original_degree[first.pixel[1], first.pixel[0]] != 1
                    or original_degree[last.pixel[1], last.pixel[0]] != 1):
                continue
            for start, end, curve_points in ((first, last, curve.points_xy), (last, first, curve.points_xy[::-1])):
                absolute_source = tuple(np.asarray(start.pixel, dtype=float) + origin_array)
                prefix = [tuple(root_array)] if absolute_source == tuple(root_array) else baseline.trace(absolute_source)
                if prefix is None:
                    continue
                points = ((tuple(root_array),) if tuple(prefix[0]) != tuple(root_array) else ()) + tuple(prefix) + tuple(tuple(np.asarray(p) + origin_array) for p in curve_points[1:])
                prefix_length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
                if prefix_length > 2 * max(native.shape):
                    continue
                allowed = outside.copy()
                # An arm other than the chosen far fragment is a barrier to
                # this suffix. It cannot escape through a neighbouring label
                # entry and then claim that neighbour as the continuation.
                for arm in arms:
                    if arm.pixel != end.pixel:
                        allowed[arm.pixel[1], arm.pixel[0]] = False
                indices = np.flatnonzero(allowed.ravel()).astype(np.int32)
                lookup = np.full(native.shape, -1, dtype=np.int32)
                lookup.ravel()[indices] = np.arange(len(indices), dtype=np.int32)
                target_node = int(lookup[end.pixel[1], end.pixel[0]])
                if target_node < 0:
                    continue
                compact = graph[indices][:, indices].tocsr()
                cancel()
                distances, predecessors = dijkstra(compact, directed=True, indices=target_node, return_predecessors=True)
                cancel()
                for value in (indices, lookup, distances, predecessors):
                    value.setflags(write=False)
                routes.append(_Route(
                    indices, lookup, predecessors, distances, points, prefix_length, target_node,
                    region_index, general_text, len(points) - len(curve_points) + 1,
                ))
                retained.append((start.pixel, end.pixel, curve_points, region_index))
                if len(routes) > 16:
                    return None
    if not routes:
        return None
    union.setflags(write=False)
    outside.setflags(write=False)
    outside_degree = _degree(outside)
    outside_degree.setflags(write=False)
    text_cores = []
    for region, core in zip(regions, cores):
        core.setflags(write=False)
        text_cores.append(core if getattr(region, "core_mask", None) is not None else None)
    return OccludedConnectionTree(
        tuple(root_array), tuple(origin_array), tuple(native.shape), union,
        outside, tuple(routes), tuple(retained), baseline, outside_degree, tuple(text_cores),
    )
