"""Bounded routes through the native observed centreline only.

The optional compact tree samples the existing Live-Wire graph instead of
inventing links between nearby strokes. Its purpose is to keep a continuous
observed bend when the pixel graph prefers a shorter cross-contour route.
It cannot establish contour identity at an actual junction or fill a gap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence

import numpy as np


class ObservedConnectionCancelled(RuntimeError):
    """A cancelled compact-tree build must not publish a partial result."""


@dataclass(frozen=True)
class ObservedConnectionTree:
    """Immutable predecessor data; target queries inspect a small pixel box."""

    root: tuple[float, float]
    origin: tuple[int, int]
    shape: tuple[int, int]
    native_indices: np.ndarray
    predecessors: np.ndarray
    distances: np.ndarray
    node_lookup: np.ndarray
    gap_outward: Optional[np.ndarray] = None

    def preserves_collinear_gap(self, path, *, max_gap_pixels=96.0) -> bool:
        """Protect an existing short, tangential pixel-route gap crossing.

        This only recognizes geometry already present in the pixel route.
        It neither adds a connector nor chooses a new continuation beneath
        a label. Non-endpoint cross-stroke shortcuts do not qualify.
        """

        if self.gap_outward is None or len(path) < 3:
            return False
        points = np.asarray(path, dtype=np.float64) - np.asarray(self.origin)
        pixels = np.rint(points).astype(np.int64)
        height, width = self.shape
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        nodes = np.full(len(points), -1, dtype=np.int32)
        nodes[valid] = self.node_lookup[pixels[valid, 1], pixels[valid, 0]]
        present = np.flatnonzero(nodes >= 0)
        endpoint_positions = present[
            np.any(self.gap_outward[nodes[present]], axis=1)
        ].tolist()
        maximum_steps = int(math.ceil(min(max_gap_pixels, 2 * max(self.shape)) * 2)) + 2
        for first_index, start in enumerate(endpoint_positions):
            for end in endpoint_positions[first_index + 1:]:
                if end - start > maximum_steps:
                    break
                if end - start < 2 or np.all(nodes[start:end + 1] >= 0):
                    continue
                delta = points[end] - points[start]
                distance = float(np.hypot(delta[0], delta[1]))
                if not 2.0 <= distance <= max_gap_pixels:
                    continue
                direction = delta / distance
                if (
                    float(np.dot(self.gap_outward[nodes[start]], direction)) < 0.95
                    or float(np.dot(self.gap_outward[nodes[end]], -direction)) < 0.95
                ):
                    continue
                section = points[start:end + 1] - points[start]
                along = section @ direction
                normal = np.abs(section[:, 0] * direction[1] - section[:, 1] * direction[0])
                if np.all(normal <= 1.0) and np.all((along >= -0.5) & (along <= distance + 0.5)):
                    return True
        return False

    def trace(self, target: Sequence[float]):
        """Return an unblended observed route, or ``None`` for pixel fallback."""

        point = np.asarray(target, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            return None
        local_x, local_y = point - np.asarray(self.origin)
        height, width = self.shape
        if not (0 <= local_x < width and 0 <= local_y < height):
            return None
        if np.linalg.norm(point - self.root) <= 1e-9:
            return None

        radius = 2.0
        yy, xx = np.mgrid[
            max(0, int(math.floor(local_y - radius))):
            min(height, int(math.ceil(local_y + radius)) + 1),
            max(0, int(math.floor(local_x - radius))):
            min(width, int(math.ceil(local_x + radius)) + 1),
        ]
        offsets = np.hypot(xx - local_x, yy - local_y)
        nodes = self.node_lookup[yy, xx]
        valid = (nodes >= 0) & (offsets <= radius + 1e-6)
        if not np.any(valid):
            return None
        # A cheaper neighbouring stroke must not win by moving a precise
        # click to a more distant pixel inside the anchor search radius.
        valid &= offsets <= float(offsets[valid].min()) + 0.25
        candidates = nodes[valid]
        costs = self.distances[candidates] + offsets[valid] * 4.0
        best = int(np.argmin(costs))
        if not math.isfinite(float(costs[best])):
            return None
        node = int(candidates[best])

        count = len(self.native_indices)
        backwards = []
        chord = float(np.linalg.norm(point - self.root))
        length_limit = min(2 * max(self.shape), max(8.0, chord * 6.0))
        length = float(offsets[valid][best])
        previous = point
        for _ in range(count + 1):
            if node == count:  # Virtual source, connected only to root anchors.
                length += math.hypot(
                    previous[0] - self.root[0], previous[1] - self.root[1]
                )
                if length > length_limit:
                    return None
                break
            native = int(self.native_indices[node])
            current = (
                float(native % width + self.origin[0]),
                float(native // width + self.origin[1]),
            )
            # Stop while walking. A giant connected network must not allocate
            # and traverse its entire route on every mouse movement merely
            # to discover afterwards that it exceeds the bounded allowance.
            if backwards:
                length += math.hypot(current[0] - previous[0], current[1] - previous[1])
            if length > length_limit:
                return None
            backwards.append(current)
            previous = current
            node = int(self.predecessors[node])
            if node < 0:
                return None
        else:
            return None

        points = np.asarray(
            [self.root] + backwards[::-1] + [tuple(point)], dtype=np.float64
        )
        points = points[np.r_[True, np.any(np.diff(points, axis=0), axis=1)]]
        if len(points) < 2:
            return None
        return [(float(x), float(y)) for x, y in points]


def build_observed_connection(
    graph,
    centerline: np.ndarray,
    root: Sequence[float],
    origin: Sequence[int],
    *,
    sparse,
    dijkstra,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[ObservedConnectionTree]:
    """Sample existing graph edges at native skeleton nodes, in the worker.

    ``sparse`` and ``dijkstra`` come from Live-Wire's already checked optional
    SciPy runtime. No detector, model, image filtering or imports run on hover.
    """

    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise ObservedConnectionCancelled("Observed connection was cancelled")

    check_cancelled()
    mask = np.asarray(centerline, dtype=bool)
    if mask.ndim != 2 or min(mask.shape, default=0) < 1:
        raise ValueError("centerline must be a non-empty 2D array")
    height, width = mask.shape
    if graph.shape != (height * width, height * width):
        raise ValueError("graph dimensions must match centerline")
    root_array = np.asarray(root, dtype=np.float64)
    origin_array = np.asarray(origin, dtype=np.int64)
    if root_array.shape != (2,) or origin_array.shape != (2,):
        raise ValueError("root and origin must contain two coordinates")
    local_root = root_array - origin_array
    if not np.isfinite(root_array).all() or not (
        0 <= local_root[0] < width and 0 <= local_root[1] < height
    ):
        raise ValueError("root must be inside the centreline window")
    native = np.flatnonzero(mask.ravel()).astype(np.int32)
    count = len(native)
    if not count:
        return None
    offsets = np.hypot(native % width - local_root[0], native // width - local_root[1])
    anchors = np.flatnonzero(
        (offsets <= 2.0 + 1e-6) & (offsets <= float(offsets.min()) + 0.25)
    )
    if not len(anchors):
        return None
    lookup = np.full(height * width, -1, dtype=np.int32)
    lookup[native] = np.arange(count, dtype=np.int32)
    lookup = lookup.reshape(mask.shape)

    # Sampling preserves finite grid costs and incoming heading. No dilation,
    # closing, distance transform or added inter-stroke edge is permitted.
    selected = graph[native][:, native].tocsr()
    gap_outward = _endpoint_outward(native, selected, width)
    selected = selected.tocoo()
    check_cancelled()
    rows = np.r_[selected.row, np.full(len(anchors), count, dtype=np.int32)]
    columns = np.r_[selected.col, anchors]
    values = np.r_[selected.data, offsets[anchors] * 4.0]
    compact = sparse.csr_matrix(
        (values, (rows, columns)), shape=(count + 1, count + 1)
    )
    check_cancelled()
    distances, predecessors = dijkstra(
        compact, directed=True, indices=count, return_predecessors=True
    )
    check_cancelled()
    arrays = (
        native,
        np.asarray(predecessors, dtype=np.int32),
        np.asarray(distances, dtype=np.float64),
        lookup,
    )
    for array in arrays:
        array.setflags(write=False)
    if gap_outward is not None:
        gap_outward.setflags(write=False)
    return ObservedConnectionTree(
        tuple(float(value) for value in root_array),
        tuple(int(value) for value in origin_array),
        (height, width),
        *arrays,
        gap_outward=gap_outward,
    )


def _endpoint_outward(native, graph, width):
    """Estimate only unbranched, nearly straight degree-1 endpoint tangents."""

    degree = np.diff(graph.indptr)
    endpoints = np.flatnonzero(degree == 1)
    if not len(endpoints):
        return None
    current = endpoints.copy()
    previous = np.full(len(endpoints), -1, dtype=np.int32)
    travelled = np.zeros(len(endpoints), dtype=np.float64)
    moving = np.ones(len(endpoints), dtype=bool)
    for step in range(6):
        moving &= degree[current] == (1 if step == 0 else 2)
        active = np.flatnonzero(moving)
        if not len(active):
            break
        first_slot = graph.indptr[current[active]]
        first = graph.indices[first_slot]
        if step == 0:
            following = first
        else:
            second = graph.indices[first_slot + 1]
            following = np.where(first == previous[active], second, first)
        before = native[current[active]]
        after = native[following]
        travelled[active] += np.hypot(after % width - before % width,
                                       after // width - before // width)
        previous[active] = current[active]
        current[active] = following
    start = native[endpoints]
    finish = native[current]
    dx = start % width - finish % width
    dy = start // width - finish // width
    chord = np.hypot(dx, dy)
    usable = (travelled >= 3.0) & (chord >= 2.75) & (chord >= travelled * 0.95)
    if not np.any(usable):
        return None
    outward = np.zeros((len(native), 2), dtype=np.float32)
    outward[endpoints[usable], 0] = dx[usable] / chord[usable]
    outward[endpoints[usable], 1] = dy[usable] / chord[usable]
    return outward
