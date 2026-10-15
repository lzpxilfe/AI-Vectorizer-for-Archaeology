"""One bounded inferred gap between otherwise native contour fragments.

The second virtual source starts on the far side of a connector. Its cost
already includes the native root-to-near-end route, so no downstream edge can
cross a second gap. Curve geometry is retained separately from observed ink.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .observed_connection import ObservedConnectionCancelled, ObservedConnectionTree


@dataclass(frozen=True)
class LabelGapConnectionTree:
    native: ObservedConnectionTree
    predecessors: np.ndarray
    distances: np.ndarray
    prefixes: Mapping

    def trace(self, target):
        """Return native strokes plus one retained curve, never two gaps."""
        point = np.asarray(target, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            return None
        height, width = self.native.shape
        local_x, local_y = point - np.asarray(self.native.origin)
        if not (0 <= local_x < width and 0 <= local_y < height):
            return None
        chord = float(np.linalg.norm(point - self.native.root))
        if chord <= 1e-9:
            return None
        yy, xx = np.mgrid[
            max(0, int(math.floor(local_y - 2))):min(height, int(math.ceil(local_y + 2)) + 1),
            max(0, int(math.floor(local_x - 2))):min(width, int(math.ceil(local_x + 2)) + 1),
        ]
        offsets = np.hypot(xx - local_x, yy - local_y)
        nodes = self.native.node_lookup[yy, xx]
        valid = (nodes >= 0) & (offsets <= 2.0 + 1e-6)
        if not np.any(valid):
            return None
        valid &= offsets <= float(offsets[valid].min()) + 0.25
        candidates = nodes[valid]
        costs = self.distances[candidates] + offsets[valid] * 4.0
        best = int(np.argmin(costs))
        if not math.isfinite(float(costs[best])):
            return None
        node = int(candidates[best])
        if math.isfinite(float(self.native.distances[node])):
            return None
        limit = min(2 * max(self.native.shape), max(8.0, chord * 6.0))
        length = float(offsets[valid][best])
        backwards = []
        previous = point
        count = len(self.native.native_indices)
        for _ in range(min(count + 1, int(math.ceil(limit)) + 2)):
            pixel = int(self.native.native_indices[node])
            current = (float(pixel % width + self.native.origin[0]),
                       float(pixel // width + self.native.origin[1]))
            if backwards:
                length += math.hypot(current[0] - previous[0], current[1] - previous[1])
            if length > limit:
                return None
            backwards.append(current)
            previous = current
            parent = int(self.predecessors[node])
            if parent == count:
                prefix = self.prefixes.get(node)
                if prefix is None or length + prefix[1] > limit:
                    return None
                points = [*prefix[0], *backwards[-2::-1], tuple(point)]
                return [p for i, p in enumerate(points) if not i or p != points[i - 1]]
            if parent < 0 or parent >= count:
                return None
            node = parent
        return None


def build_label_gap_connection(graph, native, connectors, *, sparse, dijkstra, cancel_check=None):
    """Build a second tree with exactly one connector in any accepted route."""
    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise ObservedConnectionCancelled("Label-gap connection was cancelled")

    check_cancelled()
    count = len(native.native_indices)
    height, width = native.shape
    origin = np.asarray(native.origin)
    seeds = {}
    for connector in connectors:
        check_cancelled()
        sx, sy = connector.source_xy
        tx, ty = connector.target_xy
        source = int(native.node_lookup[sy, sx])
        target = int(native.node_lookup[ty, tx])
        if source < 0 or target < 0:
            continue
        source_reached = math.isfinite(float(native.distances[source]))
        target_reached = math.isfinite(float(native.distances[target]))
        if source_reached == target_reached:
            continue
        curve = connector.points_xy
        if target_reached:
            source, target = target, source
            curve = curve[::-1]
        source_xy = tuple(np.asarray(curve[0], dtype=float) + origin)
        prefix = ([native.root] if source_xy == native.root else native.trace(source_xy))
        if prefix is None:
            continue
        points = tuple(prefix) + tuple(tuple(np.asarray(p, dtype=float) + origin) for p in curve[1:])
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
        if length > 2 * max(native.shape):
            continue
        cost = float(native.distances[source]) + 4.0 + 2.0 * connector.path_length_pixels
        if target not in seeds or cost < seeds[target][0]:
            seeds[target] = (cost, points, length)
    if not seeds:
        return None
    check_cancelled()
    selected = graph[native.native_indices][:, native.native_indices].tocoo()
    targets = np.asarray(list(seeds), dtype=np.int32)
    compact = sparse.csr_matrix(
        (np.r_[selected.data, [seeds[int(n)][0] for n in targets]],
         (np.r_[selected.row, np.full(len(targets), count, dtype=np.int32)],
          np.r_[selected.col, targets])),
        shape=(count + 1, count + 1),
    )
    check_cancelled()
    distances, predecessors = dijkstra(compact, directed=True, indices=count, return_predecessors=True)
    check_cancelled()
    predecessors = np.asarray(predecessors, dtype=np.int32)
    distances = np.asarray(distances, dtype=np.float64)
    predecessors.setflags(write=False)
    distances.setflags(write=False)
    return LabelGapConnectionTree(
        native, predecessors, distances,
        MappingProxyType({node: (value[1], value[2]) for node, value in seeds.items()}),
    )
