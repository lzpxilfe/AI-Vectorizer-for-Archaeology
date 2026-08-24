"""Fast, direction-aware Live-Wire tracing for scanned line maps.

The expensive shortest-path tree is built once for an accepted anchor.  A
mouse-move then only traces predecessor indices, which keeps the green QGIS
preview responsive.  This module deliberately has no QGIS imports so its
geometry and performance can be tested outside QGIS.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np


Pixel = Tuple[int, int]
FloatPixel = Tuple[float, float]
MAX_LIVEWIRE_WINDOW_SIZE = 1_024


class LiveWireUnavailable(RuntimeError):
    """Raised when the optional SciPy runtime is not available."""


class LiveWireCancelled(RuntimeError):
    """Raised when a background tree build was cancelled."""


@dataclass(frozen=True)
class LiveWireConfig:
    """Tuning parameters for a human-led contour Live-Wire tree."""

    max_window_size: int = 384
    forward_window_bias: float = 0.18
    edge_threshold: float = 128.0
    edge_sigma: float = 2.4
    local_background_sigma: float = 6.0
    gradient_sigma: float = 0.8
    tensor_sigma: float = 1.4
    line_cost_weight: float = 4.5
    direction_cost_weight: float = 5.0
    heading_cost_weight: float = 3.0
    heading_decay_pixels: float = 22.0
    target_snap_radius: int = 6
    target_snap_penalty: float = 4.0
    max_detour_ratio: float = 3.0
    min_direct_distance_for_detour_check: float = 8.0

    def validate(self) -> "LiveWireConfig":
        if (
            isinstance(self.max_window_size, bool)
            or not isinstance(self.max_window_size, int)
            or not 32 <= self.max_window_size <= MAX_LIVEWIRE_WINDOW_SIZE
        ):
            raise ValueError(
                "max_window_size must be an integer between 32 and "
                f"{MAX_LIVEWIRE_WINDOW_SIZE} pixels"
            )
        if (
            isinstance(self.target_snap_radius, bool)
            or not isinstance(self.target_snap_radius, int)
        ):
            raise ValueError("target_snap_radius must be a non-negative integer")
        finite_values = (
            self.forward_window_bias,
            self.edge_threshold,
            self.edge_sigma,
            self.local_background_sigma,
            self.gradient_sigma,
            self.tensor_sigma,
            self.line_cost_weight,
            self.direction_cost_weight,
            self.heading_cost_weight,
            self.heading_decay_pixels,
            self.target_snap_penalty,
            self.max_detour_ratio,
            self.min_direct_distance_for_detour_check,
        )
        try:
            all_finite = all(
                not isinstance(value, bool) and math.isfinite(value)
                for value in finite_values
            )
        except TypeError:
            all_finite = False
        if not all_finite:
            raise ValueError("Live-Wire tuning values must be finite numbers")
        if not 0.0 <= self.forward_window_bias <= 0.4:
            raise ValueError("forward_window_bias must be between 0 and 0.4")
        if self.edge_threshold < 0:
            raise ValueError("edge_threshold cannot be negative")
        if self.edge_sigma <= 0:
            raise ValueError("edge_sigma must be positive")
        if self.local_background_sigma <= 0:
            raise ValueError("local_background_sigma must be positive")
        if self.gradient_sigma < 0 or self.tensor_sigma <= 0:
            raise ValueError("gradient/tensor sigma values are invalid")
        if min(
            self.line_cost_weight,
            self.direction_cost_weight,
            self.heading_cost_weight,
        ) < 0:
            raise ValueError("cost weights cannot be negative")
        if self.heading_decay_pixels <= 0:
            raise ValueError("heading_decay_pixels must be positive")
        if self.target_snap_radius < 0 or self.target_snap_penalty < 0:
            raise ValueError("target snapping values cannot be negative")
        if self.max_detour_ratio < 1.0:
            raise ValueError("max_detour_ratio must be at least 1")
        if self.min_direct_distance_for_detour_check < 0:
            raise ValueError("min_direct_distance_for_detour_check cannot be negative")
        return self


@dataclass(frozen=True)
class LiveWireTree:
    """One single-source path tree over a cache-local image window."""

    root: Pixel
    origin: Pixel
    shape: Tuple[int, int]
    predecessors: np.ndarray
    distances: np.ndarray
    strength: float
    config: LiveWireConfig

    @property
    def width(self) -> int:
        return int(self.shape[1])

    @property
    def height(self) -> int:
        return int(self.shape[0])

    def contains(self, pixel: Sequence[float]) -> bool:
        x = int(round(float(pixel[0]))) - self.origin[0]
        y = int(round(float(pixel[1]))) - self.origin[1]
        return 0 <= x < self.width and 0 <= y < self.height

    def trace(self, target: Sequence[float]) -> list[FloatPixel]:
        """Return a strength-blended path from the root to ``target``.

        At 0% this is the literal root-to-cursor segment.  At 100% it is the
        complete Live-Wire route, including a small endpoint snap.  Values in
        between blend every route vertex toward that literal segment.
        """

        target_x = float(target[0])
        target_y = float(target[1])
        direct = [
            (float(self.root[0]), float(self.root[1])),
            (target_x, target_y),
        ]
        if self.strength <= 0.0 or not self.contains(target):
            return direct

        end_index = self._select_target_index(target_x, target_y)
        if end_index is None:
            return direct

        root_local_x = self.root[0] - self.origin[0]
        root_local_y = self.root[1] - self.origin[1]
        root_index = root_local_y * self.width + root_local_x
        local_indices = _trace_predecessors(
            self.predecessors,
            root_index,
            end_index,
        )
        if not local_indices:
            return direct

        assisted = [
            (
                float(index % self.width + self.origin[0]),
                float(index // self.width + self.origin[1]),
            )
            for index in local_indices
        ]
        if not _detour_is_reasonable(
            assisted,
            self.root,
            (target_x, target_y),
            self.config,
        ):
            return direct

        return blend_path_with_cursor(
            assisted,
            self.root,
            (target_x, target_y),
            self.strength,
        )

    def _select_target_index(self, target_x: float, target_y: float) -> Optional[int]:
        local_x = int(round(target_x)) - self.origin[0]
        local_y = int(round(target_y)) - self.origin[1]
        radius = int(round(self.config.target_snap_radius * self.strength))
        radius = max(0, radius)

        x_min = max(0, local_x - radius)
        x_max = min(self.width - 1, local_x + radius)
        y_min = max(0, local_y - radius)
        y_max = min(self.height - 1, local_y + radius)
        if x_min > x_max or y_min > y_max:
            return None

        candidate_x, candidate_y = np.meshgrid(
            np.arange(x_min, x_max + 1, dtype=np.int32),
            np.arange(y_min, y_max + 1, dtype=np.int32),
        )
        offsets = np.hypot(
            candidate_x + self.origin[0] - target_x,
            candidate_y + self.origin[1] - target_y,
        )
        within_radius = offsets <= radius + 0.25 if radius else offsets <= 0.75
        indices = candidate_y * self.width + candidate_x
        route_cost = self.distances[indices]
        valid = within_radius & np.isfinite(route_cost)
        if not np.any(valid):
            return None

        # The endpoint penalty keeps the cursor authoritative.  A nearby line
        # wins only when its route is sufficiently better than the exact
        # cursor pixel; this avoids visible endpoint jumps.
        score = route_cost + offsets * self.config.target_snap_penalty
        score = np.where(valid, score, np.inf)
        flat_best = int(np.argmin(score))
        return int(indices.ravel()[flat_best])


def _import_livewire_runtime():
    from scipy import ndimage, sparse
    from scipy.sparse.csgraph import dijkstra

    return ndimage, sparse, dijkstra


@lru_cache(maxsize=1)
def _get_livewire_runtime():
    """Import the complete optional runtime once, including permanent errors."""
    try:
        return _import_livewire_runtime(), None
    except Exception as exc:  # pragma: no cover - exact failure is environment-specific
        return None, f"{type(exc).__name__}: {exc}"


def is_livewire_available() -> bool:
    """Return whether all required SciPy graph operations can be imported."""

    runtime, _error = _get_livewire_runtime()
    return runtime is not None


def build_livewire_tree(
    image: np.ndarray,
    edges: np.ndarray,
    root: Sequence[int],
    *,
    strength: float = 1.0,
    incoming_direction: Optional[Sequence[float]] = None,
    config: LiveWireConfig = LiveWireConfig(),
    cancel_check: Optional[Callable[[], bool]] = None,
) -> LiveWireTree:
    """Build one direction-aware shortest-path tree around ``root``.

    The graph is intentionally bounded to a square around the user's latest
    checkpoint.  This keeps latency predictable and prevents the suggestion
    from roaming across an unrelated part of a dense historical map.
    """

    config = config.validate()
    strength = _clamp_strength(strength)
    edge_array = np.asarray(edges)
    if edge_array.ndim != 2:
        raise ValueError("edges must be a 2D array")
    if min(edge_array.shape) < 2:
        raise ValueError("edges must be at least 2x2")
    if max(edge_array.shape) > MAX_LIVEWIRE_WINDOW_SIZE:
        raise ValueError(
            "edge image dimensions must not exceed "
            f"{MAX_LIVEWIRE_WINDOW_SIZE}x{MAX_LIVEWIRE_WINDOW_SIZE}"
        )

    root_x = int(round(float(root[0])))
    root_y = int(round(float(root[1])))
    height, width = edge_array.shape
    if not (0 <= root_x < width and 0 <= root_y < height):
        raise ValueError("root is outside the edge image")

    runtime, import_error = _get_livewire_runtime()
    if runtime is None:
        raise LiveWireUnavailable(
            "Direction-aware Live-Wire requires a working SciPy runtime; "
            f"local snapping remains available. ({import_error})"
        )

    _raise_if_cancelled(cancel_check)
    ndimage, sparse, dijkstra = runtime

    normalized_incoming = _normalize_direction(incoming_direction)
    if normalized_incoming is None:
        window_center_x = root_x
        window_center_y = root_y
    else:
        forward_bias = config.max_window_size * config.forward_window_bias
        window_center_x = root_x + normalized_incoming[0] * forward_bias
        window_center_y = root_y + normalized_incoming[1] * forward_bias
    x0, x1 = _bounded_window(window_center_x, width, config.max_window_size)
    y0, y1 = _bounded_window(window_center_y, height, config.max_window_size)
    crop_edges = np.ascontiguousarray(edge_array[y0:y1, x0:x1])
    crop_gray = _to_grayscale(image, edge_array.shape)[y0:y1, x0:x1]
    local_root = (root_x - x0, root_y - y0)

    _raise_if_cancelled(cancel_check)
    line_confidence, tangent_x, tangent_y, coherence = _line_features(
        crop_gray,
        crop_edges,
        ndimage,
        config,
    )
    _raise_if_cancelled(cancel_check)

    graph = _build_sparse_graph(
        sparse,
        line_confidence,
        tangent_x,
        tangent_y,
        coherence,
        local_root,
        1.0 if strength > 0.0 else 0.0,
        normalized_incoming,
        config,
        cancel_check,
    )
    _raise_if_cancelled(cancel_check)

    crop_height, crop_width = crop_edges.shape
    root_index = local_root[1] * crop_width + local_root[0]
    distances, predecessors = dijkstra(
        graph,
        directed=True,
        indices=root_index,
        return_predecessors=True,
    )
    _raise_if_cancelled(cancel_check)
    return LiveWireTree(
        root=(root_x, root_y),
        origin=(x0, y0),
        shape=(crop_height, crop_width),
        predecessors=np.asarray(predecessors, dtype=np.int32),
        distances=np.asarray(distances, dtype=np.float64),
        strength=strength,
        config=config,
    )


def blend_path_with_cursor(
    assisted_path: Iterable[Sequence[float]],
    root: Sequence[float],
    cursor_target: Sequence[float],
    strength: float,
) -> list[FloatPixel]:
    """Blend a routed polyline with the literal root-to-cursor segment."""

    alpha = _clamp_strength(strength)
    start = (float(root[0]), float(root[1]))
    target = (float(cursor_target[0]), float(cursor_target[1]))
    points = _deduplicate_points(assisted_path)
    if alpha <= 0.0 or len(points) < 2:
        return [start, target]

    points[0] = start
    deltas = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    segment_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return [start, target]
    fractions = cumulative / total

    blended = []
    for fraction, point in zip(fractions, points):
        direct_x = start[0] + (target[0] - start[0]) * float(fraction)
        direct_y = start[1] + (target[1] - start[1]) * float(fraction)
        blended.append(
            (
                direct_x * (1.0 - alpha) + point[0] * alpha,
                direct_y * (1.0 - alpha) + point[1] * alpha,
            )
        )
    blended[0] = start
    return _deduplicate_points(blended)


def _to_grayscale(image: np.ndarray, expected_shape: Tuple[int, int]) -> np.ndarray:
    values = np.asarray(image)
    if (
        values.ndim == 3
        and values.shape[:2] == expected_shape
        and values.shape[2] >= 3
    ):
        rgb = values[..., :3].astype(np.float32, copy=False)
        gray = (
            rgb[..., 0] * np.float32(0.299)
            + rgb[..., 1] * np.float32(0.587)
            + rgb[..., 2] * np.float32(0.114)
        )
    elif values.ndim == 2 and values.shape == expected_shape:
        gray = values.astype(np.float32, copy=False)
    else:
        raise ValueError("image dimensions must match edges")

    finite = np.isfinite(gray)
    if not finite.any():
        return np.zeros(expected_shape, dtype=np.float32)
    low = float(np.nanpercentile(gray[finite], 1.0))
    high = float(np.nanpercentile(gray[finite], 99.0))
    if high <= low + 1e-6:
        return np.zeros(expected_shape, dtype=np.float32)
    normalized = (gray - low) / (high - low)
    # A single corrupt pixel must not poison Gaussian filters across the
    # complete crop.  Bright background is the conservative replacement: it
    # does not invent a dark line for the assisted path to follow.
    normalized = np.where(finite, normalized, 1.0)
    return np.ascontiguousarray(np.clip(normalized, 0.0, 1.0), dtype=np.float32)


def _line_features(gray, edges, ndimage, config):
    edge_mask = np.asarray(edges, dtype=np.float32) > config.edge_threshold
    distance = ndimage.distance_transform_edt(~edge_mask).astype(np.float32)
    edge_confidence = np.exp(-distance / config.edge_sigma).astype(np.float32)

    background = ndimage.gaussian_filter(
        gray,
        sigma=config.local_background_sigma,
        mode="nearest",
    )
    relative_darkness = np.clip(background - gray, 0.0, None)
    positive_darkness = relative_darkness[relative_darkness > 0]
    if positive_darkness.size:
        darkness_scale = max(
            float(np.percentile(positive_darkness, 90.0)),
            1e-4,
        )
        ink_confidence = np.clip(relative_darkness / darkness_scale, 0.0, 1.0)
    else:
        ink_confidence = np.zeros_like(gray, dtype=np.float32)
    line_confidence = np.maximum(edge_confidence, ink_confidence).astype(np.float32)

    smoothed = ndimage.gaussian_filter(
        gray,
        sigma=config.gradient_sigma,
        mode="nearest",
    )
    gradient_x = ndimage.sobel(smoothed, axis=1, mode="nearest").astype(np.float32)
    gradient_y = ndimage.sobel(smoothed, axis=0, mode="nearest").astype(np.float32)
    tensor_xx = ndimage.gaussian_filter(
        gradient_x * gradient_x,
        sigma=config.tensor_sigma,
        mode="nearest",
    )
    tensor_yy = ndimage.gaussian_filter(
        gradient_y * gradient_y,
        sigma=config.tensor_sigma,
        mode="nearest",
    )
    tensor_xy = ndimage.gaussian_filter(
        gradient_x * gradient_y,
        sigma=config.tensor_sigma,
        mode="nearest",
    )
    discriminant = np.sqrt(
        np.maximum((tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy ** 2, 0.0)
    )
    coherence = discriminant / (tensor_xx + tensor_yy + 1e-6)
    coherence = np.clip(coherence * line_confidence, 0.0, 1.0).astype(np.float32)

    gradient_angle = 0.5 * np.arctan2(2.0 * tensor_xy, tensor_xx - tensor_yy)
    tangent_angle = gradient_angle + np.float32(math.pi / 2.0)
    tangent_x = np.cos(tangent_angle).astype(np.float32)
    tangent_y = np.sin(tangent_angle).astype(np.float32)
    return line_confidence, tangent_x, tangent_y, coherence


def _build_sparse_graph(
    sparse,
    line_confidence,
    tangent_x,
    tangent_y,
    coherence,
    local_root,
    strength,
    incoming_direction,
    config,
    cancel_check,
):
    height, width = line_confidence.shape
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    rows = []
    columns = []
    costs = []

    heading = _normalize_direction(incoming_direction)
    if heading is not None:
        grid_y, grid_x = np.ogrid[:height, :width]
        root_distance = np.hypot(
            grid_x - local_root[0],
            grid_y - local_root[1],
        ).astype(np.float32)
        heading_decay = np.exp(
            -root_distance / config.heading_decay_pixels
        ).astype(np.float32)
    else:
        heading_decay = None

    for delta_x, delta_y in (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ):
        _raise_if_cancelled(cancel_check)
        source_y, target_y = _paired_slices(height, delta_y)
        source_x, target_x_slice = _paired_slices(width, delta_x)
        source_indices = indices[source_y, source_x]
        target_indices = indices[target_y, target_x_slice]

        unit_length = math.hypot(delta_x, delta_y)
        unit_x = delta_x / unit_length
        unit_y = delta_y / unit_length
        source_line = line_confidence[source_y, source_x]
        target_line = line_confidence[target_y, target_x_slice]
        mean_line = (source_line + target_line) * 0.5

        source_alignment = np.abs(
            tangent_x[source_y, source_x] * unit_x
            + tangent_y[source_y, source_x] * unit_y
        )
        target_alignment = np.abs(
            tangent_x[target_y, target_x_slice] * unit_x
            + tangent_y[target_y, target_x_slice] * unit_y
        )
        mean_alignment = (source_alignment + target_alignment) * 0.5
        mean_coherence = (
            coherence[source_y, source_x]
            + coherence[target_y, target_x_slice]
        ) * 0.5
        direction_penalty = (1.0 - mean_alignment) ** 2 * mean_coherence

        assist_penalty = (
            config.line_cost_weight * (1.0 - mean_line)
            + config.direction_cost_weight * direction_penalty
        )
        if heading is not None:
            signed_alignment = max(-1.0, min(1.0, unit_x * heading[0] + unit_y * heading[1]))
            heading_penalty = (1.0 - signed_alignment) ** 2
            assist_penalty = assist_penalty + (
                config.heading_cost_weight
                * heading_penalty
                * heading_decay[source_y, source_x]
            )

        move_cost = unit_length * (1.0 + strength * assist_penalty)
        rows.append(source_indices.ravel())
        columns.append(target_indices.ravel())
        costs.append(np.asarray(move_cost, dtype=np.float32).ravel())

    row_array = np.concatenate(rows)
    column_array = np.concatenate(columns)
    cost_array = np.concatenate(costs)
    return sparse.csr_matrix(
        (cost_array, (row_array, column_array)),
        shape=(height * width, height * width),
        dtype=np.float32,
    )


def _paired_slices(length: int, delta: int):
    if delta < 0:
        return slice(-delta, length), slice(0, length + delta)
    if delta > 0:
        return slice(0, length - delta), slice(delta, length)
    return slice(0, length), slice(0, length)


def _trace_predecessors(predecessors, root_index: int, target_index: int) -> list[int]:
    if target_index == root_index:
        return [root_index]
    result = [int(target_index)]
    current = int(target_index)
    maximum_steps = int(np.asarray(predecessors).size) + 1
    for _ in range(maximum_steps):
        previous = int(predecessors[current])
        if previous < 0 or previous >= maximum_steps - 1:
            return []
        result.append(previous)
        if previous == root_index:
            result.reverse()
            return result
        current = previous
    return []


def _detour_is_reasonable(path, root, target, config):
    direct_distance = math.hypot(target[0] - root[0], target[1] - root[1])
    if direct_distance < config.min_direct_distance_for_detour_check:
        return True
    route_length = sum(
        math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
        for point_a, point_b in zip(path, path[1:])
    )
    # Every non-zero slider value blends against the same full-assist route,
    # so the slider is a literal geometric continuum rather than a second,
    # hidden change to graph semantics. The route itself still has a strict
    # roaming bound before any blend is applied.
    return route_length <= direct_distance * config.max_detour_ratio


def _bounded_window(center: float, length: int, maximum: int) -> Tuple[int, int]:
    window = min(int(maximum), int(length))
    start = int(round(float(center))) - window // 2
    start = max(0, min(start, length - window))
    return start, start + window


def _normalize_direction(direction):
    if direction is None:
        return None
    x = float(direction[0])
    y = float(direction[1])
    norm = math.hypot(x, y)
    if norm <= 1e-6:
        return None
    return x / norm, y / norm


def _clamp_strength(strength: float) -> float:
    try:
        value = float(strength)
    except (TypeError, ValueError) as exc:
        raise ValueError("strength must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError("strength must be a finite number")
    return max(0.0, min(1.0, value))


def _deduplicate_points(points: Iterable[Sequence[float]]) -> list[FloatPixel]:
    result = []
    for point in points:
        value = (float(point[0]), float(point[1]))
        if not result or value != result[-1]:
            result.append(value)
    return result


def _raise_if_cancelled(cancel_check):
    if cancel_check is not None and cancel_check():
        raise LiveWireCancelled("Live-Wire tree build was cancelled")


__all__ = [
    "LiveWireCancelled",
    "LiveWireConfig",
    "LiveWireTree",
    "LiveWireUnavailable",
    "blend_path_with_cursor",
    "build_livewire_tree",
    "is_livewire_available",
]
