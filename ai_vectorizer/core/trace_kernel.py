# -*- coding: utf-8 -*-
"""QGIS-independent tracing primitives used by UI and benchmark workers.

The product's interactive tracer operates in ``(x, y)`` pixel coordinates.
This module deliberately has no NumPy, OpenCV, or QGIS import so the exact A*
policy can be exercised in a small worker process and in ordinary unit tests.
Array-like objects with a two-dimensional ``shape`` (including NumPy arrays)
and rectangular Python sequences are accepted as cost maps.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import struct
from typing import Any, Iterable, Sequence, Tuple


Pixel = Tuple[int, int]
Point = Tuple[float, float]


PATH_MOVE_COST_STRAIGHT = 1.0
PATH_MOVE_COST_DIAGONAL = 1.41421356237
PATH_MAX_ITER_BASE = 100_000
PATH_MAX_ITER_DISTANCE_FACTOR = 500
PATH_SMOOTH_WINDOW_SIZE = 5
CHAIKIN_ITERATIONS = 3
CHAIKIN_Q_WEIGHT = 0.75
CHAIKIN_R_WEIGHT = 0.25
A_STAR_NEIGHBORS: tuple[Pixel, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


class TraceInputError(ValueError):
    """Raised when a trace request cannot be evaluated safely."""


@dataclass(frozen=True)
class TraceConfig:
    """Deterministic product A* policy and resource limits."""

    straight_move_cost: float = PATH_MOVE_COST_STRAIGHT
    diagonal_move_cost: float = PATH_MOVE_COST_DIAGONAL
    max_iterations_base: int = PATH_MAX_ITER_BASE
    max_iterations_distance_factor: int = PATH_MAX_ITER_DISTANCE_FACTOR
    minimum_pixel_cost: float = 1.0
    max_width: int = 1_024
    max_height: int = 1_024
    max_cells: int = 1_048_576
    validate_all_costs: bool = True
    validate_accessed_costs: bool = True
    neighbors: tuple[Pixel, ...] = A_STAR_NEIGHBORS

    def __post_init__(self) -> None:
        positive_numbers = (
            self.straight_move_cost,
            self.diagonal_move_cost,
            self.minimum_pixel_cost,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive_numbers):
            raise ValueError("move costs and minimum_pixel_cost must be finite and positive")
        if self.minimum_pixel_cost * self.straight_move_cost < 1.0:
            raise ValueError(
                "minimum_pixel_cost * straight_move_cost must be at least 1 "
                "for the product heuristic to remain admissible"
            )
        if (
            self.minimum_pixel_cost * self.diagonal_move_cost
            < PATH_MOVE_COST_DIAGONAL
        ):
            raise ValueError(
                "minimum_pixel_cost * diagonal_move_cost must be at least the "
                "product diagonal cost for the heuristic to remain admissible"
            )
        if self.max_iterations_base < 0 or self.max_iterations_distance_factor < 0:
            raise ValueError("iteration limits must be non-negative")
        if self.max_width < 1 or self.max_height < 1 or self.max_cells < 1:
            raise ValueError("cost-map resource limits must be positive")
        if not isinstance(self.validate_all_costs, bool) or not isinstance(
            self.validate_accessed_costs,
            bool,
        ):
            raise ValueError("cost validation flags must be bool")
        if tuple(self.neighbors) != A_STAR_NEIGHBORS:
            raise ValueError("neighbors must use the fixed product 8-neighbor policy")


@dataclass(frozen=True)
class TraceResult:
    """A trace result whose ``path`` excludes the clamped start pixel."""

    path: tuple[Pixel, ...]
    start: Pixel
    target: Pixel
    endpoint: Pixel
    reached_target: bool
    used_partial: bool
    iterations: int
    total_cost: float | None
    status: str
    limit_hit: bool

    @property
    def points_xy(self) -> tuple[Pixel, ...]:
        """Raw ordered A* points, including the start and reached endpoint."""

        return (self.start, *self.path)


DEFAULT_TRACE_CONFIG = TraceConfig()


def _numeric_coordinate(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise TraceInputError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TraceInputError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise TraceInputError(f"{name} must be a finite number")
    return numeric


def _numeric_point(point: Sequence[Any], *, name: str) -> Point:
    if isinstance(point, (str, bytes)) or len(point) != 2:
        raise TraceInputError(f"{name} must contain exactly two coordinates")
    return (
        _numeric_coordinate(point[0], name=f"{name}.x"),
        _numeric_coordinate(point[1], name=f"{name}.y"),
    )


def quantize_pixel_point(
    point: Sequence[Any],
    *,
    mode: str = "truncate",
    name: str = "point",
) -> Pixel:
    """Quantize a pixel point with an explicit product boundary policy.

    ``truncate`` matches ``SmartTraceTool.map_to_pixel`` (the end-to-end UI
    path). ``round`` matches its lower-level A* compatibility wrapper.
    """

    numeric = _numeric_point(point, name=name)
    if mode == "truncate":
        return int(numeric[0]), int(numeric[1])
    if mode == "round":
        return int(round(numeric[0])), int(round(numeric[1]))
    if mode == "reject_noninteger":
        if not numeric[0].is_integer() or not numeric[1].is_integer():
            raise TraceInputError(f"{name} must use integer pixel coordinates")
        return int(numeric[0]), int(numeric[1])
    raise ValueError("mode must be 'truncate', 'round', or 'reject_noninteger'")


def _cost_map_dimensions(cost_map: Any) -> tuple[int, int]:
    shape = getattr(cost_map, "shape", None)
    if shape is not None:
        try:
            if len(shape) != 2:
                raise TraceInputError("cost_map must be two-dimensional")
            height, width = int(shape[0]), int(shape[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise TraceInputError("cost_map has an invalid shape") from exc
        return height, width

    if isinstance(cost_map, (str, bytes)):
        raise TraceInputError("cost_map must be a rectangular two-dimensional sequence")
    try:
        height = len(cost_map)
        width = len(cost_map[0]) if height else 0
    except (TypeError, IndexError, KeyError) as exc:
        raise TraceInputError("cost_map must be a rectangular two-dimensional sequence") from exc
    return height, width


def _validate_cost_map(cost_map: Any, config: TraceConfig) -> tuple[int, int]:
    height, width = _cost_map_dimensions(cost_map)
    if height < 1 or width < 1:
        raise TraceInputError("cost_map must not be empty")
    if width > config.max_width or height > config.max_height:
        raise TraceInputError(
            f"cost_map dimensions {width}x{height} exceed the "
            f"{config.max_width}x{config.max_height} limit"
        )
    if width * height > config.max_cells:
        raise TraceInputError(
            f"cost_map contains {width * height} cells; limit is {config.max_cells}"
        )

    for y in range(height):
        try:
            row = cost_map[y]
            if len(row) != width:
                raise TraceInputError("cost_map rows must all have the same width")
        except TypeError:
            # Some ndarray-like row proxies omit len but still support indexing.
            row = cost_map[y]
        except (IndexError, KeyError) as exc:
            raise TraceInputError("cost_map rows must all be readable") from exc

        if not config.validate_all_costs:
            continue
        for x in range(width):
            try:
                raw_cost = row[x]
                if isinstance(raw_cost, bool):
                    raise TypeError
                pixel_cost = float(raw_cost)
            except (TypeError, ValueError, IndexError, KeyError) as exc:
                raise TraceInputError(f"cost_map[{y}][{x}] is not numeric") from exc
            if not math.isfinite(pixel_cost) or pixel_cost < config.minimum_pixel_cost:
                raise TraceInputError(
                    f"cost_map[{y}][{x}] must be finite and at least "
                    f"{config.minimum_pixel_cost}"
                )
    return height, width


def _read_pixel_cost(cost_map: Any, x: int, y: int, config: TraceConfig) -> float:
    try:
        raw_cost = cost_map[y][x]
        if isinstance(raw_cost, bool):
            raise TypeError
        pixel_cost = float(raw_cost)
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise TraceInputError(f"cost_map[{y}][{x}] is not numeric") from exc
    if not math.isfinite(pixel_cost) or pixel_cost < config.minimum_pixel_cost:
        raise TraceInputError(
            f"cost_map[{y}][{x}] must be finite and at least {config.minimum_pixel_cost}"
        )
    return pixel_cost


def _clamp(point: Pixel, *, width: int, height: int) -> Pixel:
    return (
        min(max(point[0], 0), width - 1),
        min(max(point[1], 0), height - 1),
    )


def _trace_path(
    cost_map: Any,
    start_xy: Sequence[Any],
    target_xy: Sequence[Any],
    *,
    allow_partial: bool,
    config: TraceConfig,
    bounds_policy: str,
    quantization: str,
) -> TraceResult:
    if not isinstance(config, TraceConfig):
        raise TypeError("config must be a TraceConfig")
    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be bool")
    if bounds_policy not in ("clamp", "reject"):
        raise ValueError("bounds_policy must be 'clamp' or 'reject'")

    height, width = _validate_cost_map(cost_map, config)
    numeric_start = _numeric_point(start_xy, name="start_xy")
    numeric_target = _numeric_point(target_xy, name="target_xy")
    requested_start = quantize_pixel_point(
        numeric_start,
        mode=quantization,
        name="start_xy",
    )
    requested_target = quantize_pixel_point(
        numeric_target,
        mode=quantization,
        name="target_xy",
    )
    if bounds_policy == "reject":
        for label, raw_point in (("start_xy", numeric_start), ("target_xy", numeric_target)):
            if not (0.0 <= raw_point[0] <= width - 1 and 0.0 <= raw_point[1] <= height - 1):
                raise TraceInputError(f"{label} is outside the {width}x{height} cost map")
        start = requested_start
        target = requested_target
    else:
        start = _clamp(requested_start, width=width, height=height)
        target = _clamp(requested_target, width=width, height=height)

    frontier: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    came_from: dict[Pixel, Pixel] = {}
    cost_so_far: dict[Pixel, float] = {start: 0.0}
    max_iterations = max(
        config.max_iterations_base,
        (abs(target[0] - start[0]) + abs(target[1] - start[1]))
        * config.max_iterations_distance_factor,
    )

    found = False
    limit_hit = False
    iterations = 0
    best_node = start
    minimum_remaining_distance = abs(target[0] - start[0]) + abs(target[1] - start[1])

    while frontier:
        priority, current_x, current_y = heapq.heappop(frontier)
        current = (current_x, current_y)

        # A node can be queued more than once before its cheapest route is
        # known.  Expanding an older entry again is both wasted work and, more
        # importantly, used to consume the product iteration budget.  Large
        # weighted edge maps could therefore hit the limit even though the
        # target was reachable within the configured number of real
        # expansions.
        current_cost = cost_so_far[current]
        current_heuristic = math.sqrt(
            (target[0] - current_x) ** 2 + (target[1] - current_y) ** 2
        )
        if current != start and priority > current_cost + current_heuristic + 1e-12:
            continue

        iterations += 1
        if iterations > max_iterations:
            limit_hit = True
            break

        remaining_distance = abs(target[0] - current_x) + abs(target[1] - current_y)
        if remaining_distance < minimum_remaining_distance:
            minimum_remaining_distance = remaining_distance
            best_node = current

        if current == target:
            found = True
            best_node = current
            break

        for dx, dy in config.neighbors:
            next_x = current_x + dx
            next_y = current_y + dy
            if not (0 <= next_x < width and 0 <= next_y < height):
                continue

            movement_cost = (
                config.diagonal_move_cost if dx != 0 and dy != 0 else config.straight_move_cost
            )
            if config.validate_accessed_costs:
                pixel_cost = _read_pixel_cost(cost_map, next_x, next_y, config)
            else:
                # EdgeDetector and SAM produce trusted, float32 costs >= 1.
                # Keep this hot product path equivalent to the original UI.
                pixel_cost = float(cost_map[next_y, next_x])
            next_node = (next_x, next_y)
            new_cost = cost_so_far[current] + pixel_cost * movement_cost
            if not math.isfinite(new_cost):
                raise TraceInputError(
                    "accumulated path cost overflowed; cost_map values are too large"
                )

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                came_from[next_node] = current
                heuristic = math.sqrt((target[0] - next_x) ** 2 + (target[1] - next_y) ** 2)
                heapq.heappush(frontier, (new_cost + heuristic, next_x, next_y))

    if found:
        endpoint = target
        used_partial = False
    elif allow_partial and best_node != start:
        endpoint = best_node
        used_partial = True
    else:
        return TraceResult(
            path=(),
            start=start,
            target=target,
            endpoint=start,
            reached_target=False,
            used_partial=False,
            iterations=iterations,
            total_cost=None,
            status="iteration_limit" if limit_hit else "no_path",
            limit_hit=limit_hit,
        )

    path: list[Pixel] = []
    current = endpoint
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return TraceResult(
        path=tuple(path),
        start=start,
        target=target,
        endpoint=endpoint,
        reached_target=found,
        used_partial=used_partial,
        iterations=iterations,
        total_cost=cost_so_far[endpoint],
        status=("complete" if found else "partial_iteration_limit"),
        limit_hit=limit_hit,
    )


def find_path(
    cost_map: Any,
    start_xy: Sequence[Any],
    target_xy: Sequence[Any],
    *,
    allow_partial: bool = True,
    config: TraceConfig = DEFAULT_TRACE_CONFIG,
) -> TraceResult:
    """Compatibility entry point for ``SmartTraceTool._run_a_star_path``.

    Coordinates are rounded then clamped, and ``result.path`` excludes the
    start exactly as the historical lower-level UI method did.
    """

    return _trace_path(
        cost_map,
        start_xy,
        target_xy,
        allow_partial=allow_partial,
        config=config,
        bounds_policy="clamp",
        quantization="round",
    )


def trace_path(
    cost_map: Any,
    start_xy: Sequence[Any],
    target_xy: Sequence[Any],
    *,
    allow_partial: bool = False,
    config: TraceConfig = DEFAULT_TRACE_CONFIG,
    quantization: str = "truncate",
) -> TraceResult:
    """Strict worker entry point for the product's A* policy.

    Out-of-bounds inputs are rejected instead of silently clamped. The default
    truncation policy matches the UI's map-to-pixel conversion; benchmark
    manifests preserve the original floating-point prompts separately.
    """

    return _trace_path(
        cost_map,
        start_xy,
        target_xy,
        allow_partial=allow_partial,
        config=config,
        bounds_policy="reject",
        quantization=quantization,
    )


def smooth_pixel_path(
    points: Iterable[Sequence[Any]],
    *,
    window_size: int = PATH_SMOOTH_WINDOW_SIZE,
) -> tuple[Point, ...]:
    """Apply the tracer's centered moving-average smoothing in pixel space."""

    if isinstance(window_size, bool) or window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    normalized: list[Point] = []
    for index, point in enumerate(points):
        if isinstance(point, (str, bytes)) or len(point) != 2:
            raise TraceInputError(f"points[{index}] must contain exactly two coordinates")
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError) as exc:
            raise TraceInputError(f"points[{index}] must contain finite numbers") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise TraceInputError(f"points[{index}] must contain finite numbers")
        normalized.append((x, y))

    # This condition deliberately mirrors the historical product code.
    if len(normalized) <= window_size:
        return tuple(normalized)

    radius = window_size // 2
    smoothed: list[Point] = []
    for index in range(len(normalized)):
        window = normalized[max(0, index - radius) : min(len(normalized), index + radius + 1)]
        count = len(window)
        # The historical implementation used a float32 NumPy array. Pixel
        # coordinates are small integers, so only the final division needs an
        # explicit IEEE-754 single-precision round to reproduce it exactly.
        mean_x = struct.unpack("!f", struct.pack("!f", sum(point[0] for point in window) / count))[0]
        mean_y = struct.unpack("!f", struct.pack("!f", sum(point[1] for point in window) / count))[0]
        smoothed.append(
            (mean_x, mean_y)
        )
    return tuple(smoothed)


def chaikin_smooth_path(
    points: Iterable[Sequence[Any]],
    *,
    iterations: int = CHAIKIN_ITERATIONS,
    q_weight: float = CHAIKIN_Q_WEIGHT,
    r_weight: float = CHAIKIN_R_WEIGHT,
) -> tuple[Point, ...]:
    """Apply the product's open-path Chaikin pass without QGIS or NumPy."""

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    if (
        not math.isfinite(q_weight)
        or not math.isfinite(r_weight)
        or q_weight < 0
        or r_weight < 0
        or not math.isclose(q_weight + r_weight, 1.0, abs_tol=1e-12)
    ):
        raise ValueError("Chaikin weights must be finite, non-negative, and sum to 1")

    normalized: list[Point] = []
    for index, point in enumerate(points):
        if isinstance(point, (str, bytes)) or len(point) != 2:
            raise TraceInputError(f"points[{index}] must contain exactly two coordinates")
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError) as exc:
            raise TraceInputError(f"points[{index}] must contain finite numbers") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise TraceInputError(f"points[{index}] must contain finite numbers")
        normalized.append((x, y))
    if len(normalized) < 3 or iterations == 0:
        return tuple(normalized)

    for _ in range(iterations):
        if len(normalized) < 3:
            break
        next_points: list[Point] = [normalized[0]]
        for index in range(len(normalized) - 1):
            p0 = normalized[index]
            p1 = normalized[index + 1]
            next_points.extend(
                (
                    (
                        p0[0] * q_weight + p1[0] * r_weight,
                        p0[1] * q_weight + p1[1] * r_weight,
                    ),
                    (
                        p0[0] * r_weight + p1[0] * q_weight,
                        p0[1] * r_weight + p1[1] * q_weight,
                    ),
                )
            )
        next_points.append(normalized[-1])
        normalized = next_points
    return tuple(normalized)


def centerline_points(
    result: TraceResult,
    *,
    smooth: bool = True,
    window_size: int = PATH_SMOOTH_WINDOW_SIZE,
    chaikin_iterations: int = CHAIKIN_ITERATIONS,
    segment_start_xy: Sequence[Any] | None = None,
    segment_target_xy: Sequence[Any] | None = None,
) -> tuple[Point, ...]:
    """Build the historical product centerline, including the segment start.

    For a long automatic segment this intentionally reproduces the current UI
    pipeline: centered moving average on the A* extension, three open-path
    Chaikin passes, then append that extension to the untouched start point.
    """

    if not isinstance(result, TraceResult):
        raise TypeError("result must be a TraceResult")
    if segment_start_xy is None:
        segment_start = (float(result.start[0]), float(result.start[1]))
    else:
        segment_start = _numeric_point(segment_start_xy, name="segment_start_xy")
    if not result.path and result.reached_target and segment_target_xy is not None:
        segment_target = _numeric_point(segment_target_xy, name="segment_target_xy")
        if segment_target != segment_start:
            # SmartTraceTool falls back to the untouched target when distinct
            # float prompts truncate to the same A* pixel.
            return (segment_start, segment_target)
    segment = (
        smooth_pixel_path(result.path, window_size=window_size)
        if smooth
        else tuple((float(x), float(y)) for x, y in result.path)
    )
    if smooth and len(segment) > 2:
        segment = chaikin_smooth_path(segment, iterations=chaikin_iterations)
    return (segment_start, *segment)


__all__ = [
    "A_STAR_NEIGHBORS",
    "CHAIKIN_ITERATIONS",
    "CHAIKIN_Q_WEIGHT",
    "CHAIKIN_R_WEIGHT",
    "DEFAULT_TRACE_CONFIG",
    "PATH_MAX_ITER_BASE",
    "PATH_MAX_ITER_DISTANCE_FACTOR",
    "PATH_MOVE_COST_DIAGONAL",
    "PATH_MOVE_COST_STRAIGHT",
    "PATH_SMOOTH_WINDOW_SIZE",
    "Pixel",
    "Point",
    "TraceConfig",
    "TraceInputError",
    "TraceResult",
    "centerline_points",
    "chaikin_smooth_path",
    "find_path",
    "quantize_pixel_point",
    "smooth_pixel_path",
    "trace_path",
]
