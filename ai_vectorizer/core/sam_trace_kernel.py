# -*- coding: utf-8 -*-
"""QGIS-independent SAM mask-to-centerline product pipeline.

This module owns the boundary after a segmentation backend has selected one
two-dimensional mask.  It deliberately does not import NumPy, OpenCV,
scikit-image, or the trace kernel at module import time.  Callers may inject
those dependencies (the isolated benchmark worker does this), while the QGIS
product can use the lazy defaults.

The two mask-closing stages are intentional and match ``SmartTraceTool``:

1. :func:`postprocess_mask` thresholds, closes, and applies the area guard.
2. :func:`build_cost_map` closes the accepted mask again before skeletonizing.

``trace_mask`` and ``trace_mask_centerline`` therefore require an already
postprocessed boolean mask.  They never hide a third close or another area
guard.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence


Pixel = tuple[int, int]
Point = tuple[float, float]
ThinBinaryMask = Callable[[Any], Any]


PRODUCT_NEIGHBORS: tuple[Pixel, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


@dataclass(frozen=True)
class SamTraceConfig:
    """Versioned constants for the current Smart Trace SAM policy."""

    mask_min_pixels: int = 24
    mask_max_area_ratio: float = 0.35
    mask_close_kernel: tuple[int, int] = (3, 3)
    nearest_active_radius: int = 20
    edge_pixel_threshold: int = 128
    outside_cost: float = 12.0
    inside_cost: float = 2.5
    edge_cost: float = 1.6
    skeleton_cost: float = 1.0
    centerline_bonus: float = 0.75
    straight_move_cost: float = 1.0
    diagonal_move_cost: float = 1.41421356237
    max_iterations_base: int = 100_000
    max_iterations_distance_factor: int = 500
    max_dimension: int = 1_000
    smooth_window_size: int = 5
    chaikin_iterations: int = 3
    neighbors: tuple[Pixel, ...] = PRODUCT_NEIGHBORS

    def __post_init__(self) -> None:
        if (
            len(self.mask_close_kernel) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.mask_close_kernel
            )
        ):
            raise ValueError("mask_close_kernel must contain two positive integers")
        if (
            isinstance(self.mask_min_pixels, bool)
            or not isinstance(self.mask_min_pixels, int)
            or self.mask_min_pixels < 0
        ):
            raise ValueError("mask_min_pixels must be a non-negative integer")
        if not math.isfinite(self.mask_max_area_ratio) or not (
            0.0 <= self.mask_max_area_ratio <= 1.0
        ):
            raise ValueError("mask_max_area_ratio must be between zero and one")
        if (
            isinstance(self.nearest_active_radius, bool)
            or not isinstance(self.nearest_active_radius, int)
            or self.nearest_active_radius < 1
        ):
            raise ValueError("nearest_active_radius must be a positive integer")
        if (
            isinstance(self.max_dimension, bool)
            or not isinstance(self.max_dimension, int)
            or self.max_dimension < 1
        ):
            raise ValueError("max_dimension must be a positive integer")
        if (
            isinstance(self.smooth_window_size, bool)
            or not isinstance(self.smooth_window_size, int)
            or self.smooth_window_size < 1
            or self.smooth_window_size % 2 == 0
        ):
            raise ValueError("smooth_window_size must be a positive odd integer")
        if (
            isinstance(self.chaikin_iterations, bool)
            or not isinstance(self.chaikin_iterations, int)
            or self.chaikin_iterations < 0
        ):
            raise ValueError("chaikin_iterations must be a non-negative integer")
        if tuple(self.neighbors) != PRODUCT_NEIGHBORS:
            raise ValueError("neighbors must use the fixed product 8-neighbor policy")
        costs = (
            self.outside_cost,
            self.inside_cost,
            self.edge_cost,
            self.skeleton_cost,
            self.straight_move_cost,
            self.diagonal_move_cost,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in costs):
            raise ValueError("SAM and movement costs must be finite and positive")
        if not math.isfinite(self.centerline_bonus) or self.centerline_bonus < 0.0:
            raise ValueError("centerline_bonus must be finite and non-negative")


DEFAULT_CONFIG = SamTraceConfig()


@dataclass(frozen=True)
class SamTraceResult:
    """Intermediate product evidence retained for UI and worker callers."""

    closed_mask: Any
    skeleton: Any
    cost_map: Any
    snapped_start: Pixel
    snapped_end: Pixel
    trace_result: Any

    @property
    def path(self) -> tuple[Pixel, ...]:
        """Raw A* extension, excluding the snapped start pixel."""

        return tuple(self.trace_result.path)


def _numpy(np_module: Any | None) -> Any:
    if np_module is not None:
        return np_module
    try:
        import numpy as imported_numpy
    except Exception as exc:  # pragma: no cover - environment-specific detail.
        raise ImportError("SAM tracing requires NumPy") from exc
    return imported_numpy


def _cv2(cv2_module: Any | None) -> Any:
    if cv2_module is not None:
        return cv2_module
    from .dependencies import require_cv2

    return require_cv2("SAM tracing")


def _thin_binary_mask(thin_binary_mask: ThinBinaryMask | None) -> ThinBinaryMask:
    if thin_binary_mask is not None:
        return thin_binary_mask
    from .edge_detector import EdgeDetector

    return EdgeDetector.thin_binary_mask


def _trace_kernel(trace_kernel: Any | None) -> Any:
    if trace_kernel is not None:
        return trace_kernel
    from . import trace_kernel as product_trace_kernel

    return product_trace_kernel


def postprocess_mask(
    mask: Any,
    *,
    cv2_module: Any | None = None,
    np_module: Any | None = None,
    config: SamTraceConfig = DEFAULT_CONFIG,
) -> Any | None:
    """Threshold, close, and area-guard one backend-selected 2-D mask.

    ``None`` is the product's existing signal that SAM should not supply a
    path.  The threshold remains ``mask > 0``; sigmoid conversion is neither
    needed nor performed here.
    """

    if not isinstance(config, SamTraceConfig):
        raise TypeError("config must be a SamTraceConfig")
    np = _numpy(np_module)
    candidate = np.asarray(mask)
    if candidate.ndim != 2:
        return None
    cv2 = _cv2(cv2_module)

    closed = cv2.morphologyEx(
        (candidate > 0).astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones(config.mask_close_kernel, np.uint8),
    )
    active_pixels = int(np.count_nonzero(closed))
    if active_pixels < config.mask_min_pixels:
        return None
    if active_pixels > int(closed.size * config.mask_max_area_ratio):
        return None
    return closed > 0


def build_cost_map(
    mask: Any,
    edges: Any | None = None,
    *,
    cv2_module: Any | None = None,
    np_module: Any | None = None,
    thin_binary_mask: ThinBinaryMask | None = None,
    config: SamTraceConfig = DEFAULT_CONFIG,
) -> tuple[Any, Any, Any]:
    """Build the product SAM mask, skeleton, and float32 A* cost map.

    ``mask`` must already be the boolean result of :func:`postprocess_mask`.
    The second close in this function is intentional historical behavior.
    """

    if not isinstance(config, SamTraceConfig):
        raise TypeError("config must be a SamTraceConfig")
    np = _numpy(np_module)
    candidate = np.asarray(mask)
    if candidate.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    cv2 = _cv2(cv2_module)

    closed_mask = cv2.morphologyEx(
        candidate.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones(config.mask_close_kernel, np.uint8),
    ) > 0
    skeleton = np.asarray(_thin_binary_mask(thin_binary_mask)(closed_mask)).astype(bool)
    if skeleton.shape != closed_mask.shape:
        raise ValueError("thin_binary_mask must preserve the mask shape")

    cost_map = np.full(closed_mask.shape, config.outside_cost, dtype=np.float32)
    cost_map[closed_mask] = config.inside_cost

    if edges is not None:
        edge_pixels = np.asarray(edges) > config.edge_pixel_threshold
        if edge_pixels.shape != closed_mask.shape:
            raise ValueError("edges must have the same shape as mask")
        cost_map[np.logical_and(closed_mask, edge_pixels)] = config.edge_cost

    distance_to_background = cv2.distanceTransform(
        closed_mask.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    max_distance = float(distance_to_background.max())
    if max_distance > 0.0:
        normalized_distance = distance_to_background / max_distance
        cost_map[closed_mask] -= (
            normalized_distance[closed_mask] * config.centerline_bonus
        )

    cost_map[skeleton] = config.skeleton_cost
    cost_map = np.clip(cost_map, config.skeleton_cost, None)
    return closed_mask, skeleton, cost_map


def _clamp_pixel(px: Any, py: Any, width: int, height: int) -> Pixel:
    return (
        max(0, min(width - 1, int(round(px)))),
        max(0, min(height - 1, int(round(py)))),
    )


def nearest_active_pixel(
    binary_mask: Any,
    px: Any,
    py: Any,
    max_radius: int | None = None,
    *,
    config: SamTraceConfig = DEFAULT_CONFIG,
) -> Pixel | None:
    """Return the product's deterministic nearest active ring pixel.

    Ring scan order and first-tie retention intentionally match
    ``SmartTraceTool._nearest_active_pixel``.
    """

    if binary_mask is None:
        return None
    if not isinstance(config, SamTraceConfig):
        raise TypeError("config must be a SamTraceConfig")

    shape = getattr(binary_mask, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("binary_mask must be a two-dimensional array")
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        return None
    px, py = _clamp_pixel(px, py, width, height)
    if binary_mask[py, px]:
        return (px, py)

    # ``or`` preserves the historical max_radius=0 behavior.
    radius_limit = max_radius or config.nearest_active_radius
    best = None
    best_distance = None
    for radius in range(1, radius_limit + 1):
        x_min = max(0, px - radius)
        x_max = min(width - 1, px + radius)
        y_min = max(0, py - radius)
        y_max = min(height - 1, py + radius)

        for ny in range(y_min, y_max + 1):
            for nx in range(x_min, x_max + 1):
                if nx not in (x_min, x_max) and ny not in (y_min, y_max):
                    continue
                if not binary_mask[ny, nx]:
                    continue
                distance = (nx - px) ** 2 + (ny - py) ** 2
                if best is None or distance < best_distance:
                    best = (nx, ny)
                    best_distance = distance

        if best is not None:
            return best
    return None


def _pixel_in_bounds(point: Pixel, width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height


def trace_mask(
    mask: Any,
    edges: Any | None,
    start_xy: Sequence[Any],
    end_xy: Sequence[Any],
    *,
    trace_kernel: Any | None = None,
    cv2_module: Any | None = None,
    np_module: Any | None = None,
    thin_binary_mask: ThinBinaryMask | None = None,
    config: SamTraceConfig = DEFAULT_CONFIG,
) -> SamTraceResult | None:
    """Trace an already-postprocessed SAM mask with the strict product A* API.

    Endpoints first snap to the skeleton.  If either skeleton snap fails, both
    are recomputed against the closed mask, matching the product's all-or-none
    fallback.  ``None`` means no SAM path boundary could be established.
    """

    if not isinstance(config, SamTraceConfig):
        raise TypeError("config must be a SamTraceConfig")
    kernel = _trace_kernel(trace_kernel)
    closed_mask, skeleton, cost_map = build_cost_map(
        mask,
        edges,
        cv2_module=cv2_module,
        np_module=np_module,
        thin_binary_mask=thin_binary_mask,
        config=config,
    )
    height, width = closed_mask.shape
    start_pixel = kernel.quantize_pixel_point(
        start_xy,
        mode="truncate",
        name="start_xy",
    )
    end_pixel = kernel.quantize_pixel_point(
        end_xy,
        mode="truncate",
        name="end_xy",
    )
    if not _pixel_in_bounds(start_pixel, width, height) or not _pixel_in_bounds(
        end_pixel,
        width,
        height,
    ):
        return None

    snapped_start = nearest_active_pixel(
        skeleton,
        start_pixel[0],
        start_pixel[1],
        config=config,
    )
    snapped_end = nearest_active_pixel(
        skeleton,
        end_pixel[0],
        end_pixel[1],
        config=config,
    )
    if snapped_start is None or snapped_end is None:
        snapped_start = nearest_active_pixel(
            closed_mask,
            start_pixel[0],
            start_pixel[1],
            config=config,
        )
        snapped_end = nearest_active_pixel(
            closed_mask,
            end_pixel[0],
            end_pixel[1],
            config=config,
        )
    if snapped_start is None or snapped_end is None:
        return None

    trace_config = kernel.TraceConfig(
        straight_move_cost=config.straight_move_cost,
        diagonal_move_cost=config.diagonal_move_cost,
        max_iterations_base=config.max_iterations_base,
        max_iterations_distance_factor=config.max_iterations_distance_factor,
        max_width=config.max_dimension,
        max_height=config.max_dimension,
        max_cells=config.max_dimension * config.max_dimension,
        validate_all_costs=False,
        validate_accessed_costs=False,
        neighbors=tuple(config.neighbors),
    )
    result = kernel.trace_path(
        cost_map,
        snapped_start,
        snapped_end,
        allow_partial=False,
        config=trace_config,
    )
    return SamTraceResult(
        closed_mask=closed_mask,
        skeleton=skeleton,
        cost_map=cost_map,
        snapped_start=snapped_start,
        snapped_end=snapped_end,
        trace_result=result,
    )


def trace_mask_centerline(
    mask: Any,
    edges: Any | None,
    start_xy: Sequence[Any],
    end_xy: Sequence[Any],
    *,
    trace_kernel: Any | None = None,
    cv2_module: Any | None = None,
    np_module: Any | None = None,
    thin_binary_mask: ThinBinaryMask | None = None,
    config: SamTraceConfig = DEFAULT_CONFIG,
) -> tuple[Point, ...]:
    """Return the worker-ready ordered centerline for a processed mask.

    The historical profile is applied in its existing order: the A* extension
    excludes the snapped start, then receives a centered five-point moving
    average and three open-path Chaikin passes, and finally the untouched raw
    segment start is prepended.  An empty tuple means the product would not use
    this SAM result and would continue to its ordinary edge-path fallback.
    """

    kernel = _trace_kernel(trace_kernel)
    traced = trace_mask(
        mask,
        edges,
        start_xy,
        end_xy,
        trace_kernel=kernel,
        cv2_module=cv2_module,
        np_module=np_module,
        thin_binary_mask=thin_binary_mask,
        config=config,
    )
    if (
        traced is None
        or traced.trace_result.status != "complete"
        or not traced.trace_result.path
    ):
        return ()
    return tuple(
        kernel.centerline_points(
            traced.trace_result,
            smooth=True,
            window_size=config.smooth_window_size,
            chaikin_iterations=config.chaikin_iterations,
            segment_start_xy=start_xy,
            segment_target_xy=end_xy,
        )
    )


__all__ = [
    "DEFAULT_CONFIG",
    "PRODUCT_NEIGHBORS",
    "Pixel",
    "Point",
    "SamTraceConfig",
    "SamTraceResult",
    "build_cost_map",
    "nearest_active_pixel",
    "postprocess_mask",
    "trace_mask",
    "trace_mask_centerline",
]
