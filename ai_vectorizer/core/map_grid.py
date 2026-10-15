"""Conservative, NumPy-only detection of near-axis printed map grids.

This is geometric evidence, not a semantic classifier. Each perpendicular
family must have at least two separated lines; isolated strokes stay alone.
The scores only raise finite tracing costs and never remove raster pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Tuple

import numpy as np


class MapGridCancelled(RuntimeError):
    """Raised when a caller cancels grid detection."""


@dataclass(frozen=True)
class MapGridPrior:
    horizontal_score: np.ndarray
    vertical_score: np.ndarray
    horizontal_slope: float = 0.0
    vertical_slope: float = 0.0

    def __post_init__(self) -> None:
        shape = None
        for name in ("horizontal_score", "vertical_score"):
            try:
                value = np.array(getattr(self, name), dtype=np.float32, order="C", copy=True)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("grid scores must be finite numeric arrays") from exc
            if value.ndim != 2 or min(value.shape, default=0) < 1:
                raise ValueError("grid scores must be non-empty 2D arrays")
            if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                raise ValueError("grid scores must be finite values in [0, 1]")
            if shape is not None and shape != value.shape:
                raise ValueError("grid scores must have matching dimensions")
            shape = value.shape
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("horizontal_slope", "vertical_slope"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 0.05:
                raise ValueError("grid slopes must be finite numbers in [-0.05, 0.05]")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.horizontal_score.shape

    @property
    def score(self) -> np.ndarray:
        """Combined evidence for scalar consumers (finite penalty only)."""
        value = np.maximum(self.horizontal_score, self.vertical_score)
        value.setflags(write=False)
        return value


GridLinePrior = MapGridPrior


def crop_grid_prior(prior: MapGridPrior, bounds) -> MapGridPrior:
    if not isinstance(prior, MapGridPrior):
        raise TypeError("grid_prior must be a MapGridPrior instance")
    if len(bounds) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in bounds):
        raise ValueError("grid bounds must be four integer pixel coordinates")
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= prior.shape[1] and 0 <= y0 < y1 <= prior.shape[0]):
        raise ValueError("grid bounds lie outside the prior")
    return MapGridPrior(prior.horizontal_score[y0:y1, x0:x1], prior.vertical_score[y0:y1, x0:x1], prior.horizontal_slope, prior.vertical_slope)


def _check_cancel(cancel_check) -> None:
    if cancel_check is not None and cancel_check():
        raise MapGridCancelled("map grid detection cancelled")


def _line_family(dark: np.ndarray, cancel_check):
    """Find near-horizontal full-span strokes, using sparse projections."""
    height, width = dark.shape
    if min(height, width) < 64:
        return [], 0.0
    ys, xs = np.nonzero(dark)
    if len(xs) < width * 0.75:
        return [], 0.0
    center = (width - 1) * 0.5
    shifted_x = xs.astype(np.float32) - center
    sample_x = np.arange(width)
    best = []
    # ±1.8 degrees covers modest scanner rotation. A 3px verification band
    # accommodates the residual half-step error without resampling the map.
    for slope in np.linspace(-0.032, 0.032, 17):
        _check_cancel(cancel_check)
        bins = np.rint(ys - slope * shifted_x).astype(np.int32)
        valid = (bins >= 0) & (bins < height)
        projection = np.bincount(bins[valid], minlength=height)
        band = np.convolve(projection, np.ones(3), mode="same")
        peaks = np.flatnonzero((band >= width * 0.84) & (band >= np.roll(band, 1)) & (band >= np.roll(band, -1)))
        # Only the strongest few dozen candidates need detailed verification.
        peaks = peaks[np.argsort(band[peaks])[-48:]]
        for intercept in peaks:
            line_y = np.rint(intercept + slope * (sample_x - center)).astype(np.int32)
            in_image = (line_y >= 2) & (line_y < height - 2)
            if np.count_nonzero(in_image) < width * 0.90:
                continue
            coverage = np.zeros(width, dtype=bool)
            for offset in range(-1, 2):
                rows = np.clip(line_y + offset, 0, height - 1)
                coverage |= dark[rows, sample_x] & in_image
            fraction = float(np.mean(coverage))
            if fraction < 0.88 or min(float(np.mean(part)) for part in np.array_split(coverage, 8)) < 0.65:
                continue
            # Refine from dark samples inside the verified 3px band. This
            # avoids a tie selecting a parallel line one pixel off-centre.
            sums = np.zeros(width, dtype=np.float64)
            counts = np.zeros(width, dtype=np.float64)
            for offset in range(-1, 2):
                rows = np.clip(line_y + offset, 0, height - 1)
                hit = dark[rows, sample_x] & in_image
                sums[hit] += rows[hit]
                counts[hit] += 1
            keep = counts > 0
            sample_centered = sample_x[keep] - center
            sample_rows = sums[keep] / counts[keep]
            mean_x = float(np.mean(sample_centered))
            mean_y = float(np.mean(sample_rows))
            centered_x = sample_centered - mean_x
            refined_slope = float(np.dot(centered_x, sample_rows - mean_y) / max(1.0, np.dot(centered_x, centered_x)))
            refined_intercept = mean_y - refined_slope * mean_x
            best.append((fraction, refined_intercept, refined_slope))
    if not best:
        return [], 0.0
    best.sort(reverse=True)
    selected = []
    minimum_separation = max(12.0, min(32.0, min(height, width) * 0.04))
    for confidence, intercept, slope in best:
        # One stroke must not be counted repeatedly at slightly different
        # slopes, or as two parallel sides of a thick scan line.
        if all(abs(intercept - old[1]) > minimum_separation for old in selected):
            selected.append((confidence, intercept, slope))
    # A printed family should share the same scanner rotation. Re-verify all
    # candidates at its strongest angle to keep the directional cost precise.
    family_slope = selected[0][2]
    return selected, family_slope


def _paint_family(dark: np.ndarray, lines) -> np.ndarray:
    height, width = dark.shape
    score = np.zeros((height, width), dtype=np.float32)
    xs = np.arange(width)
    for confidence, intercept, candidate_slope in lines:
        # Candidate angle is used for location; the common family angle is
        # used only for routing alignment, a difference bounded to 1.8 degrees.
        ys = np.rint(intercept + candidate_slope * (xs - (width - 1) * 0.5)).astype(np.int32)
        for offset in range(-1, 2):
            rows = ys + offset
            valid = (rows >= 0) & (rows < height)
            # Cover the fitted stroke and scan antialiasing only. A broad
            # band needlessly charges nearby contours at diagonal crossings.
            # Visible-ink gating separately prevents white halo shortcuts.
            score[rows[valid], xs[valid]] = np.maximum(score[rows[valid], xs[valid]], np.float32(confidence))
    return score


def detect_map_grid(image: np.ndarray, cancel_check: Optional[Callable[[], bool]] = None) -> MapGridPrior:
    """Detect long, dark, nearly perpendicular grid strokes conservatively.

    Byte grayscale and RGB(A) are supported. Unsupported numeric raster
    ranges produce zero evidence; the raster itself is never normalized or
    changed. Detection is performed on the supplied cache once, not on each
    moving Live-Wire window. Cancellation is checked between projections.
    """
    _check_cancel(cancel_check)
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (3, 4)) or min(array.shape[:2], default=0) < 1:
        raise ValueError("map grid image must be non-empty grayscale or RGB(A)")
    shape = array.shape[:2]
    zero = np.zeros(shape, dtype=np.float32)
    if array.dtype != np.uint8:
        return MapGridPrior(zero, zero)
    if array.ndim == 3:
        rgb = array[:, :, :3]
        maximum = np.max(rgb, axis=2)
        minimum = np.min(rgb, axis=2)
        dark = (maximum < 185) & ((maximum.astype(np.int16) - minimum) <= 48)
        if array.shape[2] == 4:
            dark &= array[:, :, 3] >= 240
    else:
        dark = array < 160
    horizontal, h_slope = _line_family(dark, cancel_check)
    vertical, v_slope = _line_family(dark.T, cancel_check)
    _check_cancel(cancel_check)
    if len(horizontal) < 2 or len(vertical) < 2:
        return MapGridPrior(zero, zero)
    h_score = _paint_family(dark, horizontal)
    v_score = _paint_family(dark.T, vertical).T
    return MapGridPrior(h_score, v_score, h_slope, v_slope)


__all__ = ["MapGridPrior", "GridLinePrior", "MapGridCancelled", "detect_map_grid", "crop_grid_prior"]
