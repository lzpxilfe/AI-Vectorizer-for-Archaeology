"""Preserve an existing route whose visible arms already match a label gap.

This check cannot establish the hidden contour. It only prevents replacing a
well-supported visible route with another inferred shape inside the same
independently detected label. All points use absolute native image XY; only
``mask`` is window-local, with its upper-left pixel at ``origin``.

The operation reads immutable geometry prepared by the worker. It performs no
image filtering, graph search, OCR or model inference during cursor movement.
"""

from __future__ import annotations

import math

import numpy as np

from .confirmed_gap_curve import _self_intersects


_MAX_POINTS = 8192
_MAX_SAMPLES = 16384
_VISIBLE_TOLERANCE = 1.25


def _points(value):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if (result.ndim != 2 or result.shape[1] != 2
            or not 2 <= len(result) <= _MAX_POINTS
            or not np.isfinite(result).all()):
        return None
    # Consecutive duplicate points have no geometric meaning, and could
    # otherwise turn an ordinary endpoint into an alleged self-contact.
    result = result[np.r_[True, np.any(np.diff(result, axis=0), axis=1)]]
    return result if len(result) >= 2 else None


def _dense(points):
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    counts = np.maximum(1, np.ceil(lengths * 2)).astype(np.int64)
    if int(counts.sum()) + 1 > _MAX_SAMPLES:
        return None, None
    result = [points[:1]]
    for first, last, count in zip(points[:-1], points[1:], counts):
        result.append(first + (last - first) * np.linspace(0, 1, int(count) + 1)[1:, None])
    return np.concatenate(result), float(lengths.sum())


def _outside(points, mask, origin):
    with np.errstate(over="ignore", invalid="ignore"):
        relative = points - origin
    if (not np.isfinite(relative).all() or np.any(relative < -0.5)
            or np.any(relative[:, 0] >= mask.shape[1] - 0.5)
            or np.any(relative[:, 1] >= mask.shape[0] - 0.5)):
        return None
    local = np.rint(relative).astype(np.int64)
    if (np.any(local[:, 0] < 0) or np.any(local[:, 0] >= mask.shape[1])
            or np.any(local[:, 1] < 0) or np.any(local[:, 1] >= mask.shape[0])):
        return None
    return ~mask[local[:, 1], local[:, 0]]


def _covered(first, second):
    """Bounded spatial hash agreement, avoiding an unbounded distance matrix."""
    radius = _VISIBLE_TOLERANCE
    buckets = {}
    for x, y in second:
        key = (math.floor(x / radius), math.floor(y / radius))
        bucket = buckets.setdefault(key, [])
        # A dense, repeatedly revisited visible point is not an admissible
        # simple continuation. Decline before pathological work accumulates.
        if len(bucket) >= 64:
            return False
        bucket.append((float(x), float(y)))
    limit = radius * radius + 1e-9
    for x, y in first:
        bx, by = math.floor(x / radius), math.floor(y / radius)
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if any((x - px) ** 2 + (y - py) ** 2 <= limit
                       for px, py in buckets.get((bx + dx, by + dy), ())):
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


def preserves_visible_route(
    ordinary, inferred, *, mask, origin, source_xy, target_xy,
):
    """Whether an ordinary route already retains the chosen native gap arms.

    ``source_xy`` and ``target_xy`` are the exact, ordered original-native
    terminals selected for ``inferred``. They cannot be manufactured from a
    cut glyph mask. A positive result preserves the ordinary route verbatim;
    it makes no accuracy claim about its unobserved interval.
    """
    old, new = _points(ordinary), _points(inferred)
    if old is None or new is None:
        return False
    try:
        region = np.asarray(mask)
        offset = np.asarray(origin, dtype=np.float64)
        source = np.asarray(source_xy, dtype=np.float64)
        target = np.asarray(target_xy, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return False
    if (region.dtype != np.bool_ or region.ndim != 2 or not region.size
            or region.size > 1024 * 1024 or not np.any(region)
            or any(p.shape != (2,) or not np.isfinite(p).all()
                   for p in (offset, source, target))):
        return False
    if (np.linalg.norm(old[0] - new[0]) > 0.25
            or np.linalg.norm(old[-1] - new[-1]) > 0.25):
        return False
    # Check the bounded window before lengths or integer sample counts. A
    # finite but enormous malformed coordinate must not overflow that work.
    if _outside(old, region, offset) is None or _outside(new, region, offset) is None:
        return False
    chord = float(np.linalg.norm(target - source))
    if not 3.0 <= chord <= 96.0:
        return False
    exact_source = np.linalg.norm(new - source, axis=1)
    exact_target = np.linalg.norm(new - target, axis=1)
    ni, nj = int(np.argmin(exact_source)), int(np.argmin(exact_target))
    if exact_source[ni] > 1e-6 or exact_target[nj] > 1e-6 or ni >= nj:
        return False
    old_dense, old_length = _dense(old)
    new_dense, new_length = _dense(new)
    if old_dense is None or new_dense is None:
        return False
    if old_length > max(new_length * 1.25, new_length + 4.0):
        return False
    distances_source = np.linalg.norm(old_dense - source, axis=1)
    distances_target = np.linalg.norm(old_dense - target, axis=1)
    oi, oj = int(np.argmin(distances_source)), int(np.argmin(distances_target))
    if (distances_source[oi] > _VISIBLE_TOLERANCE
            or distances_target[oj] > _VISIBLE_TOLERANCE or oi >= oj):
        return False
    gap = old_dense[oi:oj + 1]
    if len(gap) > 512:
        return False
    gap_length = float(np.linalg.norm(np.diff(gap, axis=0), axis=1).sum())
    if gap_length > min(192.0, chord * 1.8):
        return False
    if _self_intersects([tuple(point) for point in gap]):
        return False
    old_outside = _outside(old_dense, region, offset)
    new_outside = _outside(new_dense, region, offset)
    if old_outside is None or new_outside is None:
        return False
    # There must be exactly one contiguous passage through independently
    # detected text, lying between the two established native terminals.
    inside_indices = np.flatnonzero(~old_outside)
    if not len(inside_indices):
        return False
    entry, exit_ = int(inside_indices[0]), int(inside_indices[-1])
    if np.any(old_outside[entry:exit_ + 1]):
        return False
    if (entry < oi and np.linalg.norm(old_dense[entry] - source) > 2.0
            or exit_ > oj and np.linalg.norm(old_dense[exit_] - target) > 2.0):
        return False
    old_visible, new_visible = old_dense[old_outside], new_dense[new_outside]
    if not len(old_visible) or not len(new_visible):
        return False
    return (_covered(old_visible, new_visible)
            and _covered(new_visible, old_visible))
