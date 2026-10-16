"""Distinguish sustained glyph following from a brief text crossing.

This optional veto is used only after the ordinary route has retained the
same genuine ends and visible arms as an admissible text connector. It reads
source-ink snapshots, never an OCR result or a hidden-contour annotation.
"""

import numpy as np

from .occlusion_route_guard import _dense, _points


def _interval(points, source, target):
    dense, _length = _dense(points)
    if dense is None:
        return None
    first = np.linalg.norm(dense - source, axis=1)
    last = np.linalg.norm(dense - target, axis=1)
    start, end = int(np.argmin(first)), int(np.argmin(last))
    if first[start] > 1.25 or last[end] > 1.25 or start >= end or end - start > 512:
        return None
    gap = dense[start:end + 1]
    lengths = np.linalg.norm(np.diff(gap, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    length = float(cumulative[-1])
    if not 3.0 <= length <= 192.0:
        return None
    # Uniform arclength makes the measurement independent of raster vertex
    # counts and of the cubic connector's denser subdivision.
    distances = np.linspace(0.0, length, int(np.ceil(length * 4)) + 1)
    samples = np.column_stack([np.interp(distances, cumulative, gap[:, axis]) for axis in (0, 1)])
    return samples, np.diff(distances), length


def _contacts(interval, core, origin):
    samples, lengths, total = interval
    middle = (samples[:-1] + samples[1:]) * 0.5 - origin
    if (np.any(middle < 0) or np.any(middle[:, 0] > core.shape[1] - 1)
            or np.any(middle[:, 1] > core.shape[0] - 1)):
        return None
    pixels = np.rint(middle).astype(np.int64)
    contact = core[pixels[:, 1], pixels[:, 0]]
    longest = current = 0.0
    for touches, length in zip(contact, lengths):
        current = current + float(length) if touches else 0.0
        longest = max(longest, current)
    return float(lengths[contact].sum()), longest, total


def follows_text_ink(ordinary, inferred, *, core, origin, source_xy, target_xy):
    """Whether a materially longer old gap follows substantially more glyph ink.

    A long run (8 native pixels) and reductions in both continuous and total
    actual-ink contact are required. A short transverse crossing or an equally
    good old line keeps priority. Ambiguous/malformed input also keeps priority.
    This predicate alone does not authorize replacing a route.
    """
    old, new = _points(ordinary), _points(inferred)
    if old is None or new is None:
        return False
    try:
        ink = np.asarray(core)
        offset, source, target = [np.asarray(value, dtype=np.float64)
                                  for value in (origin, source_xy, target_xy)]
    except (TypeError, ValueError, OverflowError):
        return False
    if (ink.dtype != np.bool_ or ink.ndim != 2 or not ink.size or ink.size > 1024 * 1024
            or any(value.shape != (2,) or not np.isfinite(value).all()
                   for value in (offset, source, target))):
        return False
    # Reject huge finite coordinates before dense sample-count conversion.
    for points in (old, new):
        relative = points - offset
        if (np.any(relative < 0) or np.any(relative[:, 0] > ink.shape[1] - 1)
                or np.any(relative[:, 1] > ink.shape[0] - 1)):
            return False
    old_interval, new_interval = _interval(old, source, target), _interval(new, source, target)
    if old_interval is None or new_interval is None:
        return False
    old_contact, new_contact = _contacts(old_interval, ink, offset), _contacts(new_interval, ink, offset)
    if old_contact is None or new_contact is None:
        return False
    old_total, old_run, old_length = old_contact
    new_total, new_run, new_length = new_contact
    return bool(
        old_run >= 8.0 and old_run >= 2.0 * new_run and old_run - new_run >= 4.0
        and old_total - new_total >= 4.0 and old_total >= 1.5 * new_total
        and old_length - new_length >= 4.0
    )
