"""Conservative, sparse geometric proposals for missing contour fragments.

Only native skeleton endpoints can be paired.  A pair needs sustained,
unbranched context on both sides and a mutually unique continuation.  The
retained cubic is an inference, not observed ink or recovered ground truth.
No RGB threshold, model, dilation or broad support corridor is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Callable, Optional, Tuple

import numpy as np

from .observed_connection import ObservedConnectionCancelled


Point = Tuple[float, float]
Pixel = Tuple[int, int]


@dataclass(frozen=True)
class LabelGapConfig:
    """Hard bounds and deliberately conservative continuation tolerances."""

    min_gap_pixels: float = 4.0
    max_gap_pixels: float = 96.0
    min_context_pixels: float = 12.0
    context_pixels: float = 24.0
    min_alignment: float = 0.80
    max_context_turn_radians: float = math.radians(40.0)
    max_context_fit_error: float = 1.10
    max_curve_deviation_ratio: float = 0.20
    ambiguity_score_margin: float = 0.30
    max_endpoints: int = 256
    max_candidate_pairs: int = 1024
    max_window_pixels: int = 1024 * 1024

    def validate(self):
        for name in (
            "min_gap_pixels", "max_gap_pixels", "min_context_pixels",
            "context_pixels", "min_alignment", "max_context_turn_radians",
            "max_context_fit_error", "max_curve_deviation_ratio",
            "ambiguity_score_margin",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if not 2.0 <= self.min_gap_pixels <= self.max_gap_pixels <= 96.0:
            raise ValueError("label gap length must be between 2 and 96 pixels")
        if not 8.0 <= self.min_context_pixels <= self.context_pixels <= 32.0:
            raise ValueError("label gap context must be between 8 and 32 pixels")
        if not 0.70 <= self.min_alignment <= 1.0:
            raise ValueError("label gap alignment is invalid")
        if not 0.0 < self.max_context_turn_radians <= math.pi / 3:
            raise ValueError("label gap context turn is invalid")
        if not 0.25 <= self.max_context_fit_error <= 2.0:
            raise ValueError("label gap fit error is invalid")
        if not 0.0 < self.max_curve_deviation_ratio <= 0.25:
            raise ValueError("label gap curve deviation is invalid")
        if not 0.0 < self.ambiguity_score_margin <= 1.0:
            raise ValueError("label gap ambiguity margin is invalid")
        for name, maximum in (
            ("max_endpoints", 1024), ("max_candidate_pairs", 4096),
            ("max_window_pixels", 1024 * 1024),
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} is outside its work bound")
        return self


DEFAULT_LABEL_GAP_CONFIG = LabelGapConfig().validate()


@dataclass(frozen=True)
class LabelGapConnector:
    """One immutable proposal in the input skeleton's local XY coordinates."""

    source_xy: Pixel
    target_xy: Pixel
    points_xy: Tuple[Point, ...]
    gap_length_pixels: float
    path_length_pixels: float
    score: float


@dataclass(frozen=True)
class _Endpoint:
    pixel: Pixel
    component: int
    outward: np.ndarray
    context: np.ndarray
    length: float


def _degree(mask):
    """Use native adjacency, suppressing only redundant diagonal shortcuts.

    A one-pixel digital bend often has an L-shaped triangle under ordinary
    8-connectivity.  The diagonal is redundant when either shared orthogonal
    neighbour exists.  Removing it only for context walking avoids labelling
    that ordinary bend as a branch.  Component identity still uses native
    8-connectivity, and no routing graph is changed here.
    """

    padded = np.pad(mask, 1)
    height, width = mask.shape
    degree = np.zeros(mask.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            neighbours = padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            if dx and dy:
                horizontal = padded[1:1 + height, 1 + dx:1 + dx + width]
                vertical = padded[1 + dy:1 + dy + height, 1:1 + width]
                neighbours = neighbours & ~horizontal & ~vertical
            degree += neighbours
    degree[~mask] = 0
    return degree


def _neighbours(mask, pixel):
    x, y = pixel
    height, width = mask.shape
    found = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height and mask[ny, nx]):
                continue
            if dx and dy and (mask[y, nx] or mask[ny, x]):
                continue
            found.append((nx, ny))
    return found


def _endpoint(mask, degree, components, pixel, config):
    previous = None
    current = pixel
    points = [pixel]
    length = 0.0
    # At most 33 native steps fit the allowed 32-pixel context.  The extra
    # step is only for its terminal sub-pixel length, never an unbounded walk.
    for _ in range(int(math.ceil(config.context_pixels)) + 1):
        x, y = current
        expected = 1 if previous is None else 2
        if int(degree[y, x]) != expected:
            # A branch or another end inside the required context is not a
            # reliable tangent.  A later branch is outside sampled context.
            if length < config.min_context_pixels:
                return None
            break
        following = [value for value in _neighbours(mask, current) if value != previous]
        if len(following) != 1:
            return None
        following = following[0]
        length += math.hypot(following[0] - current[0], following[1] - current[1])
        previous, current = current, following
        points.append(current)
        if length >= config.context_pixels:
            break
    if length < config.min_context_pixels:
        return None
    context = np.asarray(points, dtype=np.float64)
    origin = context[0]
    chord = float(np.linalg.norm(context[-1] - origin))
    if chord < length * 0.90:
        return None
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(context, axis=0), axis=1))]
    # Signed arclength is negative inward, so the fitted first derivative
    # points outward from the last visible native pixel into the gap.
    signed = -distances
    coefficients = np.linalg.lstsq(
        np.column_stack((signed, signed ** 2)), context - origin, rcond=None
    )[0]
    outward = coefficients[0]
    norm = float(np.linalg.norm(outward))
    if norm < 0.5:
        return None
    outward = outward / norm
    # Independently compare two sustained secants; a fitted tangent must not
    # conceal a tight hook or a broad turn just before the occlusion.
    halfway = max(1, int(np.searchsorted(distances, distances[-1] * 0.5)))
    first = origin - context[halfway]
    second = context[halfway] - context[-1]
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    if float(np.dot(first, second)) < math.cos(config.max_context_turn_radians):
        return None
    if float(np.dot(outward, first)) < math.cos(config.max_context_turn_radians):
        return None
    context.setflags(write=False)
    outward.setflags(write=False)
    x, y = pixel
    return _Endpoint(pixel, int(components[y, x]), outward, context, length)


def _cubic_normal(t, first_slope, last_slope):
    # Chordwise Hermite interpolation: longitudinal position is exactly t.
    # Its endpoint normal displacement is zero and endpoint slopes survive
    # unchanged.  No clipping can turn incompatible tangents into a bridge.
    return ((t ** 3 - 2 * t ** 2 + t) * first_slope
            + (t ** 3 - t ** 2) * last_slope)


def _proposal(first, last, config):
    start = np.asarray(first.pixel, dtype=np.float64)
    end = np.asarray(last.pixel, dtype=np.float64)
    difference = end - start
    gap = float(np.linalg.norm(difference))
    if not config.min_gap_pixels <= gap <= config.max_gap_pixels:
        return None
    if min(first.length, last.length) < max(config.min_context_pixels, gap * 0.20):
        return None
    axis = difference / gap
    normal = np.asarray((-axis[1], axis[0]))
    source_alignment = float(np.dot(first.outward, axis))
    target_alignment = float(np.dot(-last.outward, axis))
    if min(source_alignment, target_alignment) < config.min_alignment:
        return None
    source_slope = float(np.dot(first.outward, normal)) / source_alignment
    target_slope = float(np.dot(-last.outward, normal)) / target_alignment
    # Compare the extrapolated cubic with the *visible* native context.  A
    # lateral jump to a parallel contour bends the extrapolation away from
    # those known pixels even if two axial tangent scores happen to agree.
    errors = []
    maxima = []
    for endpoint in (first, last):
        local = endpoint.context - start
        along = local @ axis
        across = local @ normal
        predicted = gap * _cubic_normal(along / gap, source_slope, target_slope)
        error = np.abs(across - predicted)
        # Very long context compared with a tiny gap makes cubic extension
        # ill-conditioned.  Twelve visible pixels still constrain each end;
        # additional native context was already used to reject branches.
        from_endpoint = np.linalg.norm(endpoint.context - endpoint.context[0], axis=1)
        error = error[from_endpoint <= max(config.min_context_pixels, gap * 0.5)]
        errors.append(float(np.sqrt(np.mean(error ** 2))))
        maxima.append(float(np.max(error)))
    fit_error = max(errors)
    if fit_error > config.max_context_fit_error or max(maxima) > config.max_context_fit_error * 2.5:
        return None
    steps = max(2, min(256, int(math.ceil(gap * 2)) + 1))
    t = np.linspace(0.0, 1.0, steps)
    lateral = gap * _cubic_normal(t, source_slope, target_slope)
    if float(np.max(np.abs(lateral))) > gap * config.max_curve_deviation_ratio:
        return None
    curve = start + (gap * t)[:, None] * axis + lateral[:, None] * normal
    curve[0], curve[-1] = start, end
    path_length = float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())
    if path_length > gap * 1.15:
        return None
    score = fit_error + (2.0 - source_alignment - target_alignment) * 2.0 + gap / 256.0
    return LabelGapConnector(
        first.pixel, last.pixel,
        tuple((float(x), float(y)) for x, y in curve),
        gap, path_length, score,
    )


def _crosses_long_native(mask, degree, components, compact_components, connector, config):
    """Reject contact with sustained native strokes, allowing compact marks.

    A long straight digit can be indistinguishable from a contour here and
    is deliberately rejected too.  This routine does not classify text.
    """

    first = np.asarray(connector.source_xy)
    last = np.asarray(connector.target_xy)
    curve = np.asarray(connector.points_xy)
    rounded = np.rint(curve).astype(int)
    height, width = mask.shape
    if np.any((rounded[:, 0] < 0) | (rounded[:, 0] >= width)
              | (rounded[:, 1] < 0) | (rounded[:, 1] >= height)):
        return True
    visited = set()
    contacts = set()
    for px, py in curve:
        for y in range(max(0, int(math.floor(py - 1.0))), min(height, int(math.ceil(py + 1.0)) + 1)):
            for x in range(max(0, int(math.floor(px - 1.0))), min(width, int(math.ceil(px + 1.0)) + 1)):
                if mask[y, x] and (x - px) ** 2 + (y - py) ** 2 <= 1.0 + 1e-9:
                    contacts.add((x, y))
    # Inspect a one-pixel native neighbourhood, not just rounded curve cells:
    # two diagonal strokes can intersect between their pixel centres.
    for x, y in contacts:
        pixel = (int(x), int(y))
        if pixel in visited:
            continue
        visited.add(pixel)
        # Curves necessarily touch their exact native endpoints.  Their
        # first two pixels can overlap the endpoint stroke at raster scale.
        if min(np.linalg.norm(np.asarray(pixel) - first),
               np.linalg.norm(np.asarray(pixel) - last)) <= 2.0:
            continue
        # A compact disconnected mark can occupy the hidden interval.  This
        # is a size bound, not a text classification: a large glyph or one
        # fused to a contour deliberately remains a blocking uncertainty.
        if compact_components[int(components[y, x])]:
            continue
        neighbours = _neighbours(mask, pixel)
        if len(neighbours) != 2:
            # Crossings and forks cannot be reinterpreted as compact labels.
            if len(neighbours) > 2:
                return True
            continue
        lengths = []
        for neighbour in neighbours:
            previous, current = pixel, neighbour
            length = math.hypot(current[0] - previous[0], current[1] - previous[1])
            for _ in range(int(math.ceil(config.min_context_pixels)) + 1):
                if length >= config.min_context_pixels:
                    break
                cx, cy = current
                if int(degree[cy, cx]) != 2:
                    break
                following = [p for p in _neighbours(mask, current) if p != previous]
                if len(following) != 1:
                    break
                following = following[0]
                length += math.hypot(following[0] - current[0], following[1] - current[1])
                previous, current = current, following
            lengths.append(length)
        if sum(lengths) >= config.min_context_pixels * 1.5:
            return True
    return False


def build_label_gap_connectors(
    centerline: np.ndarray,
    *,
    config: LabelGapConfig = DEFAULT_LABEL_GAP_CONFIG,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[LabelGapConnector, ...]:
    """Build bounded, mutually unique endpoint proposals in a worker.

    Inputs and the native graph remain unchanged.  The caller must keep any
    available native-only route ahead of these inferred curves, and keep the
    normal preview, explicit acceptance and Undo boundaries.  Exceeding a
    work budget returns no proposals rather than comparing an arbitrary
    truncated set of endpoints.  Cancellation never publishes a partial set.
    """

    config.validate()

    def check_cancelled():
        if cancel_check is not None and cancel_check():
            raise ObservedConnectionCancelled("Label-gap proposals were cancelled")

    check_cancelled()
    mask = np.asarray(centerline, dtype=bool)
    if mask.ndim != 2 or min(mask.shape, default=0) < 1:
        raise ValueError("centerline must be a non-empty 2D array")
    if mask.size > config.max_window_pixels:
        return ()
    if not np.any(mask):
        return ()
    degree = _degree(mask)
    ys, xs = np.nonzero(degree == 1)
    # Small text/noise endpoints are cheap to reject by their short native
    # context.  Budget the expensive pair comparisons on eligible endpoints,
    # while retaining a separate absolute bound on those context walks.
    if len(xs) > 4096:
        return ()
    height, width = mask.shape
    # A crop boundary is an artificial cut, not a text occlusion endpoint.
    interior = (xs > 1) & (xs < width - 2) & (ys > 1) & (ys < height - 2)
    pixels = list(zip(xs[interior].tolist(), ys[interior].tolist()))
    if len(pixels) < 2:
        return ()
    check_cancelled()
    try:
        from scipy.ndimage import find_objects, label
    except ImportError as exc:
        raise RuntimeError("SciPy is required for label-gap component checks") from exc
    components, _ = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    objects = find_objects(components)
    compact_components = np.zeros(len(objects) + 1, dtype=bool)
    for index, bounds in enumerate(objects, start=1):
        if bounds is not None:
            compact_components[index] = max(
                axis.stop - axis.start for axis in bounds
            ) <= config.min_context_pixels * 1.5
    endpoints = []
    for pixel in pixels:
        check_cancelled()
        endpoint = _endpoint(mask, degree, components, pixel, config)
        if endpoint is not None:
            endpoints.append(endpoint)
            if len(endpoints) > config.max_endpoints:
                return ()
    if len(endpoints) < 2:
        return ()
    pairs = []
    candidates = [[] for _ in endpoints]
    considered = 0
    for first_index, first in enumerate(endpoints):
        check_cancelled()
        for last_index in range(first_index + 1, len(endpoints)):
            last = endpoints[last_index]
            if first.component == last.component:
                continue
            dx = last.pixel[0] - first.pixel[0]
            dy = last.pixel[1] - first.pixel[1]
            if not config.min_gap_pixels ** 2 <= dx * dx + dy * dy <= config.max_gap_pixels ** 2:
                continue
            considered += 1
            if considered > config.max_candidate_pairs:
                return ()
            if considered % 32 == 0:
                check_cancelled()
            connector = _proposal(first, last, config)
            if connector is None:
                continue
            if _crosses_long_native(mask, degree, components, compact_components, connector, config):
                continue
            index = len(pairs)
            pairs.append(connector)
            candidates[first_index].append((connector.score, index))
            candidates[last_index].append((connector.score, index))
    check_cancelled()
    votes = np.zeros(len(pairs), dtype=np.uint8)
    for options in candidates:
        options.sort()
        if not options:
            continue
        if len(options) > 1 and options[1][0] - options[0][0] <= config.ambiguity_score_margin:
            continue
        votes[options[0][1]] += 1
    selected = [connector for connector, count in zip(pairs, votes) if count == 2]
    # Independently unique endpoint choices can still imply crossing hidden
    # curves.  Reject both rather than infer a contour junction in the gap.
    # Sparse raster-cell occupancy keeps this bounded by retained vertices,
    # without allocating or altering an image-sized support corridor.
    occupancy = {}
    crossing = set()
    for index, connector in enumerate(selected):
        check_cancelled()
        cells = set()
        for first, last in zip(connector.points_xy, connector.points_xy[1:]):
            # A segment's closed cell bounding box is a conservative
            # supercover.  Opposite diagonals through one pixel square must
            # not escape the crossing guard by rounding to different pixels.
            for x in range(int(math.floor(min(first[0], last[0]))),
                           int(math.floor(max(first[0], last[0]))) + 1):
                for y in range(int(math.floor(min(first[1], last[1]))),
                               int(math.floor(max(first[1], last[1]))) + 1):
                    cells.add((x, y))
        for cell in cells:
            other = occupancy.get(cell)
            if other is not None:
                crossing.update((index, other))
            else:
                occupancy[cell] = index
    return tuple(connector for index, connector in enumerate(selected) if index not in crossing)
