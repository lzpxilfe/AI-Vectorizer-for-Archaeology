"""Dependency-free centerline and topology metrics for the M1 benchmark.

Masks are rectangular, row-major iterables.  Reference paths are supplied as
``(pixels, closed)`` pairs, where every pixel is an ``(x, y)`` pair matching
the canonical ``pixel_xy`` artifact coordinate space.  A
single path pair is accepted as a convenience, as is an iterable of path pairs.

The public result contains only JSON-native values.  In particular, undefined
distances are represented by ``None`` rather than infinity or NaN.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, List, Optional, Sequence, Tuple


Pixel = Tuple[int, int]
Mask = List[List[bool]]
MAX_METRIC_PIXELS = 1024 * 1024
MAX_TOLERANCE_PX = 8.0
MAX_REFERENCE_PATH_PIXELS = 100_000
MAX_BRANCH_ZONES = 512
MAX_BRANCH_ZONE_PAIRS = 512 * 512

_ORTHOGONAL_NEIGHBORS = (
    (-1, 0),
    (0, -1),
    (0, 1),
    (1, 0),
)

_DIAGONAL_NEIGHBORS = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

# Clockwise ordering is required by the crossing-number calculation.
_CROSSING_NEIGHBORS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)


def compute_metrics(
    prediction_mask: Iterable[Iterable[object]],
    reference_mask: Iterable[Iterable[object]],
    reference_paths=(),
    tolerances: Iterable[float] = (0.0, 1.0, 2.0),
    primary_tolerance: float = 1.0,
):
    """Compute centerline, distance, tolerance, and topology metrics.

    ``cldice`` is the exact foreground Dice because benchmark artifacts are
    already one-pixel centerlines.  Symmetric mean distance is the arithmetic
    mean of the two directed means.  Symmetric p95 is the maximum of the two
    directed nearest-rank p95 values.

    Connectivity maps each ordered reference pixel to the nearest 8-connected
    prediction component no farther than ``primary_tolerance``.  Fragments are
    maximal runs mapped to one component.  Breaks are runs of failed links
    between consecutive reference pixels; closed paths include the last-to-first
    link and merge runs across that boundary.
    """

    prediction = _normalise_mask(prediction_mask, "prediction_mask")
    reference = _normalise_mask(reference_mask, "reference_mask")
    prediction_shape = _shape(prediction)
    reference_shape = _shape(reference)
    if prediction_shape != reference_shape:
        raise ValueError(
            "prediction_mask and reference_mask must have the same shape "
            "(got %r and %r)" % (prediction_shape, reference_shape)
        )
    if prediction_shape[0] * prediction_shape[1] > MAX_METRIC_PIXELS:
        raise ValueError(
            "benchmark masks may contain at most %d pixels" % MAX_METRIC_PIXELS
        )

    tolerance_values = [_normalise_tolerance(value) for value in tolerances]
    if not tolerance_values:
        raise ValueError("tolerances must contain at least one value")
    primary = _normalise_tolerance(primary_tolerance)
    paths = _normalise_paths(reference_paths, reference)
    if sum(len(pixels) for pixels, _closed in paths) > MAX_REFERENCE_PATH_PIXELS:
        raise ValueError(
            "ordered reference paths may contain at most %d pixels"
            % MAX_REFERENCE_PATH_PIXELS
        )

    prediction_count = _foreground_count(prediction)
    reference_count = _foreground_count(reference)
    intersection_count = sum(
        1
        for row_index, row in enumerate(prediction)
        for column_index, value in enumerate(row)
        if value and reference[row_index][column_index]
    )

    if prediction_count + reference_count == 0:
        cldice = 1.0
    else:
        cldice = (2.0 * intersection_count) / (
            prediction_count + reference_count
        )

    prediction_edt = _distance_transform(prediction)
    reference_edt = _distance_transform(reference)
    prediction_to_reference = _directed_distances(prediction, reference_edt)
    reference_to_prediction = _directed_distances(reference, prediction_edt)

    prediction_distance_summary = _distance_summary(
        prediction_to_reference, prediction_count
    )
    reference_distance_summary = _distance_summary(
        reference_to_prediction, reference_count
    )
    directed_means = (
        prediction_distance_summary["mean"],
        reference_distance_summary["mean"],
    )
    directed_p95s = (
        prediction_distance_summary["p95"],
        reference_distance_summary["p95"],
    )
    if None in directed_means:
        symmetric_mean = None
    else:
        symmetric_mean = (directed_means[0] + directed_means[1]) / 2.0
    if None in directed_p95s:
        symmetric_p95 = None
    else:
        symmetric_p95 = max(directed_p95s)

    tolerance_metrics = []
    for tolerance in tolerance_values:
        matched_prediction = _matched_count(
            prediction, reference_edt, tolerance
        )
        matched_reference = _matched_count(reference, prediction_edt, tolerance)
        precision, recall, f1 = _precision_recall_f1(
            matched_prediction,
            prediction_count,
            matched_reference,
            reference_count,
        )
        tolerance_metrics.append(
            {
                "tolerance": tolerance,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_prediction_pixels": matched_prediction,
                "total_prediction_pixels": prediction_count,
                "matched_reference_pixels": matched_reference,
                "total_reference_pixels": reference_count,
            }
        )

    prediction_labels, prediction_components = _component_labels(prediction)
    _, reference_components = _component_labels(reference)
    _, prediction_branch_zones, prediction_topology = (
        _skeleton_topology(prediction, len(prediction_components))
    )
    _, reference_branch_zones, reference_topology = (
        _skeleton_topology(reference, len(reference_components))
    )
    if (
        len(prediction_branch_zones) > MAX_BRANCH_ZONES
        or len(reference_branch_zones) > MAX_BRANCH_ZONES
    ):
        raise ValueError(
            "branch-zone count exceeds the benchmark limit of %d"
            % MAX_BRANCH_ZONES
        )
    matched_branch_zones = _matched_branch_zone_count(
        prediction_branch_zones,
        reference_branch_zones,
        primary,
    )
    unmatched_prediction_branch_zones = (
        len(prediction_branch_zones) - matched_branch_zones
    )
    unmatched_reference_branch_zones = (
        len(reference_branch_zones) - matched_branch_zones
    )

    connectivity = _connectivity_summary(
        paths,
        prediction_labels,
        primary,
    )

    return {
        "cldice": cldice,
        "pixel_counts": {
            "prediction": prediction_count,
            "reference": reference_count,
            "intersection": intersection_count,
        },
        "distance": {
            "prediction_to_reference": prediction_distance_summary,
            "reference_to_prediction": reference_distance_summary,
            "symmetric_mean": symmetric_mean,
            "symmetric_p95": symmetric_p95,
        },
        "tolerance_metrics": tolerance_metrics,
        "primary_tolerance": primary,
        "topology": {
            "prediction": prediction_topology,
            "reference": reference_topology,
            "matched_prediction_branch_zones": matched_branch_zones,
            "unmatched_prediction_branch_zones": (
                unmatched_prediction_branch_zones
            ),
            "unmatched_reference_branch_zones": (
                unmatched_reference_branch_zones
            ),
            "branch_match_tolerance": primary,
        },
        "connectivity": connectivity,
    }


def _normalise_mask(mask: Iterable[Iterable[object]], name: str) -> Mask:
    try:
        rows = [list(row) for row in mask]
    except TypeError as error:
        raise ValueError("%s must be an iterable of rows" % name) from error
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("%s must be rectangular" % name)
    return [[bool(value) for value in row] for row in rows]


def _shape(mask: Mask) -> Tuple[int, int]:
    return len(mask), len(mask[0]) if mask else 0


def _normalise_tolerance(value: float) -> float:
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("tolerances must be finite, non-negative numbers") from error
    if (
        not math.isfinite(tolerance)
        or tolerance < 0.0
        or tolerance > MAX_TOLERANCE_PX
    ):
        raise ValueError(
            "tolerances must be finite numbers between 0 and %.0f"
            % MAX_TOLERANCE_PX
        )
    return tolerance


def _looks_like_single_path(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[1], bool)
    )


def _normalise_paths(reference_paths, reference: Mask):
    if reference_paths is None:
        raw_paths = []
    elif _looks_like_single_path(reference_paths):
        raw_paths = [reference_paths]
    else:
        try:
            raw_paths = list(reference_paths)
        except TypeError as error:
            raise ValueError(
                "reference_paths must contain (pixels, closed) pairs"
            ) from error

    height, width = _shape(reference)
    paths = []
    for path_index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, (list, tuple)) or len(raw_path) != 2:
            raise ValueError(
                "reference path %d must be a (pixels, closed) pair" % path_index
            )
        raw_pixels, raw_closed = raw_path
        if not isinstance(raw_closed, bool):
            raise ValueError("reference path closed flags must be bool values")
        try:
            raw_pixels = list(raw_pixels)
        except TypeError as error:
            raise ValueError("reference path pixels must be iterable") from error

        pixels = []
        for pixel_index, raw_pixel in enumerate(raw_pixels):
            if not isinstance(raw_pixel, (list, tuple)) or len(raw_pixel) != 2:
                raise ValueError(
                    "reference path %d pixel %d must be (x, y)"
                    % (path_index, pixel_index)
                )
            x = _pixel_index(raw_pixel[0], path_index, pixel_index)
            y = _pixel_index(raw_pixel[1], path_index, pixel_index)
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    "reference path %d pixel %d is outside the mask"
                    % (path_index, pixel_index)
                )
            if not reference[y][x]:
                raise ValueError(
                    "reference path %d pixel %d is not foreground in "
                    "reference_mask" % (path_index, pixel_index)
                )
            # Internal algorithms use row/column indexing; only the public
            # path boundary speaks the artifact's pixel_xy convention.
            pixels.append((y, x))

        # Closed paths are often exported with the first vertex repeated.  The
        # cyclic analysis already supplies that closing edge, so remove it.
        if raw_closed and len(pixels) > 1 and pixels[0] == pixels[-1]:
            pixels.pop()
        paths.append((pixels, raw_closed))
    return paths


def _pixel_index(value, path_index: int, pixel_index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "reference path %d pixel %d coordinates must be integers"
            % (path_index, pixel_index)
        )
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "reference path %d pixel %d coordinates must be integers"
            % (path_index, pixel_index)
        ) from error
    try:
        is_integral = value == result
    except Exception:
        is_integral = False
    if not is_integral:
        raise ValueError(
            "reference path %d pixel %d coordinates must be integers"
            % (path_index, pixel_index)
        )
    return result


def _foreground_count(mask: Mask) -> int:
    return sum(1 for row in mask for value in row if value)


def _edt_1d(values: Sequence[float]) -> List[float]:
    """Felzenszwalb-Huttenlocher exact squared Euclidean transform in 1-D."""

    length = len(values)
    finite_sites = [index for index, value in enumerate(values) if math.isfinite(value)]
    if not finite_sites:
        return [math.inf] * length

    sites = [0] * len(finite_sites)
    boundaries = [0.0] * (len(finite_sites) + 1)
    sites[0] = finite_sites[0]
    boundaries[0] = -math.inf
    boundaries[1] = math.inf
    envelope_size = 0

    for site in finite_sites[1:]:
        current = sites[envelope_size]
        boundary = (
            (values[site] + site * site)
            - (values[current] + current * current)
        ) / (2.0 * (site - current))
        while boundary <= boundaries[envelope_size]:
            envelope_size -= 1
            current = sites[envelope_size]
            boundary = (
                (values[site] + site * site)
                - (values[current] + current * current)
            ) / (2.0 * (site - current))
        envelope_size += 1
        sites[envelope_size] = site
        boundaries[envelope_size] = boundary
        boundaries[envelope_size + 1] = math.inf

    result = [0.0] * length
    envelope_index = 0
    for index in range(length):
        while boundaries[envelope_index + 1] < index:
            envelope_index += 1
        site = sites[envelope_index]
        result[index] = (index - site) ** 2 + values[site]
    return result


def _distance_transform(mask: Mask) -> Optional[List[List[float]]]:
    """Return exact squared distance to foreground for every mask pixel."""

    height, width = _shape(mask)
    if height == 0 or width == 0 or _foreground_count(mask) == 0:
        return None
    first_pass = []
    for row in mask:
        first_pass.append(_edt_1d([0.0 if value else math.inf for value in row]))

    result = [[0.0] * width for _ in range(height)]
    for column in range(width):
        transformed = _edt_1d(
            [first_pass[row][column] for row in range(height)]
        )
        for row in range(height):
            result[row][column] = transformed[row]
    return result


def _directed_distances(
    source: Mask, target_edt: Optional[List[List[float]]]
) -> Optional[List[float]]:
    if target_edt is None or _foreground_count(source) == 0:
        return None
    return [
        math.sqrt(target_edt[row_index][column_index])
        for row_index, row in enumerate(source)
        for column_index, value in enumerate(row)
        if value
    ]


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(1, int(math.ceil(quantile * len(ordered))))
    return ordered[rank - 1]


def _distance_summary(distances: Optional[List[float]], source_count: int):
    if not distances:
        return {"mean": None, "p95": None, "count": source_count}
    return {
        "mean": sum(distances) / len(distances),
        "p95": _nearest_rank(distances, 0.95),
        "count": source_count,
    }


def _matched_count(
    source: Mask,
    target_edt: Optional[List[List[float]]],
    tolerance: float,
) -> int:
    if target_edt is None:
        return 0
    maximum_squared_distance = tolerance * tolerance
    return sum(
        1
        for row_index, row in enumerate(source)
        for column_index, value in enumerate(row)
        if value
        and target_edt[row_index][column_index]
        <= maximum_squared_distance + 1e-12
    )


def _precision_recall_f1(
    matched_prediction: int,
    prediction_count: int,
    matched_reference: int,
    reference_count: int,
) -> Tuple[float, float, float]:
    if prediction_count == 0 and reference_count == 0:
        return 1.0, 1.0, 1.0
    # A missing side is treated as a complete failure instead of assigning an
    # undefined or misleadingly perfect precision/recall value.
    if prediction_count == 0 or reference_count == 0:
        return 0.0, 0.0, 0.0
    precision = matched_prediction / prediction_count
    recall = matched_reference / reference_count
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _component_labels(mask: Mask):
    height, width = _shape(mask)
    labels = [[-1] * width for _ in range(height)]
    components = []
    for start_row in range(height):
        for start_column in range(width):
            if not mask[start_row][start_column] or labels[start_row][start_column] >= 0:
                continue
            label = len(components)
            labels[start_row][start_column] = label
            queue = deque([(start_row, start_column)])
            component = []
            while queue:
                row, column = queue.popleft()
                component.append((row, column))
                for neighbor_row, neighbor_column in _connected_neighbors(
                    mask, row, column
                ):
                    if (
                        mask[neighbor_row][neighbor_column]
                        and labels[neighbor_row][neighbor_column] < 0
                    ):
                        labels[neighbor_row][neighbor_column] = label
                        queue.append((neighbor_row, neighbor_column))
            components.append(component)
    return labels, components


def _connected_neighbors(mask: Mask, row: int, column: int):
    """Yield neighbors using the manifest's 8/no-corner-cut rule.

    Orthogonal foreground pixels always connect.  A diagonal foreground pixel
    connects only when both pixels at the intervening orthogonal positions are
    background.  This keeps genuine one-pixel diagonals connected without
    introducing shortcut edges around L corners and junctions.
    """

    height, width = _shape(mask)
    for row_offset, column_offset in _ORTHOGONAL_NEIGHBORS:
        neighbor_row = row + row_offset
        neighbor_column = column + column_offset
        if (
            0 <= neighbor_row < height
            and 0 <= neighbor_column < width
            and mask[neighbor_row][neighbor_column]
        ):
            yield neighbor_row, neighbor_column
    for row_offset, column_offset in _DIAGONAL_NEIGHBORS:
        neighbor_row = row + row_offset
        neighbor_column = column + column_offset
        if not (
            0 <= neighbor_row < height
            and 0 <= neighbor_column < width
            and mask[neighbor_row][neighbor_column]
        ):
            continue
        if not mask[row][neighbor_column] and not mask[neighbor_row][column]:
            yield neighbor_row, neighbor_column


def _crossing_number(mask: Mask, row: int, column: int) -> int:
    height, width = _shape(mask)
    neighbors = []
    for row_offset, column_offset in _CROSSING_NEIGHBORS:
        neighbor_row = row + row_offset
        neighbor_column = column + column_offset
        connected = (
            0 <= neighbor_row < height
            and 0 <= neighbor_column < width
            and mask[neighbor_row][neighbor_column]
        )
        if connected and row_offset != 0 and column_offset != 0:
            connected = (
                not mask[row][neighbor_column]
                and not mask[neighbor_row][column]
            )
        neighbors.append(1 if connected else 0)
    transitions = sum(
        abs(neighbors[index] - neighbors[(index + 1) % len(neighbors)])
        for index in range(len(neighbors))
    )
    return transitions // 2


def _skeleton_topology(mask: Mask, component_count: int):
    height, width = _shape(mask)
    branch_mask = [[False] * width for _ in range(height)]
    endpoints = 0
    branch_pixels = 0
    for row in range(height):
        for column in range(width):
            if not mask[row][column]:
                continue
            crossing_number = _crossing_number(mask, row, column)
            if crossing_number == 1:
                endpoints += 1
            elif crossing_number >= 3:
                branch_mask[row][column] = True
                branch_pixels += 1
    _, branch_zones = _component_labels(branch_mask)
    return (
        branch_mask,
        branch_zones,
        {
            "components": component_count,
            "branch_zones": len(branch_zones),
            "branch_pixels": branch_pixels,
            "endpoints": endpoints,
        },
    )


def _matched_branch_zone_count(
    prediction_zones: Sequence[Sequence[Pixel]],
    reference_zones: Sequence[Sequence[Pixel]],
    tolerance: float,
) -> int:
    if not prediction_zones or not reference_zones:
        return 0
    if len(prediction_zones) * len(reference_zones) > MAX_BRANCH_ZONE_PAIRS:
        raise ValueError("branch-zone candidate pair budget exceeded")
    maximum_squared_distance = tolerance * tolerance
    radius = int(math.ceil(tolerance))
    neighbor_offsets = sorted(
        (
            row_offset * row_offset + column_offset * column_offset,
            row_offset,
            column_offset,
        )
        for row_offset in range(-radius, radius + 1)
        for column_offset in range(-radius, radius + 1)
        if row_offset * row_offset + column_offset * column_offset
        <= maximum_squared_distance + 1e-12
    )
    reference_sets = [set(zone) for zone in reference_zones]
    reference_bounds = [
        (
            min(row for row, _column in zone),
            max(row for row, _column in zone),
            min(column for _row, column in zone),
            max(column for _row, column in zone),
        )
        for zone in reference_zones
    ]
    candidates = []
    for prediction_zone in prediction_zones:
        prediction_set = set(prediction_zone)
        prediction_bounds = (
            min(row for row, _column in prediction_zone),
            max(row for row, _column in prediction_zone),
            min(column for _row, column in prediction_zone),
            max(column for _row, column in prediction_zone),
        )
        zone_candidates = []
        for reference_index, reference_zone in enumerate(reference_zones):
            reference_bounds_for_zone = reference_bounds[reference_index]
            row_gap = max(
                reference_bounds_for_zone[0] - prediction_bounds[1],
                prediction_bounds[0] - reference_bounds_for_zone[1],
                0,
            )
            column_gap = max(
                reference_bounds_for_zone[2] - prediction_bounds[3],
                prediction_bounds[2] - reference_bounds_for_zone[3],
                0,
            )
            if row_gap * row_gap + column_gap * column_gap > maximum_squared_distance:
                continue
            reference_set = reference_sets[reference_index]
            if not prediction_set.isdisjoint(reference_set):
                minimum_squared_distance = 0
            else:
                smaller, larger = (
                    (prediction_set, reference_set)
                    if len(prediction_set) <= len(reference_set)
                    else (reference_set, prediction_set)
                )
                minimum_squared_distance = None
                for row, column in smaller:
                    for distance_squared, row_offset, column_offset in neighbor_offsets:
                        if (
                            minimum_squared_distance is not None
                            and distance_squared >= minimum_squared_distance
                        ):
                            break
                        if (row + row_offset, column + column_offset) in larger:
                            minimum_squared_distance = distance_squared
                            break
                    if minimum_squared_distance == 1:
                        break
            if minimum_squared_distance is not None:
                zone_candidates.append(
                    (minimum_squared_distance, reference_index)
                )
        candidates.append(sorted(zone_candidates))

    # Deterministic augmenting-path matching gives maximum cardinality while
    # ensuring that a reference junction cannot explain multiple predictions.
    reference_matches = {}

    def augment(prediction_index: int, seen_references: set) -> bool:
        for _, reference_index in candidates[prediction_index]:
            if reference_index in seen_references:
                continue
            seen_references.add(reference_index)
            previous_prediction = reference_matches.get(reference_index)
            if previous_prediction is None or augment(
                previous_prediction, seen_references
            ):
                reference_matches[reference_index] = prediction_index
                return True
        return False

    matched = 0
    for prediction_index in range(len(prediction_zones)):
        if augment(prediction_index, set()):
            matched += 1
    return matched


def _nearest_component(
    pixel: Pixel,
    labels: List[List[int]],
    tolerance: float,
) -> Optional[int]:
    height = len(labels)
    width = len(labels[0]) if labels else 0
    if height == 0 or width == 0:
        return None
    row, column = pixel
    radius = int(math.ceil(tolerance))
    maximum_squared_distance = tolerance * tolerance
    best = None
    for candidate_row in range(max(0, row - radius), min(height, row + radius + 1)):
        for candidate_column in range(
            max(0, column - radius), min(width, column + radius + 1)
        ):
            label = labels[candidate_row][candidate_column]
            if label < 0:
                continue
            distance_squared = (candidate_row - row) ** 2 + (
                candidate_column - column
            ) ** 2
            if distance_squared > maximum_squared_distance + 1e-12:
                continue
            candidate = (distance_squared, label, candidate_row, candidate_column)
            if best is None or candidate < best:
                best = candidate
    return None if best is None else best[1]


def _fragment_count(labels: Sequence[Optional[int]], closed: bool) -> int:
    if not labels:
        return 0
    if not closed:
        return sum(
            1
            for index, label in enumerate(labels)
            if label is not None
            and (index == 0 or labels[index - 1] != label)
        )
    starts = sum(
        1
        for index, label in enumerate(labels)
        if label is not None and labels[index - 1] != label
    )
    if starts == 0 and labels[0] is not None:
        return 1
    return starts


def _longest_fragment(labels: Sequence[Optional[int]], closed: bool) -> int:
    if not labels or all(label is None for label in labels):
        return 0
    if closed and all(label == labels[0] and label is not None for label in labels):
        return len(labels)
    sequence = list(labels) + (list(labels) if closed else [])
    longest = 0
    current = 0
    previous = None
    for label in sequence:
        if label is not None and label == previous:
            current += 1
        elif label is not None:
            current = 1
        else:
            current = 0
        previous = label
        longest = max(longest, min(current, len(labels)))
    return longest


def _linear_true_runs(values: Sequence[bool]) -> int:
    return sum(
        1
        for index, value in enumerate(values)
        if value and (index == 0 or not values[index - 1])
    )


def _cyclic_true_runs(values: Sequence[bool]) -> int:
    if not values or not any(values):
        return 0
    if all(values):
        return 1
    return sum(
        1
        for index, value in enumerate(values)
        if value and not values[index - 1]
    )


def _break_count(labels: Sequence[Optional[int]], closed: bool) -> int:
    if len(labels) < 2 or all(label is None for label in labels):
        return 0
    if closed:
        working_labels = list(labels)
    else:
        # Endpoint omissions on an open trace affect coverage but are not
        # internal breaks.  Restrict link analysis to the first and last
        # recovered samples.
        first_matched = next(
            index for index, label in enumerate(labels) if label is not None
        )
        last_matched = len(labels) - 1 - next(
            index
            for index, label in enumerate(reversed(labels))
            if label is not None
        )
        working_labels = list(labels[first_matched : last_matched + 1])
        if len(working_labels) < 2:
            return 0

    edge_count = len(working_labels) if closed else len(working_labels) - 1
    failed_links = []
    for index in range(edge_count):
        next_index = (index + 1) % len(working_labels)
        failed_links.append(
            working_labels[index] is None
            or working_labels[next_index] is None
            or working_labels[index] != working_labels[next_index]
        )
    if closed:
        return _cyclic_true_runs(failed_links)
    return _linear_true_runs(failed_links)


def _connectivity_summary(
    paths: Sequence[Tuple[Sequence[Pixel], bool]],
    prediction_labels: List[List[int]],
    tolerance: float,
):
    path_results = []
    for path_index, (pixels, closed) in enumerate(paths):
        labels = [
            _nearest_component(pixel, prediction_labels, tolerance)
            for pixel in pixels
        ]
        total_pixels = len(labels)
        matched_pixels = sum(label is not None for label in labels)
        fragments = _fragment_count(labels, closed)
        breaks = _break_count(labels, closed)
        missed = total_pixels > 0 and matched_pixels == 0
        fragment_excess = max(fragments - 1, 0)
        longest_fragment_pixels = _longest_fragment(labels, closed)
        path_results.append(
            {
                "index": path_index,
                "closed": closed,
                "total_pixels": total_pixels,
                "matched_pixels": matched_pixels,
                "coverage_ratio": (
                    matched_pixels / total_pixels if total_pixels else None
                ),
                "missed": missed,
                "fragments": fragments,
                "fragment_excess": fragment_excess,
                "breaks": breaks,
                "longest_fragment_pixels": longest_fragment_pixels,
                "longest_fragment_ratio": (
                    longest_fragment_pixels / total_pixels if total_pixels else None
                ),
            }
        )

    total_pixels = sum(path["total_pixels"] for path in path_results)
    matched_pixels = sum(path["matched_pixels"] for path in path_results)
    sum_longest = sum(path["longest_fragment_pixels"] for path in path_results)
    coverage_values = [
        path["coverage_ratio"]
        for path in path_results
        if path["coverage_ratio"] is not None
    ]
    longest_values = [
        path["longest_fragment_ratio"]
        for path in path_results
        if path["longest_fragment_ratio"] is not None
    ]
    return {
        "paths": path_results,
        "summary": {
            "path_count": len(path_results),
            "open_paths": sum(not path["closed"] for path in path_results),
            "closed_paths": sum(path["closed"] for path in path_results),
            "total_pixels": total_pixels,
            "matched_pixels": matched_pixels,
            "coverage_ratio": (
                matched_pixels / total_pixels if total_pixels else None
            ),
            "fragments": sum(path["fragments"] for path in path_results),
            "fragment_excess": sum(
                path["fragment_excess"] for path in path_results
            ),
            "breaks": sum(path["breaks"] for path in path_results),
            "missed_paths": sum(path["missed"] for path in path_results),
            "sum_longest_fragment_pixels": sum_longest,
            "longest_fragment_ratio": (
                sum_longest / total_pixels if total_pixels else None
            ),
            "mean_path_coverage_ratio": (
                sum(coverage_values) / len(coverage_values)
                if coverage_values
                else None
            ),
            "mean_path_longest_fragment_ratio": (
                sum(longest_values) / len(longest_values)
                if longest_values
                else None
            ),
        },
    }


__all__ = ["compute_metrics"]
