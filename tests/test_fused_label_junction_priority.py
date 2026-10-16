"""Fused text evidence cannot make a distant native junction authoritative."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from ai_vectorizer.core.label_occlusion import detect_label_occlusions
from ai_vectorizer.core.observed_connection import ObservedConnectionCancelled
from ai_vectorizer.core.occluded_connection import build_occluded_connection


def _round_counters(count=3, *, spacing=7):
    image = np.full((80, 100), 255, dtype=np.uint8)
    for index in range(count):
        x = 34 + index * spacing
        image[30:36, x:x + 5] = 0
        image[32:34, x + 1:x + 3] = 255
    # One dark scan connection makes all glyphs a single ink component.
    image[30, 34:34 + (count - 1) * spacing + 5] = 0
    return image


@pytest.mark.parametrize("turns", range(4))
def test_three_round_counters_in_one_compact_component_are_a_label(turns):
    image = np.rot90(_round_counters(), turns)
    regions = detect_label_occlusions(image)
    assert len(regions) == 1
    region = regions[0]
    assert region.counter_count == 3
    assert not region.mask.flags.writeable
    # Square2x2 white counters cannot pass the original elongated-pair rule.
    # Their independent enclosure and common baseline supply the extra evidence.
    assert len(region.polygon_xy) >= 4


@pytest.mark.parametrize("turns", range(4))
def test_single_two_counter_glyph_does_not_become_a_label_group(turns):
    assert detect_label_occlusions(np.rot90(_round_counters(count=2), turns)) == ()


def test_a_long_connected_grid_cannot_supply_a_fused_label_parent():
    image = _round_counters()
    image[30, :] = 0
    assert detect_label_occlusions(image) == ()


def test_irregularly_spaced_counters_decline():
    image = np.full((80, 100), 255, dtype=np.uint8)
    for x in (20, 27, 49):
        image[30:36, x:x + 5] = 0
        image[32:34, x + 1:x + 3] = 255
    image[30, 20:54] = 0
    assert detect_label_occlusions(image) == ()


def test_fused_detection_cancellation_never_publishes_a_partial_group():
    calls = []

    def cancel():
        calls.append(True)
        return len(calls) >= 3

    with pytest.raises(ObservedConnectionCancelled):
        detect_label_occlusions(_round_counters(), cancel_check=cancel)


@pytest.mark.parametrize("groups, expected_regions", ((170, 64), (171, 0)))
def test_dense_round_counters_share_the_total_candidate_budget(groups, expected_regions):
    # Each scan fragment has three separately enclosed square counters. They
    # bypass the elongated-hole rule, but must still consume its bounded
    # candidate budget: 510 is allowed; 513 abandons the entire noisy window.
    image = np.full((146, 552), 255, dtype=np.uint8)
    for index in range(groups):
        y = 10 + (index // 19) * 14
        left = 10 + (index % 19) * 28
        for offset in (0, 7, 14):
            x = left + offset
            image[y:y + 6, x:x + 5] = 0
            image[y + 2:y + 4, x + 1:x + 3] = 255
        image[y, left:left + 19] = 0
    assert len(detect_label_occlusions(image)) == expected_regions


def _graph(shape):
    height, width = shape
    pixels = np.arange(height * width).reshape(shape)
    rows, columns, costs = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        first = pixels[y0:y1, x0:x1].ravel()
        last = pixels[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        rows.extend((first, last))
        columns.extend((last, first))
        costs.extend((np.full(len(first), np.hypot(dx, dy)),) * 2)
    return sparse.csr_matrix(
        (np.concatenate(costs), (np.concatenate(rows), np.concatenate(columns))),
        shape=(height * width, height * width),
    )


def _native_scene(*, junction=False, arm_branch=False, turns=0):
    native = np.zeros((96, 96), dtype=bool)
    native[30:65, 40] = True
    native[30:61, 64] = True
    native[30, 40:65] = True
    native[30:65, 72] = True
    if junction:
        native[42, 40:65] = True
    if arm_branch:
        native[54, 40:43] = True
    image = np.full(native.shape, 255, dtype=np.uint8)
    image[native] = 0
    for x in (47, 56):
        image[63:75, x:x + 7] = 0
        image[65:73, x + 2:x + 5] = 255
    root, target, neighbor = (40, 46), (64, 46), (72, 46)
    for _ in range(turns):
        native, image = np.rot90(native), np.rot90(image)
        root, target, neighbor = [(y, 95 - x) for x, y in (root, target, neighbor)]
    regions = detect_label_occlusions(image)
    tree = build_occluded_connection(
        _graph(native.shape), native, root, (0, 0), regions,
        image=image, sparse=sparse, dijkstra=dijkstra,
    )
    assert tree is not None
    return tree, root, target, neighbor


@pytest.mark.parametrize("turns", range(4))
def test_junction_network_does_not_outvote_unbranched_true_gap_arms(turns):
    tree, _root, target, neighbor = _native_scene(junction=True, turns=turns)
    outside = tree.outside_connection.trace(target)
    candidate = tree.trace(target)
    assert outside is not None and candidate is not None
    assert tree._has_native_junction(outside) is True
    assert tree.outside_route(target) is None
    assert tree.accepts_display(candidate)
    assert tree.trace(neighbor) is None
    assert not tree.outside_degree.flags.writeable


@pytest.mark.parametrize("turns", range(4))
def test_unbranched_continuous_outside_contour_keeps_priority(turns):
    tree, _root, target, _neighbor = _native_scene(turns=turns)
    outside = tree.outside_connection.trace(target)
    assert outside is not None and tree.trace(target) is not None
    assert tree._has_native_junction(outside) is False
    assert tree.outside_route(target) == outside


def test_candidate_visible_arm_with_a_branch_cannot_displace_outside_route():
    tree, _root, target, _neighbor = _native_scene(junction=True, arm_branch=True)
    outside = tree.outside_connection.trace(target)
    candidate = tree.trace(target)
    assert outside is not None and candidate is not None
    assert tree._has_native_junction(outside) is True
    assert tree.outside_route(target) == outside


def test_unit_diagonal_must_not_hide_its_shared_native_t_junction():
    tree, _root, _target, _neighbor = _native_scene(junction=True)
    # The genuine8-neighbor route can step around the pruned degree3 center.
    diagonal = [(40.0, 43.0), (41.0, 42.0)]
    assert tree.outside_degree[42, 40] == 3
    assert tree.outside_degree[43, 40] == tree.outside_degree[42, 41] == 2
    assert tree._has_native_junction(diagonal) is True
    # Only shared orthogonal corners of a unit diagonal qualify; a longer
    # virtual gap or a merely nearby branch may not manufacture that result.
    assert tree._has_native_junction([(40.0, 44.0), (42.0, 42.0)]) is False


def test_legacy_snapshot_without_prepared_degrees_keeps_outside_priority():
    tree, _root, target, _neighbor = _native_scene(junction=True)
    legacy = replace(tree, outside_degree=None)
    assert legacy.outside_route(target) == tree.outside_connection.trace(target)
