"""Natural inferred gaps, explicit uncertainty and adjacent-contour rejection."""

from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from ai_vectorizer.core.label_gap import (
    DEFAULT_LABEL_GAP_CONFIG,
    LabelGapConfig,
    build_label_gap_connectors,
)
from ai_vectorizer.core.observed_connection import ObservedConnectionCancelled


pytest.importorskip("scipy.ndimage")


def _curved_label_scene():
    mask = np.zeros((256, 256), dtype=bool)
    for x in range(40, 161):
        y = round(60 + 0.003 * (x - 100) ** 2)
        if x <= 74 or x >= 126:
            mask[y, x] = True
    # A continuous parallel contour must remain a separate line.
    for x in range(30, 171):
        mask[round(60 + 0.003 * (x - 100) ** 2) + 14, x] = True
    # Compact disconnected number-like strokes occupy the missing interval.
    mask[56:65, 98] = True
    mask[64, 96:101] = True
    mask[55:67, 108] = True
    mask[55:67, 114] = True
    mask[55, 108:115] = True
    mask[66, 108:115] = True
    return mask


def _straight_scene():
    mask = np.zeros((200, 220), dtype=bool)
    mask[70, 20:71] = True
    mask[70, 121:201] = True
    return mask


def _pairs(connectors):
    return {frozenset((connector.source_xy, connector.target_xy)) for connector in connectors}


def test_curved_occlusion_keeps_observed_endpoints_and_natural_sag():
    mask = _curved_label_scene()
    original = mask.copy()
    connectors = build_label_gap_connectors(mask)
    assert len(connectors) == 1
    connector = connectors[0]
    assert connector.source_xy == (74, 62)
    assert connector.target_xy == (126, 62)
    points = np.asarray(connector.points_xy)
    assert tuple(points[0]) == (74.0, 62.0)
    assert tuple(points[-1]) == (126.0, 62.0)
    assert np.all(np.diff(points[:, 0]) > 0)
    # This synthetic arc has known geometry; actual hidden map fragments do
    # not.  Straight interpolation has a two-pixel midpoint error here.
    true_y = 60 + 0.003 * (points[:, 0] - 100) ** 2
    assert np.max(np.abs(points[:, 1] - true_y)) < 0.5
    assert min(points[:, 1]) < 60.5
    assert 1 < connector.path_length_pixels / connector.gap_length_pixels < 1.02
    # Near-end derivative directions follow the observed contour, not text.
    assert points[1, 1] < points[0, 1]
    assert points[-1, 1] > points[-2, 1]
    np.testing.assert_array_equal(mask, original)
    with pytest.raises(FrozenInstanceError):
        connector.score = 0
    with pytest.raises(TypeError):
        connector.points_xy[0] = (0.0, 0.0)


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("reflection", (False, True))
def test_curve_pairing_is_equivariant_under_quarter_turns_and_reflection(turns, reflection):
    mask = _curved_label_scene()
    expected = np.zeros(mask.shape, dtype=bool)
    expected[62, (74, 126)] = True
    if reflection:
        mask, expected = np.fliplr(mask), np.fliplr(expected)
    mask, expected = np.rot90(mask, turns), np.rot90(expected, turns)
    connectors = build_label_gap_connectors(mask)
    assert len(connectors) == 1
    ys, xs = np.nonzero(expected)
    assert _pairs(connectors) == {frozenset(zip(xs.tolist(), ys.tolist()))}


def test_parallel_contours_interrupted_by_the_same_label_do_not_cross_pair():
    mask = np.zeros((200, 220), dtype=bool)
    for y in (60, 74):
        mask[y, 20:71] = True
        mask[y, 121:201] = True
    connectors = build_label_gap_connectors(mask)
    assert _pairs(connectors) == {
        frozenset(((70, 60), (121, 60))),
        frozenset(((70, 74), (121, 74))),
    }
    assert all(np.ptp(np.asarray(connector.points_xy)[:, 1]) == 0 for connector in connectors)


def test_nearer_neighbour_does_not_displace_the_tangential_continuation():
    mask = _straight_scene()
    # The wrong contour begins 20px sooner and only six pixels below.
    mask[76, 101:201] = True
    assert _pairs(build_label_gap_connectors(mask)) == {frozenset(((70, 70), (121, 70)))}


def test_equally_plausible_parallel_endpoints_are_declined():
    mask = _straight_scene()
    mask[70, 121:201] = False
    mask[68, 121:201] = True
    mask[72, 121:201] = True
    assert build_label_gap_connectors(mask) == ()


def test_short_glyph_stroke_cannot_supply_a_contour_tangent():
    mask = _straight_scene()
    mask[70, 129:201] = False
    assert build_label_gap_connectors(mask) == ()


def test_junction_close_to_an_endpoint_is_not_hidden_by_a_fitted_tangent():
    mask = _straight_scene()
    mask[60:81, 65] = True
    assert build_label_gap_connectors(mask) == ()


def test_continuous_native_u_is_never_replaced_by_an_inferred_shortcut():
    mask = np.zeros((160, 160), dtype=bool)
    mask[20:121, 40] = True
    mask[20:121, 100] = True
    mask[120, 40:101] = True
    assert build_label_gap_connectors(mask) == ()


@pytest.mark.parametrize("diagonal", (False, True))
def test_crossing_an_uninterrupted_long_contour_is_rejected(diagonal):
    mask = _straight_scene()
    if diagonal:
        # Cross between the missing line's raster centres, not just exactly
        # through a sampled curve pixel.
        for i in range(-35, 36):
            mask[70 + i, 96 + i] = True
    else:
        mask[20:121, 96] = True
    assert build_label_gap_connectors(mask) == ()


def test_large_or_contour_fused_mark_remains_uncertain_and_blocks_the_curve():
    mask = _straight_scene()
    mask[50:91, 96] = True
    mask[50, 96:104] = True
    assert build_label_gap_connectors(mask) == ()


@pytest.mark.parametrize("diagonal", (False, True))
def test_individually_plausible_hidden_curves_cannot_create_a_new_intersection(diagonal):
    mask = np.zeros((200, 220), dtype=bool)
    if diagonal:
        for x in range(20, 151):
            if x <= 60 or x >= 110:
                mask[x, x] = True
                mask[169 - x, x] = True
        # The inferred diagonals meet at (84.5,84.5), between native centres.
    else:
        mask[70, 20:71] = True
        mask[70, 121:201] = True
        mask[10:45, 96] = True
        mask[96:161, 96] = True
    assert build_label_gap_connectors(mask) == ()


def test_crop_boundary_is_not_treated_as_a_text_occlusion():
    mask = np.zeros((80, 160), dtype=bool)
    mask[1, 20:61] = True
    mask[1, 91:141] = True
    assert build_label_gap_connectors(mask) == ()


@pytest.mark.parametrize("distance", (3, 97))
def test_gap_length_bounds_do_not_get_clipped_into_an_allowed_curve(distance):
    mask = np.zeros((120, 260), dtype=bool)
    mask[60, 20:71] = True
    mask[60, 70 + distance:241] = True
    assert build_label_gap_connectors(mask) == ()


def test_read_only_native_mask_is_accepted_without_mutation():
    mask = _straight_scene()
    mask.setflags(write=False)
    assert len(build_label_gap_connectors(mask)) == 1


def test_endpoint_and_pair_work_budgets_fail_closed_without_truncation():
    mask = _straight_scene()
    assert build_label_gap_connectors(mask, config=replace(DEFAULT_LABEL_GAP_CONFIG, max_endpoints=1)) == ()
    for y in (40, 100):
        mask[y, 20:71] = True
        mask[y, 121:201] = True
    assert build_label_gap_connectors(
        mask, config=replace(DEFAULT_LABEL_GAP_CONFIG, max_candidate_pairs=1)
    ) == ()
    assert build_label_gap_connectors(
        mask, config=replace(DEFAULT_LABEL_GAP_CONFIG, max_window_pixels=100)
    ) == ()


def test_cancellation_after_work_begins_never_returns_partial_connectors():
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 5
    with pytest.raises(ObservedConnectionCancelled):
        build_label_gap_connectors(_straight_scene(), cancel_check=cancelled)
    assert calls == 5


def test_blank_mask_and_masks_without_endpoints_return_no_proposals():
    assert build_label_gap_connectors(np.zeros((100, 100), dtype=bool)) == ()
    assert build_label_gap_connectors(np.ones((100, 100), dtype=bool)) == ()


@pytest.mark.parametrize("kwargs", (
    {"min_gap_pixels": 1}, {"max_gap_pixels": 97},
    {"min_context_pixels": 7}, {"context_pixels": 33},
    {"min_alignment": 0.6}, {"max_context_turn_radians": math.pi},
    {"max_context_fit_error": 0}, {"max_curve_deviation_ratio": 0.9},
    {"ambiguity_score_margin": 0}, {"max_endpoints": True},
    {"max_candidate_pairs": 4097}, {"max_window_pixels": 0},
    {"max_gap_pixels": float("nan")},
))
def test_configuration_rejects_unbounded_or_nonfinite_options(kwargs):
    with pytest.raises(ValueError):
        build_label_gap_connectors(_straight_scene(), config=LabelGapConfig(**kwargs))


@pytest.mark.parametrize("mask", (np.zeros((0, 3)), np.zeros(3), np.zeros((2, 3, 4))))
def test_invalid_shape_is_rejected(mask):
    with pytest.raises(ValueError):
        build_label_gap_connectors(mask)
