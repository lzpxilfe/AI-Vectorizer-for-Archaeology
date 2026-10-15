"""Signed native evidence for explicitly selected broad contour turns."""

import math

import numpy as np
import pytest

from ai_vectorizer.core.confirmed_gap_evidence import (
    confirmed_gap_curve_is_clear, sample_confirmed_gap_tangents,
)
from ai_vectorizer.core.line_evidence import LineEvidence


def _evidence(mask):
    return LineEvidence(mask.astype(np.float32), mask)


def _u_fragments():
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:51, 30] = True
    mask[20:51, 60] = True
    return mask


def test_broad_turn_keeps_signed_outward_directions_without_chord_cone():
    mask = _u_fragments()
    tangents = sample_confirmed_gap_tangents(_evidence(mask), (30, 50), (60, 50))
    np.testing.assert_allclose(tangents, ((0, 1), (0, 1)), atol=1e-7)


def test_endpoint_sampling_does_not_require_click_to_equal_terminal_pixel():
    mask = _u_fragments()
    start, end = (30.0, 49.0), (60.0, 48.0)
    original = _evidence(mask)
    before = original.centerline.tobytes()
    np.testing.assert_allclose(
        sample_confirmed_gap_tangents(original, start, end), ((0, 1), (0, 1)), atol=1e-7,
    )
    assert start == (30.0, 49.0) and end == (60.0, 48.0)
    assert original.centerline.tobytes() == before


def test_nearby_contour_cannot_supply_the_clicked_interior_endpoint():
    mask = _u_fragments()
    mask[15:70, 64] = True
    assert sample_confirmed_gap_tangents(_evidence(mask), (30, 50), (64, 49)) is None


def test_nearest_short_glyph_cannot_borrow_long_neighbour_context():
    mask = _u_fragments()
    mask[47:51, 64] = True
    assert sample_confirmed_gap_tangents(_evidence(mask), (30, 50), (64, 50)) is None


def test_equally_near_competing_strokes_are_ambiguous_even_when_parallel():
    mask = _u_fragments()
    mask[20:51, 64] = True
    assert sample_confirmed_gap_tangents(_evidence(mask), (30, 50), (62, 50)) is None


def test_farther_explicit_pair_cannot_leap_over_a_nearer_credible_endpoint():
    mask = _u_fragments()
    mask[20:51, 66] = True
    evidence = _evidence(mask)
    assert sample_confirmed_gap_tangents(evidence, (30, 50), (60, 50)) is not None
    assert sample_confirmed_gap_tangents(evidence, (30, 50), (66, 50)) is None
    assert sample_confirmed_gap_tangents(evidence, (66, 50), (30, 50)) is None


def test_short_inner_glyph_does_not_become_a_competing_contour_endpoint():
    mask = _u_fragments()
    mask[47:51, 54] = True
    assert sample_confirmed_gap_tangents(_evidence(mask), (30, 50), (60, 50)) is not None


def test_nested_inner_u_endpoint_outside_anchor_corridor_is_not_a_skipped_end():
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:51, 30] = True
    mask[20:41, 60] = True
    mask[20:45, 35] = True
    # The nearby inner endpoint is earlier along the gap chord but over
    # four pixels to its side. A length-scaled cone would confuse nesting
    # with bypassing an endpoint directly between the confirmed pair.
    assert sample_confirmed_gap_tangents(_evidence(mask), (30, 50), (60, 40)) is not None


def test_branch_within_required_native_context_is_not_an_endpoint_tangent():
    mask = _u_fragments()
    mask[46, 26:35] = True
    assert sample_confirmed_gap_tangents(_evidence(mask), (30, 50), (60, 50)) is None


def test_trusted_incoming_heading_only_rejects_disagreement():
    evidence = _evidence(_u_fragments())
    assert sample_confirmed_gap_tangents(evidence, (30, 50), (60, 50), incoming_direction=(0, 1))
    assert sample_confirmed_gap_tangents(evidence, (30, 50), (60, 50), incoming_direction=(0, -1)) is None
    mask = _u_fragments()
    mask[20:50, 30] = False
    assert sample_confirmed_gap_tangents(_evidence(mask), (30, 50), (60, 50), incoming_direction=(0, 1)) is None


def test_digital_bends_do_not_look_like_foreground_forks():
    mask = np.zeros((96, 96), dtype=bool)
    for x in range(15, 41):
        mask[int(round(30 + (x - 15) * 0.25)), x] = True
    mask[10:40, 65] = True
    tangents = sample_confirmed_gap_tangents(_evidence(mask), (40, 36), (65, 39))
    assert tangents is not None
    assert tangents[0][0] > 0.85 and tangents[0][1] > 0.05
    np.testing.assert_allclose(tangents[1], (0, 1), atol=1e-7)


@pytest.mark.parametrize("rotation", range(4))
def test_rotation_preserves_signed_orientation(rotation):
    mask = np.rot90(_u_fragments(), rotation)
    def transform(point):
        x, y = point
        for _ in range(rotation):
            x, y = y, 95 - x
        return x, y
    direction = (0.0, 1.0)
    for _ in range(rotation):
        direction = direction[1], -direction[0]
    tangents = sample_confirmed_gap_tangents(_evidence(mask), transform((30, 50)), transform((60, 50)))
    np.testing.assert_allclose(tangents, (direction, direction), atol=1e-7)


@pytest.mark.parametrize("arguments", [
    (None, (30, 50), (60, 50)),
    (_evidence(_u_fragments()), (math.nan, 50), (60, 50)),
    (_evidence(_u_fragments()), (30, 50), (30, 50)),
    (_evidence(_u_fragments()), (-1, 50), (60, 50)),
    (_evidence(_u_fragments()), (True, 50), (60, 50)),
])
def test_invalid_inputs_raise_without_changing_evidence(arguments):
    with pytest.raises(ValueError):
        sample_confirmed_gap_tangents(*arguments)


def test_bad_heading_and_cache_boundary_endpoint_are_rejected():
    evidence = _evidence(_u_fragments())
    with pytest.raises(ValueError, match="non-zero"):
        sample_confirmed_gap_tangents(evidence, (30, 50), (60, 50), incoming_direction=(0, 0))
    mask = _u_fragments()
    mask[20:51, 1] = True
    assert sample_confirmed_gap_tangents(_evidence(mask), (1, 50), (60, 50)) is None


def test_curve_clearance_blocks_long_foreign_native_strokes():
    mask = np.zeros((80, 80), dtype=bool)
    mask[5:61, 35] = True
    curve = tuple((float(x), 30.0) for x in range(10, 61))
    assert not confirmed_gap_curve_is_clear(_evidence(mask), curve)


def test_curve_clearance_catches_diagonal_crossing_between_native_cells():
    mask = np.zeros((80, 80), dtype=bool)
    for coordinate in range(5, 66):
        mask[coordinate, coordinate] = True
    curve = ((10.0, 59.0), (59.0, 10.0))
    assert not confirmed_gap_curve_is_clear(_evidence(mask), curve)


def test_compact_loop_and_fused_distant_network_do_not_supply_long_local_arms():
    mask = np.zeros((80, 80), dtype=bool)
    mask[26:35, 30] = True
    mask[26:35, 38] = True
    mask[26, 30:39] = True
    mask[34, 30:39] = True
    curve = tuple((float(x), 30.0) for x in range(10, 61))
    assert confirmed_gap_curve_is_clear(_evidence(mask), curve)
    # A far connection makes the component large, but its tightly curved
    # local loop still does not become an uninterrupted contour crossing.
    mask[26, 38:75] = True
    assert confirmed_gap_curve_is_clear(_evidence(mask), curve)


def test_short_broken_but_elongated_contour_is_not_treated_as_a_glyph():
    mask = np.zeros((80, 80), dtype=bool)
    for x, y in [(25, 24), (25, 25), (26, 26), (27, 27), (28, 27), (29, 28),
                 (30, 28), (31, 29), (32, 29), (33, 29), (34, 29), (35, 29)]:
        mask[y, x] = True
    curve = tuple((float(x), 30.0) for x in range(10, 61))
    assert not confirmed_gap_curve_is_clear(_evidence(mask), curve)


def test_curve_clearance_does_not_reject_its_exact_native_end_caps():
    mask = np.zeros((80, 80), dtype=bool)
    mask[30, 1:12] = True
    mask[30, 59:76] = True
    curve = tuple((float(x), 30.0) for x in range(10, 61))
    assert confirmed_gap_curve_is_clear(_evidence(mask), curve)


@pytest.mark.parametrize("curve", [
    (), ((math.nan, 20), (40, 20)), ((-1, 20), (40, 20)),
    ((0, 0), (400, 400)), ((1, 2, 3), (4, 5, 6)),
])
def test_curve_clearance_fails_closed_on_invalid_or_unbounded_proposals(curve):
    assert not confirmed_gap_curve_is_clear(_evidence(np.zeros((512, 512), dtype=bool)), curve)
