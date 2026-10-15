"""Exact user anchors, signed broad-turn tangents and bounded explicit curves."""

from dataclasses import FrozenInstanceError, replace
import math

import pytest

from ai_vectorizer.core.confirmed_gap_curve import (
    DEFAULT_CONFIRMED_GAP_CURVE_CONFIG,
    ConfirmedGapCurveConfig,
    ConfirmedGapCurveError,
    _self_intersects,
    build_confirmed_gap_curve,
)


def _direction(first, last):
    dx, dy = last[0] - first[0], last[1] - first[1]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def test_broad_u_respects_exact_confirmed_endpoints_and_signed_tangents():
    curve = build_confirmed_gap_curve((30, 30), (70, 30), (0, 1), (0, 1))
    assert curve.points_xy[0] == (30.0, 30.0)
    assert curve.points_xy[-1] == (70.0, 30.0)
    assert curve.source_outward == (0.0, 1.0)
    assert curve.target_outward == (0.0, 1.0)
    assert _direction(*curve.points_xy[:2])[1] > 0.999
    assert _direction(*curve.points_xy[-2:])[1] < -0.999
    assert 49.9 < max(y for _x, y in curve.points_xy) <= 50
    assert 1.58 < curve.detour_ratio < 1.60
    assert not _self_intersects(curve.points_xy)
    assert len(curve.points_xy) <= 512
    with pytest.raises(FrozenInstanceError):
        curve.path_length_pixels = 0


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("reflection", (False, True))
def test_broad_turn_is_equivariant_without_choosing_different_endpoints(turns, reflection):
    def transform(point):
        x, y = point
        if reflection:
            x = -x
        for _ in range(turns):
            x, y = -y, x
        return x, y
    original = build_confirmed_gap_curve((30, 30), (70, 30), (0, 1), (0, 1))
    transformed = build_confirmed_gap_curve(
        transform((30, 30)), transform((70, 30)), transform((0, 1)), transform((0, 1)),
    )
    assert transformed.gap_length_pixels == original.gap_length_pixels
    assert transformed.path_length_pixels == pytest.approx(original.path_length_pixels)
    for actual, expected in zip(transformed.points_xy, map(transform, original.points_xy)):
        assert actual == pytest.approx(expected)


def test_source_may_face_partly_against_chord_when_user_has_confirmed_the_pair():
    # A hidden broad bend can require the source to continue down before
    # turning up toward the explicitly selected target.  No hidden raster
    # points are used or asserted as truth in this geometry-only fixture.
    curve = build_confirmed_gap_curve((222, 294), (243, 285), (0.1, 1), (0, 1))
    assert curve.points_xy[0] == (222.0, 294.0)
    assert curve.points_xy[-1] == (243.0, 285.0)
    assert max(y for _x, y in curve.points_xy) > 301
    assert curve.detour_ratio < 1.8
    assert _direction(*curve.points_xy[:2])[1] > 0.99
    assert _direction(*curve.points_xy[-2:])[1] < -0.99


def test_signed_target_outward_is_not_mistaken_for_forward_arrival_direction():
    curve = build_confirmed_gap_curve((10, 20), (60, 20), (1, 0), (-1, 0))
    assert all(y == pytest.approx(20) for _x, y in curve.points_xy)
    assert curve.path_length_pixels == pytest.approx(50)
    assert all(b[0] > a[0] for a, b in zip(curve.points_xy, curve.points_xy[1:]))
    with pytest.raises(ConfirmedGapCurveError, match="directions point away"):
        build_confirmed_gap_curve((10, 20), (60, 20), (1, 0), (1, 0))


@pytest.mark.parametrize("first,last", (((-1, 0), (1, 0)), ((-1, 0), (-1, 0)), ((1, 0), (1, 0))))
def test_directions_pointing_away_cannot_be_clipped_into_a_plausible_curve(first, last):
    with pytest.raises(ConfirmedGapCurveError, match="directions point away"):
        build_confirmed_gap_curve((0, 0), (30, 0), first, last)


def test_geometry_bounds_reject_excessive_detour_and_sag():
    with pytest.raises(ConfirmedGapCurveError, match="detour"):
        build_confirmed_gap_curve(
            (10, 10), (50, 10), (0, 1), (0, 1),
            config=replace(DEFAULT_CONFIRMED_GAP_CURVE_CONFIG, max_detour_ratio=1.1),
        )
    with pytest.raises(ConfirmedGapCurveError, match="bends too far"):
        build_confirmed_gap_curve(
            (10, 10), (50, 10), (0, 1), (0, 1),
            config=replace(DEFAULT_CONFIRMED_GAP_CURVE_CONFIG, max_deviation_ratio=0.1),
        )


def test_self_intersection_guard_covers_loops_touching_and_overlapping_segments():
    assert _self_intersects(((0, 0), (10, 10), (0, 10), (10, 0)))
    assert _self_intersects(((0, 0), (10, 0), (10, 10), (0, 0)))
    assert _self_intersects(((0, 0), (10, 0), (20, 0), (5, 0)))
    assert not _self_intersects(((0, 0), (10, 0), (10, 10), (0, 10)))


@pytest.mark.parametrize("distance", (0, 2.99, 128.01, 1000))
def test_chord_bounds_are_finite_and_do_not_move_explicit_points(distance):
    with pytest.raises(ConfirmedGapCurveError, match="length"):
        build_confirmed_gap_curve((0, 0), (distance, 0), (1, 0), (-1, 0))


def test_extreme_finite_tangent_components_are_normalised_without_overflow():
    curve = build_confirmed_gap_curve((0, 0), (128, 0), (1e308, 0), (-1e308, 0))
    assert curve.source_outward == (1.0, 0.0)
    assert curve.target_outward == (-1.0, 0.0)
    assert curve.path_length_pixels == pytest.approx(128)
    assert len(curve.points_xy) <= 512


@pytest.mark.parametrize("point", ((float("nan"), 0), (float("inf"), 0), (True, 0), "12", (1,), None))
def test_invalid_explicit_endpoints_are_rejected(point):
    with pytest.raises(ConfirmedGapCurveError):
        build_confirmed_gap_curve(point, (30, 0), (1, 0), (-1, 0))


@pytest.mark.parametrize("direction", ((0, 0), (float("nan"), 0), (float("inf"), 0), (True, 0), "12", None))
def test_invalid_signed_directions_are_rejected(direction):
    with pytest.raises(ConfirmedGapCurveError):
        build_confirmed_gap_curve((0, 0), (30, 0), direction, (-1, 0))


@pytest.mark.parametrize("options", (
    {"max_gap_pixels": 129}, {"min_gap_pixels": 0},
    {"handle_ratio": 0.9}, {"max_detour_ratio": 1.81},
    {"max_deviation_ratio": 0.8}, {"min_forward_alignment": -0.51},
    {"max_vertices": 1025}, {"max_vertices": True}, {"max_vertices": 15},
    {"handle_ratio": float("inf")},
))
def test_work_and_geometry_limits_cannot_be_disabled_by_configuration(options):
    with pytest.raises(ConfirmedGapCurveError):
        build_confirmed_gap_curve(
            (0, 0), (30, 0), (1, 0), (-1, 0), config=ConfirmedGapCurveConfig(**options),
        )
