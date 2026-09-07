"""Contract tests for explicit, geometry-only contour-gap bridges."""

import math

import pytest

from ai_vectorizer.core.manual_gap_bridge import (
    ManualGapBridgeConfig,
    ManualGapBridgeError,
    build_manual_gap_bridge,
)


def test_manual_gap_bridge_preserves_explicit_endpoints_and_curves_to_tangents():
    bridge = build_manual_gap_bridge(
        (10.0, 30.0),
        (70.0, 26.0),
        (1.0, -0.25),
        (1.0, 0.25),
    )

    assert bridge.points_xy[0] == (10.0, 30.0)
    assert bridge.points_xy[-1] == (70.0, 26.0)
    assert len(bridge.points_xy) == 62
    assert bridge.source_tangent_slope < 0.0
    assert bridge.target_tangent_slope > 0.0
    assert bridge.detour_ratio <= 1.25
    assert min(point[1] for point in bridge.points_xy) < 25.0


def test_manual_gap_bridge_tangent_sign_is_axial_not_directional():
    forward = build_manual_gap_bridge(
        (5.0, 20.0),
        (45.0, 20.0),
        (1.0, -0.2),
        (1.0, 0.2),
    )
    reversed_tangents = build_manual_gap_bridge(
        (5.0, 20.0),
        (45.0, 20.0),
        (-1.0, 0.2),
        (-1.0, -0.2),
    )

    for expected, actual in zip(forward.points_xy, reversed_tangents.points_xy):
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "arguments, message",
    (
        (((1, 1), (2, 1), (1, 0), (1, 0)), "outside"),
        (((1, 1), (300, 1), (1, 0), (1, 0)), "outside"),
        (((1, 1), (10, 1), (0, 0), (1, 0)), "non-zero"),
        (((1, 1), (10, 1), (1, 0), (float("nan"), 0)), "finite"),
    ),
)
def test_manual_gap_bridge_rejects_ambiguous_or_unbounded_input(arguments, message):
    with pytest.raises(ManualGapBridgeError, match=message):
        build_manual_gap_bridge(*arguments)


def test_manual_gap_bridge_config_rejects_invalid_limits():
    configs = (
        ManualGapBridgeConfig(max_gap_pixels=2.0),
        ManualGapBridgeConfig(max_abs_tangent_slope=-0.1),
        ManualGapBridgeConfig(max_detour_ratio=0.9),
        ManualGapBridgeConfig(max_vertices=1),
    )
    for config in configs:
        with pytest.raises(ManualGapBridgeError):
            config.validate()


def test_manual_gap_bridge_is_geometry_only_and_deterministic():
    first = build_manual_gap_bridge(
        (20.0, 40.0),
        (80.0, 40.0),
        (1.0, -0.3),
        (1.0, 0.3),
    )
    second = build_manual_gap_bridge(
        (20.0, 40.0),
        (80.0, 40.0),
        (1.0, -0.3),
        (1.0, 0.3),
    )

    assert first == second
    assert math.isfinite(first.path_length_pixels)
