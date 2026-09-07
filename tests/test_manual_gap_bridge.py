"""Contract tests for explicit, geometry-only contour-gap bridges."""

import math
import subprocess
import sys

import pytest

from ai_vectorizer.core.manual_gap_bridge import (
    ManualGapBridgeConfig,
    ManualGapBridgeError,
    build_manual_gap_bridge,
    sample_manual_gap_bridge_tangents,
    sample_manual_gap_tangent,
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


def test_perpendicular_tangents_do_not_become_plausible_bridges():
    for tangent in ((0, 1), (0, -1), (1, 2), (-1, -2)):
        with pytest.raises(ManualGapBridgeError, match="direction is incompatible"):
            build_manual_gap_bridge((0, 0), (40, 0), tangent, (1, 0))
        with pytest.raises(ManualGapBridgeError, match="direction is incompatible"):
            build_manual_gap_bridge((0, 0), (40, 0), (1, 0), tangent)


def test_reasonably_aligned_noisy_tangents_keep_the_bounded_preview():
    bridge = build_manual_gap_bridge((0, 0), (40, 0), (1, -0.5), (1, 0.5))
    assert bridge.source_tangent_slope == pytest.approx(-0.35)
    assert bridge.target_tangent_slope == pytest.approx(0.35)
    assert bridge.points_xy[0] == (0, 0)
    assert bridge.points_xy[-1] == (40, 0)
    assert bridge.detour_ratio <= 1.25


def test_manual_gap_bridge_rotation_preserves_the_geometry_contract():
    angle = math.radians(61)

    def rotated(point):
        x, y = point
        return (x * math.cos(angle) - y * math.sin(angle),
                x * math.sin(angle) + y * math.cos(angle))

    arguments = ((10, 30), (65, 25), (1, -0.25), (-1, -0.2))
    original = build_manual_gap_bridge(*arguments)
    transformed = build_manual_gap_bridge(*(rotated(point) for point in arguments))
    assert len(transformed.points_xy) == len(original.points_xy)
    for point, actual in zip(original.points_xy, transformed.points_xy):
        assert actual == pytest.approx(rotated(point))
    assert transformed.detour_ratio == pytest.approx(original.detour_ratio)


def test_numeric_contract_rejects_booleans_text_and_malformed_sequences():
    for point in (None, 2, "12", {0: 1, 1: 2}, (True, 1), ("1", 2), (1,), (1, 2, 3)):
        with pytest.raises(ManualGapBridgeError):
            build_manual_gap_bridge(point, (40, 0), (1, 0), (1, 0))
    for value in (True, "3", None, float("inf"), float("nan"), 10 ** 1000):
        with pytest.raises(ManualGapBridgeError):
            ManualGapBridgeConfig(min_gap_pixels=value).validate()
    for config in (
        ManualGapBridgeConfig(max_vertices=10 ** 100),
        ManualGapBridgeConfig(max_vertices=True),
        ManualGapBridgeConfig(max_abs_tangent_slope=10),
        ManualGapBridgeConfig(min_tangent_alignment=0),
        ManualGapBridgeConfig(min_tangent_alignment=1.1),
    ):
        with pytest.raises(ManualGapBridgeError):
            config.validate()


def test_large_finite_direction_vectors_normalize_without_overflow():
    huge = sys.float_info.max
    bridge = build_manual_gap_bridge((0, 0), (40, 40), (huge, huge), (huge, huge))
    assert bridge.detour_ratio == pytest.approx(1)
    assert all(math.isfinite(value) for point in bridge.points_xy for value in point)


def _evidence(samples, size=12):
    np = pytest.importorskip("numpy")
    from ai_vectorizer.core.line_evidence import LineEvidence

    score = np.zeros((size, size), dtype=np.float32)
    centerline = np.zeros((size, size), dtype=bool)
    tangent_x = np.zeros((size, size), dtype=np.float32)
    tangent_y = np.zeros((size, size), dtype=np.float32)
    coherence = np.zeros((size, size), dtype=np.float32)
    for x, y, tangent, confidence, support in samples:
        score[y, x] = support
        centerline[y, x] = True
        tangent_x[y, x], tangent_y[y, x] = tangent
        coherence[y, x] = confidence
    return LineEvidence(score, centerline, tangent_x, tangent_y, coherence)


def test_sampler_prefers_nearby_contour_over_stronger_text_direction():
    evidence = _evidence((
        (5, 5, (1, 0), 0.3, 1),
        (4, 5, (-1, 0), 0.4, 1),
        (7, 5, (0, 1), 1, 1),
    ))
    tangent = sample_manual_gap_tangent(evidence, (5, 5))
    assert tangent == pytest.approx((1, 0))
    assert evidence.tangent_y[5, 7] == 1  # Snapshot remains intact.


def test_sampler_rejects_competing_directions_and_unreliable_support():
    for samples in (
        ((4, 5, (1, 0), 0.8, 1), (6, 5, (0, 1), 0.8, 1)),
        ((5, 5, (1, 0), 0.8, 1), (5, 6, (0, 1), 0.8, 1)),
        ((5, 5, (1, 0), 0.1, 1),),
        ((5, 5, (1, 0), 1, 0.1),),
        ((5, 5, (0, 0), 1, 1),),
    ):
        assert sample_manual_gap_tangent(_evidence(samples), (5, 5)) is None


def test_sampler_uses_a_circle_around_the_fractional_cursor():
    corner_only = _evidence(((7, 7, (1, 0), 1, 1),))
    assert sample_manual_gap_tangent(corner_only, (5, 5), radius_pixels=2.5) is None

    evidence = _evidence(((4, 5, (0, 1), 1, 1), (5, 5, (1, 0), 0.3, 1)))
    tangent = sample_manual_gap_tangent(evidence, (4.51, 5), radius_pixels=0.5)
    assert tangent == pytest.approx((1, 0))


def test_sampler_rotation_and_axial_signs_preserve_direction():
    samples = (
        (5, 5, (1, 0.2), 0.4, 1),
        (4, 5, (-1, -0.2), 0.5, 1),
        (7, 5, (0, 1), 1, 1),
    )
    original = sample_manual_gap_tangent(_evidence(samples), (5, 5))
    transformed = tuple(
        (11 - y, x, (-tangent[1], tangent[0]), confidence, support)
        for x, y, tangent, confidence, support in samples
    )
    rotated = sample_manual_gap_tangent(_evidence(transformed), (6, 5))
    assert rotated == pytest.approx((-original[1], original[0]))


def test_sampler_rejects_invalid_radius_and_out_of_cache_endpoints():
    evidence = _evidence(((5, 5, (1, 0), 1, 1),))
    for radius in (0, -1, True, "3", 33, float("inf"), float("nan")):
        with pytest.raises(ManualGapBridgeError):
            sample_manual_gap_tangent(evidence, (5, 5), radius_pixels=radius)
    for point in ((-0.1, 5), (12, 5), (5, 12), (False, 5), (float("nan"), 5)):
        with pytest.raises(ManualGapBridgeError):
            sample_manual_gap_tangent(evidence, point)
    with pytest.raises(ManualGapBridgeError, match="LineEvidence"):
        sample_manual_gap_tangent(None, (5, 5))


def test_contextual_pair_uses_one_sided_contour_support_not_inner_glyphs():
    intended = [
        (x, 10, (1, 0), 1, 1)
        for x in tuple(range(4, 11)) + tuple(range(30, 37))
    ]
    # These imitate number strokes within the explicitly selected blank.  They
    # have strong evidence but lie on the wrong side of each endpoint.
    glyph = [(x, y, (0, 1), 1, 1) for x in range(12, 29) for y in (7, 8)]
    tangents = sample_manual_gap_bridge_tangents(
        _evidence(tuple(intended + glyph), size=48), (10, 10), (30, 10),
    )

    assert tangents is not None
    assert abs(tangents[0][0]) > 0.99
    assert abs(tangents[0][1]) < 0.01
    assert abs(tangents[1][0]) > 0.99
    assert abs(tangents[1][1]) < 0.01


def test_contextual_pair_rejects_inner_glyphs_and_remote_parallel_support():
    glyph = [(x, y, (1, 0), 1, 1) for x in range(12, 29) for y in (7, 8)]
    parallel = [
        (x, 18, (1, 0), 1, 1)
        for x in tuple(range(1, 11)) + tuple(range(30, 40))
    ]
    evidence = _evidence(tuple(glyph + parallel), size=48)

    assert sample_manual_gap_bridge_tangents(evidence, (10, 10), (30, 10)) is None


def test_contextual_pair_rejects_invalid_or_degenerate_requests():
    evidence = _evidence(((5, 5, (1, 0), 1, 1),))
    for radius in (True, "12", 11, 33, float("nan")):
        with pytest.raises(ManualGapBridgeError):
            sample_manual_gap_bridge_tangents(
                evidence, (5, 5), (8, 5), support_radius_pixels=radius,
            )
    with pytest.raises(ManualGapBridgeError, match="non-zero"):
        sample_manual_gap_bridge_tangents(evidence, (5, 5), (5, 5))


def test_geometry_import_and_build_do_not_require_numpy():
    code = """
import builtins
original_import = builtins.__import__
def without_numpy(name, *args, **kwargs):
    if name == 'numpy' or name.startswith('numpy.'):
        raise ImportError('NumPy deliberately unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_numpy
from ai_vectorizer.core.manual_gap_bridge import (
    build_manual_gap_bridge, sample_manual_gap_tangent, ManualGapBridgeError,
)
assert build_manual_gap_bridge((0, 0), (40, 0), (1, 0), (1, 0)).points_xy[-1] == (40, 0)
try:
    sample_manual_gap_tangent(None, (1, 1))
except ManualGapBridgeError as exc:
    assert 'NumPy is required' in str(exc)
else:
    raise AssertionError('sampler silently ignored unavailable NumPy')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
