import math

import numpy as np
import pytest

import ai_vectorizer.core.livewire as livewire_module
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.livewire import (
    LiveWireCancelled,
    LiveWireConfig,
    blend_path_with_cursor,
    build_livewire_tree,
    is_livewire_available,
)


pytestmark = pytest.mark.skipif(
    not is_livewire_available(),
    reason="SciPy is not installed",
)


def _dense_contour_fixture():
    height, width = 128, 160
    image = np.full((height, width), 245, dtype=np.uint8)
    edges = np.zeros((height, width), dtype=np.uint8)

    desired = []
    adjacent = []
    for x in range(8, 152):
        y = int(round(60 + 7 * math.sin((x - 8) / 30.0)))
        adjacent_y = y + 11
        desired.append((x, y))
        adjacent.append((x, adjacent_y))
        # A short scan break in the desired contour is intentional.
        if not 72 <= x <= 79:
            image[y, x] = 15
            edges[y, x] = 255
        image[adjacent_y, x] = 20
        edges[adjacent_y, x] = 255

    # Number/symbol-like strokes intersect the contours and create tempting
    # branches. The expected route must preserve the incoming contour.
    image[39:87, 52] = 5
    edges[39:87, 52] = 255
    image[42, 48:58] = 5
    edges[42, 48:58] = 255
    image[82, 48:58] = 5
    edges[82, 48:58] = 255
    return image, edges, desired, adjacent


def test_zero_strength_is_literal_cursor_segment():
    image, edges, desired, _ = _dense_contour_fixture()
    root = desired[2]
    target = (desired[-3][0] + 0.4, desired[-3][1] - 0.3)
    tree = build_livewire_tree(image, edges, root, strength=0.0)

    assert tree.trace(target) == [
        (float(root[0]), float(root[1])),
        (float(target[0]), float(target[1])),
    ]


def test_zero_strength_remains_literal_with_optional_evidence():
    image, edges, desired, _ = _dense_contour_fixture()
    root = desired[2]
    target = (desired[-3][0] + 0.4, desired[-3][1] - 0.3)
    evidence = LineEvidence(
        center_score=np.ones(edges.shape, dtype=np.float32),
        centerline=np.ones(edges.shape, dtype=bool),
    )
    tree = build_livewire_tree(
        image,
        edges,
        root,
        strength=0.0,
        evidence=evidence,
    )

    assert tree.trace(target) == [
        (float(root[0]), float(root[1])),
        (float(target[0]), float(target[1])),
    ]


def test_explicit_none_evidence_is_v1_compatible():
    image, edges, desired, _ = _dense_contour_fixture()
    root = desired[4]
    kwargs = {
        "strength": 1.0,
        "incoming_direction": (1.0, 0.0),
        "config": LiveWireConfig(max_window_size=160, target_snap_radius=3),
    }

    historical = build_livewire_tree(image, edges, root, **kwargs)
    explicit_none = build_livewire_tree(
        image,
        edges,
        root,
        evidence=None,
        **kwargs,
    )

    np.testing.assert_array_equal(
        historical.predecessors,
        explicit_none.predecessors,
    )
    np.testing.assert_array_equal(historical.distances, explicit_none.distances)
    assert historical.trace(desired[-5]) == explicit_none.trace(desired[-5])


def test_direct_centerline_evidence_can_supply_a_missing_route():
    image = np.full((64, 64), 255, dtype=np.uint8)
    edges = np.zeros((64, 64), dtype=np.uint8)
    root = (5, 32)
    target = (58, 32)
    centerline = np.zeros(edges.shape, dtype=bool)

    def draw_segment(start, end):
        steps = int(max(abs(end[0] - start[0]), abs(end[1] - start[1])))
        for index in range(steps + 1):
            fraction = index / max(1, steps)
            x = int(round(start[0] + (end[0] - start[0]) * fraction))
            y = int(round(start[1] + (end[1] - start[1]) * fraction))
            centerline[y, x] = True

    draw_segment(root, (18, 18))
    draw_segment((18, 18), (45, 18))
    draw_segment((45, 18), target)
    evidence = LineEvidence(
        # The direct mask remains usable even if a producer has not supplied
        # a separate continuous score yet.
        center_score=np.zeros(edges.shape, dtype=np.float32),
        centerline=centerline,
    )

    without_evidence = build_livewire_tree(
        image,
        edges,
        root,
        strength=1.0,
        config=LiveWireConfig(max_window_size=64, target_snap_radius=0),
    ).trace(target)
    with_evidence = build_livewire_tree(
        image,
        edges,
        root,
        strength=1.0,
        evidence=evidence,
        config=LiveWireConfig(max_window_size=64, target_snap_radius=0),
    ).trace(target)

    assert min(y for _x, y in without_evidence) > 25.0
    assert min(y for _x, y in with_evidence) <= 20.0
    supported = sum(
        bool(centerline[int(round(y)), int(round(x))])
        for x, y in with_evidence
    )
    assert supported >= int(len(with_evidence) * 0.75)


def test_precomputed_evidence_skips_duplicate_image_feature_passes(monkeypatch):
    image = np.full((48, 48), 255, dtype=np.uint8)
    edges = np.zeros((48, 48), dtype=np.uint8)
    centerline = np.zeros(edges.shape, dtype=bool)
    centerline[24, 4:44] = True
    tangent_x = np.zeros(edges.shape, dtype=np.float32)
    tangent_x[centerline] = 1.0
    coherence = np.zeros(edges.shape, dtype=np.float32)
    coherence[centerline] = 1.0
    evidence = LineEvidence(
        center_score=centerline.astype(np.float32),
        centerline=centerline,
        tangent_x=tangent_x,
        tangent_y=np.zeros_like(tangent_x),
        coherence=coherence,
    )

    def unexpected_feature_pass(*_args, **_kwargs):
        raise AssertionError("precomputed evidence must bypass image features")

    runtime, _error = livewire_module._get_livewire_runtime()
    ndimage, _sparse, _dijkstra = runtime
    monkeypatch.setattr(livewire_module, "_to_grayscale", unexpected_feature_pass)
    monkeypatch.setattr(
        ndimage,
        "distance_transform_edt",
        unexpected_feature_pass,
    )
    monkeypatch.setattr(ndimage, "gaussian_filter", unexpected_feature_pass)
    monkeypatch.setattr(ndimage, "sobel", unexpected_feature_pass)

    tree = build_livewire_tree(
        image,
        edges,
        (4, 24),
        evidence=evidence,
        config=LiveWireConfig(max_window_size=48, target_snap_radius=0),
    )

    assert tree.trace((43, 24))[0] == (4.0, 24.0)
    assert tree.trace((43, 24))[-1] == (43.0, 24.0)


def test_evidence_type_and_dimensions_are_checked():
    image = np.full((32, 32), 255, dtype=np.uint8)
    edges = np.zeros_like(image)
    with pytest.raises(TypeError, match="LineEvidence"):
        build_livewire_tree(image, edges, (4, 4), evidence={})

    evidence = LineEvidence(
        center_score=np.zeros((16, 16), dtype=np.float32),
        centerline=np.zeros((16, 16), dtype=bool),
    )
    with pytest.raises(ValueError, match="dimensions"):
        build_livewire_tree(image, edges, (4, 4), evidence=evidence)

    matching_evidence = LineEvidence(
        center_score=np.zeros(edges.shape, dtype=np.float32),
        centerline=np.zeros(edges.shape, dtype=bool),
    )
    with pytest.raises(ValueError, match="image dimensions"):
        build_livewire_tree(
            image[:, :-1],
            edges,
            (4, 4),
            evidence=matching_evidence,
        )


def test_direction_aware_route_bridges_gap_without_switching_contours():
    image, edges, desired, _ = _dense_contour_fixture()
    root = desired[4]
    target = desired[-5]
    tree = build_livewire_tree(
        image,
        edges,
        root,
        strength=1.0,
        incoming_direction=(1.0, 0.0),
        config=LiveWireConfig(max_window_size=160, target_snap_radius=3),
    )

    path = tree.trace(target)
    assert len(path) > 60
    # Compare every route point with the fixture's intended curve. It may
    # move a few pixels through the scan break, but must not jump 11 pixels
    # onto the adjacent contour.
    expected_y = {
        x: int(round(60 + 7 * math.sin((x - 8) / 30.0)))
        for x in range(root[0], target[0] + 1)
    }
    deviations = [
        abs(y - expected_y.get(int(round(x)), int(round(y))))
        for x, y in path
        if root[0] <= int(round(x)) <= target[0]
    ]
    assert np.percentile(deviations, 95) <= 3.0
    assert max(deviations) < 8.0


def test_livewire_window_bounds_wandering_and_falls_back_outside():
    image = np.full((300, 300), 255, dtype=np.uint8)
    edges = np.zeros_like(image)
    image[150, :] = 0
    edges[150, :] = 255
    tree = build_livewire_tree(
        image,
        edges,
        (150, 150),
        strength=1.0,
        config=LiveWireConfig(max_window_size=96),
    )

    assert tree.shape == (96, 96)
    assert tree.contains((190, 150))
    assert not tree.contains((260, 150))
    assert tree.trace((260, 150)) == [(150.0, 150.0), (260.0, 150.0)]


def test_input_dimensions_are_bounded_before_feature_allocations():
    image = np.zeros((2, 1025), dtype=np.uint8)
    edges = np.zeros_like(image)

    with pytest.raises(ValueError, match="must not exceed 1024x1024"):
        build_livewire_tree(image, edges, (1, 1))


def test_blend_strength_is_a_true_geometry_continuum():
    assisted = [(0, 0), (5, 5), (10, 0)]
    assert blend_path_with_cursor(assisted, (0, 0), (10, 0), 0.0) == [
        (0.0, 0.0),
        (10.0, 0.0),
    ]
    half = blend_path_with_cursor(assisted, (0, 0), (10, 0), 0.5)
    full = blend_path_with_cursor(assisted, (0, 0), (10, 0), 1.0)

    assert half[1] == pytest.approx((5.0, 2.5))
    assert full[1] == pytest.approx((5.0, 5.0))
    assert half[-1] == pytest.approx((10.0, 0.0))


def test_background_build_honors_early_cancellation():
    image = np.full((64, 64), 255, dtype=np.uint8)
    edges = np.zeros_like(image)
    with pytest.raises(LiveWireCancelled):
        build_livewire_tree(
            image,
            edges,
            (32, 32),
            cancel_check=lambda: True,
        )


def test_invalid_root_is_rejected_before_graph_build():
    image = np.full((32, 32), 255, dtype=np.uint8)
    edges = np.zeros_like(image)
    with pytest.raises(ValueError, match="outside"):
        build_livewire_tree(image, edges, (-1, 4))


@pytest.mark.parametrize(
    "config",
    (
        LiveWireConfig(edge_sigma=float("nan")),
        LiveWireConfig(line_cost_weight=float("nan")),
        LiveWireConfig(max_window_size=32.5),
        LiveWireConfig(max_window_size=2048),
    ),
)
def test_invalid_or_unbounded_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config.validate()


def test_nonfinite_image_pixel_does_not_poison_the_shortest_path_tree():
    image = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    image[32, 32] = np.nan
    edges = np.zeros((64, 64), dtype=np.uint8)
    edges[32, :] = 255

    tree = build_livewire_tree(image, edges, (10, 32))

    assert np.isfinite(tree.distances).all()
    assert len(tree.trace((52, 32))) > 2


def test_nonfinite_strength_is_rejected():
    with pytest.raises(ValueError, match="strength must be a finite number"):
        blend_path_with_cursor(
            [(0, 0), (1, 1), (2, 0)],
            (0, 0),
            (2, 0),
            float("nan"),
        )
