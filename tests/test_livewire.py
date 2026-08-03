import math

import numpy as np
import pytest

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
