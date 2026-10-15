"""Printed-grid geometry, finite crossing costs and blank shortcut controls."""

from dataclasses import replace
import math

import numpy as np
import pytest

from ai_vectorizer.core.map_grid import MapGridCancelled, MapGridPrior, crop_grid_prior, detect_map_grid
from ai_vectorizer.core.learned_prior import LearnedLinePrior
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree, is_livewire_available
from ai_vectorizer.core.smart_recovery import build_corridor_cost_map


def _printed_grid(size=256, slope=0.0):
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    xy = np.arange(size)
    for intercept in (48, 128, 208):
        shifted = np.rint(intercept + slope * (xy - (size - 1) * 0.5)).astype(int)
        valid = (shifted >= 0) & (shifted < size)
        image[shifted[valid], xy[valid]] = 30
        image[xy[valid], shifted[valid]] = 30
    return image


@pytest.mark.parametrize("slope", (0.0, -0.024, 0.024))
def test_detects_both_printed_grid_families_with_small_scan_rotation(slope):
    image = _printed_grid(slope=slope)
    before = image.copy()
    prior = detect_map_grid(image)
    assert prior.shape == (256, 256)
    assert np.count_nonzero(prior.horizontal_score) >= 700
    assert np.count_nonzero(prior.vertical_score) >= 700
    assert np.count_nonzero(prior.score) < 11000
    assert abs(prior.horizontal_slope - slope) <= 0.008
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize("kind", ("straight_contour", "parallel_contours", "curved_contours", "short_symbols", "single_axis_grid", "isolated_cross"))
def test_conservative_negative_controls_leave_contours_alone(kind):
    image = np.full((256, 256, 3), 245, dtype=np.uint8)
    if kind == "straight_contour":
        image[128] = 20
    elif kind == "isolated_cross":
        image[128] = 20
        image[:, 128] = 20
    elif kind in ("parallel_contours", "single_axis_grid"):
        image[32::40] = 20
    elif kind == "short_symbols":
        for y in range(10, 240, 20):
            for x in range(10, 240, 20):
                image[y:y+8, x] = 20
                image[y, x:x+8] = 20
    else:
        for x in range(256):
            for base in range(25, 230, 25):
                y = int(round(base + 12 * math.sin(x / 24)))
                image[y, x] = 20
    assert not np.any(detect_map_grid(image).score)


def test_chromatic_blue_and_brown_contours_do_not_form_a_dark_grid():
    image = _printed_grid()
    mask = image[:, :, 0] == 30
    image[mask] = (155, 90, 50)
    assert not np.any(detect_map_grid(image).score)
    image[mask] = (30, 60, 180)
    assert not np.any(detect_map_grid(image).score)


def test_prior_is_immutable_and_crop_keeps_pixel_coordinates():
    horizontal = np.zeros((12, 15))
    horizontal[5, :] = 1
    prior = MapGridPrior(horizontal, np.zeros_like(horizontal))
    horizontal[:] = 0
    assert prior.horizontal_score[5, 6] == 1
    crop = crop_grid_prior(prior, (3, 2, 10, 8))
    assert crop.shape == (6, 7)
    assert crop.horizontal_score[3, 3] == 1
    with pytest.raises(ValueError):
        crop.horizontal_score[3, 3] = 0
    for bounds in ((0, 0, 100, 12), (2, 2, 2, 5), (0.0, 0, 10, 10)):
        with pytest.raises(ValueError):
            crop_grid_prior(prior, bounds)


@pytest.mark.parametrize("invalid", (np.full((4, 4), np.nan), np.ones((4, 4))*1.1, np.zeros(4)))
def test_invalid_score_boundaries_are_rejected(invalid):
    with pytest.raises(ValueError):
        MapGridPrior(invalid, invalid)


def test_unsupported_raster_range_is_not_reinterpreted_and_cancellation_propagates():
    assert not np.any(detect_map_grid(_printed_grid().astype(np.uint16)).score)
    with pytest.raises(MapGridCancelled):
        detect_map_grid(_printed_grid(), cancel_check=lambda: True)


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
def test_directional_grid_cost_penalizes_following_but_preserves_transverse_crossing():
    shape = (64, 64)
    horizontal = np.zeros(shape)
    horizontal[32, :] = 1
    prior = MapGridPrior(horizontal, np.zeros(shape))
    # Uniform observation isolates the grid contribution from detector effects.
    evidence = LineEvidence(np.ones(shape), np.ones(shape, dtype=bool))
    image = np.full((*shape, 3), 245, dtype=np.uint8)
    edges = np.zeros(shape, dtype=np.uint8)
    config = LiveWireConfig(max_window_size=64, target_snap_radius=0, direction_cost_weight=0, heading_cost_weight=0)
    baseline = build_livewire_tree(image, edges, (20, 32), evidence=evidence, config=config)
    guided = build_livewire_tree(image, edges, (20, 32), evidence=evidence, grid_prior=prior, config=config)
    assert guided.distances[32*64+21] > baseline.distances[32*64+21]
    assert guided.distances[33*64+20] == baseline.distances[33*64+20]
    assert guided.distances[33*64+21] < 2.0
    assert np.isfinite(guided.distances).all()
    assert guided.trace((20, 45))[-1] == (20.0, 45.0)
    literal = build_livewire_tree(image, edges, (20, 32), evidence=evidence, grid_prior=prior, config=config, strength=0)
    assert literal.trace((40.2, 35.3)) == [(20.0, 32.0), (40.2, 35.3)]


def test_recovery_grid_penalty_remains_finite_and_local():
    evidence = LineEvidence(np.ones((32, 32)), np.ones((32, 32), dtype=bool))
    horizontal = np.zeros((32, 32))
    horizontal[16, :] = 1
    prior = MapGridPrior(horizontal, np.zeros((32, 32)))
    cost = build_corridor_cost_map(evidence, np.ones((32, 32)), grid_prior=prior)
    assert cost[16, 15] > cost[15, 15] == 1
    assert np.isfinite(cost).all()


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
def test_detected_grid_no_longer_wins_over_one_visible_curved_contour():
    image = _printed_grid()
    centerline = image[:, :, 0] == 30
    for x in range(15, 241):
        y = int(round(128 + 48 * math.sin((x-15) * math.pi / 225)))
        centerline[y, x] = True
        image[y, x] = 30
    prior = detect_map_grid(image)
    evidence = LineEvidence(centerline.astype(np.float32), centerline)
    config = LiveWireConfig(max_window_size=256, target_snap_radius=0, direction_cost_weight=0, heading_cost_weight=0)
    kwargs = dict(evidence=evidence, config=config)
    baseline = build_livewire_tree(image, np.zeros((256, 256)), (15, 128), **kwargs).trace((240, 128))
    guided = build_livewire_tree(image, np.zeros((256, 256)), (15, 128), grid_prior=prior, **kwargs).trace((240, 128))
    def contour_error(path):
        return np.mean([abs(y - (128 + 48 * math.sin((x-15) * math.pi / 225))) for x,y in path])
    assert contour_error(baseline) > 20
    assert contour_error(guided) < 3
    assert max(y for x,y in guided) > 173


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
def test_learned_support_cannot_turn_blank_inter_contour_space_into_a_shortcut():
    size = 256
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    centerline = np.zeros((size, size), dtype=bool)
    # Both anchors lie on one visible contour, curving around white space.
    for x in range(10, 150):
        y = int(round(30 + 140 * math.sin((x-10) * math.pi / 139)))
        centerline[y, x] = True
    # Rasterize diagonal adjacency; max slope >1 needs intervening pixels.
    for x in range(10, 149):
        ys = np.flatnonzero(centerline[:, x])
        next_y = np.flatnonzero(centerline[:, x+1])[0]
        centerline[min(ys[0], next_y):max(ys[0], next_y)+1, x] = True
    image[centerline] = 25
    evidence = LineEvidence(centerline.astype(np.float32), centerline)
    prior = LearnedLinePrior(np.ones(centerline.shape), np.zeros(centerline.shape))
    config = LiveWireConfig(max_window_size=size, target_snap_radius=0, direction_cost_weight=0, heading_cost_weight=0, learned_observation_gate=True, observed_connections=False)
    kwargs = dict(evidence=evidence, learned_prior=prior)
    ungated = build_livewire_tree(image, np.zeros(centerline.shape), (10, 30), config=replace(config, learned_observation_gate=False), **kwargs).trace((149, 30))
    gated = build_livewire_tree(image, np.zeros(centerline.shape), (10, 30), config=config, **kwargs).trace((149, 30))
    def blank_fraction(path):
        return np.mean([not np.any(centerline[max(0,int(y)-2):min(size,int(y)+3), max(0,int(x)-2):min(size,int(x)+3)]) for x,y in path])
    assert blank_fraction(ungated) > 0.30
    assert blank_fraction(gated) < 0.02
    assert max(y for x,y in gated) > 165


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
def test_real_detector_dense_prior_does_not_hop_to_separate_neighbor_contour():
    from ai_vectorizer.core.edge_detector import EdgeDetector

    size = 256
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    for x in range(20, 237):
        y = int(round(50 + 25 * math.sin((x-20) * math.pi / 216)))
        image[y, x] = 20
    image[44, 20:237] = 20
    evidence = EdgeDetector(method=EdgeDetector.METHOD_INK).detect_ink_evidence(image)
    edges = np.where(evidence.centerline, 255, 0).astype(np.uint8)
    prior = LearnedLinePrior(np.ones((size, size)), np.zeros((size, size)))
    config = LiveWireConfig(max_window_size=size, target_snap_radius=0, learned_observation_gate=True, observed_connections=False)
    kwargs = dict(evidence=evidence, config=config)
    baseline = build_livewire_tree(image, edges, (20, 50), **kwargs).trace((236, 50))
    guided = build_livewire_tree(image, edges, (20, 50), learned_prior=prior, **kwargs).trace((236, 50))
    old = build_livewire_tree(image, edges, (20, 50), learned_prior=prior, evidence=evidence, config=replace(config, learned_observation_gate=False)).trace((236, 50))
    assert max(y for x,y in baseline) == 75
    assert np.mean([y < 49 for x,y in old]) > 0.90
    assert max(y for x,y in guided) == 75
    assert not any(y < 49 for x,y in guided)


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
def test_learned_route_guard_keeps_observed_contour_when_both_heads_favor_neighbor():
    from ai_vectorizer.core.edge_detector import EdgeDetector

    size = 256
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    avoid = np.zeros((size, size), dtype=np.float32)
    for x in range(20, 237):
        y = int(round(50 + 25 * math.sin((x-20) * math.pi / 216)))
        image[y, x] = 20
        avoid[max(0,y-2):y+3, x] = 1
    image[44, 20:237] = 20
    evidence = EdgeDetector.detect_ink_evidence(image)
    edges = np.where(evidence.centerline, 255, 0).astype(np.uint8)
    prior = LearnedLinePrior(np.ones((size, size)), avoid)
    config = LiveWireConfig(max_window_size=size, target_snap_radius=0, learned_observation_gate=True, learned_route_guard=True, learned_avoidance_cost_weight=6)
    # A guard can preserve the wrong Ink path too. It stays diagnostic;
    # the normal product must not silently enable either rejected candidate.
    assert not LiveWireConfig().learned_observation_gate
    assert not LiveWireConfig().learned_route_guard
    kwargs = dict(evidence=evidence, learned_prior=prior)
    unguarded = build_livewire_tree(image, edges, (20,50), config=replace(config, learned_route_guard=False), **kwargs)
    guarded = build_livewire_tree(image, edges, (20,50), config=config, **kwargs)
    np.testing.assert_array_equal(guarded.distances, unguarded.distances)
    assert guarded.champion_tree is not None
    assert guarded.route_evidence is evidence
    ordinary = build_livewire_tree(image, edges, (20,50), config=LiveWireConfig(max_window_size=size, target_snap_radius=0), **kwargs)
    assert ordinary.champion_tree is None
    assert guarded.trace((20, 50)) == [(20.0, 50.0), (20.0, 50.0)]
    assert guarded.trace((20.2, 50.1)) == [(20.0, 50.0), (20.2, 50.1)]
    assert np.mean([y < 49 for x,y in unguarded.trace((236,50))]) > 0.90
    assert max(y for x,y in guarded.trace((236,50))) == 75
    assert not any(y < 49 for x,y in guarded.trace((236,50)))
    partial = build_livewire_tree(image, edges, (20,50), config=config, strength=0.5, **kwargs)
    assert max(y for x,y in partial.trace((236,50))) == pytest.approx(62.5)
    literal = build_livewire_tree(image, edges, (20,50), config=config, strength=0, **kwargs)
    assert literal.champion_tree is None
    assert literal.trace((236.2,50.3)) == [(20.0,50.0),(236.2,50.3)]
