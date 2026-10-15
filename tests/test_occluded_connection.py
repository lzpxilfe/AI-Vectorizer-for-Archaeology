"""Geometry/lifecycle contracts for independent-label automatic proposals."""
from types import SimpleNamespace

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from ai_vectorizer.core.label_occlusion import detect_label_occlusions
from ai_vectorizer.core.observed_connection import ObservedConnectionCancelled
from ai_vectorizer.core.occluded_connection import (
    OccludedConnectionTree, _Route, build_occluded_connection,
)


def _graph(shape):
    height, width = shape
    pixels = np.arange(height * width).reshape(shape)
    rows, columns, values = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        first = pixels[y0:y1, x0:x1].ravel()
        last = pixels[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        rows.extend((first, last))
        columns.extend((last, first))
        values.extend((np.full(len(first), np.hypot(dx, dy)),) * 2)
    return sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=(height * width,) * 2)


def _fixture(rotation=0):
    # The two visible arms already belong to the same component through a
    # distant native bend. This catches the old "finite native distance means
    # no gap" restriction. The third arm is an independent contour competitor.
    native = np.zeros((96, 96), dtype=bool)
    native[30:65, 40] = True
    native[30:61, 64] = True
    native[30, 40:65] = True
    native[30:65, 72] = True
    image = np.full(native.shape, 255, dtype=np.uint8)
    image[native] = 0
    for x in (47, 56):
        image[63:75, x:x + 7] = 0
        image[65:73, x + 2:x + 5] = 255
    points = [(40, 46), (64, 46), (72, 46)]
    for _ in range(rotation):
        native, image = np.rot90(native), np.rot90(image)
        points = [(y, 95 - x) for x, y in points]
    regions = detect_label_occlusions(image)
    assert len(regions) == 1
    return image, native, regions, points


def _build(image, native, regions, root, **kwargs):
    return build_occluded_connection(_graph(native.shape), native, root, kwargs.pop("origin", (0, 0)), regions, sparse=sparse, dijkstra=dijkstra, image=image, **kwargs)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_detected_label_has_one_natural_gap_and_never_maps_to_neighbor(rotation):
    image, native, regions, (root, target, neighbor) = _fixture(rotation)
    tree = _build(image, native, regions, root)
    assert tree is not None
    route = tree.trace(target)
    assert route is not None
    assert route[0] == root and route[-1] == target
    assert len(route) > 40
    assert tree.intersects(route)
    assert tree.accepts_display(route)
    assert tree.trace(neighbor) is None
    # The original continuous detour remains an observed fact, but the new
    # route turns into the label's uncertainty instead of following that loop.
    center = np.asarray(route)
    if rotation == 0:
        assert center[:, 1].max() > 68 and center[:, 1].min() >= 46


def test_native_evidence_and_detector_masks_are_not_changed():
    image, native, regions, (root, target, _) = _fixture()
    native_before, image_before = native.copy(), image.copy()
    masks = [r.mask.copy() for r in regions]
    tree = _build(image, native, regions, root)
    assert tree.trace(target)
    np.testing.assert_array_equal(native, native_before)
    np.testing.assert_array_equal(image, image_before)
    for region, before in zip(regions, masks):
        np.testing.assert_array_equal(region.mask, before)
    for array in (tree.mask, tree.outside_native):
        assert not array.flags.writeable
    for route in tree.routes:
        for array in (route.native_indices, route.lookup, route.predecessors, route.distances):
            assert not array.flags.writeable


def test_nonzero_origin_preserves_exact_authoritative_clicks():
    image, native, regions, (root, target, _) = _fixture()
    origin = (123, 456)
    shifted_root = tuple(np.asarray(root) + origin)
    shifted_target = tuple(np.asarray(target) + origin)
    tree = _build(image, native, regions, shifted_root, origin=origin)
    route = tree.trace(shifted_target)
    assert route[0] == shifted_root and route[-1] == shifted_target
    assert tree.accepts_display(route)


def test_root_inside_uncertainty_does_not_choose_an_arm():
    image, native, regions, _ = _fixture()
    x0, y0, _, _ = regions[0].bbox_xyxy
    ys, xs = np.nonzero(regions[0].mask)
    root = (int(xs[0] + x0), int(ys[0] + y0))
    assert _build(image, native, regions, root) is None


def test_root_beyond_three_pixels_cannot_borrow_a_visible_arm():
    image, native, regions, _ = _fixture()
    assert _build(image, native, regions, (46, 46)) is None


def test_three_pixel_root_attachment_keeps_literal_root():
    image, native, regions, (_, target, _) = _fixture()
    root = (42.2, 46.0)
    tree = _build(image, native, regions, root)
    assert tree is not None
    assert tree.trace(target)[0] == root


def test_equidistant_separate_root_components_are_ambiguous():
    image, native, regions, _ = _fixture()
    # At x68 the gap between right and competitor arms is four pixels.
    assert _build(image, native, regions, (68, 46)) is None


@pytest.mark.parametrize("target", [(np.nan, 1), (1, np.inf), (-1, 2), (96, 1), (40, 46), (50, 68)])
def test_invalid_or_uncertain_targets_decline(target):
    image, native, regions, (root, _, _) = _fixture()
    tree = _build(image, native, regions, root)
    assert tree.trace(target) is None


def test_empty_regions_and_exceeded_work_budget_decline():
    image, native, regions, (root, _, _) = _fixture()
    assert _build(image, native, (), root) is None
    assert _build(image, native, regions * 17, root) is None
    big = np.zeros((1025, 1024), dtype=bool)
    assert build_occluded_connection(None, big, (0, 0), (0, 0), (), sparse=sparse, dijkstra=dijkstra) is None


def test_full_mask_and_bbox_local_mask_have_identical_results():
    image, native, regions, (root, target, _) = _fixture()
    region = regions[0]
    full = np.zeros(native.shape, dtype=bool)
    x0, y0, x1, y1 = region.bbox_xyxy
    full[y0:y1, x0:x1] = region.mask
    other = SimpleNamespace(bbox_xyxy=region.bbox_xyxy, mask=full, baseline_xy=region.baseline_xy)
    first = _build(image, native, regions, root)
    second = _build(image, native, (other,), root)
    np.testing.assert_allclose(first.trace(target), second.trace(target))


def test_hover_uses_only_completed_snapshot(monkeypatch):
    image, native, regions, (root, target, _) = _fixture()
    tree = _build(image, native, regions, root)
    expected = tree.trace(target)
    def forbidden(*args, **kwargs):
        raise AssertionError("hover tried to process the source again")
    monkeypatch.setattr("ai_vectorizer.core.occluded_connection.build_observed_connection", forbidden)
    monkeypatch.setattr("scipy.sparse.csgraph.dijkstra", forbidden)
    monkeypatch.setattr("ai_vectorizer.core.occluded_connection._widths", forbidden)
    assert tree.trace(target) == expected
    assert tree.accepts_display(expected)


def test_cancellation_never_publishes_partial_routes():
    image, native, regions, (root, _, _) = _fixture()
    with pytest.raises(ObservedConnectionCancelled):
        _build(image, native, regions, root, cancel_check=lambda: True)
    calls = []
    def cancel_after_native_tree():
        calls.append(True)
        return len(calls) >= 8
    with pytest.raises(ObservedConnectionCancelled):
        _build(image, native, regions, root, cancel_check=cancel_after_native_tree)


def test_intersection_samples_segment_interior_and_bounds_work():
    image, native, regions, (root, _, _) = _fixture()
    tree = _build(image, native, regions, root)
    assert tree.intersects([(35, 68), (75, 68)])
    assert not tree.intersects([(35, 20), (75, 20)])
    assert not tree.intersects([(np.nan, 1), (2, 3)])
    assert not tree.intersects([(0, 0)] * 8193)


def test_blended_display_must_not_hit_foreign_visible_contour():
    image, native, regions, (root, target, _) = _fixture()
    tree = _build(image, native, regions, root)
    full = tree.trace(target)
    assert tree.accepts_display(full)
    invalid = [full[0], (72.0, 50.0), full[-1]]
    assert not tree.accepts_display(invalid)
    assert not tree.accepts_display([(np.nan, 1), target])


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
@pytest.mark.parametrize("origin", [(0, 0), (123, 456)])
def test_virtual_curve_cannot_own_a_nearby_foreign_stroke(rotation, origin):
    from ai_vectorizer.core.confirmed_gap_curve import (
        ConfirmedGapCurveConfig, build_confirmed_gap_curve,
    )
    from ai_vectorizer.core.livewire import blend_path_with_cursor

    # The full cubic clears this independent eleven-pixel line by >1px.
    # Its prepared inference used to claim all native pixels within1.25px,
    # allowing a fractional blend to land directly on that foreign line.
    curve = build_confirmed_gap_curve(
        (8, 45), (104, 45), (0, 1), (0, 1),
        config=ConfirmedGapCurveConfig(handle_ratio=18.249 / 0.75 / 96),
    )
    prefix = np.asarray([(x, 45) for x in range(4, 8)], dtype=float)
    gap = np.asarray(curve.points_xy)
    suffix = np.asarray([(x, 45) for x in range(104, 109)], dtype=int)
    foreign = np.asarray([(x, 62) for x in range(51, 62)], dtype=int)

    def rotate(points):
        result = np.asarray(points).copy()
        for _ in range(rotation):
            result = np.column_stack((result[:, 1], 119 - result[:, 0]))
        return result

    prefix, gap, suffix, foreign = map(rotate, (prefix, gap, suffix, foreign))
    shape = (120, 120)
    outside = np.zeros(shape, dtype=bool)
    observed = np.concatenate((prefix.astype(int), suffix, foreign))
    outside[observed[:, 1], observed[:, 0]] = True
    indices = (suffix[:, 1] * shape[1] + suffix[:, 0]).astype(np.int32)
    lookup = np.full(shape, -1, dtype=np.int32)
    lookup.ravel()[indices] = np.arange(len(indices))
    offset = np.asarray(origin)
    root, target = tuple(prefix[0] + offset), tuple(suffix[-1] + offset)
    source_xy, target_xy = tuple(gap[0]), tuple(gap[-1])
    route_prefix = tuple(map(tuple, np.concatenate((prefix, gap)) + offset))
    length = float(np.linalg.norm(np.diff(np.asarray(route_prefix), axis=0), axis=1).sum())
    route = _Route(
        indices, lookup, np.asarray([-9999, 0, 1, 2, 3]),
        np.arange(len(indices), dtype=float), route_prefix, length, 0, 0,
    )
    connector = (source_xy, target_xy, tuple(map(tuple, gap)), 0)
    tree = OccludedConnectionTree(
        root, origin, shape, np.zeros(shape, dtype=bool), outside, (route,), (connector,),
    )
    full = tree.trace(target)
    assert full is not None
    assert tree.accepts_display(full)
    full_gap = gap + offset
    absolute_foreign = foreign + offset
    assert np.linalg.norm(full_gap[:, None] - absolute_foreign[None], axis=2).min() > 1.0
    blended = blend_path_with_cursor(full, root, target, 17 / 18.249)
    assert np.linalg.norm(np.asarray(blended)[:, None] - absolute_foreign[None], axis=2).min() < 1e-8
    assert not tree.accepts_display(blended)


def test_global_nearest_target_arm_cannot_be_replaced_by_deleted_neighbor():
    # The exact clicked arm is deliberately absent from the route graph. A
    # per-route nearest search would borrow x12, two pixels away, and append
    # the x10 click to it. The original all-arm distance must prohibit that.
    shape = (20, 20)
    outside = np.zeros(shape, dtype=bool)
    outside[5, 10] = True
    outside[5, 12] = True
    lookup = np.full(shape, -1, dtype=np.int32)
    lookup[5, 12] = 0
    route = _Route(np.asarray([5 * 20 + 12]), lookup, np.asarray([-9999]), np.asarray([0.0]), ((1.0, 5.0), (12.0, 5.0)), 11.0, 0, 0)
    tree = OccludedConnectionTree((1.0, 5.0), (0, 0), shape, np.zeros(shape, dtype=bool), outside, (route,), ())
    assert tree.trace((10.0, 5.0)) is None
    assert tree.trace((12.0, 5.0)) is not None


def test_short_branch_on_correct_arm_blocks_neighbor_substitution():
    # Independent review counterexample: rejecting the noisy correct tangent
    # must not delete that visible arm from the boundary's ordinal evidence.
    image, native, _, (root, target, _) = _fixture()
    native[:, 72] = False
    image[:, 72] = 255
    native[30:65, 70] = True
    native[54, 64:67] = True
    image[native] = 0
    regions = detect_label_occlusions(image)
    tree = _build(image, native, regions, root)
    assert tree is None or tree.trace((70, 46)) is None
    # No assertion that the noisy positive is solved: uncertainty declines.


def test_nearest_target_authority_also_holds_for_fractional_clicks():
    shape = (20, 20)
    outside = np.zeros(shape, dtype=bool)
    outside[5, 10] = outside[5, 12] = True
    lookup = np.full(shape, -1, dtype=np.int32)
    lookup[5, 12] = 0
    route = _Route(np.asarray([112]), lookup, np.asarray([-9999]), np.asarray([0.0]), ((1.0, 5.0), (12.0, 5.0)), 11.0, 0, 0)
    tree = OccludedConnectionTree((1.0, 5.0), (0, 0), shape, np.zeros(shape, dtype=bool), outside, (route,), ())
    assert tree.trace((10.2, 5.0)) is None
    assert tree.trace((10.75, 5.0)) is None
    assert tree.trace((11.0, 5.0)) is None
    assert tree.trace((10.9, 5.0)) is None


def test_one_route_cannot_cross_two_separate_occlusions():
    native = np.zeros((64, 112), dtype=bool)
    native[32, 4:30] = native[32, 42:65] = native[32, 77:108] = True
    image = np.full(native.shape, 255, dtype=np.uint8)
    image[native] = 0
    regions = tuple(SimpleNamespace(
        bbox_xyxy=(left, 27, left + 12, 38), mask=np.ones((11, 12), dtype=bool),
        baseline_xy=(1.0, 0.0),
    ) for left in (30, 65))
    tree = _build(image, native, regions, (15, 32))
    assert tree is not None
    assert tree.trace((50, 32)) is not None
    assert tree.trace((95, 32)) is None


def test_admissible_observed_outside_route_is_exposed_for_priority():
    image, native, regions, (root, target, _) = _fixture()
    tree = _build(image, native, regions, root)
    # The normal product wrapper preserves this still-observed alternative;
    # candidate construction itself remains independent of UI arbitration.
    outside = tree.outside_route(target)
    assert outside is not None
    assert min(y for _, y in outside) == 30
    assert not tree.intersects(outside)


def test_occlusion_mask_cannot_manufacture_an_endpoint_on_continuous_ink():
    image, native, regions, (root, target, _) = _fixture()
    # Extend the far visible arm into the glyph region. It is no longer a
    # real observed terminal; the detector mask would create a false end.
    native[60:95, 64] = True
    image[native] = 0
    regions = detect_label_occlusions(image)
    tree = _build(image, native, regions, root)
    assert tree is None or tree.trace(target) is None


def test_invalid_region_or_image_has_an_explicit_boundary_error():
    image, native, regions, (root, _, _) = _fixture()
    with pytest.raises(ValueError, match="outside"):
        _build(image, native, (SimpleNamespace(bbox_xyxy=(-1, 0, 2, 2), mask=np.ones((2, 3))),), root)
    with pytest.raises(ValueError, match="dimensions"):
        _build(image, native, (SimpleNamespace(bbox_xyxy=(1, 1, 3, 3), mask=np.ones((1, 1))),), root)
    with pytest.raises(ValueError, match="Byte"):
        _build(image.astype(float), native, regions, root)
