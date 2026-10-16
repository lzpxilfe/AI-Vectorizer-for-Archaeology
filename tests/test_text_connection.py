"""Source text proposals use genuine ends and preserve the Ink fallback."""

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from ai_vectorizer.core import edge_detector
from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree, LiveWireCancelled
from ai_vectorizer.core.observed_connection import ObservedConnectionCancelled
from ai_vectorizer.core.occluded_connection import _glyph_core, build_occluded_connection
from ai_vectorizer.core.text_occlusion import detect_text_occlusions
from tests.test_occluded_connection import _graph
from tests.test_text_occlusion import _word


def _scene(turns=0):
    image = _word("blue")
    native = np.zeros(image.shape[:2], dtype=bool)
    native[8:26, 39] = native[48:86, 39] = True
    native[8:86, 82] = True
    image[native] = (15, 15, 15)
    points = [(39, 15), (39, 75), (82, 75)]
    for _ in range(turns):
        width = image.shape[1]
        image, native = np.rot90(image), np.rot90(native)
        points = [(y, width - 1 - x) for x, y in points]
    return image, native, points


@pytest.mark.parametrize("turns", range(4))
def test_general_text_connector_requires_original_ends_and_excludes_neighbor(turns):
    image, native, (root, target, neighbor) = _scene(turns)
    before = native.copy()
    regions = detect_text_occlusions(image)
    tree = build_occluded_connection(
        _graph(native.shape), native, root, (0, 0), regions,
        image=image, sparse=sparse, dijkstra=dijkstra,
    )
    assert tree is not None
    route = tree.trace(target)
    assert route and route[0] == root and route[-1] == target
    assert tree.accepts_display(route)
    assert tree.trace(neighbor) is None
    np.testing.assert_array_equal(native, before)


def test_text_context_hull_cannot_erase_visible_ink_in_glyph_holes():
    region = detect_text_occlusions(_word("blue"))[0]
    full = np.zeros((96, 140), dtype=bool)
    x0, y0, x1, y1 = region.bbox_xyxy
    full[y0:y1, x0:x1] = region.mask
    core = _glyph_core(region, full)
    assert full[33, 39] and not core[33, 39]
    assert core[30, 34]
    with pytest.raises(ValueError, match="dimensions"):
        _glyph_core(SimpleNamespace(bbox_xyxy=region.bbox_xyxy, core_mask=np.ones((1, 1))), full)
    with pytest.raises(ValueError, match="inside"):
        _glyph_core(region, np.zeros_like(full))


@pytest.mark.parametrize("reverse", [False, True])
def test_text_bridge_cannot_escape_via_a_distant_grid_to_the_neighbor(reverse):
    image, native, (root, target, neighbor) = _scene()
    native[80, 39:83] = True
    image[native] = (15, 15, 15)
    regions = detect_text_occlusions(image)
    if reverse:
        root, neighbor = neighbor, root
    tree = build_occluded_connection(
        _graph(native.shape), native, root, (0, 0), regions,
        image=image, sparse=sparse, dijkstra=dijkstra,
    )
    assert tree is not None
    assert tree.trace(neighbor) is None
    if not reverse:
        assert tree.trace(target) is not None
    # The rejected route really is reachable through this native network;
    # ordinary global connectivity cannot establish contour identity.
    unsafe = replace(tree, routes=tuple(replace(route, require_unbranched=False) for route in tree.routes))
    assert unsafe.trace(neighbor) is not None


def _livewire_inputs():
    image, _native, (root, target, _neighbor) = _scene()
    with mock.patch.object(edge_detector, "_skimage_skeletonize", None), mock.patch.object(
        edge_detector, "get_cv2", return_value=None,
    ):
        evidence = edge_detector.EdgeDetector.detect_ink_evidence(image)
    return image, evidence, root, target


def test_bad_optional_text_snapshot_preserves_exact_existing_ink_tree():
    image, evidence, root, target = _livewire_inputs()
    args = (image, evidence.centerline.astype(np.uint8) * 255, root)
    kwargs = dict(strength=1.0, evidence=evidence, config=LiveWireConfig(max_window_size=140))
    with mock.patch("ai_vectorizer.core.text_occlusion.detect_text_occlusions", return_value=()):
        before = build_livewire_tree(*args, **kwargs)
    with mock.patch("ai_vectorizer.core.text_occlusion.detect_text_occlusions", side_effect=ValueError("bad proposal")):
        after = build_livewire_tree(*args, **kwargs)
    assert after.text_occluded_connection is None
    assert after.trace(target) == before.trace(target)
    np.testing.assert_array_equal(after.predecessors, before.predecessors)


def test_text_cancellation_propagates_and_disabled_modes_do_not_detect():
    image, evidence, root, _target = _livewire_inputs()
    args = (image, evidence.centerline.astype(np.uint8) * 255, root)
    config = LiveWireConfig(max_window_size=140)
    with mock.patch("ai_vectorizer.core.text_occlusion.detect_text_occlusions", side_effect=ObservedConnectionCancelled("cancel")):
        with pytest.raises(LiveWireCancelled):
            build_livewire_tree(*args, strength=1, evidence=evidence, config=config)
    with mock.patch("ai_vectorizer.core.text_occlusion.detect_text_occlusions") as detector:
        build_livewire_tree(*args, strength=0, evidence=evidence, config=config)
        build_livewire_tree(*args, strength=1, evidence=evidence, config=replace(config, automatic_label_occlusion=False))
        detector.assert_not_called()
