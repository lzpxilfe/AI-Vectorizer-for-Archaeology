"""Native glyph proposals do not themselves authorize a contour bridge."""

import builtins
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

pytest.importorskip("scipy")

from ai_vectorizer.core.label_occlusion import (
    LabelOcclusion,
    combined_occlusion_mask,
    detect_label_occlusions,
)
from ai_vectorizer.core.observed_connection import ObservedConnectionCancelled


def _ring(image, x, y, width=6, height=8):
    image[y:y + height, x:x + width] = 0
    image[y + 2:y + height - 2, x + 2:x + width - 2] = 255


def _label():
    image = np.full((80, 100), 255, dtype=np.uint8)
    _ring(image, 44, 32)
    _ring(image, 52, 32)
    # An open leading glyph and an unrelated long nearby contour.
    image[32:34, 36:42] = 0
    image[34:40, 39:41] = 0
    image[8:68, 65] = 0
    return image


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_two_compact_counters_find_native_or_rotated_label(turns):
    image = np.rot90(_label(), turns)
    regions = detect_label_occlusions(image)
    assert len(regions) == 1
    region = regions[0]
    mask = combined_occlusion_mask(regions, image.shape)
    assert region.counter_count >= 2
    assert 0.0 < region.score <= 1.0
    # The detector includes the open glyph and white inter-glyph gap.
    expected = np.rot90(np.indices((80, 100))[1] == 65, turns)
    assert not mask[expected].any()
    ink = np.rot90(_label() == 0, turns)
    assert int((mask & ink).sum()) > 60
    assert int((mask & ~ink).sum()) > 10
    assert tuple(mask.shape) == tuple(image.shape)
    assert not mask.flags.writeable
    assert not region.mask.flags.writeable
    assert len(region.polygon_xy) >= 4


def test_rgb_blue_ink_and_scan_background_use_same_native_geometry():
    gray = _label()
    rgb = np.full((*gray.shape, 3), (225, 230, 220), dtype=np.uint8)
    rgb[gray == 0] = (10, 20, 140)
    before = rgb.copy()
    regions = detect_label_occlusions(rgb)
    assert len(regions) == 1
    assert np.array_equal(rgb, before)
    assert regions[0].bbox_xyxy == detect_label_occlusions(gray)[0].bbox_xyxy


@pytest.mark.parametrize("angle", [-45, -30, -15, 15, 30, 45, 60])
def test_diagonal_native_glyph_groups(angle):
    from scipy.ndimage import rotate

    image = rotate(_label(), angle, reshape=True, order=0, cval=255)
    regions = detect_label_occlusions(image)
    assert len(regions) == 1
    assert regions[0].counter_count >= 2


def test_curved_baseline_includes_nearby_leading_open_glyph():
    image = np.full((80, 100), 255, dtype=np.uint8)
    _ring(image, 44, 30)
    _ring(image, 52, 28)
    image[35:37, 38:42] = 0
    image[37:43, 38:40] = 0
    regions = detect_label_occlusions(image)
    assert len(regions) == 1
    mask = combined_occlusion_mask(regions, image.shape)
    assert mask[41, 38]
    # Axis-aligned bbox corner is still excluded by the oriented footprint.
    x0, y0, _, _ = regions[0].bbox_xyxy
    assert not mask[y0, x0]


@pytest.mark.parametrize("kind", ["u", "loop", "nested", "grid", "letters", "double-loop"])
def test_contours_grid_and_open_letter_strokes_are_not_text_proposals(kind):
    image = np.full((96, 96), 255, dtype=np.uint8)
    if kind == "u":
        image[10:80, 20] = image[10:80, 70] = 0
        image[79, 20:71] = 0
    elif kind == "loop":
        _ring(image, 30, 30, 10, 14)
    elif kind == "nested":
        for offset in [0, 8, 16]:
            x0, x1 = 10 + offset, 85 - offset
            image[10:80 - offset, x0] = image[10:80 - offset, x1] = 0
            image[79 - offset, x0:x1 + 1] = 0
    elif kind == "grid":
        image[::10, :] = image[:, ::10] = 0
    elif kind == "letters":
        for x in [20, 32, 44]:
            image[30:42, x:x + 2] = 0
            image[40:42, x:x + 8] = 0
    else:
        _ring(image, 30, 30)
        _ring(image, 38, 30)
        image[33:35, 35:39] = 0
    assert detect_label_occlusions(image) == ()


def test_ring_grid_cells_have_no_elongated_counter_agreement():
    image = np.full((80, 100), 255, dtype=np.uint8)
    for x in [20, 30, 40]:
        _ring(image, x, 30, width=8, height=8)
    assert detect_label_occlusions(image) == ()


def test_oversized_contour_parent_cannot_be_cut_as_compact_glyph():
    image = _label()
    image[36, 57:95] = 0  # Attach a long contour to one zero.
    assert detect_label_occlusions(image) == ()


def test_parent_shared_by_two_counters_is_rejected():
    image = _label()
    image[35, 49:53] = 0
    assert detect_label_occlusions(image) == ()


@pytest.mark.parametrize("image", [
    np.zeros((30, 30), dtype=np.uint8),
    np.full((30, 30), 255, dtype=np.uint8),
    np.full((30, 30, 3), 150, dtype=np.uint8),
])
def test_uniform_map_does_not_produce_occlusion(image):
    assert detect_label_occlusions(image) == ()


def test_dense_counter_noise_abandons_whole_optional_detector():
    image = np.full((512, 512), 255, dtype=np.uint8)
    for y in range(4, 500, 12):
        for x in range(4, 500, 12):
            _ring(image, x, y)
    assert detect_label_occlusions(image) == ()


def test_retained_proposals_have_a_fixed_region_limit():
    image = np.full((160, 450), 255, dtype=np.uint8)
    for y in range(5, 150, 16):
        for x in range(5, 430, 48):
            _ring(image, x, y)
            _ring(image, x + 8, y)
    assert len(detect_label_occlusions(image)) == 64


@pytest.mark.parametrize("image", [
    np.empty((0, 10), dtype=np.uint8),
    np.zeros((10, 10), dtype=np.float32),
    np.zeros((10, 10, 4), dtype=np.uint8),
    np.zeros((1025, 1025), dtype=np.uint8),
    np.zeros((5,), dtype=np.uint8),
])
def test_input_validation_rejects_unsupported_native_images(image):
    with pytest.raises(ValueError):
        detect_label_occlusions(image)


def test_cancel_before_scipy_or_retention():
    with pytest.raises(ObservedConnectionCancelled):
        detect_label_occlusions(_label(), cancel_check=lambda: True)
    calls = []

    def cancel_later():
        calls.append(1)
        return len(calls) >= 3

    with pytest.raises(ObservedConnectionCancelled):
        detect_label_occlusions(_label(), cancel_check=cancel_later)


def test_missing_scipy_can_be_handled_by_existing_route(monkeypatch):
    original = builtins.__import__

    def without_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("optional SciPy unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_scipy)
    with pytest.raises(ImportError):
        detect_label_occlusions(_label())


def test_snapshot_is_copied_immutable_and_bounds_checked():
    mask = np.ones((3, 4), dtype=bool)
    region = LabelOcclusion((2, 3, 6, 6), mask, (1, 0), 2, 0.8)
    mask[:] = False
    assert region.mask.all()
    with pytest.raises(FrozenInstanceError):
        region.score = 0.5
    with pytest.raises(ValueError):
        region.mask[0, 0] = False
    with pytest.raises(ValueError):
        combined_occlusion_mask([region], (5, 6))
    with pytest.raises(ValueError):
        combined_occlusion_mask([object()], (10, 10))
    with pytest.raises(ValueError):
        combined_occlusion_mask([], (1025, 1025))


@pytest.mark.parametrize("bounds,mask,axis,count,score", [
    ((0, 0, 0, 3), np.ones((3, 0)), (1, 0), 2, 0.5),
    ((0, 0, 4, 3), np.ones((3, 3)), (1, 0), 2, 0.5),
    ((0, 0, 4, 3), np.ones((3, 4)), (0, 0), 2, 0.5),
    ((0, 0, 4, 3), np.ones((3, 4)), (1, 0), 1, 0.5),
    ((0, 0, 4, 3), np.ones((3, 4)), (1, 0), 2, float("nan")),
])
def test_invalid_occlusion_snapshots_fail(bounds, mask, axis, count, score):
    with pytest.raises(ValueError):
        LabelOcclusion(bounds, mask, axis, count, score)
