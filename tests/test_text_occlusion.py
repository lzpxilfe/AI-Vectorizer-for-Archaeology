"""General text uncertainty never changes native contour observations."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

pytest.importorskip("scipy")

from ai_vectorizer.core.label_occlusion import detect_label_occlusions
from ai_vectorizer.core.observed_connection import ObservedConnectionCancelled
from ai_vectorizer.core.text_occlusion import TextOcclusion, detect_text_occlusions


def _word(ink_class="gray"):
    """Three open, differently shaped letters; no counters or font package."""
    gray = np.full((96, 140), 255, dtype=np.uint8)
    for x, kind in ((34, "E"), (50, "H"), (66, "T")):
        if kind == "E":
            gray[30:44, x:x + 2] = 0
            for y in (30, 36, 42):
                gray[y:y + 2, x:x + 10] = 0
        elif kind == "H":
            gray[30:44, x:x + 2] = gray[30:44, x + 8:x + 10] = 0
            gray[36:38, x:x + 10] = 0
        else:
            gray[30:32, x:x + 10] = 0
            gray[32:44, x + 4:x + 6] = 0
    if ink_class == "gray":
        return gray
    rgb = np.full((*gray.shape, 3), (230, 234, 225), dtype=np.uint8)
    rgb[gray == 0] = (10, 15, 140)
    return rgb


@pytest.mark.parametrize("ink_class", ["gray", "blue"])
@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_open_letter_groups_have_separate_native_ink_snapshots(ink_class, turns):
    image = np.rot90(_word(ink_class), turns)
    before = image.copy()
    assert detect_label_occlusions(image) == ()
    regions = detect_text_occlusions(image)
    assert len(regions) == 3
    assert all(isinstance(region, TextOcclusion) for region in regions)
    assert all(region.ink_class == ink_class for region in regions)
    assert all(region.glyph_count == 3 and region.branched_count >= 2 for region in regions)
    assert all(not hasattr(region, "counter_count") for region in regions)
    np.testing.assert_array_equal(image, before)
    ink = np.any(image < 150, axis=2) if image.ndim == 3 else image == 0
    core_union = np.zeros(image.shape[:2], dtype=bool)
    for region in regions:
        x0, y0, x1, y1 = region.bbox_xyxy
        core_union[y0:y1, x0:x1] |= region.core_mask
        assert not region.mask.flags.writeable
        assert not region.core_mask.flags.writeable
        assert np.all(region.core_mask <= region.mask)
    np.testing.assert_array_equal(core_union, ink)


def test_word_spaces_and_visible_black_ink_in_blue_glyph_hole_are_preserved():
    image = _word("blue")
    # Inside the E footprint, this is independent source ink, not blue text.
    image[33:35, 38:42] = (15, 15, 15)
    regions = detect_text_occlusions(image)
    assert len(regions) == 3
    core = np.zeros(image.shape[:2], dtype=bool)
    context = np.zeros_like(core)
    for region in regions:
        x0, y0, x1, y1 = region.bbox_xyxy
        core[y0:y1, x0:x1] |= region.core_mask
        context[y0:y1, x0:x1] |= region.mask
    assert context[33, 39]
    assert not core[33:35, 38:42].any()
    assert not core[:, 46:48].any()
    assert not context[:, 46:48].any()


@pytest.mark.parametrize("kind", [
    "single", "pair", "contours", "river", "grid", "dashes", "elbows",
    "closed-islands", "repeated-trees", "oversized", "border",
])
@pytest.mark.parametrize("ink_class", ["gray", "blue"])
def test_non_text_native_geometry_cannot_authorize_a_text_group(kind, ink_class):
    from scipy import ndimage as ndi

    source = _word()
    image = np.full(source.shape, 255, dtype=np.uint8)
    if kind == "single":
        image[:, 30:47] = source[:, 30:47]
    elif kind == "pair":
        image[:, 30:63] = source[:, 30:63]
    elif kind in ("contours", "river"):
        for row in (22, 45, 68):
            image[row, 8:132] = 0
            if kind == "river":
                for step in range(12):
                    image[row - step, 50 + step] = 0
    elif kind == "grid":
        image[::12, :] = image[:, ::12] = 0
    elif kind == "dashes":
        for x in (30, 45, 60, 75):
            image[34:36, x:x + 8] = 0
    elif kind == "elbows":
        for x in (30, 45, 60, 75):
            image[30:44, x:x + 2] = 0
            image[42:44, x:x + 9] = 0
    elif kind == "closed-islands":
        for x in (30, 48, 66):
            image[30:44, x:x + 10] = 0
            image[32:42, x + 2:x + 8] = 255
    elif kind == "repeated-trees":
        for x in (30, 48, 66):
            image[30:44, x + 4:x + 6] = 0
            image[33:35, x:x + 10] = image[39:41, x:x + 10] = 0
    elif kind == "oversized":
        image = ndi.zoom(source, 3, order=0)
    else:
        image = np.roll(source, -32, axis=1)
    if ink_class == "blue":
        rgb = np.full((*image.shape, 3), 255, dtype=np.uint8)
        rgb[image == 0] = (10, 15, 140)
        image = rgb
    assert detect_text_occlusions(image) == ()


def test_long_blue_stroke_attached_to_a_glyph_is_not_partly_swallowed():
    image = _word("blue")
    image[31, 5:35] = (10, 15, 140)
    # The whole E component is now a long stroke. The remaining pair alone
    # cannot identify text or borrow the rejected E's former local pixels.
    assert detect_text_occlusions(image) == ()


@pytest.mark.parametrize("ink_class", ["gray", "blue"])
def test_two_letters_and_a_collinear_slash_decline_without_invalid_polygon(ink_class):
    image = _word()
    image[:, 64:] = 255
    for offset in range(8):
        image[32 + offset, 66 + offset] = 0
    if ink_class == "blue":
        rgb = np.full((*image.shape, 3), 255, dtype=np.uint8)
        rgb[image == 0] = (10, 15, 140)
        image = rgb
    assert detect_text_occlusions(image) == ()


@pytest.mark.parametrize("ink_class", ["gray", "blue"])
def test_repeated_tree_layout_declines_despite_scan_scale_variation(ink_class):
    image = np.full((96, 140), 255, dtype=np.uint8)
    for x, height, width in ((30, 14, 10), (48, 13, 10), (66, 12, 8)):
        image[30:30 + height, x + width // 2:x + width // 2 + 2] = 0
        image[33:35, x:x + width] = image[39:41, x:x + width] = 0
    if ink_class == "blue":
        rgb = np.full((*image.shape, 3), 255, dtype=np.uint8)
        rgb[image == 0] = (10, 15, 140)
        image = rgb
    assert detect_text_occlusions(image) == ()


def _structural_letter(heading=True):
    image = np.full((100, 130), 255, dtype=np.uint8)
    image[38:54, 52:68] = 0
    image[40:52, 54:58] = image[40:52, 61:66] = 255
    if heading:
        image[32:34, 53:66] = 0
    return image


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_structural_two_cell_letter_requires_its_separate_heading(turns):
    image = np.rot90(_structural_letter(), turns)
    assert detect_label_occlusions(image) == ()
    regions = detect_text_occlusions(image)
    assert len(regions) == 1
    region = regions[0]
    assert region.evidence_kind == "cells_and_heading"
    assert region.glyph_count == 2 and region.branched_count == 1 and region.cell_count == 2
    assert region.ink_class == "gray"
    assert not hasattr(region, "counter_count")
    core = np.zeros(image.shape, dtype=bool)
    x0, y0, x1, y1 = region.bbox_xyxy
    core[y0:y1, x0:x1] = region.core_mask
    np.testing.assert_array_equal(core, image == 0)
    assert detect_text_occlusions(np.rot90(_structural_letter(False), turns)) == ()


@pytest.mark.parametrize("kind", ["misaligned-heading", "far-heading", "ambiguous-heading", "attached-heading", "long-parent"])
def test_structural_signature_declines_unrelated_or_ambiguous_lines(kind):
    image = _structural_letter(False)
    if kind == "misaligned-heading":
        image[22:35, 58:60] = 0
    elif kind == "far-heading":
        image[24:26, 53:66] = 0
    elif kind == "ambiguous-heading":
        image[32:34, 53:66] = image[58:60, 53:66] = 0
    elif kind == "attached-heading":
        image[32:34, 53:66] = 0
        image[34:39, 58:60] = 0
    else:
        image[32:34, 53:66] = 0
        image[45, 67:110] = 0
    assert detect_text_occlusions(image) == ()


def test_cancellation_propagates_before_and_during_group_search():
    with pytest.raises(ObservedConnectionCancelled):
        detect_text_occlusions(_word(), cancel_check=lambda: True)
    calls = []

    def cancel_during():
        calls.append(True)
        return len(calls) >= 4

    with pytest.raises(ObservedConnectionCancelled):
        detect_text_occlusions(_word(), cancel_check=cancel_during)
    assert len(calls) == 4


def test_crowded_channel_declines_without_partial_letter_groups():
    image = np.full((512, 512), 255, dtype=np.uint8)
    letter = _word()[30:44, 34:44]
    for y in range(30, 480, 16):
        for x in range(30, 480, 16):
            image[y:y + 14, x:x + 10] = letter
    assert detect_text_occlusions(image) == ()


@pytest.mark.parametrize("image", [
    np.empty((0, 10), dtype=np.uint8),
    np.zeros((30, 30), dtype=np.float32),
    np.zeros((30, 30, 4), dtype=np.uint8),
    np.zeros((1025, 1025), dtype=np.uint8),
])
def test_unsupported_source_pixels_are_rejected(image):
    with pytest.raises(ValueError):
        detect_text_occlusions(image)


def test_snapshot_copies_source_masks_and_is_frozen():
    region = detect_text_occlusions(_word())[0]
    mask, core = region.mask.copy(), region.core_mask.copy()
    snapshot = replace(region, mask=mask, core_mask=core)
    mask[:] = core[:] = False
    assert snapshot.mask.any() and snapshot.core_mask.any()
    with pytest.raises(FrozenInstanceError):
        snapshot.score = 0.0


@pytest.mark.parametrize("changes", [
    {"bbox_xyxy": (0, 0, 0, 4)},
    {"baseline_xy": (2.0, 0.0)},
    {"glyph_count": 2},
    {"glyph_count": True},
    {"branched_count": 1},
    {"score": float("nan")},
    {"ink_class": "contour"},
    {"polygon_xy": ()},
    {"core_mask": np.zeros((1, 1), dtype=bool)},
    {"evidence_kind": "cells_and_heading"},
    {"evidence_kind": "unknown"},
    {"cell_count": 2},
])
def test_invalid_snapshot_cannot_claim_text_authority(changes):
    region = detect_text_occlusions(_word())[0]
    with pytest.raises(ValueError):
        replace(region, **changes)
