"""Label-gap and ink-colour augmentation keep the contour target honest."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from learning.augmentation import (
    augment_sample, recolor_contour_ink, sample_rng, synthetic_label_gaps,
)


def contour_sample(colour=(150, 90, 50)):
    """One horizontal printed contour on paper plus an unrelated black road."""
    image = np.full((512, 512, 3), 232, np.uint8)
    image[255:258, 30:482] = colour
    image[400:403, 30:482] = (25, 25, 25)
    rows = np.arange(512, dtype=np.float32)[:, None]
    centerline = np.exp(-((rows - 256.0) ** 2) / (2 * 1.5 ** 2)) * np.ones((1, 512), np.float32)
    centerline[:, :30] = 0
    centerline[:, 482:] = 0
    other = np.zeros((512, 512), np.float32)
    other[399:404, 30:482] = 1.0
    return {"image": image, "centerline": centerline.astype(np.float32), "other_ink": other,
            "valid_mask": np.ones((512, 512), bool), "sample_id": "synthetic"}


def first_cut(sample):
    for seed in range(50):
        result = synthetic_label_gaps(dict(sample), np.random.default_rng(seed))
        if not np.array_equal(result["image"], sample["image"]):
            return result
    raise AssertionError("no label gap was synthesized")


def test_label_gap_cuts_printed_ink_but_keeps_the_contour_target_continuous():
    sample = contour_sample()
    result = first_cut(sample)
    np.testing.assert_array_equal(result["centerline"], sample["centerline"])
    line = result["image"][256, 30:482].astype(int)
    assert (line.mean(axis=1) > 200).any(), "the printed contour was not interrupted"
    corridor = sample["centerline"] >= 0.3
    assert not result["other_ink"][corridor].any(), "the bridge corridor must stay passable"
    added = (result["other_ink"] > 0) & (sample["other_ink"] == 0)
    assert added.any(), "the label glyph must be taught as non-contour ink"
    np.testing.assert_array_equal(result["image"][380:420], sample["image"][380:420])


def test_ink_recolour_changes_contour_ink_only():
    sample = contour_sample()
    result = recolor_contour_ink(dict(sample), np.random.default_rng(0))
    assert not np.array_equal(result["image"][256, 40:470], sample["image"][256, 40:470])
    np.testing.assert_array_equal(result["image"][380:420], sample["image"][380:420])
    np.testing.assert_array_equal(result["centerline"], sample["centerline"])


def test_new_augmentations_are_reproducible_and_keep_supervision_geometry():
    sample = contour_sample()
    rng = lambda: sample_rng(3407, 4, sample["sample_id"])
    first = augment_sample(sample, rng(), ink_recolor=True, label_gaps=True)
    second = augment_sample(sample, rng(), ink_recolor=True, label_gaps=True)
    for key in ("image", "centerline", "other_ink", "valid_mask"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["valid_mask"].sum() == sample["valid_mask"].sum()
    assert first["centerline"].sum() == pytest.approx(sample["centerline"].sum())
    plain = augment_sample(sample, rng())
    legacy = augment_sample(sample, rng(), ink_recolor=False, label_gaps=False)
    np.testing.assert_array_equal(plain["image"], legacy["image"])
