"""Label-gap and ink-colour augmentation keep the contour target honest."""

import numpy as np
import pytest

pytest.importorskip("PIL")

from learning.augmentation import (
    augment_sample, recolor_contour_ink, sample_rng, synthetic_label_gaps, synthetic_word_gaps,
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


def parallel_sample():
    sample = contour_sample()
    rows = np.arange(512, dtype=np.float32)[:, None]
    neighbor = np.exp(-((rows - 272.0) ** 2) / (2 * 1.5 ** 2)) * np.ones((1, 512), np.float32)
    neighbor[:, :30] = neighbor[:, 482:] = 0
    sample["centerline"] = np.maximum(sample["centerline"], neighbor).astype(np.float32)
    sample["image"][271:274, 30:482] = (30, 30, 30)
    return sample


def word_examples(sample, count=40):
    for seed in range(count):
        yield synthetic_word_gaps(dict(sample), sample_rng(seed, 0, sample["sample_id"]))


def test_synthetic_words_keep_both_parallel_contours_positive_and_target_geometry_exact():
    sample = parallel_sample()
    examples = list(word_examples(sample))
    changed = [value for value in examples if value.get("synthetic_word_gap_events")]
    assert changed
    assert any(event["parallel_contour_pixels"] >= 8 for value in changed for event in value["synthetic_word_gap_events"])
    assert {event["style"] for value in changed for event in value["synthetic_word_gap_events"]} == {"overprint", "white_clearance"}
    assert any(any(ord(character) > 127 for character in event["text"])
               for value in changed for event in value["synthetic_word_gap_events"])
    for value in changed:
        np.testing.assert_array_equal(value["centerline"], sample["centerline"])
        np.testing.assert_array_equal(value["valid_mask"], sample["valid_mask"])
        assert not value["other_ink"][sample["centerline"] >= 0.3].any()
        assert value["centerline"][256, 100] == 1.0
        assert value["centerline"][272, 100] == 1.0
        assert ((value["other_ink"] > 0) & (sample["other_ink"] == 0)).any()
        np.testing.assert_array_equal(value["image"][380:420], sample["image"][380:420])


def test_word_curriculum_never_paints_ignore_or_invents_preexisting_hidden_truth():
    sample = parallel_sample()
    sample["valid_mask"][220:300, 200:280] = False
    sample["centerline"][:, 235:245] = 0
    for value in word_examples(sample, count=20):
        np.testing.assert_array_equal(value["image"][~sample["valid_mask"]], sample["image"][~sample["valid_mask"]])
        np.testing.assert_array_equal(value["centerline"], sample["centerline"])
        assert not value["centerline"][:, 235:245].any()


def test_word_curriculum_rng_is_stable_and_default_flag_preserves_legacy_bytes():
    sample = parallel_sample()
    rng = lambda: sample_rng(3407, 4, sample["sample_id"])
    first = augment_sample(sample, rng(), ink_recolor=True, label_gaps=True, word_gaps=True)
    second = augment_sample(sample, rng(), ink_recolor=True, label_gaps=True, word_gaps=True)
    for key in ("image", "centerline", "other_ink", "valid_mask"):
        np.testing.assert_array_equal(first[key], second[key])
    ordinary = augment_sample(sample, rng(), ink_recolor=True, label_gaps=True)
    disabled = augment_sample(sample, rng(), ink_recolor=True, label_gaps=True, word_gaps=False)
    for key in ("image", "centerline", "other_ink", "valid_mask"):
        np.testing.assert_array_equal(ordinary[key], disabled[key])


def test_stroke_atlas_requires_no_host_font_and_refuses_missing_characters(monkeypatch):
    from PIL import ImageFont
    from learning.text_glyphs import render_word_mask

    monkeypatch.setattr(ImageFont, "load_default", lambda *_a, **_k: pytest.fail("unexpected host font"))
    monkeypatch.setattr(ImageFont, "truetype", lambda *_a, **_k: pytest.fail("unexpected external font"))
    first = render_word_mask("北面", scale=2, angle=35, tracking=1, stroke=1)
    second = render_word_mask("北面", scale=2, angle=35, tracking=1, stroke=1)
    np.testing.assert_array_equal(first, second)
    assert first.ndim == 2 and first.max() == 1.0 and first.min() == 0.0
    with pytest.raises(ValueError, match="declared"):
        render_word_mask("🙂", scale=2, angle=0)


def test_word_training_option_is_explicit_and_atlas_bytes_are_bound():
    from dataclasses import asdict
    from learning.train import TrainConfig, code_identity

    assert TrainConfig().word_gaps is False
    assert TrainConfig(**{key: value for key, value in asdict(TrainConfig()).items() if key != "word_gaps"}).word_gaps is False
    with pytest.raises(ValueError, match="word_gaps"):
        TrainConfig(word_gaps=1).validate()
    identity = code_identity()
    assert "learning/text_glyphs.py" in identity["source_files"]
