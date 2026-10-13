"""Explicit colour sampling and route discrimination on controlled map ink."""

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from ai_vectorizer.core.edge_detector import EdgeDetector
from ai_vectorizer.core.ink_color import (
    InkColorProfile,
    guidance_from_ink_color,
    sample_ink_color,
)
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.livewire import (
    LiveWireConfig,
    build_livewire_tree,
    is_livewire_available,
)
from ai_vectorizer.core.smart_recovery import build_corridor_cost_map


BROWN = (156, 104, 60)
BLACK = (24, 24, 24)
BLUE = (45, 105, 170)


def _evidence(mask, *, score=None, tangent_x=None, tangent_y=None):
    return LineEvidence(
        center_score=mask.astype(np.float32) if score is None else score,
        centerline=mask,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        coherence=mask.astype(np.float32) if tangent_x is not None else None,
    )


def _sample_fixture():
    image = np.full((17, 21, 3), 245, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[8, 3:18] = True
    image[mask] = BROWN
    return image, _evidence(mask)


def test_profile_is_immutable_and_independent_of_the_image():
    image, evidence = _sample_fixture()
    profile = sample_ink_color(image, evidence, (8, 8))
    assert isinstance(profile, InkColorProfile)
    assert profile.sample_rgb == BROWN
    assert isinstance(profile.chroma_direction, tuple)
    assert math.isclose(np.linalg.norm(profile.chroma_direction), 1.0)
    image[8, 8] = BLUE
    assert profile.sample_rgb == BROWN
    with pytest.raises(FrozenInstanceError):
        profile.sample_rgb = BLUE


def test_sample_requires_bounded_nearby_observed_ink():
    image, evidence = _sample_fixture()
    assert sample_ink_color(image, evidence, (8, 10)).sample_rgb == BROWN
    assert sample_ink_color(image, evidence, (8, 10.01)) is None
    assert sample_ink_color(image, evidence, (8, 2)) is None
    assert sample_ink_color(image, evidence, (-0.1, 8)) is None
    blank_evidence = _evidence(np.zeros(evidence.shape, dtype=bool))
    assert sample_ink_color(image, blank_evidence, (8, 8)) is None


def test_strong_continuous_support_can_be_sampled_without_a_binary_line():
    image, evidence = _sample_fixture()
    mask = np.zeros(evidence.shape, dtype=bool)
    score = evidence.center_score.copy()
    supported = _evidence(mask, score=score)
    assert sample_ink_color(image, supported, (8, 8)).sample_rgb == BROWN
    score *= np.float32(0.2)
    assert sample_ink_color(image, _evidence(mask, score=score), (8, 8)) is None


@pytest.mark.parametrize("other_color", [BLACK, BLUE])
def test_equally_near_different_styles_are_refused(other_color):
    image, evidence = _sample_fixture()
    mask = evidence.centerline.copy()
    mask[10, 3:18] = True
    image[10, 3:18] = other_color
    supported = _evidence(mask)
    assert sample_ink_color(image, supported, (8, 9)) is None
    # A precise click on one of those lines remains authoritative.
    assert sample_ink_color(image, supported, (8, 8)).sample_rgb == BROWN


def test_unsupported_stretches_grayscale_and_neutral_ink_are_refused():
    image, evidence = _sample_fixture()
    for unsupported in (
        image.astype(np.uint16),
        image.astype(np.float32),
        image.astype(np.float32) / 255.0,
        image[..., 0],
        image[..., :1],
    ):
        assert sample_ink_color(unsupported, evidence, (8, 8)) is None
    image[evidence.centerline] = BLACK
    assert sample_ink_color(image, evidence, (8, 8)) is None


def test_bad_input_contracts_are_rejected():
    image, evidence = _sample_fixture()
    for point in ((8, np.nan), (8, np.inf), (True, 8), (8,), "8,8", None):
        with pytest.raises(ValueError, match="point"):
            sample_ink_color(image, evidence, point)
    for radius in (-1, np.nan, np.inf, True, 9):
        with pytest.raises(ValueError, match="sample_radius"):
            sample_ink_color(image, evidence, (8, 8), sample_radius=radius)
    with pytest.raises(ValueError, match="dimensions"):
        sample_ink_color(image[:-1], evidence, (8, 8))
    with pytest.raises(TypeError, match="evidence"):
        sample_ink_color(image, None, (8, 8))
    for rgb in ((12, 12, 12), (256, 20, 0), (1, 2), (1.0, 2, 3), (True, 2, 3)):
        with pytest.raises(ValueError, match="sample_rgb"):
            InkColorProfile(rgb)
    with pytest.raises(TypeError, match="profile"):
        guidance_from_ink_color(image, evidence, None)
    with pytest.raises(ValueError, match="native Byte"):
        guidance_from_ink_color(image.astype(np.uint16), evidence, InkColorProfile(BROWN))


def test_guidance_retains_matches_and_gaps_and_never_mutates_evidence():
    image, evidence = _sample_fixture()
    image[8, 10:13] = BLACK
    image[8, 15:18] = BLUE
    score = evidence.center_score.copy()
    mask = evidence.centerline.copy()
    score[8, 13:15] = 0.0
    mask[8, 13:15] = False
    supported = _evidence(mask, score=score)
    profile = sample_ink_color(image, supported, (5, 8))
    original_score = supported.center_score.copy()
    original_mask = supported.centerline.copy()
    guidance = guidance_from_ink_color(image, supported, profile)
    penalty = guidance.avoidance_score
    assert not penalty.flags.writeable
    assert np.isfinite(penalty).all()
    assert np.all(penalty[8, 3:10] == 0.0)
    assert np.all(penalty[8, 10:13] == 1.0)
    assert np.all(penalty[8, 13:15] == 0.0)
    assert np.all(penalty[8, 15:18] == 1.0)
    assert np.all(penalty[~mask] == 0.0)
    np.testing.assert_array_equal(supported.center_score, original_score)
    np.testing.assert_array_equal(supported.centerline, original_mask)


def test_whiteward_antialias_and_rotated_byte_rgb_preserve_the_selected_style():
    image, evidence = _sample_fixture()
    profile = sample_ink_color(image, evidence, (5, 8))
    faded = np.rint(255.0 - (255.0 - np.asarray(BROWN)) * 0.22).astype(np.uint8)
    image[8, 9:18] = faded
    guidance = guidance_from_ink_color(image, evidence, profile)
    assert np.all(guidance.avoidance_score == 0.0)
    rotated_image = np.rot90(image)
    rotated_evidence = _evidence(np.rot90(evidence.centerline))
    rotated_guidance = guidance_from_ink_color(rotated_image, rotated_evidence, profile)
    np.testing.assert_array_equal(rotated_guidance.avoidance_score, np.rot90(guidance.avoidance_score))


def test_recovery_cost_uses_the_same_finite_soft_penalty():
    image, evidence = _sample_fixture()
    profile = sample_ink_color(image, evidence, (5, 8))
    image[8, 11:18] = BLACK
    guidance = guidance_from_ink_color(image, evidence, profile)
    corridor = np.ones(evidence.shape, dtype=np.float32)
    base = build_corridor_cost_map(evidence, corridor)
    guided = build_corridor_cost_map(evidence, corridor, guidance=guidance)
    assert np.isfinite(guided).all()
    np.testing.assert_array_equal(guided[8, 3:11], base[8, 3:11])
    assert np.all(guided[8, 11:18] > base[8, 11:18])


def _brown_shortcut_fixture():
    height, width = 82, 145
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    tangent_x = np.zeros((height, width), dtype=np.float32)
    tangent_y = np.zeros((height, width), dtype=np.float32)
    desired = []
    for x in range(8, 137):
        phase = math.pi * (x - 8) / 128.0
        y = int(round(52.0 - 27.0 * math.sin(phase)))
        desired.append((x, y))
        mask[y, x] = True
        image[y, x] = BROWN
        tangent_x[y, x] = 1.0
        tangent_y[y, x] = -27.0 * math.pi / 128.0 * math.cos(phase)
    # A black road cuts between the two contour endpoints.  A crossing
    # glyph-like stem offers another distracting route near the contour.
    mask[52, 8:137] = True
    image[52, 8:137] = BLACK
    tangent_x[52, 8:137] = 1.0
    tangent_y[52, 8:137] = 0.0
    mask[18:60, 82] = True
    image[18:60, 82] = BLACK
    tangent_x[18:60, 82] = 0.0
    tangent_y[18:60, 82] = 1.0
    return image, _evidence(mask, tangent_x=tangent_x, tangent_y=tangent_y), desired


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("detected", [False, True])
def test_selected_brown_contour_wins_over_black_shortcut_and_crossing(rotate, detected):
    image, evidence, desired = _brown_shortcut_fixture()
    profile = sample_ink_color(image, evidence, desired[10])
    assert profile is not None
    if rotate:
        original_width = image.shape[1]
        image = np.rot90(image)
        evidence = _evidence(
            np.rot90(evidence.centerline),
            tangent_x=np.rot90(evidence.tangent_y),
            tangent_y=-np.rot90(evidence.tangent_x),
        )
        desired = [(y, original_width - 1 - x) for x, y in desired]
    if detected:
        # Exercise real multiscale Ink evidence as well as the controlled
        # graph fixture; a hand-made direction field is not enough to show
        # the detector-to-preview behaviour the user actually receives.
        evidence = EdgeDetector.detect_ink_evidence(image)
    edges = np.zeros(evidence.shape, dtype=np.uint8)
    config = LiveWireConfig(max_window_size=160, target_snap_radius=0)
    root, target = desired[0], desired[-1]
    baseline = build_livewire_tree(image, edges, root, evidence=evidence, config=config)
    guidance = guidance_from_ink_color(image, evidence, profile)
    guided_tree = build_livewire_tree(
        image, edges, root, evidence=evidence, guidance=guidance, config=config
    )
    baseline_path = np.asarray(baseline.trace(target))
    guided_path = np.asarray(guided_tree.trace(target))
    reference = np.asarray(desired)
    def distance_to_contour(path):
        return np.sqrt(((path[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    assert float(distance_to_contour(baseline_path).max()) > 20.0
    assert float(distance_to_contour(guided_path).max()) <= 2.0
    assert guided_tree.trace(target)[0] == tuple(float(value) for value in root)
    assert guided_tree.trace(target)[-1] == tuple(float(value) for value in target)
    # Clearing a profile uses the exact historical graph.  Colour guidance
    # changes no global feature, pixel mask or default kernel configuration.
    cleared = build_livewire_tree(image, edges, root, evidence=evidence, guidance=None, config=config)
    np.testing.assert_array_equal(cleared.predecessors, baseline.predecessors)
    np.testing.assert_array_equal(cleared.distances, baseline.distances)
    zero = build_livewire_tree(
        image, edges, root, evidence=evidence, guidance=guidance,
        strength=0.0, config=config,
    )
    assert zero.trace(target) == [tuple(float(value) for value in root), tuple(float(value) for value in target)]


def test_a_wrong_profile_penalizes_the_selected_brown_ink():
    image, evidence = _sample_fixture()
    brown = guidance_from_ink_color(image, evidence, InkColorProfile(BROWN))
    wrong = guidance_from_ink_color(image, evidence, InkColorProfile(BLUE))
    assert np.all(brown.avoidance_score[evidence.centerline] == 0.0)
    assert np.all(wrong.avoidance_score[evidence.centerline] == 1.0)
