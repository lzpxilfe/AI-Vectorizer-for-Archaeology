"""A valid raw gap may retain its Ink route without changing unrelated AI paths."""

from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("scipy")

from ai_vectorizer.core import label_occlusion, livewire
from ai_vectorizer.core.edge_detector import EdgeDetector
from ai_vectorizer.core.learned_prior import LearnedLinePrior
from ai_vectorizer.core.livewire import LiveWireCancelled, LiveWireConfig, LiveWireTree
from ai_vectorizer.core.occluded_connection import OccludedConnectionTree
from ai_vectorizer.core.trace_guidance import TraceGuidance


def _scene(*, inner=False):
    image = np.full((192, 192, 3), 245, dtype=np.uint8)
    image[60:101, 74] = image[60:101, 100] = 35
    for x in (78, 90):
        image[100:114, x:x + 7] = 35
        image[102:112, x + 2:x + 5] = 245
    image[55:130, 65] = image[55:130, 109] = 35
    image[129, 65:110] = 35
    if inner:
        image[70:97, 78] = image[70:97, 96] = 35
        image[96, 78:97] = 35
    evidence = EdgeDetector.detect_ink_evidence(image)
    edges = evidence.centerline.astype(np.uint8) * 255
    # Deliberately broad support can make a short blank-space pixel route
    # inexpensive; it must not change ownership of the observed gap arms.
    prior = LearnedLinePrior(np.ones(evidence.shape, np.float32), np.zeros(evidence.shape, np.float32))
    return image, edges, evidence, prior, (74, 84), (100, 84), (109, 84)


def _build(scene, **kwargs):
    image, edges, evidence, prior, root, _target, _neighbor = scene
    return livewire.build_livewire_tree(
        image, edges, root, evidence=evidence,
        learned_prior=kwargs.pop("learned_prior", prior), **kwargs,
    )


def _matching_reference(tree, target):
    connection = tree.occluded_connection
    inferred = connection.trace(target)
    assert inferred is not None
    reference = [tuple(point) for point in inferred]
    changed = 0
    ox, oy = connection.origin
    for index, (x, y) in enumerate(reference[1:-1], 1):
        cx, cy = int(round(x - ox)), int(round(y - oy))
        if (1 <= cx < connection.shape[1] - 1 and 1 <= cy < connection.shape[0] - 1
                and connection.mask[cy - 1:cy + 2, cx - 1:cx + 2].all()):
            reference[index] = (x, y + 0.4)
            changed += 1
    assert changed > 5
    assert connection.preserves_visible_route(reference, inferred)
    return inferred, reference


def test_worker_prepares_unlearned_reference_only_for_eligible_active_support():
    scene = _scene()
    tree = _build(scene)
    reference = tree.occlusion_reference_tree
    assert tree.occluded_connection is not None and reference is not None
    assert reference.strength == 1.0
    assert reference.occluded_connection is None
    assert reference.occlusion_reference_tree is None
    assert tree.champion_tree is None
    assert not tree.config.learned_route_guard and not tree.config.learned_observation_gate
    expected = _build(scene, learned_prior=None, config=replace(tree.config, automatic_label_occlusion=False))
    assert reference.trace(scene[5]) == expected.trace(scene[5])


def test_unrelated_learned_target_keeps_its_original_saved_vertices():
    scene = _scene()
    tree = _build(scene)
    neighbor = scene[6]
    assert tree.occlusion_reference_tree is not None
    assert tree.occluded_connection.trace(neighbor) is None
    assert tree.trace(neighbor) == replace(tree, occlusion_reference_tree=None).trace(neighbor)


def test_a_map_without_a_detected_label_keeps_the_learned_pixel_route():
    image = np.full((128, 128, 3), 245, dtype=np.uint8)
    image[30:100, 40] = image[30:100, 76] = 35
    evidence = EdgeDetector.detect_ink_evidence(image)
    prior = LearnedLinePrior(np.ones(evidence.shape), np.zeros(evidence.shape))
    tree = livewire.build_livewire_tree(image, evidence.centerline.astype(np.uint8) * 255,
                                       (40, 50), evidence=evidence, learned_prior=prior)
    assert tree.occluded_connection is None and tree.occlusion_reference_tree is None
    original = livewire.build_livewire_tree(
        image, evidence.centerline.astype(np.uint8) * 255, (40, 50),
        evidence=evidence, learned_prior=prior,
        config=LiveWireConfig(automatic_label_occlusion=False),
    )
    assert tree.trace((76, 90)) == original.trace((76, 90))


@pytest.mark.parametrize("mode", ["zero", "zero-head", "zero-cap", "disabled", "no-prior"])
def test_no_reference_for_inactive_support_or_disabled_feature(mode):
    scene = _scene()
    kwargs = {}
    if mode == "zero":
        kwargs["strength"] = 0.0
    elif mode == "zero-head":
        kwargs["learned_prior"] = LearnedLinePrior(np.zeros(scene[2].shape), np.zeros(scene[2].shape))
    elif mode == "zero-cap":
        kwargs["config"] = LiveWireConfig(learned_support_cap=0.0)
    elif mode == "disabled":
        kwargs["config"] = LiveWireConfig(automatic_label_occlusion=False)
    else:
        kwargs["learned_prior"] = None
    tree = _build(scene, **kwargs)
    assert tree.occlusion_reference_tree is None
    if mode == "zero":
        assert tree.trace(scene[5]) == [tuple(map(float, scene[4])), tuple(map(float, scene[5]))]


@pytest.mark.parametrize("mode", ["manual", "learned"])
def test_active_avoidance_never_acquires_an_ink_reference_override(mode):
    scene = _scene()
    avoidance = np.zeros(scene[2].shape, np.float32)
    avoidance[98:118, 74:101] = 1.0
    config = LiveWireConfig(learned_avoidance_cost_weight=6.0)
    kwargs = {"config": config}
    if mode == "manual":
        kwargs["guidance"] = TraceGuidance(avoidance)
    else:
        kwargs["learned_prior"] = LearnedLinePrior(scene[3].contour_score, avoidance)
    tree = _build(scene, **kwargs)
    assert tree.occluded_connection is None and tree.occlusion_reference_tree is None
    previous = _build(scene, **dict(kwargs, config=replace(config, automatic_label_occlusion=False)))
    assert tree.trace(scene[5]) == previous.trace(scene[5])


@pytest.mark.parametrize("strength", [1.0, 0.95])
def test_matching_reference_wins_only_for_real_gap_and_blends_once(strength):
    scene = _scene()
    tree = _build(scene, strength=strength)
    target, neighbor = scene[5:7]
    _inferred, matching = _matching_reference(tree, target)
    reference_tree = tree.occlusion_reference_tree
    direct = [tuple(map(float, scene[4])), tuple(map(float, target))]
    reference_calls = []

    def ordinary(selected, cursor):
        if selected is reference_tree:
            reference_calls.append(tuple(cursor))
            assert tuple(cursor) == target, "An unrelated target consulted the reference"
            return matching
        return [direct[0], tuple(map(float, cursor))]

    with mock.patch.object(LiveWireTree, "_trace_without_occlusion", autospec=True, side_effect=ordinary):
        route, provenance = tree.trace_with_provenance(target)
        assert provenance == "ink"
        assert route == livewire.blend_path_with_cursor(matching, tree.root, target, strength)
        assert reference_calls == [target]
        assert tree.occluded_connection.trace(neighbor) is None
        unrelated, _provenance = tree.trace_with_provenance(neighbor)
        assert unrelated == [direct[0], tuple(map(float, neighbor))]
        assert reference_calls == [target]


def test_partial_reference_is_rejected_if_its_display_crosses_neighbor():
    scene = _scene(inner=True)
    tree = _build(scene, strength=0.5)
    target = scene[5]
    inferred, matching = _matching_reference(tree, target)
    assert tree.occluded_connection.accepts_display(matching)
    displayed = livewire.blend_path_with_cursor(matching, tree.root, target, tree.strength)
    assert not tree.occluded_connection.accepts_display(displayed)
    reference = tree.occlusion_reference_tree
    direct = [tuple(map(float, scene[4])), tuple(map(float, target))]
    original_accepts = OccludedConnectionTree.accepts_display
    with mock.patch.object(
        LiveWireTree, "_trace_without_occlusion", autospec=True,
        side_effect=lambda selected, _cursor: matching if selected is reference else direct,
    ), mock.patch.object(type(tree.occluded_connection), "accepts_display",
                         autospec=True, wraps=None,
                         side_effect=lambda connection, path: original_accepts(connection, path)) as check:
        route, provenance = tree.trace_with_provenance(target)
    assert provenance == "ink" and route == direct
    np.testing.assert_allclose(check.call_args_list[0].args[1], displayed)
    assert not tree.occluded_connection.accepts_display(
        livewire.blend_path_with_cursor(inferred, tree.root, target, 0.5)
    )


@pytest.mark.parametrize("failure", [MemoryError, RuntimeError])
def test_optional_reference_failure_retains_established_candidate(failure):
    scene = _scene()
    original_build = livewire.build_livewire_tree
    attempts = []

    def fail_reference(*args, **kwargs):
        if kwargs.get("learned_prior") is None and not kwargs["config"].automatic_label_occlusion:
            attempts.append(True)
            raise failure("optional reference unavailable")
        return original_build(*args, **kwargs)

    with mock.patch.object(livewire, "build_livewire_tree", side_effect=fail_reference):
        tree = original_build(scene[0], scene[1], scene[4], evidence=scene[2], learned_prior=scene[3])
    assert attempts == [True]
    assert tree.occlusion_reference_tree is None and tree.occluded_connection is not None
    route, provenance = tree.trace_with_provenance(scene[5])
    assert route and provenance == "automatic-label-gap"


def test_cancellation_during_reference_preparation_never_returns_partial_tree():
    scene = _scene()
    original_build = livewire.build_livewire_tree
    cancelled = [False]

    def cancel_reference(*args, **kwargs):
        if kwargs.get("learned_prior") is None and not kwargs["config"].automatic_label_occlusion:
            cancelled[0] = True
        return original_build(*args, **kwargs)

    with mock.patch.object(livewire, "build_livewire_tree", side_effect=cancel_reference):
        with pytest.raises(LiveWireCancelled):
            original_build(scene[0], scene[1], scene[4], evidence=scene[2], learned_prior=scene[3],
                           cancel_check=lambda: cancelled[0])
    assert cancelled[0]


def test_hover_uses_prepared_reference_without_detection_or_graph_builds():
    scene = _scene()
    tree = _build(scene)
    assert tree.occlusion_reference_tree is not None
    expected = tree.trace(scene[5])
    with mock.patch.object(livewire, "build_livewire_tree", side_effect=AssertionError("hover build")), \
         mock.patch.object(label_occlusion, "detect_label_occlusions", side_effect=AssertionError("hover detection")), \
         mock.patch.object(livewire, "_line_features", side_effect=AssertionError("hover filtering")):
        for _ in range(3):
            assert tree.trace(scene[5]) == expected
