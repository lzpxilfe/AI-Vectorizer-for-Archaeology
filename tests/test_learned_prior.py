"""Independent optional prior and finite Live-Wire composition contracts."""

import numpy as np
import pytest

from ai_vectorizer.core.learned_prior import (
    LearnedLinePrior, compose_learned_guidance, crop_learned_prior,
)
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree, is_livewire_available
from ai_vectorizer.core.trace_guidance import TraceGuidance


def test_prior_copies_both_arrays_and_crop_is_immutable():
    source = np.full((5, 7), 0.4, dtype=np.float32)
    prior = LearnedLinePrior(source, source)
    source[:] = 1.0
    assert prior.shape == (5, 7)
    assert np.all(prior.contour_score == np.float32(0.4))
    assert not prior.contour_score.flags.writeable
    assert not prior.avoidance_score.flags.writeable
    crop = crop_learned_prior(prior, (1, 2, 6, 5))
    np.testing.assert_array_equal(crop.contour_score, prior.contour_score[2:5, 1:6])
    assert crop.shape == (3, 5)
    assert not crop.avoidance_score.flags.writeable
    with pytest.raises(ValueError):
        crop.contour_score[0, 0] = 1.0


@pytest.mark.parametrize("bad", [np.zeros((0, 3)), np.zeros(3), [[np.nan]], [[np.inf]], [[1.1]], [[-0.1]]])
def test_invalid_prior_scores_are_rejected(bad):
    with pytest.raises(ValueError):
        LearnedLinePrior(bad, bad)


def test_crop_and_shapes_are_checked():
    with pytest.raises(ValueError, match="match"):
        LearnedLinePrior(np.zeros((2, 2)), np.zeros((3, 2)))
    prior = LearnedLinePrior(np.zeros((4, 4)), np.zeros((4, 4)))
    for bounds in ((0, 0, 5, 4), (1, 1, 1, 2), (0.0, 0, 2, 2), "0,0,2,2"):
        with pytest.raises(ValueError):
            crop_learned_prior(prior, bounds)


@pytest.mark.parametrize("learned_weight", [2.0, 4.0, 6.0])
def test_weighted_composition_preserves_manual_cost(learned_weight):
    manual = TraceGuidance(np.array([[1.0, 0.0]], dtype=np.float32))
    prior = LearnedLinePrior(np.zeros((1, 2)), np.ones((1, 2)))
    guidance, weight = compose_learned_guidance(manual, prior, learned_weight=learned_weight)
    assert weight == 6.0
    np.testing.assert_allclose(guidance.avoidance_score * weight, [[6.0, learned_weight]])
    absent, old_weight = compose_learned_guidance(manual, None)
    assert absent is manual and old_weight == 6.0
    zero = LearnedLinePrior(np.ones((1, 2)), np.zeros((1, 2)))
    assert compose_learned_guidance(manual, zero)[0] is manual
    with pytest.raises(ValueError):
        compose_learned_guidance(manual, prior, learned_weight=np.inf)


def _straight_evidence():
    shape = (40, 40)
    active = np.ones(shape, dtype=bool)
    evidence = LineEvidence(
        active.astype(np.float32), active, tangent_x=np.ones(shape),
        tangent_y=np.zeros(shape), coherence=np.ones(shape),
    )
    return np.full((*shape, 3), 245, dtype=np.uint8), np.zeros(shape, dtype=np.uint8), evidence


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
def test_optional_none_and_zero_prior_preserve_exact_baseline():
    image, edges, evidence = _straight_evidence()
    baseline = build_livewire_tree(image, edges, (10, 20), evidence=evidence)
    explicit = build_livewire_tree(image, edges, (10, 20), evidence=evidence, learned_prior=None)
    zero = LearnedLinePrior(np.zeros(evidence.shape), np.zeros(evidence.shape))
    empty = build_livewire_tree(image, edges, (10, 20), evidence=evidence, learned_prior=zero)
    for tree in (explicit, empty):
        np.testing.assert_array_equal(tree.distances, baseline.distances)
        np.testing.assert_array_equal(tree.predecessors, baseline.predecessors)
    literal = build_livewire_tree(image, edges, (10, 20), evidence=evidence, learned_prior=zero, strength=0.0)
    assert literal.trace((25.2, 30.3)) == [(10.0, 20.0), (25.2, 30.3)]


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
@pytest.mark.parametrize("learned_weight", [2.0, 4.0, 6.0])
def test_graph_cost_max_composes_weights_and_keeps_routes_reachable(learned_weight):
    image, edges, evidence = _straight_evidence()
    prior = LearnedLinePrior(np.zeros(evidence.shape), np.ones(evidence.shape))
    manual = TraceGuidance(np.ones(evidence.shape))
    config = LiveWireConfig(target_snap_radius=0, learned_avoidance_cost_weight=learned_weight)
    combined = build_livewire_tree(image, edges, (10, 20), evidence=evidence, guidance=manual, learned_prior=prior, config=config)
    assert combined.distances[20 * 40 + 11] == pytest.approx(7.0)
    learned = build_livewire_tree(image, edges, (10, 20), evidence=evidence, learned_prior=prior, config=config)
    assert learned.distances[20 * 40 + 11] == pytest.approx(1.0 + learned_weight)
    assert np.isfinite(combined.distances).all()
    assert combined.trace((11, 20))[-1] == (11.0, 20.0)


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
def test_prior_boundary_requires_matching_observed_ink():
    image, edges, evidence = _straight_evidence()
    prior = LearnedLinePrior(np.zeros(evidence.shape), np.zeros(evidence.shape))
    with pytest.raises(ValueError, match="observed"):
        build_livewire_tree(image, edges, (10, 20), learned_prior=prior)
    with pytest.raises(ValueError, match="dimensions"):
        build_livewire_tree(image, edges, (10, 20), evidence=evidence, learned_prior=LearnedLinePrior(np.zeros((2, 2)), np.zeros((2, 2))))
    with pytest.raises(TypeError, match="LearnedLinePrior"):
        build_livewire_tree(image, edges, (10, 20), evidence=evidence, learned_prior=object())
    for config in (LiveWireConfig(learned_support_cap=0.9), LiveWireConfig(learned_support_cap=np.nan), LiveWireConfig(learned_avoidance_cost_weight=-1)):
        with pytest.raises(ValueError):
            config.validate()
