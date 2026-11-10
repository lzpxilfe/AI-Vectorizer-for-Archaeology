"""Opt-in decision records must explain routing without changing its output."""
import json
from dataclasses import replace

import numpy as np
import pytest

from ai_vectorizer.core import livewire
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.learned_prior import LearnedLinePrior
from ai_vectorizer.core.trace_guidance import TraceGuidance

pytestmark = pytest.mark.skipif(not livewire.is_livewire_available(), reason="SciPy unavailable")


def _fixture():
    native = np.zeros((48, 48), dtype=bool)
    native[12, 8:39] = native[34, 8:39] = True
    native[12:35, 8] = native[12:35, 38] = True
    image = np.full(native.shape, 255, dtype=np.uint8)
    image[native] = 0
    evidence = LineEvidence(center_score=native.astype(np.float32), centerline=native)
    return image, native.astype(np.uint8) * 255, evidence


def test_build_and_hover_records_preserve_geometry_and_are_independent_snapshots():
    image, edges, evidence = _fixture()
    prior = LearnedLinePrior(np.full(edges.shape, .75), np.full(edges.shape, .3))
    kwargs = dict(evidence=evidence, learned_prior=prior,
                  config=livewire.LiveWireConfig(max_window_size=48, automatic_label_occlusion=False))
    baseline = livewire.build_livewire_tree(image, edges, (8, 20), **kwargs)
    sink = {}
    recorded = livewire.build_livewire_tree(image, edges, (8, 20), diagnostics=sink, **kwargs)
    np.testing.assert_array_equal(recorded.predecessors, baseline.predecessors)
    np.testing.assert_array_equal(recorded.distances, baseline.distances)
    path, provenance, decision = recorded.trace_with_diagnostics((38, 25))
    assert (path, provenance) == baseline.trace_with_provenance((38, 25))
    assert decision["ordinary"]["branch"] == "observed_native"
    assert decision["selection"]["branch"] == "observed_native"
    assert sink["mixing"]["learned_support_dominant_pixels"] > 0
    assert sink["mixing"]["native_skeleton_pixels_changed"] == 0
    assert sink["mixing"]["native_skeleton_pixels"] == int(evidence.centerline.sum())
    assert not sink["mixing"]["active_avoidance"]
    json.dumps(decision, allow_nan=False)
    saved = recorded.diagnostics_json
    sink.clear()
    decision["build"].clear()
    again = recorded.trace_with_diagnostics((38, 25))[2]
    assert recorded.diagnostics_json == saved
    assert again["build"]["mixing"]["learned_present"] is True
    assert baseline.diagnostics_json is None


@pytest.mark.parametrize("mode,reason", [("strength", "strength_zero"),
                                          ("evidence", "no_evidence"),
                                          ("avoidance", "active_avoidance")])
def test_build_skip_reasons_explain_real_gating_without_changing_paths(mode, reason):
    image, edges, evidence = _fixture()
    kwargs = {"evidence": evidence, "config": livewire.LiveWireConfig(max_window_size=48)}
    if mode == "strength":
        kwargs["strength"] = 0.0
    elif mode == "evidence":
        kwargs.pop("evidence")
    else:
        kwargs["guidance"] = TraceGuidance(np.full(edges.shape, .5))
    plain = livewire.build_livewire_tree(image, edges, (8, 20), **kwargs)
    sink = {}
    diagnosed = livewire.build_livewire_tree(image, edges, (8, 20), diagnostics=sink, **kwargs)
    assert diagnosed.trace((38, 25)) == plain.trace((38, 25))
    assert sink["observed"]["reason"] == reason
    assert sink["numeric_gap"]["reason"] == reason
    assert sink["text_gap"]["reason"] == reason


def test_optional_runtime_error_reason_does_not_replace_the_pixel_fallback(monkeypatch):
    image, edges, evidence = _fixture()
    def unavailable(*args, **kwargs):
        raise MemoryError("private details must not enter diagnostics")
    monkeypatch.setattr(livewire, "build_observed_connection", unavailable)
    config = livewire.LiveWireConfig(max_window_size=48, automatic_label_occlusion=False)
    sink = {}
    tree = livewire.build_livewire_tree(image, edges, (8, 20), evidence=evidence,
                                     config=config, diagnostics=sink)
    assert sink["observed"] == {"status": "error", "reason": "optional_runtime_error", "error_type": "MemoryError"}
    path, provenance, trace = tree.trace_with_diagnostics((38, 25))
    assert (path, provenance) == tree.trace_with_provenance((38, 25))
    assert trace["selection"]["branch"] == "pixel"
    assert "private details" not in json.dumps(trace)


def test_diagnostics_capture_cancellation_and_direct_target_reasons():
    image, edges, evidence = _fixture()
    sink = {}
    with pytest.raises(livewire.LiveWireCancelled):
        livewire.build_livewire_tree(image, edges, (8, 20), diagnostics=sink,
                                    cancel_check=lambda: True)
    assert sink["status"] == "cancelled"
    tree = livewire.build_livewire_tree(image, edges, (8, 20), evidence=evidence, strength=0)
    path, _provenance, decision = tree.trace_with_diagnostics((38, 25))
    assert path == [(8.0, 20.0), (38.0, 25.0)]
    assert decision["ordinary"]["reason"] == "strength_zero"
    outside = replace(tree, strength=1).trace_with_diagnostics((-1, 25))[2]
    assert outside["ordinary"]["reason"] == "target_outside_window"


@pytest.mark.parametrize("strength", [0, 1])
def test_learned_prior_without_native_evidence_keeps_existing_validation_on_and_off(strength):
    image, edges, _evidence = _fixture()
    prior = LearnedLinePrior(np.full(edges.shape, .75), np.full(edges.shape, .3))
    for diagnostics in (None, {}):
        with pytest.raises(ValueError, match="learned_prior requires observed Ink LineEvidence"):
            livewire.build_livewire_tree(image, edges, (8, 20), learned_prior=prior,
                                        evidence=None, strength=strength, diagnostics=diagnostics)
