"""Optional detected-label routing preserves the existing fallback contract."""

from dataclasses import replace

import numpy as np
import pytest

from ai_vectorizer.core import livewire
from ai_vectorizer.core import label_occlusion
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.trace_guidance import TraceGuidance


pytestmark = pytest.mark.skipif(not livewire.is_livewire_available(), reason="SciPy is not installed")


def _scene():
    line = np.zeros((80, 96), dtype=bool)
    line[20:61, 24] = line[20:61, 64] = True
    line[60, 24:65] = True
    return np.where(line, 20, 245).astype(np.uint8), line


def _tree(image, line, **kwargs):
    return livewire.build_livewire_tree(
        image, np.uint8(line) * 255, (24, 20),
        evidence=LineEvidence(line.astype(np.float32), line),
        config=kwargs.pop("config", livewire.LiveWireConfig(max_window_size=96)),
        **kwargs,
    )


@pytest.mark.parametrize("mode", ("off", "zero", "avoidance", "no-native", "float", "rgba"))
def test_ineligible_context_never_prepares_or_uses_occlusion(monkeypatch, mode):
    image, line = _scene()
    kwargs = {}
    if mode in ("off", "no-native"):
        kwargs["config"] = livewire.LiveWireConfig(
            max_window_size=96, automatic_label_occlusion=mode != "off",
            observed_connections=mode != "no-native",
        )
    elif mode == "zero":
        kwargs["strength"] = 0.0
    elif mode == "avoidance":
        kwargs["guidance"] = TraceGuidance(np.ones(line.shape, dtype=np.float32))
    elif mode == "float":
        image = image.astype(np.float32)
    elif mode == "rgba":
        image = np.repeat(image[..., None], 4, axis=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("ineligible context prepared label occlusions")

    monkeypatch.setattr(label_occlusion, "detect_label_occlusions", forbidden)
    tree = _tree(image, line, **kwargs)
    assert tree.occluded_connection is None
    assert tree.trace_with_provenance((64, 20))[1] == "ink"


@pytest.mark.parametrize("failure", (RuntimeError, MemoryError))
def test_optional_worker_failure_retains_identical_existing_route(monkeypatch, failure):
    image, line = _scene()
    baseline = _tree(image, line, config=livewire.LiveWireConfig(
        max_window_size=96, automatic_label_occlusion=False,
    ))

    def unavailable(*args, **kwargs):
        raise failure("optional preparation failed")

    monkeypatch.setattr(label_occlusion, "detect_label_occlusions", unavailable)
    tree = _tree(image, line)
    assert tree.occluded_connection is None
    assert tree.trace((64, 20)) == baseline.trace((64, 20))


def test_cancellation_propagates_instead_of_publishing_a_partial_tree(monkeypatch):
    image, line = _scene()

    def cancelled(*args, **kwargs):
        raise livewire.ObservedConnectionCancelled("cancelled during detection")

    monkeypatch.setattr(label_occlusion, "detect_label_occlusions", cancelled)
    with pytest.raises(livewire.LiveWireCancelled, match="cancelled during detection"):
        _tree(image, line)


@pytest.mark.parametrize("broken,available,clear", ((False, True, True), (True, False, True), (True, True, False), (True, True, True)))
def test_only_a_broken_outside_connection_and_clear_display_can_replace_existing_route(broken, available, clear):
    image, line = _scene()
    base = _tree(image, line, strength=0.75, config=livewire.LiveWireConfig(
        max_window_size=96, automatic_label_occlusion=False,
    ))
    target = (64.25, 20.0)
    candidate = [(24.0, 20.0), (44.0, 65.0), target]
    checked = []

    class Proposal:
        def preserves_visible_route(self, ordinary, inferred):
            return False

        def outside_route(self, point):
            return None if broken else base.trace(target)

        def trace(self, point):
            return candidate if available else None

        def accepts_display(self, route):
            checked.append(route)
            return clear

    tree = replace(base, occluded_connection=Proposal())
    route, provenance = tree.trace_with_provenance(target)
    if broken and available:
        assert checked == [livewire.blend_path_with_cursor(candidate, tree.root, target, 0.75)]
    else:
        assert not checked
    if broken and available and clear:
        assert route == checked[0]
        assert provenance == "automatic-label-gap"
    else:
        assert route == base.trace(target)
        assert provenance == "ink"


def test_visible_route_preservation_uses_full_evidence_and_keeps_the_original_partial_blend():
    image, line = _scene()
    base = _tree(image, line, strength=0.75, config=livewire.LiveWireConfig(
        max_window_size=96, automatic_label_occlusion=False,
    ))
    target = (64.0, 20.0)
    candidate = [(24.0, 20.0), (44.0, 65.0), target]
    full_old = replace(base, strength=1.0)._trace_without_occlusion(target)

    class Proposal:
        def outside_route(self, point):
            return None

        def trace(self, point):
            return candidate

        def preserves_visible_route(self, ordinary, inferred):
            assert ordinary == full_old
            assert inferred == candidate
            return True

        def accepts_display(self, points):
            raise AssertionError("preserved geometry must not be blended again")

    tree = replace(base, occluded_connection=Proposal())
    assert tree.trace_with_provenance(target) == (base.trace(target), "ink")
