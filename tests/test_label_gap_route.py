"""One inferred connector, bounded hover and exact observed-route precedence."""

import numpy as np
import pytest

from ai_vectorizer.core import livewire
from ai_vectorizer.core.label_gap_route import build_label_gap_connection
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.observed_connection import ObservedConnectionCancelled

pytestmark = pytest.mark.skipif(not livewire.is_livewire_available(), reason="SciPy unavailable")


def _build(mask, root, **kwargs):
    evidence = LineEvidence(mask.astype(np.float32), mask)
    return livewire.build_livewire_tree(
        np.where(mask, 20, 245).astype(np.uint8), mask.astype(np.uint8) * 255,
        root, evidence=evidence, config=livewire.LiveWireConfig(
            max_window_size=max(mask.shape), target_snap_radius=0,
            heading_cost_weight=0, direction_cost_weight=0, **kwargs,
        ),
    )


def test_enabled_gap_keeps_a_straight_contour_and_never_uses_parallel_neighbor():
    mask = np.zeros((100, 160), dtype=bool)
    mask[40, 15:61] = mask[40, 90:146] = True
    mask[49, 15:146] = True
    tree = _build(mask, (20, 40))
    assert tree.observed_connection.trace((140, 40)) is None
    assert tree.label_gap_connection is not None
    route = tree.trace((140, 40))
    assert route[0] == (20., 40.) and route[-1] == (140., 40.)
    assert max(abs(y - 40) for x, y in route) < 1e-8
    assert all(a[0] <= b[0] for a, b in zip(route, route[1:]))


def test_two_separate_gaps_cannot_chain_into_one_accepted_inferred_route():
    mask = np.zeros((100, 190), dtype=bool)
    mask[40, 10:51] = mask[40, 75:116] = mask[40, 140:181] = True
    tree = _build(mask, (20, 40))
    assert tree.label_gap_connection is not None
    assert tree.label_gap_connection.trace((100, 40)) is not None
    assert tree.label_gap_connection.trace((170, 40)) is None


def test_existing_native_connection_stays_exact_when_gap_feature_is_enabled():
    mask = np.zeros((140, 160), dtype=bool)
    mask[20:111, 40] = mask[20:111, 110] = mask[110, 40:111] = True
    mask[60, 5:26] = mask[60, 130:151] = True
    tree = _build(mask, (40, 20))
    old = _build(mask, (40, 20), label_gap_connections=False)
    assert tree.trace((110, 20)) == old.trace((110, 20))
    assert max(y for x, y in tree.trace((110, 20))) == 110


def test_gap_tree_is_immutable_and_cancelled_worker_never_publishes(monkeypatch):
    mask = np.zeros((100, 160), dtype=bool)
    mask[40, 15:61] = mask[40, 90:146] = True
    tree = _build(mask, (20, 40))
    gap = tree.label_gap_connection
    assert gap is not None
    assert not gap.predecessors.flags.writeable and not gap.distances.flags.writeable
    with pytest.raises(TypeError):
        gap.prefixes[0] = ()
    with pytest.raises(ObservedConnectionCancelled):
        build_label_gap_connection(None, tree.observed_connection, (), sparse=None,
                                   dijkstra=None, cancel_check=lambda: True)
    import ai_vectorizer.core.label_gap as kernel
    def cancelled(*args, **kwargs):
        raise ObservedConnectionCancelled("during endpoint analysis")
    monkeypatch.setattr(kernel, "build_label_gap_connectors", cancelled)
    with pytest.raises(livewire.LiveWireCancelled):
        _build(mask, (20, 40))


def test_optional_gap_worker_failure_keeps_previous_pixel_route(monkeypatch):
    mask = np.zeros((100, 160), dtype=bool)
    mask[40, 15:61] = mask[40, 90:146] = True
    old = _build(mask, (20, 40), label_gap_connections=False)
    import ai_vectorizer.core.label_gap as kernel
    def failed(*args, **kwargs):
        raise MemoryError("optional bridge allocation failed")
    monkeypatch.setattr(kernel, "build_label_gap_connectors", failed)
    tree = _build(mask, (20, 40))
    assert tree.label_gap_connection is None
    assert tree.trace((140, 40)) == old.trace((140, 40))


def test_zero_strength_skips_all_gap_preparation_and_keeps_literal_cursor(monkeypatch):
    import ai_vectorizer.core.label_gap as kernel
    def unexpected(*args, **kwargs):
        raise AssertionError("0% prepared an inferred connector")
    monkeypatch.setattr(kernel, "build_label_gap_connectors", unexpected)
    mask = np.zeros((100, 160), dtype=bool)
    mask[40, 15:61] = mask[40, 90:146] = True
    tree = livewire.build_livewire_tree(
        np.where(mask, 20, 245).astype(np.uint8), mask.astype(np.uint8) * 255,
        (20, 40), evidence=LineEvidence(mask.astype(np.float32), mask), strength=0,
    )
    assert tree.label_gap_connection is None
    assert tree.trace((140.3, 41.2)) == [(20., 40.), (140.3, 41.2)]
