"""Connected bends, endpoint authority and safe pixel-graph fallbacks."""

from dataclasses import replace
import math

import numpy as np
import pytest

import ai_vectorizer.core.livewire as livewire
from ai_vectorizer.core.learned_prior import LearnedLinePrior
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.map_grid import MapGridPrior
from ai_vectorizer.core.observed_connection import (
    ObservedConnectionCancelled,
    ObservedConnectionTree,
    build_observed_connection,
)
from ai_vectorizer.core.trace_guidance import TraceGuidance


pytestmark = pytest.mark.skipif(
    not livewire.is_livewire_available(), reason="SciPy is not installed"
)


def _stroke(mask, first, last):
    first = np.asarray(first, dtype=float)
    last = np.asarray(last, dtype=float)
    count = max(1, int(np.linalg.norm(last - first) * 4))
    points = np.rint(
        first + (last - first) * np.linspace(0, 1, count + 1)[:, None]
    ).astype(int)
    mask[points[:, 1], points[:, 0]] = True


def _deep_bend():
    mask = np.zeros((192, 192), dtype=bool)
    vertices = [(50, 30), (50, 110), (60, 132), (90, 152),
                (120, 132), (130, 110), (130, 30)]
    for first, last in zip(vertices[:-1], vertices[1:]):
        _stroke(mask, first, last)
    # Separate nearby ink tempts the pixel route to leave and rejoin the U.
    mask[33, 53:128] = True
    return mask, vertices[0], vertices[-1]


def _tree(mask, root, *, config=None, **kwargs):
    image = np.where(mask, 20, 245).astype(np.uint8)
    evidence = LineEvidence(mask.astype(np.float32), mask)
    config = config or livewire.LiveWireConfig(
        max_window_size=max(mask.shape), target_snap_radius=0,
        direction_cost_weight=0, heading_cost_weight=0,
    )
    return livewire.build_livewire_tree(
        image, np.uint8(mask) * 255, root, evidence=evidence,
        config=config, **kwargs,
    )


def test_connected_deep_bend_wins_over_a_short_adjacent_stroke():
    mask, root, target = _deep_bend()
    tree = _tree(mask, root)
    previous = _tree(mask, root, config=replace(tree.config, observed_connections=False))
    path = tree.trace(target)
    assert tree.observed_connection is not None
    assert max(y for _x, y in previous.trace(target)) < 40
    assert max(y for _x, y in path) == 152
    assert path[0] == tuple(map(float, root))
    assert path[-1] == tuple(map(float, target))
    assert all(mask[int(y), int(x)] for x, y in path)
    assert not tree.observed_connection.preserves_collinear_gap(previous.trace(target))
    # A connected route has a separate bounded detour allowance; the old
    # pixel-graph guard is retained verbatim for its fallback.
    assert not livewire._detour_is_reasonable(path, root, target, tree.config)


def test_precise_click_cannot_move_to_a_cheaper_neighbour_two_pixels_away():
    mask, root, target = _deep_bend()
    # Both ends of a separate shortcut are exactly two pixels from a click.
    mask[30, 52:129] = True
    tree = _tree(mask, root)
    path = tree.trace(target)
    assert max(y for _x, y in path) == 152
    assert path[1][1] > root[1]


def test_missing_connection_and_missing_endpoint_keep_identical_pixel_fallback():
    mask = np.zeros((80, 100), dtype=bool)
    mask[40, 10:40] = True
    mask[40, 43:90] = True
    # Isolate native-only fallback; inferred gaps have their own geometry tests.
    tree = _tree(mask, (10, 40), config=livewire.LiveWireConfig(
        max_window_size=100, target_snap_radius=0,
        direction_cost_weight=0, heading_cost_weight=0, label_gap_connections=False,
    ))
    previous = _tree(mask, (10, 40), config=replace(tree.config, observed_connections=False))
    assert tree.observed_connection is not None
    for target in ((89, 40), (89, 50)):
        assert tree.observed_connection.trace(target) is None
        assert tree.trace(target) == previous.trace(target)
    root_in_paper = _tree(mask, (10, 50))
    assert root_in_paper.observed_connection is None


@pytest.mark.parametrize("limit", ("absolute", "relative"))
def test_long_network_detours_remain_bounded_and_keep_pixel_fallback(limit):
    mask = np.zeros((100, 100), dtype=bool)
    if limit == "absolute":
        vertices = [(5, 5), (95, 5), (95, 15), (5, 15), (5, 25),
                    (95, 25), (95, 35), (5, 35), (5, 45), (95, 45)]
    else:
        vertices = [(40, 10), (40, 70), (48, 70), (48, 10)]
    for first, last in zip(vertices[:-1], vertices[1:]):
        _stroke(mask, first, last)
    root, target = vertices[0], vertices[-1]
    tree = _tree(mask, root)
    previous = _tree(mask, root, config=replace(tree.config, observed_connections=False))
    assert tree.observed_connection is not None
    assert tree.observed_connection.trace(target) is None
    assert tree.trace(target) == previous.trace(target)


def test_overlong_serpentine_stops_predecessor_access_at_the_native_length_limit():
    size = 64
    native = np.concatenate([
        np.arange(row * size, (row + 1) * size)[::1 if row % 2 == 0 else -1]
        for row in range(size)
    ])
    count = len(native)
    lookup = np.empty(count, dtype=np.int32)
    lookup[native] = np.arange(count)
    class CountingPredecessors:
        accesses = 0
        def __getitem__(self, node):
            self.accesses += 1
            return count if node == 0 else node - 1
    predecessors = CountingPredecessors()
    connection = ObservedConnectionTree(
        (0.0, 0.0), (0, 0), (size, size), native, predecessors,
        np.arange(count + 1, dtype=float), lookup.reshape((size, size)),
    )
    assert connection.trace((0, size - 1)) is None
    # Native steps are one pixel long. The absolute allowance is 2*window;
    # the 4095-step route must stop without traversing or materializing it.
    assert 1 < predecessors.accesses <= size * 2 + 1


def test_legacy_gap_inspection_bounds_iterations_without_rejecting_high_native_indices():
    reads = []
    class CountingArray(np.ndarray):
        def __getitem__(self, node):
            reads.append(node)
            return super().__getitem__(node)
    predecessors = np.arange(-1, 4095, dtype=np.int32).view(CountingArray)
    assert livewire._trace_predecessors(predecessors, 0, 4095, max_steps=129) == []
    assert len(reads) == 129


@pytest.mark.parametrize("strength", (0.0, 0.25, 0.5, 1.0))
def test_zero_strength_root_hover_and_fractional_blend_are_exact(strength):
    mask, root, target = _deep_bend()
    tree = _tree(mask, root, strength=strength)
    assert tree.trace(root) == [tuple(map(float, root)), tuple(map(float, root))]
    if strength == 0:
        assert tree.observed_connection is None
        assert tree.trace(target) == [tuple(map(float, root)), tuple(map(float, target))]
    else:
        full = tree.observed_connection.trace(target)
        assert tree.trace(target) == livewire.blend_path_with_cursor(full, root, target, strength)


@pytest.mark.parametrize("strength", (0.0, 0.5, 1.0))
def test_narrow_observed_u_survives_legacy_snap_back_to_root(strength):
    shape = (64, 64)
    root, target = (30, 20), (35, 20)
    mask = np.zeros(shape, dtype=bool)
    mask[20:31, 30] = True
    mask[20:31, 35] = True
    mask[30, 30:36] = True
    tangent_x = np.zeros(shape, dtype=np.float32)
    tangent_y = mask.astype(np.float32)
    tangent_x[30, 30:36] = 1
    tangent_y[30, 30:36] = 0
    evidence = LineEvidence(
        mask.astype(np.float32), mask, tangent_x=tangent_x,
        tangent_y=tangent_y, coherence=mask.astype(np.float32),
    )
    tree = livewire.build_livewire_tree(
        np.where(mask, 20, 245).astype(np.uint8), np.uint8(mask) * 255,
        root, evidence=evidence, strength=strength,
        config=livewire.LiveWireConfig(max_window_size=64),
    )
    assert tree.trace(root) == [tuple(map(float, root)), tuple(map(float, root))]
    if strength == 0:
        assert tree.observed_connection is None
        assert tree.trace(target) == [tuple(map(float, root)), tuple(map(float, target))]
        return
    observed = tree.observed_connection.trace(target)
    assert max(y for _x, y in observed) == 30
    if strength == 1:
        # With the default six-pixel snap, the old target query chooses the
        # root's zero-cost route over the narrow bend's more distant end.
        assert tree._select_target_index(*target) == root[1] * 64 + root[0]
    path = tree.trace(target)
    assert path == livewire.blend_path_with_cursor(observed, root, target, strength)
    assert path[-1] == tuple(map(float, target))


def test_grid_cost_preserves_a_transverse_crossing_and_rejects_grid_shortcut():
    mask = np.zeros((121, 121), dtype=bool)
    for first, last in [((30, 20), (30, 100)), ((30, 100), (90, 100)),
                        ((90, 100), (90, 20))]:
        _stroke(mask, first, last)
    mask[60, :] = True
    horizontal = np.zeros(mask.shape, dtype=np.float32)
    horizontal[60, :] = 1
    prior = MapGridPrior(horizontal, np.zeros_like(horizontal))
    no_grid = _tree(mask, (30, 20)).trace((90, 20))
    tree = _tree(mask, (30, 20), grid_prior=prior)
    path = tree.trace((90, 20))
    assert max(y for _x, y in no_grid) == 60
    assert max(y for _x, y in path) == 100
    assert sum(y == 60 for _x, y in path) >= 2
    assert np.isfinite(tree.distances).all()


@pytest.mark.parametrize("kind", ("manual", "color", "learned"))
def test_active_avoidance_preserves_existing_finite_cost_behavior(kind):
    mask, root, target = _deep_bend()
    cost = np.zeros(mask.shape, dtype=np.float32)
    cost[90:140, 40:140] = 1
    config = livewire.LiveWireConfig(max_window_size=192, target_snap_radius=0)
    kwargs = {}
    if kind in ("manual", "color"):
        # Both UI sources arrive as the same immutable max-composed guide.
        kwargs["guidance"] = TraceGuidance(cost)
    else:
        kwargs["learned_prior"] = LearnedLinePrior(np.zeros_like(cost), cost)
        config = replace(config, learned_avoidance_cost_weight=4.0)
    guided = _tree(mask, root, config=config, **kwargs)
    previous = _tree(mask, root, config=replace(config, observed_connections=False), **kwargs)
    assert guided.observed_connection is None
    assert guided.trace(target) == previous.trace(target)
    np.testing.assert_array_equal(guided.distances, previous.distances)
    assert np.isfinite(guided.distances).all()


def test_inactive_avoidance_does_not_disable_observed_connection():
    mask, root, target = _deep_bend()
    zero = np.zeros(mask.shape, dtype=np.float32)
    prior = LearnedLinePrior(np.ones_like(zero), np.ones_like(zero))
    tree = _tree(mask, root, guidance=TraceGuidance(zero), learned_prior=prior)
    assert tree.observed_connection is not None
    assert max(y for _x, y in tree.trace(target)) == 152


def test_colored_scan_gap_bridge_keeps_its_existing_pixel_route():
    shape = (72, 168)
    image = np.full((*shape, 3), (232, 220, 196), dtype=np.uint8)
    mask = np.zeros(shape, dtype=bool)
    for x in range(8, 160):
        image[43, x] = (55, 45, 35)
        mask[43, x] = True
        if not 61 <= x <= 108:
            image[31, x] = (175, 225, 225)
            mask[31, x] = True
    evidence = LineEvidence(mask.astype(np.float32), mask,
                            tangent_x=mask.astype(np.float32),
                            tangent_y=np.zeros(shape), coherence=mask.astype(np.float32))
    config = livewire.LiveWireConfig(max_window_size=168, target_snap_radius=0)
    kwargs = dict(evidence=evidence, incoming_direction=(1.0, 0.0))
    tree = livewire.build_livewire_tree(image, np.uint8(mask) * 255, (8, 31), config=config, **kwargs)
    previous = livewire.build_livewire_tree(image, np.uint8(mask) * 255, (8, 31),
        config=replace(config, observed_connections=False), **kwargs)
    assert tree.observed_connection is None
    assert tree.trace((159, 31)) == previous.trace((159, 31))
    assert max(y for _x, y in tree.trace((159, 31))) < 39


@pytest.mark.parametrize("missing", (3, 8, 20))
@pytest.mark.parametrize("transpose", (False, True))
def test_existing_collinear_black_gap_is_not_replaced_by_connected_lower_network(missing, transpose):
    mask = np.zeros((100, 120), dtype=bool)
    mask[25, 10:101] = True
    mask[25, 50:50 + missing] = False
    mask[25:75, 25] = True
    mask[25:75, 85] = True
    mask[74, 25:86] = True
    root, target = (10, 25), (100, 25)
    if transpose:
        mask = mask.T.copy()
        root, target = root[::-1], target[::-1]
    tree = _tree(mask, root)
    previous = _tree(mask, root, config=replace(tree.config, observed_connections=False))
    native = tree.observed_connection.trace(target)
    axis = 0 if transpose else 1
    # The continuous alternative exists, but describes the neighbouring
    # lower network. Protect the old pixel route through the genuine gap.
    assert max(point[axis] for point in native) == 74
    assert max(point[axis] for point in previous.trace(target)) == 25
    assert tree.observed_connection.preserves_collinear_gap(previous.trace(target))
    if missing == 20:
        assert not tree.observed_connection.preserves_collinear_gap(
            previous.trace(target), max_gap_pixels=8.0
        )
    assert tree.trace(target) == previous.trace(target)


def test_gap_protection_cannot_preserve_a_pixel_route_that_legacy_would_reject():
    mask = np.zeros((192, 192), dtype=bool)
    observed_bend = [(20, 20), (20, 80), (100, 80), (100, 20)]
    longer_network = [(20, 20), (10, 20), (10, 100), (130, 100),
                      (130, 20), (100, 20)]
    for vertices in (observed_bend, longer_network):
        for first, last in zip(vertices[:-1], vertices[1:]):
            _stroke(mask, first, last)
    mask[100, 80:86] = False
    horizontal = np.zeros(mask.shape)
    vertical = np.zeros(mask.shape)
    horizontal[80, 20:101] = 1
    vertical[20:81, 20] = 1
    vertical[20:81, 100] = 1
    config = livewire.LiveWireConfig(
        max_window_size=192, target_snap_radius=0, line_cost_weight=10,
        direction_cost_weight=0, heading_cost_weight=0,
    )
    root, target = observed_bend[0], observed_bend[-1]
    tree = _tree(mask, root, config=config, grid_prior=MapGridPrior(horizontal, vertical))
    end = tree._select_target_index(*target)
    indices = livewire._trace_predecessors(tree.predecessors, root[1] * 192 + root[0], end)
    raw = [(node % 192, node // 192) for node in indices]
    assert tree.observed_connection.preserves_collinear_gap(raw)
    assert not livewire._detour_is_reasonable(raw, root, target, config)
    # The genuine gap on a rejected 318px detour cannot suppress the
    # admissible 198px observed bend or turn its preview into a literal chord.
    assert max(y for _x, y in tree.trace(target)) == 80


@pytest.mark.parametrize("kind", ("branch", "circle"))
def test_observed_junctions_and_loops_keep_clicks_and_never_cut_through_paper(kind):
    mask = np.zeros((101, 101), dtype=bool)
    if kind == "branch":
        for first, last in [((10, 20), (50, 50)), ((90, 20), (50, 50)),
                            ((50, 50), (50, 90))]:
            _stroke(mask, first, last)
        root, target = (10, 20), (50, 90)
    else:
        angle = np.linspace(0, 2 * math.pi, 1000)
        xy = np.rint(np.c_[50 + 30 * np.cos(angle), 50 + 30 * np.sin(angle)]).astype(int)
        mask[xy[:, 1], xy[:, 0]] = True
        root, target = (20, 50), (80, 50)
    path = _tree(mask, root).trace(target)
    assert path[-1] == tuple(map(float, target))
    assert all(mask[int(y), int(x)] for x, y in path)
    if kind == "circle":
        assert min(math.hypot(x - 50, y - 50) for x, y in path) > 28


def test_compact_data_are_immutable_and_hover_needs_no_worker_work(monkeypatch):
    mask, root, target = _deep_bend()
    tree = _tree(mask, root)
    connection = tree.observed_connection
    for array in (connection.native_indices, connection.predecessors,
                  connection.distances, connection.node_lookup, connection.gap_outward):
        assert not array.flags.writeable
    def unexpected(*_args, **_kwargs):
        raise AssertionError("worker construction ran during hover")
    monkeypatch.setattr(livewire, "build_observed_connection", unexpected)
    assert max(y for _x, y in tree.trace(target)) == 152


def test_cancellation_from_compact_worker_propagates(monkeypatch):
    mask, root, _target = _deep_bend()
    def cancelled(*_args, **_kwargs):
        raise ObservedConnectionCancelled("cancelled while building compact graph")
    monkeypatch.setattr(livewire, "build_observed_connection", cancelled)
    with pytest.raises(livewire.LiveWireCancelled):
        _tree(mask, root)


@pytest.mark.parametrize("failure", (RuntimeError, MemoryError))
def test_optional_worker_failure_preserves_completed_pixel_tree(monkeypatch, failure):
    mask, root, target = _deep_bend()
    previous = _tree(mask, root, config=livewire.LiveWireConfig(
        max_window_size=192, target_snap_radius=0, observed_connections=False))
    def failed(*_args, **_kwargs):
        raise failure("optional compact tree failed")
    monkeypatch.setattr(livewire, "build_observed_connection", failed)
    current = _tree(mask, root, config=replace(previous.config, observed_connections=True))
    assert current.observed_connection is None
    np.testing.assert_array_equal(current.distances, previous.distances)
    np.testing.assert_array_equal(current.predecessors, previous.predecessors)
    assert current.trace(target) == previous.trace(target)


def test_compact_builder_checks_cancellation_after_sampling():
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    mask = np.ones((4, 4), dtype=bool)
    calls = []
    def cancelled():
        calls.append(True)
        return len(calls) == 2
    with pytest.raises(ObservedConnectionCancelled):
        build_observed_connection(sparse.eye(16, format="csr"), mask, (0, 0), (0, 0),
                                  sparse=sparse, dijkstra=dijkstra, cancel_check=cancelled)


def test_invalid_boolean_and_missing_runtime_keep_existing_contract(monkeypatch):
    with pytest.raises(ValueError, match="observed_connections"):
        livewire.LiveWireConfig(observed_connections=1).validate()
    monkeypatch.setattr(livewire, "_get_livewire_runtime", lambda: (None, "missing SciPy"))
    mask, root, _target = _deep_bend()
    with pytest.raises(livewire.LiveWireUnavailable):
        _tree(mask, root)
