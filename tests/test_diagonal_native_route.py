"""Source-derived diagonal topology: keep the visible arch before its gap.

The small fixture stores observed skeleton bits and an independent detector
footprint from a public AMS map. It supplies no hidden curve ground truth.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from ai_vectorizer.core.label_occlusion import LabelOcclusion
from ai_vectorizer.core.occluded_connection import build_occluded_connection


FIXTURE = Path(__file__).parent / "fixtures" / "diagonal_600_native_topology.json"


def _unpack(rows, width):
    packed = np.asarray([list(bytes.fromhex(row)) for row in rows], dtype=np.uint8)
    return np.unpackbits(packed, axis=1, bitorder="little")[:, :width].astype(bool)


def _point(point, size, rotation, reflected):
    x, y = point
    if reflected:
        x = size - 1 - x
    for _ in range(rotation):
        x, y = y, size - 1 - x
    return x, y


def _eight_neighbor_graph(shape):
    """Only genuine neighboring pixels; no prewired contour or gap edges."""
    height, width = shape
    pixels = np.arange(height * width).reshape(shape)
    rows, columns, costs = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        first = pixels[y0:y1, x0:x1].ravel()
        last = pixels[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        rows.extend((first, last))
        columns.extend((last, first))
        costs.extend((np.full(len(first), np.hypot(dx, dy)),) * 2)
    return sparse.csr_matrix(
        (np.concatenate(costs), (np.concatenate(rows), np.concatenate(columns))),
        shape=(height * width, height * width),
    )


def _fixture(rotation=0, reflected=False):
    data = json.loads(FIXTURE.read_text())
    size = data["window_xywh"][2]
    native = _unpack(data["native_rows_hex"], size)
    region = data["region"]
    x0, y0, x1, y1 = region["bbox_xyxy"]
    mask = np.zeros(native.shape, dtype=bool)
    mask[y0:y1, x0:x1] = _unpack(region["mask_rows_hex"], x1 - x0)
    if reflected:
        native, mask = np.fliplr(native), np.fliplr(mask)
    native, mask = np.rot90(native, rotation), np.rot90(mask, rotation)
    transform = lambda p: _point(p, size, rotation, reflected)
    corners = np.asarray([transform(p) for p in
                          ((x0, y0), (x1 - 1, y0), (x1 - 1, y1 - 1), (x0, y1 - 1))])
    xmin, ymin = corners.min(axis=0)
    xmax, ymax = corners.max(axis=0) + 1
    polygon = tuple(transform(p) for p in region["polygon_xy"])
    if reflected:
        polygon = polygon[::-1]
    baseline = np.asarray(transform(region["baseline_xy"])) - transform((0, 0))
    proposal = LabelOcclusion(
        tuple(map(int, (xmin, ymin, xmax, ymax))), mask[ymin:ymax, xmin:xmax],
        tuple(baseline), region["counter_count"], region["score"], polygon,
    )
    points = {key: transform(value) for key, value in data["points_native_xy"].items()}
    visible = np.asarray([transform(p) for p in data["preserved_outer_native_xy"]])
    return native, proposal, points, visible


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("reflected", [False, True])
def test_diagonal_digit_gap_keeps_original_arch_and_rejects_adjacent_contour(rotation, reflected):
    native, region, points, visible = _fixture(rotation, reflected)
    graph = _eight_neighbor_graph(native.shape)
    before = native.copy()
    tree = build_occluded_connection(
        graph, native, points["root"], (0, 0), (region,),
        sparse=sparse, dijkstra=dijkstra,
    )
    assert tree is not None
    pair = frozenset((points["source_endpoint"], points["target_endpoint"]))
    assert {frozenset((first, last)) for first, last, _curve, _index in tree.connectors} == {pair}
    route = tree.trace(points["target"])
    assert route is not None
    assert route[0] == points["root"] and route[-1] == points["target"]
    assert tree.accepts_display(route)
    # The observed outer arm wraps around 6 before reaching its real endpoint.
    # A detector halo cannot cut that arm at the former fake endpoint.
    for x, y in visible:
        assert tree.outside_native[y, x]
    observed_distances = np.linalg.norm(visible[:, None] - np.asarray(route)[None, :], axis=2)
    assert observed_distances.min(axis=1).max() <= 1.0
    assert tree.trace(points["adjacent_inner_endpoint"]) is None
    assert tree.trace(points["adjacent_inner_target"]) is None
    wrong_tree = build_occluded_connection(
        graph, native, points["old_wrong_root"], (0, 0), (region,),
        sparse=sparse, dijkstra=dijkstra,
    )
    assert wrong_tree is None or wrong_tree.trace(points["adjacent_inner_target"]) is None
    np.testing.assert_array_equal(native, before)


def test_topology_fixture_keeps_source_rights_and_uncertainty_record():
    data = json.loads(FIXTURE.read_text())
    provenance = data["provenance"]
    assert provenance["publisher"] == "U.S. Army Map Service"
    assert provenance["rights_status"] == "public_domain"
    assert provenance["image_sha256"] == "ac04da0fce6279e2a7f52b7fd42802864cee129152a39a66832f7cedadab1996"
    assert "No hidden bridge geometry" in data["scope"]
