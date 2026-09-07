"""Explicit neutral-contour label-gap experiment, outside the QGIS product.

The coloured Ink v2 bridge intentionally fails closed on neutral contours.
This shadow experiment records what happens when a user explicitly confirms
the two ends of one dark contour around a printed elevation label.  It does not
inspect a reference while generating the bridge, run a model, alter a QGIS
preview, or select endpoints automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ai_vectorizer.core.edge_detector import EdgeDetector
from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree
from ai_vectorizer.core.manual_gap_bridge import build_manual_gap_bridge
from benchmarks.complex_synthetic import (
    _path_metrics,
    _write_ppm,
    build_neutral_label_gap_case,
    candidate_smoke_gate,
)


# These are a recorded manual prompt, not labels inferred by a detector.  They
# deliberately sit just outside the blank 289 label and describe the visible
# incoming/outgoing contour directions in source-pixel coordinates.
MANUAL_START_XY = (116.0, 118.0)
MANUAL_END_XY = (164.0, 113.0)
MANUAL_START_TANGENT_XY = (1.0, -0.25)
MANUAL_END_TANGENT_XY = (1.0, 0.25)


def _endpoint_preserved(
    path: Sequence[Sequence[float]],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> bool:
    return bool(
        path
        and tuple(float(value) for value in path[0]) == start
        and tuple(float(value) for value in path[-1]) == end
    )


def _score_segment(
    path: Sequence[Sequence[float]],
    reference: Sequence[Tuple[int, int]],
    parallel: Sequence[Tuple[int, int]],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Dict[str, object]:
    metrics = _path_metrics(path, reference, parallel)
    metrics["endpoint_preserved"] = _endpoint_preserved(path, start, end)
    metrics["candidate_smoke_gate"] = candidate_smoke_gate(metrics)
    return metrics


def run_neutral_manual_gap_shadow() -> Dict[str, object]:
    """Compare neutral Ink control with a recorded explicit manual bridge."""

    case = build_neutral_label_gap_case()
    detector = EdgeDetector(method=EdgeDetector.METHOD_INK)
    evidence = detector.detect_ink_evidence(case.image_rgb, tile_origin=(0, 0))
    edges = np.where(evidence.centerline, 255, 0).astype(np.uint8)
    tree = build_livewire_tree(
        case.image_rgb,
        edges,
        case.start_xy,
        strength=1.0,
        incoming_direction=(1.0, 0.0),
        evidence=evidence,
        config=LiveWireConfig(max_window_size=320, target_snap_radius=3),
    )
    neutral_ink_path = tree.trace(case.end_xy)
    neutral_ink_metrics = _path_metrics(
        neutral_ink_path,
        case.reference_xy,
        case.parallel_xy,
    )
    neutral_ink_metrics["endpoint_preserved"] = _endpoint_preserved(
        neutral_ink_path,
        tuple(float(value) for value in case.start_xy),
        tuple(float(value) for value in case.end_xy),
    )
    neutral_ink_metrics["candidate_smoke_gate"] = candidate_smoke_gate(
        neutral_ink_metrics
    )

    bridge = build_manual_gap_bridge(
        MANUAL_START_XY,
        MANUAL_END_XY,
        MANUAL_START_TANGENT_XY,
        MANUAL_END_TANGENT_XY,
    )
    reference_segment = tuple(
        point
        for point in case.reference_xy
        if MANUAL_START_XY[0] <= point[0] <= MANUAL_END_XY[0]
    )
    parallel_segment = tuple(
        point
        for point in case.parallel_xy
        if MANUAL_START_XY[0] <= point[0] <= MANUAL_END_XY[0]
    )
    manual_metrics = _score_segment(
        bridge.points_xy,
        reference_segment,
        parallel_segment,
        MANUAL_START_XY,
        MANUAL_END_XY,
    )
    return {
        "challenge_id": case.name,
        "historical_map_evidence": False,
        "publication_ranking_eligible": False,
        "image_sha256": hashlib.sha256(case.image_rgb.tobytes()).hexdigest(),
        "ink_livewire_v2_neutral_control": {
            "evidence_centerline_pixels": int(np.count_nonzero(evidence.centerline)),
            **neutral_ink_metrics,
        },
        "manual_gap_bridge_shadow_v1": {
            "requires_explicit_user_anchors": True,
            "automatic_endpoint_selection": False,
            "model_or_ocr_used": False,
            "start_xy": list(MANUAL_START_XY),
            "end_xy": list(MANUAL_END_XY),
            "start_tangent_xy": list(MANUAL_START_TANGENT_XY),
            "end_tangent_xy": list(MANUAL_END_TANGENT_XY),
            "gap_length_pixels": bridge.gap_length_pixels,
            "path_length_pixels": bridge.path_length_pixels,
            "detour_ratio": bridge.detour_ratio,
            "score": manual_metrics,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new or empty directory for the JSON evidence and PPM fixture",
    )
    args = parser.parse_args(argv)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise SystemExit("--output-dir must be empty")

    case = build_neutral_label_gap_case()
    result = run_neutral_manual_gap_shadow()
    _write_ppm(destination / "neutral-manual-gap-challenge.ppm", case.image_rgb)
    (destination / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests.
    raise SystemExit(main())
