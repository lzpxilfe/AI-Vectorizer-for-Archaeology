"""Reproducible product-kernel checks for explicitly requested label-gap bridges.

The fixed manual prompts are inputs, not inferred annotations. Both methods
receive the same start/end points; the bridge samples directions from the same
Ink evidence using the helper called by QGIS. Rejection retains the Ink path.
References are consulted only when scoring the completed paths. These synthetic
checks are neither historical-map accuracy evidence nor a human usability study.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ai_vectorizer.core.edge_detector import EdgeDetector
from ai_vectorizer.core.livewire import (
    LiveWireConfig,
    blend_path_with_cursor,
    build_livewire_tree,
)
from ai_vectorizer.core.manual_gap_bridge import (
    DEFAULT_MANUAL_GAP_BRIDGE_CONFIG,
    ManualGapBridgeError,
    build_manual_gap_bridge,
    sample_manual_gap_bridge_tangents,
    sample_manual_gap_tangent,
)
from ai_vectorizer.core.trace_kernel import smooth_pixel_path
from benchmarks.complex_synthetic import (
    ComplexTraceCase,
    _path_metrics,
    _paint_digit,
    _paint_path,
    _write_ppm,
    build_neutral_label_gap_case,
    candidate_smoke_gate,
)


# Preserve the original recorded prompt, including difficult near-glyph points.
# No tangent or reference-derived slope is supplied to the bridge.
MANUAL_START_XY = (116.0, 118.0)
MANUAL_END_XY = (164.0, 113.0)
TANGENT_RADIUS_PIXELS = 3
STRENGTH = 1.0
SMOOTH_WINDOW_SIZE = 5
LIVEWIRE_CONFIG = LiveWireConfig(max_window_size=320, target_snap_radius=6)


@dataclass(frozen=True)
class ManualGapShadowCase:
    name: str
    image_rgb: np.ndarray
    reference_case: ComplexTraceCase
    grayscale: bool
    quarter_turns: int
    canonical_start_xy: Tuple[float, float] = MANUAL_START_XY
    canonical_end_xy: Tuple[float, float] = MANUAL_END_XY
    fixture_family: str = "glyph-adjacent-original"

    def to_image_xy(self, point: Sequence[float]) -> Tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        return (y, self.image_rgb.shape[0] - 1.0 - x) if self.quarter_turns else (x, y)

    def to_reference_xy(self, point: Sequence[float]) -> Tuple[float, float]:
        x, y = float(point[0]), float(point[1])
        return (self.image_rgb.shape[0] - 1.0 - y, x) if self.quarter_turns else (x, y)


def build_manual_gap_shadow_cases() -> Tuple[ManualGapShadowCase, ...]:
    """Return brown/achromatic fixtures at two predeclared sheet orientations.

    The original fixture's so-called neutral ink is brown. The gray variants
    make every R/G/B value equal. Rotation applies losslessly to the complete
    sheet and prompts, including the printed label, rather than interpolating
    the raster or selecting easier endpoint pixels.
    """

    base = build_neutral_label_gap_case()
    cases = []
    for grayscale in (False, True):
        for quarter_turns in (0, 1):
            image = base.image_rgb.copy()
            if grayscale:
                gray = np.rint(np.mean(image.astype(np.float32), axis=2)).astype(np.uint8)
                image = np.repeat(gray[..., None], 3, axis=2)
            image = np.ascontiguousarray(np.rot90(image, quarter_turns))
            image.setflags(write=False)
            color = "gray" if grayscale else "brown"
            cases.append(ManualGapShadowCase(
                name="{}-label-gap-{}deg-v2".format(color, quarter_turns * 90),
                image_rgb=image,
                reference_case=base,
                grayscale=grayscale,
                quarter_turns=quarter_turns,
            ))
    # A positive control places both explicit anchors on visible, straight
    # contour sections six pixels back from the blank. Digits remain inside
    # the gap. The simple input validates evidence sampling + preview creation
    # without pretending that every glyph-adjacent endpoint is recoverable.
    clear_image = np.full((256, 256, 3), 238, dtype=np.uint8)
    reference = tuple((x, 128) for x in range(16, 241))
    parallel = tuple((x, 152) for x in range(16, 241))
    _paint_path(clear_image, reference, (80, 80, 80), 1, omitted_x=(118, 162))
    _paint_path(clear_image, parallel, (64, 64, 64), 1)
    for digit, x in zip("289", (127, 137, 147)):
        _paint_digit(clear_image, digit, x, 119, scale=2)
    gray = np.rint(np.mean(clear_image.astype(np.float32), axis=2)).astype(np.uint8)
    clear_image = np.repeat(gray[..., None], 3, axis=2)
    clear_base = ComplexTraceCase(
        name="gray-clear-label-gap-v2", image_rgb=clear_image,
        start_xy=reference[0], end_xy=reference[-1],
        reference_xy=reference, parallel_xy=parallel,
    )
    for quarter_turns in (0, 1):
        image = np.ascontiguousarray(np.rot90(clear_image, quarter_turns))
        image.setflags(write=False)
        cases.append(ManualGapShadowCase(
            name="gray-clear-label-gap-{}deg-v2".format(quarter_turns * 90),
            image_rgb=image, reference_case=clear_base, grayscale=True,
            quarter_turns=quarter_turns,
            canonical_start_xy=(110.0, 128.0),
            canonical_end_xy=(170.0, 128.0),
            fixture_family="clear-gap-positive-control",
        ))
    # This negative control intentionally has no target contour at either
    # anchor.  A dark, parallel contour and the numeric glyphs are real Ink
    # evidence, so a successful bridge here would prove the fallback had
    # started inventing a route instead of requiring outward support.
    ambiguous_image = np.full((256, 256, 3), 238, dtype=np.uint8)
    ambiguous_reference = tuple((x, 128) for x in range(16, 241))
    ambiguous_parallel = tuple((x, 142) for x in range(16, 241))
    _paint_path(ambiguous_image, ambiguous_parallel, (64, 64, 64), 1)
    for digit, x in zip("289", (127, 137, 147)):
        _paint_digit(ambiguous_image, digit, x, 119, scale=2)
    ambiguous_gray = np.rint(
        np.mean(ambiguous_image.astype(np.float32), axis=2)
    ).astype(np.uint8)
    ambiguous_image = np.repeat(ambiguous_gray[..., None], 3, axis=2)
    ambiguous_base = ComplexTraceCase(
        name="gray-glyph-parallel-no-contour-v1", image_rgb=ambiguous_image,
        start_xy=ambiguous_reference[0], end_xy=ambiguous_reference[-1],
        reference_xy=ambiguous_reference, parallel_xy=ambiguous_parallel,
    )
    for quarter_turns in (0, 1):
        image = np.ascontiguousarray(np.rot90(ambiguous_image, quarter_turns))
        image.setflags(write=False)
        cases.append(ManualGapShadowCase(
            name="gray-glyph-parallel-no-contour-{}deg-v1".format(quarter_turns * 90),
            image_rgb=image, reference_case=ambiguous_base, grayscale=True,
            quarter_turns=quarter_turns,
            canonical_start_xy=(116.0, 128.0),
            canonical_end_xy=(164.0, 128.0),
            fixture_family="glyph-parallel-negative-control",
        ))
    return tuple(cases)


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _points_json(points: Sequence[Sequence[float]]) -> list:
    return [[float(x), float(y)] for x, y in points]


def _source_hashes() -> Dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = (
        "benchmarks/manual_gap_shadow.py",
        "benchmarks/complex_synthetic.py",
        "ai_vectorizer/core/edge_detector.py",
        "ai_vectorizer/core/line_evidence.py",
        "ai_vectorizer/core/livewire.py",
        "ai_vectorizer/core/manual_gap_bridge.py",
        "ai_vectorizer/core/trace_kernel.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def _configuration() -> Dict[str, object]:
    return {
        "livewire": asdict(LIVEWIRE_CONFIG),
        "manual_gap_bridge": asdict(DEFAULT_MANUAL_GAP_BRIDGE_CONFIG),
        "tangent_radius_pixels": TANGENT_RADIUS_PIXELS,
        "strength": STRENGTH,
        "livewire_smooth_window_size": SMOOTH_WINDOW_SIZE,
        "tile_origin_xy": [0, 0],
    }


def _score_path(case, path, start, end, *, full_trace=False):
    reference = case.reference_case.reference_xy
    parallel = case.reference_case.parallel_xy
    if not full_trace:
        reference = tuple(p for p in reference if case.canonical_start_xy[0] <= p[0] <= case.canonical_end_xy[0])
        parallel = tuple(p for p in parallel if case.canonical_start_xy[0] <= p[0] <= case.canonical_end_xy[0])
    # A rigid inverse rotation makes the existing x-interval label-gap metric
    # applicable. Distances are unchanged; emitted routes stay in image pixels.
    metrics = _path_metrics([case.to_reference_xy(point) for point in path], reference, parallel)
    metrics["scoring_frame"] = "canonical_unrotated_source_pixels"
    metrics["endpoint_preserved"] = bool(
        path and tuple(path[0]) == tuple(start) and tuple(path[-1]) == tuple(end)
    )
    metrics["candidate_smoke_gate"] = candidate_smoke_gate(metrics)
    return metrics


def _ink_path(case, evidence, start, end, incoming_direction=None):
    tree = build_livewire_tree(
        case.image_rgb,
        np.where(evidence.centerline, 255, 0).astype(np.uint8),
        start,
        strength=STRENGTH,
        incoming_direction=incoming_direction,
        evidence=evidence,
        config=LIVEWIRE_CONFIG,
    )
    kernel_path = tree.trace(end)
    path = list(kernel_path)
    # Match the QGIS cursor-preview smoothing, preserving the routed endpoints.
    if len(path) > SMOOTH_WINDOW_SIZE:
        path = list(smooth_pixel_path(path, window_size=SMOOTH_WINDOW_SIZE))
        path[0], path[-1] = kernel_path[0], kernel_path[-1]
    return path, kernel_path


def run_manual_gap_case(case: ManualGapShadowCase) -> Dict[str, object]:
    """Run a fixed case without consulting its reference during path generation."""

    evidence = EdgeDetector.detect_ink_evidence(case.image_rgb, tile_origin=(0, 0))
    start = case.to_image_xy(case.canonical_start_xy)
    end = case.to_image_xy(case.canonical_end_xy)
    prompt = {"start_xy": list(start), "end_xy": list(end), "previous_xy": None}
    control, raw_control = _ink_path(case, evidence, start, end)
    start_tangent = sample_manual_gap_tangent(evidence, start, radius_pixels=TANGENT_RADIUS_PIXELS)
    end_tangent = sample_manual_gap_tangent(evidence, end, radius_pixels=TANGENT_RADIUS_PIXELS)
    bridge = None
    reason = None
    tangent_source = "sample_manual_gap_tangent(ink_evidence)"
    try:
        if start_tangent is None or end_tangent is None:
            raise ManualGapBridgeError("supported unambiguous endpoint tangents unavailable")
        bridge = build_manual_gap_bridge(
            start, end, start_tangent, end_tangent,
            config=DEFAULT_MANUAL_GAP_BRIDGE_CONFIG,
        )
        effective_path = blend_path_with_cursor(bridge.points_xy, start, end, STRENGTH)
    except ManualGapBridgeError as exc:
        contextual_tangents = sample_manual_gap_bridge_tangents(evidence, start, end)
        if contextual_tangents is None:
            reason = str(exc)
            effective_path = list(control)
        else:
            try:
                start_tangent, end_tangent = contextual_tangents
                bridge = build_manual_gap_bridge(
                    start, end, start_tangent, end_tangent,
                    config=DEFAULT_MANUAL_GAP_BRIDGE_CONFIG,
                )
                effective_path = blend_path_with_cursor(
                    bridge.points_xy, start, end, STRENGTH,
                )
                tangent_source = (
                    "sample_manual_gap_bridge_tangents(ink_evidence, explicit_anchors)"
                )
            except ManualGapBridgeError:
                bridge = None
                reason = str(exc)
                effective_path = list(control)

    # Full-trace context uses different endpoints and is deliberately outside
    # the paired intervention comparison.
    full_start = case.to_image_xy(case.reference_case.start_xy)
    full_end = case.to_image_xy(case.reference_case.end_xy)
    full_incoming = (0.0, -1.0) if case.quarter_turns else (1.0, 0.0)
    full_path, _ = _ink_path(case, evidence, full_start, full_end, full_incoming)
    config = _configuration()
    control_json = _points_json(control)
    effective_json = _points_json(effective_path)
    return {
        "challenge_id": case.name,
        "fixture_family": case.fixture_family,
        "historical_map_evidence": False,
        "publication_ranking_eligible": False,
        "human_usability_study": False,
        "image_sha256": hashlib.sha256(case.image_rgb.tobytes()).hexdigest(),
        "image_shape": list(case.image_rgb.shape),
        "reference_sha256": _json_sha256({
            "ordered_reference_xy": case.reference_case.reference_xy,
            "parallel_xy": case.reference_case.parallel_xy,
        }),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
        },
        "transform": {"grayscale": case.grayscale, "counterclockwise_degrees": 90 * case.quarter_turns},
        "prompt": prompt,
        "prompt_sha256": _json_sha256(prompt),
        "configuration": config,
        "configuration_sha256": _json_sha256(config),
        "source_sha256": _source_hashes(),
        "evidence_centerline_pixels": int(np.count_nonzero(evidence.centerline)),
        "ink_same_segment_control": {
            "points_xy": control_json,
            "kernel_points_xy": _points_json(raw_control),
            "route_sha256": _json_sha256(control_json),
            "score": _score_path(case, control, start, end),
        },
        "manual_gap_bridge_product_kernel_v2": {
            "requires_explicit_user_anchors": True,
            "automatic_endpoint_selection": False,
            "model_or_ocr_used": False,
            "tangent_source": tangent_source,
            "sampled_start_tangent_xy": None if start_tangent is None else list(start_tangent),
            "sampled_end_tangent_xy": None if end_tangent is None else list(end_tangent),
            "status": "preview" if bridge is not None else "ink_fallback",
            "reason": reason,
            "bridge_points_xy": None if bridge is None else _points_json(bridge.points_xy),
            "effective_points_xy": effective_json,
            "route_sha256": _json_sha256(effective_json),
            "gap_length_pixels": None if bridge is None else bridge.gap_length_pixels,
            "detour_ratio": None if bridge is None else bridge.detour_ratio,
            "score": _score_path(case, effective_path, start, end),
        },
        "full_trace_context": {
            "included_in_paired_comparison": False,
            "prompt": {"start_xy": list(full_start), "end_xy": list(full_end), "incoming_direction_xy": list(full_incoming)},
            "points_xy": _points_json(full_path),
            "score": _score_path(case, full_path, full_start, full_end, full_trace=True),
        },
    }


def run_neutral_manual_gap_shadow() -> Dict[str, object]:
    """Compatibility entry point for the original (brown, not gray) fixture."""

    return run_manual_gap_case(build_manual_gap_shadow_cases()[0])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="new or empty directory for JSON and lossless PPM fixtures")
    args = parser.parse_args(argv)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise SystemExit("--output-dir must be empty")

    cases = build_manual_gap_shadow_cases()
    result = {
        "suite_id": "manual-gap-product-parity-v2",
        "historical_map_evidence": False,
        "publication_ranking_eligible": False,
        "human_usability_study": False,
        "cases": [run_manual_gap_case(case) for case in cases],
    }
    for case in cases:
        _write_ppm(destination / (case.name + ".ppm"), case.image_rgb)
    (destination / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests.
    raise SystemExit(main())
