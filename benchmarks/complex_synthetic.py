"""A reproducible, adversarial synthetic map challenge for Ink tracing.

This is a *regression and integration* fixture, not historical-map accuracy
evidence.  It intentionally combines several map-like failure modes in one
small deterministic image so a contributor can run the real product kernels
before proposing another detector or a larger model:

* a coloured, locally faded, thick target contour;
* a short scan break, a numeric-label gap and nearby darker parallel contour;
* number glyphs and a pale grid line crossing the target;
* paper grain, stain/bleed patches and isolated dark speckles.

The baseline calls the same ``EdgeDetector`` and ``build_livewire_tree`` APIs
as the plugin.  It neither fabricates evidence nor treats a segmentation mask
as a final line.  Its compact JSON output is useful for local candidate-model
experiments, but must never be published as a historical-map ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ai_vectorizer.core.edge_detector import EdgeDetector
from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree


Point = Tuple[int, int]
FloatPoint = Tuple[float, float]
CANDIDATE_SMOKE_REQUIREMENTS = {
    "target_p95_distance_px": 4.0,
    "bidirectional_coverage_within_4px": 0.92,
    "numeric_label_gap_coverage_within_4px": 0.85,
    "parallel_switch_fraction": 0.03,
}
NUMERIC_LABEL_GAP_X = (118, 162)


@dataclass(frozen=True)
class ComplexTraceCase:
    """One immutable-looking input and its independently specified target."""

    name: str
    image_rgb: np.ndarray
    start_xy: Point
    end_xy: Point
    reference_xy: Tuple[Point, ...]
    parallel_xy: Tuple[Point, ...]


def _curve_y(x: int) -> int:
    return int(round(121.0 + 9.0 * math.sin((x - 16) / 29.0)))


def _paint_disc(
    image: np.ndarray,
    x: int,
    y: int,
    color: Sequence[int],
    radius: int,
    *,
    alpha: float = 1.0,
) -> None:
    """Paint a clipped circular stroke without an image-library dependency."""

    height, width = image.shape[:2]
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
    selection = (grid_x - x) ** 2 + (grid_y - y) ** 2 <= radius**2
    source = image[y0:y1, x0:x1]
    if alpha >= 1.0:
        source[selection] = np.asarray(color, dtype=np.uint8)
        return
    blended = (
        source[selection].astype(np.float32) * (1.0 - alpha)
        + np.asarray(color, dtype=np.float32) * alpha
    )
    source[selection] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _paint_path(
    image: np.ndarray,
    points: Iterable[Point],
    color: Sequence[int],
    radius: int,
    *,
    omitted_x: Optional[Tuple[int, int]] = None,
    omitted_ranges: Sequence[Tuple[int, int]] = (),
) -> None:
    for x, y in points:
        if (omitted_x is not None and omitted_x[0] <= x <= omitted_x[1]) or any(
            start <= x <= end for start, end in omitted_ranges
        ):
            continue
        _paint_disc(image, x, y, color, radius)


_DIGITS = {
    "2": ("111", "001", "111", "100", "111"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def _paint_digit(
    image: np.ndarray,
    digit: str,
    x: int,
    y: int,
    *,
    scale: int = 4,
) -> None:
    pattern = _DIGITS[digit]
    for row, values in enumerate(pattern):
        for column, value in enumerate(values):
            if value != "1":
                continue
            for offset_y in range(scale):
                for offset_x in range(scale):
                    _paint_disc(
                        image,
                        x + column * scale + offset_x,
                        y + row * scale + offset_y,
                        (42, 35, 30),
                        0,
                    )


def _build_complex_trace_case(
    *,
    name: str,
    target_color: Sequence[int],
    faded_color: Sequence[int],
) -> ComplexTraceCase:
    """Build one deterministic map-like image for an explicitly named case."""

    height = width = 256
    generator = np.random.default_rng(20260907)
    paper = np.array((232, 220, 196), dtype=np.int16)
    grain = generator.normal(0.0, 4.0, size=(height, width, 1))
    image = np.clip(paper + grain, 0, 255).astype(np.uint8)

    reference = tuple((x, _curve_y(x)) for x in range(16, 241))
    parallel = tuple((x, _curve_y(x) + 14) for x in range(16, 241))

    # A darker nearby contour is deliberately attractive to a pure edge map.
    _paint_path(image, parallel, (70, 58, 48), 1)

    # The intended contour is blue and thick.  The map's elevation label is
    # conventionally set over an *actual blank* in the contour -- its line is
    # not merely hidden under dark glyph pixels.  A separate nine-pixel scan
    # loss forces a route to bridge both kinds of discontinuity rather than
    # merely skeletonize a continuous stroke.
    _paint_path(
        image,
        reference,
        target_color,
        2,
        omitted_x=NUMERIC_LABEL_GAP_X,
        omitted_ranges=((174, 182),),
    )
    faded = tuple(point for point in reference if 82 <= point[0] <= 116)
    _paint_path(image, faded, faded_color, 2)

    # Pale survey grid and an elevation label occupy the target's blank label
    # gap.  The target reference remains independently specified above, so the
    # evaluator measures whether a tracer reconnects the intended contour.
    for y in range(66, 194):
        _paint_disc(image, 78, y, (147, 131, 108), 0, alpha=0.72)
    for x in range(25, 233):
        _paint_disc(image, x, 164, (153, 138, 116), 0, alpha=0.58)
    _paint_digit(image, "2", 118, 105)
    _paint_digit(image, "8", 135, 105)
    _paint_digit(image, "9", 151, 105)

    # Scan stains/bleed-through appear as broad, low-contrast dark patches;
    # none lies exactly on the target's known centreline.
    for x, y, radius in ((43, 93, 12), (101, 169, 15), (186, 86, 13), (217, 179, 11)):
        _paint_disc(image, x, y, (124, 102, 88), radius, alpha=0.13)
    for x, y in ((29, 41), (58, 211), (96, 51), (202, 48), (229, 212), (242, 92)):
        _paint_disc(image, x, y, (30, 28, 25), 1)

    image.setflags(write=False)
    return ComplexTraceCase(
        name=name,
        image_rgb=image,
        start_xy=reference[0],
        end_xy=reference[-1],
        reference_xy=reference,
        parallel_xy=parallel,
    )


def build_complex_trace_case() -> ComplexTraceCase:
    """Return the fixed coloured map-like image used by the Ink smoke gate."""

    return _build_complex_trace_case(
        name="coloured-faded-text-parallel-stain-v1",
        target_color=(185, 226, 226),
        faded_color=(207, 231, 231),
    )


def build_neutral_label_gap_case() -> ComplexTraceCase:
    """Return the dark-contour variant for an explicit manual-gap experiment.

    This intentionally removes the colour distinction which lets the automatic
    Ink v2 bridge fail closed.  The matching manual shadow can then evaluate a
    user-confirmed bridge without pretending a detector can infer the answer.
    """

    return _build_complex_trace_case(
        name="neutral-faded-text-parallel-stain-v1",
        target_color=(102, 84, 70),
        faded_color=(130, 112, 98),
    )


def _path_metrics(
    path: Sequence[FloatPoint],
    reference: Sequence[Point],
    parallel: Sequence[Point],
) -> Dict[str, Union[float, int, bool, str]]:
    predicted = np.asarray(path, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    parallel_array = np.asarray(parallel, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1:] != (2,) or len(predicted) < 2:
        raise ValueError("trace path must contain at least two (x, y) points")

    reference_distance = np.sqrt(
        np.min(
            np.sum((predicted[:, None, :] - reference_array[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )
    reference_to_path_distance = np.sqrt(
        np.min(
            np.sum((reference_array[:, None, :] - predicted[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )
    parallel_distance = np.sqrt(
        np.min(
            np.sum((predicted[:, None, :] - parallel_array[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )
    interior = (predicted[:, 0] >= reference_array[0, 0] + 8) & (
        predicted[:, 0] <= reference_array[-1, 0] - 8
    )
    interior_count = max(1, int(np.count_nonzero(interior)))
    switched = np.count_nonzero(
        (parallel_distance + 1.0 < reference_distance) & interior
    )
    label_reference = reference_array[
        (reference_array[:, 0] >= NUMERIC_LABEL_GAP_X[0])
        & (reference_array[:, 0] <= NUMERIC_LABEL_GAP_X[1])
    ]
    label_distance = np.sqrt(
        np.min(
            np.sum(
                (label_reference[:, None, :] - predicted[None, :, :]) ** 2,
                axis=2,
            ),
            axis=1,
        )
    )
    encoded = json.dumps(
        [[round(float(x), 4), round(float(y), 4)] for x, y in predicted],
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "vertex_count": int(len(predicted)),
        "target_mean_distance_px": float(np.mean(reference_distance)),
        "target_p95_distance_px": float(np.percentile(reference_distance, 95)),
        "target_coverage_within_4px": float(np.mean(reference_distance <= 4.0)),
        "reference_coverage_within_4px": float(
            np.mean(reference_to_path_distance <= 4.0)
        ),
        "numeric_label_gap_coverage_within_4px": float(
            np.mean(label_distance <= 4.0)
        ),
        "numeric_label_gap_p95_distance_px": float(np.percentile(label_distance, 95)),
        "parallel_switch_fraction": float(switched / interior_count),
        "path_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def candidate_smoke_gate(
    metrics: Dict[str, Union[float, int, bool, str]],
) -> Dict[str, object]:
    """Assess a candidate without turning this fixture into a quality claim.

    The limits are deliberately strict enough to reject the current difficult
    baseline: an integration candidate should not be advertised merely because
    it returns an endpoint-preserving line.  The caller records the reasons
    beside the path metrics; this function does not tune a model or inspect a
    reference while a model runs.
    """

    reasons = []
    if metrics.get("endpoint_preserved") is not True:
        reasons.append("endpoint_not_preserved")
    if float(metrics["target_p95_distance_px"]) > CANDIDATE_SMOKE_REQUIREMENTS[
        "target_p95_distance_px"
    ]:
        reasons.append("target_p95_exceeds_4px")
    coverage = min(
        float(metrics["target_coverage_within_4px"]),
        float(metrics["reference_coverage_within_4px"]),
    )
    if coverage < CANDIDATE_SMOKE_REQUIREMENTS[
        "bidirectional_coverage_within_4px"
    ]:
        reasons.append("bidirectional_coverage_below_0.92")
    if float(metrics["numeric_label_gap_coverage_within_4px"]) < (
        CANDIDATE_SMOKE_REQUIREMENTS["numeric_label_gap_coverage_within_4px"]
    ):
        reasons.append("numeric_label_gap_coverage_below_0.85")
    if float(metrics["parallel_switch_fraction"]) > CANDIDATE_SMOKE_REQUIREMENTS[
        "parallel_switch_fraction"
    ]:
        reasons.append("parallel_switch_fraction_exceeds_0.03")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "requirements": dict(CANDIDATE_SMOKE_REQUIREMENTS),
    }


def score_complex_candidate_path(
    path: Sequence[Sequence[float]],
) -> Dict[str, object]:
    """Score an externally produced ordered route against the fixed fixture.

    This is the boundary used for future shadow adapters: models receive the
    image and start/end prompt while their output is produced; only afterwards
    does this evaluator see the stored ordered route and reference.  A finite
    size and in-image check avoids accidental quadratic allocations from a
    malformed local experiment file.
    """

    case = build_complex_trace_case()
    try:
        point_count = len(path)
    except TypeError as exc:
        raise ValueError("candidate path must be a sequence of (x, y) points") from exc
    if point_count < 2 or point_count > 4_096:
        raise ValueError("candidate path must contain between 2 and 4096 points")
    try:
        points = np.asarray(path, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate path must contain numeric (x, y) points") from exc
    if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
        raise ValueError("candidate path must contain finite (x, y) points")
    height, width = case.image_rgb.shape[:2]
    if (
        np.any(points[:, 0] < 0.0)
        or np.any(points[:, 0] >= width)
        or np.any(points[:, 1] < 0.0)
        or np.any(points[:, 1] >= height)
    ):
        raise ValueError("candidate path leaves the challenge image bounds")

    metrics = _path_metrics(points, case.reference_xy, case.parallel_xy)
    metrics["endpoint_preserved"] = bool(
        tuple(points[0]) == tuple(map(float, case.start_xy))
        and tuple(points[-1]) == tuple(map(float, case.end_xy))
    )
    metrics["candidate_smoke_gate"] = candidate_smoke_gate(metrics)
    return metrics


def run_complex_ink_challenge() -> Dict[str, object]:
    """Run both real Ink adapters and return JSON-safe diagnostic evidence."""

    case = build_complex_trace_case()
    config = LiveWireConfig(
        max_window_size=320,
        target_snap_radius=3,
    )
    detector = EdgeDetector(method=EdgeDetector.METHOD_INK)

    legacy_edges = detector.detect_edges(case.image_rgb)
    legacy_tree = build_livewire_tree(
        case.image_rgb,
        legacy_edges,
        case.start_xy,
        strength=1.0,
        incoming_direction=(1.0, 0.0),
        config=config,
    )
    legacy_path = legacy_tree.trace(case.end_xy)

    evidence = detector.detect_ink_evidence(case.image_rgb, tile_origin=(0, 0))
    v2_edges = np.where(evidence.centerline, 255, 0).astype(np.uint8)
    v2_tree = build_livewire_tree(
        case.image_rgb,
        v2_edges,
        case.start_xy,
        strength=1.0,
        incoming_direction=(1.0, 0.0),
        evidence=evidence,
        config=config,
    )
    v2_path = v2_tree.trace(case.end_xy)

    legacy_metrics = score_complex_candidate_path(legacy_path)
    v2_metrics = score_complex_candidate_path(v2_path)
    result = {
        "challenge_id": case.name,
        "historical_map_evidence": False,
        "publication_ranking_eligible": False,
        "image_sha256": hashlib.sha256(case.image_rgb.tobytes()).hexdigest(),
        "input": {
            "width": int(case.image_rgb.shape[1]),
            "height": int(case.image_rgb.shape[0]),
            "start_xy": list(case.start_xy),
            "end_xy": list(case.end_xy),
            "failure_modes": [
                "coloured_thick_target",
                "faded_target",
                "scan_break",
                "numeric_label_blank_gap",
                "text_and_grid_crossing",
                "nearby_parallel_contour",
                "survey_grid",
                "paper_grain_stain_and_speckles",
            ],
        },
        "ink_livewire_v1": {
            "edge_pixels": int(np.count_nonzero(legacy_edges)),
            **legacy_metrics,
        },
        "ink_livewire_v2": {
            "evidence_centerline_pixels": int(np.count_nonzero(evidence.centerline)),
            "evidence_score_p95": float(np.percentile(evidence.center_score, 95)),
            **v2_metrics,
        },
    }
    return result


def _write_ppm(path: Path, image: np.ndarray) -> None:
    """Write an inspectable lossless PPM without adding a runtime dependency."""

    height, width = image.shape[:2]
    payload = b"P6\n%d %d\n255\n" % (width, height) + image.tobytes()
    path.write_bytes(payload)


def _load_candidate_path(path: Path) -> Sequence[Sequence[float]]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("candidate path JSON could not be read") from exc
    if isinstance(payload, dict):
        payload = payload.get("points_xy")
    if isinstance(payload, (str, bytes)) or not hasattr(payload, "__len__"):
        raise ValueError("candidate JSON must be a points_xy array or object")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new or empty directory for the JSON evidence and PPM fixture",
    )
    parser.add_argument(
        "--candidate-path",
        help="optional JSON array (or points_xy object) from a shadow adapter",
    )
    args = parser.parse_args(argv)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise SystemExit("--output-dir must be empty")

    case = build_complex_trace_case()
    result = run_complex_ink_challenge()
    if args.candidate_path:
        result["candidate"] = score_complex_candidate_path(
            _load_candidate_path(Path(args.candidate_path))
        )
    _write_ppm(destination / "complex-trace-challenge.ppm", case.image_rgb)
    (destination / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI command.
    raise SystemExit(main())
