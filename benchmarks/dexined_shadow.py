"""Explicit, CPU-only DexiNed shadow test for the complex Ink fixture.

This module is intentionally a benchmark experiment, not a QGIS backend or a
model downloader.  It accepts only a user-supplied, fixed-hash ONNX file and
keeps the current Ink v2 binary centreline and tangents authoritative.  The
learned edge response is added only as continuous soft support before the
shared bounded Live-Wire consumes it.  The resulting ordered path is scored
*after inference* by :mod:`benchmarks.complex_synthetic`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ai_vectorizer.core.edge_detector import EdgeDetector
from ai_vectorizer.core.line_evidence import LineEvidence
from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree
from benchmarks.complex_synthetic import (
    build_complex_trace_case,
    run_complex_ink_challenge,
    score_complex_candidate_path,
)


DEXINED_MODEL_ID = "opencv/edge_detection_dexined"
DEXINED_MODEL_REVISION = "01a752eb4006688376e576dc24941c54608f6dd0"
DEXINED_MODEL_FILE = "edge_detection_dexined_2024sep.onnx"
DEXINED_MODEL_SHA256 = (
    "a50d01dc8481549c7dedb9eb3e0123b810a016520df75e4669a504609982cdd0"
)
DEXINED_MODEL_SIZE_BYTES = 47_235_563
DEXINED_MODEL_LICENSE = "MIT"
DEXINED_MODEL_URL = (
    "https://huggingface.co/opencv/edge_detection_dexined/resolve/"
    + DEXINED_MODEL_REVISION
    + "/"
    + DEXINED_MODEL_FILE
)
DEXINED_INPUT_SHAPE = (1, 3, 480, 640)
DEXINED_SOFT_SUPPORT_WEIGHT = 0.65


class DexiNedShadowError(RuntimeError):
    """Raised when a candidate run cannot meet its explicit local contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dexined_model(path: Path) -> Dict[str, object]:
    """Verify the only model binary accepted by this experimental adapter."""

    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise DexiNedShadowError("DexiNed model must be a regular local file")
    size = candidate.stat().st_size
    if size != DEXINED_MODEL_SIZE_BYTES:
        raise DexiNedShadowError(
            "DexiNed model size does not match the pinned experiment binary"
        )
    digest = _sha256_file(candidate)
    if digest != DEXINED_MODEL_SHA256:
        raise DexiNedShadowError(
            "DexiNed model SHA-256 does not match the pinned experiment binary"
        )
    return {
        "model_id": DEXINED_MODEL_ID,
        "revision": DEXINED_MODEL_REVISION,
        "file": DEXINED_MODEL_FILE,
        "sha256": digest,
        "size_bytes": size,
        "license": DEXINED_MODEL_LICENSE,
        "source_url": DEXINED_MODEL_URL,
    }


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise DexiNedShadowError(
            "DexiNed shadow testing requires onnxruntime; it is not a plugin dependency"
        ) from exc
    return ort


def _open_cpu_session(model_path: Path) -> Tuple[Any, Dict[str, object]]:
    ort = _require_onnxruntime()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise DexiNedShadowError("DexiNed CPU ONNX session could not be opened") from exc
    providers = tuple(session.get_providers())
    if providers != ("CPUExecutionProvider",):
        raise DexiNedShadowError(
            "DexiNed shadow run requires exactly CPUExecutionProvider"
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or tuple(inputs[0].shape) != DEXINED_INPUT_SHAPE:
        raise DexiNedShadowError("DexiNed ONNX input shape differs from pinned contract")
    if len(outputs) != 7 or any(
        tuple(output.shape) != DEXINED_INPUT_SHAPE[:1] + (1,) + DEXINED_INPUT_SHAPE[2:]
        for output in outputs
    ):
        raise DexiNedShadowError("DexiNed ONNX output shape differs from pinned contract")
    return session, {
        "providers": list(providers),
        "input_name": inputs[0].name,
        "input_shape": list(DEXINED_INPUT_SHAPE),
        "output_names": [output.name for output in outputs],
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
    }


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    try:
        from scipy.ndimage import zoom
    except ImportError as exc:
        raise DexiNedShadowError(
            "DexiNed shadow testing requires SciPy for deterministic resize"
        ) from exc
    values = np.asarray(image, dtype=np.float32)
    if values.ndim not in (2, 3) or min(values.shape[:2]) < 1:
        raise DexiNedShadowError("DexiNed image must be a non-empty gray or RGB array")
    factors = (float(height) / values.shape[0], float(width) / values.shape[1])
    if values.ndim == 3:
        factors = factors + (1.0,)
    resized = zoom(values, factors, order=1, mode="nearest", prefilter=False)
    expected = (height, width) if values.ndim == 2 else (height, width, values.shape[2])
    if resized.shape != expected:
        raise DexiNedShadowError("DexiNed resize did not produce the requested shape")
    return np.ascontiguousarray(resized, dtype=np.float32)


def _dexined_input(image_rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(image_rgb)
    if values.ndim != 3 or values.shape[2] < 3:
        raise DexiNedShadowError("DexiNed shadow input requires three RGB channels")
    _batch, _channels, height, width = DEXINED_INPUT_SHAPE
    resized_rgb = _resize_bilinear(values[..., :3], height, width)
    # OpenCV's reference implementation calls blobFromImage on its BGR image
    # with this BGR mean.  Reproduce that visible preprocessing exactly while
    # keeping the benchmark fixture itself in RGB product order.
    bgr = resized_rgb[..., ::-1]
    blob = np.transpose(bgr, (2, 0, 1))[None, ...]
    blob = blob - np.asarray((103.5, 116.2, 123.6), dtype=np.float32)[
        None, :, None, None
    ]
    return np.ascontiguousarray(blob, dtype=np.float32)


def _normalised_final_edge_score(
    output: Any,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    logits = np.asarray(output, dtype=np.float32)
    if logits.shape != (1, 1, DEXINED_INPUT_SHAPE[2], DEXINED_INPUT_SHAPE[3]):
        raise DexiNedShadowError("DexiNed final output shape is invalid")
    clipped = np.clip(logits[0, 0], -50.0, 50.0)
    probability = 1.0 / (1.0 + np.exp(-clipped))
    low = float(probability.min())
    high = float(probability.max())
    if not np.isfinite(probability).all() or high <= low + np.finfo(np.float32).eps:
        raise DexiNedShadowError("DexiNed final edge output is non-finite or constant")
    normalised = (probability - low) / (high - low)
    return _resize_bilinear(normalised, image_shape[0], image_shape[1])


def trace_dexined_shadow(
    session: Any,
    image_rgb: np.ndarray,
    start_xy: Sequence[float],
    end_xy: Sequence[float],
) -> Tuple[Sequence[Tuple[float, float]], Dict[str, float]]:
    """Use DexiNed only as continuous support over an Ink v2 route.

    The binary ``centerline`` and tangent field come from Ink v2.  In
    particular, a DexiNed threshold never becomes a saved line or a binary OR
    with Ink; this keeps the shadow test aligned with the product's proposed
    model-combination safety boundary.
    """

    started = time.perf_counter_ns()
    detector = EdgeDetector(method=EdgeDetector.METHOD_INK)
    ink = detector.detect_ink_evidence(image_rgb, tile_origin=(0, 0))
    model_input = _dexined_input(image_rgb)
    try:
        output = session.run(None, {session.get_inputs()[0].name: model_input})[-1]
    except Exception as exc:
        # Do not substitute OpenCV DNN, another ONNX Runtime provider, or an
        # older model in the background.  A candidate whose pinned CPU path
        # fails is evidence against integrating that exact artifact.
        raise DexiNedShadowError("DexiNed CPU inference failed") from exc
    dexined_score = _normalised_final_edge_score(output, ink.shape)
    combined_score = np.maximum(
        ink.center_score,
        np.clip(dexined_score * DEXINED_SOFT_SUPPORT_WEIGHT, 0.0, 1.0),
    ).astype(np.float32)
    combined = LineEvidence(
        center_score=combined_score,
        centerline=ink.centerline,
        tangent_x=ink.tangent_x,
        tangent_y=ink.tangent_y,
        coherence=ink.coherence,
        scale_px=ink.scale_px,
    )
    edges = np.where(ink.centerline, 255, 0).astype(np.uint8)
    tree = build_livewire_tree(
        image_rgb,
        edges,
        (int(start_xy[0]), int(start_xy[1])),
        strength=1.0,
        incoming_direction=(1.0, 0.0),
        evidence=combined,
        config=LiveWireConfig(max_window_size=320, target_snap_radius=3),
    )
    path = tree.trace((float(end_xy[0]), float(end_xy[1])))
    return path, {
        "wall_ms": float((time.perf_counter_ns() - started) / 1_000_000.0),
        "dexined_score_p95": float(np.percentile(dexined_score, 95)),
        "combined_score_p95": float(np.percentile(combined_score, 95)),
    }


def run_dexined_shadow(model_path: Path) -> Dict[str, object]:
    """Execute the pinned candidate once and score its ordered final route."""

    model = validate_dexined_model(model_path)
    session_started = time.perf_counter_ns()
    session, runtime = _open_cpu_session(Path(model_path))
    runtime["model_load_wall_ms"] = float(
        (time.perf_counter_ns() - session_started) / 1_000_000.0
    )
    case = build_complex_trace_case()
    path, trace_runtime = trace_dexined_shadow(
        session,
        case.image_rgb,
        case.start_xy,
        case.end_xy,
    )
    return {
        "candidate_id": "dexined-shadow-soft-evidence-v1",
        "historical_map_evidence": False,
        "publication_ranking_eligible": False,
        "combination_contract": (
            "max(ink_v2_center_score, 0.65 * dexined_final_edge_score); "
            "Ink v2 centerline/tangent retained; no binary OR"
        ),
        "model": model,
        "runtime": runtime,
        "trace_runtime": trace_runtime,
        "score": score_complex_candidate_path(path),
        "baseline_ink_v2": run_complex_ink_challenge()["ink_livewire_v2"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="explicit local DexiNed ONNX path")
    parser.add_argument("--output-dir", required=True, help="new or empty result directory")
    args = parser.parse_args(argv)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise SystemExit("--output-dir must be empty")

    try:
        result = run_dexined_shadow(Path(args.model))
    except DexiNedShadowError as exc:
        result = {
            "candidate_id": "dexined-shadow-soft-evidence-v1",
            "historical_map_evidence": False,
            "publication_ranking_eligible": False,
            "model_contract": {
                "model_id": DEXINED_MODEL_ID,
                "revision": DEXINED_MODEL_REVISION,
                "file": DEXINED_MODEL_FILE,
                "expected_sha256": DEXINED_MODEL_SHA256,
            },
            "execution": {"status": "failed", "reason": str(exc)},
        }
    (destination / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 2 if result.get("execution", {}).get("status") == "failed" else 0


if __name__ == "__main__":  # pragma: no cover - CLI only.
    raise SystemExit(main())
