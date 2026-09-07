"""Dependency-light contract tests for the optional DexiNed shadow runner."""

import json

import numpy as np
import pytest

from benchmarks.dexined_shadow import (
    DEXINED_INPUT_SHAPE,
    DEXINED_MODEL_REVISION,
    DEXINED_MODEL_SHA256,
    DEXINED_MODEL_URL,
    DexiNedShadowError,
    _dexined_input,
    _normalised_final_edge_score,
    main,
    trace_dexined_shadow,
    validate_dexined_model,
)


def test_dexined_pinned_experiment_identity_is_complete():
    assert len(DEXINED_MODEL_REVISION) == 40
    assert len(DEXINED_MODEL_SHA256) == 64
    assert DEXINED_MODEL_REVISION in DEXINED_MODEL_URL


def test_dexined_preprocess_and_final_score_have_fixed_shapes():
    image = np.full((12, 16, 3), 220, dtype=np.uint8)
    image[5:8, 4:12] = (180, 225, 225)
    blob = _dexined_input(image)
    assert blob.shape == DEXINED_INPUT_SHAPE
    assert blob.dtype == np.float32

    logits = np.linspace(
        -5.0,
        5.0,
        num=DEXINED_INPUT_SHAPE[2] * DEXINED_INPUT_SHAPE[3],
        dtype=np.float32,
    ).reshape((1, 1, DEXINED_INPUT_SHAPE[2], DEXINED_INPUT_SHAPE[3]))
    score = _normalised_final_edge_score(logits, image.shape[:2])
    assert score.shape == image.shape[:2]
    assert np.isfinite(score).all()
    assert float(score.min()) >= 0.0
    assert float(score.max()) <= 1.0


def test_dexined_model_validation_fails_closed_for_wrong_local_artifact(tmp_path):
    path = tmp_path / "not-a-model.onnx"
    path.write_bytes(b"not a model")
    with pytest.raises(DexiNedShadowError, match="size"):
        validate_dexined_model(path)


def test_dexined_inference_failure_is_not_silently_replaced():
    class FailingSession:
        @staticmethod
        def get_inputs():
            return [type("Input", (), {"name": "img"})()]

        @staticmethod
        def run(*_args, **_kwargs):
            raise RuntimeError("unsupported quantization")

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(DexiNedShadowError, match="CPU inference failed"):
        trace_dexined_shadow(FailingSession(), image, (1, 1), (14, 14))


def test_dexined_cli_writes_a_failed_record_and_nonzero_exit(tmp_path, capsys):
    model = tmp_path / "wrong.onnx"
    model.write_bytes(b"wrong")
    output = tmp_path / "result"

    assert main(["--model", str(model), "--output-dir", str(output)]) == 2
    record = json.loads(capsys.readouterr().out)
    assert record["execution"]["status"] == "failed"
    assert record["model_contract"]["revision"] == DEXINED_MODEL_REVISION
