from types import SimpleNamespace

import numpy as np
import pytest

from ai_vectorizer.core.efficientsam_recovery import EfficientSAMRecoveryEngine
from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT


class _FakeBundle:
    def __init__(self):
        self.reads = []

    def read_bytes(self, identifier):
        self.reads.append(identifier)
        return (identifier + "-verified").encode("ascii")


class _FakeOnnxEngine:
    def __init__(self, encoder_bytes, decoder_bytes, threads):
        assert encoder_bytes == b"encoder-verified"
        assert decoder_bytes == b"decoder-verified"
        assert threads == 1
        self.metadata = {"providers": ["CPUExecutionProvider"]}
        self.encode_calls = 0
        self.predict_calls = 0

    def encode(self, image):
        self.encode_calls += 1
        return SimpleNamespace(
            kind="embedding",
            ordinal=self.encode_calls,
            image_size=tuple(image.shape[:2]),
        )

    def predict(self, encoding, points, labels):
        self.predict_calls += 1
        image_size = tuple(encoding.image_size)
        return SimpleNamespace(
            mask=np.ones(image_size, dtype=bool),
            selected_logits=np.ones(image_size, dtype=np.float32),
            encoding=encoding,
            points=np.asarray(points),
            labels=np.asarray(labels),
        )


def _open_engine():
    captured = {}
    bundle = _FakeBundle()

    def resolver(root, spec):
        captured["root"] = root
        captured["spec"] = spec
        return bundle

    engine = EfficientSAMRecoveryEngine(
        "/verified/cache",
        bundle_resolver=resolver,
        engine_factory=_FakeOnnxEngine,
    )
    return engine, bundle, captured


def test_recovery_engine_resolves_only_the_pinned_verified_bundle():
    engine, bundle, captured = _open_engine()
    assert engine.is_ready is True
    assert captured == {
        "root": "/verified/cache",
        "spec": EFFICIENTSAM_TI_SPLIT,
    }
    assert bundle.reads == ["encoder", "decoder"]
    assert engine.metadata["provider_id"] == "efficientsam-ti-onnx-v1"
    assert engine.metadata["model_bundle_id"] == EFFICIENTSAM_TI_SPLIT.identifier
    assert len(engine.metadata["model_bundle_sha256"]) == 64
    assert engine.metadata["model_source_commit"] == EFFICIENTSAM_TI_SPLIT.source_commit
    assert engine.metadata["model_artifacts_sha256"] == {
        artifact.identifier: artifact.sha256
        for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
    }


def test_image_encoding_is_reused_only_for_identical_bytes():
    engine, _bundle, _captured = _open_engine()
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    first = engine.set_image(image)
    second = engine.set_image(image.copy())
    assert first is second
    assert engine._engine.encode_calls == 1

    image[0, 0, 0] = 1
    third = engine.set_image(image)
    assert third is not first
    assert engine._engine.encode_calls == 2


def test_image_cache_identity_binds_shape_and_preserves_non_square_output():
    engine, _bundle, _captured = _open_engine()
    source = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    first = engine.set_image(source)
    first_prediction = engine.predict(first, [[2, 1]], [1])
    assert first_prediction.mask.shape == (4, 5)

    # The byte stream is identical, but its source coordinate system is not.
    # Cache identity must therefore include shape as well as pixel bytes.
    reshaped = source.reshape(5, 4, 3)
    second = engine.set_image(reshaped)
    second_prediction = engine.predict([[2, 1]], [1])
    assert second is not first
    assert engine._engine.encode_calls == 2
    assert second_prediction.mask.shape == (5, 4)


def test_prediction_contract_rejects_wrong_source_shape_or_dtype():
    engine, _bundle, _captured = _open_engine()
    encoding = engine.set_image(np.zeros((4, 5, 3), dtype=np.uint8))

    engine._engine.predict = lambda *_args: SimpleNamespace(
        mask=np.ones((5, 4), dtype=bool),
        selected_logits=np.ones((5, 4), dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="source-sized bool"):
        engine.predict(encoding, [[2, 1]], [1])

    engine._engine.predict = lambda *_args: SimpleNamespace(
        mask=np.ones((4, 5), dtype=np.uint8),
        selected_logits=np.ones((4, 5), dtype=np.float32),
    )
    with pytest.raises(RuntimeError, match="source-sized bool"):
        engine.predict(encoding, [[2, 1]], [1])


def test_point_prompt_compatibility_returns_mask_and_keeps_evidence():
    engine, _bundle, _captured = _open_engine()
    encoding = engine.encode(np.zeros((4, 5, 3), dtype=np.uint8))
    mask = engine.predict_point([[1, 1], [3, 2]], [1, 1])
    assert mask.dtype == bool
    assert mask.all()
    assert engine.last_prediction is not None
    assert engine._engine.predict_calls == 1
    explicit = engine.predict(encoding, [[2, 1]], [1])
    assert explicit.encoding is encoding
    assert engine._engine.predict_calls == 2

    engine.clear_image()
    assert engine.last_prediction is None
    with pytest.raises(RuntimeError, match="set_image"):
        engine.predict_point([[1, 1]], [1])


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((4, 5), dtype=np.uint8),
        np.zeros((4, 5, 3), dtype=np.float32),
    ],
)
def test_recovery_engine_rejects_non_product_images(image):
    engine, _bundle, _captured = _open_engine()
    with pytest.raises(ValueError, match="Smart Recovery image"):
        engine.set_image(image)
