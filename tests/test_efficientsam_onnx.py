"""QGIS-, NumPy-, and ONNX Runtime-independent EfficientSAM adapter tests."""

import builtins
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

from ai_vectorizer.core import efficientsam_onnx
from ai_vectorizer.core.efficientsam_onnx import (
    EfficientSAMEncoding,
    EfficientSAMOnnxContractError,
    EfficientSAMOnnxEngine,
    EfficientSAMOnnxInputError,
)


class _FakeDType:
    def __init__(self, name):
        self.name = name

    def __call__(self, value):
        if self.name == "float32":
            return float(value)
        if self.name == "int64":
            return int(value)
        return value

    def __repr__(self):
        return "fake.{}".format(self.name)


class _FakeArray:
    def __init__(
        self,
        shape,
        dtype,
        data=None,
        kind="generic",
        finite=True,
        candidates=None,
        selected_index=None,
        divisor=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.data = data
        self.kind = kind
        self.finite = finite
        self.candidates = candidates
        self.selected_index = selected_index
        self.divisor = divisor

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], tuple):
            axes = axes[0]
        return _FakeArray(
            tuple(self.shape[index] for index in axes),
            self.dtype,
            data=self.data,
            kind=self.kind,
            finite=self.finite,
            selected_index=self.selected_index,
            divisor=self.divisor,
        )

    def astype(self, dtype):
        return _FakeArray(
            self.shape,
            dtype,
            data=self.data,
            kind=self.kind,
            finite=self.finite,
            selected_index=self.selected_index,
            divisor=self.divisor,
        )

    def reshape(self, *shape):
        return _FakeArray(
            shape,
            self.dtype,
            data=self.data,
            kind=self.kind,
            finite=self.finite,
            selected_index=self.selected_index,
            divisor=self.divisor,
        )

    def tolist(self):
        return self.data

    def __truediv__(self, divisor):
        return _FakeArray(
            self.shape,
            self.dtype,
            data=self.data,
            kind=self.kind,
            finite=self.finite,
            selected_index=self.selected_index,
            divisor=divisor,
        )

    def __getitem__(self, key):
        if self.kind == "output_masks" and key == (0, 0):
            return _FakeArray(
                self.shape[2:],
                self.dtype,
                kind="all_logits",
                finite=self.finite,
                candidates=self.candidates,
            )
        if self.kind == "all_logits" and isinstance(key, int):
            return self.candidates[key]
        if self.kind == "iou_predictions" and key == (0, 0):
            return _FakeArray(
                (3,),
                self.dtype,
                data=list(self.data),
                kind="candidate_ious",
                finite=self.finite,
            )
        raise AssertionError("Unexpected fake array index {!r} for {}".format(key, self.kind))

    def __ge__(self, threshold):
        return _FakeArray(
            self.shape,
            _FAKE_BOOL,
            kind="mask",
            finite=self.finite,
            selected_index=self.selected_index,
            divisor=threshold,
        )


_FAKE_UINT8 = _FakeDType("uint8")
_FAKE_FLOAT32 = _FakeDType("float32")
_FAKE_INT64 = _FakeDType("int64")
_FAKE_OBJECT = _FakeDType("object")
_FAKE_BOOL = _FakeDType("bool")


def _nested_shape(value):
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    child_shape = _nested_shape(value[0])
    if any(_nested_shape(child) != child_shape for child in value[1:]):
        raise ValueError("ragged array")
    return (len(value),) + child_shape


class _FakeFiniteResult:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self.value


class _FakeNumpy:
    uint8 = _FAKE_UINT8
    float32 = _FAKE_FLOAT32
    int64 = _FAKE_INT64

    def asarray(self, value, dtype=None):
        if isinstance(value, _FakeArray):
            return value if dtype is None or value.dtype == dtype else value.astype(dtype)
        return _FakeArray(
            _nested_shape(value),
            dtype or _FAKE_OBJECT,
            data=list(value) if isinstance(value, (list, tuple)) else value,
        )

    def expand_dims(self, array, axis):
        shape = list(array.shape)
        shape.insert(axis, 1)
        return _FakeArray(
            shape,
            array.dtype,
            data=array.data,
            kind=array.kind,
            finite=array.finite,
            selected_index=array.selected_index,
            divisor=array.divisor,
        )

    @staticmethod
    def isfinite(array):
        return _FakeFiniteResult(array.finite)

    @staticmethod
    def argmax(array):
        return max(range(len(array.data)), key=lambda index: array.data[index])


class _FakeNode:
    def __init__(self, name, value_type, shape):
        self.name = name
        self.type = value_type
        self.shape = list(shape)


def _nodes(contract):
    return [_FakeNode(name, value_type, shape) for name, value_type, shape in contract]


class _FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.execution_mode = None
        self.graph_optimization_level = None


class _FakeExecutionMode:
    ORT_SEQUENTIAL = "fake-sequential"


class _FakeGraphOptimizationLevel:
    ORT_ENABLE_ALL = "fake-enable-all"


def _copy_session_options(options):
    copied = _FakeSessionOptions()
    copied.intra_op_num_threads = options.intra_op_num_threads
    copied.inter_op_num_threads = options.inter_op_num_threads
    copied.execution_mode = options.execution_mode
    copied.graph_optimization_level = options.graph_optimization_level
    return copied


class _FakeSession:
    def __init__(self, kind, owner, options, requested_providers):
        self.kind = kind
        self.owner = owner
        self.options = options
        self.readback_options = _copy_session_options(options)
        self.requested_providers = list(requested_providers)
        self.last_output_names = None
        self.last_feeds = None

        if kind == "encoder":
            self.inputs = _nodes(efficientsam_onnx._ENCODER_INPUTS)
            self.outputs = _nodes(efficientsam_onnx._ENCODER_OUTPUTS)
        else:
            self.inputs = _nodes(efficientsam_onnx._DECODER_INPUTS)
            self.outputs = _nodes(efficientsam_onnx._DECODER_OUTPUTS)

        if owner.bad_contract_kind == kind:
            self.inputs[0].name = "wrong_input_name"
        if owner.bad_options_kind == kind:
            self.readback_options.intra_op_num_threads = 2

    def get_providers(self):
        return list(self.owner.attested_providers)

    def get_session_options(self):
        return self.readback_options

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, output_names, feeds):
        self.last_output_names = list(output_names)
        self.last_feeds = feeds
        if self.kind == "encoder":
            return [
                _FakeArray(
                    (1, 256, 64, 64),
                    _FAKE_FLOAT32,
                    kind="embedding",
                )
            ]

        height, width = feeds["orig_im_size"].data
        output_height = height + 1 if self.owner.bad_decoder_shape else height
        candidates = [
            _FakeArray(
                (output_height, width),
                _FAKE_FLOAT32,
                kind="candidate_logits",
                selected_index=index,
            )
            for index in range(3)
        ]
        masks = _FakeArray(
            (1, 1, 3, output_height, width),
            _FAKE_FLOAT32,
            kind="output_masks",
            candidates=candidates,
        )
        ious = _FakeArray(
            (1, 1, 3),
            _FAKE_FLOAT32,
            data=[0.2, 0.9, 0.4],
            kind="iou_predictions",
        )
        return [masks, ious]


class _FakeOnnxRuntime:
    __version__ = "1.99.fake"
    SessionOptions = _FakeSessionOptions
    ExecutionMode = _FakeExecutionMode
    GraphOptimizationLevel = _FakeGraphOptimizationLevel

    def __init__(
        self,
        attested_providers=("CPUExecutionProvider",),
        bad_contract_kind=None,
        bad_options_kind=None,
        bad_decoder_shape=False,
    ):
        self.attested_providers = tuple(attested_providers)
        self.bad_contract_kind = bad_contract_kind
        self.bad_options_kind = bad_options_kind
        self.bad_decoder_shape = bad_decoder_shape
        self.sessions = []

    def InferenceSession(self, model_bytes, sess_options, providers):
        kind = model_bytes.decode("ascii")
        session = _FakeSession(kind, self, sess_options, providers)
        self.sessions.append(session)
        return session


class EfficientSAMOnnxTests(unittest.TestCase):
    def _engine(self, fake_ort=None):
        fake_numpy = _FakeNumpy()
        fake_ort = fake_ort or _FakeOnnxRuntime()

        def load_dependency(name):
            if name == "numpy":
                return fake_numpy
            if name == "onnxruntime":
                return fake_ort
            raise AssertionError("Unexpected dependency {}".format(name))

        patcher = mock.patch.object(
            efficientsam_onnx.importlib,
            "import_module",
            side_effect=load_dependency,
        )
        with patcher:
            engine = EfficientSAMOnnxEngine(b"encoder", b"decoder")
        return engine, fake_numpy, fake_ort

    def test_module_import_does_not_import_numpy_or_onnxruntime(self):
        source_path = Path(efficientsam_onnx.__file__)
        probe_name = "_efficientsam_onnx_lazy_import_probe"
        spec = importlib.util.spec_from_file_location(probe_name, source_path)
        module = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"numpy", "onnxruntime"}:
                raise AssertionError("optional runtime imported at module import time")
            return original_import(name, *args, **kwargs)

        sys.modules[probe_name] = module
        try:
            with mock.patch("builtins.__import__", side_effect=guarded_import):
                spec.loader.exec_module(module)
        finally:
            sys.modules.pop(probe_name, None)

        self.assertTrue(hasattr(module, "EfficientSAMOnnxEngine"))

    def test_initialization_attests_cpu_and_deterministic_options(self):
        engine, _fake_numpy, fake_ort = self._engine()

        self.assertEqual(len(fake_ort.sessions), 2)
        for session in fake_ort.sessions:
            self.assertEqual(session.requested_providers, ["CPUExecutionProvider"])
            self.assertEqual(session.options.intra_op_num_threads, 1)
            self.assertEqual(session.options.inter_op_num_threads, 1)
            self.assertEqual(session.options.execution_mode, _FakeExecutionMode.ORT_SEQUENTIAL)
            self.assertEqual(
                session.options.graph_optimization_level,
                _FakeGraphOptimizationLevel.ORT_ENABLE_ALL,
            )
        self.assertEqual(
            engine.metadata["session_options"]["graph_optimization_level"],
            "ORT_ENABLE_ALL",
        )
        self.assertEqual(
            engine.metadata["session_options_by_session"],
            {
                "encoder": {
                    "intra_op_num_threads": 1,
                    "inter_op_num_threads": 1,
                    "execution_mode": "ORT_SEQUENTIAL",
                    "graph_optimization_level": "ORT_ENABLE_ALL",
                },
                "decoder": {
                    "intra_op_num_threads": 1,
                    "inter_op_num_threads": 1,
                    "execution_mode": "ORT_SEQUENTIAL",
                    "graph_optimization_level": "ORT_ENABLE_ALL",
                },
            },
        )

        metadata = engine.metadata
        self.assertEqual(metadata["onnxruntime_version"], "1.99.fake")
        self.assertEqual(metadata["providers"]["encoder"], ["CPUExecutionProvider"])
        self.assertEqual(metadata["providers"]["decoder"], ["CPUExecutionProvider"])
        self.assertEqual(metadata["session_options"]["execution_mode"], "ORT_SEQUENTIAL")
        self.assertGreaterEqual(metadata["timing_ms"]["session_initialization"], 0.0)

    def test_initialization_rejects_provider_fallback(self):
        fake_ort = _FakeOnnxRuntime(
            attested_providers=("CoreMLExecutionProvider", "CPUExecutionProvider")
        )
        with self.assertRaisesRegex(
            EfficientSAMOnnxContractError,
            "provider attestation failed",
        ):
            self._engine(fake_ort)

        # Decoder construction never follows a failed encoder attestation.
        self.assertEqual(len(fake_ort.sessions), 1)

    def test_initialization_rejects_fixed_io_contract_mismatch(self):
        fake_ort = _FakeOnnxRuntime(bad_contract_kind="decoder")
        with self.assertRaisesRegex(
            EfficientSAMOnnxContractError,
            "decoder input contract mismatch",
        ):
            self._engine(fake_ort)

    def test_initialization_rejects_session_option_readback_mismatch(self):
        fake_ort = _FakeOnnxRuntime(bad_options_kind="encoder")
        with self.assertRaisesRegex(
            EfficientSAMOnnxContractError,
            "encoder session options attestation failed",
        ):
            self._engine(fake_ort)

        # Decoder construction never follows a failed encoder attestation.
        self.assertEqual(len(fake_ort.sessions), 1)

    def test_non_deterministic_thread_count_is_rejected_before_lazy_import(self):
        with mock.patch.object(efficientsam_onnx.importlib, "import_module") as importer:
            with self.assertRaisesRegex(EfficientSAMOnnxInputError, "threads must be 1"):
                EfficientSAMOnnxEngine(b"encoder", b"decoder", threads=2)
        importer.assert_not_called()

    def test_encode_and_predict_obey_tensor_and_selection_contract(self):
        engine, _fake_numpy, fake_ort = self._engine()
        image = _FakeArray((2, 3, 3), _FAKE_UINT8, kind="rgb")

        encoding = engine.encode(image)
        prediction = engine.predict(
            encoding,
            points_xy=[[0.0, 0.0], [2.5, 1.5]],
            labels=[1, 0],
        )

        self.assertIsInstance(encoding, EfficientSAMEncoding)
        self.assertEqual(encoding.image_size, (2, 3))
        encoder_feed = fake_ort.sessions[0].last_feeds["batched_images"]
        self.assertEqual(encoder_feed.shape, (1, 3, 2, 3))
        self.assertIs(encoder_feed.dtype, _FAKE_FLOAT32)
        self.assertEqual(encoder_feed.divisor, 255.0)
        self.assertEqual(fake_ort.sessions[0].last_output_names, ["image_embeddings"])

        decoder = fake_ort.sessions[1]
        self.assertEqual(
            decoder.last_output_names,
            ["output_masks", "iou_predictions"],
        )
        self.assertEqual(decoder.last_feeds["batched_point_coords"].shape, (1, 1, 2, 2))
        self.assertIs(decoder.last_feeds["batched_point_coords"].dtype, _FAKE_FLOAT32)
        self.assertEqual(decoder.last_feeds["batched_point_labels"].shape, (1, 1, 2))
        self.assertIs(decoder.last_feeds["batched_point_labels"].dtype, _FAKE_FLOAT32)
        self.assertEqual(decoder.last_feeds["orig_im_size"].data, [2, 3])
        self.assertIs(decoder.last_feeds["orig_im_size"].dtype, _FAKE_INT64)

        self.assertEqual(prediction.selected_index, 1)
        self.assertEqual(prediction.selected_logits.selected_index, 1)
        self.assertEqual(prediction.mask.kind, "mask")
        self.assertEqual(prediction.mask.selected_index, 1)
        self.assertEqual(prediction.mask.divisor, 0.0)
        self.assertEqual(prediction.all_logits.shape, (3, 2, 3))
        self.assertEqual(prediction.iou_predictions.data, [0.2, 0.9, 0.4])
        self.assertEqual(prediction.metadata["image_size"], [2, 3])
        self.assertEqual(prediction.metadata["point_count"], 2)
        self.assertGreaterEqual(prediction.metadata["timing_ms"]["encoder"], 0.0)
        self.assertGreaterEqual(prediction.metadata["timing_ms"]["decoder"], 0.0)

    def test_inputs_reject_wrong_image_prompt_labels_and_bounds(self):
        engine, _fake_numpy, _fake_ort = self._engine()

        with self.assertRaisesRegex(EfficientSAMOnnxInputError, "uint8"):
            engine.encode(_FakeArray((2, 3, 3), _FAKE_FLOAT32))
        with self.assertRaisesRegex(EfficientSAMOnnxInputError, "HWC shape"):
            engine.encode(_FakeArray((2, 3), _FAKE_UINT8))
        with self.assertRaisesRegex(EfficientSAMOnnxInputError, "must not exceed 1024x1024"):
            engine.encode(_FakeArray((1, 1025, 3), _FAKE_UINT8))

        encoding = engine.encode(_FakeArray((2, 3, 3), _FAKE_UINT8))
        with self.assertRaisesRegex(EfficientSAMOnnxInputError, "1..6"):
            engine.predict(encoding, [[1.0, 1.0]] * 7, [1] * 7)
        with self.assertRaisesRegex(EfficientSAMOnnxInputError, "bounding-box"):
            engine.predict(encoding, [[1.0, 1.0]], [2])
        with self.assertRaisesRegex(EfficientSAMOnnxInputError, "outside image bounds"):
            engine.predict(encoding, [[3.0, 1.0]], [1])

    def test_runtime_output_shape_mismatch_is_rejected(self):
        engine, _fake_numpy, _fake_ort = self._engine(
            _FakeOnnxRuntime(bad_decoder_shape=True)
        )
        encoding = engine.encode(_FakeArray((2, 3, 3), _FAKE_UINT8))

        with self.assertRaisesRegex(
            EfficientSAMOnnxContractError,
            "output_masks shape mismatch",
        ):
            engine.predict(encoding, [[1.0, 1.0]], [1])

    def test_embedding_from_another_engine_is_rejected(self):
        first, _fake_numpy, _fake_ort = self._engine()
        second, _fake_numpy, _fake_ort = self._engine()
        encoding = first.encode(_FakeArray((2, 3, 3), _FAKE_UINT8))

        with self.assertRaisesRegex(EfficientSAMOnnxInputError, "different"):
            second.predict(encoding, [[1.0, 1.0]], [1])


if __name__ == "__main__":
    unittest.main()
