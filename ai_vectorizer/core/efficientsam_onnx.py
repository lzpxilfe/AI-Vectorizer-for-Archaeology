# -*- coding: utf-8 -*-
"""Deterministic CPU adapter for the pinned EfficientSAM-Ti split ONNX model.

This module deliberately does not own model discovery, download, or integrity
verification.  Callers must pass the already verified encoder and decoder model
bytes to :class:`EfficientSAMOnnxEngine`.

NumPy and ONNX Runtime remain optional plugin dependencies: neither package is
imported until an engine instance is created.
"""

from dataclasses import dataclass, field
import importlib
import time
from typing import Any, Dict, Mapping, Tuple


CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
MAX_POINT_PROMPTS = 6
MAX_IMAGE_DIMENSION = 1_024


class EfficientSAMOnnxError(RuntimeError):
    """Base error raised by the EfficientSAM ONNX adapter."""


class EfficientSAMOnnxDependencyError(ImportError):
    """Raised when a lazy runtime dependency cannot be imported."""


class EfficientSAMOnnxContractError(EfficientSAMOnnxError):
    """Raised when a session or runtime result violates the pinned contract."""


class EfficientSAMOnnxInputError(ValueError):
    """Raised when an image or prompt is outside the supported input contract."""


@dataclass(frozen=True)
class EfficientSAMEncoding:
    """An encoder result retaining the source image coordinate system."""

    image_embeddings: Any
    image_size: Tuple[int, int]
    encoder_ms: float
    _engine_token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class EfficientSAMOnnxPrediction:
    """Best EfficientSAM mask plus the evidence needed to reproduce selection."""

    mask: Any
    selected_logits: Any
    all_logits: Any
    iou_predictions: Any
    selected_index: int
    metadata: Mapping[str, Any]


# These symbolic dimension names are part of the pinned ONNX binaries, not a
# permissive description of any EfficientSAM export.  A model update must update
# this contract and its integrity specification together.
_ENCODER_INPUTS = (
    ("batched_images", "tensor(float)", ("batch", 3, "height", "width")),
)
_ENCODER_OUTPUTS = (
    (
        "image_embeddings",
        "tensor(float)",
        (
            "Addimage_embeddings_dim_0",
            256,
            "Addimage_embeddings_dim_2",
            "Addimage_embeddings_dim_3",
        ),
    ),
)
_DECODER_INPUTS = (
    ("image_embeddings", "tensor(float)", ("batch", 256, 64, 64)),
    ("batched_point_coords", "tensor(float)", (1, 1, "num_points", 2)),
    ("batched_point_labels", "tensor(float)", (1, 1, "num_points")),
    ("orig_im_size", "tensor(int64)", (2,)),
)
_DECODER_OUTPUTS = (
    (
        "output_masks",
        "tensor(float)",
        (
            "Reshapeoutput_masks_dim_0",
            "Reshapeoutput_masks_dim_1",
            "Reshapeoutput_masks_dim_2",
            "Reshapeoutput_masks_dim_3",
            "Reshapeoutput_masks_dim_4",
        ),
    ),
    ("iou_predictions", "tensor(float)", (1, 1, "Reshapeiou_predictions_dim_2")),
    (
        "onnx::Shape_1830",
        "tensor(float)",
        (
            "Sliceonnx::Shape_1830_dim_0",
            "Reshapeiou_predictions_dim_2",
            "Sliceonnx::Shape_1830_dim_2",
            "Sliceonnx::Shape_1830_dim_3",
        ),
    ),
)


def _load_dependency(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise EfficientSAMOnnxDependencyError(
            "EfficientSAM ONNX requires the optional '{}' package: {}".format(
                module_name,
                exc,
            )
        ) from exc


def _model_bytes(value, label):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("{} must be verified ONNX model bytes".format(label))
    result = value if isinstance(value, bytes) else bytes(value)
    if not result:
        raise ValueError("{} must not be empty".format(label))
    return result


def _node_contract(nodes):
    result = []
    for node in nodes:
        shape = getattr(node, "shape", None)
        result.append(
            (
                getattr(node, "name", None),
                getattr(node, "type", None),
                tuple(shape) if shape is not None else None,
            )
        )
    return tuple(result)


def _attest_io_contract(session, session_name, expected_inputs, expected_outputs):
    actual_inputs = _node_contract(session.get_inputs())
    actual_outputs = _node_contract(session.get_outputs())
    if actual_inputs != expected_inputs:
        raise EfficientSAMOnnxContractError(
            "{} input contract mismatch: expected {!r}, got {!r}".format(
                session_name,
                expected_inputs,
                actual_inputs,
            )
        )
    if actual_outputs != expected_outputs:
        raise EfficientSAMOnnxContractError(
            "{} output contract mismatch: expected {!r}, got {!r}".format(
                session_name,
                expected_outputs,
                actual_outputs,
            )
        )


def _attest_cpu_provider(session, session_name):
    actual = tuple(session.get_providers())
    expected = (CPU_EXECUTION_PROVIDER,)
    if actual != expected:
        raise EfficientSAMOnnxContractError(
            "{} provider attestation failed: expected {!r}, got {!r}".format(
                session_name,
                expected,
                actual,
            )
        )
    return actual


def _attest_session_options(session, session_name, ort_module):
    try:
        observed = session.get_session_options()
    except Exception as exc:
        raise EfficientSAMOnnxContractError(
            "{} session options could not be read back: {}".format(
                session_name,
                exc,
            )
        ) from exc

    intra_threads = getattr(observed, "intra_op_num_threads", None)
    inter_threads = getattr(observed, "inter_op_num_threads", None)
    execution_mode = getattr(observed, "execution_mode", None)
    graph_optimization_level = getattr(observed, "graph_optimization_level", None)
    matches = (
        not isinstance(intra_threads, bool)
        and intra_threads == 1
        and not isinstance(inter_threads, bool)
        and inter_threads == 1
        and execution_mode == ort_module.ExecutionMode.ORT_SEQUENTIAL
        and graph_optimization_level
        == ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    if not matches:
        raise EfficientSAMOnnxContractError(
            "{} session options attestation failed: got "
            "intra={!r}, inter={!r}, execution_mode={!r}, "
            "graph_optimization_level={!r}".format(
                session_name,
                intra_threads,
                inter_threads,
                execution_mode,
                graph_optimization_level,
            )
        )
    return {
        "intra_op_num_threads": int(intra_threads),
        "inter_op_num_threads": int(inter_threads),
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization_level": "ORT_ENABLE_ALL",
    }


def _elapsed_ms(started_at):
    return max(0.0, (time.perf_counter() - started_at) * 1000.0)


class EfficientSAMOnnxEngine:
    """Run the pinned EfficientSAM-Ti split ONNX model on deterministic CPU.

    Args:
        encoder_bytes: Integrity-verified ``efficient_sam_vitt_encoder.onnx`` bytes.
        decoder_bytes: Integrity-verified ``efficient_sam_vitt_decoder.onnx`` bytes.
        threads: Must remain one for the deterministic runtime contract.
    """

    def __init__(self, encoder_bytes, decoder_bytes, threads=1):
        if isinstance(threads, bool) or threads != 1:
            raise EfficientSAMOnnxInputError(
                "threads must be 1 for deterministic EfficientSAM CPU inference"
            )

        encoder_model = _model_bytes(encoder_bytes, "encoder_bytes")
        decoder_model = _model_bytes(decoder_bytes, "decoder_bytes")

        # Optional heavy dependencies are intentionally imported here, never at
        # module import time.
        self._np = _load_dependency("numpy")
        self._ort = _load_dependency("onnxruntime")
        self._engine_token = object()

        started_at = time.perf_counter()
        options = self._ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = self._ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            self._encoder = self._ort.InferenceSession(
                encoder_model,
                sess_options=options,
                providers=[CPU_EXECUTION_PROVIDER],
            )
            # Attest immediately; never construct or inspect the decoder after a
            # provider fallback on the encoder.
            self._encoder_providers = _attest_cpu_provider(self._encoder, "encoder")
            self._encoder_session_options = _attest_session_options(
                self._encoder,
                "encoder",
                self._ort,
            )

            self._decoder = self._ort.InferenceSession(
                decoder_model,
                sess_options=options,
                providers=[CPU_EXECUTION_PROVIDER],
            )
            self._decoder_providers = _attest_cpu_provider(self._decoder, "decoder")
            self._decoder_session_options = _attest_session_options(
                self._decoder,
                "decoder",
                self._ort,
            )
        except EfficientSAMOnnxContractError:
            raise
        except Exception as exc:
            raise EfficientSAMOnnxError(
                "Could not initialize EfficientSAM ONNX sessions: {}".format(exc)
            ) from exc

        _attest_io_contract(
            self._encoder,
            "encoder",
            _ENCODER_INPUTS,
            _ENCODER_OUTPUTS,
        )
        _attest_io_contract(
            self._decoder,
            "decoder",
            _DECODER_INPUTS,
            _DECODER_OUTPUTS,
        )

        self._session_initialization_ms = _elapsed_ms(started_at)
        self._onnxruntime_version = str(getattr(self._ort, "__version__", "unknown"))

    @property
    def metadata(self):
        """Return a fresh JSON-friendly runtime attestation snapshot."""
        return {
            "onnxruntime_version": self._onnxruntime_version,
            "providers": {
                "encoder": list(self._encoder_providers),
                "decoder": list(self._decoder_providers),
            },
            "session_options": dict(self._encoder_session_options),
            "session_options_by_session": {
                "encoder": dict(self._encoder_session_options),
                "decoder": dict(self._decoder_session_options),
            },
            "timing_ms": {
                "session_initialization": self._session_initialization_ms,
            },
        }

    def _require_finite(self, array, label, error_type):
        try:
            finite = bool(self._np.isfinite(array).all())
        except Exception as exc:
            raise error_type("{} could not be checked for finite values".format(label)) from exc
        if not finite:
            raise error_type("{} must contain only finite values".format(label))

    def _require_runtime_array(self, array, label, expected_shape, expected_dtype):
        actual_shape = tuple(getattr(array, "shape", ()))
        if actual_shape != tuple(expected_shape):
            raise EfficientSAMOnnxContractError(
                "{} shape mismatch: expected {!r}, got {!r}".format(
                    label,
                    tuple(expected_shape),
                    actual_shape,
                )
            )
        if getattr(array, "dtype", None) != expected_dtype:
            raise EfficientSAMOnnxContractError(
                "{} dtype mismatch: expected {!r}, got {!r}".format(
                    label,
                    expected_dtype,
                    getattr(array, "dtype", None),
                )
            )
        self._require_finite(array, label, EfficientSAMOnnxContractError)

    def encode(self, rgb_u8):
        """Encode an RGB uint8 HWC image and retain its original ``(H, W)``."""
        image = self._np.asarray(rgb_u8)
        image_shape = tuple(getattr(image, "shape", ()))
        if len(image_shape) != 3 or image_shape[2] != 3:
            raise EfficientSAMOnnxInputError(
                "rgb_u8 must have HWC shape (height, width, 3); got {!r}".format(
                    image_shape
                )
            )
        height, width, _channels = image_shape
        if isinstance(height, bool) or isinstance(width, bool) or height < 1 or width < 1:
            raise EfficientSAMOnnxInputError("rgb_u8 height and width must be positive")
        if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
            raise EfficientSAMOnnxInputError(
                "rgb_u8 dimensions must not exceed {}x{}; got {}x{}".format(
                    MAX_IMAGE_DIMENSION,
                    MAX_IMAGE_DIMENSION,
                    width,
                    height,
                )
            )
        if getattr(image, "dtype", None) != self._np.uint8:
            raise EfficientSAMOnnxInputError(
                "rgb_u8 must have uint8 dtype; got {!r}".format(
                    getattr(image, "dtype", None)
                )
            )

        tensor = image.transpose(2, 0, 1)
        tensor = self._np.expand_dims(tensor, axis=0).astype(self._np.float32)
        tensor = tensor / self._np.float32(255.0)

        expected_tensor_shape = (1, 3, height, width)
        if tuple(getattr(tensor, "shape", ())) != expected_tensor_shape:
            raise EfficientSAMOnnxContractError(
                "preprocessed image shape mismatch: expected {!r}, got {!r}".format(
                    expected_tensor_shape,
                    tuple(getattr(tensor, "shape", ())),
                )
            )
        if getattr(tensor, "dtype", None) != self._np.float32:
            raise EfficientSAMOnnxContractError(
                "preprocessed image must remain float32 after scaling"
            )
        self._require_finite(
            tensor,
            "preprocessed image",
            EfficientSAMOnnxInputError,
        )

        started_at = time.perf_counter()
        try:
            embeddings = self._encoder.run(
                ["image_embeddings"],
                {"batched_images": tensor},
            )[0]
        except Exception as exc:
            raise EfficientSAMOnnxError(
                "EfficientSAM encoder inference failed: {}".format(exc)
            ) from exc
        encoder_ms = _elapsed_ms(started_at)

        self._require_runtime_array(
            embeddings,
            "image_embeddings",
            (1, 256, 64, 64),
            self._np.float32,
        )
        return EfficientSAMEncoding(
            image_embeddings=embeddings,
            image_size=(int(height), int(width)),
            encoder_ms=encoder_ms,
            _engine_token=self._engine_token,
        )

    def _prompt_tensors(self, image_size, points_xy, labels):
        height, width = image_size
        try:
            points = self._np.asarray(points_xy, dtype=self._np.float32)
        except Exception as exc:
            raise EfficientSAMOnnxInputError(
                "points_xy must be a rectangular numeric array"
            ) from exc

        point_shape = tuple(getattr(points, "shape", ()))
        if len(point_shape) != 2 or point_shape[1] != 2:
            raise EfficientSAMOnnxInputError(
                "points_xy must have shape (N, 2); got {!r}".format(point_shape)
            )
        point_count = point_shape[0]
        if point_count < 1 or point_count > MAX_POINT_PROMPTS:
            raise EfficientSAMOnnxInputError(
                "EfficientSAM requires 1..{} point prompts; got {}".format(
                    MAX_POINT_PROMPTS,
                    point_count,
                )
            )
        self._require_finite(points, "points_xy", EfficientSAMOnnxInputError)

        for index, point in enumerate(points.tolist()):
            x, y = point
            if x < 0 or x >= width or y < 0 or y >= height:
                raise EfficientSAMOnnxInputError(
                    "points_xy[{}] ({}, {}) is outside image bounds x=[0, {}), y=[0, {})".format(
                        index,
                        x,
                        y,
                        width,
                        height,
                    )
                )

        try:
            raw_labels = self._np.asarray(labels)
        except Exception as exc:
            raise EfficientSAMOnnxInputError("labels must be a one-dimensional array") from exc
        label_shape = tuple(getattr(raw_labels, "shape", ()))
        if label_shape != (point_count,):
            raise EfficientSAMOnnxInputError(
                "labels must have shape ({},); got {!r}".format(
                    point_count,
                    label_shape,
                )
            )
        label_values = raw_labels.tolist()
        if any(value not in (0, 1) for value in label_values):
            raise EfficientSAMOnnxInputError(
                "labels must contain only point labels {0, 1}; bounding-box labels {2, 3} are unsupported"
            )

        point_tensor = points.reshape(1, 1, point_count, 2)
        label_tensor = self._np.asarray(
            label_values,
            dtype=self._np.float32,
        ).reshape(1, 1, point_count)
        original_size = self._np.asarray([height, width], dtype=self._np.int64)
        return point_tensor, label_tensor, original_size, point_count

    def predict(self, embedding, points_xy, labels):
        """Predict and select the maximum-IoU mask for point prompts.

        ``embedding`` must be the :class:`EfficientSAMEncoding` returned by this
        engine. Labels ``0`` and ``1`` are negative and positive points;
        bounding-box labels are intentionally rejected.
        """
        if not isinstance(embedding, EfficientSAMEncoding):
            raise EfficientSAMOnnxInputError(
                "embedding must be an EfficientSAMEncoding returned by encode()"
            )
        if embedding._engine_token is not self._engine_token:
            raise EfficientSAMOnnxInputError(
                "embedding was produced by a different EfficientSAM engine"
            )

        height, width = embedding.image_size
        self._require_runtime_array(
            embedding.image_embeddings,
            "image_embeddings",
            (1, 256, 64, 64),
            self._np.float32,
        )
        points, prompt_labels, original_size, point_count = self._prompt_tensors(
            embedding.image_size,
            points_xy,
            labels,
        )

        feeds = {
            "image_embeddings": embedding.image_embeddings,
            "batched_point_coords": points,
            "batched_point_labels": prompt_labels,
            "orig_im_size": original_size,
        }
        started_at = time.perf_counter()
        try:
            output_masks, iou_predictions = self._decoder.run(
                ["output_masks", "iou_predictions"],
                feeds,
            )
        except Exception as exc:
            raise EfficientSAMOnnxError(
                "EfficientSAM decoder inference failed: {}".format(exc)
            ) from exc
        decoder_ms = _elapsed_ms(started_at)

        self._require_runtime_array(
            output_masks,
            "output_masks",
            (1, 1, 3, height, width),
            self._np.float32,
        )
        self._require_runtime_array(
            iou_predictions,
            "iou_predictions",
            (1, 1, 3),
            self._np.float32,
        )

        candidate_ious = iou_predictions[0, 0]
        selected_index = int(self._np.argmax(candidate_ious))
        all_logits = output_masks[0, 0]
        selected_logits = all_logits[selected_index]
        mask = selected_logits >= self._np.float32(0.0)

        metadata: Dict[str, Any] = self.metadata
        metadata["timing_ms"].update(
            {
                "encoder": embedding.encoder_ms,
                "decoder": decoder_ms,
                "total_inference": embedding.encoder_ms + decoder_ms,
            }
        )
        metadata["image_size"] = [height, width]
        metadata["point_count"] = point_count

        return EfficientSAMOnnxPrediction(
            mask=mask,
            selected_logits=selected_logits,
            all_logits=all_logits,
            iou_predictions=candidate_ious,
            selected_index=selected_index,
            metadata=metadata,
        )
