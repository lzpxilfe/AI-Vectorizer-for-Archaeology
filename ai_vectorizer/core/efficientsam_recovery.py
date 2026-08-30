"""Offline product adapter for the optional EfficientSAM recovery provider.

Construction verifies the content-addressed model bundle and loads ONNX bytes
locally.  Network access is intentionally absent; the UI must call the model
store's explicit ``fetch_bundle`` action in a separate user-authorised task.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Callable

from .efficientsam_onnx import EfficientSAMOnnxEngine, MAX_IMAGE_DIMENSION
from .efficientsam_spec import EFFICIENTSAM_TI_SPLIT, bundle_fingerprint
from .model_store import inspect_bundle, resolve_bundle


class EfficientSAMRecoveryEngine:
    """Expose pinned EfficientSAM through the existing point-prompt surface."""

    provider_id = "efficientsam-ti-onnx-v1"

    def __init__(
        self,
        cache_root,
        *,
        bundle_resolver: Callable = resolve_bundle,
        engine_factory: Callable = EfficientSAMOnnxEngine,
    ):
        bundle = bundle_resolver(cache_root, EFFICIENTSAM_TI_SPLIT)
        encoder_bytes = bundle.read_bytes("encoder")
        decoder_bytes = bundle.read_bytes("decoder")
        self._engine = engine_factory(encoder_bytes, decoder_bytes, threads=1)
        self._encoding = None
        self._image_sha256 = None
        self._last_prediction = None
        self.is_ready = True

    @staticmethod
    def inspect(cache_root):
        """Inspect local files only; this function never creates or downloads."""

        return inspect_bundle(cache_root, EFFICIENTSAM_TI_SPLIT)

    @property
    def metadata(self):
        metadata = dict(getattr(self._engine, "metadata", {}))
        metadata["provider_id"] = self.provider_id
        metadata["model_bundle_id"] = EFFICIENTSAM_TI_SPLIT.identifier
        metadata["model_bundle_sha256"] = bundle_fingerprint(EFFICIENTSAM_TI_SPLIT)
        metadata["model_source_commit"] = EFFICIENTSAM_TI_SPLIT.source_commit
        metadata["model_artifacts_sha256"] = {
            artifact.identifier: artifact.sha256
            for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
        }
        return metadata

    @property
    def last_prediction(self):
        return self._last_prediction

    def set_image(self, rgb_u8):
        """Encode a cache image once, invalidating prior prompt results."""

        import numpy as np

        image = np.asarray(rgb_u8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Smart Recovery image must have HWC RGB shape")
        if image.dtype != np.uint8:
            raise ValueError("Smart Recovery image must use uint8 pixels")
        height, width, channels = image.shape
        if height < 1 or width < 1:
            raise ValueError("Smart Recovery image dimensions must be positive")
        if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
            raise ValueError(
                "Smart Recovery image dimensions must not exceed "
                "{}x{}".format(MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)
            )
        image = np.ascontiguousarray(image)
        digest = hashlib.sha256()
        digest.update(b"archaeotrace-recovery-rgb/1\0")
        digest.update(struct.pack(">QQQ", int(height), int(width), int(channels)))
        digest.update(image.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(image).cast("B"))
        image_hash = digest.hexdigest()
        if self._encoding is not None and image_hash == self._image_sha256:
            return self._encoding
        encoding = self._engine.encode(image)
        encoded_size = getattr(encoding, "image_size", None)
        if encoded_size is not None:
            try:
                encoded_size = tuple(int(value) for value in encoded_size)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Smart Recovery encoding exposed an invalid image_size"
                ) from exc
            if encoded_size != (height, width):
                raise RuntimeError(
                    "Smart Recovery encoding changed the source image dimensions"
                )
        self._encoding = encoding
        self._image_sha256 = image_hash
        self._last_prediction = None
        return self._encoding

    def encode(self, rgb_u8):
        """Background-task alias retaining the verified image cache contract."""

        return self.set_image(rgb_u8)

    def predict(self, *args):
        """Return the full prediction using cached or explicit encoding.

        ``predict(points, labels)`` serves point-prompt callers. Background
        tasks may pass ``predict(encoding, points, labels)`` so an immutable
        encoding can be handed between task instances without touching QGIS.
        """

        if len(args) == 2:
            encoding = self._encoding
            points_xy, labels = args
        elif len(args) == 3:
            encoding, points_xy, labels = args
        else:
            raise TypeError("predict expects points/labels or encoding/points/labels")
        if encoding is None:
            raise RuntimeError("set_image() must be called before Smart Recovery")
        prediction = self._engine.predict(encoding, points_xy, labels)
        self._validate_prediction(prediction, encoding)
        self._last_prediction = prediction
        return prediction

    @staticmethod
    def _validate_prediction(prediction, encoding):
        """Fail closed if a provider changes the source-grid mask contract."""

        import numpy as np

        image_size = getattr(encoding, "image_size", None)
        if image_size is None:
            return
        try:
            expected_shape = tuple(int(value) for value in image_size)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Smart Recovery encoding exposed an invalid image_size"
            ) from exc
        if len(expected_shape) != 2 or min(expected_shape) < 1:
            raise RuntimeError("Smart Recovery encoding exposed an invalid image_size")

        mask = np.asarray(getattr(prediction, "mask", None))
        if mask.shape != expected_shape or mask.dtype != np.bool_:
            raise RuntimeError(
                "Smart Recovery prediction mask must be a source-sized bool array"
            )
        selected_logits = getattr(prediction, "selected_logits", None)
        if selected_logits is not None:
            logits = np.asarray(selected_logits)
            if logits.shape != expected_shape or not np.isfinite(logits).all():
                raise RuntimeError(
                    "Smart Recovery selected logits must be finite and source-sized"
                )

    def predict_point(self, points_xy, labels):
        """Compatibility adapter returning the selected boolean corridor mask."""

        return self.predict(points_xy, labels).mask

    def clear_image(self):
        self._encoding = None
        self._image_sha256 = None
        self._last_prediction = None


__all__ = ["EfficientSAMRecoveryEngine"]
