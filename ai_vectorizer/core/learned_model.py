"""Offline verified ONNX contour priors over source-anchored fixed tiles.

No inference or import operation opens the network.  Additional normalization
belongs inside the model graph; the adapter only converts Byte RGB to /255.
Anchoring each fixed tile makes overlapping views repeatable.  The 64-pixel
context is not a claim of equivalence to whole-image ResNet inference.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Callable, Optional, Tuple

import numpy as np

from .learned_prior import LearnedLinePrior


BUNDLE_SCHEMA = "archaeotrace-learned-contour-bundle/1"
MAX_MODEL_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
TILE_SIZE = 512
CORE_SIZE = 384
HALO_SIZE = 64
MAX_CROP_DIMENSION = 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

MODEL_CONTRACT = {
    "input_name": "rgb",
    "output_name": "logits",
    "input_shape": [1, 3, TILE_SIZE, TILE_SIZE],
    "output_shape": [1, 2, TILE_SIZE, TILE_SIZE],
    "input_dtype": "float32",
    "output_dtype": "float32",
    "preprocessing": "rgb_u8_div255",
    "normalization": "imagenet_inside_graph",
    "heads": ["contour_support", "other_ink_risk"],
    "core_size": CORE_SIZE,
    "halo": HALO_SIZE,
    "padding": "edge",
    "tile_alignment": "source_grid",
    "context_limit": "fixed_tiles_not_full_image_equivalent",
}


class LearnedModelError(RuntimeError):
    """Invalid artifacts, source grids or inference results."""


class LearnedModelUnavailable(LearnedModelError):
    """An optional local runtime dependency is missing."""


class LearnedModelCancelled(LearnedModelError):
    """A cancelled prediction must never become a current preview."""


@dataclass(frozen=True)
class BundleInspection:
    state: str
    detail: str
    path: Path
    model_sha256: Optional[str] = None
    bundle_sha256: Optional[str] = None
    manifest: Optional[dict] = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"


@dataclass(frozen=True)
class SourceTile:
    """One globally anchored core and its clipped true-raster read."""

    core_origin: Tuple[int, int]
    read_bounds: Tuple[int, int, int, int]
    padding: Tuple[int, int, int, int]  # top, bottom, left, right


def build_bundle_manifest(
    model_bytes: bytes,
    provenance: dict,
    *,
    model_id: str = "learned-contour-v1",
) -> dict:
    """Build the canonical manifest used by the separate training exporter."""

    if not isinstance(model_bytes, bytes) or not 0 < len(model_bytes) <= MAX_MODEL_BYTES:
        raise LearnedModelError("model must be non-empty bytes no larger than 64 MiB")
    document = {
        "schema_version": BUNDLE_SCHEMA,
        "model_id": model_id,
        "model": {
            "filename": "model.onnx",
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
            "size_bytes": len(model_bytes),
        },
        "contract": json.loads(json.dumps(MODEL_CONTRACT)),
        "provenance": provenance,
    }
    return _validate_manifest(document)


def model_guidance_parameters(manifest: Optional[dict]) -> Tuple[float, float]:
    """Read sealed path costs, or use contour support alone for a pilot.

    An exported second head is not evidence that its path penalty has been
    selected and evaluated. Only the manifest's validated, report-bound
    ``selected_guidance`` may enable learned avoidance. This does not affect
    the separate avoidance regions drawn by the user.
    """

    if manifest is None:
        return 0.8, 0.0
    validated = _validate_manifest(manifest)
    selected = validated["provenance"].get("selected_guidance")
    if selected is None:
        return 0.8, 0.0
    return float(selected["support"]), float(selected["avoidance"])


def inspect_bundle(bundle_dir) -> BundleInspection:
    """Inspect local immutable bytes only; never create or download files."""

    path = Path(bundle_dir).absolute()
    try:
        manifest, _model_bytes, digest = _read_bundle(path)
        _validate_onnx_bytes(_model_bytes)
        return BundleInspection(
            "ready", "Local model size and SHA-256 verified", path,
            manifest["model"]["sha256"], digest, manifest,
        )
    except FileNotFoundError:
        return BundleInspection("missing", "Local learned bundle is missing", path)
    except LearnedModelUnavailable as exc:
        return BundleInspection("runtime_missing", str(exc), path)
    except (OSError, LearnedModelError, ValueError) as exc:
        return BundleInspection("corrupt", str(exc), path)


def import_local_bundle(source_dir, cache_root) -> Path:
    """Validate and atomically publish a local bundle in an isolated cache.

    A bundle includes one self-contained ONNX graph and ``manifest.json``.
    Model and manifest symlinks, external tensors and custom operators are
    refused.  Existing content-addressed objects are verified and reused.
    """

    manifest, model_bytes, digest = _read_bundle(Path(source_dir).absolute())
    _validate_onnx_bytes(model_bytes)
    destination_parent = Path(cache_root).absolute() / "learned-contours"
    manifest_bytes = _canonical_json(manifest)
    with _safe_directory(destination_parent, create=True) as parent_fd:
        destination = destination_parent / digest
        def verify_destination():
            existing, _bytes, existing_digest = _read_bundle(destination)
            if existing_digest != digest or existing != manifest:
                raise LearnedModelError("existing cache bundle does not match import")

        if os.path.lexists(destination):
            verify_destination()
            return destination
        staging = Path(tempfile.mkdtemp(prefix=".import-", dir=destination_parent))
        try:
            for filename, data in (("model.onnx", model_bytes), ("manifest.json", manifest_bytes)):
                with (staging / filename).open("xb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            if os.path.lexists(destination):
                verify_destination()
            else:
                try:
                    _publish_directory_no_replace(staging, destination, parent_fd)
                except FileExistsError:
                    # Another successful importer can win the same object.
                    verify_destination()
            verify_destination()
            if parent_fd is not None:
                os.fsync(parent_fd)
            return destination
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def tile_plan(
    bounds: Tuple[int, int, int, int],
    raster_size: Tuple[int, int],
) -> Tuple[SourceTile, ...]:
    """Plan fixed source-grid tiles; size order is ``(width, height)``."""

    width, height = _integer_tuple(raster_size, 2, "raster_size")
    if width < 1 or height < 1:
        raise LearnedModelError("raster_size dimensions must be positive")
    x0, y0, x1, y1 = _integer_tuple(bounds, 4, "bounds")
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise LearnedModelError("bounds must be a non-empty rectangle inside the raster")
    if max(x1 - x0, y1 - y0) > MAX_CROP_DIMENSION:
        raise LearnedModelError("learned crop dimensions must not exceed 1024 pixels")
    tiles = []
    for core_y in range(y0 // CORE_SIZE * CORE_SIZE, y1, CORE_SIZE):
        for core_x in range(x0 // CORE_SIZE * CORE_SIZE, x1, CORE_SIZE):
            left, top = core_x - HALO_SIZE, core_y - HALO_SIZE
            right, bottom = left + TILE_SIZE, top + TILE_SIZE
            read_bounds = (max(0, left), max(0, top), min(width, right), min(height, bottom))
            padding = (max(0, -top), max(0, bottom - height), max(0, -left), max(0, right - width))
            tiles.append(SourceTile((core_x, core_y), read_bounds, padding))
    return tuple(tiles)


plan_source_tiles = tile_plan


class LearnedContourEngine:
    """A verified CPU-only runtime; call predictions from a background task."""

    provider_id = "learned-contour-onnx-v1"

    def __init__(self, bundle_dir):
        manifest, model_bytes, digest = _read_bundle(Path(bundle_dir).absolute())
        _validate_onnx_bytes(model_bytes)
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise LearnedModelUnavailable(f"Local learned inference needs onnxruntime: {exc}") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise LearnedModelError(f"Unable to initialize learned ONNX graph: {exc}") from exc
        self._validate_session(session)
        self._session = session
        self.manifest = manifest
        # Resolve the policy once at preparation, not on mouse movement.
        self.guidance_parameters = model_guidance_parameters(manifest)
        self.has_selected_guidance = manifest["provenance"].get("selected_guidance") is not None
        self.bundle_sha256 = digest
        self.model_sha256 = manifest["model"]["sha256"]
        self.is_ready = True
        self.metadata = {
            "provider_id": self.provider_id,
            "model_id": manifest["model_id"],
            "model_sha256": self.model_sha256,
            "bundle_sha256": digest,
            "provider": "CPUExecutionProvider",
            "onnxruntime_version": ort.__version__,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "tile_size": TILE_SIZE,
            "core_size": CORE_SIZE,
            "halo": HALO_SIZE,
            "context_limit": MODEL_CONTRACT["context_limit"],
        }

    @classmethod
    def from_bundle(cls, bundle_dir):
        return cls(bundle_dir)

    @staticmethod
    def inspect(bundle_dir):
        return inspect_bundle(bundle_dir)

    @staticmethod
    def _validate_session(session) -> None:
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise LearnedModelError("learned inference must use only CPUExecutionProvider")
        options = session.get_session_options()
        if options.intra_op_num_threads != 1 or options.inter_op_num_threads != 1:
            raise LearnedModelError("learned inference must use one CPU thread")
        if getattr(options.execution_mode, "name", str(options.execution_mode)) != "ORT_SEQUENTIAL":
            raise LearnedModelError("learned inference must use sequential execution")
        for label, values, name, shape in (
            ("input", session.get_inputs(), "rgb", [1, 3, TILE_SIZE, TILE_SIZE]),
            ("output", session.get_outputs(), "logits", [1, 2, TILE_SIZE, TILE_SIZE]),
        ):
            if len(values) != 1 or values[0].name != name or values[0].type != "tensor(float)" or values[0].shape != shape:
                raise LearnedModelError(f"learned ONNX {label} violates the fixed FP32 contract")

    def predict_tile(self, image: np.ndarray) -> LearnedLinePrior:
        """Run one full 512px Byte RGB tile; normalization is inside ONNX."""

        values = np.asarray(image)
        if values.shape != (TILE_SIZE, TILE_SIZE, 3) or values.dtype != np.uint8:
            raise LearnedModelError("learned tile requires 512x512 native Byte RGB")
        tensor = np.ascontiguousarray(values.transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= np.float32(255.0)
        try:
            outputs = self._session.run(["logits"], {"rgb": tensor})
        except Exception as exc:
            raise LearnedModelError(f"Learned ONNX inference failed: {exc}") from exc
        if len(outputs) != 1:
            raise LearnedModelError("learned graph returned an unexpected output count")
        logits = np.asarray(outputs[0])
        if logits.shape != (1, 2, TILE_SIZE, TILE_SIZE) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise LearnedModelError("learned logits must be finite FP32 1x2x512x512")
        scores = np.empty_like(logits)
        positive = logits >= 0.0
        scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        negative_exp = np.exp(logits[~positive])
        scores[~positive] = negative_exp / (1.0 + negative_exp)
        return LearnedLinePrior(scores[0, 0], scores[0, 1])

    def predict_crop(
        self,
        read_tile: Callable,
        bounds: Tuple[int, int, int, int],
        raster_size: Tuple[int, int],
        *,
        cancel_check: Optional[Callable] = None,
    ) -> LearnedLinePrior:
        """Read pure array snapshots and stitch valid cores in source pixels.

        ``read_tile((x0,y0,x1,y1))`` returns clipped native Byte RGB.  The
        callback must not access QGIS objects from a worker.  Padding occurs
        only at declared raster edges, never at an arbitrary viewport edge.
        """

        if not callable(read_tile) or (cancel_check is not None and not callable(cancel_check)):
            raise TypeError("read_tile and optional cancel_check must be callable")
        plan = tile_plan(bounds, raster_size)
        x0, y0, x1, y1 = (int(value) for value in bounds)
        contour = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        avoidance = np.zeros_like(contour)
        for tile in plan:
            _cancel(cancel_check)
            left, top, right, bottom = tile.read_bounds
            values = np.asarray(read_tile(tile.read_bounds))
            if values.shape != (bottom - top, right - left, 3) or values.dtype != np.uint8:
                raise LearnedModelError("source tile read must preserve native Byte RGB dimensions")
            pad_top, pad_bottom, pad_left, pad_right = tile.padding
            values = np.pad(values, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
            _cancel(cancel_check)
            prediction = self.predict_tile(values)
            _cancel(cancel_check)
            core_x, core_y = tile.core_origin
            target_left, target_top = max(x0, core_x), max(y0, core_y)
            target_right, target_bottom = min(x1, core_x + CORE_SIZE), min(y1, core_y + CORE_SIZE)
            target = np.s_[target_top - y0 : target_bottom - y0, target_left - x0 : target_right - x0]
            source = np.s_[HALO_SIZE + target_top - core_y : HALO_SIZE + target_bottom - core_y, HALO_SIZE + target_left - core_x : HALO_SIZE + target_right - core_x]
            contour[target] = prediction.contour_score[source]
            avoidance[target] = prediction.avoidance_score[source]
        return LearnedLinePrior(contour, avoidance)


def _cancel(cancel_check) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise LearnedModelCancelled("learned inference was cancelled")


def _integer_tuple(value, length, label):
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != length or any(isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) for item in value):
        raise LearnedModelError(f"{label} must contain {length} integer coordinates")
    return tuple(int(item) for item in value)


def _canonical_json(value) -> bytes:
    try:
        result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise LearnedModelError("manifest must contain finite JSON values") from exc
    if len(result) > MAX_MANIFEST_BYTES:
        raise LearnedModelError("manifest exceeds one MiB")
    return result


def _validate_manifest(document) -> dict:
    if not isinstance(document, dict) or set(document) != {"schema_version", "model_id", "model", "contract", "provenance"}:
        raise LearnedModelError("learned manifest fields are invalid")
    if document["schema_version"] != BUNDLE_SCHEMA or not isinstance(document["model_id"], str) or not _MODEL_ID.fullmatch(document["model_id"]):
        raise LearnedModelError("learned manifest schema or model id is invalid")
    model = document["model"]
    if not isinstance(model, dict) or set(model) != {"filename", "sha256", "size_bytes"} or model["filename"] != "model.onnx":
        raise LearnedModelError("manifest must identify exactly model.onnx")
    if not isinstance(model["sha256"], str) or not _SHA256.fullmatch(model["sha256"]):
        raise LearnedModelError("model SHA-256 must contain 64 lowercase hex digits")
    if isinstance(model["size_bytes"], bool) or not isinstance(model["size_bytes"], int) or not 0 < model["size_bytes"] <= MAX_MODEL_BYTES:
        raise LearnedModelError("model size must be between one byte and 64 MiB")
    if _canonical_json(document["contract"]) != _canonical_json(MODEL_CONTRACT):
        raise LearnedModelError("manifest violates the fixed learned tile contract")
    provenance = document["provenance"]
    required = {"dataset_manifest_sha256", "training_code_commit", "export_code_commit", "architecture", "seed", "license_spdx"}
    if not isinstance(provenance, dict) or not required <= set(provenance):
        raise LearnedModelError("training and export provenance fields are missing")
    for key, pattern in (("dataset_manifest_sha256", _SHA256), ("training_code_commit", _COMMIT), ("export_code_commit", _COMMIT)):
        if not isinstance(provenance[key], str) or not pattern.fullmatch(provenance[key]):
            raise LearnedModelError(f"provenance.{key} is invalid")
    for key in ("architecture", "license_spdx"):
        if not isinstance(provenance[key], str) or not provenance[key].strip() or len(provenance[key]) > 256:
            raise LearnedModelError(f"provenance.{key} is invalid")
    if isinstance(provenance["seed"], bool) or not isinstance(provenance["seed"], int) or not 0 <= provenance["seed"] < 2**63:
        raise LearnedModelError("provenance.seed must be a non-negative integer")
    selected = provenance.get("selected_guidance")
    if selected is not None:
        required_guidance = {"support", "avoidance", "calibration_sha256", "holdout_report_sha256", "pilot_gates_passed"}
        if not isinstance(selected, dict) or set(selected) not in (required_guidance, required_guidance | {"source_bundle_sha256"}):
            raise LearnedModelError("selected guidance provenance fields are invalid")
        if type(selected["support"]) not in (int, float) or selected["support"] not in (0.4, 0.8) or type(selected["avoidance"]) not in (int, float) or selected["avoidance"] not in (2, 4, 6) or selected["pilot_gates_passed"] is not True:
            raise LearnedModelError("selected guidance must identify an approved grid configuration")
        for key in ("calibration_sha256", "holdout_report_sha256", "source_bundle_sha256"):
            if key in selected and (not isinstance(selected[key], str) or not _SHA256.fullmatch(selected[key])):
                raise LearnedModelError("selected guidance report hashes are invalid")
    return json.loads(_canonical_json(document))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise LearnedModelError(f"manifest contains duplicate key: {key}")
        result[key] = value
    return result


def _read_bundle(path):
    manifest_bytes = _read_regular(path, "manifest.json", MAX_MANIFEST_BYTES)
    try:
        document = json.loads(manifest_bytes, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise LearnedModelError("manifest is not valid JSON") from exc
    manifest = _validate_manifest(document)
    model_bytes = _read_regular(path, "model.onnx", MAX_MODEL_BYTES)
    if len(model_bytes) != manifest["model"]["size_bytes"] or hashlib.sha256(model_bytes).hexdigest() != manifest["model"]["sha256"]:
        raise LearnedModelError("model size or SHA-256 does not match manifest")
    return manifest, model_bytes, hashlib.sha256(_canonical_json(manifest)).hexdigest()


def _publish_directory_no_replace(source, destination, parent_fd) -> None:
    """Atomically publish a complete generation without replacing any entry."""

    if os.name == "nt":
        # Windows rename refuses an existing destination, including empty
        # directories. POSIX rename would replace an empty one.
        os.rename(source, destination)
        return
    if parent_fd is None:
        raise LearnedModelError("atomic bundle import requires pinned directories")
    library = ctypes.CDLL(None, use_errno=True)
    symbol, flags = ("renameatx_np", 0x4) if sys.platform == "darwin" else ("renameat2", 1)
    operation = getattr(library, symbol, None)
    if operation is None:
        raise LearnedModelError("atomic no-replace directory publication is unavailable")
    operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    result = operation(parent_fd, os.fsencode(source.name), parent_fd, os.fsencode(destination.name), flags)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


@contextmanager
def _safe_directory(path, *, create=False):
    """Walk and pin directories without following symlink components."""

    path = Path(path).absolute()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
        descriptor = os.open(path.anchor, flags)
        try:
            for part in path.parts[1:]:
                if part in (".", ".."):
                    raise LearnedModelError("bundle paths cannot contain traversal")
                if create:
                    try:
                        os.mkdir(part, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            yield descriptor
        finally:
            os.close(descriptor)
    else:  # Windows lacks openat/O_NOFOLLOW; check each component explicitly.
        current = Path(path.anchor)
        for part in path.parts[1:]:
            if part in (".", ".."):
                raise LearnedModelError("bundle paths cannot contain traversal")
            current /= part
            if create:
                current.mkdir(exist_ok=True)
            info = current.lstat()
            if not stat.S_ISDIR(info.st_mode) or current.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                raise LearnedModelError("bundle directories cannot be symlinks or reparse points")
        yield None


def _read_regular(path, filename, limit):
    with _safe_directory(path) as directory_fd:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        target = Path(path) / filename
        if directory_fd is None:
            info = target.lstat()
            if target.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
                raise LearnedModelError("bundle files cannot be symlinks or reparse points")
            descriptor = os.open(target, flags)
        else:
            descriptor = os.open(filename, flags, dir_fd=directory_fd)
        with os.fdopen(descriptor, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
                raise LearnedModelError("bundle file is not a bounded regular file")
            data = source.read(limit + 1)
            after = os.fstat(source.fileno())
            if len(data) != before.st_size or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise LearnedModelError("bundle file changed during verification")
            return data


def _validate_onnx_bytes(model_bytes) -> None:
    try:
        import onnx
    except Exception as exc:
        raise LearnedModelUnavailable(f"Local learned validation needs onnx: {exc}") from exc
    try:
        model = onnx.load_model_from_string(model_bytes)
        if model.functions or any(opset.domain not in ("", "ai.onnx") for opset in model.opset_import):
            raise LearnedModelError("learned ONNX cannot contain custom operators or functions")
        for label, values, name, shape in (
            ("input", model.graph.input, "rgb", [1, 3, TILE_SIZE, TILE_SIZE]),
            ("output", model.graph.output, "logits", [1, 2, TILE_SIZE, TILE_SIZE]),
        ):
            if len(values) != 1:
                raise LearnedModelError(f"learned ONNX requires exactly one {label}")
            value = values[0]
            tensor_type = value.type.tensor_type
            dimensions = [dimension.dim_value for dimension in tensor_type.shape.dim]
            if value.name != name or tensor_type.elem_type != onnx.TensorProto.FLOAT or dimensions != shape:
                raise LearnedModelError(f"learned ONNX {label} violates the fixed FP32 contract")

        def tensor_check(tensor):
            if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
                raise LearnedModelError("learned ONNX must not reference external tensor data")

        def graph_check(graph):
            for tensor in graph.initializer:
                tensor_check(tensor)
            for sparse_tensor in graph.sparse_initializer:
                tensor_check(sparse_tensor.values)
                tensor_check(sparse_tensor.indices)
            for node in graph.node:
                if node.domain not in ("", "ai.onnx"):
                    raise LearnedModelError("learned ONNX cannot contain custom operators")
                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.TENSOR:
                        tensor_check(attribute.t)
                    elif attribute.type == onnx.AttributeProto.TENSORS:
                        for tensor in attribute.tensors:
                            tensor_check(tensor)
                    elif attribute.type == onnx.AttributeProto.GRAPH:
                        graph_check(attribute.g)
                    elif attribute.type == onnx.AttributeProto.GRAPHS:
                        for child in attribute.graphs:
                            graph_check(child)
        graph_check(model.graph)
        onnx.checker.check_model(model)
    except LearnedModelError:
        raise
    except Exception as exc:
        raise LearnedModelError(f"invalid self-contained ONNX graph: {exc}") from exc


__all__ = [
    "LearnedContourEngine", "LearnedModelError", "LearnedModelUnavailable",
    "LearnedModelCancelled", "LearnedLinePrior", "BundleInspection", "SourceTile",
    "build_bundle_manifest", "inspect_bundle", "import_local_bundle", "tile_plan", "plan_source_tiles",
    "MODEL_CONTRACT", "BUNDLE_SCHEMA", "CORE_SIZE", "HALO_SIZE", "TILE_SIZE",
]
