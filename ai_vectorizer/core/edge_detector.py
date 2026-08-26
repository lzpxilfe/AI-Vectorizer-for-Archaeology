# -*- coding: utf-8 -*-
"""
Line Evidence Module - Multiple detection methods for historical maps
Supports: Ink Centerline, Canny+Adaptive, LSD, and HED
"""

import hashlib
import hmac
import os
import shutil
import stat
import tempfile
import urllib.parse
import urllib.error
import urllib.request
from contextlib import contextmanager

import numpy as np

from ..config import PLUGIN_NAME
from .dependencies import build_missing_cv2_message, get_cv2, is_cv2_available, require_cv2

try:
    from scipy import ndimage as _scipy_ndimage
except Exception:
    _scipy_ndimage = None

try:
    from skimage.filters import threshold_otsu as _skimage_threshold_otsu
except Exception:
    _skimage_threshold_otsu = None

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
except Exception:
    _skimage_skeletonize = None


class _HEDNoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Make redirect responses fail closed before contacting a second URL."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


@contextmanager
def _quietly_closing_hed_response(response):
    """Release a response without masking a completed verified download."""

    try:
        yield response
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            # Cleanup must not mask a verified download result.
            except Exception:  # nosec B110
                pass


class EdgeDetector:
    METHOD_INK = 'ink'
    METHOD_CANNY = 'canny'
    METHOD_LSD = 'lsd'
    METHOD_HED = 'hed'

    EDGE_MAX_VALUE = 255
    EDGE_PRESENCE_THRESHOLD = 50

    DEFAULT_CANNY_LOW_THRESHOLD = 30
    DEFAULT_CANNY_HIGH_THRESHOLD = 100

    # A black top-hat isolates narrow dark ink from slowly varying paper and
    # scan shading. The binary response is then reduced to one centerline,
    # avoiding the paired outlines produced by gradient edge detectors.
    # 15px still isolates ordinary 1-3px map strokes while preserving a
    # centerline when QGIS zoom/resampling expands a stroke to roughly 11px.
    INK_BACKGROUND_SIZE = 15
    INK_RESPONSE_PERCENTILE = 99.0
    INK_MIN_NORMALIZED_RESPONSE = 0.04
    INK_MIN_COMPONENT_SIZE = 5

    CANNY_ADAPTIVE_BLOCK_SIZE = 21
    CANNY_ADAPTIVE_C = 10
    CANNY_BLUR_KERNEL = (3, 3)
    CANNY_CLOSE_KERNEL = (2, 2)

    LSD_LINE_WIDTH = 2
    LSD_ADAPTIVE_BLOCK_SIZE = 21
    LSD_ADAPTIVE_C = 8
    LSD_CLOSE_KERNEL = (3, 3)

    HED_MEAN = (104.00698793, 116.66876762, 122.67891434)
    HED_OUTPUT_LAYER = "sigmoid-fuse"
    HED_BINARY_THRESHOLD = 50
    HED_CLOSE_KERNEL = (2, 2)

    # At edge_weight=0 the edge distance must have no influence on an
    # explicitly requested Auto Path proposal.  The proposal itself remains
    # opt-in and can still choose a shortest grid path; this constant only
    # controls edge attraction within that path search.
    EDGE_COST_BASE_MULTIPLIER = 0.0
    EDGE_COST_WEIGHT_SCALE = 0.9
    DIST_TRANSFORM_MASK_SIZE = 5

    HED_PACKAGE_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    HED_MODEL_DIR = HED_PACKAGE_MODEL_DIR
    HED_PROTOTXT = os.path.join(HED_MODEL_DIR, 'hed_deploy.prototxt')
    HED_CAFFEMODEL = os.path.join(HED_MODEL_DIR, 'hed_pretrained_bsds.caffemodel')
    HED_PROTOTXT_URL = (
        "https://raw.githubusercontent.com/s9xie/hed/"
        "912632b986acc6dd6cc33b95603b2f279d7bd9f2/"
        "examples/hed/deploy.prototxt"
    )
    HED_CAFFEMODEL_URL = 'https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'
    HED_PROTOTXT_SIZE_BYTES = 8_186
    HED_PROTOTXT_SHA256 = "378a9246383da889cf8e0290c47554d75dcf9c5b6bbabd8ab6c481c34aa12b8a"
    HED_CAFFEMODEL_SIZE_BYTES = 58_876_104
    HED_CAFFEMODEL_SHA256 = "4b6937684bce9be1ef5163c78ec812dff9a23653bfbb451925210a64ecfaaac7"
    HED_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    HED_REQUEST_HEADERS = {
        "User-Agent": PLUGIN_NAME,
        "Accept-Encoding": "identity",
    }
    HED_ALLOWED_DOWNLOAD_SCHEMES = {"https"}
    HED_ALLOWED_DOWNLOAD_HOSTS = {
        "raw.githubusercontent.com",
        "vcl.ucsd.edu",
    }
    HED_MODEL_SIZE_MB = 56
    HED_VALIDATION_IMAGE_SIZE = 64

    _hed_runtime_status_cache = None
    _hed_runtime_status_signature = None
    _hed_crop_layer_registered = False

    class _HEDCropLayer:
        """OpenCV DNN compatibility layer for Caffe Crop nodes used by HED."""

        def __init__(self, _params, _blobs):
            self.x_start = 0
            self.x_end = 0
            self.y_start = 0
            self.y_end = 0

        def getMemoryShapes(self, inputs):
            input_shape, target_shape = inputs[0], inputs[1]
            target_h = int(target_shape[2])
            target_w = int(target_shape[3])
            self.y_start = max(0, int((input_shape[2] - target_h) / 2))
            self.x_start = max(0, int((input_shape[3] - target_w) / 2))
            self.y_end = self.y_start + target_h
            self.x_end = self.x_start + target_w
            return [[input_shape[0], input_shape[1], target_h, target_w]]

        def forward(self, inputs):
            return [inputs[0][:, :, self.y_start:self.y_end, self.x_start:self.x_end]]

    @staticmethod
    def thin_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
        """Return a thin centerline mask with graceful runtime fallbacks."""
        binary = np.asarray(binary_mask).astype(bool)
        if not binary.any():
            return binary

        if _skimage_skeletonize is not None:
            return _skimage_skeletonize(binary)

        cv2 = get_cv2()
        if cv2 is not None:
            ximgproc = getattr(cv2, "ximgproc", None)
            if ximgproc is not None and hasattr(ximgproc, "thinning"):
                thinned = ximgproc.thinning(binary.astype(np.uint8) * 255)
                return thinned > 0

        return EdgeDetector._thin_binary_mask_numpy(binary)

    @staticmethod
    def _thin_binary_mask_numpy(binary_mask: np.ndarray) -> np.ndarray:
        """Skeletonize a mask with the Zhang-Suen algorithm."""
        skeleton = np.asarray(binary_mask, dtype=bool).copy()
        if not skeleton.any():
            return skeleton

        def neighborhood(values):
            padded = np.pad(values, 1, mode="constant", constant_values=False)
            return (
                padded[:-2, 1:-1],
                padded[:-2, 2:],
                padded[1:-1, 2:],
                padded[2:, 2:],
                padded[2:, 1:-1],
                padded[2:, :-2],
                padded[1:-1, :-2],
                padded[:-2, :-2],
            )

        while True:
            changed = False
            neighbors = neighborhood(skeleton)
            neighbor_count = sum(neighbors)
            transitions = sum(
                (~current & following).astype(np.uint8)
                for current, following in zip(neighbors, neighbors[1:] + neighbors[:1])
            )
            p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbors
            remove = (
                skeleton
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & ~(p2 & p4 & p6)
                & ~(p4 & p6 & p8)
            )
            if remove.any():
                skeleton[remove] = False
                changed = True

            neighbors = neighborhood(skeleton)
            neighbor_count = sum(neighbors)
            transitions = sum(
                (~current & following).astype(np.uint8)
                for current, following in zip(neighbors, neighbors[1:] + neighbors[:1])
            )
            p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbors
            remove = (
                skeleton
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & ~(p2 & p4 & p8)
                & ~(p2 & p6 & p8)
            )
            if remove.any():
                skeleton[remove] = False
                changed = True

            if not changed:
                return skeleton

    @classmethod
    def get_ink_runtime_status(cls):
        """Report the currently active, always-operational Ink backends."""
        background_backend = "scipy" if _scipy_ndimage is not None else "numpy"
        if _skimage_skeletonize is not None:
            thinning_backend = "scikit-image"
        else:
            cv2 = get_cv2()
            ximgproc = getattr(cv2, "ximgproc", None) if cv2 is not None else None
            thinning_backend = (
                "opencv-ximgproc"
                if ximgproc is not None and hasattr(ximgproc, "thinning")
                else "numpy"
            )

        optimized = background_backend == "scipy" and thinning_backend != "numpy"
        return {
            "ok": True,
            "reason": "ready" if optimized else "numpy_fallback",
            "message": (
                "Ink Centerline is ready "
                f"(background: {background_backend}, thinning: {thinning_backend})."
            ),
            "background_backend": background_backend,
            "thinning_backend": thinning_backend,
        }

    @staticmethod
    def _prepare_input_images(image: np.ndarray):
        """Normalize raster input into gray + BGR variants for detectors."""
        cv2 = get_cv2()
        if len(image.shape) == 3:
            rgb = np.ascontiguousarray(image[..., :3])
            if cv2 is not None:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                rgb_float = rgb.astype(np.float32, copy=False)
                gray = np.tensordot(
                    rgb_float,
                    np.array([0.299, 0.587, 0.114], dtype=np.float32),
                    axes=([-1], [0]),
                )
                gray = np.clip(gray, 0, 255).astype(np.uint8)
                color_bgr = np.ascontiguousarray(rgb[..., ::-1])
        else:
            gray = np.asarray(image)
            if gray.dtype != np.uint8:
                gray_float = gray.astype(np.float32, copy=False)
                finite = np.isfinite(gray_float)
                if not finite.any():
                    gray = np.zeros(gray_float.shape, dtype=np.uint8)
                else:
                    low = float(np.nanmin(gray_float[finite]))
                    high = float(np.nanmax(gray_float[finite]))
                    if high <= low:
                        gray = np.zeros(gray_float.shape, dtype=np.uint8)
                    else:
                        gray = np.clip(
                            (gray_float - low) * (255.0 / (high - low)),
                            0,
                            255,
                        ).astype(np.uint8)
            gray = np.ascontiguousarray(gray)
            if cv2 is not None:
                color_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                color_bgr = np.repeat(gray[..., None], 3, axis=2)

        return gray, color_bgr

    @classmethod
    def _missing_opencv_status(cls, feature_name):
        return {
            "ok": False,
            "reason": "missing_opencv",
            "message": build_missing_cv2_message(feature_name),
        }

    @classmethod
    def _require_cv2_runtime(cls, feature_name):
        return require_cv2(feature_name)

    def __init__(self, method=METHOD_INK):
        """
        Args:
            method: 'ink', 'canny', 'lsd', or 'hed'
        """
        self.method = method
        self.hed_net = None
        self.cv2 = None
        self.lsd = None

        # Ink Centerline never needs OpenCV. Legacy Canny also remains usable
        # in a clean QGIS Python environment through its NumPy fallback.
        if method == self.METHOD_CANNY:
            self.cv2 = get_cv2()
        elif method in (self.METHOD_LSD, self.METHOD_HED):
            self.cv2 = self._require_cv2_runtime(f"{method.upper()} edge detection")

        # LSD detector instance
        if method == self.METHOD_LSD:
            self.lsd = self.cv2.createLineSegmentDetector(self.cv2.LSD_REFINE_STD)

        # HED network
        if method == self.METHOD_HED:
            self._init_hed()

    @classmethod
    def _hed_asset_specs(cls):
        return (
            {
                "name": "HED prototxt",
                "url": cls.HED_PROTOTXT_URL,
                "path": cls.HED_PROTOTXT,
                "size": cls.HED_PROTOTXT_SIZE_BYTES,
                "sha256": cls.HED_PROTOTXT_SHA256,
                "temp_prefix": "hed_prototxt_",
                "temp_suffix": ".prototxt",
            },
            {
                "name": "HED caffemodel",
                "url": cls.HED_CAFFEMODEL_URL,
                "path": cls.HED_CAFFEMODEL,
                "size": cls.HED_CAFFEMODEL_SIZE_BYTES,
                "sha256": cls.HED_CAFFEMODEL_SHA256,
                "temp_prefix": "hed_weights_",
                "temp_suffix": ".caffemodel",
            },
        )

    @classmethod
    def _verify_hed_asset_file(cls, path, spec):
        """Verify one regular HED asset without following symlinks."""
        expected_size = spec.get("size")
        expected_sha256 = spec.get("sha256")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            return False, f"{spec['name']} is missing an exact expected byte size."
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            return False, f"{spec['name']} is missing a valid expected SHA-256 digest."

        try:
            cls._require_hed_model_directory(os.path.dirname(os.path.abspath(path)))
            initial = os.lstat(path)
        except FileNotFoundError:
            return False, f"Missing {spec['name']}: {path}"
        except (OSError, RuntimeError) as exc:
            return False, f"Cannot inspect {spec['name']}: {exc}"
        if not stat.S_ISREG(initial.st_mode):
            return False, f"{spec['name']} must be a regular file; symlinks are rejected."
        if initial.st_size != expected_size:
            return False, (
                f"{spec['name']} size mismatch: expected {expected_size} bytes, "
                f"got {initial.st_size} bytes."
            )

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = None
        handle = None
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != initial.st_dev
                or opened.st_ino != initial.st_ino
            ):
                return False, f"{spec['name']} changed while verification started."
            if opened.st_size != expected_size:
                return False, (
                    f"{spec['name']} size mismatch: expected {expected_size} bytes, "
                    f"got {opened.st_size} bytes."
                )

            handle = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = None
            digest = hashlib.sha256()
            total = 0
            while total <= expected_size:
                chunk = handle.read(
                    min(cls.HED_DOWNLOAD_CHUNK_SIZE, expected_size - total + 1)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_size:
                    return False, f"{spec['name']} exceeds its expected byte size."
                digest.update(chunk)

            finished = os.fstat(handle.fileno())
            if (
                finished.st_size != opened.st_size
                or getattr(finished, "st_mtime_ns", None)
                != getattr(opened, "st_mtime_ns", None)
                or getattr(finished, "st_ctime_ns", None)
                != getattr(opened, "st_ctime_ns", None)
            ):
                return False, f"{spec['name']} changed during integrity verification."
            if total != expected_size:
                return False, (
                    f"{spec['name']} size mismatch: expected {expected_size} bytes, "
                    f"got {total} bytes."
                )
            if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
                return False, (
                    f"{spec['name']} SHA-256 mismatch; the asset is corrupt or untrusted."
                )
            return True, None
        except OSError as exc:
            return False, f"Cannot read {spec['name']} safely: {exc}"
        finally:
            if handle is not None:
                handle.close()
            elif descriptor is not None:
                os.close(descriptor)

    @classmethod
    def _verify_hed_asset_pair(cls, prototxt_path=None, caffemodel_path=None):
        paths = (
            prototxt_path or cls.HED_PROTOTXT,
            caffemodel_path or cls.HED_CAFFEMODEL,
        )
        for path, spec in zip(paths, cls._hed_asset_specs()):
            verified, error = cls._verify_hed_asset_file(path, spec)
            if not verified:
                return False, error
        return True, None

    @classmethod
    def _read_verified_hed_asset_bytes(cls, path, spec):
        """Read and verify one asset through one descriptor for in-memory loading."""
        try:
            cls._require_hed_model_directory(os.path.dirname(os.path.abspath(path)))
            initial = os.lstat(path)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"Cannot inspect {spec['name']}: {exc}") from exc
        if not stat.S_ISREG(initial.st_mode):
            raise RuntimeError(
                f"{spec['name']} must be a regular file; symlinks are rejected."
            )

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = None
        handle = None
        try:
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != initial.st_dev
                or opened.st_ino != initial.st_ino
            ):
                raise RuntimeError(
                    f"{spec['name']} changed while secure loading started."
                )
            if opened.st_size != spec["size"]:
                raise RuntimeError(
                    f"{spec['name']} size mismatch: expected {spec['size']} bytes, "
                    f"got {opened.st_size} bytes."
                )

            handle = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = None
            payload = bytearray()
            digest = hashlib.sha256()
            while len(payload) <= spec["size"]:
                chunk = handle.read(
                    min(
                        cls.HED_DOWNLOAD_CHUNK_SIZE,
                        spec["size"] - len(payload) + 1,
                    )
                )
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > spec["size"]:
                    raise RuntimeError(
                        f"{spec['name']} exceeds its expected byte size."
                    )
                digest.update(chunk)

            finished = os.fstat(handle.fileno())
            if (
                finished.st_size != opened.st_size
                or getattr(finished, "st_mtime_ns", None)
                != getattr(opened, "st_mtime_ns", None)
                or getattr(finished, "st_ctime_ns", None)
                != getattr(opened, "st_ctime_ns", None)
            ):
                raise RuntimeError(
                    f"{spec['name']} changed during secure loading."
                )
            if len(payload) != spec["size"]:
                raise RuntimeError(
                    f"{spec['name']} size mismatch: expected {spec['size']} bytes, "
                    f"got {len(payload)} bytes."
                )
            if not hmac.compare_digest(digest.hexdigest(), spec["sha256"]):
                raise RuntimeError(
                    f"{spec['name']} SHA-256 mismatch; the asset is corrupt or untrusted."
                )
            return payload
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read {spec['name']} safely: {exc}"
            ) from exc
        finally:
            if handle is not None:
                handle.close()
            elif descriptor is not None:
                os.close(descriptor)

    @classmethod
    def _hed_file_signature(cls):
        signature = []
        try:
            for spec in cls._hed_asset_specs():
                file_stat = os.lstat(spec["path"])
                signature.extend(
                    (
                        file_stat.st_dev,
                        file_stat.st_ino,
                        file_stat.st_mode,
                        file_stat.st_size,
                        getattr(file_stat, "st_mtime_ns", None),
                        getattr(file_stat, "st_ctime_ns", None),
                    )
                )
        except OSError:
            return None
        return tuple(signature)

    @classmethod
    def _invalidate_hed_status_cache(cls):
        cls._hed_runtime_status_cache = None
        cls._hed_runtime_status_signature = None

    @classmethod
    def configure_hed_storage(cls, models_dir, legacy_models_dir=None):
        """Use persistent profile storage and migrate trusted plugin-local assets."""
        target_dir = os.path.abspath(os.fspath(models_dir))
        legacy_dir = os.path.abspath(
            os.fspath(legacy_models_dir or cls.HED_PACKAGE_MODEL_DIR)
        )
        cls.HED_MODEL_DIR = target_dir
        cls.HED_PROTOTXT = os.path.join(target_dir, "hed_deploy.prototxt")
        cls.HED_CAFFEMODEL = os.path.join(
            target_dir,
            "hed_pretrained_bsds.caffemodel",
        )
        cls._invalidate_hed_status_cache()
        cls._migrate_legacy_hed_assets(legacy_dir)

    @staticmethod
    def _require_hed_model_directory(directory):
        try:
            information = os.lstat(directory)
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect the HED model directory: {exc}") from exc
        if not stat.S_ISDIR(information.st_mode):
            raise RuntimeError(
                "The HED model directory must be a real directory; symlinks are rejected."
            )
        return information

    @classmethod
    def _ensure_hed_model_directory(cls, directory):
        os.makedirs(directory, mode=0o700, exist_ok=True)
        return cls._require_hed_model_directory(directory)

    @classmethod
    def _migrate_legacy_hed_assets(cls, legacy_models_dir):
        """Copy only exact pinned legacy assets into persistent storage once."""
        migrated = []
        for spec in cls._hed_asset_specs():
            destination = spec["path"]
            source = os.path.join(legacy_models_dir, os.path.basename(destination))
            if (
                os.path.abspath(source) == os.path.abspath(destination)
                or os.path.lexists(destination)
                or not os.path.lexists(source)
            ):
                continue

            temp_path = None
            try:
                payload = cls._read_verified_hed_asset_bytes(source, spec)
                cls._ensure_hed_model_directory(cls.HED_MODEL_DIR)
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=f".{os.path.basename(destination)}.",
                    suffix=".migration",
                    dir=cls.HED_MODEL_DIR,
                )
                with os.fdopen(descriptor, "wb") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                verified, verification_error = cls._verify_hed_asset_file(
                    temp_path,
                    spec,
                )
                if not verified:
                    raise RuntimeError(
                        f"Migrated {spec['name']} failed verification: "
                        f"{verification_error}"
                    )
                os.replace(temp_path, destination)
                temp_path = None
                verified, verification_error = cls._verify_hed_asset_file(
                    destination,
                    spec,
                )
                if not verified:
                    try:
                        os.remove(destination)
                    except OSError as cleanup_error:
                        raise RuntimeError(
                            f"Published migrated {spec['name']} failed verification: "
                            f"{verification_error}; could not remove it: {cleanup_error}"
                        ) from cleanup_error
                    raise RuntimeError(
                        f"Published migrated {spec['name']} failed verification: "
                        f"{verification_error}"
                    )
                migrated.append(destination)
            except Exception as exc:
                print(f"Failed to migrate legacy {spec['name']}: {exc}")
            finally:
                if temp_path and os.path.lexists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
        if migrated:
            cls._fsync_hed_directory(cls.HED_MODEL_DIR)
        cls._invalidate_hed_status_cache()
        return tuple(migrated)

    @classmethod
    def _register_hed_layers(cls):
        if cls._hed_crop_layer_registered:
            return

        cv2 = cls._require_cv2_runtime("HED edge detection")
        register_fn = getattr(cv2.dnn, "registerLayer", None)
        if register_fn is None:
            register_fn = getattr(cv2, "dnn_registerLayer", None)

        if register_fn is None:
            return

        try:
            register_fn("Crop", cls._HEDCropLayer)
        except Exception as exc:
            if "already" not in str(exc).lower():
                raise

        cls._hed_crop_layer_registered = True

    @staticmethod
    def _hed_caffe_reader(cv2):
        """Return the OpenCV 4 Caffe reader or explain the OpenCV 5 boundary."""
        reader = getattr(getattr(cv2, "dnn", None), "readNetFromCaffe", None)
        if reader is None:
            reader = getattr(cv2, "dnn_readNetFromCaffe", None)
        if reader is None:
            version = getattr(cv2, "__version__", "unknown")
            raise RuntimeError(
                "HED requires OpenCV 4.x because OpenCV 5 removed the Caffe "
                f"importer (installed OpenCV: {version})."
            )
        return reader

    @classmethod
    def _create_hed_net(cls, prototxt_path=None, caffemodel_path=None, validate_forward=False):
        prototxt = prototxt_path or cls.HED_PROTOTXT
        caffemodel = caffemodel_path or cls.HED_CAFFEMODEL
        specs = cls._hed_asset_specs()
        prototxt_bytes = cls._read_verified_hed_asset_bytes(prototxt, specs[0])
        caffemodel_bytes = cls._read_verified_hed_asset_bytes(caffemodel, specs[1])
        cv2 = cls._require_cv2_runtime("HED edge detection")
        read_net_from_caffe = cls._hed_caffe_reader(cv2)
        cls._register_hed_layers()
        # OpenCV 4 accepts uint8 buffers. Loading the already-verified bytes
        # closes the path-replacement window between hashing and Caffe parse.
        net = read_net_from_caffe(
            np.frombuffer(prototxt_bytes, dtype=np.uint8),
            np.frombuffer(caffemodel_bytes, dtype=np.uint8),
        )
        if validate_forward:
            cls._validate_hed_net(net)
        return net

    @classmethod
    def _validate_hed_net(cls, net):
        cv2 = cls._require_cv2_runtime("HED edge detection")
        dummy = np.zeros(
            (cls.HED_VALIDATION_IMAGE_SIZE, cls.HED_VALIDATION_IMAGE_SIZE, 3),
            dtype=np.uint8,
        )
        blob = cv2.dnn.blobFromImage(
            dummy,
            scalefactor=1.0,
            size=(cls.HED_VALIDATION_IMAGE_SIZE, cls.HED_VALIDATION_IMAGE_SIZE),
            mean=cls.HED_MEAN,
            swapRB=False,
            crop=False,
        )
        net.setInput(blob)
        output = net.forward(cls.HED_OUTPUT_LAYER)
        if output is None or np.asarray(output).size == 0:
            raise RuntimeError("HED forward pass returned no output")

    @classmethod
    def get_hed_runtime_status(cls, force_refresh=False):
        """Return whether HED assets are present and actually loadable."""
        if not is_cv2_available():
            return cls._missing_opencv_status("HED edge detection")
        if not os.path.lexists(cls.HED_PROTOTXT):
            return {
                "ok": False,
                "reason": "missing_prototxt",
                "message": f"Missing HED prototxt: {cls.HED_PROTOTXT}",
            }
        if not os.path.lexists(cls.HED_CAFFEMODEL):
            return {
                "ok": False,
                "reason": "missing_weights",
                "message": f"Missing HED weights: {cls.HED_CAFFEMODEL}",
            }

        signature = cls._hed_file_signature()
        if (
            not force_refresh
            and cls._hed_runtime_status_cache is not None
            and cls._hed_runtime_status_signature == signature
        ):
            return dict(cls._hed_runtime_status_cache)

        verified, verification_error = cls._verify_hed_asset_pair()
        if not verified:
            status = {
                "ok": False,
                "reason": "invalid_asset",
                "message": verification_error,
            }
            cls._hed_runtime_status_cache = dict(status)
            cls._hed_runtime_status_signature = signature
            return dict(status)

        try:
            cls._create_hed_net(validate_forward=True)
            status = {
                "ok": True,
                "reason": "ready",
                "message": "HED model loaded successfully.",
            }
        except Exception as exc:
            status = {
                "ok": False,
                "reason": "invalid_runtime",
                "message": str(exc),
            }

        cls._hed_runtime_status_cache = dict(status)
        cls._hed_runtime_status_signature = signature
        return dict(status)

    def _init_hed(self):
        """Initialize HED network if available."""
        try:
            status = self.get_hed_runtime_status()
            if status.get("ok"):
                self.hed_net = self._create_hed_net(validate_forward=False)
                print("HED model loaded successfully")
            else:
                print(f"HED runtime is not ready. Will fallback to Canny: {status.get('message')}")
        except Exception as e:
            print(f"HED init error: {e}")
            self.hed_net = None

    def detect_edges(
        self,
        image: np.ndarray,
        low_threshold=DEFAULT_CANNY_LOW_THRESHOLD,
        high_threshold=DEFAULT_CANNY_HIGH_THRESHOLD,
    ) -> np.ndarray:
        """
        Detect edges using selected method.
        """
        gray, color = self._prepare_input_images(image)

        if self.method == self.METHOD_INK:
            edges = self._detect_ink_centerline(gray, low_threshold, high_threshold)
        elif self.method == self.METHOD_LSD:
            edges = self._detect_lsd(gray)
        elif self.method == self.METHOD_HED:
            edges = self._detect_hed(color, gray)
        else:
            edges = self._detect_canny(gray, low_threshold, high_threshold)

        # Ink Centerline already returns a one-pixel line. Other detectors
        # keep the shared thinning pass for backward-compatible output.
        if self.method != self.METHOD_INK:
            try:
                binary = edges > self.EDGE_PRESENCE_THRESHOLD
                skeleton = self.thin_binary_mask(binary)
                edges = (skeleton * self.EDGE_MAX_VALUE).astype(np.uint8)
            except Exception as e:
                print(f"Skeletonize error: {e}")

        return edges

    @classmethod
    def _detect_ink_centerline(
        cls,
        gray: np.ndarray,
        low_threshold=DEFAULT_CANNY_LOW_THRESHOLD,
        high_threshold=DEFAULT_CANNY_HIGH_THRESHOLD,
    ) -> np.ndarray:
        """Extract a single centerline from locally dark map ink.

        Historical scans commonly contain uneven paper tone, faded strokes,
        text, and symbols. A grayscale black top-hat removes slow background
        variation without turning both sides of every stroke into competing
        edges. Small isolated responses are discarded before skeletonization.

        SciPy and scikit-image provide optimized operations when present. Pure
        NumPy closing, component filtering, and thinning preserve the same
        centerline semantics in dependency-light QGIS runtimes.
        """
        values = np.asarray(gray, dtype=np.float32)
        if values.ndim != 2 or min(values.shape) < 2:
            return np.zeros(values.shape, dtype=np.uint8)
        finite = np.isfinite(values)
        if not finite.any():
            return np.zeros(values.shape, dtype=np.uint8)
        if not finite.all():
            fill_value = float(np.median(values[finite]))
            values = np.where(finite, values, fill_value)

        value_min = float(values.min())
        value_max = float(values.max())
        if value_max <= value_min:
            return np.zeros(values.shape, dtype=np.uint8)
        values = (values - value_min) / (value_max - value_min)

        if _scipy_ndimage is not None:
            background = _scipy_ndimage.grey_closing(
                values,
                size=(cls.INK_BACKGROUND_SIZE, cls.INK_BACKGROUND_SIZE),
                mode="nearest",
            )
        else:
            background = cls._numpy_grey_closing(values, cls.INK_BACKGROUND_SIZE)
        response = np.maximum(background - values, 0.0)
        positive = response[response > 0]
        if positive.size == 0:
            return np.zeros(values.shape, dtype=np.uint8)

        scale = max(
            float(np.percentile(positive, cls.INK_RESPONSE_PERCENTILE)),
            np.finfo(np.float32).eps,
        )
        normalized = np.clip(response / scale, 0.0, 1.0)
        candidate = cls._remove_small_ink_components(
            normalized >= cls.INK_MIN_NORMALIZED_RESPONSE,
        )
        normalized_positive = normalized[candidate]
        if normalized_positive.size == 0:
            return np.zeros(values.shape, dtype=np.uint8)

        if _skimage_threshold_otsu is not None and normalized_positive.size > 1:
            try:
                threshold = float(_skimage_threshold_otsu(normalized_positive))
            except (TypeError, ValueError):
                threshold = cls.INK_MIN_NORMALIZED_RESPONSE
        else:
            threshold = float(np.percentile(normalized_positive, 50.0))
        threshold = max(cls.INK_MIN_NORMALIZED_RESPONSE, threshold)
        # Otsu returns the sole sample value for a perfectly uniform stroke.
        # Step just below the observed maximum so that such a line is kept.
        response_max = float(normalized_positive.max())
        threshold = min(
            threshold,
            float(np.nextafter(response_max, -np.inf)),
        )
        mask = cls._remove_small_ink_components(
            candidate & (normalized >= threshold),
        )

        centerline = cls.thin_binary_mask(mask)
        return centerline.astype(np.uint8) * cls.EDGE_MAX_VALUE

    @classmethod
    def _remove_small_ink_components(cls, mask: np.ndarray) -> np.ndarray:
        """Remove isolated scan speckles without a version-sensitive API."""
        active = np.asarray(mask, dtype=bool)
        if _scipy_ndimage is not None:
            labels, component_count = _scipy_ndimage.label(
                active,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            if not component_count:
                return np.zeros(labels.shape, dtype=bool)
            sizes = np.bincount(labels.ravel())
            keep = sizes >= cls.INK_MIN_COMPONENT_SIZE
            keep[0] = False
            return keep[labels]

        height, width = active.shape
        visited = np.zeros(active.shape, dtype=bool)
        retained = np.zeros(active.shape, dtype=bool)
        for start in np.flatnonzero(active):
            start = int(start)
            start_y, start_x = divmod(start, width)
            if visited[start_y, start_x]:
                continue
            visited[start_y, start_x] = True
            stack = [(start_y, start_x)]
            component = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for next_y in range(max(0, y - 1), min(height, y + 2)):
                    for next_x in range(max(0, x - 1), min(width, x + 2)):
                        if active[next_y, next_x] and not visited[next_y, next_x]:
                            visited[next_y, next_x] = True
                            stack.append((next_y, next_x))
            if len(component) >= cls.INK_MIN_COMPONENT_SIZE:
                rows, columns = zip(*component)
                retained[rows, columns] = True
        return retained

    @staticmethod
    def _numpy_window_filter(values, size, axis, reducer):
        before = int(size) // 2
        after = int(size) - before - 1
        padding = [(0, 0)] * values.ndim
        padding[axis] = (before, after)
        padded = np.pad(values, padding, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            window_shape=int(size),
            axis=axis,
        )
        return reducer(windows, axis=-1)

    @classmethod
    def _numpy_grey_closing(cls, values, size):
        dilated = cls._numpy_window_filter(values, size, 0, np.max)
        dilated = cls._numpy_window_filter(dilated, size, 1, np.max)
        closed = cls._numpy_window_filter(dilated, size, 0, np.min)
        return cls._numpy_window_filter(closed, size, 1, np.min)

    def _detect_canny(
        self,
        gray: np.ndarray,
        low_threshold=DEFAULT_CANNY_LOW_THRESHOLD,
        high_threshold=DEFAULT_CANNY_HIGH_THRESHOLD,
    ) -> np.ndarray:
        """Canny + Adaptive threshold method."""
        if self.cv2 is None:
            return self._detect_numpy_edges(gray, low_threshold, high_threshold)

        cv2 = self.cv2
        # Adaptive threshold for dark lines
        dark_mask = cv2.adaptiveThreshold(
            gray,
            self.EDGE_MAX_VALUE,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.CANNY_ADAPTIVE_BLOCK_SIZE,
            self.CANNY_ADAPTIVE_C,
        )

        # Canny for fine edges
        blurred = cv2.GaussianBlur(gray, self.CANNY_BLUR_KERNEL, 0)
        canny = cv2.Canny(blurred, low_threshold, high_threshold)

        # Combine
        combined = cv2.bitwise_or(dark_mask, canny)

        # Clean up
        kernel = np.ones(self.CANNY_CLOSE_KERNEL, np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        return combined

    @classmethod
    def _detect_numpy_edges(
        cls,
        gray: np.ndarray,
        low_threshold=DEFAULT_CANNY_LOW_THRESHOLD,
        high_threshold=DEFAULT_CANNY_HIGH_THRESHOLD,
    ) -> np.ndarray:
        """Detect local intensity edges without requiring OpenCV.

        This deliberately stays small and conservative: it is a fallback for
        the default mouse-led assist, not a replacement for LSD/HED/SAM.
        ``np.gradient`` is sufficient to create a nearby attraction mask and
        avoids installing a large package just to start tracing.
        """
        values = np.asarray(gray, dtype=np.float32)
        if values.ndim != 2 or min(values.shape) < 2:
            return np.zeros(values.shape, dtype=np.uint8)

        gradient_y, gradient_x = np.gradient(values)
        magnitude = np.hypot(gradient_x, gradient_y)
        finite = magnitude[np.isfinite(magnitude) & (magnitude > 0)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.uint8)

        percentile_threshold = float(np.percentile(finite, 65.0))
        threshold = max(
            2.0,
            min(
                float(high_threshold),
                max(float(low_threshold) * 0.25, percentile_threshold),
            ),
        )
        return (magnitude >= threshold).astype(np.uint8) * cls.EDGE_MAX_VALUE

    def _detect_lsd(self, gray: np.ndarray) -> np.ndarray:
        """
        LSD Line Segment Detector - detects line segments directly.
        Much smoother and more accurate for contour lines.
        """
        cv2 = self.cv2 or self._require_cv2_runtime("LSD edge detection")
        # Detect line segments
        lines, widths, precs, nfas = self.lsd.detect(gray)
        line_segments = self._normalize_lsd_lines(lines)

        # Create edge mask from detected lines
        edge_mask = np.zeros(gray.shape, dtype=np.uint8)

        for line in line_segments:
            x1, y1, x2, y2 = line.astype(int)
            # Draw thicker lines for better pathfinding
            cv2.line(edge_mask, (x1, y1), (x2, y2), self.EDGE_MAX_VALUE, self.LSD_LINE_WIDTH)

        # Also add dark line detection for completeness
        dark_mask = cv2.adaptiveThreshold(
            gray,
            self.EDGE_MAX_VALUE,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.LSD_ADAPTIVE_BLOCK_SIZE,
            self.LSD_ADAPTIVE_C,
        )

        # Combine LSD with dark detection
        combined = cv2.bitwise_or(edge_mask, dark_mask)

        # Morphological closing for continuity
        kernel = np.ones(self.LSD_CLOSE_KERNEL, np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        return combined

    @staticmethod
    def _normalize_lsd_lines(lines: np.ndarray) -> np.ndarray:
        """Normalize OpenCV 4/5 LSD coordinates to an ``(N, 4)`` array.

        OpenCV 4 Python wheels return ``(N, 1, 4)`` while OpenCV 5 wheels
        return ``(N, 4)``.  Reject other layouts instead of silently drawing
        corrupt segments or falling back to another detector.
        """
        if lines is None:
            return np.empty((0, 4), dtype=np.float32)

        line_array = np.asarray(lines)
        if line_array.ndim == 3 and line_array.shape[1:] == (1, 4):
            line_array = line_array[:, 0, :]
        elif line_array.ndim == 2 and line_array.shape[1] == 4:
            pass
        else:
            raise ValueError(
                "Unexpected OpenCV LSD line array shape: "
                f"{line_array.shape}; expected (N, 1, 4) or (N, 4)."
            )

        if not np.issubdtype(line_array.dtype, np.number):
            raise ValueError("OpenCV LSD line coordinates must be numeric.")
        if not np.isfinite(line_array).all():
            raise ValueError("OpenCV LSD line coordinates must be finite.")
        return line_array.astype(np.float32, copy=False)

    def _detect_hed(self, color: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        HED (Holistically-Nested Edge Detection) - Deep learning based.
        Produces smoother, more natural edges than traditional methods.
        """
        cv2 = self.cv2 or self._require_cv2_runtime("HED edge detection")
        if self.hed_net is None:
            # Fallback to Canny if HED not available
            print("HED not available, falling back to Canny")
            return self._detect_canny(gray)

        try:
            h, w = color.shape[:2]

            # Prepare input blob
            # HED expects specific preprocessing
            blob = cv2.dnn.blobFromImage(
                color,
                scalefactor=1.0,
                size=(w, h),
                mean=self.HED_MEAN,
                swapRB=False,
                crop=False
            )

            self.hed_net.setInput(blob)
            hed_output = self.hed_net.forward(self.HED_OUTPUT_LAYER)

            # Post-process: convert to 0-255 range
            hed_edges = hed_output[0, 0]
            hed_edges = cv2.resize(hed_edges, (w, h))
            hed_edges = (self.EDGE_MAX_VALUE * hed_edges).astype(np.uint8)

            # Threshold to binary
            _, binary = cv2.threshold(
                hed_edges,
                self.HED_BINARY_THRESHOLD,
                self.EDGE_MAX_VALUE,
                cv2.THRESH_BINARY,
            )

            # Optional: thin the edges
            kernel = np.ones(self.HED_CLOSE_KERNEL, np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            return binary

        except Exception as e:
            print(f"HED detection error: {e}")
            return self._detect_canny(gray)

    def get_edge_cost_map(self, edges: np.ndarray, edge_weight: float = 0.5) -> np.ndarray:
        """
        Create cost map for pathfinding.
        edge_weight: 0.0 = free draw, 1.0 = strict edge follow
        """
        cv2 = self.cv2 or self._require_cv2_runtime("OpenCV edge cost mapping")
        edge_weight = max(0.0, min(1.0, float(edge_weight)))
        # Distance to nearest edge
        inverted = cv2.bitwise_not(edges)
        dist = cv2.distanceTransform(inverted, cv2.DIST_L2, self.DIST_TRANSFORM_MASK_SIZE)

        # Lower multiplier = more freedom
        multiplier = self.EDGE_COST_BASE_MULTIPLIER + edge_weight * self.EDGE_COST_WEIGHT_SCALE
        cost_map = 1.0 + dist * multiplier

        return cost_map.astype(np.float32)

    @classmethod
    def is_hed_available(cls):
        """Check if HED model files are available."""
        return cls.get_hed_runtime_status().get("ok", False)

    @classmethod
    def _validate_download_url(cls, url: str) -> str:
        """Return a validated remote download URL for HED assets."""
        parsed = urllib.parse.urlparse(url)
        host = (parsed.hostname or "").lower()
        scheme = parsed.scheme.lower()

        if scheme not in cls.HED_ALLOWED_DOWNLOAD_SCHEMES:
            raise ValueError(
                f"Unsupported HED download scheme: {parsed.scheme or '<missing>'}"
            )
        if host not in cls.HED_ALLOWED_DOWNLOAD_HOSTS:
            raise ValueError(f"Unsupported HED download host: {host or '<missing>'}")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"Invalid HED download URL: {exc}") from exc
        if port not in (None, 443):
            raise ValueError(f"Unsupported HED download port: {port}")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("HED download URLs cannot contain credentials or fragments.")
        return url

    @classmethod
    def _open_hed_response(cls, url, timeout):
        request = urllib.request.Request(
            url,
            headers=cls.HED_REQUEST_HEADERS,
            method="GET",
        )
        opener = urllib.request.build_opener(_HEDNoRedirectHandler())
        try:
            return opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                status = int(exc.code)
            except (TypeError, ValueError):
                status = 0
            try:
                exc.close()
            # Preserve the original HTTP error.
            except Exception:  # nosec B110
                pass
            if 300 <= status < 400:
                raise RuntimeError("HED asset redirects are not accepted.") from exc
            raise

    @classmethod
    def _download_hed_asset(cls, spec, temp_path, descriptor, timeout):
        """Stream one pinned asset into its still-open ``mkstemp`` file."""
        url = cls._validate_download_url(spec["url"])
        response = cls._open_hed_response(url, timeout)
        with _quietly_closing_hed_response(response):
            final_url = response.geturl()
            cls._validate_download_url(final_url)
            if final_url != url:
                raise RuntimeError(
                    f"{spec['name']} response came from an unexpected final URL."
                )
            status = response.getcode()
            if status is None:
                status = getattr(response, "status", None)
            if int(status or 0) != 200:
                raise RuntimeError(
                    f"{spec['name']} download returned unexpected HTTP status {status}."
                )

            headers = getattr(response, "headers", {})
            content_encoding = headers.get("Content-Encoding")
            if content_encoding and content_encoding.strip().lower() != "identity":
                raise RuntimeError("Compressed HED asset responses are not accepted.")
            announced_size = headers.get("Content-Length")
            if announced_size is not None:
                try:
                    announced_size = int(announced_size)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"{spec['name']} returned an invalid Content-Length."
                    ) from exc
                if announced_size != spec["size"]:
                    raise RuntimeError(
                        f"{spec['name']} Content-Length mismatch: expected "
                        f"{spec['size']} bytes, got {announced_size}."
                    )

            digest = hashlib.sha256()
            total = 0
            # Keep writing through the descriptor returned by mkstemp. Reopening
            # the pathname here would permit a local symlink swap between file
            # creation and the first byte written.
            with os.fdopen(descriptor, "wb", closefd=False) as out_file:
                out_file.seek(0)
                out_file.truncate()
                while total <= spec["size"]:
                    chunk = response.read(
                        min(cls.HED_DOWNLOAD_CHUNK_SIZE, spec["size"] - total + 1)
                    )
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > spec["size"]:
                        raise RuntimeError(
                            f"{spec['name']} download exceeded the "
                            f"{spec['size']}-byte limit."
                        )
                    digest.update(chunk)
                    out_file.write(chunk)
                out_file.flush()
                os.fsync(out_file.fileno())

        if total != spec["size"]:
            raise RuntimeError(
                f"Incomplete {spec['name']} download: expected {spec['size']} bytes, "
                f"got {total}."
            )
        if not hmac.compare_digest(digest.hexdigest(), spec["sha256"]):
            raise RuntimeError(
                f"Downloaded {spec['name']} SHA-256 does not match the pinned artifact."
            )
        verified, verification_error = cls._verify_hed_asset_file(temp_path, spec)
        if not verified:
            raise RuntimeError(
                f"Downloaded {spec['name']} failed disk verification: {verification_error}"
            )

    @staticmethod
    def _fsync_hed_directory(directory):
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = None
        try:
            descriptor = os.open(directory, flags)
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @classmethod
    def _create_hed_backup(cls, path, directory):
        if not os.path.lexists(path):
            return None
        file_stat = os.lstat(path)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(
                f"Cannot replace non-regular HED asset safely: {path}"
            )

        fd, backup_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".rollback",
            dir=directory,
        )
        os.close(fd)
        try:
            os.remove(backup_path)
            try:
                os.link(path, backup_path, follow_symlinks=False)
            except (OSError, NotImplementedError, TypeError):
                shutil.copy2(path, backup_path, follow_symlinks=False)
                with open(backup_path, "r+b") as backup_file:
                    backup_file.flush()
                    os.fsync(backup_file.fileno())
            return backup_path
        except Exception:
            try:
                os.remove(backup_path)
            except OSError:
                pass
            raise

    @classmethod
    def _publish_hed_asset_pair(cls, staged_paths, specs, directory):
        original_exists = {
            spec["path"]: os.path.lexists(spec["path"])
            for spec in specs
        }
        backups = {}
        publish_started = False
        preserve_backups = False
        try:
            for spec in specs:
                backups[spec["path"]] = cls._create_hed_backup(
                    spec["path"],
                    directory,
                )

            publish_started = True
            for staged_path, spec in zip(staged_paths, specs):
                os.replace(staged_path, spec["path"])
            for spec in specs:
                verified, verification_error = cls._verify_hed_asset_file(
                    spec["path"],
                    spec,
                )
                if not verified:
                    raise RuntimeError(
                        f"Published {spec['name']} failed final verification: "
                        f"{verification_error}"
                    )
            cls._fsync_hed_directory(directory)
        except Exception as publish_error:
            rollback_errors = []
            if publish_started:
                for spec in reversed(specs):
                    destination = spec["path"]
                    backup_path = backups.get(destination)
                    try:
                        if original_exists[destination]:
                            if not backup_path or not os.path.lexists(backup_path):
                                raise RuntimeError(
                                    f"Missing rollback copy for {destination}"
                                )
                            os.replace(backup_path, destination)
                            # POSIX rename is a no-op when both names are hard
                            # links to the same inode, so remove that surviving
                            # rollback name explicitly.
                            if os.path.lexists(backup_path):
                                os.remove(backup_path)
                            backups[destination] = None
                        elif os.path.lexists(destination):
                            os.remove(destination)
                    except Exception as rollback_error:
                        rollback_errors.append(str(rollback_error))
                cls._fsync_hed_directory(directory)
            if rollback_errors:
                preserve_backups = True
                recovery_paths = [
                    backup_path
                    for backup_path in backups.values()
                    if backup_path and os.path.lexists(backup_path)
                ]
                recovery_detail = ""
                if recovery_paths:
                    recovery_detail = (
                        "; recovery backups preserved at: "
                        + ", ".join(recovery_paths)
                    )
                raise RuntimeError(
                    f"HED asset publish failed: {publish_error}; rollback failed: "
                    + "; ".join(rollback_errors)
                    + recovery_detail
                ) from publish_error
            raise
        finally:
            if not preserve_backups:
                for backup_path in backups.values():
                    if backup_path and os.path.lexists(backup_path):
                        try:
                            os.remove(backup_path)
                        except OSError:
                            pass

    @classmethod
    def download_hed_assets(cls, timeout=60):
        """Download, validate, and transactionally publish the pinned HED pair."""
        specs = cls._hed_asset_specs()
        target_directories = {
            os.path.abspath(os.path.dirname(spec["path"]))
            for spec in specs
        }
        if len(target_directories) != 1:
            return False, "HED assets must share one destination directory."
        model_dir = target_directories.pop()
        try:
            cls._ensure_hed_model_directory(model_dir)
        except Exception as exc:
            return False, str(exc)

        temp_paths = []
        try:
            for spec in specs:
                cls._validate_download_url(spec["url"])
                fd, temp_path = tempfile.mkstemp(
                    prefix=spec["temp_prefix"],
                    suffix=spec["temp_suffix"],
                    dir=model_dir,
                )
                temp_paths.append(temp_path)
                try:
                    cls._download_hed_asset(spec, temp_path, fd, timeout)
                finally:
                    os.close(fd)

            cls._create_hed_net(
                prototxt_path=temp_paths[0],
                caffemodel_path=temp_paths[1],
                validate_forward=True,
            )

            cls._publish_hed_asset_pair(temp_paths, specs, model_dir)
            cls._invalidate_hed_status_cache()
            return True, None
        except Exception as exc:
            return False, str(exc)
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @classmethod
    def get_hed_download_info(cls):
        """Get info about downloading HED model."""
        return {
            'prototxt_url': cls.HED_PROTOTXT_URL,
            'caffemodel_url': cls.HED_CAFFEMODEL_URL,
            'prototxt_path': cls.HED_PROTOTXT,
            'caffemodel_path': cls.HED_CAFFEMODEL,
            'prototxt_size_bytes': cls.HED_PROTOTXT_SIZE_BYTES,
            'prototxt_sha256': cls.HED_PROTOTXT_SHA256,
            'caffemodel_size_bytes': cls.HED_CAFFEMODEL_SIZE_BYTES,
            'caffemodel_sha256': cls.HED_CAFFEMODEL_SHA256,
            'size_mb': cls.HED_MODEL_SIZE_MB
        }
