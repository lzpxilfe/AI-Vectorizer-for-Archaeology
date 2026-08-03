# -*- coding: utf-8 -*-
"""
Edge Detection Module - Multiple detection methods for historical maps
Supports: Canny+Adaptive, LSD Line Detector, HED (Holistically-Nested Edge Detection)
"""

import numpy as np
import os
import shutil
import tempfile
import urllib.parse
import urllib.request

from .dependencies import build_missing_cv2_message, get_cv2, is_cv2_available, require_cv2

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
except Exception:
    _skimage_skeletonize = None


class EdgeDetector:
    METHOD_CANNY = 'canny'
    METHOD_LSD = 'lsd'
    METHOD_HED = 'hed'

    EDGE_MAX_VALUE = 255
    EDGE_PRESENCE_THRESHOLD = 50

    DEFAULT_CANNY_LOW_THRESHOLD = 30
    DEFAULT_CANNY_HIGH_THRESHOLD = 100
    CANNY_ADAPTIVE_BLOCK_SIZE = 21
    CANNY_ADAPTIVE_C = 10
    CANNY_BLUR_KERNEL = (3, 3)
    CANNY_CLOSE_KERNEL = (2, 2)

    LSD_LINE_WIDTH = 2
    LSD_ADAPTIVE_BLOCK_SIZE = 21
    LSD_ADAPTIVE_C = 8
    LSD_CLOSE_KERNEL = (3, 3)

    HED_MEAN = (104.00698793, 116.66876762, 122.67891434)
    HED_BINARY_THRESHOLD = 50
    HED_CLOSE_KERNEL = (2, 2)

    # At edge_weight=0 the edge distance must have no influence on an
    # explicitly requested Auto Path proposal.  The proposal itself remains
    # opt-in and can still choose a shortest grid path; this constant only
    # controls edge attraction within that path search.
    EDGE_COST_BASE_MULTIPLIER = 0.0
    EDGE_COST_WEIGHT_SCALE = 0.9
    DIST_TRANSFORM_MASK_SIZE = 5

    HED_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    HED_PROTOTXT = os.path.join(HED_MODEL_DIR, 'hed_deploy.prototxt')
    HED_CAFFEMODEL = os.path.join(HED_MODEL_DIR, 'hed_pretrained_bsds.caffemodel')
    HED_PROTOTXT_URL = 'https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt'
    HED_CAFFEMODEL_URL = 'https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel'
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
        if cv2 is None:
            return binary

        ximgproc = getattr(cv2, "ximgproc", None)
        if ximgproc is not None and hasattr(ximgproc, "thinning"):
            thinned = ximgproc.thinning(binary.astype(np.uint8) * 255)
            return thinned > 0

        return binary

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

    def __init__(self, method=METHOD_CANNY):
        """
        Args:
            method: 'canny', 'lsd', or 'hed'
        """
        self.method = method
        self.hed_net = None
        self.cv2 = None
        self.lsd = None

        # The default Canny-style Human Assist must remain usable in a clean
        # QGIS Python environment. LSD and HED still require OpenCV, while
        # Canny can use the small NumPy fallback below when cv2 is absent.
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
    def _hed_file_signature(cls):
        if not os.path.exists(cls.HED_PROTOTXT) or not os.path.exists(cls.HED_CAFFEMODEL):
            return None
        return (
            os.path.getsize(cls.HED_PROTOTXT),
            os.path.getmtime(cls.HED_PROTOTXT),
            os.path.getsize(cls.HED_CAFFEMODEL),
            os.path.getmtime(cls.HED_CAFFEMODEL),
        )

    @classmethod
    def _invalidate_hed_status_cache(cls):
        cls._hed_runtime_status_cache = None
        cls._hed_runtime_status_signature = None

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

    @classmethod
    def _create_hed_net(cls, prototxt_path=None, caffemodel_path=None, validate_forward=False):
        prototxt = prototxt_path or cls.HED_PROTOTXT
        caffemodel = caffemodel_path or cls.HED_CAFFEMODEL
        cv2 = cls._require_cv2_runtime("HED edge detection")
        cls._register_hed_layers()
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
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
        output = net.forward()
        if output is None or np.asarray(output).size == 0:
            raise RuntimeError("HED forward pass returned no output")

    @classmethod
    def get_hed_runtime_status(cls, force_refresh=False):
        """Return whether HED assets are present and actually loadable."""
        if not is_cv2_available():
            return cls._missing_opencv_status("HED edge detection")
        if not os.path.exists(cls.HED_PROTOTXT):
            return {
                "ok": False,
                "reason": "missing_prototxt",
                "message": f"Missing HED prototxt: {cls.HED_PROTOTXT}",
            }
        if not os.path.exists(cls.HED_CAFFEMODEL):
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

        if self.method == self.METHOD_LSD:
            edges = self._detect_lsd(gray)
        elif self.method == self.METHOD_HED:
            edges = self._detect_hed(color, gray)
        else:
            edges = self._detect_canny(gray, low_threshold, high_threshold)

        # SKELETONIZATION: Ensure edges are 1px wide
        # This prevents "walking inside the edge" and reduces jitter
        try:
            # Normalize to binary (0 or 1)
            binary = edges > self.EDGE_PRESENCE_THRESHOLD

            # Convert back to uint8 (0 or 255)
            skeleton = self.thin_binary_mask(binary)
            edges = (skeleton * self.EDGE_MAX_VALUE).astype(np.uint8)
        except Exception as e:
            print(f"Skeletonize error: {e}")

        return edges

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
            hed_output = self.hed_net.forward()

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
        return url

    @classmethod
    def download_hed_assets(cls, timeout=60):
        """Download HED assets atomically and validate them before replacing local files."""
        info = cls.get_hed_download_info()
        prototxt_url = cls._validate_download_url(info["prototxt_url"])
        caffemodel_url = cls._validate_download_url(info["caffemodel_url"])
        model_dir = os.path.dirname(info["caffemodel_path"])
        os.makedirs(model_dir, exist_ok=True)

        temp_paths = []

        try:
            fd, temp_prototxt = tempfile.mkstemp(
                prefix="hed_prototxt_",
                suffix=".prototxt",
                dir=model_dir,
            )
            os.close(fd)
            temp_paths.append(temp_prototxt)
            prototxt_response = urllib.request.urlopen(  # nosec B310
                prototxt_url,
                timeout=timeout,
            )
            with prototxt_response as response, open(
                temp_prototxt,
                "wb",
            ) as out_file:
                shutil.copyfileobj(response, out_file)

            fd, temp_caffemodel = tempfile.mkstemp(
                prefix="hed_weights_",
                suffix=".caffemodel",
                dir=model_dir,
            )
            os.close(fd)
            temp_paths.append(temp_caffemodel)
            caffemodel_response = urllib.request.urlopen(  # nosec B310
                caffemodel_url,
                timeout=timeout,
            )
            with caffemodel_response as response, open(
                temp_caffemodel,
                "wb",
            ) as out_file:
                shutil.copyfileobj(response, out_file)

            cls._create_hed_net(
                prototxt_path=temp_prototxt,
                caffemodel_path=temp_caffemodel,
                validate_forward=True,
            )

            os.replace(temp_prototxt, info["prototxt_path"])
            temp_paths.remove(temp_prototxt)
            os.replace(temp_caffemodel, info["caffemodel_path"])
            temp_paths.remove(temp_caffemodel)
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
            'size_mb': cls.HED_MODEL_SIZE_MB
        }
