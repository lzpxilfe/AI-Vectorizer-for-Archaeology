# -*- coding: utf-8 -*-
"""
Unified SAM engine for MobileSAM and Meta Segment Anything backends.
"""

import importlib.util
import json
import os
import tempfile

import numpy as np

from ..config import (
    DEFAULT_FULL_SAM_MODEL_TYPE,
    DEFAULT_MOBILE_SAM_MODEL_TYPE,
    PLUGIN_NAME,
    SAM_BACKEND_FULL,
    SAM_BACKEND_MOBILE,
)


MOBILE_SAM_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("mobile_sam") is not None
)
SEGMENT_ANYTHING_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("segment_anything") is not None
)


class SAMEngine:
    DOWNLOAD_CHUNK_SIZE = 8192
    DOWNLOAD_TIMEOUT_SECONDS = 60
    REQUEST_HEADERS = {"User-Agent": PLUGIN_NAME}

    BACKEND_SPECS = {
        SAM_BACKEND_MOBILE: {
            "display_name": "MobileSAM",
            "default_model_type": DEFAULT_MOBILE_SAM_MODEL_TYPE,
            "module_name": "mobile_sam",
            "models": {
                "vit_t": {
                    "weights_filename": "mobile_sam.pt",
                    "weights_url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
                    "size_hint_mb": 39,
                },
            },
        },
        SAM_BACKEND_FULL: {
            "display_name": "SAM",
            "default_model_type": DEFAULT_FULL_SAM_MODEL_TYPE,
            "module_name": "segment_anything",
            "models": {
                "vit_b": {
                    "weights_filename": "sam_vit_b_01ec64.pth",
                    "weights_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                    "size_hint_mb": 358,
                },
                "vit_l": {
                    "weights_filename": "sam_vit_l_0b3195.pth",
                    "weights_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "size_hint_mb": 1247,
                },
                "vit_h": {
                    "weights_filename": "sam_vit_h_4b8939.pth",
                    "weights_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "size_hint_mb": 2445,
                },
            },
        },
    }

    def __init__(self, backend=SAM_BACKEND_MOBILE, model_type=None, device=None):
        """
        Initialize a SAM backend.

        Args:
            backend (str): SAM backend key.
            model_type (str): model family key (e.g. vit_t or vit_b).
            device (str): 'cuda' or 'cpu'. Auto-detect if None.
        """
        self.backend = backend
        self.model_type = model_type or self.default_model_type_for_backend(backend)
        self.predictor = None
        self.is_ready = False
        self.model_spec = self._resolve_model_spec(self.backend, self.model_type)
        self.display_name = self.display_name_for_backend(self.backend, self.model_type)

        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.weights_path = os.path.join(models_dir, self.model_spec["weights_filename"])
        self.weights_meta_path = os.path.join(
            models_dir,
            f"{os.path.splitext(self.model_spec['weights_filename'])[0]}.meta.json",
        )

        if not self.is_backend_available(self.backend):
            self.device = None
            return

        try:
            import torch

            self.device = "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
        except Exception:
            self.device = device or "cpu"

    @classmethod
    def _backend_spec(cls, backend):
        spec = cls.BACKEND_SPECS.get(backend)
        if spec is None:
            raise ValueError(f"Unsupported SAM backend: {backend}")
        return spec

    @classmethod
    def _resolve_model_spec(cls, backend, model_type):
        backend_spec = cls._backend_spec(backend)
        model_spec = backend_spec["models"].get(model_type)
        if model_spec is None:
            raise ValueError(f"Unsupported model type '{model_type}' for backend '{backend}'")
        return model_spec

    @classmethod
    def is_backend_available(cls, backend):
        backend_spec = cls._backend_spec(backend)
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec(backend_spec["module_name"]) is not None
        )

    @classmethod
    def default_model_type_for_backend(cls, backend):
        return cls._backend_spec(backend)["default_model_type"]

    @classmethod
    def display_name_for_backend(cls, backend, model_type=None):
        backend_spec = cls._backend_spec(backend)
        resolved_model_type = model_type or backend_spec["default_model_type"]
        if backend == SAM_BACKEND_FULL:
            return f"{backend_spec['display_name']} ({resolved_model_type.upper()})"
        return backend_spec["display_name"]

    @classmethod
    def size_hint_mb_for_backend(cls, backend, model_type=None):
        resolved_model_type = model_type or cls.default_model_type_for_backend(backend)
        return cls._resolve_model_spec(backend, resolved_model_type).get("size_hint_mb")

    def _ensure_models_dir(self):
        models_dir = os.path.dirname(self.weights_path)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

    @staticmethod
    def _import_requests():
        try:
            import requests
            return requests, None
        except Exception as exc:
            return None, str(exc)

    def _read_local_meta(self):
        if not os.path.exists(self.weights_meta_path):
            return {}
        try:
            with open(self.weights_meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_local_meta(self, remote_info):
        meta = {
            "url": self.model_spec["weights_url"],
            "backend": self.backend,
            "model_type": self.model_type,
            "etag": remote_info.get("etag"),
            "last_modified": remote_info.get("last_modified"),
            "content_length": remote_info.get("content_length"),
        }
        try:
            with open(self.weights_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"Failed to write SAM metadata: {exc}")

    @staticmethod
    def _parse_remote_headers(headers):
        content_length = headers.get("Content-Length")
        try:
            content_length = int(content_length) if content_length is not None else None
        except Exception:
            content_length = None
        return {
            "etag": headers.get("ETag"),
            "last_modified": headers.get("Last-Modified"),
            "content_length": content_length,
        }

    def get_remote_weights_info(self):
        """Fetch remote metadata for the selected SAM backend weights."""
        url = self.model_spec["weights_url"]
        requests, import_error = self._import_requests()
        if requests is None:
            return {"ok": False, "error": f"requests is unavailable: {import_error}"}
        try:
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            if response.status_code >= 400 or (
                "Content-Length" not in response.headers and "ETag" not in response.headers
            ):
                response = requests.get(
                    url,
                    stream=True,
                    timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                    headers=self.REQUEST_HEADERS,
                )
                response.raise_for_status()
            else:
                response.raise_for_status()

            info = self._parse_remote_headers(response.headers)
            return {"ok": True, **info}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def get_local_weights_info(self):
        """Get local weights presence and metadata."""
        exists = os.path.exists(self.weights_path)
        size = os.path.getsize(self.weights_path) if exists else None
        meta = self._read_local_meta()
        return {
            "exists": exists,
            "size": size,
            "backend": meta.get("backend"),
            "model_type": meta.get("model_type"),
            "etag": meta.get("etag"),
            "last_modified": meta.get("last_modified"),
            "content_length": meta.get("content_length"),
        }

    def check_weights_update(self):
        """
        Compare local weights with remote metadata.
        Returns:
            dict with keys:
            - ok (bool)
            - status (str): not_installed|update_available|up_to_date|unknown|check_failed
            - message (str)
            - local (dict)
            - remote (dict|None)
        """
        local = self.get_local_weights_info()
        remote = self.get_remote_weights_info()

        if not remote.get("ok"):
            return {
                "ok": False,
                "status": "check_failed",
                "message": f"Failed to fetch remote metadata: {remote.get('error', 'unknown error')}",
                "local": local,
                "remote": None,
            }

        if not local["exists"]:
            return {
                "ok": True,
                "status": "not_installed",
                "message": "Local weights file is missing.",
                "local": local,
                "remote": remote,
            }

        if remote.get("etag") and local.get("etag") and remote["etag"] != local["etag"]:
            return {
                "ok": True,
                "status": "update_available",
                "message": "Remote ETag differs from local metadata.",
                "local": local,
                "remote": remote,
            }

        if remote.get("content_length") and local.get("size"):
            if int(remote["content_length"]) != int(local["size"]):
                return {
                    "ok": True,
                    "status": "update_available",
                    "message": "Remote file size differs from local file size.",
                    "local": local,
                    "remote": remote,
                }

        if (
            remote.get("last_modified")
            and local.get("last_modified")
            and remote["last_modified"] != local["last_modified"]
            and not remote.get("etag")
        ):
            return {
                "ok": True,
                "status": "update_available",
                "message": "Remote Last-Modified differs from local metadata.",
                "local": local,
                "remote": remote,
            }

        if not remote.get("etag") and not remote.get("content_length") and not remote.get("last_modified"):
            return {
                "ok": True,
                "status": "unknown",
                "message": "Remote host did not expose comparable metadata.",
                "local": local,
                "remote": remote,
            }

        return {
            "ok": True,
            "status": "up_to_date",
            "message": "Local weights appear up-to-date.",
            "local": local,
            "remote": remote,
        }

    def _load_predictor(self):
        if self.backend == SAM_BACKEND_MOBILE:
            from mobile_sam import SamPredictor, sam_model_registry
        else:
            from segment_anything import SamPredictor, sam_model_registry
        return SamPredictor, sam_model_registry

    def load_model(self):
        """Load the selected SAM backend from the local checkpoint file."""
        self.predictor = None
        self.is_ready = False

        if not self.is_backend_available(self.backend):
            return False, f"{self.display_name} library not installed."

        try:
            import torch  # noqa: F401

            SamPredictor, sam_model_registry = self._load_predictor()
        except Exception as exc:
            return False, f"{self.display_name} dependencies are not ready: {str(exc)}"

        if not os.path.exists(self.weights_path):
            return False, f"Model weights not found at {self.weights_path}"

        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.weights_path)
            sam.to(device=self.device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            self.is_ready = True
            return True, "Model loaded successfully."
        except Exception as exc:
            return False, f"Error loading model: {str(exc)}"

    def download_weights(self):
        """Download the selected SAM backend weights."""
        url = self.model_spec["weights_url"]
        self._ensure_models_dir()
        requests, import_error = self._import_requests()
        if requests is None:
            print(f"Download failed: requests is unavailable: {import_error}")
            return False

        temp_path = None
        try:
            print(f"Downloading {self.display_name} weights from {url}...")
            response = requests.get(
                url,
                stream=True,
                timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            response.raise_for_status()
            remote_info = self._parse_remote_headers(response.headers)

            fd, temp_path = tempfile.mkstemp(
                prefix=f"{os.path.basename(self.weights_path)}.",
                suffix=".download",
                dir=os.path.dirname(self.weights_path),
            )
            os.close(fd)

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=self.DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            if remote_info.get("content_length") is not None:
                local_size = os.path.getsize(temp_path)
                if int(local_size) != int(remote_info["content_length"]):
                    raise RuntimeError(
                        f"Incomplete download: expected {remote_info['content_length']} bytes, got {local_size} bytes"
                    )
            os.replace(temp_path, self.weights_path)
            temp_path = None
            self._write_local_meta(remote_info)
            print("Download complete.")
            return True
        except Exception as exc:
            print(f"Download failed: {exc}")
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def set_image(self, image: np.ndarray):
        """
        Set image for embedding calculation.

        Args:
            image (np.ndarray): RGB image (H, W, 3).
        """
        if self.predictor:
            self.predictor.set_image(image)

    def predict_point(self, points, labels):
        """
        Predict mask from point prompts.

        Args:
            points (np.ndarray or list): [[x, y], ...]
            labels (np.ndarray or list): [1, 0, ...] (1: fg, 0: bg)

        Returns:
            mask (np.ndarray): Best binary mask.
        """
        if not self.predictor:
            return None

        masks, _scores, _logits = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=False,
        )
        return masks[0]
