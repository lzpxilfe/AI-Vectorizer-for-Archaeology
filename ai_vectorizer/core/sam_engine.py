# -*- coding: utf-8 -*-
"""
SAM Engine Module for AI Vectorizer
Uses MobileSAM for efficient segmentation on historical maps.
"""

import os
import json
import importlib.util
import numpy as np
import requests
from ..config import DEFAULT_SAM_MODEL_TYPE

MOBILE_SAM_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("mobile_sam") is not None
)

class SAMEngine:
    WEIGHTS_FILENAME = "mobile_sam.pt"
    WEIGHTS_META_FILENAME = "mobile_sam.meta.json"
    WEIGHTS_DOWNLOAD_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    DOWNLOAD_CHUNK_SIZE = 8192
    DOWNLOAD_TIMEOUT_SECONDS = 60
    REQUEST_HEADERS = {"User-Agent": "ArchaeoTrace/1.0"}

    def __init__(self, model_type=DEFAULT_SAM_MODEL_TYPE, device=None):
        """
        Initialize SAM Engine.
        
        Args:
            model_type (str): 'vit_t' for MobileSAM.
            device (str): 'cuda' or 'cpu'. Auto-detect if None.
        """
        self.predictor = None
        self.is_ready = False
        self.model_type = model_type
        
        # Keep these paths available even when torch/mobile_sam is not installed.
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.weights_path = os.path.join(models_dir, self.WEIGHTS_FILENAME)
        self.weights_meta_path = os.path.join(models_dir, self.WEIGHTS_META_FILENAME)
        
        if not MOBILE_SAM_AVAILABLE:
            self.device = None
            return

        try:
            import torch
            self.device = "cuda" if device is None and torch.cuda.is_available() else (device or "cpu")
        except Exception:
            self.device = device or "cpu"

    def _ensure_models_dir(self):
        models_dir = os.path.dirname(self.weights_path)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

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
            "url": self.WEIGHTS_DOWNLOAD_URL,
            "etag": remote_info.get("etag"),
            "last_modified": remote_info.get("last_modified"),
            "content_length": remote_info.get("content_length"),
        }
        try:
            with open(self.weights_meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to write SAM metadata: {e}")

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
        """Fetch remote metadata for MobileSAM weights."""
        try:
            response = requests.head(
                self.WEIGHTS_DOWNLOAD_URL,
                allow_redirects=True,
                timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            # Some hosts may not provide useful HEAD responses.
            if response.status_code >= 400 or (
                "Content-Length" not in response.headers and "ETag" not in response.headers
            ):
                response = requests.get(
                    self.WEIGHTS_DOWNLOAD_URL,
                    stream=True,
                    timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                    headers=self.REQUEST_HEADERS,
                )
                response.raise_for_status()
            else:
                response.raise_for_status()

            info = self._parse_remote_headers(response.headers)
            return {
                "ok": True,
                **info,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": str(e),
            }

    def get_local_weights_info(self):
        """Get local weights presence and metadata."""
        exists = os.path.exists(self.weights_path)
        size = os.path.getsize(self.weights_path) if exists else None
        meta = self._read_local_meta()
        return {
            "exists": exists,
            "size": size,
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

        # Prefer strong metadata when available.
        if remote.get("etag") and local.get("etag") and remote["etag"] != local["etag"]:
            return {
                "ok": True,
                "status": "update_available",
                "message": "Remote ETag differs from local metadata.",
                "local": local,
                "remote": remote,
            }

        # Fall back to size comparison.
        if remote.get("content_length") and local.get("size"):
            if int(remote["content_length"]) != int(local["size"]):
                return {
                    "ok": True,
                    "status": "update_available",
                    "message": "Remote file size differs from local file size.",
                    "local": local,
                    "remote": remote,
                }

        # Weak fallback: last-modified.
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

        # If we cannot compare anything concrete, mark as unknown.
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

    def load_model(self):
        """Load the MobileSAM model from weights file."""
        if not MOBILE_SAM_AVAILABLE:
            return False, "MobileSAM library not installed."
        try:
            import torch
            from mobile_sam import sam_model_registry, SamPredictor
        except Exception as e:
            return False, f"MobileSAM dependencies are not ready: {str(e)}"
            
        if not os.path.exists(self.weights_path):
            return False, f"Model weights not found at {self.weights_path}"
            
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.weights_path)
            sam.to(device=self.device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            self.is_ready = True
            return True, "Model loaded successfully."
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def download_weights(self):
        """Download MobileSAM weights if missing."""
        url = self.WEIGHTS_DOWNLOAD_URL
        self._ensure_models_dir()
            
        try:
            print(f"Downloading MobileSAM weights from {url}...")
            response = requests.get(
                url,
                stream=True,
                timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            response.raise_for_status()
            remote_info = self._parse_remote_headers(response.headers)
            
            with open(self.weights_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            if remote_info.get("content_length") is not None:
                local_size = os.path.getsize(self.weights_path)
                if int(local_size) != int(remote_info["content_length"]):
                    raise RuntimeError(
                        f"Incomplete download: expected {remote_info['content_length']} bytes, got {local_size} bytes"
                    )
            self._write_local_meta(remote_info)
            print("Download complete.")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

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
            
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels),
            multimask_output=False # We want the best single mask for contour
        )
        
        # masks shape: (1, H, W) -> return (H, W)
        return masks[0]
