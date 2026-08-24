# -*- coding: utf-8 -*-
"""
Unified SAM engine for MobileSAM and Meta Segment Anything backends.
"""

import gc
import hashlib
import hmac
import importlib.util
import json
import os
import shutil
import stat
import tempfile
import urllib.parse
from contextlib import contextmanager

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
    DOWNLOAD_CHUNK_SIZE = 1024 * 1024
    DOWNLOAD_TIMEOUT_SECONDS = 60
    MAX_METADATA_BYTES = 64 * 1024
    REQUEST_HEADERS = {
        "User-Agent": PLUGIN_NAME,
        "Accept-Encoding": "identity",
    }

    BACKEND_SPECS = {
        SAM_BACKEND_MOBILE: {
            "display_name": "MobileSAM",
            "default_model_type": DEFAULT_MOBILE_SAM_MODEL_TYPE,
            "module_name": "mobile_sam",
            "models": {
                "vit_t": {
                    "weights_filename": "mobile_sam.pt",
                    "weights_url": (
                        "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/"
                        "f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt"
                    ),
                    "weights_size_bytes": 40_728_226,
                    "weights_sha256": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
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
                    "weights_size_bytes": 375_042_383,
                    "weights_sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
                    "size_hint_mb": 358,
                },
                "vit_l": {
                    "weights_filename": "sam_vit_l_0b3195.pth",
                    "weights_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "weights_size_bytes": 1_249_524_607,
                    "weights_sha256": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
                    "size_hint_mb": 1247,
                },
                "vit_h": {
                    "weights_filename": "sam_vit_h_4b8939.pth",
                    "weights_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "weights_size_bytes": 2_564_550_879,
                    "weights_sha256": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
                    "size_hint_mb": 2445,
                },
            },
        },
    }

    def __init__(
        self,
        backend=SAM_BACKEND_MOBILE,
        model_type=None,
        device=None,
        models_dir=None,
        legacy_models_dir=None,
    ):
        """
        Initialize a SAM backend.

        Args:
            backend (str): SAM backend key.
            model_type (str): model family key (e.g. vit_t or vit_b).
            device (str): 'cuda' or 'cpu'. Auto-detect if None.
            models_dir (str|os.PathLike): persistent user model directory.
            legacy_models_dir (str|os.PathLike): old plugin-local directory.
        """
        self.backend = backend
        self.model_type = model_type or self.default_model_type_for_backend(backend)
        self.predictor = None
        self.is_ready = False
        self.model_spec = self._resolve_model_spec(self.backend, self.model_type)
        self.display_name = self.display_name_for_backend(self.backend, self.model_type)

        package_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        self.models_dir = os.path.abspath(os.fspath(models_dir or package_models_dir))
        self.legacy_models_dir = os.path.abspath(
            os.fspath(legacy_models_dir or package_models_dir)
        )
        self.weights_path = os.path.join(
            self.models_dir,
            self.model_spec["weights_filename"],
        )
        self.weights_meta_path = os.path.join(
            self.models_dir,
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
        os.makedirs(self.models_dir, mode=0o700, exist_ok=True)
        try:
            information = os.lstat(self.models_dir)
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect the SAM model directory: {exc}") from exc
        if not stat.S_ISDIR(information.st_mode):
            raise RuntimeError(
                "The SAM model directory must be a real directory; symlinks are rejected."
            )

    def _validated_artifact_spec(self):
        url = self.model_spec.get("weights_url")
        expected_size = self.model_spec.get("weights_size_bytes")
        expected_sha256 = self.model_spec.get("weights_sha256")
        parsed_url = urllib.parse.urlsplit(url or "")
        try:
            port = parsed_url.port
        except ValueError as exc:
            raise ValueError("SAM weights use an invalid source URL port.") from exc
        if (
            parsed_url.scheme != "https"
            or not parsed_url.hostname
            or port not in (None, 443)
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("SAM weights require a fixed HTTPS source URL.")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise ValueError("SAM weights are missing an exact expected byte size.")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("SAM weights are missing a valid expected SHA-256 digest.")
        return url, expected_size, expected_sha256

    @contextmanager
    def _open_verified_weights_file(self, path=None):
        """Yield one verified file descriptor, pinned across checkpoint loading."""
        target_path = os.path.abspath(os.fspath(path or self.weights_path))
        _url, expected_size, expected_sha256 = self._validated_artifact_spec()

        target_directory = os.path.dirname(target_path)
        try:
            directory_information = os.lstat(target_directory)
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect the SAM model directory: {exc}") from exc
        if not stat.S_ISDIR(directory_information.st_mode):
            raise RuntimeError(
                "The SAM model directory must be a real directory; symlinks are rejected."
            )

        try:
            initial = os.lstat(target_path)
        except FileNotFoundError:
            raise RuntimeError(f"Model weights not found at {target_path}")
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect model weights: {exc}") from exc
        if not stat.S_ISREG(initial.st_mode):
            raise RuntimeError(
                "Model weights must be a regular file (symlinks are not accepted)."
            )
        if initial.st_size != expected_size:
            raise RuntimeError(
                f"Model weight size mismatch: expected {expected_size} bytes, "
                f"got {initial.st_size} bytes."
            )

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = None
        checkpoint = None
        try:
            descriptor = os.open(target_path, flags)
        except OSError as exc:
            raise RuntimeError(f"Cannot open model weights safely: {exc}") from exc

        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != initial.st_dev
                or opened.st_ino != initial.st_ino
            ):
                raise RuntimeError(
                    "Model weights changed while integrity verification started."
                )
            if opened.st_size != expected_size:
                raise RuntimeError(
                    f"Model weight size mismatch: expected {expected_size} bytes, "
                    f"got {opened.st_size} bytes."
                )

            checkpoint = os.fdopen(descriptor, "rb", closefd=True)
            descriptor = None
            digest = hashlib.sha256()
            total = 0
            while total <= expected_size:
                chunk = checkpoint.read(
                    min(self.DOWNLOAD_CHUNK_SIZE, expected_size - total + 1)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_size:
                    raise RuntimeError("Model weights exceed the expected byte size.")
                digest.update(chunk)

            finished = os.fstat(checkpoint.fileno())
            if (
                finished.st_size != opened.st_size
                or getattr(finished, "st_mtime_ns", None)
                != getattr(opened, "st_mtime_ns", None)
                or getattr(finished, "st_ctime_ns", None)
                != getattr(opened, "st_ctime_ns", None)
            ):
                raise RuntimeError("Model weights changed during integrity verification.")
            if total != expected_size:
                raise RuntimeError(
                    f"Model weight size mismatch: expected {expected_size} bytes, "
                    f"got {total} bytes."
                )
            if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
                raise RuntimeError(
                    "Model weight SHA-256 mismatch: the checkpoint is corrupt or untrusted."
                )

            checkpoint.seek(0)
            yield checkpoint
        finally:
            if checkpoint is not None:
                checkpoint.close()
            elif descriptor is not None:
                os.close(descriptor)

    def _verify_weights_file(self, path=None):
        """Verify a checkpoint without exposing its file descriptor."""
        try:
            with self._open_verified_weights_file(path):
                return True, None
        except Exception as exc:
            return False, str(exc)

    def _fsync_models_dir(self):
        """Best-effort durability for checkpoint publication and rollback."""
        descriptor = None
        try:
            descriptor = os.open(
                self.models_dir,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _publish_verified_weights(self, staged_path):
        """Replace the checkpoint transactionally and verify the published path."""
        staged_path = os.path.abspath(os.fspath(staged_path))
        if os.path.dirname(staged_path) != self.models_dir:
            raise RuntimeError("Staged SAM weights must be inside the model directory.")

        backup_path = None
        old_checkpoint_moved = False
        published = False
        preserve_backup = False
        try:
            if os.path.lexists(self.weights_path):
                backup_descriptor, backup_path = tempfile.mkstemp(
                    prefix=f".{os.path.basename(self.weights_path)}.",
                    suffix=".rollback",
                    dir=self.models_dir,
                )
                os.close(backup_descriptor)
                os.replace(self.weights_path, backup_path)
                old_checkpoint_moved = True

            os.replace(staged_path, self.weights_path)
            published = True
            verified, verification_error = self._verify_weights_file(self.weights_path)
            if not verified:
                raise RuntimeError(
                    "Published SAM checkpoint failed final integrity verification: "
                    f"{verification_error}"
                )
            self._fsync_models_dir()
        except Exception as publish_error:
            restore_errors = []
            if published and os.path.lexists(self.weights_path):
                try:
                    os.remove(self.weights_path)
                except OSError as exc:
                    restore_errors.append(str(exc))
            if old_checkpoint_moved and backup_path and os.path.lexists(backup_path):
                try:
                    os.replace(backup_path, self.weights_path)
                    backup_path = None
                except OSError as exc:
                    restore_errors.append(str(exc))
            self._fsync_models_dir()
            if restore_errors:
                preserve_backup = bool(
                    backup_path and os.path.lexists(backup_path)
                )
                recovery_detail = (
                    f" Recovery backup preserved at {backup_path}."
                    if preserve_backup
                    else ""
                )
                raise RuntimeError(
                    f"SAM checkpoint publication failed: {publish_error}; "
                    f"rollback failed: {'; '.join(restore_errors)}."
                    f"{recovery_detail}"
                ) from publish_error
            raise
        finally:
            if (
                not preserve_backup
                and backup_path
                and os.path.lexists(backup_path)
            ):
                try:
                    os.remove(backup_path)
                except OSError:
                    pass

    def _legacy_weight_path(self):
        return os.path.join(
            self.legacy_models_dir,
            self.model_spec["weights_filename"],
        )

    def _migrate_legacy_weights(self):
        """Copy an old plugin-local weight into persistent storage once."""
        if os.path.exists(self.weights_path):
            return False

        legacy_path = self._legacy_weight_path()
        if (
            os.path.abspath(legacy_path) == os.path.abspath(self.weights_path)
            or not os.path.isfile(legacy_path)
        ):
            return False

        temp_path = None
        descriptor = None
        try:
            verified, verification_error = self._verify_weights_file(legacy_path)
            if not verified:
                raise RuntimeError(
                    f"Legacy SAM checkpoint failed integrity verification: {verification_error}"
                )
            self._ensure_models_dir()
            descriptor, temp_path = tempfile.mkstemp(
                prefix=f"{os.path.basename(self.weights_path)}.",
                suffix=".migration",
                dir=self.models_dir,
            )
            with self._open_verified_weights_file(legacy_path) as source:
                with os.fdopen(descriptor, "wb", closefd=True) as destination:
                    descriptor = None
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
            verified, verification_error = self._verify_weights_file(temp_path)
            if not verified:
                raise RuntimeError(
                    f"Migrated SAM checkpoint failed integrity verification: {verification_error}"
                )
            self._publish_verified_weights(temp_path)
            temp_path = None
            self._write_local_meta({})
            return True
        except Exception as exc:
            print(f"Failed to migrate legacy SAM weights: {exc}")
            return False
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _import_requests():
        try:
            import requests
            return requests, None
        except Exception as exc:
            return None, str(exc)

    def _read_local_meta(self):
        try:
            initial = os.lstat(self.weights_meta_path)
        except OSError:
            return {}
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_size > self.MAX_METADATA_BYTES
        ):
            return {}

        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = None
        try:
            descriptor = os.open(self.weights_meta_path, flags)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_dev != initial.st_dev
                or opened.st_ino != initial.st_ino
                or opened.st_size > self.MAX_METADATA_BYTES
            ):
                return {}
            with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as handle:
                descriptor = None
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _write_local_meta(self, remote_info):
        _url, expected_size, expected_sha256 = self._validated_artifact_spec()
        meta = {
            "url": self.model_spec["weights_url"],
            "backend": self.backend,
            "model_type": self.model_type,
            "etag": remote_info.get("etag"),
            "last_modified": remote_info.get("last_modified"),
            "content_length": remote_info.get("content_length") or expected_size,
            "verified_size_bytes": expected_size,
            "verified_sha256": expected_sha256,
        }
        descriptor = None
        temp_path = None
        try:
            self._ensure_models_dir()
            descriptor, temp_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(self.weights_meta_path)}.",
                suffix=".tmp",
                dir=self.models_dir,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
                descriptor = None
                json.dump(meta, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.weights_meta_path)
            temp_path = None

            directory_descriptor = None
            try:
                directory_descriptor = os.open(
                    self.models_dir,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                if directory_descriptor is not None:
                    os.close(directory_descriptor)
        except Exception as exc:
            print(f"Failed to write SAM metadata: {exc}")
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temp_path and os.path.lexists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

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

    @staticmethod
    def _close_response(response):
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def _validate_response_source(response, expected_url):
        status_code = int(getattr(response, "status_code", 0))
        if 300 <= status_code < 400:
            raise RuntimeError("SAM checkpoint redirects are not accepted.")
        final_url = getattr(response, "url", expected_url)
        if final_url != expected_url:
            raise RuntimeError("SAM checkpoint response came from an unexpected URL.")

    def get_remote_weights_info(self):
        """Fetch remote metadata for the selected SAM backend weights."""
        try:
            url, expected_size, _expected_sha256 = self._validated_artifact_spec()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        requests, import_error = self._import_requests()
        if requests is None:
            return {"ok": False, "error": f"requests is unavailable: {import_error}"}
        response = None
        try:
            response = requests.head(
                url,
                allow_redirects=False,
                timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            if response.status_code >= 400 or (
                "Content-Length" not in response.headers and "ETag" not in response.headers
            ):
                self._close_response(response)
                response = requests.get(
                    url,
                    stream=True,
                    allow_redirects=False,
                    timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                    headers=self.REQUEST_HEADERS,
                )
            self._validate_response_source(response, url)
            response.raise_for_status()

            info = self._parse_remote_headers(response.headers)
            if (
                info.get("content_length") is not None
                and info["content_length"] != expected_size
            ):
                raise RuntimeError(
                    "Remote SAM checkpoint size does not match the pinned artifact."
                )
            return {"ok": True, **info}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            self._close_response(response)

    def get_local_weights_info(self):
        """Get local weights presence and metadata."""
        self._migrate_legacy_weights()
        exists = os.path.lexists(self.weights_path)
        size = None
        integrity_ok = None
        integrity_error = None
        if exists:
            try:
                information = os.lstat(self.weights_path)
                if stat.S_ISREG(information.st_mode):
                    size = information.st_size
            except OSError as exc:
                integrity_ok = False
                integrity_error = f"Cannot inspect model weights: {exc}"
            else:
                integrity_ok, integrity_error = self._verify_weights_file()
        meta = self._read_local_meta()
        return {
            "exists": exists,
            "size": size,
            "integrity_ok": integrity_ok,
            "integrity_error": integrity_error,
            "backend": meta.get("backend"),
            "model_type": meta.get("model_type"),
            "etag": meta.get("etag"),
            "last_modified": meta.get("last_modified"),
            "content_length": meta.get("content_length"),
            "expected_size_bytes": self.model_spec.get("weights_size_bytes"),
            "expected_sha256": self.model_spec.get("weights_sha256"),
        }

    def check_weights_update(self):
        """
        Check the local file against the immutable pinned artifact identity.

        Remote metadata is consulted only when the artifact is absent, to
        confirm that its configured source is currently available.

        Returns:
            dict with keys:
            - ok (bool)
            - status (str): not_installed|invalid|up_to_date|check_failed
            - message (str)
            - local (dict)
            - remote (dict|None)
        """
        local = self.get_local_weights_info()
        if local["exists"] and not local["integrity_ok"]:
            return {
                "ok": True,
                "status": "invalid",
                "message": (
                    "Local checkpoint failed pinned integrity verification: "
                    f"{local['integrity_error']}"
                ),
                "local": local,
                "remote": None,
            }
        if local["exists"] and local["integrity_ok"]:
            return {
                "ok": True,
                "status": "up_to_date",
                "message": (
                    "Local weights match the pinned size and SHA-256 artifact identity."
                ),
                "local": local,
                "remote": None,
            }
        remote = self.get_remote_weights_info()

        if not remote.get("ok"):
            return {
                "ok": False,
                "status": "check_failed",
                "message": f"Failed to fetch remote metadata: {remote.get('error', 'unknown error')}",
                "local": local,
                "remote": None,
            }

        return {
            "ok": True,
            "status": "not_installed",
            "message": "Local weights file is missing.",
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
        self._migrate_legacy_weights()

        if not os.path.exists(self.weights_path):
            return False, f"Model weights not found at {self.weights_path}"

        if not self.is_backend_available(self.backend):
            return False, f"{self.display_name} library not installed."

        try:
            with self._open_verified_weights_file() as checkpoint:
                try:
                    import torch

                    SamPredictor, sam_model_registry = self._load_predictor()
                except Exception as exc:
                    return False, (
                        f"{self.display_name} dependencies are not ready: {str(exc)}"
                    )
                sam = sam_model_registry[self.model_type](checkpoint=None)
                state_dict = torch.load(
                    checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )
                sam.load_state_dict(state_dict)
            sam.to(device=self.device)
            sam.eval()
            self.predictor = SamPredictor(sam)
            self.is_ready = True
            return True, "Model loaded successfully."
        except Exception as exc:
            return False, f"Error loading model: {str(exc)}"

    def unload_model(self):
        """Release the active predictor so model switching does not stack RAM."""
        self.predictor = None
        self.is_ready = False
        try:
            import torch

            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def download_weights(self):
        """Download the selected SAM backend weights."""
        try:
            url, expected_size, expected_sha256 = self._validated_artifact_spec()
        except Exception as exc:
            print(f"Download failed: {exc}")
            return False
        requests, import_error = self._import_requests()
        if requests is None:
            print(f"Download failed: requests is unavailable: {import_error}")
            return False

        try:
            self._ensure_models_dir()
        except Exception as exc:
            print(f"Download failed: {exc}")
            return False
        temp_path = None
        descriptor = None
        response = None
        try:
            print(f"Downloading {self.display_name} weights from {url}...")
            response = requests.get(
                url,
                stream=True,
                allow_redirects=False,
                timeout=self.DOWNLOAD_TIMEOUT_SECONDS,
                headers=self.REQUEST_HEADERS,
            )
            self._validate_response_source(response, url)
            response.raise_for_status()
            remote_info = self._parse_remote_headers(response.headers)
            content_encoding = response.headers.get("Content-Encoding")
            if content_encoding and content_encoding.lower() != "identity":
                raise RuntimeError("Compressed SAM checkpoint responses are not accepted.")
            if (
                remote_info.get("content_length") is not None
                and remote_info["content_length"] != expected_size
            ):
                raise RuntimeError(
                    f"Unexpected checkpoint size: expected {expected_size} bytes, "
                    f"server announced {remote_info['content_length']} bytes."
                )

            descriptor, temp_path = tempfile.mkstemp(
                prefix=f"{os.path.basename(self.weights_path)}.",
                suffix=".download",
                dir=os.path.dirname(self.weights_path),
            )

            digest = hashlib.sha256()
            total = 0
            with os.fdopen(descriptor, "wb", closefd=True) as f:
                descriptor = None
                for chunk in response.iter_content(chunk_size=self.DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        total += len(chunk)
                        if total > expected_size:
                            raise RuntimeError(
                                f"Checkpoint download exceeded the {expected_size}-byte limit."
                            )
                        digest.update(chunk)
                        f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            if total != expected_size:
                raise RuntimeError(
                    f"Incomplete download: expected {expected_size} bytes, got {total} bytes."
                )
            if not hmac.compare_digest(digest.hexdigest(), expected_sha256):
                raise RuntimeError(
                    "Downloaded checkpoint SHA-256 does not match the pinned artifact."
                )
            verified, verification_error = self._verify_weights_file(temp_path)
            if not verified:
                raise RuntimeError(
                    f"Downloaded checkpoint failed disk verification: {verification_error}"
                )
            self._publish_verified_weights(temp_path)
            temp_path = None
            self._write_local_meta(remote_info)
            print("Download complete.")
            return True
        except Exception as exc:
            print(f"Download failed: {exc}")
            return False
        finally:
            self._close_response(response)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temp_path and os.path.lexists(temp_path):
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
