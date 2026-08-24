#!/usr/bin/env python3
"""Emit a machine-readable status report for the selected SAM backend."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Tuple


EXIT_OK = 0
EXIT_DIAGNOSTIC_ERROR = 1
EXIT_BACKEND_UNAVAILABLE = 2


def module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def safe_import_version(package_name: str) -> Optional[str]:
    try:
        import importlib.metadata as metadata

        return metadata.version(package_name)
    except Exception:
        return None


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recommendation(update_status: Optional[str]) -> str:
    if update_status == "not_installed":
        return "SAM weights are missing. Download is recommended."
    if update_status == "invalid":
        return "Local SAM weights failed pinned integrity verification. Re-download is recommended."
    if update_status == "up_to_date":
        return "SAM weights appear up-to-date."
    if update_status == "unknown":
        return "Could not compare exact versions. Re-download if you suspect mismatch."
    if update_status == "check_failed":
        return "Version check failed. Verify internet/proxy/firewall and retry."
    return "No recommendation."


def resolve_models_dir(explicit: Optional[str]) -> Tuple[Optional[Path], str]:
    """Resolve the same persistent model directory used by the QGIS UI when possible."""
    if explicit:
        return Path(explicit).expanduser().resolve(), "command_line"

    configured = os.environ.get("ARCHAEOTRACE_SAM_MODELS_DIR")
    if configured:
        return Path(configured).expanduser().resolve(), "environment"

    try:
        from qgis.core import QgsApplication

        settings_dir = QgsApplication.qgisSettingsDirPath()
    except Exception:
        settings_dir = ""
    if settings_dir:
        return Path(settings_dir).resolve() / "ai_vectorizer" / "models", "qgis_profile"

    return None, "package_default"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default=os.environ.get("ARCHAEOTRACE_SAM_BACKEND", "mobile_sam"),
        help="SAM backend key (default: ARCHAEOTRACE_SAM_BACKEND or mobile_sam).",
    )
    parser.add_argument(
        "--models-dir",
        help=(
            "Persistent SAM model directory. Defaults to ARCHAEOTRACE_SAM_MODELS_DIR, "
            "then the active QGIS profile, then the engine package fallback."
        ),
    )
    return parser.parse_args(argv)


def _base_report() -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cwd": str(Path.cwd()),
        },
        "modules": {
            "numpy": module_exists("numpy"),
            "requests": module_exists("requests"),
            "torch": module_exists("torch"),
            "mobile_sam": module_exists("mobile_sam"),
            "segment_anything": module_exists("segment_anything"),
            "qgis": module_exists("qgis"),
        },
        "versions": {
            "numpy": safe_import_version("numpy"),
            "requests": safe_import_version("requests"),
            "torch": safe_import_version("torch"),
            "mobile_sam": safe_import_version("mobile-sam"),
            "segment_anything": safe_import_version("segment-anything"),
        },
        "env": {
            "QGIS_PREFIX_PATH": os.environ.get("QGIS_PREFIX_PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "ARCHAEOTRACE_SAM_MODELS_DIR": os.environ.get(
                "ARCHAEOTRACE_SAM_MODELS_DIR"
            ),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent
    repo_str = str(repo_root)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    importlib.invalidate_caches()

    report = _base_report()
    exit_code = EXIT_OK

    try:
        sam_module = importlib.import_module("ai_vectorizer.core.sam_engine")
        engine_class = sam_module.SAMEngine
        required_methods = (
            "get_local_weights_info",
            "get_remote_weights_info",
            "check_weights_update",
        )
        missing = [name for name in required_methods if not hasattr(engine_class, name)]
        if missing:
            report["sam_engine"] = {
                "error": "Loaded SAMEngine does not include required litmus methods.",
                "loaded_module_file": getattr(sam_module, "__file__", None),
                "missing_methods": missing,
            }
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return EXIT_DIAGNOSTIC_ERROR

        models_dir, models_dir_source = resolve_models_dir(args.models_dir)
        engine_kwargs: dict[str, object] = {"backend": args.backend}
        if models_dir is not None:
            engine_kwargs["models_dir"] = models_dir
        engine = engine_class(**engine_kwargs)

        if not isinstance(getattr(engine, "model_spec", None), dict):
            raise RuntimeError("SAMEngine.model_spec is unavailable")
        weights_url = engine.model_spec.get("weights_url")
        if not weights_url:
            raise RuntimeError("SAMEngine.model_spec.weights_url is unavailable")

        # The pinned digest is authoritative. check_weights_update stays offline
        # for a present local artifact and queries remote availability only when
        # the artifact is missing; never query the endpoint a second time here.
        update = engine.check_weights_update()
        local = update.get("local") or engine.get_local_weights_info()
        remote = update.get("remote")
        backend_available = bool(engine_class.is_backend_available(args.backend))

        local_file_info: dict[str, object] = {
            "path": str(engine.weights_path),
            "exists": bool(local.get("exists")),
            "meta_path": str(engine.weights_meta_path),
            "meta_exists": os.path.exists(engine.weights_meta_path),
        }
        if local.get("exists"):
            local_file_info.update(
                {
                    "size_bytes": os.path.getsize(engine.weights_path),
                    "sha256": file_sha256(engine.weights_path),
                }
            )

        report["sam_engine"] = {
            "loaded_module_file": getattr(sam_module, "__file__", None),
            "backend": args.backend,
            "backend_available": backend_available,
            "models_dir": str(engine.models_dir),
            "models_dir_source": models_dir_source,
            "weights_url": weights_url,
            "expected_size_bytes": engine.model_spec.get("weights_size_bytes"),
            "expected_sha256": engine.model_spec.get("weights_sha256"),
            "download_timeout_sec": engine.DOWNLOAD_TIMEOUT_SECONDS,
            "local": local,
            "remote": remote,
            "remote_check_performed": remote is not None,
            "update_check": update,
            "local_file_info": local_file_info,
            "recommendation": recommendation(update.get("status")),
        }
        if not backend_available or not update.get("ok"):
            exit_code = EXIT_BACKEND_UNAVAILABLE
    except Exception as error:
        report["sam_engine"] = {
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        exit_code = EXIT_DIAGNOSTIC_ERROR

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
