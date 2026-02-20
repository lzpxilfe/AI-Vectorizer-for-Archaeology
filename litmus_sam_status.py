#!/usr/bin/env python3
"""
Lightweight SAM status litmus test for ArchaeoTrace.

Run:
  python litmus_sam_status.py
"""

import json
import os
import platform
import sys
import traceback
import hashlib
import importlib
from datetime import datetime, timezone
from pathlib import Path
import importlib.util


def module_exists(name):
    return importlib.util.find_spec(name) is not None


def safe_import_version(package_name):
    try:
        import importlib.metadata as md
        return md.version(package_name)
    except Exception:
        return None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recommendation(update_status):
    if update_status == "not_installed":
        return "SAM weights are missing. Download is recommended."
    if update_status == "update_available":
        return "A newer SAM weights file appears available. Update is recommended."
    if update_status == "up_to_date":
        return "SAM weights appear up-to-date."
    if update_status == "unknown":
        return "Could not compare exact versions. Re-download if you suspect mismatch."
    if update_status == "check_failed":
        return "Version check failed. Verify internet/proxy/firewall and retry."
    return "No recommendation."


def purge_module_prefix(prefix):
    keys = [k for k in sys.modules.keys() if k == prefix or k.startswith(prefix + ".")]
    for k in keys:
        sys.modules.pop(k, None)


def main():
    repo_root = Path(__file__).resolve().parent
    repo_str = str(repo_root)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    importlib.invalidate_caches()
    purge_module_prefix("ai_vectorizer")

    out = {
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
            "requests": module_exists("requests"),
            "torch": module_exists("torch"),
            "mobile_sam": module_exists("mobile_sam"),
            "sam3": module_exists("sam3"),
            "huggingface_hub": module_exists("huggingface_hub"),
            "yaml": module_exists("yaml"),
            "qgis": module_exists("qgis"),
        },
        "versions": {
            "requests": safe_import_version("requests"),
            "torch": safe_import_version("torch"),
            "mobile_sam": safe_import_version("mobile_sam"),
            "sam3": safe_import_version("sam3"),
            "huggingface_hub": safe_import_version("huggingface_hub"),
            "PyYAML": safe_import_version("PyYAML"),
        },
        "env": {
            "QGIS_PREFIX_PATH": os.environ.get("QGIS_PREFIX_PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
    }

    try:
        sam_mod = importlib.import_module("ai_vectorizer.core.sam_engine")
        SAMEngine = sam_mod.SAMEngine
        required_methods = (
            "get_local_weights_info",
            "get_remote_weights_info",
            "check_weights_update",
        )
        missing = [m for m in required_methods if not hasattr(SAMEngine, m)]
        if missing:
            out["sam_engine"] = {
                "error": "Loaded SAMEngine does not include required litmus methods.",
                "loaded_module_file": getattr(sam_mod, "__file__", None),
                "missing_methods": missing,
                "available_methods_sample": sorted(
                    [n for n in dir(SAMEngine) if not n.startswith("_")]
                )[:50],
            }
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return

        backend = os.environ.get("ARCHAEOTRACE_SAM_BACKEND", "mobile_sam")
        engine = SAMEngine(backend=backend)
        local = engine.get_local_weights_info()
        remote = engine.get_remote_weights_info()
        update = engine.check_weights_update()

        local_file_info = {}
        if local.get("exists"):
            try:
                local_file_info = {
                    "path": engine.weights_path,
                    "size_bytes": os.path.getsize(engine.weights_path),
                    "sha256": file_sha256(engine.weights_path),
                    "meta_path": engine.weights_meta_path,
                    "meta_exists": os.path.exists(engine.weights_meta_path),
                }
            except Exception as e:
                local_file_info = {"error": str(e)}
        else:
            local_file_info = {
                "path": engine.weights_path,
                "exists": False,
                "meta_path": engine.weights_meta_path,
                "meta_exists": os.path.exists(engine.weights_meta_path),
            }

        out["sam_engine"] = {
            "loaded_module_file": getattr(sam_mod, "__file__", None),
            "backend": backend,
            "weights_url": engine.WEIGHTS_DOWNLOAD_URL,
            "download_timeout_sec": engine.DOWNLOAD_TIMEOUT_SECONDS,
            "local": local,
            "remote": remote,
            "update_check": update,
            "local_file_info": local_file_info,
            "recommendation": recommendation(update.get("status")),
        }
    except Exception as e:
        out["sam_engine"] = {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
