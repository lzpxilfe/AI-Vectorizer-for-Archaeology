#!/usr/bin/env python3
"""Exercise Smart Recovery in a disposable real-QGIS process.

The outer process creates an isolated QGIS profile, disables the macOS
password-helper prompt only in that profile, and launches QGIS with this file
as ``--code``. The in-process bootstrap checks missing, installed, and corrupt
model states, runs one real EfficientSAM-Ti CPU inference, and verifies that an
inference error preserves the visible Ink champion.

The two pinned artifacts are downloaded only with ``--allow-network``. Both
the profile and model cache are removed after the run; the user's normal QGIS
profile and model cache are never read or written.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Sequence

PROFILE_NAME = "archaeotrace-recovery-smoke"
BOOTSTRAP_ENV = "ARCHAEOTRACE_RECOVERY_SMOKE_BOOTSTRAP"
PROFILE_ROOT_ENV = "ARCHAEOTRACE_RECOVERY_SMOKE_PROFILE_ROOT"
RESULT_PATH_ENV = "ARCHAEOTRACE_RECOVERY_SMOKE_RESULT"
NETWORK_ENV = "ARCHAEOTRACE_RECOVERY_SMOKE_NETWORK"
DEFAULT_TIMEOUT_SECONDS = 180


class SmokeFailure(RuntimeError):
    """A deterministic recovery lifecycle contract failed."""


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_qgis_bootstrap() -> None:
    """Run inside QgisApp, whose entitlement permits pip native extensions."""

    from qgis.core import Qgis
    from qgis.PyQt.QtCore import QCoreApplication, QTimer

    profile_root = Path(os.environ[PROFILE_ROOT_ENV]).resolve(strict=True)
    result_path = Path(os.environ[RESULT_PATH_ENV])
    try:
        result_path.resolve().relative_to(profile_root)
    except ValueError as exc:
        raise SmokeFailure("Result path escaped the disposable profile root.") from exc

    report = {"qgis_version": Qgis.QGIS_VERSION, "success": False}
    try:
        if os.environ.get(NETWORK_ENV) != "1":
            raise SmokeFailure(
                f"Refusing model download without explicit {NETWORK_ENV}=1."
            )

        import numpy as np
        import onnxruntime
        import qgis.utils

        from ai_vectorizer.core.model_store import (
            ModelIntegrityError,
            fetch_bundle,
            inspect_bundle,
            resolve_bundle,
        )
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool
        from ai_vectorizer.ui.main_dialog import (
            AIVectorizerDock,
            _RecoveryInstallTask,
            _RecoveryPrepareTask,
        )

        report["numpy_version"] = np.__version__
        report["onnxruntime_version"] = onnxruntime.__version__
        dock = AIVectorizerDock(qgis.utils.iface)
        dock.current_language = "en"
        if dock.smart_recovery_check.isChecked():
            raise SmokeFailure("Smart Recovery did not default OFF.")
        if not dock.recovery_status.text().startswith("Ink"):
            raise SmokeFailure("Default recovery state is not Ink.")
        if "Install Recovery Model" not in dock.recovery_install_btn.text():
            raise SmokeFailure("Recovery install action is missing from the dock.")
        dock.smart_recovery_check.blockSignals(True)
        dock.smart_recovery_check.setChecked(True)
        dock.smart_recovery_check.blockSignals(False)
        cache = Path(dock._sam_models_dir()).resolve()
        try:
            cache.relative_to(profile_root)
        except ValueError as exc:
            raise SmokeFailure("Recovery cache escaped the isolated profile.") from exc

        def cache_snapshot():
            if not cache.exists():
                return ()
            return tuple(
                sorted(
                    (
                        path.relative_to(cache).as_posix(),
                        path.is_dir(),
                    )
                    for path in cache.rglob("*")
                )
            )

        before_missing_inspection = cache_snapshot()
        missing_prepare = _RecoveryPrepareTask(
            cache, 1, True, lambda *_args: None
        )
        if not missing_prepare.run():
            raise SmokeFailure(
                f"Missing-model preparation failed: {missing_prepare.error}"
            )
        if missing_prepare.status.ready or missing_prepare.engine is not None:
            raise SmokeFailure("Missing model unexpectedly produced an engine.")
        if cache_snapshot() != before_missing_inspection:
            raise SmokeFailure("Offline missing-model inspection created cache state.")
        report["missing_model"] = {
            "engine_created": False,
            "states": [
                artifact.state for artifact in missing_prepare.status.artifacts
            ],
        }
        dock._recovery_model_status = missing_prepare.status
        dock._recovery_prepare_error = ""
        dock._refresh_recovery_availability()
        if dock.recovery_install_btn.isHidden():
            raise SmokeFailure("Missing model did not expose the install action.")
        if not dock.recovery_status.text().startswith("Ink fallback"):
            raise SmokeFailure("Missing model did not report Ink fallback.")
        report["missing_model"].update(
            {
                "install_button_visible": True,
                "ui_state": dock.recovery_status.text(),
            }
        )

        install = _RecoveryInstallTask(cache, lambda *_args: None)
        if not install.run():
            raise SmokeFailure(f"Model installation failed: {install.error}")
        if install.bundle is None or not inspect_bundle(cache).ready:
            raise SmokeFailure("Installed bundle did not verify as ready.")

        prepare = _RecoveryPrepareTask(cache, 2, True, lambda *_args: None)
        if not prepare.run():
            raise SmokeFailure(f"Model preparation failed: {prepare.error}")
        if not prepare.status.ready or prepare.engine is None:
            raise SmokeFailure("Verified model did not produce a ready engine.")

        engine = prepare.engine
        image = np.full((192, 256, 3), 242, dtype=np.uint8)
        for x in range(16, 240):
            y = round(96 + 28 * np.sin(x / 31.0))
            image[max(0, y - 2) : min(192, y + 3), x] = (30, 38, 48)
        encoding = engine.set_image(image)
        prediction = engine.predict(
            encoding,
            np.asarray([[16.0, 110.0], [239.0, 124.0]], dtype=np.float32),
            np.asarray([1, 1], dtype=np.int64),
        )
        if prediction.mask.shape != (192, 256):
            raise SmokeFailure("Recovery mask changed the source-grid shape.")
        if prediction.mask.dtype != np.bool_:
            raise SmokeFailure("Recovery mask is not boolean.")
        if not np.isfinite(prediction.selected_logits).all():
            raise SmokeFailure("Recovery logits contain non-finite values.")
        report["installed_model"] = {
            "engine_created": True,
            "mask_shape": list(prediction.mask.shape),
            "mask_true_pixels": int(prediction.mask.sum()),
            "providers": engine.metadata["providers"],
            "session_options": engine.metadata["session_options"],
        }
        dock._recovery_model_status = prepare.status
        dock._recovery_prepare_error = ""
        dock.recovery_engine = engine
        dock._refresh_recovery_availability()
        if not dock.recovery_install_btn.isHidden():
            raise SmokeFailure("Ready model left the install action visible.")
        if not dock.recovery_status.text().startswith("Ink"):
            raise SmokeFailure("Ready model did not return the UI to Ink.")
        report["installed_model"].update(
            {
                "install_button_visible": False,
                "ui_state": dock.recovery_status.text(),
            }
        )

        runtime_missing = _RecoveryPrepareTask(
            cache, 3, False, lambda *_args: None
        )
        if not runtime_missing.run() or runtime_missing.engine is not None:
            raise SmokeFailure("Missing runtime did not keep the engine disabled.")
        if not runtime_missing.status.ready:
            raise SmokeFailure("Missing-runtime check lost the verified model state.")
        report["missing_runtime"] = {
            "engine_created": False,
            "model_ready": True,
        }
        dock.recovery_engine = None
        dock._recovery_model_status = runtime_missing.status
        dock._recovery_prepare_error = ""
        dock._recovery_runtime_available = lambda: False
        dock._refresh_recovery_availability()
        if not dock.recovery_status.text().startswith("Ink fallback"):
            raise SmokeFailure("Missing runtime did not report Ink fallback.")
        if dock.recovery_runtime_guide.isHidden():
            raise SmokeFailure("Missing runtime did not expose install guidance.")
        if "onnxruntime" not in dock.recovery_runtime_cmd.text():
            raise SmokeFailure("Missing runtime guidance lost its install command.")
        if not dock.recovery_install_btn.isHidden():
            raise SmokeFailure("Verified model exposed a redundant install action.")
        report["missing_runtime"].update(
            {
                "runtime_guide_visible": True,
                "ui_state": dock.recovery_status.text(),
            }
        )
        del dock._recovery_runtime_available

        # A current ONNX error may update UI state, but it must not mutate the
        # already-visible Ink champion.
        task = type(
            "TaskSnapshot",
            (),
            {
                "request_generation": 8,
                "cache_generation": 3,
                "preview_identity": ("champion",),
                "encoding": encoding,
            },
        )()
        tool = SmartTraceTool.__new__(SmartTraceTool)
        tool._recovery_task = task
        tool._recovery_request = {
            "request_generation": 8,
            "cache_generation": 3,
        }
        tool._recovery_generation = 8
        tool._cache_generation = 3
        tool._recovery_preview_identity = ("champion",)
        tool._pending_livewire_accept_point = None
        tool._pending_livewire_recovery_identity = None
        tool._disposed = False
        tool.is_tracing = True
        tool.smart_recovery_enabled = True
        tool.preview_path = ["immutable Ink champion"]
        emitted = []
        tool._emit_recovery_state = (
            lambda state, detail="": emitted.append((state, detail))
        )
        tool._on_recovery_preview_finished(
            task,
            False,
            None,
            None,
            None,
            RuntimeError("injected ONNX inference failure"),
        )
        if tool.preview_path != ["immutable Ink champion"]:
            raise SmokeFailure("Inference failure replaced the Ink champion.")
        if tool._recovery_request is not None:
            raise SmokeFailure("Inference failure retained a stale recovery request.")
        if not emitted or emitted[-1][0] != "Ink fallback":
            raise SmokeFailure("Inference failure did not report Ink fallback.")
        report["inference_failure"] = {
            "champion_preserved": True,
            "state": emitted[-1][0],
            "detail": emitted[-1][1],
        }

        decoder = install.bundle.path("decoder")
        corrupt_bytes = bytearray(decoder.read_bytes())
        corrupt_bytes[0] ^= 1
        decoder.write_bytes(corrupt_bytes)
        corrupt_prepare = _RecoveryPrepareTask(
            cache, 4, True, lambda *_args: None
        )
        if not corrupt_prepare.run():
            raise SmokeFailure(
                f"Corrupt-model inspection failed: {corrupt_prepare.error}"
            )
        decoder_status = corrupt_prepare.status.artifact("decoder")
        if decoder_status.state != "corrupt":
            raise SmokeFailure("Corrupt decoder did not fail SHA-256 inspection.")
        if corrupt_prepare.engine is not None:
            raise SmokeFailure("Corrupt decoder produced a recovery engine.")
        for label, action in (
            ("resolve", lambda: resolve_bundle(cache)),
            ("fetch", lambda: fetch_bundle(cache)),
        ):
            try:
                action()
            except ModelIntegrityError:
                pass
            else:
                raise SmokeFailure(f"Corrupt decoder unexpectedly passed {label}.")
        report["corrupt_model"] = {
            "engine_created": False,
            "state": decoder_status.state,
            "detail": decoder_status.detail,
            "redownload_attempted": False,
        }
        dock._recovery_model_status = corrupt_prepare.status
        dock._recovery_prepare_error = ""
        dock.recovery_engine = None
        dock._refresh_recovery_availability()
        if dock.recovery_install_btn.isHidden():
            raise SmokeFailure("Corrupt model did not expose the repair action.")
        if "Repair Recovery Model" not in dock.recovery_install_btn.text():
            raise SmokeFailure("Corrupt model exposed a dead install action, not repair.")
        if not dock.recovery_status.text().startswith("Ink fallback"):
            raise SmokeFailure("Corrupt model did not report Ink fallback.")
        if "decoder:corrupt" not in dock.recovery_status.text():
            raise SmokeFailure("Corrupt model state was not identified in the UI.")
        report["corrupt_model"].update(
            {
                "install_button_visible": True,
                "repair_button_visible": True,
                "ui_state": dock.recovery_status.text(),
            }
        )

        repair = _RecoveryInstallTask(
            cache,
            lambda *_args: None,
            repair_corrupt=True,
        )
        if not repair.run():
            raise SmokeFailure(f"Corrupt-model repair failed: {repair.error}")
        if repair.bundle is None or not inspect_bundle(cache).ready:
            raise SmokeFailure("Repaired model bundle did not verify as ready.")
        report["corrupt_model"].update(
            {
                "redownload_attempted": True,
                "repair_verified": True,
            }
        )

        engine.clear_image()
        dock.cleanup(permanent=True)
        dock.deleteLater()
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        _atomic_json(result_path, report)
        QTimer.singleShot(0, QCoreApplication.quit)


def _runtime_site() -> Path:
    try:
        import onnxruntime
    except Exception as exc:
        raise SmokeFailure(
            "The outer smoke requires onnxruntime. Run it in the documented "
            "temporary uv environment."
        ) from exc
    return Path(onnxruntime.__file__).resolve().parent.parent


def _qgis_site(qgis_executable: Path) -> Path:
    candidates = sorted(
        (qgis_executable.parent.parent / "Frameworks" / "lib").glob(
            "python3.*/site-packages"
        )
    )
    if len(candidates) != 1 or not (candidates[0] / "qgis").is_dir():
        raise SmokeFailure("Could not resolve QGIS Python site-packages.")
    return candidates[0].resolve(strict=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qgis-executable",
        type=Path,
        default=Path("/Applications/QGIS.app/Contents/MacOS/QGIS"),
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Explicitly allow the pinned 39.4 MiB model download.",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    return parser.parse_args(argv)


def run_outer(args: argparse.Namespace) -> dict:
    if not args.allow_network:
        raise SmokeFailure("Pass --allow-network to authorize the pinned download.")
    if args.timeout <= 0 or args.timeout > 600:
        raise SmokeFailure("Timeout must be between 1 and 600 seconds.")
    qgis_executable = args.qgis_executable.expanduser().resolve(strict=True)
    if not os.access(qgis_executable, os.X_OK):
        raise SmokeFailure(f"QGIS executable is not runnable: {qgis_executable}")

    runtime_site = _runtime_site()
    qgis_site = _qgis_site(qgis_executable)
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(
        prefix="archaeotrace-recovery-profile-"
    ) as temporary:
        profile_root = Path(temporary).resolve(strict=True)
        settings = (
            profile_root
            / "profiles"
            / PROFILE_NAME
            / "qgis.org"
            / "QGIS3.ini"
        )
        settings.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        settings.write_text(
            "[auth]\nuse_password_helper=false\n", encoding="utf-8"
        )
        result_path = profile_root / "recovery-lifecycle-result.json"
        environment = os.environ.copy()
        python_paths = [qgis_site, runtime_site, source_root]
        prior_python_path = environment.get("PYTHONPATH", "")
        if prior_python_path:
            python_paths.extend(Path(item) for item in prior_python_path.split(os.pathsep))
        environment.update(
            {
                BOOTSTRAP_ENV: "1",
                PROFILE_ROOT_ENV: str(profile_root),
                RESULT_PATH_ENV: str(result_path),
                NETWORK_ENV: "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": os.pathsep.join(map(os.fspath, python_paths)),
                "QT_QPA_PLATFORM": "offscreen",
            }
        )
        command = [
            str(qgis_executable),
            "--nologo",
            "--noversioncheck",
            "--noplugins",
            "--defaultui",
            "--hide-browser",
            "--profiles-path",
            str(profile_root),
            "--profile",
            PROFILE_NAME,
            "--code",
            str(Path(__file__).resolve()),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=profile_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SmokeFailure(
                f"QGIS recovery smoke timed out after {args.timeout}s."
            ) from exc
        if not result_path.is_file():
            raise SmokeFailure(
                "QGIS produced no recovery result.\n"
                f"exit={completed.returncode}\n"
                f"stdout={completed.stdout[-4000:]}\n"
                f"stderr={completed.stderr[-4000:]}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update(
            {
                "isolated_profile_removed_after_run": True,
                "qgis_exit_code": completed.returncode,
            }
        )
        if completed.returncode != 0:
            raise SmokeFailure(
                f"QGIS exited with {completed.returncode}: "
                f"{completed.stderr[-4000:]}"
            )
        if not result.get("success"):
            raise SmokeFailure(
                "Recovery lifecycle bootstrap failed:\n"
                + json.dumps(result, indent=2, sort_keys=True)
            )
        return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run_outer(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if os.environ.get(BOOTSTRAP_ENV) == "1":
    _run_qgis_bootstrap()
elif __name__ == "__main__":
    raise SystemExit(main())
