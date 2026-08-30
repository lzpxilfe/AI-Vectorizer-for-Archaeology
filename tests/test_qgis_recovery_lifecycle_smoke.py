"""Dependency-free contracts for the optional real-QGIS recovery smoke."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qgis_recovery_lifecycle_smoke.py"


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "qgis_recovery_lifecycle_smoke",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class RecoveryLifecycleSmokeContractTests(unittest.TestCase):
    def test_outer_module_imports_without_qgis_or_onnx_runtime(self):
        module = _load_script()
        self.assertEqual(module.PROFILE_NAME, "archaeotrace-recovery-smoke")
        self.assertTrue(callable(module.run_outer))

    def test_outer_requires_explicit_network_before_any_environment_probe(self):
        module = _load_script()
        arguments = SimpleNamespace(
            allow_network=False,
            timeout=180,
            qgis_executable=Path("/definitely/not/a/qgis/executable"),
        )
        with self.assertRaisesRegex(module.SmokeFailure, "--allow-network"):
            module.run_outer(arguments)

    def test_result_publication_is_atomic_json(self):
        module = _load_script()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "result.json"
            module._atomic_json(target, {"success": True})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8")),
                {"success": True},
            )
            self.assertFalse(target.with_suffix(".json.tmp").exists())

    def test_bootstrap_binds_to_isolated_qgis_model_directory(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("cache = Path(dock._sam_models_dir()).resolve()", source)
        self.assertIn("cache.relative_to(profile_root)", source)
        self.assertIn("Offline missing-model inspection created cache state", source)
        self.assertIn("Inference failure retained a stale recovery request", source)


if __name__ == "__main__":
    unittest.main()
