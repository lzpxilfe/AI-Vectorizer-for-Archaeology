import json
import tempfile
import unittest
from pathlib import Path

from ai_vectorizer.config import SAM_BACKEND_MOBILE
from ai_vectorizer.core.sam_engine import SAMEngine


class SamEngineStorageTests(unittest.TestCase):
    def test_legacy_plugin_weight_is_migrated_to_persistent_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            persistent = root / "profile" / "models"
            legacy = root / "plugin" / "models"
            legacy.mkdir(parents=True)

            legacy_weight = legacy / "mobile_sam.pt"
            legacy_weight.write_bytes(b"legacy model bytes")
            legacy_meta = legacy / "mobile_sam.meta.json"
            legacy_meta.write_text(
                json.dumps({"backend": SAM_BACKEND_MOBILE, "model_type": "vit_t"}),
                encoding="utf-8",
            )

            engine = SAMEngine(
                backend=SAM_BACKEND_MOBILE,
                model_type="vit_t",
                models_dir=persistent,
                legacy_models_dir=legacy,
            )

            info = engine.get_local_weights_info()

            self.assertTrue(info["exists"])
            self.assertEqual(engine.weights_path, str(persistent / "mobile_sam.pt"))
            self.assertEqual(Path(engine.weights_path).read_bytes(), b"legacy model bytes")
            self.assertEqual(
                json.loads(Path(engine.weights_meta_path).read_text(encoding="utf-8"))["model_type"],
                "vit_t",
            )
            self.assertTrue(legacy_weight.exists())

    def test_existing_persistent_weight_is_not_overwritten_by_legacy_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            persistent = root / "profile" / "models"
            legacy = root / "plugin" / "models"
            persistent.mkdir(parents=True)
            legacy.mkdir(parents=True)
            (persistent / "mobile_sam.pt").write_bytes(b"persistent")
            (legacy / "mobile_sam.pt").write_bytes(b"legacy")

            engine = SAMEngine(
                backend=SAM_BACKEND_MOBILE,
                model_type="vit_t",
                models_dir=persistent,
                legacy_models_dir=legacy,
            )

            engine.get_local_weights_info()

            self.assertEqual(
                (persistent / "mobile_sam.pt").read_bytes(),
                b"persistent",
            )


if __name__ == "__main__":
    unittest.main()
