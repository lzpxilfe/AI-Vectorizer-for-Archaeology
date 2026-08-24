import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import ai_vectorizer.core.sam_engine as sam_engine_module
from ai_vectorizer.config import SAM_BACKEND_MOBILE
from ai_vectorizer.core.sam_engine import SAMEngine


def _configure_test_artifact(engine, payload):
    engine.model_spec = {
        **engine.model_spec,
        "weights_url": "https://example.invalid/mobile_sam.pt",
        "weights_size_bytes": len(payload),
        "weights_sha256": hashlib.sha256(payload).hexdigest(),
    }


class _FakeResponse:
    def __init__(self, url, chunks, headers=None, status_code=200):
        self.url = url
        self._chunks = list(chunks)
        self.headers = dict(headers or {})
        self.status_code = status_code
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        del chunk_size
        yield from self._chunks

    def close(self):
        self.closed = True


class _CloseFailingResponse(_FakeResponse):
    def close(self):
        super().close()
        raise OSError("injected response cleanup failure")


class _FakeRequests:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


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
            _configure_test_artifact(engine, b"legacy model bytes")

            info = engine.get_local_weights_info()

            self.assertTrue(info["exists"])
            self.assertEqual(engine.weights_path, str(persistent / "mobile_sam.pt"))
            self.assertEqual(Path(engine.weights_path).read_bytes(), b"legacy model bytes")
            self.assertEqual(
                json.loads(Path(engine.weights_meta_path).read_text(encoding="utf-8"))["model_type"],
                "vit_t",
            )
            self.assertTrue(legacy_weight.exists())

    def test_unverified_legacy_checkpoint_is_not_migrated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            persistent = root / "profile" / "models"
            legacy = root / "plugin" / "models"
            legacy.mkdir(parents=True)
            (legacy / "mobile_sam.pt").write_bytes(b"untrusted")

            engine = SAMEngine(
                backend=SAM_BACKEND_MOBILE,
                model_type="vit_t",
                models_dir=persistent,
                legacy_models_dir=legacy,
            )

            info = engine.get_local_weights_info()

            self.assertFalse(info["exists"])
            self.assertFalse((persistent / "mobile_sam.pt").exists())

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

    def test_metadata_publish_replaces_symlink_without_overwriting_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            engine = SAMEngine(
                backend=SAM_BACKEND_MOBILE,
                model_type="vit_t",
                models_dir=root / "models",
                legacy_models_dir=root / "legacy",
            )
            Path(engine.models_dir).mkdir(parents=True)
            victim = root / "victim.json"
            victim.write_text('{"keep": true}\n', encoding="utf-8")
            try:
                os.symlink(victim, engine.weights_meta_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")

            engine._write_local_meta({"etag": "test-etag"})

            self.assertEqual(victim.read_text(encoding="utf-8"), '{"keep": true}\n')
            self.assertFalse(Path(engine.weights_meta_path).is_symlink())
            metadata = engine._read_local_meta()
            self.assertEqual(metadata["etag"], "test-etag")
            self.assertEqual(
                metadata["verified_sha256"],
                engine.model_spec["weights_sha256"],
            )

    def test_metadata_reader_rejects_links_oversized_files_and_non_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            engine = SAMEngine(
                backend=SAM_BACKEND_MOBILE,
                model_type="vit_t",
                models_dir=root / "models",
                legacy_models_dir=root / "legacy",
            )
            meta_path = Path(engine.weights_meta_path)
            meta_path.parent.mkdir(parents=True)

            meta_path.write_text("[]\n", encoding="utf-8")
            self.assertEqual(engine._read_local_meta(), {})

            meta_path.write_bytes(b"x" * (engine.MAX_METADATA_BYTES + 1))
            self.assertEqual(engine._read_local_meta(), {})

            meta_path.unlink()
            victim = root / "metadata-victim.json"
            victim.write_text('{"trusted": false}\n', encoding="utf-8")
            try:
                os.symlink(victim, meta_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            self.assertEqual(engine._read_local_meta(), {})


class SamEngineIntegrityTests(unittest.TestCase):
    def _engine(self, root):
        return SAMEngine(
            backend=SAM_BACKEND_MOBILE,
            model_type="vit_t",
            models_dir=Path(root) / "models",
            legacy_models_dir=Path(root) / "legacy",
        )

    def test_release_model_specs_have_complete_pinned_integrity(self):
        for backend_spec in SAMEngine.BACKEND_SPECS.values():
            for model_spec in backend_spec["models"].values():
                self.assertTrue(model_spec["weights_url"].startswith("https://"))
                self.assertGreater(model_spec["weights_size_bytes"], 0)
                self.assertRegex(model_spec["weights_sha256"], r"^[0-9a-f]{64}$")
        mobile_url = SAMEngine.BACKEND_SPECS[SAM_BACKEND_MOBILE]["models"]["vit_t"][
            "weights_url"
        ]
        self.assertIn("f706ad9c4eb7f219c00d9050e46328518ffb65d2", mobile_url)
        self.assertNotIn("/master/", mobile_url)

    def test_download_accepts_only_the_exact_verified_artifact(self):
        payload = b"small verified test checkpoint"
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, payload)
            response = _FakeResponse(
                engine.model_spec["weights_url"],
                (payload[:7], payload[7:]),
                {"Content-Length": str(len(payload))},
            )
            requests = _FakeRequests(response)

            with patch.object(engine, "_import_requests", return_value=(requests, None)):
                downloaded = engine.download_weights()

            self.assertTrue(downloaded)
            self.assertTrue(response.closed)
            self.assertEqual(Path(engine.weights_path).read_bytes(), payload)
            verified, error = engine._verify_weights_file()
            self.assertTrue(verified, error)
            meta = json.loads(Path(engine.weights_meta_path).read_text(encoding="utf-8"))
            self.assertEqual(meta["verified_sha256"], hashlib.sha256(payload).hexdigest())

    def test_response_cleanup_failure_does_not_change_successful_download_result(self):
        payload = b"small verified test checkpoint"
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, payload)
            response = _CloseFailingResponse(
                engine.model_spec["weights_url"],
                (payload,),
                {"Content-Length": str(len(payload))},
            )
            requests = _FakeRequests(response)

            with patch.object(engine, "_import_requests", return_value=(requests, None)):
                downloaded = engine.download_weights()

            self.assertTrue(downloaded)
            self.assertTrue(response.closed)
            self.assertEqual(Path(engine.weights_path).read_bytes(), payload)

    def test_oversized_download_is_stopped_without_replacing_existing_weights(self):
        payload = b"expected"
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, payload)
            Path(engine.models_dir).mkdir(parents=True)
            Path(engine.weights_path).write_bytes(b"previous")
            response = _FakeResponse(
                engine.model_spec["weights_url"],
                (payload, b"unexpected trailing bytes"),
            )
            requests = _FakeRequests(response)

            with patch.object(engine, "_import_requests", return_value=(requests, None)):
                downloaded = engine.download_weights()

            self.assertFalse(downloaded)
            self.assertTrue(response.closed)
            self.assertEqual(Path(engine.weights_path).read_bytes(), b"previous")
            self.assertEqual(list(Path(engine.models_dir).glob("*.download")), [])

    def test_download_keeps_mkstemp_descriptor_across_path_swap(self):
        payload = b"small verified test checkpoint"
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, payload)
            response = _FakeResponse(
                engine.model_spec["weights_url"],
                (payload,),
                {"Content-Length": str(len(payload))},
            )
            requests = _FakeRequests(response)
            victim = Path(folder) / "victim.bin"
            victim.write_bytes(b"do not overwrite")
            real_mkstemp = tempfile.mkstemp

            def swapped_mkstemp(*args, **kwargs):
                descriptor, path = real_mkstemp(*args, **kwargs)
                os.remove(path)
                try:
                    os.symlink(victim, path)
                except (OSError, NotImplementedError) as exc:
                    os.close(descriptor)
                    self.skipTest(f"symlink creation is unavailable: {exc}")
                return descriptor, path

            with patch.object(
                engine,
                "_import_requests",
                return_value=(requests, None),
            ):
                with patch.object(
                    sam_engine_module.tempfile,
                    "mkstemp",
                    side_effect=swapped_mkstemp,
                ):
                    downloaded = engine.download_weights()

            self.assertFalse(downloaded)
            self.assertEqual(victim.read_bytes(), b"do not overwrite")

    def test_download_rejects_post_verify_swap_and_restores_previous_checkpoint(self):
        trusted = b"trusted1"
        untrusted = b"untrust1"
        previous = b"previous"
        self.assertEqual(len(trusted), len(untrusted))
        self.assertEqual(len(trusted), len(previous))
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, trusted)
            Path(engine.models_dir).mkdir(parents=True)
            Path(engine.weights_path).write_bytes(previous)
            response = _FakeResponse(
                engine.model_spec["weights_url"],
                (trusted,),
                {"Content-Length": str(len(trusted))},
            )
            requests = _FakeRequests(response)
            real_replace = os.replace

            def swap_staged_path(source, destination):
                source_path = Path(source)
                if (
                    Path(destination) == Path(engine.weights_path)
                    and source_path.suffix == ".download"
                ):
                    source_path.unlink()
                    source_path.write_bytes(untrusted)
                return real_replace(source, destination)

            with patch.object(engine, "_import_requests", return_value=(requests, None)):
                with patch.object(
                    sam_engine_module.os,
                    "replace",
                    side_effect=swap_staged_path,
                ):
                    downloaded = engine.download_weights()

            self.assertFalse(downloaded)
            self.assertTrue(response.closed)
            self.assertEqual(Path(engine.weights_path).read_bytes(), previous)
            self.assertEqual(list(Path(engine.models_dir).glob("*.download")), [])
            self.assertEqual(list(Path(engine.models_dir).glob("*.rollback")), [])

    def test_legacy_migration_rejects_post_verify_path_swap(self):
        trusted = b"trusted1"
        untrusted = b"untrust1"
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, trusted)
            legacy_path = Path(engine.legacy_models_dir) / "mobile_sam.pt"
            legacy_path.parent.mkdir(parents=True)
            legacy_path.write_bytes(trusted)
            real_replace = os.replace

            def swap_staged_path(source, destination):
                source_path = Path(source)
                if (
                    Path(destination) == Path(engine.weights_path)
                    and source_path.suffix == ".migration"
                ):
                    source_path.unlink()
                    source_path.write_bytes(untrusted)
                return real_replace(source, destination)

            with patch.object(
                sam_engine_module.os,
                "replace",
                side_effect=swap_staged_path,
            ):
                migrated = engine._migrate_legacy_weights()

            self.assertFalse(migrated)
            self.assertFalse(Path(engine.weights_path).exists())
            self.assertEqual(legacy_path.read_bytes(), trusted)
            self.assertEqual(list(Path(engine.models_dir).glob("*.migration")), [])

    def test_symlinked_model_directory_is_rejected_before_network_access(self):
        payload = b"trusted1"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outside = root / "outside"
            outside.mkdir()
            outside_checkpoint = outside / "mobile_sam.pt"
            outside_checkpoint.write_bytes(payload)
            models_link = root / "models"
            try:
                os.symlink(outside, models_link, target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")
            engine = SAMEngine(
                backend=SAM_BACKEND_MOBILE,
                model_type="vit_t",
                models_dir=models_link,
                legacy_models_dir=root / "legacy",
            )
            _configure_test_artifact(engine, payload)
            response = _FakeResponse(
                engine.model_spec["weights_url"],
                (payload,),
                {"Content-Length": str(len(payload))},
            )
            requests = _FakeRequests(response)

            with patch.object(engine, "_import_requests", return_value=(requests, None)):
                downloaded = engine.download_weights()

            self.assertFalse(downloaded)
            self.assertEqual(requests.calls, [])
            self.assertEqual(outside_checkpoint.read_bytes(), payload)

            with patch.object(engine, "get_remote_weights_info") as remote_info:
                update = engine.check_weights_update()

            remote_info.assert_not_called()
            self.assertEqual(update["status"], "invalid")
            self.assertIn("symlinks are rejected", update["local"]["integrity_error"])

    def test_same_size_wrong_hash_is_rejected_before_backend_load(self):
        trusted = b"trusted1"
        untrusted = b"untrust1"
        self.assertEqual(len(trusted), len(untrusted))
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, trusted)
            Path(engine.models_dir).mkdir(parents=True)
            Path(engine.weights_path).write_bytes(untrusted)

            with patch.object(
                SAMEngine,
                "is_backend_available",
                return_value=True,
            ), patch.object(engine, "_load_predictor") as load_predictor:
                loaded, message = engine.load_model()

            self.assertFalse(loaded)
            self.assertIn("SHA-256 mismatch", message)
            load_predictor.assert_not_called()

    def test_corrupt_local_checkpoint_is_reported_without_remote_request(self):
        trusted = b"trusted1"
        untrusted = b"untrust1"
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, trusted)
            Path(engine.models_dir).mkdir(parents=True)
            Path(engine.weights_path).write_bytes(untrusted)

            with patch.object(engine, "get_remote_weights_info") as remote_info:
                update = engine.check_weights_update()

            remote_info.assert_not_called()
            self.assertEqual(update["status"], "invalid")
            self.assertFalse(update["local"]["integrity_ok"])
            self.assertIn("SHA-256 mismatch", update["local"]["integrity_error"])

    def test_broken_symlink_checkpoint_is_invalid_without_remote_request(self):
        payload = b"trusted1"
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, payload)
            Path(engine.models_dir).mkdir(parents=True)
            try:
                os.symlink(Path(folder) / "missing.pt", engine.weights_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with patch.object(engine, "get_remote_weights_info") as remote_info:
                update = engine.check_weights_update()

            remote_info.assert_not_called()
            self.assertEqual(update["status"], "invalid")
            self.assertTrue(update["local"]["exists"])
            self.assertFalse(update["local"]["integrity_ok"])
            self.assertIn("regular file", update["local"]["integrity_error"])

    def test_verified_pinned_checkpoint_is_current_without_remote_metadata(self):
        payload = b"trusted checkpoint"
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, payload)
            Path(engine.models_dir).mkdir(parents=True)
            Path(engine.weights_path).write_bytes(payload)
            Path(engine.weights_meta_path).write_text(
                json.dumps(
                    {
                        "etag": "stale transport tag",
                        "last_modified": "yesterday",
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(engine, "get_remote_weights_info") as remote_info:
                update = engine.check_weights_update()

            remote_info.assert_not_called()
            self.assertTrue(update["ok"])
            self.assertEqual(update["status"], "up_to_date")
            self.assertIsNone(update["remote"])
            self.assertTrue(update["local"]["integrity_ok"])

    def test_path_replacement_after_verification_loads_the_pinned_file_descriptor(self):
        trusted = b"trusted checkpoint bytes"
        untrusted = b"untrusted replacement!!!"
        self.assertEqual(len(trusted), len(untrusted))
        with tempfile.TemporaryDirectory() as folder:
            engine = self._engine(folder)
            _configure_test_artifact(engine, trusted)
            Path(engine.models_dir).mkdir(parents=True)
            Path(engine.weights_path).write_bytes(trusted)
            replacement = Path(folder) / "replacement.pt"
            replacement.write_bytes(untrusted)

            loaded_payloads = []
            load_kwargs = []
            fake_torch = types.ModuleType("torch")

            def fake_torch_load(checkpoint, **kwargs):
                loaded_payloads.append(checkpoint.read())
                load_kwargs.append(kwargs)
                return {"verified": True}

            fake_torch.load = fake_torch_load

            class FakeSam:
                def __init__(self):
                    self.state_dict = None

                def load_state_dict(self, state_dict):
                    self.state_dict = state_dict

                def to(self, device):
                    self.device = device

                def eval(self):
                    self.evaluated = True

            sam = FakeSam()
            registry_checkpoints = []

            def build_sam(checkpoint):
                registry_checkpoints.append(checkpoint)
                os.replace(replacement, engine.weights_path)
                return sam

            registry = {engine.model_type: build_sam}
            predictor = object()
            with patch.object(
                SAMEngine,
                "is_backend_available",
                return_value=True,
            ), patch.object(
                engine,
                "_load_predictor",
                return_value=(lambda _sam: predictor, registry),
            ), patch.dict(sys.modules, {"torch": fake_torch}):
                loaded, message = engine.load_model()

            self.assertTrue(loaded, message)
            self.assertEqual(registry_checkpoints, [None])
            self.assertEqual(loaded_payloads, [trusted])
            self.assertEqual(load_kwargs, [{"map_location": "cpu", "weights_only": True}])
            self.assertEqual(Path(engine.weights_path).read_bytes(), untrusted)
            self.assertEqual(sam.state_dict, {"verified": True})
            self.assertIs(engine.predictor, predictor)


if __name__ == "__main__":
    unittest.main()
