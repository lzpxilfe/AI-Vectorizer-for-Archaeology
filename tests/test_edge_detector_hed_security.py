import hashlib
import io
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ai_vectorizer.core.edge_detector as edge_detector_module
from ai_vectorizer.core.edge_detector import EdgeDetector


class _FakeResponse:
    def __init__(self, data, url, *, headers=None, status=200):
        self._data = data
        self._offset = 0
        self._url = url
        self.headers = dict(headers or {})
        self.status = status
        self.closed = False

    def geturl(self):
        return self._url

    def getcode(self):
        return self.status

    def read(self, size):
        chunk = self._data[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self):
        self.closed = True


class _CloseFailingResponse(_FakeResponse):
    def close(self):
        super().close()
        raise OSError("injected response cleanup failure")


class HedAssetSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.prototxt = b"pinned prototxt\n"
        self.caffemodel = b"pinned caffemodel bytes"
        self.prototxt_url = "https://raw.githubusercontent.com/test/hed.prototxt"
        self.caffemodel_url = "https://vcl.ucsd.edu/test/hed.caffemodel"
        self.prototxt_path = self.root / "hed_deploy.prototxt"
        self.caffemodel_path = self.root / "hed_pretrained_bsds.caffemodel"
        self.constants = patch.multiple(
            EdgeDetector,
            HED_MODEL_DIR=str(self.root),
            HED_PROTOTXT=str(self.prototxt_path),
            HED_CAFFEMODEL=str(self.caffemodel_path),
            HED_PROTOTXT_URL=self.prototxt_url,
            HED_CAFFEMODEL_URL=self.caffemodel_url,
            HED_PROTOTXT_SIZE_BYTES=len(self.prototxt),
            HED_PROTOTXT_SHA256=hashlib.sha256(self.prototxt).hexdigest(),
            HED_CAFFEMODEL_SIZE_BYTES=len(self.caffemodel),
            HED_CAFFEMODEL_SHA256=hashlib.sha256(self.caffemodel).hexdigest(),
        )
        self.constants.start()
        EdgeDetector._invalidate_hed_status_cache()

    def tearDown(self):
        EdgeDetector._invalidate_hed_status_cache()
        self.constants.stop()
        self.temp_dir.cleanup()

    def _response(self, data, url, **kwargs):
        headers = kwargs.pop("headers", {"Content-Length": str(len(data))})
        return _FakeResponse(data, url, headers=headers, **kwargs)

    def _download_with_responses(self, responses, replace_net=True):
        net_patch = (
            patch.object(EdgeDetector, "_create_hed_net", return_value=object())
            if replace_net
            else patch.object(EdgeDetector, "_create_hed_net", wraps=EdgeDetector._create_hed_net)
        )
        with patch.object(
            EdgeDetector,
            "_open_hed_response",
            side_effect=responses,
        ):
            with net_patch:
                return EdgeDetector.download_hed_assets(timeout=1)

    def test_production_specs_are_pinned_to_exact_artifacts(self):
        # Inspect the class dictionary because setUp temporarily overrides the
        # runtime attributes used by the remaining tests.
        self.constants.stop()
        try:
            info = EdgeDetector.get_hed_download_info()
            self.assertIn("912632b986acc6dd6cc33b95603b2f279d7bd9f2", info["prototxt_url"])
            self.assertNotIn("/master/", info["prototxt_url"])
            self.assertEqual(info["prototxt_size_bytes"], 8_186)
            self.assertEqual(
                info["prototxt_sha256"],
                "378a9246383da889cf8e0290c47554d75dcf9c5b6bbabd8ab6c481c34aa12b8a",
            )
            self.assertEqual(info["caffemodel_size_bytes"], 58_876_104)
            self.assertEqual(
                info["caffemodel_sha256"],
                "4b6937684bce9be1ef5163c78ec812dff9a23653bfbb451925210a64ecfaaac7",
            )
            bundled_prototxt = Path(info["prototxt_path"])
            self.assertEqual(bundled_prototxt.stat().st_size, 8_186)
            self.assertEqual(
                hashlib.sha256(bundled_prototxt.read_bytes()).hexdigest(),
                info["prototxt_sha256"],
            )
        finally:
            self.constants.start()

    def test_exact_pair_is_validated_before_transactional_publish(self):
        responses = [
            self._response(self.prototxt, self.prototxt_url),
            self._response(self.caffemodel, self.caffemodel_url),
        ]

        success, error = self._download_with_responses(responses)

        self.assertTrue(success, error)
        self.assertEqual(self.prototxt_path.read_bytes(), self.prototxt)
        self.assertEqual(self.caffemodel_path.read_bytes(), self.caffemodel)
        self.assertTrue(all(response.closed for response in responses))

    def test_response_cleanup_failure_does_not_discard_verified_asset(self):
        responses = [
            _CloseFailingResponse(
                self.prototxt,
                self.prototxt_url,
                headers={"Content-Length": str(len(self.prototxt))},
            ),
            self._response(self.caffemodel, self.caffemodel_url),
        ]

        success, error = self._download_with_responses(responses)

        self.assertTrue(success, error)
        self.assertTrue(responses[0].closed)
        self.assertEqual(self.prototxt_path.read_bytes(), self.prototxt)
        self.assertEqual(self.caffemodel_path.read_bytes(), self.caffemodel)

    def test_http_error_response_is_closed_before_it_is_re_raised(self):
        body = io.BytesIO(b"not found")
        error = urllib.error.HTTPError(
            self.prototxt_url,
            404,
            "Not Found",
            {},
            body,
        )
        opener = SimpleNamespace(open=lambda request, timeout: (_ for _ in ()).throw(error))

        with patch.object(
            edge_detector_module.urllib.request,
            "build_opener",
            return_value=opener,
        ):
            with self.assertRaises(urllib.error.HTTPError):
                EdgeDetector._open_hed_response(self.prototxt_url, timeout=1)

        self.assertTrue(body.closed)

    def test_verified_legacy_assets_migrate_to_persistent_storage(self):
        legacy = self.root / "legacy"
        persistent = self.root / "profile" / "models"
        legacy.mkdir()
        (legacy / self.prototxt_path.name).write_bytes(self.prototxt)
        (legacy / self.caffemodel_path.name).write_bytes(self.caffemodel)

        EdgeDetector.configure_hed_storage(persistent, legacy)

        self.assertEqual(Path(EdgeDetector.HED_MODEL_DIR), persistent)
        self.assertEqual(Path(EdgeDetector.HED_PROTOTXT).read_bytes(), self.prototxt)
        self.assertEqual(
            Path(EdgeDetector.HED_CAFFEMODEL).read_bytes(),
            self.caffemodel,
        )
        self.assertTrue((legacy / self.prototxt_path.name).exists())
        self.assertTrue((legacy / self.caffemodel_path.name).exists())

    def test_corrupt_legacy_asset_is_not_migrated(self):
        legacy = self.root / "legacy"
        persistent = self.root / "profile" / "models"
        legacy.mkdir()
        (legacy / self.prototxt_path.name).write_bytes(
            b"x" * len(self.prototxt)
        )
        (legacy / self.caffemodel_path.name).write_bytes(self.caffemodel)

        EdgeDetector.configure_hed_storage(persistent, legacy)

        self.assertFalse(Path(EdgeDetector.HED_PROTOTXT).exists())
        self.assertEqual(
            Path(EdgeDetector.HED_CAFFEMODEL).read_bytes(),
            self.caffemodel,
        )

    def test_legacy_migration_rejects_post_verify_path_swap(self):
        legacy = self.root / "legacy"
        persistent = self.root / "profile" / "models"
        legacy.mkdir()
        legacy_prototxt = legacy / self.prototxt_path.name
        legacy_prototxt.write_bytes(self.prototxt)
        untrusted = b"x" * len(self.prototxt)
        real_replace = os.replace

        def swap_staged_path(source, destination):
            source_path = Path(source)
            if (
                Path(destination).name == self.prototxt_path.name
                and source_path.suffix == ".migration"
            ):
                source_path.unlink()
                source_path.write_bytes(untrusted)
            return real_replace(source, destination)

        with patch.object(
            edge_detector_module.os,
            "replace",
            side_effect=swap_staged_path,
        ):
            EdgeDetector.configure_hed_storage(persistent, legacy)

        self.assertFalse(Path(EdgeDetector.HED_PROTOTXT).exists())
        self.assertEqual(legacy_prototxt.read_bytes(), self.prototxt)
        self.assertEqual(list(persistent.glob("*.migration")), [])

    def test_oversized_stream_is_bounded_and_temp_file_is_removed(self):
        response = self._response(
            self.prototxt + b"unexpected",
            self.prototxt_url,
            headers={},
        )

        success, error = self._download_with_responses([response])

        self.assertFalse(success)
        self.assertIn("exceeded", error)
        self.assertTrue(response.closed)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_same_size_wrong_hash_is_rejected(self):
        wrong = b"x" * len(self.prototxt)
        response = self._response(wrong, self.prototxt_url)

        success, error = self._download_with_responses([response])

        self.assertFalse(success)
        self.assertIn("SHA-256", error)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_download_keeps_mkstemp_descriptor_across_path_swap(self):
        victim = self.root / "victim.bin"
        victim.write_bytes(b"do not overwrite")
        descriptor, temp_path = tempfile.mkstemp(dir=self.root)
        try:
            os.remove(temp_path)
            try:
                os.symlink(victim, temp_path)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation is unavailable: {exc}")
            response = self._response(self.prototxt, self.prototxt_url)
            spec = EdgeDetector._hed_asset_specs()[0]

            with patch.object(
                EdgeDetector,
                "_open_hed_response",
                return_value=response,
            ):
                with self.assertRaisesRegex(RuntimeError, "disk verification"):
                    EdgeDetector._download_hed_asset(
                        spec,
                        temp_path,
                        descriptor,
                        timeout=1,
                    )

            self.assertEqual(victim.read_bytes(), b"do not overwrite")
        finally:
            os.close(descriptor)

    def test_final_url_mismatch_is_rejected_before_reading(self):
        response = self._response(
            self.prototxt,
            "https://raw.githubusercontent.com/other/asset.prototxt",
        )

        success, error = self._download_with_responses([response])

        self.assertFalse(success)
        self.assertIn("unexpected final URL", error)
        self.assertEqual(response._offset, 0)
        self.assertTrue(response.closed)

    def test_compressed_response_is_rejected(self):
        response = self._response(
            self.prototxt,
            self.prototxt_url,
            headers={
                "Content-Length": str(len(self.prototxt)),
                "Content-Encoding": "gzip",
            },
        )

        success, error = self._download_with_responses([response])

        self.assertFalse(success)
        self.assertIn("Compressed", error)
        self.assertEqual(response._offset, 0)

    def test_announced_size_mismatch_is_rejected(self):
        response = self._response(
            self.prototxt,
            self.prototxt_url,
            headers={"Content-Length": str(len(self.prototxt) + 1)},
        )

        success, error = self._download_with_responses([response])

        self.assertFalse(success)
        self.assertIn("Content-Length mismatch", error)
        self.assertEqual(response._offset, 0)

    def test_corrupt_local_asset_is_rejected_before_opencv_load(self):
        self.prototxt_path.write_bytes(self.prototxt)
        self.caffemodel_path.write_bytes(b"x" * len(self.caffemodel))

        with patch.object(EdgeDetector, "_require_cv2_runtime") as require_cv2:
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                EdgeDetector._create_hed_net()

        require_cv2.assert_not_called()

    def test_opencv_without_caffe_importer_has_actionable_runtime_error(self):
        fake_cv2 = SimpleNamespace(__version__="5.0.0", dnn=SimpleNamespace())

        with self.assertRaisesRegex(
            RuntimeError,
            r"OpenCV 4\.x.*OpenCV 5 removed the Caffe importer",
        ):
            EdgeDetector._hed_caffe_reader(fake_cv2)

    def test_opencv_loads_the_exact_verified_in_memory_buffers(self):
        self.prototxt_path.write_bytes(self.prototxt)
        self.caffemodel_path.write_bytes(self.caffemodel)
        loaded = {}

        def read_net(prototxt, caffemodel):
            loaded["prototxt"] = bytes(prototxt)
            loaded["caffemodel"] = bytes(caffemodel)
            return object()

        fake_cv2 = SimpleNamespace(
            __version__="4.11.0",
            dnn=SimpleNamespace(readNetFromCaffe=read_net),
        )
        with patch.object(
            EdgeDetector,
            "_require_cv2_runtime",
            return_value=fake_cv2,
        ):
            with patch.object(EdgeDetector, "_register_hed_layers"):
                net = EdgeDetector._create_hed_net()

        self.assertIsNotNone(net)
        self.assertEqual(loaded["prototxt"], self.prototxt)
        self.assertEqual(loaded["caffemodel"], self.caffemodel)

    def test_symlinked_local_asset_is_rejected(self):
        self.prototxt_path.write_bytes(self.prototxt)
        real_model = self.root / "real.caffemodel"
        real_model.write_bytes(self.caffemodel)
        try:
            os.symlink(real_model, self.caffemodel_path)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")

        verified, error = EdgeDetector._verify_hed_asset_pair()

        self.assertFalse(verified)
        self.assertIn("symlinks are rejected", error)

    def test_symlinked_model_directory_is_rejected_before_network_access(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "hed_deploy.prototxt").write_bytes(self.prototxt)
        (outside / "hed_pretrained_bsds.caffemodel").write_bytes(self.caffemodel)
        model_link = self.root / "linked-models"
        try:
            os.symlink(outside, model_link, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")

        with patch.multiple(
            EdgeDetector,
            HED_MODEL_DIR=str(model_link),
            HED_PROTOTXT=str(model_link / "hed_deploy.prototxt"),
            HED_CAFFEMODEL=str(model_link / "hed_pretrained_bsds.caffemodel"),
        ):
            with patch.object(EdgeDetector, "_open_hed_response") as open_response:
                success, error = EdgeDetector.download_hed_assets(timeout=1)
            verified, verification_error = EdgeDetector._verify_hed_asset_pair()

        self.assertFalse(success)
        self.assertIn("symlinks are rejected", error)
        open_response.assert_not_called()
        self.assertFalse(verified)
        self.assertIn("symlinks are rejected", verification_error)
        self.assertEqual(
            (outside / "hed_deploy.prototxt").read_bytes(),
            self.prototxt,
        )

    def test_second_publish_failure_restores_existing_pair(self):
        old_prototxt = b"previous prototxt"
        old_caffemodel = b"previous caffemodel"
        self.prototxt_path.write_bytes(old_prototxt)
        self.caffemodel_path.write_bytes(old_caffemodel)
        responses = [
            self._response(self.prototxt, self.prototxt_url),
            self._response(self.caffemodel, self.caffemodel_url),
        ]
        real_replace = os.replace

        def fail_second_publish(source, destination):
            if (
                str(destination) == str(self.caffemodel_path)
                and Path(source).name.startswith("hed_weights_")
            ):
                raise OSError("injected second publish failure")
            return real_replace(source, destination)

        with patch.object(
            EdgeDetector,
            "_open_hed_response",
            side_effect=responses,
        ):
            with patch.object(
                EdgeDetector,
                "_create_hed_net",
                return_value=object(),
            ):
                with patch.object(
                    edge_detector_module.os,
                    "replace",
                    side_effect=fail_second_publish,
                ):
                    success, error = EdgeDetector.download_hed_assets(timeout=1)

        self.assertFalse(success)
        self.assertIn("injected second publish failure", error)
        self.assertEqual(self.prototxt_path.read_bytes(), old_prototxt)
        self.assertEqual(self.caffemodel_path.read_bytes(), old_caffemodel)
        self.assertEqual(
            sorted(path.name for path in self.root.iterdir()),
            sorted([self.prototxt_path.name, self.caffemodel_path.name]),
        )

    def test_post_publish_verification_failure_restores_existing_pair(self):
        old_prototxt = b"previous prototxt"
        old_caffemodel = b"previous caffemodel"
        self.prototxt_path.write_bytes(old_prototxt)
        self.caffemodel_path.write_bytes(old_caffemodel)
        staged_prototxt = self.root / "staged.prototxt"
        staged_caffemodel = self.root / "staged.caffemodel"
        staged_prototxt.write_bytes(self.prototxt)
        staged_caffemodel.write_bytes(self.caffemodel)
        real_verify = EdgeDetector._verify_hed_asset_file

        def fail_final_caffemodel(path, spec):
            if Path(path) == self.caffemodel_path:
                return False, "injected final verification failure"
            return real_verify(path, spec)

        with patch.object(
            EdgeDetector,
            "_verify_hed_asset_file",
            side_effect=fail_final_caffemodel,
        ):
            with self.assertRaisesRegex(RuntimeError, "final verification failure"):
                EdgeDetector._publish_hed_asset_pair(
                    (str(staged_prototxt), str(staged_caffemodel)),
                    EdgeDetector._hed_asset_specs(),
                    str(self.root),
                )

        self.assertEqual(self.prototxt_path.read_bytes(), old_prototxt)
        self.assertEqual(self.caffemodel_path.read_bytes(), old_caffemodel)

    def test_failed_rollback_preserves_the_last_recovery_copy(self):
        old_prototxt = b"previous prototxt"
        old_caffemodel = b"previous caffemodel"
        self.prototxt_path.write_bytes(old_prototxt)
        self.caffemodel_path.write_bytes(old_caffemodel)
        responses = [
            self._response(self.prototxt, self.prototxt_url),
            self._response(self.caffemodel, self.caffemodel_url),
        ]
        real_replace = os.replace

        def fail_publish_and_first_restore(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                destination_path == self.caffemodel_path
                and source_path.name.startswith("hed_weights_")
            ):
                raise OSError("injected second publish failure")
            if (
                destination_path == self.prototxt_path
                and source_path.name.endswith(".rollback")
            ):
                raise OSError("injected prototxt rollback failure")
            return real_replace(source, destination)

        with patch.object(
            EdgeDetector,
            "_open_hed_response",
            side_effect=responses,
        ):
            with patch.object(
                EdgeDetector,
                "_create_hed_net",
                return_value=object(),
            ):
                with patch.object(
                    edge_detector_module.os,
                    "replace",
                    side_effect=fail_publish_and_first_restore,
                ):
                    success, error = EdgeDetector.download_hed_assets(timeout=1)

        self.assertFalse(success)
        self.assertIn("injected second publish failure", error)
        self.assertIn("injected prototxt rollback failure", error)
        self.assertIn("recovery backups preserved at", error)
        self.assertEqual(self.prototxt_path.read_bytes(), self.prototxt)
        self.assertEqual(self.caffemodel_path.read_bytes(), old_caffemodel)
        recovery_copies = list(
            self.root.glob(f".{self.prototxt_path.name}.*.rollback")
        )
        self.assertEqual(len(recovery_copies), 1)
        self.assertEqual(recovery_copies[0].read_bytes(), old_prototxt)


if __name__ == "__main__":
    unittest.main()
