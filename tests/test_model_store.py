"""Tests for the pinned, offline-first model artifact store."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import urllib.error
from unittest import mock

from ai_vectorizer.core.efficientsam_spec import (
    ArtifactSpec,
    EFFICIENTSAM_SOURCE_COMMIT,
    EFFICIENTSAM_TI_SPLIT,
    ModelBundleSpec,
    bundle_fingerprint,
)
from ai_vectorizer.core import model_store
from ai_vectorizer.core.model_store import (
    DOWNLOAD_CHUNK_BYTES,
    MAX_ARTIFACT_BYTES,
    ModelCacheSafetyError,
    ModelDownloadCancelled,
    ModelDownloadError,
    ModelIntegrityError,
    ModelNotFoundError,
    STATE_CORRUPT,
    STATE_MISSING,
    STATE_READY,
    STATE_UNSAFE,
    fetch_bundle,
    inspect_bundle,
    read_verified_bytes,
    repair_bundle,
    resolve_bundle,
)


TEST_COMMIT = "1" * 40
_AUTO_LENGTH = object()


def _artifact(identifier, payload, *, filename=None, expected_payload=None, size=None, url=None):
    filename = filename or f"{identifier}.onnx"
    expected_payload = payload if expected_payload is None else expected_payload
    digest = hashlib.sha256(expected_payload).hexdigest()
    return ArtifactSpec(
        identifier=identifier,
        filename=filename,
        url=url
        or (
            "https://raw.githubusercontent.com/example/models/"
            f"{TEST_COMMIT}/weights/{filename}"
        ),
        sha256=digest,
        size_bytes=len(expected_payload) if size is None else size,
    )


def _bundle(*artifacts):
    return ModelBundleSpec(
        identifier="test-split-onnx",
        version="v1",
        source_repository="https://github.com/example/models",
        source_commit=TEST_COMMIT,
        license_spdx="Apache-2.0",
        license_url=(
            "https://github.com/example/models/blob/"
            f"{TEST_COMMIT}/LICENSE"
        ),
        artifacts=tuple(artifacts),
    )


def _cache_path(root, artifact):
    return Path(root) / "objects" / "sha256" / artifact.sha256[:2] / artifact.sha256


class FakeResponse:
    def __init__(
        self,
        body,
        *,
        url,
        status=200,
        content_length=_AUTO_LENGTH,
        content_encoding=None,
    ):
        self._stream = io.BytesIO(body)
        self._url = url
        self.status = status
        self.closed = False
        self.headers = {}
        if content_length is _AUTO_LENGTH:
            self.headers["Content-Length"] = str(len(body))
        elif content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding

    def read(self, size=-1):
        return self._stream.read(size)

    def geturl(self):
        return self._url

    def close(self):
        self.closed = True
        self._stream.close()


class CloseFailingResponse(FakeResponse):
    def close(self):
        super().close()
        raise OSError("injected response cleanup failure")


class FakeTransport:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        payload = self.payloads[request.full_url]
        if isinstance(payload, BaseException):
            raise payload
        if callable(payload):
            return payload(request)
        return FakeResponse(payload, url=request.full_url)


class ModelSpecificationTests(unittest.TestCase):
    def test_official_split_contract_is_exact_and_fingerprinted(self):
        self.assertEqual(EFFICIENTSAM_SOURCE_COMMIT, "d525f622e6f640acf5a0fc37c7ca1f243da5bde0")
        self.assertEqual(EFFICIENTSAM_TI_SPLIT.id, EFFICIENTSAM_TI_SPLIT.identifier)
        self.assertEqual(
            [artifact.identifier for artifact in EFFICIENTSAM_TI_SPLIT.artifacts],
            ["encoder", "decoder"],
        )
        self.assertEqual(
            [artifact.id for artifact in EFFICIENTSAM_TI_SPLIT.artifacts],
            ["encoder", "decoder"],
        )
        self.assertEqual(
            [artifact.size_bytes for artifact in EFFICIENTSAM_TI_SPLIT.artifacts],
            [24_799_761, 16_565_728],
        )
        self.assertEqual(
            [artifact.sha256 for artifact in EFFICIENTSAM_TI_SPLIT.artifacts],
            [
                "84ed466ffcc5c1f8d08409bc34a23bb364ab2c15e402cb12d4335a42be0e0951",
                "a62f8fa5ea080447c0689418d69e58f1e83e0b7adf9c142e2bd9bcc8045c0b11",
            ],
        )
        self.assertTrue(
            all(
                f"/{EFFICIENTSAM_SOURCE_COMMIT}/weights/" in artifact.url
                for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
            )
        )
        self.assertEqual(
            bundle_fingerprint(EFFICIENTSAM_TI_SPLIT),
            "f9d4b88041640ca39ca9b484629eb9476fabcd1a15f0cc0b71ab435e12602b8c",
        )
        self.assertEqual(
            model_store.bundle_fingerprint(EFFICIENTSAM_TI_SPLIT),
            bundle_fingerprint(EFFICIENTSAM_TI_SPLIT),
        )

    def test_bundle_fingerprint_binds_artifact_order_and_every_identity_field(self):
        first = _artifact("encoder", b"encoder")
        second = _artifact("decoder", b"decoder")
        original = _bundle(first, second)

        self.assertNotEqual(
            bundle_fingerprint(original),
            bundle_fingerprint(replace(original, artifacts=(second, first))),
        )
        self.assertNotEqual(
            bundle_fingerprint(original),
            bundle_fingerprint(replace(original, version="v2")),
        )


class ModelStoreTests(unittest.TestCase):
    def _pair(self):
        payloads = {"encoder": b"encoder-model", "decoder": b"decoder-model"}
        artifacts = tuple(
            _artifact(identifier, payload)
            for identifier, payload in payloads.items()
        )
        spec = _bundle(*artifacts)
        by_url = {
            artifact.url: payloads[artifact.identifier]
            for artifact in artifacts
        }
        return spec, payloads, by_url

    def assert_no_partial_files(self, root):
        root = Path(root)
        if root.exists():
            self.assertEqual(list(root.rglob("*.partial")), [])

    def test_inspect_and_resolve_are_offline_and_do_not_create_a_cache(self):
        spec, _payloads, _by_url = self._pair()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "not-created"

            status = inspect_bundle(root, spec)

            self.assertFalse(status.ready)
            self.assertEqual(
                [artifact.state for artifact in status.artifacts],
                [STATE_MISSING, STATE_MISSING],
            )
            self.assertEqual(status.artifact("encoder").id, "encoder")
            self.assertFalse(status.as_dict()["ready"])
            json.dumps(status.as_dict(), allow_nan=False)
            self.assertFalse(root.exists())
            with self.assertRaises(ModelNotFoundError):
                resolve_bundle(root, spec)
            self.assertFalse(root.exists())

    def test_windows_reparse_attributes_are_unsafe_for_directories_and_files(self):
        reparse_flag = getattr(
            model_store.stat,
            "FILE_ATTRIBUTE_REPARSE_POINT",
            0x400,
        )
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "directory"
            directory.mkdir()
            real_directory = os.lstat(directory)
            fake_directory = SimpleNamespace(
                st_mode=real_directory.st_mode,
                st_file_attributes=reparse_flag,
            )
            with mock.patch.object(
                model_store.os,
                "lstat",
                return_value=fake_directory,
            ):
                with self.assertRaisesRegex(
                    ModelCacheSafetyError,
                    "reparse point",
                ):
                    model_store._require_safe_directory(
                        directory,
                        "test directory",
                    )

            payload = b"model"
            artifact = _artifact("encoder", payload)
            spec = _bundle(artifact)
            root = Path(folder) / "cache"
            destination = _cache_path(root, artifact)
            destination.parent.mkdir(parents=True)
            destination.write_bytes(payload)
            real_lstat = model_store.os.lstat

            def mark_artifact(path):
                information = real_lstat(path)
                if Path(path) == destination:
                    return SimpleNamespace(
                        st_mode=information.st_mode,
                        st_file_attributes=reparse_flag,
                    )
                return information

            with mock.patch.object(
                model_store.os,
                "lstat",
                side_effect=mark_artifact,
            ), mock.patch.object(model_store.os, "supports_dir_fd", set()):
                self.assertEqual(
                    inspect_bundle(root, spec).artifact("encoder").state,
                    STATE_UNSAFE,
                )

    def test_fetch_publishes_verified_content_addressed_objects_and_reuses_them_offline(self):
        spec, payloads, by_url = self._pair()
        transport = FakeTransport(by_url)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            with mock.patch.object(
                model_store,
                "_publish_no_replace",
                wraps=model_store._publish_no_replace,
            ) as publish, mock.patch.object(
                model_store.os,
                "fsync",
                wraps=model_store.os.fsync,
            ) as fsync:
                verified = fetch_bundle(root, spec, transport=transport)

            self.assertTrue(fsync.called)
            self.assertEqual(publish.call_count, 2)
            for call in publish.call_args_list:
                temporary, destination, _parent = call.args
                self.assertEqual(temporary.parent, destination.parent)
                self.assertFalse(temporary.exists())
            self.assertEqual(len(transport.calls), 2)
            for request, timeout in transport.calls:
                self.assertEqual(request.get_header("Accept-encoding"), "identity")
                self.assertGreater(timeout, 0)
            for artifact in spec.artifacts:
                expected_path = _cache_path(root, artifact)
                self.assertEqual(verified.path(artifact.identifier), expected_path)
                self.assertEqual(verified.read_bytes(artifact.identifier), payloads[artifact.identifier])
                self.assertEqual(read_verified_bytes(root, artifact), payloads[artifact.identifier])
                self.assertEqual(expected_path.name, artifact.sha256)

            no_network = FakeTransport(
                {url: AssertionError("valid cache hit opened the network") for url in by_url}
            )
            reused = fetch_bundle(root, spec, transport=no_network)
            self.assertTrue(inspect_bundle(root, spec).ready)
            self.assertEqual(no_network.calls, [])
            self.assertEqual(reused.path("encoder"), verified.path("encoder"))
            self.assert_no_partial_files(root)

    def test_fetch_honours_cancellation_before_cache_or_network_work(self):
        spec, _payloads, by_url = self._pair()
        transport = FakeTransport(by_url)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"

            with self.assertRaises(ModelDownloadCancelled):
                fetch_bundle(
                    root,
                    spec,
                    transport=transport,
                    cancel_check=lambda: True,
                )

            self.assertEqual(transport.calls, [])
            self.assertFalse(root.exists())

    def test_ready_bundle_is_the_late_cancellation_commit_point(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        for label, corrupt_first, action in (
            ("fetch", False, fetch_bundle),
            ("repair", True, repair_bundle),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "cache"
                destination = _cache_path(root, artifact)
                if corrupt_first:
                    destination.parent.mkdir(parents=True)
                    destination.write_bytes(b"wrong")

                def cancelled_only_after_verified_publish():
                    try:
                        return destination.read_bytes() == payload
                    except FileNotFoundError:
                        return False

                verified = action(
                    root,
                    spec,
                    transport=FakeTransport({artifact.url: payload}),
                    cancel_check=cancelled_only_after_verified_publish,
                )

                self.assertEqual(verified.read_bytes("encoder"), payload)
                self.assertTrue(inspect_bundle(root, spec).ready)

    def test_fetch_honours_midstream_cancellation_and_removes_partial(self):
        payload = b"x" * (DOWNLOAD_CHUNK_BYTES + 17)
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        response = FakeResponse(payload, url=artifact.url)
        transport = FakeTransport({artifact.url: lambda _request: response})

        def cancelled_after_first_read():
            return response._stream.tell() > 0

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ModelDownloadCancelled):
                fetch_bundle(
                    folder,
                    spec,
                    transport=transport,
                    cancel_check=cancelled_after_first_read,
                )

            self.assertTrue(response.closed)
            self.assertFalse(_cache_path(folder, artifact).exists())
            self.assert_no_partial_files(folder)

    def test_fetch_rejects_noncallable_cancellation_probe(self):
        spec, _payloads, by_url = self._pair()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TypeError):
                fetch_bundle(
                    folder,
                    spec,
                    transport=FakeTransport(by_url),
                    cancel_check=True,
                )

    def test_truncated_oversized_and_hash_mismatched_downloads_leave_no_object(self):
        cases = (
            ("truncated", b"abcdef", b"abc", ModelDownloadError),
            ("oversized", b"abc", b"abcd", ModelDownloadError),
            ("wrong-hash", b"abc", b"abd", ModelIntegrityError),
        )
        for label, expected, received, error_type in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                artifact = _artifact("encoder", received, expected_payload=expected)
                spec = _bundle(artifact)

                def response(request, body=received):
                    return FakeResponse(
                        body,
                        url=request.full_url,
                        content_length=None,
                    )

                transport = FakeTransport({artifact.url: response})
                with self.assertRaises(error_type):
                    fetch_bundle(folder, spec, transport=transport)

                self.assertFalse(_cache_path(folder, artifact).exists())
                self.assert_no_partial_files(folder)

    def test_response_cleanup_failure_does_not_discard_verified_download(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)

        def response(request):
            return CloseFailingResponse(payload, url=request.full_url)

        with tempfile.TemporaryDirectory() as folder:
            verified = fetch_bundle(
                folder,
                spec,
                transport=FakeTransport({artifact.url: response}),
            )

            self.assertEqual(verified.read_bytes("encoder"), payload)
            self.assert_no_partial_files(folder)

    def test_default_transport_closes_http_error_response(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        body = io.BytesIO(b"not found")
        error = urllib.error.HTTPError(
            artifact.url,
            404,
            "Not Found",
            {},
            body,
        )

        class FailingOpener:
            def open(self, request, timeout):
                del request, timeout
                raise error

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(
                model_store.urllib.request,
                "build_opener",
                return_value=FailingOpener(),
            ):
                with self.assertRaises(ModelDownloadError):
                    fetch_bundle(folder, spec)

        self.assertTrue(body.closed)

    def test_temporary_unlink_error_does_not_hide_published_verified_object(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        real_unlink = model_store.os.unlink
        failed_once = False

        def fail_first_partial_unlink(path, *args, **kwargs):
            nonlocal failed_once
            if not failed_once and str(path).endswith(".partial"):
                failed_once = True
                raise OSError("injected temporary cleanup failure")
            return real_unlink(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(
                model_store.os,
                "unlink",
                side_effect=fail_first_partial_unlink,
            ):
                verified = fetch_bundle(
                    folder,
                    spec,
                    transport=FakeTransport({artifact.url: payload}),
                )

            self.assertTrue(failed_once)
            self.assertEqual(verified.read_bytes("encoder"), payload)
            self.assert_no_partial_files(folder)

    def test_publication_falls_back_without_opening_directory_on_no_dir_fd_platform(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            temporary = parent / ".model.partial"
            destination = parent / "model"
            temporary.write_bytes(b"verified bytes")
            expected_parent = os.lstat(parent)

            with mock.patch.object(
                model_store.os,
                "supports_dir_fd",
                set(),
            ), mock.patch.object(
                model_store.os,
                "supports_follow_symlinks",
                set(),
            ):
                with mock.patch.object(
                    model_store.os,
                    "open",
                    side_effect=AssertionError(
                        "fallback publication must not open a directory descriptor"
                    ),
                ):
                    model_store._publish_no_replace(
                        temporary,
                        destination,
                        expected_parent,
                    )

            self.assertFalse(temporary.exists())
            self.assertEqual(destination.read_bytes(), b"verified bytes")

    def test_untrusted_response_metadata_is_rejected_before_publish(self):
        expected = b"fixed-model"
        artifact = _artifact("encoder", expected)
        spec = _bundle(artifact)
        cases = (
            (
                "content-length",
                lambda request: FakeResponse(
                    expected,
                    url=request.full_url,
                    content_length=len(expected) + 1,
                ),
            ),
            (
                "content-encoding",
                lambda request: FakeResponse(
                    expected,
                    url=request.full_url,
                    content_encoding="gzip",
                ),
            ),
            (
                "redirect",
                lambda request: FakeResponse(
                    expected,
                    url="https://example.invalid/redirected.onnx",
                ),
            ),
            (
                "status",
                lambda request: FakeResponse(expected, url=request.full_url, status=206),
            ),
        )
        for label, factory in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                transport = FakeTransport({artifact.url: factory})
                with self.assertRaises(ModelDownloadError):
                    fetch_bundle(folder, spec, transport=transport)
                self.assertFalse(_cache_path(folder, artifact).exists())
                self.assert_no_partial_files(folder)

    def test_fixed_https_contract_and_size_limit_are_checked_before_network(self):
        payload = b"model"
        wrong_url = _artifact(
            "encoder",
            payload,
            url="https://example.invalid/model.onnx",
        )
        oversized = _artifact(
            "encoder",
            payload,
            size=MAX_ARTIFACT_BYTES + 1,
        )
        for label, spec in (
            ("wrong-url", _bundle(wrong_url)),
            ("oversized-spec", _bundle(oversized)),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                transport = FakeTransport({})
                with self.assertRaises(ModelDownloadError):
                    fetch_bundle(folder, spec, transport=transport)
                self.assertEqual(transport.calls, [])
                self.assertFalse((Path(folder) / "objects").exists())

    def test_symlinked_cache_components_and_objects_are_never_followed(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            outside = base / "outside"
            outside.mkdir()
            sentinel = outside / "sentinel"
            sentinel.write_bytes(b"unchanged")

            root_link = base / "root-link"
            root_link.symlink_to(outside, target_is_directory=True)
            transport = FakeTransport({artifact.url: payload})
            with self.assertRaises(ModelCacheSafetyError):
                fetch_bundle(root_link, spec, transport=transport)
            self.assertEqual(transport.calls, [])

            root = base / "cache"
            root.mkdir()
            (root / "objects").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ModelCacheSafetyError):
                fetch_bundle(root, spec, transport=transport)
            self.assertEqual(transport.calls, [])
            self.assertEqual(sentinel.read_bytes(), b"unchanged")

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            destination = _cache_path(root, artifact)
            destination.parent.mkdir(parents=True)
            outside = Path(folder) / "outside-model"
            outside.write_bytes(payload)
            destination.symlink_to(outside)

            status = inspect_bundle(root, spec).artifact("encoder")
            self.assertEqual(status.state, STATE_UNSAFE)
            transport = FakeTransport({artifact.url: payload})
            with self.assertRaises(ModelCacheSafetyError):
                fetch_bundle(root, spec, transport=transport)
            self.assertEqual(transport.calls, [])
            self.assertEqual(outside.read_bytes(), payload)

    def test_nonregular_or_corrupt_destination_is_not_overwritten(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        for label, make_destination, expected_state, error_type in (
            (
                "directory",
                lambda path: path.mkdir(),
                STATE_UNSAFE,
                ModelCacheSafetyError,
            ),
            (
                "corrupt-file",
                lambda path: path.write_bytes(b"wrong"),
                STATE_CORRUPT,
                ModelIntegrityError,
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "cache"
                destination = _cache_path(root, artifact)
                destination.parent.mkdir(parents=True)
                make_destination(destination)
                before_is_directory = destination.is_dir()
                before = None if before_is_directory else destination.read_bytes()
                transport = FakeTransport({artifact.url: payload})

                self.assertEqual(
                    inspect_bundle(root, spec).artifact("encoder").state,
                    expected_state,
                )
                with self.assertRaises(error_type):
                    fetch_bundle(root, spec, transport=transport)

                self.assertEqual(transport.calls, [])
                self.assertEqual(destination.is_dir(), before_is_directory)
                if before is not None:
                    self.assertEqual(destination.read_bytes(), before)

    def test_explicit_repair_replaces_only_corrupt_regular_objects(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            destination = _cache_path(root, artifact)
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"wrong")
            transport = FakeTransport({artifact.url: payload})

            verified = repair_bundle(root, spec, transport=transport)

            self.assertEqual(verified.read_bytes("encoder"), payload)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(len(transport.calls), 1)
            self.assertEqual(list(root.rglob("*.corrupt")), [])
            self.assertTrue(inspect_bundle(root, spec).ready)

    def test_explicit_repair_has_a_checked_no_dir_fd_fallback(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            destination = _cache_path(root, artifact)
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"wrong")
            with mock.patch.object(model_store.os, "supports_dir_fd", set()):
                verified = repair_bundle(
                    root,
                    spec,
                    transport=FakeTransport({artifact.url: payload}),
                )

            self.assertEqual(verified.read_bytes("encoder"), payload)
            self.assertEqual(list(root.rglob("*.corrupt")), [])

    def test_quarantine_keeps_exact_link_if_source_disappears_after_link(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            source = parent / "source"
            quarantine = parent / "quarantine"
            payload = b"corrupt-but-recoverable"
            source.write_bytes(payload)
            source_information = os.lstat(source)
            parent_information = model_store._safe_parent_stat(parent)
            real_lstat = model_store._entry_lstat
            source_checks = 0

            def concurrent_remove(parent_arg, name, descriptor):
                nonlocal source_checks
                if name == source.name:
                    source_checks += 1
                    if source_checks == 2:
                        model_store._unlink_entry(
                            parent_arg,
                            name,
                            descriptor,
                        )
                        raise FileNotFoundError(source)
                return real_lstat(parent_arg, name, descriptor)

            with mock.patch.object(
                model_store,
                "_entry_lstat",
                side_effect=concurrent_remove,
            ):
                moved = model_store._move_regular_no_replace(
                    source,
                    quarantine,
                    source_information,
                    parent_information,
                )

            self.assertFalse(source.exists())
            self.assertEqual(quarantine.read_bytes(), payload)
            self.assertTrue(os.path.samestat(moved, os.lstat(quarantine)))

    def test_directory_fsync_failure_does_not_hide_a_committed_quarantine(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            source = parent / "source"
            quarantine = parent / "quarantine"
            payload = b"corrupt-but-recoverable"
            source.write_bytes(payload)
            source_information = os.lstat(source)
            parent_information = model_store._safe_parent_stat(parent)

            with mock.patch.object(
                model_store.os,
                "fsync",
                side_effect=OSError("directory fsync unsupported"),
            ):
                moved = model_store._move_regular_no_replace(
                    source,
                    quarantine,
                    source_information,
                    parent_information,
                )

            self.assertFalse(source.exists())
            self.assertEqual(quarantine.read_bytes(), payload)
            self.assertTrue(os.path.samestat(moved, os.lstat(quarantine)))

    def test_restore_accepts_a_verified_concurrent_winner(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            destination = _cache_path(root, artifact)
            destination.parent.mkdir(parents=True)
            quarantine = destination.parent / ".encoder.race.corrupt"
            quarantine.write_bytes(b"wrong")
            quarantine_information = os.lstat(quarantine)

            def concurrent_winner(*_args, **_kwargs):
                destination.write_bytes(payload)
                raise FileExistsError(destination)

            with mock.patch.object(
                model_store,
                "_move_regular_no_replace",
                side_effect=concurrent_winner,
            ):
                model_store._restore_quarantined_artifacts(
                    root,
                    ((artifact, quarantine, quarantine_information),),
                )

            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(quarantine.exists())

    def test_failed_or_cancelled_repair_restores_the_corrupt_object(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        for label, transport, cancel_check, error_type in (
            (
                "download-failure",
                FakeTransport({artifact.url: OSError("offline")}),
                None,
                ModelDownloadError,
            ),
            (
                "cancel-before-fetch",
                FakeTransport({artifact.url: payload}),
                mock.Mock(side_effect=[False, False, True]),
                ModelDownloadCancelled,
            ),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "cache"
                destination = _cache_path(root, artifact)
                destination.parent.mkdir(parents=True)
                corrupt = b"wrong"
                destination.write_bytes(corrupt)

                with self.assertRaises(error_type):
                    repair_bundle(
                        root,
                        spec,
                        transport=transport,
                        cancel_check=cancel_check,
                    )

                self.assertEqual(destination.read_bytes(), corrupt)
                self.assertEqual(list(root.rglob("*.corrupt")), [])
                self.assertEqual(
                    inspect_bundle(root, spec).artifact("encoder").state,
                    STATE_CORRUPT,
                )

    def test_split_repair_keeps_verified_first_replacement_when_second_fails(self):
        spec, payloads, by_url = self._pair()
        second = spec.artifacts[1]
        for label, injected_error in (
            ("failure", ModelDownloadError("decoder failed")),
            ("cancel", ModelDownloadCancelled("decoder cancelled")),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "cache"
                original = {
                    "encoder": b"wrong-encoder",
                    "decoder": b"wrong-decoder",
                }
                for artifact in spec.artifacts:
                    destination = _cache_path(root, artifact)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(original[artifact.identifier])

                real_download = model_store._download_artifact

                def fail_second(root_arg, artifact_arg, **kwargs):
                    if artifact_arg.identifier == second.identifier:
                        raise injected_error
                    return real_download(root_arg, artifact_arg, **kwargs)

                with mock.patch.object(
                    model_store,
                    "_download_artifact",
                    side_effect=fail_second,
                ):
                    with self.assertRaises(type(injected_error)):
                        repair_bundle(
                            root,
                            spec,
                            transport=FakeTransport(by_url),
                        )

                encoder = spec.artifacts[0]
                self.assertEqual(
                    _cache_path(root, encoder).read_bytes(),
                    payloads[encoder.identifier],
                )
                self.assertEqual(
                    inspect_bundle(root, spec).artifact("encoder").state,
                    STATE_READY,
                )
                self.assertEqual(
                    _cache_path(root, second).read_bytes(),
                    original[second.identifier],
                )
                self.assertEqual(
                    inspect_bundle(root, spec).artifact("decoder").state,
                    STATE_CORRUPT,
                )
                self.assertEqual(list(root.rglob("*.corrupt")), [])

    def test_explicit_repair_never_changes_an_unsafe_object(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            destination = _cache_path(root, artifact)
            destination.parent.mkdir(parents=True)
            destination.mkdir()
            transport = FakeTransport({artifact.url: payload})

            with self.assertRaises(ModelCacheSafetyError):
                repair_bundle(root, spec, transport=transport)

            self.assertTrue(destination.is_dir())
            self.assertEqual(transport.calls, [])

    def test_concurrent_valid_winner_is_reused_but_invalid_winner_is_preserved_and_rejected(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        for label, winner, should_succeed in (
            ("valid", payload, True),
            ("invalid", b"bad", False),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                root = Path(folder) / "cache"
                transport = FakeTransport({artifact.url: payload})

                def race(_temporary, destination, _parent):
                    destination.write_bytes(winner)
                    raise FileExistsError(destination)

                with mock.patch.object(model_store, "_publish_no_replace", side_effect=race):
                    if should_succeed:
                        verified = fetch_bundle(root, spec, transport=transport)
                        self.assertEqual(verified.read_bytes("encoder"), payload)
                    else:
                        with self.assertRaises(ModelIntegrityError):
                            fetch_bundle(root, spec, transport=transport)
                        self.assertEqual(_cache_path(root, artifact).read_bytes(), winner)
                self.assert_no_partial_files(root)

    def test_split_bundle_is_not_resolved_until_every_artifact_is_ready(self):
        spec, payloads, by_url = self._pair()
        decoder = spec.artifacts[1]
        failing = dict(by_url)
        failing[decoder.url] = OSError("controlled decoder failure")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            with self.assertRaises(ModelDownloadError):
                fetch_bundle(root, spec, transport=FakeTransport(failing))

            status = inspect_bundle(root, spec)
            self.assertEqual(status.artifact("encoder").state, STATE_READY)
            self.assertEqual(status.artifact("decoder").state, STATE_MISSING)
            with self.assertRaises(ModelNotFoundError):
                resolve_bundle(root, spec)

            only_decoder = FakeTransport({decoder.url: payloads["decoder"]})
            verified = fetch_bundle(root, spec, transport=only_decoder)
            self.assertTrue(inspect_bundle(root, spec).ready)
            self.assertEqual(len(only_decoder.calls), 1)
            self.assertEqual(only_decoder.calls[0][0].full_url, decoder.url)
            self.assertEqual(verified.read_bytes("encoder"), payloads["encoder"])

    def test_verified_bundle_rechecks_bytes_after_path_replacement(self):
        payload = b"model"
        artifact = _artifact("encoder", payload)
        spec = _bundle(artifact)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            verified = fetch_bundle(
                root,
                spec,
                transport=FakeTransport({artifact.url: payload}),
            )
            destination = verified.path("encoder")
            destination.unlink()
            outside = Path(folder) / "outside"
            outside.write_bytes(payload)
            destination.symlink_to(outside)

            with self.assertRaises(ModelCacheSafetyError):
                verified.read_bytes("encoder")
            self.assertEqual(outside.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
