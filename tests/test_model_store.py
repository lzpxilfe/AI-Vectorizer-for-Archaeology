"""Tests for the pinned, offline-first model artifact store."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
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
    MAX_ARTIFACT_BYTES,
    ModelCacheSafetyError,
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
