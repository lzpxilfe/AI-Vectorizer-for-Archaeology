"""Tests for checksummed contour benchmark manifests."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmarks.evidence import (
    PROMPT_EVIDENCE_SCHEMA_VERSION,
    PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
    prompt_sha256,
)
from benchmarks.manifest import ManifestError, load_manifest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "benchmarks" / "data" / "synthetic-smoke"
RUNTIME_TEMPLATE_ROOT = REPOSITORY_ROOT / "benchmarks" / "data" / "runtime-template"
EFFICIENTSAM_TEMPLATE_ROOT = (
    REPOSITORY_ROOT / "benchmarks" / "data" / "efficientsam-runtime-template"
)


class BenchmarkManifestTests(unittest.TestCase):
    def test_loads_the_checksummed_synthetic_fixture(self):
        manifest = load_manifest(FIXTURE_ROOT / "manifest.json")

        self.assertEqual(manifest.dataset.identifier, "synthetic-smoke")
        self.assertEqual([method.identifier for method in manifest.methods], ["perfect", "offset", "broken"])
        self.assertEqual(manifest.metric_config.primary_tolerance_px, 3.0)
        self.assertEqual(manifest.samples[0].prompt.start_xy, (1.0, 4.0))
        self.assertEqual(
            manifest.samples[0].prompt.schema_version,
            PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
        )
        self.assertEqual(manifest.samples[0].source_tile_origin_xy, (0, 0))

    def test_explicit_prompt_v2_is_preserved_without_previous_xy(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self._copy_fixture(folder)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["samples"][0]["prompt"]["schema_version"] = (
                PROMPT_EVIDENCE_SCHEMA_VERSION
            )
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            manifest = load_manifest(manifest_path)

        prompt = manifest.samples[0].prompt
        self.assertEqual(prompt.schema_version, PROMPT_EVIDENCE_SCHEMA_VERSION)
        self.assertIsNone(prompt.previous_xy)
        self.assertNotEqual(
            prompt_sha256(prompt),
            prompt_sha256(
                prompt,
                schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
            ),
        )

    def test_real_v2_sample_requires_explicit_source_tile_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self._copy_fixture(folder)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["methods"][0]["id"] = "ink-livewire-v2"
            prediction = payload["samples"][0]["predictions"].pop("perfect")
            payload["samples"][0]["predictions"]["ink-livewire-v2"] = prediction
            payload["samples"][0]["source"]["url"] = "https://example.test/map"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                ManifestError,
                "source_tile_origin_xy is required",
            ):
                load_manifest(manifest_path)

    def test_loads_the_worker_generation_template(self):
        manifest = load_manifest(RUNTIME_TEMPLATE_ROOT / "manifest.json")

        self.assertEqual(manifest.dataset.identifier, "synthetic-runtime-smoke")
        self.assertEqual(
            [method.identifier for method in manifest.methods],
            ["canny-adaptive-v1", "lsd-adaptive-v1"],
        )
        for prediction in manifest.samples[0].predictions.values():
            self.assertEqual(prediction.execution.status, "failed")
            self.assertIsNone(prediction.artifact_path)
            self.assertFalse(prediction.execution.runtime["provider_verified"])

    def _copy_fixture(self, folder):
        destination = Path(folder) / "fixture"
        shutil.copytree(FIXTURE_ROOT, destination)
        return destination

    def test_rejects_a_changed_dataset_file(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self._copy_fixture(folder)
            image = fixture / "images" / "straight-line.pgm"
            image.write_bytes(image.read_bytes() + b"\n")

            with self.assertRaisesRegex(ManifestError, "checksum mismatch"):
                load_manifest(fixture / "manifest.json")

    def test_rejects_relative_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self._copy_fixture(folder)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["samples"][0]["reference"] = "../../outside.json"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ManifestError, "escapes"):
                load_manifest(manifest_path)

    def test_rejects_canvas_and_image_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self._copy_fixture(folder)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["samples"][0]["width"] = 10
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ManifestError, "not the declared"):
                load_manifest(manifest_path)

    def test_rejects_non_cpu_execution_record(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self._copy_fixture(folder)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["samples"][0]["predictions"]["perfect"]["execution"]["device"] = "cuda"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ManifestError, "device='cpu'"):
                load_manifest(manifest_path)

    def test_rejects_runtime_prompt_hash_that_does_not_match_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = self._copy_fixture(folder)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            runtime = payload["samples"][0]["predictions"]["perfect"][
                "execution"
            ]["runtime"]
            runtime["prompt_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ManifestError, "does not match the sample prompt"):
                load_manifest(manifest_path)

    def test_efficientsam_prompt_limit_counts_start_end_and_rejects_duplicates(self):
        variants = (
            (
                {
                    "positive_xy": [
                        [200, 400],
                        [300, 350],
                        [400, 300],
                        [500, 300],
                        [600, 350],
                    ],
                    "negative_xy": [],
                },
                "including start and end",
            ),
            (
                {"positive_xy": [[140, 512]], "negative_xy": []},
                "must not repeat",
            ),
        )
        for prompt_changes, message in variants:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as folder:
                fixture = Path(folder) / "fixture"
                shutil.copytree(EFFICIENTSAM_TEMPLATE_ROOT, fixture)
                manifest_path = fixture / "manifest.json"
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload["samples"][0]["prompt"].update(prompt_changes)
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ManifestError, message):
                    load_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
