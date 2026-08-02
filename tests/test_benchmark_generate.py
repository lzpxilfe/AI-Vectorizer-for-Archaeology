"""Tests for atomic manifest generation with isolated fake workers."""

import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from benchmarks import cli
from benchmarks.generate import (
    GenerationError,
    _rename_no_replace,
    generate_benchmark_dataset,
)
from benchmarks.manifest import load_manifest
from benchmarks.runner import validate_benchmark


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "benchmarks" / "data" / "synthetic-smoke"
METHOD_IDS = ("canny-adaptive-v1", "lsd-adaptive-v1")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_worker_template(folder):
    """Turn the existing valid smoke dataset into a valid worker template."""

    root = Path(folder) / "template"
    shutil.copytree(FIXTURE_ROOT, root)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_methods = payload["methods"][:2]
    original_predictions = payload["samples"][0]["predictions"]
    payload["methods"] = []
    replacements = {}
    for method_id, source_method, prediction_key in zip(
        METHOD_IDS,
        original_methods,
        ("perfect", "offset"),
    ):
        method = copy.deepcopy(source_method)
        method.update(
            id=method_id,
            label=f"Fake template for {method_id}",
            configuration={"edge_weight": 0.5},
        )
        payload["methods"].append(method)
        prediction = copy.deepcopy(original_predictions[prediction_key])
        execution = prediction["execution"]
        execution["requested_backend"] = method_id
        execution["actual_backend"] = method_id
        replacements[method_id] = prediction
    payload["samples"][0]["predictions"] = replacements
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_benchmark(manifest_path)
    return manifest_path


def _write_fake_worker(
    folder,
    *,
    fail_backend=None,
    malformed=False,
    wrong_input_hash=False,
    wrong_source_hash=False,
    leave_artifact_on_failure=False,
):
    path = Path(folder) / "fake_worker.py"
    source = f'''
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

request_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
root = request_path.parent
request = json.loads(request_path.read_text(encoding="utf-8"))
prompt_document = {{
    "schema_version": "archaeotrace-trace-prompt/1",
    "start_xy": [float(value) for value in request["prompt"]["start_xy"]],
    "end_xy": [float(value) for value in request["prompt"]["end_xy"]],
    "positive_xy": [
        [float(value) for value in point]
        for point in request["prompt"].get("positive_xy", [])
    ],
    "negative_xy": [
        [float(value) for value in point]
        for point in request["prompt"].get("negative_xy", [])
    ],
}}
prompt_digest = hashlib.sha256(
    json.dumps(
        prompt_document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
with (root / "worker-pids.log").open("a", encoding="utf-8") as handle:
    handle.write(str(os.getpid()) + "\\n")

if {malformed!r}:
    result_path.write_text("{{}}", encoding="utf-8")
    raise SystemExit(0)

backend = request["requested_backend"]
runtime = {{
    "adapter_version": "fake-isolated-worker/1",
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "cpu": platform.processor() or platform.machine() or "unknown",
    "actual_provider": "OpenCV CPU",
    "provider_kind": "opencv",
    "provider_device_type": "cpu",
    "provider_verified": True,
    "package_versions": {{"numpy": "fake", "opencv": "fake"}},
    "source_files_sha256": {{
        "fake_worker.py": (
            "0" * 64
            if {wrong_source_hash!r}
            else hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ),
    }},
    "thread_settings": {{
        "threads": request["threads"],
        "opencv_set_num_threads": 0,
        "opencl": False,
        "fake_worker": True,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    }},
    "input_sha256": (
        "0" * 64 if {wrong_input_hash!r} else request["image"]["sha256"]
    ),
    "configuration_sha256": hashlib.sha256(
        json.dumps(
            request["configuration"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest(),
    "prompt_sha256": prompt_digest,
    "latency_scope": "warmed_predict_plus_canonical_artifact_v1",
}}

if backend == {fail_backend!r}:
    if {leave_artifact_on_failure!r}:
        abandoned_path = root / request["artifact"]
        abandoned_path.parent.mkdir(parents=True, exist_ok=True)
        abandoned_path.write_text("abandoned", encoding="utf-8")
    runtime.update(
        deterministic=None,
        output_sha256_samples=[],
        image_load_wall_ns=None,
        image_decode_wall_ns=None,
        warmup_wall_ns_samples=[],
    )
    prediction = {{
        "execution": {{
            "status": "failed",
            "requested_backend": backend,
            "actual_backend": None,
            "fallback_reason": None,
            "error": "FakeDependencyError: controlled missing backend",
            "device": "cpu",
            "runtime": runtime,
            "timing": {{
                "warmup_runs": 0,
                "wall_ns_samples": [],
                "cpu_ns_samples": [],
                "model_load_wall_ns": 1,
                "peak_rss_bytes": 1024,
            }},
        }}
    }}
    return_code = 3
else:
    artifact = {{
        "schema_version": "archaeotrace-centerline/1",
        "coordinate_space": "pixel_xy",
        "image_size": {{
            "width": request["image"]["width"],
            "height": request["image"]["height"],
        }},
        "paths": [{{
            "id": "trace-0",
            "closed": False,
            "points": [
                request["prompt"]["start_xy"],
                request["prompt"]["end_xy"],
            ],
        }}],
        "metadata": {{
            "fake_worker": True,
            "actual_backend": backend,
            "configuration_sha256": runtime["configuration_sha256"],
            "input_sha256": request["image"]["sha256"],
            "prompt_sha256": prompt_digest,
            "requested_backend": backend,
            "smoothing": "smart-trace-v1-historical",
            "trace_kernel": "ai_vectorizer.core.trace_kernel",
        }},
    }}
    raw = (
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\\n"
    ).encode("utf-8")
    artifact_path = root / request["artifact"]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    repetitions = request["measurement_runs"]
    runtime.update(
        deterministic=True,
        output_sha256_samples=[digest] * repetitions,
        image_load_wall_ns=1,
        image_decode_wall_ns=1,
        warmup_wall_ns_samples=[50] * request["warmup_runs"],
    )
    prediction = {{
        "artifact": request["artifact"],
        "sha256": digest,
        "execution": {{
            "status": "ok",
            "requested_backend": backend,
            "actual_backend": backend,
            "fallback_reason": None,
            "error": None,
            "device": "cpu",
            "runtime": runtime,
            "timing": {{
                "warmup_runs": request["warmup_runs"],
                "wall_ns_samples": list(range(100, 100 + repetitions)),
                "cpu_ns_samples": list(range(90, 90 + repetitions)),
                "model_load_wall_ns": 1,
                "peak_rss_bytes": 1024,
            }},
        }},
    }}
    return_code = 0

result = {{
    "schema_version": "archaeotrace-worker-result/1",
    "request_id": request["request_id"],
    "prediction": prediction,
}}
result_path.write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
raise SystemExit(return_code)
'''
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


class BenchmarkGenerationTests(unittest.TestCase):
    def test_cli_dispatches_generate_with_the_isolated_defaults(self):
        template = REPOSITORY_ROOT / "benchmarks" / "data" / "runtime-template" / "manifest.json"
        manifest = load_manifest(template)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated"
            stdout = io.StringIO()
            with mock.patch.object(
                cli,
                "generate_benchmark_dataset",
                return_value=manifest,
            ) as generate, mock.patch("sys.stdout", stdout):
                return_code = cli.main(
                    [
                        "generate",
                        str(template),
                        "--output",
                        str(output),
                        "--python-executable",
                        "/controlled/python",
                    ]
                )

        self.assertEqual(return_code, 0)
        generate.assert_called_once_with(
            template,
            output,
            python_executable="/controlled/python",
            warmup_runs=1,
            measurement_runs=3,
            threads=1,
            timeout_seconds=600.0,
        )
        self.assertIn("GENERATED dataset=synthetic-runtime-smoke@1.0.0", stdout.getvalue())

    def test_generates_valid_dataset_in_one_fresh_process_per_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            template_bytes = template_path.read_bytes()
            template_image_hash = _sha256(template_path.parent / "images" / "straight-line.pgm")
            fake_worker = _write_fake_worker(folder)
            output = Path(folder) / "generated"

            generated = generate_benchmark_dataset(
                template_path,
                output,
                worker_command=(sys.executable, str(fake_worker)),
            )

            self.assertEqual(generated.path, (output / "manifest.json").resolve())
            self.assertEqual(template_path.read_bytes(), template_bytes)
            self.assertEqual(
                _sha256(template_path.parent / "images" / "straight-line.pgm"),
                template_image_hash,
            )
            reloaded = load_manifest(output / "manifest.json")
            validate_benchmark(reloaded)
            self.assertEqual(
                [method.identifier for method in reloaded.methods],
                list(METHOD_IDS),
            )
            for method_id in METHOD_IDS:
                prediction = reloaded.samples[0].predictions[method_id]
                self.assertEqual(prediction.execution.status, "ok")
                self.assertTrue(prediction.artifact_path.is_relative_to(output.resolve()))
                self.assertEqual(_sha256(prediction.artifact_path), prediction.artifact_sha256)

            pids = (output / "worker-pids.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(pids), len(METHOD_IDS))
            self.assertEqual(len(set(pids)), len(METHOD_IDS))

            raw_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            sample = raw_manifest["samples"][0]
            self.assertFalse(Path(sample["image"]).is_absolute())
            self.assertFalse(Path(sample["reference"]).is_absolute())
            for prediction in sample["predictions"].values():
                self.assertFalse(Path(prediction["artifact"]).is_absolute())
            self.assertEqual(list(output.glob("worker-request--*.json")), [])
            self.assertEqual(list(output.glob("worker-result--*.json")), [])

    def test_preserves_a_structured_worker_failure_in_the_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            failed_method = "lsd-adaptive-v1"
            fake_worker = _write_fake_worker(folder, fail_backend=failed_method)
            output = Path(folder) / "generated"

            generated = generate_benchmark_dataset(
                template_path,
                output,
                worker_command=(sys.executable, str(fake_worker)),
            )

            prediction = generated.samples[0].predictions[failed_method]
            self.assertEqual(prediction.execution.status, "failed")
            self.assertIn("controlled missing backend", prediction.execution.error)
            self.assertIsNone(prediction.artifact_path)
            self.assertFalse(
                (output / "predictions" / failed_method / "straight-line.json").exists()
            )
            validate_benchmark(output / "manifest.json")

    def test_protocol_failure_leaves_no_partial_output_and_does_not_edit_template(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            template_bytes = template_path.read_bytes()
            fake_worker = _write_fake_worker(folder, malformed=True)
            output = Path(folder) / "generated"

            with self.assertRaises(GenerationError):
                generate_benchmark_dataset(
                    template_path,
                    output,
                    worker_command=(sys.executable, str(fake_worker)),
                )

            self.assertFalse(output.exists())
            self.assertEqual(template_path.read_bytes(), template_bytes)
            self.assertEqual(
                list(Path(folder).glob(".generated.staging-*")),
                [],
            )

    def test_rejects_worker_evidence_that_is_not_bound_to_the_input(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            fake_worker = _write_fake_worker(folder, wrong_input_hash=True)
            output = Path(folder) / "generated"

            with self.assertRaisesRegex(GenerationError, "input checksum"):
                generate_benchmark_dataset(
                    template_path,
                    output,
                    worker_command=(sys.executable, str(fake_worker)),
                )

            self.assertFalse(output.exists())

    def test_failed_worker_cannot_leave_an_unclaimed_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            fake_worker = _write_fake_worker(
                folder,
                fail_backend="canny-adaptive-v1",
                leave_artifact_on_failure=True,
            )
            output = Path(folder) / "generated"

            with self.assertRaisesRegex(GenerationError, "left an artifact"):
                generate_benchmark_dataset(
                    template_path,
                    output,
                    worker_command=(sys.executable, str(fake_worker)),
                )

            self.assertFalse(output.exists())

    def test_rejects_a_worker_source_hash_that_changed_after_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            fake_worker = _write_fake_worker(folder, wrong_source_hash=True)
            output = Path(folder) / "generated"

            with self.assertRaisesRegex(GenerationError, "source-file checksums changed"):
                generate_benchmark_dataset(
                    template_path,
                    output,
                    worker_command=(sys.executable, str(fake_worker)),
                )

            self.assertFalse(output.exists())

    def test_atomic_publish_never_replaces_a_destination_that_appeared(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "staging"
            destination = root / "generated"
            source.mkdir()
            destination.mkdir()
            (source / "source-marker").write_text("source", encoding="utf-8")
            (destination / "destination-marker").write_text(
                "destination",
                encoding="utf-8",
            )

            with self.assertRaises(FileExistsError):
                _rename_no_replace(source, destination)

            self.assertEqual(
                (source / "source-marker").read_text(encoding="utf-8"),
                "source",
            )
            self.assertEqual(
                (destination / "destination-marker").read_text(encoding="utf-8"),
                "destination",
            )

    def test_rejects_a_dangling_output_symlink_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            output = Path(folder) / "generated"
            output.symlink_to(Path(folder) / "missing-target", target_is_directory=True)

            with self.assertRaisesRegex(GenerationError, "already exists"):
                generate_benchmark_dataset(template_path, output)

            self.assertTrue(output.is_symlink())

    def test_rejects_a_nonisolated_thread_count_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            output = Path(folder) / "generated"

            with self.assertRaisesRegex(GenerationError, "threads=1"):
                generate_benchmark_dataset(template_path, output, threads=2)

            self.assertFalse(output.exists())

    def test_rejects_nonpositive_timeout_before_creating_output_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            template_path = _make_worker_template(folder)
            for timeout_seconds in (0, -1.0):
                with self.subTest(timeout_seconds=timeout_seconds):
                    output = Path(folder) / f"absent-{timeout_seconds}" / "generated"
                    with self.assertRaisesRegex(
                        GenerationError,
                        "timeout_seconds must be finite and positive",
                    ):
                        generate_benchmark_dataset(
                            template_path,
                            output,
                            timeout_seconds=timeout_seconds,
                        )

                    self.assertFalse(output.parent.exists())


if __name__ == "__main__":
    unittest.main()
