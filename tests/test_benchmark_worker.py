"""Contract tests for the dependency-light isolated benchmark worker."""

import builtins
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import benchmarks.worker as benchmark_worker
from benchmarks.geometry import load_centerline_artifact
from benchmarks.manifest import load_manifest
from benchmarks.worker import (
    BackendInfo,
    BackendUnavailableError,
    WORKER_REQUEST_SCHEMA_VERSION,
    WorkerDependencyError,
    WorkerRequestError,
    load_worker_request,
    run_request_file,
    run_worker,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "benchmarks" / "data" / "synthetic-smoke"


class FakePipeline:
    def __init__(
        self,
        backend,
        outputs=None,
        *,
        provider="OpenCV CPU",
        provider_verified=True,
    ):
        self.info = BackendInfo(
            actual_backend=backend,
            provider_kind="opencv",
            actual_provider=provider,
            provider_device_type="cpu",
            adapter_version="fake-opencv/1",
            package_versions={"numpy": "fake", "opencv": "fake"},
            thread_settings={
                "threads": 1,
                "opencv_set_num_threads": 0,
                "opencl": False,
                **{
                    variable: "1"
                    for variable in benchmark_worker.CPU_THREAD_VARIABLES
                },
            },
            provider_verified=provider_verified,
            source_files_sha256={"fake_pipeline.py": "0" * 64},
        )
        self.outputs = list(outputs or [[(1, 4), (7, 4)]])
        self.calls = 0
        self.loaded = None

    def load_image(self, path, width, height):
        self.loaded = (Path(path), width, height)
        return b"fake-image"

    def predict(self, image, prompt, configuration):
        self.calls += 1
        return self.outputs[min(self.calls - 1, len(self.outputs) - 1)]


def _request_payload(
    image_path,
    *,
    requested="canny-adaptive-v1",
    fallback=None,
    artifact="predictions/canny/straight-line.json",
):
    raw = Path(image_path).read_bytes()
    return {
        "schema_version": WORKER_REQUEST_SCHEMA_VERSION,
        "request_id": "straight-line.canny-adaptive-v1",
        "requested_backend": requested,
        "fallback_backend": fallback,
        "device": "cpu",
        "image": {
            "path": "images/straight-line.pgm",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "width": 9,
            "height": 9,
        },
        "artifact": artifact,
        "prompt": {
            "start_xy": [1.9, 4.8],
            "end_xy": [7.2, 4.1],
            "positive_xy": [],
            "negative_xy": [],
        },
        "configuration": {"edge_weight": 0.5},
        "warmup_runs": 1,
        "measurement_runs": 3,
        "threads": 1,
    }


class WorkerContractTests(unittest.TestCase):
    def _fixture(self, folder, **request_changes):
        root = Path(folder)
        image_folder = root / "images"
        image_folder.mkdir(parents=True)
        image_path = image_folder / "straight-line.pgm"
        shutil.copyfile(FIXTURE_ROOT / "images" / "straight-line.pgm", image_path)
        payload = _request_payload(image_path)
        payload.update(request_changes)
        request_path = root / "request.json"
        request_path.write_text(json.dumps(payload), encoding="utf-8")
        return request_path, payload

    def test_module_import_does_not_require_numpy_cv2_or_qgis(self):
        script = f"""
import builtins, sys
sys.path.insert(0, {str(REPOSITORY_ROOT)!r})
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'numpy' or name == 'cv2' or name == 'qgis' or name.startswith('qgis.'):
        raise AssertionError('eager optional import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import benchmarks.worker
print(benchmarks.worker.WORKER_REQUEST_SCHEMA_VERSION)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(WORKER_REQUEST_SCHEMA_VERSION, completed.stdout)

    def test_success_records_manifest_compatible_self_observed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            request = load_worker_request(request_path)
            pipeline = FakePipeline("canny-adaptive-v1")

            result = run_worker(request, pipeline_loader=lambda _backend, _threads: pipeline)

            prediction = result["prediction"]
            execution = prediction["execution"]
            runtime = execution["runtime"]
            timing = execution["timing"]
            artifact_path = Path(folder) / prediction["artifact"]
            artifact = load_centerline_artifact(artifact_path)

        self.assertEqual(execution["status"], "ok")
        self.assertEqual(execution["requested_backend"], "canny-adaptive-v1")
        self.assertEqual(execution["actual_backend"], "canny-adaptive-v1")
        self.assertEqual(runtime["actual_provider"], "OpenCV CPU")
        self.assertEqual(runtime["provider_device_type"], "cpu")
        self.assertTrue(runtime["provider_verified"])
        self.assertTrue(runtime["deterministic"])
        self.assertEqual(len(runtime["output_sha256_samples"]), 3)
        self.assertEqual(set(runtime["output_sha256_samples"]), {prediction["sha256"]})
        self.assertEqual(len(timing["wall_ns_samples"]), 3)
        self.assertEqual(len(timing["cpu_ns_samples"]), 3)
        self.assertGreater(timing["peak_rss_bytes"], 0)
        self.assertEqual(pipeline.calls, 4)
        self.assertEqual((artifact.width, artifact.height), (9, 9))
        self.assertEqual(artifact.paths[0].points, ((1.0, 4.0), (7.0, 4.0)))

    def test_lsd_load_failure_can_record_explicit_canny_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, payload = self._fixture(folder)
            payload.update(
                request_id="straight-line.lsd-adaptive-v1",
                requested_backend="lsd-adaptive-v1",
                fallback_backend="canny-adaptive-v1",
            )
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            request = load_worker_request(request_path)
            pipeline = FakePipeline("canny-adaptive-v1")

            def loader(backend, _threads):
                if backend == "lsd-adaptive-v1":
                    raise BackendUnavailableError("LSD is absent from this OpenCV build")
                return pipeline

            result = run_worker(request, pipeline_loader=loader)

        execution = result["prediction"]["execution"]
        self.assertEqual(execution["status"], "fallback")
        self.assertEqual(execution["requested_backend"], "lsd-adaptive-v1")
        self.assertEqual(execution["actual_backend"], "canny-adaptive-v1")
        self.assertIn("LSD is absent", execution["fallback_reason"])
        self.assertIsNone(execution["error"])

    def test_fallback_prediction_failure_preserves_both_backend_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, payload = self._fixture(folder)
            payload.update(
                request_id="straight-line.lsd-adaptive-v1",
                requested_backend="lsd-adaptive-v1",
                fallback_backend="canny-adaptive-v1",
            )
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            request = load_worker_request(request_path)
            fallback_pipeline = FakePipeline("canny-adaptive-v1")

            def failing_prediction(_image, _prompt, _configuration):
                fallback_pipeline.calls += 1
                raise RuntimeError("fallback prediction exploded")

            fallback_pipeline.predict = failing_prediction

            def loader(backend, _threads):
                if backend == "lsd-adaptive-v1":
                    raise BackendUnavailableError("original backend unavailable")
                return fallback_pipeline

            result = run_worker(request, pipeline_loader=loader)

        prediction = result["prediction"]
        execution = prediction["execution"]
        self.assertEqual(execution["status"], "failed")
        self.assertEqual(execution["actual_backend"], "canny-adaptive-v1")
        self.assertIsNone(execution["fallback_reason"])
        self.assertIn("original backend unavailable", execution["error"])
        self.assertIn("fallback prediction exploded", execution["error"])
        self.assertNotIn("artifact", prediction)
        self.assertEqual(fallback_pipeline.calls, 1)

    def test_repeated_output_hashes_expose_nondeterminism(self):
        outputs = [
            [(1, 4), (7, 4)],  # warm-up
            [(1, 4), (7, 4)],
            [(1, 4), (6, 4), (7, 4)],
            [(1, 4), (7, 4)],
        ]
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            request = load_worker_request(request_path)
            pipeline = FakePipeline("canny-adaptive-v1", outputs)
            result = run_worker(request, pipeline_loader=lambda _backend, _threads: pipeline)

            prediction = result["prediction"]
            runtime = prediction["execution"]["runtime"]
            published = (Path(folder) / prediction["artifact"]).read_bytes()

        self.assertFalse(runtime["deterministic"])
        self.assertEqual(len(set(runtime["output_sha256_samples"])), 2)
        self.assertEqual(hashlib.sha256(published).hexdigest(), runtime["output_sha256_samples"][0])

    def test_failed_post_write_verification_removes_unclaimed_artifact(self):
        variants = (
            ("mismatch", "f" * 64),
            ("read-error", OSError("controlled post-write read failure")),
        )
        for label, failure in variants:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                request_path, _payload = self._fixture(folder)
                request = load_worker_request(request_path)
                pipeline = FakePipeline("canny-adaptive-v1")
                real_sha256_file = benchmark_worker._sha256_file

                def broken_artifact_hash(path):
                    if Path(path) == request.artifact_path:
                        if isinstance(failure, BaseException):
                            raise failure
                        return failure
                    return real_sha256_file(path)

                with mock.patch.object(
                    benchmark_worker,
                    "_sha256_file",
                    side_effect=broken_artifact_hash,
                ):
                    result = run_worker(
                        request,
                        pipeline_loader=lambda _backend, _threads: pipeline,
                    )

                prediction = result["prediction"]
                self.assertEqual(prediction["execution"]["status"], "failed")
                self.assertNotIn("artifact", prediction)
                self.assertFalse(request.artifact_path.exists())

    def test_success_runtime_includes_source_file_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            request = load_worker_request(request_path)
            pipeline = FakePipeline("canny-adaptive-v1")

            result = run_worker(request, pipeline_loader=lambda _backend, _threads: pipeline)

        source_hashes = result["prediction"]["execution"]["runtime"][
            "source_files_sha256"
        ]
        self.assertEqual(source_hashes, {"fake_pipeline.py": "0" * 64})
        self.assertTrue(source_hashes)
        self.assertTrue(
            all(
                len(digest) == 64
                and set(digest) <= set("0123456789abcdef")
                for digest in source_hashes.values()
            )
        )

    def test_missing_lazy_dependency_becomes_a_failed_prediction_record(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            request = load_worker_request(request_path)

            def loader(_backend, _threads):
                raise WorkerDependencyError("OpenCV is deliberately unavailable")

            result = run_worker(request, pipeline_loader=loader)

        prediction = result["prediction"]
        execution = prediction["execution"]
        self.assertNotIn("artifact", prediction)
        self.assertEqual(execution["status"], "failed")
        self.assertIsNone(execution["actual_backend"])
        self.assertIn("OpenCV is deliberately unavailable", execution["error"])
        self.assertFalse(execution["runtime"]["provider_verified"])
        self.assertIsNone(execution["runtime"]["deterministic"])
        self.assertEqual(execution["runtime"]["output_sha256_samples"], [])
        self.assertEqual(
            execution["runtime"]["thread_settings"],
            {
                "threads": 1,
                "opencv_set_num_threads": 0,
                "opencl": False,
                **{
                    variable: "1"
                    for variable in benchmark_worker.CPU_THREAD_VARIABLES
                },
            },
        )

    def test_default_loader_turns_a_missing_numpy_import_into_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            request = load_worker_request(request_path)
            real_import = builtins.__import__

            def import_without_numpy(name, *args, **kwargs):
                if name == "numpy":
                    raise ImportError("controlled missing NumPy")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=import_without_numpy):
                result = run_worker(request)

        execution = result["prediction"]["execution"]
        self.assertEqual(execution["status"], "failed")
        self.assertIn("require NumPy", execution["error"])
        self.assertFalse(execution["runtime"]["provider_verified"])

    def test_invalid_provider_attestation_cannot_publish_an_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            request = load_worker_request(request_path)
            pipeline = FakePipeline(
                "canny-adaptive-v1",
                provider="CUDAExecutionProvider",
            )
            result = run_worker(request, pipeline_loader=lambda _backend, _threads: pipeline)

        prediction = result["prediction"]
        self.assertNotIn("artifact", prediction)
        self.assertEqual(prediction["execution"]["status"], "failed")
        self.assertIn("OpenCV CPU", prediction["execution"]["error"])
        self.assertEqual(pipeline.calls, 0)

    def test_mismatched_actual_thread_count_cannot_publish_an_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            request = load_worker_request(request_path)
            pipeline = FakePipeline("canny-adaptive-v1")
            pipeline.info = BackendInfo(
                actual_backend="canny-adaptive-v1",
                provider_kind="opencv",
                actual_provider="OpenCV CPU",
                provider_device_type="cpu",
                adapter_version="fake-opencv/1",
                package_versions={"numpy": "fake", "opencv": "fake"},
                thread_settings={
                    "threads": 8,
                    "opencv_set_num_threads": 0,
                    "opencl": False,
                },
                source_files_sha256={"fake_pipeline.py": "0" * 64},
            )

            result = run_worker(request, pipeline_loader=lambda _backend, _threads: pipeline)

        prediction = result["prediction"]
        self.assertEqual(prediction["execution"]["status"], "failed")
        self.assertEqual(pipeline.calls, 0)
        self.assertNotIn("artifact", prediction)

    def test_unverified_provider_cannot_publish_an_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            request = load_worker_request(request_path)
            pipeline = FakePipeline(
                "canny-adaptive-v1",
                provider_verified=False,
            )
            result = run_worker(request, pipeline_loader=lambda _backend, _threads: pipeline)

        prediction = result["prediction"]
        self.assertNotIn("artifact", prediction)
        self.assertEqual(prediction["execution"]["status"], "failed")
        self.assertEqual(pipeline.calls, 0)

    def test_real_adapter_boundary_uses_shared_strict_trace_kernel(self):
        detector_calls = {}

        class FakeDetector:
            def __init__(self, method):
                detector_calls["method"] = method
                self.lsd = object() if method == "lsd" else None

            def detect_edges(self, image, low_threshold, high_threshold):
                detector_calls["detect"] = (image, low_threshold, high_threshold)
                return "edges"

            def get_edge_cost_map(self, edges, edge_weight):
                detector_calls["cost"] = (edges, edge_weight)
                return SimpleNamespace(shape=(9, 9))

        fake_edge_module = SimpleNamespace(
            EdgeDetector=FakeDetector,
            _skimage_skeletonize=lambda value: value,
            __file__=__file__,
        )
        fake_numpy = SimpleNamespace(__version__="fake")
        fake_cv2 = SimpleNamespace(__version__="fake", getNumThreads=lambda: 1)

        import ai_vectorizer.core as core_package

        with mock.patch.object(
            core_package,
            "edge_detector",
            fake_edge_module,
            create=True,
        ):
            pipeline = benchmark_worker._OpenCVTracePipeline(
                "canny-adaptive-v1",
                1,
                fake_numpy,
                fake_cv2,
            )

        trace_calls = {}

        class FakeTraceConfig:
            def __init__(self, *, validate_all_costs, validate_accessed_costs):
                self.validate_all_costs = validate_all_costs
                self.validate_accessed_costs = validate_accessed_costs

        class FakeTraceKernel:
            TraceConfig = FakeTraceConfig

            @staticmethod
            def trace_path(cost_map, start_xy, end_xy, *, allow_partial, config):
                trace_calls.update(
                    cost_map=cost_map,
                    start_xy=start_xy,
                    end_xy=end_xy,
                    allow_partial=allow_partial,
                    validate_all_costs=config.validate_all_costs,
                    validate_accessed_costs=config.validate_accessed_costs,
                )
                return SimpleNamespace(
                    status="complete",
                    start=start_xy,
                    end=end_xy,
                )

            @staticmethod
            def centerline_points(
                result,
                *,
                segment_start_xy=None,
                segment_target_xy=None,
            ):
                trace_calls["centerline_points"] = True
                trace_calls["segment_start_xy"] = segment_start_xy
                trace_calls["segment_target_xy"] = segment_target_xy
                return (segment_start_xy, result.end)

        pipeline._trace_kernel = FakeTraceKernel
        points = pipeline.predict(
            "image",
            benchmark_worker.TracePrompt((1.9, 4.8), (7.2, 4.1)),
            {"edge_weight": 0.5},
        )

        self.assertEqual(detector_calls["method"], "canny")
        self.assertEqual(detector_calls["detect"], ("image", 30, 100))
        self.assertEqual(detector_calls["cost"], ("edges", 0.5))
        self.assertEqual(trace_calls["start_xy"], (1, 4))
        self.assertEqual(trace_calls["end_xy"], (7, 4))
        self.assertFalse(trace_calls["allow_partial"])
        self.assertFalse(trace_calls["validate_all_costs"])
        self.assertFalse(trace_calls["validate_accessed_costs"])
        self.assertTrue(trace_calls["centerline_points"])
        self.assertEqual(trace_calls["segment_start_xy"], (1.9, 4.8))
        self.assertEqual(trace_calls["segment_target_xy"], (7.2, 4.1))
        self.assertEqual(points, ((1.9, 4.8), (7, 4)))

    def test_request_rejects_path_traversal_and_nonstandard_edge_weight(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, payload = self._fixture(folder)
            payload["artifact"] = "../escaped.json"
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(WorkerRequestError, "predictions"):
                load_worker_request(request_path)

            payload["artifact"] = "images/straight-line.pgm"
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(WorkerRequestError, "predictions"):
                load_worker_request(request_path)

            payload["artifact"] = "predictions/output.json"
            payload["configuration"]["edge_weight"] = 0.75
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(WorkerRequestError, "edge_weight=0.5"):
                load_worker_request(request_path)

            payload["configuration"]["edge_weight"] = 0.5
            payload["configuration"]["smoothing_profile"] = "endpoint-preserving-v2"
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(WorkerRequestError, "smart-trace-v1-historical"):
                load_worker_request(request_path)

            payload["configuration"].pop("smoothing_profile")
            payload["configuration"]["edge_weight"] = 0.5
            payload["prompt"]["end_xy"] = [8.5, 4]
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(WorkerRequestError, "outside"):
                load_worker_request(request_path)

            payload["prompt"]["end_xy"] = [7, 4]
            payload["threads"] = 2
            request_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(WorkerRequestError, "between 1 and 1"):
                load_worker_request(request_path)

    def test_request_rejects_an_artifact_path_with_a_symlink_component(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, _payload = self._fixture(folder)
            predictions = Path(folder) / "predictions"
            predictions.mkdir()
            symlink_target = Path(folder) / "real-canny-output"
            symlink_target.mkdir()
            try:
                (predictions / "canny").symlink_to(
                    symlink_target,
                    target_is_directory=True,
                )
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")

            with self.assertRaisesRegex(WorkerRequestError, "symbolic links"):
                load_worker_request(request_path)

            self.assertFalse((symlink_target / "straight-line.json").exists())

    def test_request_rejects_width_1001_before_inspecting_image_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            request_path, payload = self._fixture(folder)
            payload["image"]["width"] = 1001
            request_path.write_text(json.dumps(payload), encoding="utf-8")

            with mock.patch.object(
                benchmark_worker,
                "image_dimensions",
                side_effect=AssertionError("image bytes must not be inspected"),
            ):
                with self.assertRaisesRegex(WorkerRequestError, "between 1 and 1000"):
                    load_worker_request(request_path)

    def test_worker_prediction_can_be_merged_without_editing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "fixture"
            shutil.copytree(FIXTURE_ROOT, fixture)
            image_path = fixture / "images" / "straight-line.pgm"
            request_payload = _request_payload(
                image_path,
                artifact="predictions/canny-adaptive-v1/straight-line.json",
            )
            request_path = fixture / "worker-request.json"
            request_path.write_text(json.dumps(request_payload), encoding="utf-8")
            pipeline = FakePipeline("canny-adaptive-v1")
            result = run_request_file(
                request_path,
                fixture / "worker-result.json",
                pipeline_loader=lambda _backend, _threads: pipeline,
            )

            manifest_path = fixture / "manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["methods"] = [
                {
                    "id": "canny-adaptive-v1",
                    "label": "Canny adaptive v1",
                    "kind": "precomputed_centerline",
                    "source": "AI Vectorizer EdgeDetector",
                    "version": "1",
                    "license": "GPL-2.0-or-later",
                    "configuration": {"edge_weight": 0.5},
                }
            ]
            manifest_payload["samples"][0]["predictions"] = {
                "canny-adaptive-v1": copy.deepcopy(result["prediction"])
            }
            manifest_payload["samples"][0]["prompt"] = copy.deepcopy(
                request_payload["prompt"]
            )
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

            manifest = load_manifest(manifest_path)

        execution = manifest.samples[0].predictions["canny-adaptive-v1"].execution
        self.assertEqual(execution.status, "ok")
        self.assertEqual(execution.runtime["adapter_version"], "fake-opencv/1")
        self.assertEqual(execution.runtime["actual_provider"], "OpenCV CPU")


if __name__ == "__main__":
    unittest.main()
