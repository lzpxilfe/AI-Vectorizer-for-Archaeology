"""Integration-contract tests for the isolated EfficientSAM benchmark path."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT
from ai_vectorizer.core.model_store import ModelNotFoundError, bundle_fingerprint
from benchmarks import cli
from benchmarks.generate import (
    GenerationError,
    _REAL_WORKER_SOURCES,
    _validated_sam_prediction_sample,
    _verified_prediction,
    generate_benchmark_dataset,
)
from benchmarks.evidence import (
    PROMPT_EVIDENCE_SCHEMA_VERSION,
    prompt_sha256,
    sam_prompt_tensor_sha256,
)
from benchmarks.geometry import load_centerline_artifact
from benchmarks.manifest import (
    ManifestError,
    _sam_prediction_evidence,
    load_manifest,
)
import benchmarks.worker as benchmark_worker
from benchmarks.worker import (
    BackendInfo,
    EFFICIENTSAM_BACKEND,
    WORKER_REQUEST_SCHEMA_VERSION,
    WorkerRequestError,
    load_worker_request,
    run_worker,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = (
    REPOSITORY_ROOT / "benchmarks" / "data" / "efficientsam-runtime-template"
)
MODEL_FINGERPRINT = bundle_fingerprint(EFFICIENTSAM_TI_SPLIT)
EXPECTED_SAM_SOURCE_FILES = frozenset(
    {
        "benchmarks/__init__.py",
        "benchmarks/evidence.py",
        "benchmarks/geometry.py",
        "benchmarks/manifest.py",
        "benchmarks/worker.py",
        "ai_vectorizer/__init__.py",
        "ai_vectorizer/core/__init__.py",
        "ai_vectorizer/core/dependencies.py",
        "ai_vectorizer/core/edge_detector.py",
        "ai_vectorizer/core/efficientsam_onnx.py",
        "ai_vectorizer/core/efficientsam_spec.py",
        "ai_vectorizer/core/model_store.py",
        "ai_vectorizer/core/sam_trace_kernel.py",
        "ai_vectorizer/core/trace_kernel.py",
    }
)
EXPECTED_EDGE_SOURCE_FILES = frozenset(
    {
        "benchmarks/__init__.py",
        "benchmarks/evidence.py",
        "benchmarks/geometry.py",
        "benchmarks/manifest.py",
        "benchmarks/worker.py",
        "ai_vectorizer/__init__.py",
        "ai_vectorizer/core/__init__.py",
        "ai_vectorizer/core/dependencies.py",
        "ai_vectorizer/core/edge_detector.py",
        "ai_vectorizer/core/efficientsam_spec.py",
        "ai_vectorizer/core/trace_kernel.py",
    }
)


class _FakeEfficientSAMPipeline:
    def __init__(
        self,
        *,
        provider="CPUExecutionProvider",
        opencv_effective_num_threads=1,
        opencl=False,
        onnx_session_options=None,
    ):
        if onnx_session_options is None:
            onnx_session_options = {
                "encoder": dict(benchmark_worker.EFFICIENTSAM_ORT_SESSION_OPTIONS),
                "decoder": dict(benchmark_worker.EFFICIENTSAM_ORT_SESSION_OPTIONS),
            }
        self.info = BackendInfo(
            actual_backend=EFFICIENTSAM_BACKEND,
            provider_kind="onnxruntime",
            actual_provider=provider,
            provider_device_type="cpu",
            adapter_version="fake-efficientsam/1",
            package_versions={"numpy": "fake", "onnxruntime": "fake"},
            thread_settings={
                "threads": 1,
                "onnx_intra_op_num_threads": 1,
                "onnx_inter_op_num_threads": 1,
                "onnx_execution_mode": "ORT_SEQUENTIAL",
                "onnx_graph_optimization_level": "ORT_ENABLE_ALL",
                "opencv_set_num_threads": 0,
                "opencv_effective_num_threads": opencv_effective_num_threads,
                "opencl": opencl,
                **{
                    variable: "1"
                    for variable in benchmark_worker.CPU_THREAD_VARIABLES
                },
            },
            provider_verified=True,
            source_files_sha256={"fake_efficientsam.py": "0" * 64},
            model_artifacts_sha256={
                artifact.id: artifact.sha256
                for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
            },
            model_bundle_id=EFFICIENTSAM_TI_SPLIT.id,
            model_bundle_sha256=MODEL_FINGERPRINT,
            runtime_details={
                "model_source_commit": EFFICIENTSAM_TI_SPLIT.source_commit,
                "onnx_providers": {
                    "encoder": ["CPUExecutionProvider"],
                    "decoder": ["CPUExecutionProvider"],
                },
                "onnx_session_options": onnx_session_options,
                "encoder_reused_across_predictions": True,
                "session_initialization_ns": 0,
                "image_file_decode_ns": 0,
                "image_encode_wall_ns": 0,
                "edge_cache_fill_wall_ns": 0,
            },
        )
        self.calls = 0
        self._latest_evidence = None

    def load_image(self, path, width, height):
        return (Path(path), width, height)

    def predict(self, _image, _prompt, _configuration):
        self.calls += 1
        self._latest_evidence = {
            "schema_version": (
                benchmark_worker.EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION
            ),
            "selected_mask_index": 1,
            "iou_predictions": [0.1, 0.9, 0.2],
            "iou_predictions_sha256": hashlib.sha256(
                struct.pack("<3f", 0.1, 0.9, 0.2)
            ).hexdigest(),
            "selected_logits_sha256": hashlib.sha256(b"fake-logits").hexdigest(),
            "selected_binary_mask_sha256": hashlib.sha256(b"fake-mask").hexdigest(),
            "accepted_mask_sha256": hashlib.sha256(b"fake-accepted").hexdigest(),
            "decoder_wall_ns": 1,
        }
        return ((140.0, 512.0), (880.0, 512.0))

    def prediction_evidence(self):
        return copy.deepcopy(self._latest_evidence)


def _request_payload(image_path, cache_path):
    return {
        "schema_version": WORKER_REQUEST_SCHEMA_VERSION,
        "request_id": "outlined-ellipse--efficientsam-ti-onnx-v1",
        "requested_backend": EFFICIENTSAM_BACKEND,
        "fallback_backend": None,
        "device": "cpu",
        "image": {
            "path": "images/outlined-ellipse.png",
            "sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "width": 1024,
            "height": 1024,
        },
        "artifact": "predictions/efficientsam-ti-onnx-v1/outlined-ellipse.json",
        "prompt": {
            "start_xy": [140, 512],
            "end_xy": [880, 512],
            "positive_xy": [],
            "negative_xy": [[512, 512]],
        },
        "configuration": {
            "model_bundle_id": EFFICIENTSAM_TI_SPLIT.id,
            "model_bundle_sha256": MODEL_FINGERPRINT,
            "mask_logit_threshold": 0.0,
            "canny_low_threshold": 30,
            "canny_high_threshold": 100,
            "smoothing_profile": "smart-trace-v1-historical",
        },
        "model_cache": str(Path(cache_path).absolute()),
        "warmup_runs": 1,
        "measurement_runs": 3,
        "threads": 1,
    }


class EfficientSAMWorkerContractTests(unittest.TestCase):
    def _request(self, folder, mutate=None):
        root = Path(folder)
        image_folder = root / "images"
        image_folder.mkdir(parents=True)
        image_path = image_folder / "outlined-ellipse.png"
        shutil.copyfile(TEMPLATE_ROOT / "images" / image_path.name, image_path)
        payload = _request_payload(image_path, root / "model-cache")
        if mutate is not None:
            mutate(payload)
        request_path = root / "request.json"
        request_path.write_text(json.dumps(payload), encoding="utf-8")
        return load_worker_request(request_path)

    def test_fake_pipeline_records_model_and_cpu_provider_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            request = self._request(folder)
            pipeline = _FakeEfficientSAMPipeline()
            result = run_worker(
                request,
                pipeline_loader=lambda _backend, _threads: pipeline,
            )
            prediction = result["prediction"]
            runtime = prediction["execution"]["runtime"]
            artifact = load_centerline_artifact(
                Path(folder) / prediction["artifact"]
            )

        self.assertEqual(prediction["execution"]["status"], "ok")
        self.assertEqual(runtime["provider_kind"], "onnxruntime")
        self.assertEqual(runtime["actual_provider"], "CPUExecutionProvider")
        self.assertEqual(runtime["model_bundle_sha256"], MODEL_FINGERPRINT)
        self.assertEqual(runtime["prompt_sha256"], prompt_sha256(request.prompt))
        self.assertEqual(
            runtime["sam_prompt_tensor_sha256"],
            sam_prompt_tensor_sha256(request.prompt),
        )
        self.assertEqual(len(runtime["sam_prediction_samples"]), 4)
        self.assertEqual(
            runtime["model_source_commit"], EFFICIENTSAM_TI_SPLIT.source_commit
        )
        self.assertEqual(
            runtime["model_artifacts_sha256"],
            {
                artifact_spec.id: artifact_spec.sha256
                for artifact_spec in EFFICIENTSAM_TI_SPLIT.artifacts
            },
        )
        self.assertEqual(runtime["thread_settings"]["opencv_set_num_threads"], 0)
        self.assertEqual(runtime["thread_settings"]["opencv_effective_num_threads"], 1)
        self.assertIs(runtime["thread_settings"]["opencl"], False)
        self.assertEqual(
            runtime["onnx_session_options"],
            {
                "encoder": dict(benchmark_worker.EFFICIENTSAM_ORT_SESSION_OPTIONS),
                "decoder": dict(benchmark_worker.EFFICIENTSAM_ORT_SESSION_OPTIONS),
            },
        )
        self.assertEqual(pipeline.calls, 4)
        self.assertEqual(
            artifact.metadata["mask_trace_kernel"],
            "ai_vectorizer.core.sam_trace_kernel",
        )
        self.assertEqual(artifact.metadata["model_bundle_sha256"], MODEL_FINGERPRINT)
        self.assertEqual(
            artifact.metadata["model_source_commit"],
            EFFICIENTSAM_TI_SPLIT.source_commit,
        )
        self.assertEqual(
            artifact.metadata["segmentation_evidence"],
            {
                key: value
                for key, value in runtime["sam_prediction_samples"][1].items()
                if key != "decoder_wall_ns"
            },
        )

    def test_iou_selection_uses_float32_tensor_values_at_every_boundary(self):
        raw_ious = [1.00000001, 1.00000002, 0.0]
        evidence = {
            "schema_version": (
                benchmark_worker.EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION
            ),
            "selected_mask_index": 1,
            "iou_predictions": raw_ious,
            "iou_predictions_sha256": hashlib.sha256(
                struct.pack("<3f", *raw_ious)
            ).hexdigest(),
            "selected_logits_sha256": "1" * 64,
            "selected_binary_mask_sha256": "2" * 64,
            "accepted_mask_sha256": "3" * 64,
            "decoder_wall_ns": 1,
        }
        validators = (
            (
                lambda value: benchmark_worker._validated_sam_prediction_evidence(
                    value
                ),
                benchmark_worker.PredictionError,
            ),
            (
                lambda value: _validated_sam_prediction_sample(value, "sample"),
                GenerationError,
            ),
            (
                lambda value: _sam_prediction_evidence(value, "sample"),
                ManifestError,
            ),
        )
        for validator, error_type in validators:
            with self.subTest(validator=validator):
                with self.assertRaises(error_type):
                    validator(copy.deepcopy(evidence))

                accepted = copy.deepcopy(evidence)
                accepted["selected_mask_index"] = 0
                normalized = validator(accepted)
                self.assertEqual(normalized["iou_predictions"], [1.0, 1.0, 0.0])

    def test_wrong_provider_attestation_cannot_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            request = self._request(folder)
            pipeline = _FakeEfficientSAMPipeline(provider="CUDAExecutionProvider")
            result = run_worker(
                request,
                pipeline_loader=lambda _backend, _threads: pipeline,
            )

        prediction = result["prediction"]
        self.assertEqual(prediction["execution"]["status"], "failed")
        self.assertNotIn("artifact", prediction)
        self.assertIn("CPUExecutionProvider", prediction["execution"]["error"])
        self.assertEqual(pipeline.calls, 0)

    def test_unattested_opencv_state_cannot_publish(self):
        variants = (
            ({"opencv_effective_num_threads": 2}, "OpenCV"),
            ({"opencl": True}, "OpenCV"),
        )
        for pipeline_options, error_fragment in variants:
            with self.subTest(pipeline_options=pipeline_options):
                with tempfile.TemporaryDirectory() as folder:
                    request = self._request(folder)
                    pipeline = _FakeEfficientSAMPipeline(**pipeline_options)
                    result = run_worker(
                        request,
                        pipeline_loader=lambda _backend, _threads: pipeline,
                    )

                prediction = result["prediction"]
                self.assertEqual(prediction["execution"]["status"], "failed")
                self.assertNotIn("artifact", prediction)
                self.assertIn(error_fragment, prediction["execution"]["error"])
                self.assertEqual(pipeline.calls, 0)

    def test_unattested_onnx_session_readback_cannot_publish(self):
        session_options = {
            "encoder": dict(benchmark_worker.EFFICIENTSAM_ORT_SESSION_OPTIONS),
            "decoder": dict(benchmark_worker.EFFICIENTSAM_ORT_SESSION_OPTIONS),
        }
        session_options["decoder"]["inter_op_num_threads"] = 2
        with tempfile.TemporaryDirectory() as folder:
            request = self._request(folder)
            pipeline = _FakeEfficientSAMPipeline(
                onnx_session_options=session_options
            )
            result = run_worker(
                request,
                pipeline_loader=lambda _backend, _threads: pipeline,
            )

        prediction = result["prediction"]
        self.assertEqual(prediction["execution"]["status"], "failed")
        self.assertNotIn("artifact", prediction)
        self.assertIn("session evidence", prediction["execution"]["error"])
        self.assertEqual(pipeline.calls, 0)

    def test_opencv_configuration_is_read_back(self):
        class FakeOcl:
            def __init__(self):
                self.enabled = True

            def setUseOpenCL(self, enabled):
                self.enabled = enabled

            def useOpenCL(self):
                return self.enabled

        class FakeCv2:
            def __init__(self):
                self.requested_threads = None
                self.effective_threads = 8
                self.ocl = FakeOcl()

            def setNumThreads(self, threads):
                self.requested_threads = threads
                self.effective_threads = 1

            def getNumThreads(self):
                return self.effective_threads

        fake_cv2 = FakeCv2()
        observed = benchmark_worker._configure_efficientsam_opencv(fake_cv2, 1)

        self.assertEqual(fake_cv2.requested_threads, 0)
        self.assertEqual(observed, (1, False))

    def test_start_and_end_count_toward_six_prompt_limit(self):
        def add_guides(payload):
            payload["prompt"]["positive_xy"] = [
                [200, 400],
                [300, 350],
                [400, 300],
                [500, 300],
                [600, 350],
            ]
            payload["prompt"]["negative_xy"] = []

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(WorkerRequestError, "at most four guide"):
                self._request(folder, mutate=add_guides)

    def test_additional_guides_cannot_repeat_start_or_end(self):
        def repeat_start(payload):
            payload["prompt"]["positive_xy"] = [
                list(payload["prompt"]["start_xy"])
            ]

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(WorkerRequestError, "must not duplicate"):
                self._request(folder, mutate=repeat_start)

    def test_efficientsam_request_rejects_cross_family_fallback(self):
        def add_edge_fallback(payload):
            payload["fallback_backend"] = "canny-adaptive-v1"

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(
                WorkerRequestError,
                "same detector family",
            ):
                self._request(folder, mutate=add_edge_fallback)


class EfficientSAMGenerationBoundaryTests(unittest.TestCase):
    def test_sam_worker_source_snapshot_has_exact_execution_closure(self):
        self.assertEqual(
            frozenset(_REAL_WORKER_SOURCES["edge"]),
            EXPECTED_EDGE_SOURCE_FILES,
        )
        self.assertEqual(
            frozenset(_REAL_WORKER_SOURCES["sam"]),
            EXPECTED_SAM_SOURCE_FILES,
        )

    def test_manifest_rejects_sam_evidence_not_bound_to_published_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dataset"
            shutil.copytree(TEMPLATE_ROOT, root)
            image_path = root / "images" / "outlined-ellipse.png"
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(_request_payload(image_path, root / "model-cache")),
                encoding="utf-8",
            )
            request = load_worker_request(request_path)
            result = run_worker(
                request,
                pipeline_loader=lambda _backend, _threads: (
                    _FakeEfficientSAMPipeline()
                ),
            )
            manifest_path = root / "manifest.json"
            manifest_payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest_payload["samples"][0]["predictions"] = {
                EFFICIENTSAM_BACKEND: copy.deepcopy(result["prediction"])
            }
            manifest_payload["samples"][0]["prompt"]["schema_version"] = (
                PROMPT_EVIDENCE_SCHEMA_VERSION
            )
            manifest_path.write_text(
                json.dumps(manifest_payload),
                encoding="utf-8",
            )
            load_manifest(manifest_path)

            provider_changed = copy.deepcopy(manifest_payload)
            provider_runtime = provider_changed["samples"][0]["predictions"][
                EFFICIENTSAM_BACKEND
            ]["execution"]["runtime"]
            provider_runtime["provider_kind"] = "synthetic"
            provider_runtime["actual_provider"] = "synthetic CPU"
            provider_runtime.pop("onnx_providers")
            manifest_path.write_text(
                json.dumps(provider_changed),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ManifestError,
                "onnxruntime CPU provider contract",
            ):
                load_manifest(manifest_path)

            runtime_variants = (
                (
                    lambda runtime: runtime.__setitem__(
                        "provider_verified", False
                    ),
                    "provider_verified",
                ),
                (
                    lambda runtime: runtime.__setitem__(
                        "model_bundle_id", "mislabelled-bundle"
                    ),
                    "pinned EfficientSAM bundle",
                ),
                (
                    lambda runtime: runtime.__setitem__(
                        "model_source_commit", "0" * 40
                    ),
                    "pinned EfficientSAM bundle",
                ),
                (
                    lambda runtime: runtime["model_artifacts_sha256"].__setitem__(
                        "encoder", "0" * 64
                    ),
                    "pinned EfficientSAM bundle",
                ),
                (
                    lambda runtime: runtime["onnx_session_options"][
                        "decoder"
                    ].__setitem__("inter_op_num_threads", 2),
                    "onnx_session_options",
                ),
                (
                    lambda runtime: runtime.__setitem__(
                        "configuration_sha256", "0" * 64
                    ),
                    "sample image and method configuration",
                ),
            )
            for mutate_runtime, error_fragment in runtime_variants:
                with self.subTest(runtime_evidence=error_fragment):
                    changed = copy.deepcopy(manifest_payload)
                    changed_runtime = changed["samples"][0]["predictions"][
                        EFFICIENTSAM_BACKEND
                    ]["execution"]["runtime"]
                    mutate_runtime(changed_runtime)
                    manifest_path.write_text(
                        json.dumps(changed),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ManifestError, error_fragment):
                        load_manifest(manifest_path)

            for mutate_method in (
                lambda method: method.__setitem__("model_sha256", "0" * 64),
                lambda method: method["configuration"].__setitem__(
                    "model_bundle_id", "mislabelled-bundle"
                ),
                lambda method: method["configuration"].__setitem__(
                    "mask_logit_threshold", 1.0
                ),
                lambda method: method["configuration"].__setitem__(
                    "canny_low_threshold", 200
                ),
            ):
                with self.subTest(method_contract=mutate_method):
                    changed = copy.deepcopy(manifest_payload)
                    mutate_method(changed["methods"][0])
                    manifest_path.write_text(
                        json.dumps(changed),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ManifestError,
                        "must match the pinned worker model",
                    ):
                        load_manifest(manifest_path)

            wrong_dimensions = copy.deepcopy(manifest_payload)
            wrong_dimensions["samples"][0]["width"] = 1000
            manifest_path.write_text(
                json.dumps(wrong_dimensions),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ManifestError,
                "EfficientSAM inputs must be exactly 1024x1024",
            ):
                load_manifest(manifest_path)

            artifact_path = root / result["prediction"]["artifact"]
            original_artifact = artifact_path.read_bytes()
            changed_artifact = json.loads(original_artifact.decode("utf-8"))
            changed_artifact["metadata"]["requested_backend"] = "forged-backend"
            changed_artifact_raw = (
                json.dumps(
                    changed_artifact,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            artifact_path.write_bytes(changed_artifact_raw)
            changed_artifact_sha = hashlib.sha256(changed_artifact_raw).hexdigest()
            artifact_bound_manifest = copy.deepcopy(manifest_payload)
            artifact_bound_prediction = artifact_bound_manifest["samples"][0][
                "predictions"
            ][EFFICIENTSAM_BACKEND]
            artifact_bound_prediction["sha256"] = changed_artifact_sha
            artifact_bound_prediction["execution"]["runtime"][
                "output_sha256_samples"
            ] = [changed_artifact_sha] * 3
            manifest_path.write_text(
                json.dumps(artifact_bound_manifest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "artifact is not bound"):
                load_manifest(manifest_path)
            artifact_path.write_bytes(original_artifact)

            runtime = manifest_payload["samples"][0]["predictions"][
                EFFICIENTSAM_BACKEND
            ]["execution"]["runtime"]
            runtime["sam_prediction_samples"][1][
                "selected_logits_sha256"
            ] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest_payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ManifestError, "artifact is not bound"):
                load_manifest(manifest_path)

    def test_generator_rejects_changed_runtime_readback_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_folder = root / "images"
            image_folder.mkdir(parents=True)
            image_path = image_folder / "outlined-ellipse.png"
            shutil.copyfile(TEMPLATE_ROOT / "images" / image_path.name, image_path)
            request_path = root / "request.json"
            request_path.write_text(
                json.dumps(_request_payload(image_path, root / "model-cache")),
                encoding="utf-8",
            )
            request = load_worker_request(request_path)
            pipeline = _FakeEfficientSAMPipeline()
            result = run_worker(
                request,
                pipeline_loader=lambda _backend, _threads: pipeline,
            )
            runtime = result["prediction"]["execution"]["runtime"]
            result_path = root / "result.json"

            variants = (
                (
                    lambda changed: changed["prediction"]["execution"]["runtime"]
                    ["thread_settings"].__setitem__(
                        "opencv_effective_num_threads", 2
                    ),
                    "CPU thread evidence",
                ),
                (
                    lambda changed: changed["prediction"]["execution"]["runtime"]
                    ["onnx_session_options"]["decoder"].__setitem__(
                        "inter_op_num_threads", 2
                    ),
                    "raw ONNX session evidence",
                ),
                (
                    lambda changed: changed["prediction"]["execution"]["runtime"].__setitem__(
                        "prompt_sha256", "0" * 64
                    ),
                    "prompt checksum evidence",
                ),
                (
                    lambda changed: changed["prediction"]["execution"]["runtime"].__setitem__(
                        "model_bundle_id", "mislabelled-bundle"
                    ),
                    "pinned model evidence",
                ),
                (
                    lambda changed: changed["prediction"]["execution"]["runtime"].__setitem__(
                        "model_source_commit", "0" * 40
                    ),
                    "pinned model evidence",
                ),
                (
                    lambda changed: changed["prediction"]["execution"]["runtime"]
                    ["sam_prediction_samples"][1].__setitem__(
                        "selected_logits_sha256", "0" * 64
                    ),
                    "artifact metadata changed 'segmentation_evidence'",
                ),
                (
                    lambda changed: changed["prediction"]["execution"]["runtime"]
                    ["sam_prediction_samples"][0].__setitem__(
                        "iou_predictions_sha256", "0" * 64
                    ),
                    "iou_predictions_sha256 disagrees",
                ),
            )
            for mutate, error_fragment in variants:
                with self.subTest(error_fragment=error_fragment):
                    changed = copy.deepcopy(result)
                    mutate(changed)
                    result_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(GenerationError, error_fragment):
                        _verified_prediction(
                            result_path,
                            expected_request_id=request.request_id,
                            expected_backend=request.requested_backend,
                            expected_fallback_backend=request.fallback_backend,
                            expected_artifact=request.artifact_manifest_path,
                            expected_input_sha256=request.image_sha256,
                            expected_configuration_sha256=runtime[
                                "configuration_sha256"
                            ],
                            expected_prompt_sha256=prompt_sha256(request.prompt),
                            expected_source_tile_origin_xy=(0, 0),
                            expected_source_grid_input_sha256=None,
                            expected_sam_prompt_tensor_sha256=(
                                sam_prompt_tensor_sha256(request.prompt)
                            ),
                            expected_source_files_sha256=runtime[
                                "source_files_sha256"
                            ],
                            expected_model_sha256=MODEL_FINGERPRINT,
                            expected_model_artifacts_sha256={
                                artifact.id: artifact.sha256
                                for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
                            },
                            expected_model_bundle_id=EFFICIENTSAM_TI_SPLIT.id,
                            expected_model_source_commit=(
                                EFFICIENTSAM_TI_SPLIT.source_commit
                            ),
                            expected_width=request.width,
                            expected_height=request.height,
                            expected_threads=request.threads,
                            expected_warmup_runs=request.warmup_runs,
                            expected_measurement_runs=request.measurement_runs,
                            stage_root=request.root,
                            return_code=0,
                            stderr="",
                        )

    def test_missing_model_fails_offline_before_output_parent_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "not-created" / "generated"
            cache = root / "missing-cache"
            with mock.patch(
                "ai_vectorizer.core.model_store.resolve_bundle",
                side_effect=ModelNotFoundError("controlled missing model"),
            ), mock.patch(
                "urllib.request.OpenerDirector.open",
                side_effect=AssertionError("generate must remain offline"),
            ):
                with self.assertRaisesRegex(GenerationError, "not ready"):
                    generate_benchmark_dataset(
                        TEMPLATE_ROOT / "manifest.json",
                        output,
                        model_cache=cache,
                    )

            self.assertFalse(output.parent.exists())
            self.assertFalse(cache.exists())

    def test_generator_rejects_cross_family_fallback_before_model_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "generated"
            cache = root / "model-cache"
            with mock.patch(
                "ai_vectorizer.core.model_store.resolve_bundle",
                side_effect=AssertionError(
                    "invalid fallback must fail before model resolution"
                ),
            ):
                with self.assertRaisesRegex(GenerationError, "same-family"):
                    generate_benchmark_dataset(
                        TEMPLATE_ROOT / "manifest.json",
                        output,
                        fallback_backends={
                            EFFICIENTSAM_BACKEND: "canny-adaptive-v1"
                        },
                        model_cache=cache,
                    )

            self.assertFalse(output.exists())
            self.assertFalse(cache.exists())

    @unittest.skipUnless(
        os.environ.get("ARCHAEOTRACE_TEST_MODEL_CACHE"),
        "set ARCHAEOTRACE_TEST_MODEL_CACHE for the real pinned-model smoke",
    )
    def test_real_pinned_model_generates_deterministic_cpu_centerline(self):
        cache = Path(os.environ["ARCHAEOTRACE_TEST_MODEL_CACHE"])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated"
            generated = generate_benchmark_dataset(
                TEMPLATE_ROOT / "manifest.json",
                output,
                python_executable=sys.executable,
                model_cache=cache,
            )
            self.assertEqual(list(output.glob("worker-request--*.json")), [])
            self.assertEqual(list(output.glob("worker-result--*.json")), [])
            published_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*.json")
            )
            self.assertNotIn(str(cache.absolute()), published_text)
            generated_prediction = generated.samples[0].predictions[
                EFFICIENTSAM_BACKEND
            ]
            artifact = load_centerline_artifact(
                generated_prediction.artifact_path
            )

        prediction = generated.samples[0].predictions[EFFICIENTSAM_BACKEND]
        runtime = prediction.execution.runtime
        self.assertEqual(prediction.execution.status, "ok")
        self.assertEqual(runtime["actual_provider"], "CPUExecutionProvider")
        self.assertTrue(runtime["provider_verified"])
        self.assertTrue(runtime["deterministic"])
        self.assertEqual(len(set(runtime["output_sha256_samples"])), 1)
        self.assertEqual(runtime["model_bundle_sha256"], MODEL_FINGERPRINT)
        self.assertEqual(
            frozenset(runtime["source_files_sha256"]),
            EXPECTED_SAM_SOURCE_FILES,
        )
        self.assertEqual(len(runtime["sam_prediction_samples"]), 4)
        for sample in runtime["sam_prediction_samples"]:
            self.assertEqual(
                sample["schema_version"],
                benchmark_worker.EFFICIENTSAM_PREDICTION_EVIDENCE_VERSION,
            )
            self.assertEqual(len(sample["iou_predictions"]), 3)
            for key in (
                "iou_predictions_sha256",
                "selected_logits_sha256",
                "selected_binary_mask_sha256",
                "accepted_mask_sha256",
            ):
                self.assertRegex(sample[key], r"^[0-9a-f]{64}$")
        self.assertEqual(
            artifact.metadata["segmentation_evidence"],
            {
                key: value
                for key, value in runtime["sam_prediction_samples"][1].items()
                if key != "decoder_wall_ns"
            },
        )
        self.assertEqual(runtime["latency_scope"], benchmark_worker.LATENCY_SCOPE)
        self.assertEqual(
            runtime["image_load_wall_ns"],
            runtime["image_decode_wall_ns"],
        )
        self.assertEqual(len(runtime["warmup_wall_ns_samples"]), 1)
        self.assertGreaterEqual(
            runtime["warmup_wall_ns_samples"][0],
            runtime["edge_cache_fill_wall_ns"],
        )
        self.assertEqual(
            runtime["prompt_sha256"],
            prompt_sha256(generated.samples[0].prompt),
        )
        self.assertEqual(
            runtime["sam_prompt_tensor_sha256"],
            sam_prompt_tensor_sha256(generated.samples[0].prompt),
        )
        self.assertEqual(runtime["thread_settings"]["opencv_set_num_threads"], 0)
        self.assertEqual(runtime["thread_settings"]["opencv_effective_num_threads"], 1)
        self.assertIs(runtime["thread_settings"]["opencl"], False)
        self.assertEqual(
            runtime["onnx_session_options"],
            {
                "encoder": dict(benchmark_worker.EFFICIENTSAM_ORT_SESSION_OPTIONS),
                "decoder": dict(benchmark_worker.EFFICIENTSAM_ORT_SESSION_OPTIONS),
            },
        )

    def test_model_status_is_offline_and_does_not_create_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "absent-cache"
            stdout = io.StringIO()
            with mock.patch(
                "urllib.request.OpenerDirector.open",
                side_effect=AssertionError("status must remain offline"),
            ), mock.patch("sys.stdout", stdout):
                code = cli.main(
                    ["model", "status", "--model-cache", str(cache)]
                )

            self.assertEqual(code, 3)
            self.assertIn("MISSING model=efficientsam-ti-split-onnx", stdout.getvalue())
            self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
