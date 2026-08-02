"""Regression tests for benchmark validation and reporting hardening."""

from contextlib import contextmanager
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import benchmarks.runner as benchmark_runner
from benchmarks.geometry import (
    CenterlineFormatError,
    load_centerline_artifact,
    rasterize_centerlines,
)
from benchmarks.manifest import ManifestError, image_dimensions, load_manifest
from benchmarks.metrics import compute_metrics
from benchmarks.runner import evaluate_benchmark, validate_benchmark, write_reports


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "benchmarks" / "data" / "synthetic-smoke"


@contextmanager
def copied_fixture():
    """Yield an isolated synthetic fixture and its decoded manifest."""

    with tempfile.TemporaryDirectory() as folder:
        fixture = Path(folder) / "fixture"
        shutil.copytree(FIXTURE_ROOT, fixture)
        manifest_path = fixture / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        yield manifest_path, payload


def write_manifest(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class BenchmarkHardeningTests(unittest.TestCase):
    def test_failed_sample_resources_remain_in_method_timing_aggregate(self):
        with copied_fixture() as (manifest_path, payload):
            prediction = payload["samples"][0]["predictions"]["perfect"]
            prediction.pop("artifact")
            prediction.pop("sha256")
            execution = prediction["execution"]
            execution.update(
                status="failed",
                actual_backend=None,
                fallback_reason=None,
                error="synthetic out-of-memory failure",
            )
            execution["runtime"].update(
                deterministic=None,
                output_sha256_samples=[],
            )
            execution["timing"].update(
                wall_ns_samples=[9_000_000_000, 10_000_000_000, 11_000_000_000],
                cpu_ns_samples=[7_000_000_000, 8_000_000_000, 9_000_000_000],
                model_load_wall_ns=4_000_000_000,
                peak_rss_bytes=8_000_000_000,
            )
            write_manifest(manifest_path, payload)

            report = evaluate_benchmark(manifest_path, method_ids=["perfect"])

        summary = report["methods"][0]["summary"]
        timing = summary["timing"]
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["completed_count"], 0)
        self.assertEqual(timing["case_wall_ns_median"], 10_000_000_000)
        self.assertEqual(timing["case_wall_ns_mean"], 10_000_000_000)
        self.assertEqual(timing["case_wall_ns_p95"], 10_000_000_000)
        self.assertEqual(timing["estimated_dataset_pass_ns"], 10_000_000_000)
        self.assertEqual(timing["case_cpu_ns_median"], 8_000_000_000)
        self.assertEqual(timing["case_cpu_ns_mean"], 8_000_000_000)
        self.assertEqual(timing["case_cpu_ns_p95"], 8_000_000_000)
        self.assertEqual(timing["model_load_wall_ns_max"], 4_000_000_000)
        self.assertEqual(timing["worker_peak_rss_bytes_max"], 8_000_000_000)

    def test_ok_status_rejects_a_different_actual_backend(self):
        with copied_fixture() as (manifest_path, payload):
            execution = payload["samples"][0]["predictions"]["perfect"][
                "execution"
            ]
            execution["actual_backend"] = "offset"
            write_manifest(manifest_path, payload)

            with self.assertRaises(ManifestError):
                validate_benchmark(manifest_path, method_ids=["perfect"])

    def test_nonfinite_thread_setting_is_rejected_during_validation(self):
        marker = "__THREAD_SETTING_OVERFLOW__"
        with copied_fixture() as (manifest_path, payload):
            thread_settings = payload["samples"][0]["predictions"]["perfect"][
                "execution"
            ]["runtime"]["thread_settings"]
            thread_settings["overflow"] = marker
            manifest_text = json.dumps(payload, indent=2, ensure_ascii=False)
            manifest_path.write_text(
                manifest_text.replace(json.dumps(marker), "1e400"),
                encoding="utf-8",
            )

            with self.assertRaises(ManifestError):
                validate_benchmark(manifest_path, method_ids=["perfect"])

    def test_nondeterministic_result_is_scored_but_not_eligible(self):
        with copied_fixture() as (manifest_path, payload):
            prediction = payload["samples"][0]["predictions"]["perfect"]
            published_hash = prediction["sha256"]
            runtime = prediction["execution"]["runtime"]
            runtime["deterministic"] = False
            runtime["output_sha256_samples"] = [
                published_hash,
                "0" * 64,
                "1" * 64,
            ]
            write_manifest(manifest_path, payload)

            report = evaluate_benchmark(manifest_path, method_ids=["perfect"])

        method_result = report["methods"][0]
        self.assertIsNotNone(method_result["samples"][0]["metrics"])
        self.assertEqual(method_result["summary"]["nondeterministic_count"], 1)
        self.assertFalse(method_result["summary"]["eligible"])

    def test_clipped_closed_path_with_separated_endpoints_becomes_open(self):
        artifact_payload = {
            "schema_version": "archaeotrace-centerline/1",
            "coordinate_space": "pixel_xy",
            "image_size": {"width": 5, "height": 5},
            "paths": [
                {
                    "id": "clipped-ring",
                    "closed": True,
                    "points": [[-10, -10], [-10, -3], [2, 3]],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            artifact_path = Path(folder) / "clipped.json"
            artifact_path.write_text(
                json.dumps(artifact_payload),
                encoding="utf-8",
            )
            raster = rasterize_centerlines(load_centerline_artifact(artifact_path))

        self.assertEqual(len(raster.paths), 1)
        path = raster.paths[0]
        first, last = path.pixels[0], path.pixels[-1]
        self.assertGreater(
            max(abs(first[0] - last[0]), abs(first[1] - last[1])),
            1,
        )
        self.assertFalse(path.closed)

    def test_clipped_closed_path_with_adjacent_crop_endpoints_is_still_open(self):
        artifact_payload = {
            "schema_version": "archaeotrace-centerline/1",
            "coordinate_space": "pixel_xy",
            "image_size": {"width": 5, "height": 5},
            "paths": [
                {
                    "id": "clipped-u",
                    "closed": True,
                    "points": [[-2, 1], [1, 1], [1, 2], [-2, 2]],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            artifact_path = Path(folder) / "clipped-adjacent.json"
            artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
            raster = rasterize_centerlines(load_centerline_artifact(artifact_path))

        self.assertEqual(len(raster.paths), 1)
        self.assertFalse(raster.paths[0].closed)

    def test_retraced_segments_cannot_expand_beyond_canvas_pixel_budget(self):
        artifact_payload = {
            "schema_version": "archaeotrace-centerline/1",
            "coordinate_space": "pixel_xy",
            "image_size": {"width": 1024, "height": 1},
            "paths": [
                {
                    "id": "retrace",
                    "points": [[0, 0], [1023, 0], [0, 0]],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            artifact_path = Path(folder) / "retrace.json"
            artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
            artifact = load_centerline_artifact(artifact_path)
            with self.assertRaisesRegex(CenterlineFormatError, "pixel budget"):
                rasterize_centerlines(artifact)

    def test_close_tolerances_get_distinct_csv_columns(self):
        tolerances = [1.0000000000001, 1.0000000000002]
        with copied_fixture() as (manifest_path, payload):
            metric_config = payload["metric_config"]
            metric_config["tolerances_px"] = tolerances
            metric_config["primary_tolerance_px"] = tolerances[0]
            metric_config["branch_tolerance_px"] = tolerances[0]
            write_manifest(manifest_path, payload)
            report = evaluate_benchmark(manifest_path, method_ids=["perfect"])

            with tempfile.TemporaryDirectory() as output_folder:
                paths = write_reports(report, output_folder)
                with paths["samples_csv"].open(
                    encoding="utf-8", newline=""
                ) as handle:
                    header = next(csv.reader(handle))

        self.assertEqual(len(header), len(set(header)))
        for prefix in ("precision_t", "recall_t", "f1_t"):
            columns = [column for column in header if column.startswith(prefix)]
            self.assertEqual(len(columns), 2)
            self.assertEqual(len(columns), len(set(columns)))

    def test_timing_environment_must_match_across_methods(self):
        with copied_fixture() as (manifest_path, payload):
            payload["samples"][0]["predictions"]["offset"]["execution"][
                "runtime"
            ]["cpu"] = "different CPU"
            write_manifest(manifest_path, payload)

            with self.assertRaisesRegex(ManifestError, "share python, platform, CPU"):
                load_manifest(manifest_path)

    def test_one_method_cannot_mix_adapter_versions_across_samples(self):
        with copied_fixture() as (manifest_path, payload):
            second = copy.deepcopy(payload["samples"][0])
            second["id"] = "straight-line-2"
            second["predictions"]["perfect"]["execution"]["runtime"][
                "adapter_version"
            ] = "synthetic/2"
            payload["samples"].append(second)
            write_manifest(manifest_path, payload)

            with self.assertRaisesRegex(ManifestError, "adapter/provider/package"):
                load_manifest(manifest_path)

    def test_fallback_may_use_a_distinct_runtime_but_is_not_eligible(self):
        with copied_fixture() as (manifest_path, payload):
            second = copy.deepcopy(payload["samples"][0])
            second["id"] = "straight-line-fallback"
            perfect = second["predictions"]["perfect"]
            offset = second["predictions"]["offset"]
            perfect["artifact"] = offset["artifact"]
            perfect["sha256"] = offset["sha256"]
            execution = perfect["execution"]
            execution.update(
                status="fallback",
                actual_backend="offset",
                fallback_reason="synthetic requested backend failure",
                error=None,
            )
            execution["runtime"].update(
                adapter_version="opencv/1",
                provider_kind="opencv",
                actual_provider="OpenCV CPU",
                package_versions={"opencv": "synthetic"},
                output_sha256_samples=offset["execution"]["runtime"][
                    "output_sha256_samples"
                ],
            )
            payload["samples"].append(second)
            write_manifest(manifest_path, payload)

            report = evaluate_benchmark(manifest_path, method_ids=["perfect"])

        summary = report["methods"][0]["summary"]
        self.assertEqual(summary["completed_count"], 2)
        self.assertEqual(summary["fallback_count"], 1)
        self.assertFalse(summary["eligible"])

    def test_manifest_text_rejects_terminal_control_characters(self):
        marker = "__CONTROL__"
        with copied_fixture() as (manifest_path, payload):
            payload["dataset"]["version"] = marker
            manifest_path.write_text(
                json.dumps(payload).replace(json.dumps(marker), '"1.0\\nforged"'),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ManifestError, "control characters"):
                load_manifest(manifest_path)

    def test_escaped_lone_surrogate_is_rejected_during_validation(self):
        marker = "__SURROGATE__"
        with copied_fixture() as (manifest_path, payload):
            payload["methods"][0]["label"] = marker
            manifest_path.write_text(
                json.dumps(payload).replace(json.dumps(marker), '"\\ud800"'),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ManifestError, "surrogate"):
                validate_benchmark(manifest_path)

    def test_escaped_lone_surrogate_object_key_is_rejected(self):
        marker = "__SURROGATE_KEY__"
        with copied_fixture() as (manifest_path, payload):
            payload["methods"][0]["configuration"][marker] = "value"
            manifest_path.write_text(
                json.dumps(payload).replace(json.dumps(marker), '"\\ud800"'),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ManifestError, "surrogate"):
                load_manifest(manifest_path)

    def test_geometry_identifier_rejects_terminal_control_characters(self):
        artifact_payload = {
            "schema_version": "archaeotrace-centerline/1",
            "coordinate_space": "pixel_xy",
            "image_size": {"width": 2, "height": 1},
            "paths": [{"id": "\x1b[2J", "points": [[0, 0], [1, 0]]}],
        }
        with tempfile.TemporaryDirectory() as folder:
            artifact_path = Path(folder) / "control-id.json"
            artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
            with self.assertRaisesRegex(CenterlineFormatError, "control characters"):
                load_centerline_artifact(artifact_path)

    def test_truncated_pnm_image_is_rejected_even_when_hash_matches(self):
        with copied_fixture() as (manifest_path, payload):
            image_path = manifest_path.parent / payload["samples"][0]["image"]
            image_path.write_bytes(b"P2\n9 9\n")
            payload["samples"][0]["image_sha256"] = hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest()
            write_manifest(manifest_path, payload)

            with self.assertRaisesRegex(ManifestError, "PNM"):
                load_manifest(manifest_path)

    def test_png_requires_valid_chunks_crc_and_decoded_scanline(self):
        def chunk(chunk_type, data):
            return (
                struct.pack(">I", len(data))
                + chunk_type
                + data
                + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
            )

        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00\x7f"))
            + chunk(b"IEND", b"")
        )
        with tempfile.TemporaryDirectory() as folder:
            valid_path = Path(folder) / "valid.png"
            valid_path.write_bytes(png)
            self.assertEqual(image_dimensions(valid_path), (1, 1))

            corrupt_path = Path(folder) / "corrupt.png"
            corrupted = bytearray(png)
            corrupted[-1] ^= 0x01
            corrupt_path.write_bytes(corrupted)
            with self.assertRaisesRegex(ManifestError, "CRC"):
                image_dimensions(corrupt_path)

    def test_metric_tolerance_has_a_schema_v1_complexity_bound(self):
        mask = [[True]]
        with self.assertRaisesRegex(ValueError, "between 0 and 8"):
            compute_metrics(mask, mask, tolerances=[9], primary_tolerance=9)

    def test_provider_name_must_match_structured_provider_kind(self):
        with copied_fixture() as (manifest_path, payload):
            runtime = payload["samples"][0]["predictions"]["perfect"][
                "execution"
            ]["runtime"]
            runtime["actual_provider"] = "CUDAExecutionProvider CPU"
            write_manifest(manifest_path, payload)

            with self.assertRaisesRegex(ManifestError, "must equal 'synthetic CPU'"):
                load_manifest(manifest_path)

    def test_mutated_loaded_manifest_is_reloaded_before_evaluation(self):
        manifest = load_manifest(FIXTURE_ROOT / "manifest.json")
        manifest.samples[0].predictions["perfect"].execution.runtime.update(
            actual_provider="CUDAExecutionProvider CPU",
            provider_device_type="gpu",
            cpu="different CPU",
        )

        report = evaluate_benchmark(manifest, method_ids=["perfect"])

        runtime = report["methods"][0]["samples"][0]["execution"]["runtime"]
        self.assertEqual(runtime["actual_provider"], "synthetic CPU")
        self.assertEqual(runtime["provider_device_type"], "cpu")
        self.assertEqual(report["timing_environment"]["cpu"], "synthetic")

    def test_report_set_hashes_match_and_failed_write_keeps_previous_generation(self):
        report = evaluate_benchmark(FIXTURE_ROOT / "manifest.json", method_ids=["perfect"])
        with tempfile.TemporaryDirectory() as folder:
            paths = write_reports(report, folder)
            baseline = {
                key: paths[key].read_bytes()
                for key in ("json", "samples_csv", "summary_csv", "commit")
            }
            latest_path = Path(folder) / "benchmark_latest.json"
            baseline_latest = latest_path.read_bytes()

            report_payload = json.loads(baseline["json"])
            self.assertEqual(
                report_payload["report_files"]["samples_csv"]["sha256"],
                hashlib.sha256(baseline["samples_csv"]).hexdigest(),
            )
            self.assertEqual(
                report_payload["report_files"]["summary_csv"]["sha256"],
                hashlib.sha256(baseline["summary_csv"]).hexdigest(),
            )
            commit = json.loads(baseline["commit"])
            self.assertEqual(
                commit["files"]["benchmark_report.json"],
                hashlib.sha256(baseline["json"]).hexdigest(),
            )

            original_atomic_csv = benchmark_runner._atomic_csv
            calls = 0

            def fail_on_second_csv(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected report write failure")
                return original_atomic_csv(*args, **kwargs)

            changed_report = dict(report)
            changed_report["generated_at"] = "2099-01-01T00:00:00Z"
            with mock.patch.object(
                benchmark_runner,
                "_atomic_csv",
                side_effect=fail_on_second_csv,
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    write_reports(changed_report, folder)

            self.assertEqual(latest_path.read_bytes(), baseline_latest)
            for key in ("json", "samples_csv", "summary_csv", "commit"):
                self.assertEqual(paths[key].read_bytes(), baseline[key])
            self.assertEqual(
                [path.name for path in (Path(folder) / "runs").iterdir()],
                [paths["run_dir"].name],
            )

    def test_report_runs_symlink_cannot_escape_output_directory(self):
        report = evaluate_benchmark(FIXTURE_ROOT / "manifest.json", method_ids=["perfect"])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            outside = root / "outside"
            output.mkdir()
            outside.mkdir()
            (output / "runs").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(benchmark_runner.BenchmarkError, "symbolic link"):
                write_reports(report, output)

            self.assertFalse(any(outside.iterdir()))


if __name__ == "__main__":
    unittest.main()
