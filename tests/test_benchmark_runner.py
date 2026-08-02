"""Integration tests for benchmark evaluation and report writing."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmarks.cli import main as benchmark_main
from benchmarks.runner import evaluate_benchmark, write_reports


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "benchmarks" / "data" / "synthetic-smoke"


class BenchmarkRunnerTests(unittest.TestCase):
    def test_synthetic_methods_expose_overlap_distance_and_connectivity(self):
        report = evaluate_benchmark(FIXTURE_ROOT / "manifest.json")
        by_method = {
            result["method"]["id"]: result
            for result in report["methods"]
        }

        perfect = by_method["perfect"]["samples"][0]["metrics"]
        offset = by_method["offset"]["samples"][0]["metrics"]
        broken = by_method["broken"]["samples"][0]["metrics"]
        self.assertEqual(perfect["cldice"], 1.0)
        self.assertEqual(perfect["distance"]["symmetric_mean"], 0.0)
        self.assertEqual(offset["cldice"], 0.0)
        self.assertEqual(offset["distance"]["symmetric_mean"], 1.0)
        self.assertEqual(broken["topology"]["prediction"]["components"], 2)
        self.assertEqual(broken["connectivity"]["summary"]["breaks"], 1)
        self.assertEqual(broken["connectivity"]["summary"]["fragment_excess"], 1)

    def test_method_selection_preserves_requested_order(self):
        report = evaluate_benchmark(
            FIXTURE_ROOT / "manifest.json",
            method_ids=["broken", "perfect"],
        )

        self.assertEqual(
            [result["method"]["id"] for result in report["methods"]],
            ["broken", "perfect"],
        )

    def test_prompt_latency_aliases_preserve_legacy_statistics(self):
        report = evaluate_benchmark(
            FIXTURE_ROOT / "manifest.json",
            method_ids=["perfect"],
        )

        sample_timing = report["methods"][0]["samples"][0]["execution"]["timing"]
        summary_timing = report["methods"][0]["summary"]["timing"]
        self.assertEqual(
            sample_timing["prompt_wall_ns_median"],
            sample_timing["wall_ns_median"],
        )
        self.assertEqual(
            sample_timing["prompt_wall_ns_p95"],
            sample_timing["wall_ns_p95"],
        )
        self.assertIsNone(sample_timing["image_load_wall_ns"])
        self.assertIsNone(sample_timing["estimated_image_first_prompt_wall_ns"])
        self.assertEqual(
            summary_timing["case_prompt_wall_ns_median"],
            summary_timing["case_wall_ns_median"],
        )
        self.assertEqual(
            summary_timing["case_prompt_wall_ns_p95"],
            summary_timing["case_wall_ns_p95"],
        )
        self.assertEqual(
            summary_timing["estimated_warm_prompt_pass_ns"],
            summary_timing["estimated_dataset_pass_ns"],
        )
        self.assertIsNone(summary_timing["estimated_image_load_pass_ns"])

    def test_image_and_cold_estimates_require_explicit_phase_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "fixture"
            shutil.copytree(FIXTURE_ROOT, fixture)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            runtime = payload["samples"][0]["predictions"]["perfect"][
                "execution"
            ]["runtime"]
            payload["samples"][0]["predictions"]["perfect"]["execution"][
                "timing"
            ]["model_load_wall_ns"] = 50
            runtime.update(
                latency_scope="warmed_predict_plus_canonical_artifact_v1",
                image_load_wall_ns=100,
                warmup_wall_ns_samples=[200],
            )
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = evaluate_benchmark(manifest_path, method_ids=["perfect"])

        sample_timing = report["methods"][0]["samples"][0]["execution"]["timing"]
        summary_timing = report["methods"][0]["summary"]["timing"]
        self.assertEqual(sample_timing["image_load_wall_ns"], 100)
        self.assertEqual(sample_timing["estimated_image_first_prompt_wall_ns"], 300)
        self.assertEqual(
            sample_timing["estimated_cold_worker_first_prompt_wall_ns"],
            350,
        )
        self.assertEqual(summary_timing["observed_image_load_count"], 1)
        self.assertEqual(summary_timing["estimated_image_load_pass_ns"], 100)
        self.assertEqual(summary_timing["observed_image_first_prompt_count"], 1)
        self.assertEqual(
            summary_timing["estimated_image_first_prompt_pass_ns"],
            300,
        )
        self.assertEqual(
            summary_timing["estimated_cold_worker_first_prompt_pass_ns"],
            350,
        )

    def test_failed_prediction_is_not_silently_removed_from_aggregate(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "fixture"
            shutil.copytree(FIXTURE_ROOT, fixture)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            prediction = payload["samples"][0]["predictions"]["perfect"]
            prediction.pop("artifact")
            prediction.pop("sha256")
            execution = prediction["execution"]
            execution.update(
                status="failed",
                actual_backend=None,
                fallback_reason=None,
                error="synthetic failure",
            )
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = evaluate_benchmark(manifest_path, method_ids=["perfect"])

        summary = report["methods"][0]["summary"]
        sample = report["methods"][0]["samples"][0]
        self.assertIsNone(sample["metrics"])
        self.assertEqual(summary["completion_rate"], 0.0)
        self.assertEqual(summary["primary"]["failure_adjusted_macro_f1"], 0.0)
        self.assertFalse(summary["eligible"])

    def test_fallback_result_is_scored_but_not_ranking_eligible(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "fixture"
            shutil.copytree(FIXTURE_ROOT, fixture)
            manifest_path = fixture / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            execution = payload["samples"][0]["predictions"]["perfect"]["execution"]
            execution.update(
                status="fallback",
                actual_backend="broken",
                fallback_reason="requested backend unavailable",
            )
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = evaluate_benchmark(manifest_path, method_ids=["perfect"])

        summary = report["methods"][0]["summary"]
        self.assertEqual(summary["completion_rate"], 1.0)
        self.assertEqual(summary["fallback_count"], 1)
        self.assertFalse(summary["eligible"])
        self.assertIsNotNone(report["methods"][0]["samples"][0]["metrics"])

    def test_reports_are_strict_json_and_flat_csv(self):
        report = evaluate_benchmark(FIXTURE_ROOT / "manifest.json")
        with tempfile.TemporaryDirectory() as folder:
            paths = write_reports(report, folder)
            parsed = json.loads(paths["json"].read_text(encoding="utf-8"))
            with paths["samples_csv"].open(encoding="utf-8", newline="") as handle:
                sample_rows = list(csv.DictReader(handle))
            with paths["summary_csv"].open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))

        self.assertEqual(parsed["schema_version"], "archaeotrace-contour-benchmark-report/1")
        self.assertEqual(len(sample_rows), 3)
        self.assertEqual(len(summary_rows), 3)
        self.assertIn("f1_t3", sample_rows[0])
        self.assertIn("prompt_wall_ns_median", sample_rows[0])
        self.assertIn("estimated_warm_prompt_pass_ns", summary_rows[0])

    def test_cli_validate_and_evaluate_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            validate_code = benchmark_main(
                ["validate", str(FIXTURE_ROOT / "manifest.json"), "--method", "perfect"]
            )
            evaluate_code = benchmark_main(
                [
                    "evaluate",
                    str(FIXTURE_ROOT / "manifest.json"),
                    "--method",
                    "perfect",
                    "--output",
                    folder,
                    "--require-eligible",
                ]
            )

            self.assertEqual(validate_code, 0)
            self.assertEqual(evaluate_code, 0)
            latest_path = Path(folder) / "benchmark_latest.json"
            self.assertTrue(latest_path.is_file())
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertTrue(
                (Path(folder) / latest["run_directory"] / "benchmark_report.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
