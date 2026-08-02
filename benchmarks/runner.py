"""Benchmark evaluation, aggregation, and atomic report writing."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import uuid

from .geometry import load_centerline_artifact, rasterize_centerlines
from .manifest import (
    BenchmarkManifest,
    ExecutionRecord,
    MethodSpec,
    REQUIRED_STRATA,
    SampleSpec,
    load_manifest,
)
from .metrics import compute_metrics


REPORT_SCHEMA_VERSION = "archaeotrace-contour-benchmark-report/1"
HARNESS_VERSION = "0.1.0"
REPORT_JSON_NAME = "benchmark_report.json"
SAMPLE_CSV_NAME = "benchmark_samples.csv"
SUMMARY_CSV_NAME = "benchmark_summary.csv"
COMMIT_JSON_NAME = "benchmark_commit.json"
LATEST_JSON_NAME = "benchmark_latest.json"
REPORT_SET_SCHEMA_VERSION = "archaeotrace-contour-benchmark-report-set/1"
REPORT_POINTER_SCHEMA_VERSION = "archaeotrace-contour-benchmark-report-pointer/1"


class BenchmarkError(RuntimeError):
    """Raised when validated benchmark artifacts cannot be compared."""


def _mean(values: Iterable[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _nearest_rank(values: Iterable[float | int], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _median(values: Iterable[int]) -> float | None:
    values = tuple(values)
    return float(statistics.median(values)) if values else None


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _tolerance_label(tolerance: float) -> str:
    if float(tolerance).is_integer():
        return str(int(tolerance))
    return (
        repr(float(tolerance))
        .replace(".", "_")
        .replace("+", "p")
        .replace("-", "m")
    )


def _primary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    primary = float(metrics["primary_tolerance"])
    for record in metrics["tolerance_metrics"]:
        if float(record["tolerance"]) == primary:
            return record
    raise BenchmarkError(f"Metrics omitted primary tolerance {primary}.")


def _runtime_nonnegative_integer(runtime: dict[str, Any], key: str) -> int | None:
    """Return an optional timing-evidence integer without accepting booleans."""

    value = runtime.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _first_warmup_wall_ns(runtime: dict[str, Any]) -> int | None:
    samples = runtime.get("warmup_wall_ns_samples")
    if not isinstance(samples, list) or not samples:
        return None
    value = samples[0]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _execution_payload(execution: ExecutionRecord) -> dict[str, Any]:
    timing = execution.timing
    runtime = execution.runtime
    wall_median = _median(timing.wall_ns_samples)
    wall_p95 = _nearest_rank(timing.wall_ns_samples, 0.95)
    image_load_wall_ns = _runtime_nonnegative_integer(runtime, "image_load_wall_ns")
    first_warmup_wall_ns = _first_warmup_wall_ns(runtime)
    estimated_image_first_prompt_wall_ns = (
        image_load_wall_ns + first_warmup_wall_ns
        if image_load_wall_ns is not None and first_warmup_wall_ns is not None
        else None
    )
    estimated_cold_worker_first_prompt_wall_ns = (
        timing.model_load_wall_ns + estimated_image_first_prompt_wall_ns
        if timing.model_load_wall_ns is not None
        and estimated_image_first_prompt_wall_ns is not None
        else None
    )
    return {
        "status": execution.status,
        "requested_backend": execution.requested_backend,
        "actual_backend": execution.actual_backend,
        "fallback_reason": execution.fallback_reason,
        "error": execution.error,
        "device": execution.device,
        "runtime": runtime,
        "timing": {
            "warmup_runs": timing.warmup_runs,
            "wall_ns_samples": list(timing.wall_ns_samples),
            "cpu_ns_samples": list(timing.cpu_ns_samples),
            "wall_ns_median": wall_median,
            "wall_ns_p95": wall_p95,
            "prompt_wall_ns_median": wall_median,
            "prompt_wall_ns_p95": wall_p95,
            "cpu_ns_median": _median(timing.cpu_ns_samples),
            "cpu_ns_p95": _nearest_rank(timing.cpu_ns_samples, 0.95),
            "model_load_wall_ns": timing.model_load_wall_ns,
            "image_load_wall_ns": image_load_wall_ns,
            "estimated_image_first_prompt_wall_ns": (
                estimated_image_first_prompt_wall_ns
            ),
            "estimated_cold_worker_first_prompt_wall_ns": (
                estimated_cold_worker_first_prompt_wall_ns
            ),
            "peak_rss_bytes": timing.peak_rss_bytes,
        },
    }


def _method_payload(method: MethodSpec) -> dict[str, Any]:
    return {
        "id": method.identifier,
        "label": method.label,
        "kind": method.kind,
        "source": method.source,
        "version": method.version,
        "license": method.license,
        "model_sha256": method.model_sha256,
        "configuration": method.configuration,
    }


def _load_raster(
    path: Path,
    sample: SampleSpec,
    label: str,
    expected_sha256: str,
):
    artifact = load_centerline_artifact(path)
    if artifact.sha256 != expected_sha256:
        raise BenchmarkError(
            f"{label} hash changed: expected {expected_sha256}, got {artifact.sha256}."
        )
    if (artifact.width, artifact.height) != (sample.width, sample.height):
        raise BenchmarkError(
            f"{label} canvas is {artifact.width}x{artifact.height}; expected "
            f"{sample.width}x{sample.height}."
        )
    return artifact, rasterize_centerlines(artifact)


def _selected_methods(
    manifest: BenchmarkManifest,
    method_ids: Iterable[str] | None,
) -> tuple[MethodSpec, ...]:
    if method_ids is None:
        return manifest.methods
    requested = tuple(method_ids)
    if not requested:
        raise BenchmarkError("At least one method must be selected.")
    if len(set(requested)) != len(requested):
        raise BenchmarkError("Selected method ids must be unique.")
    by_id = {method.identifier: method for method in manifest.methods}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise BenchmarkError(f"Unknown benchmark methods: {', '.join(unknown)}")
    return tuple(by_id[identifier] for identifier in requested)


def validate_benchmark(
    manifest_or_path: BenchmarkManifest | str | Path,
    method_ids: Iterable[str] | None = None,
) -> BenchmarkManifest:
    """Validate hashes, dimensions, centerline formats, and selected methods."""

    if isinstance(manifest_or_path, BenchmarkManifest):
        manifest = load_manifest(manifest_or_path.path)
        if manifest.sha256 != manifest_or_path.sha256:
            raise BenchmarkError("Manifest bytes changed after the supplied object was loaded.")
    else:
        manifest = load_manifest(manifest_or_path)
    methods = _selected_methods(manifest, method_ids)
    for sample in manifest.samples:
        reference, reference_raster = _load_raster(
            sample.reference_path,
            sample,
            f"Sample {sample.identifier} reference",
            sample.reference_sha256,
        )
        if not reference_raster.paths:
            raise BenchmarkError(f"Sample {sample.identifier} reference has no visible path.")
        for method in methods:
            prediction = sample.predictions[method.identifier]
            if prediction.artifact_path is None:
                continue
            artifact, _raster = _load_raster(
                prediction.artifact_path,
                sample,
                f"Sample {sample.identifier} method {method.identifier}",
                prediction.artifact_sha256,
            )
    return manifest


def _git_environment(repository_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _dependency_versions() -> dict[str, str | None]:
    versions = {}
    for distribution in (
        "numpy",
        "opencv-python-headless",
        "scikit-image",
        "onnxruntime",
        "psutil",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _environment(repository_root: Path) -> dict[str, Any]:
    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "ORT_NUM_THREADS",
        )
    }
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "dependencies": _dependency_versions(),
        "thread_environment": thread_variables,
        "git": _git_environment(repository_root),
    }


def _harness_identity(repository_root: Path) -> dict[str, Any]:
    """Fingerprint the exact dependency-free evaluator source used for a run."""

    digest = hashlib.sha256()
    source_files = {}
    for path in sorted((repository_root / "benchmarks").glob("*.py")):
        raw = path.read_bytes()
        relative = path.relative_to(repository_root).as_posix()
        file_digest = hashlib.sha256(raw).hexdigest()
        source_files[relative] = file_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
    return {
        "name": "ArchaeoTrace contour benchmark",
        "version": HARNESS_VERSION,
        "source_sha256": digest.hexdigest(),
        "source_files": source_files,
    }


def _timing_environment(manifest: BenchmarkManifest) -> dict[str, Any]:
    first_sample = manifest.samples[0]
    first_prediction = next(iter(first_sample.predictions.values()))
    runtime = first_prediction.execution.runtime
    return {
        "python_version": runtime["python_version"],
        "platform": runtime["platform"],
        "cpu": runtime["cpu"],
        "thread_settings": runtime["thread_settings"],
    }


def _sample_result(
    sample: SampleSpec,
    method: MethodSpec,
    reference_raster,
    metric_config,
) -> dict[str, Any]:
    prediction = sample.predictions[method.identifier]
    result = {
        "sample_id": sample.identifier,
        "canvas": {"width": sample.width, "height": sample.height},
        "image_sha256": sample.image_sha256,
        "reference_sha256": sample.reference_sha256,
        "prediction_sha256": prediction.artifact_sha256,
        "strata": sample.strata,
        "source": sample.source,
        "execution": _execution_payload(prediction.execution),
        "metrics": None,
    }
    if prediction.artifact_path is None:
        return result

    _artifact, prediction_raster = _load_raster(
        prediction.artifact_path,
        sample,
        f"Sample {sample.identifier} method {method.identifier}",
        prediction.artifact_sha256,
    )
    reference_paths = tuple(
        (path.pixels, path.closed) for path in reference_raster.paths
    )
    metrics = compute_metrics(
        prediction_raster.mask,
        reference_raster.mask,
        reference_paths=reference_paths,
        tolerances=metric_config.tolerances_px,
        primary_tolerance=metric_config.primary_tolerance_px,
    )
    metrics["cldice_mode"] = "centerline_identity"
    result["metrics"] = metrics
    return result


def _strata_summary(sample_results: list[dict[str, Any]], primary: float) -> dict[str, Any]:
    output = {}
    for stratum in REQUIRED_STRATA:
        values = {}
        for result in sample_results:
            value = result["strata"][stratum]
            values.setdefault(value, []).append(result)
        output[stratum] = {}
        for value, results in sorted(values.items()):
            f1_values = [
                _primary_metrics(result["metrics"])["f1"]
                if result["metrics"] is not None
                else 0.0
                for result in results
            ]
            completed = sum(result["metrics"] is not None for result in results)
            output[stratum][value] = {
                "sample_count": len(results),
                "completed_count": completed,
                "completion_rate": completed / len(results),
                f"failure_adjusted_macro_f1_at_{_tolerance_label(primary)}px": _mean(f1_values),
            }
    return output


def _aggregate_method(
    sample_results: list[dict[str, Any]],
    metric_config,
) -> dict[str, Any]:
    total = len(sample_results)
    completed_results = [result for result in sample_results if result["metrics"] is not None]
    completed = len(completed_results)
    failed = total - completed
    fallback_count = sum(
        result["execution"]["status"] == "fallback" for result in sample_results
    )
    nondeterministic_count = sum(
        result["metrics"] is not None
        and result["execution"]["runtime"].get("deterministic") is not True
        for result in sample_results
    )
    primary = metric_config.primary_tolerance_px

    tolerance_aggregates = []
    for tolerance in metric_config.tolerances_px:
        completed_records = []
        for result in completed_results:
            record = next(
                item
                for item in result["metrics"]["tolerance_metrics"]
                if float(item["tolerance"]) == float(tolerance)
            )
            completed_records.append(record)
        matched_prediction = sum(record["matched_prediction_pixels"] for record in completed_records)
        total_prediction = sum(record["total_prediction_pixels"] for record in completed_records)
        matched_reference = sum(record["matched_reference_pixels"] for record in completed_records)
        total_reference = sum(record["total_reference_pixels"] for record in completed_records)
        if total_prediction and total_reference:
            micro_precision = matched_prediction / total_prediction
            micro_recall = matched_reference / total_reference
            micro_f1 = _f1(micro_precision, micro_recall)
        elif completed_records and all(
            record["total_prediction_pixels"] == 0 and record["total_reference_pixels"] == 0
            for record in completed_records
        ):
            micro_precision = micro_recall = micro_f1 = 1.0
        else:
            micro_precision = micro_recall = micro_f1 = 0.0
        macro_values = [record["f1"] for record in completed_records]
        tolerance_aggregates.append(
            {
                "tolerance": tolerance,
                "completed_macro_precision": _mean(record["precision"] for record in completed_records),
                "completed_macro_recall": _mean(record["recall"] for record in completed_records),
                "completed_macro_f1": _mean(macro_values),
                "failure_adjusted_macro_f1": (
                    sum(macro_values) / total if total else None
                ),
                "completed_micro_precision": micro_precision,
                "completed_micro_recall": micro_recall,
                "completed_micro_f1": micro_f1,
                "matched_prediction_pixels": matched_prediction,
                "total_prediction_pixels": total_prediction,
                "matched_reference_pixels": matched_reference,
                "total_reference_pixels": total_reference,
            }
        )

    wall_case_medians = [
        result["execution"]["timing"]["wall_ns_median"]
        for result in sample_results
        if result["execution"]["timing"]["wall_ns_median"] is not None
    ]
    cpu_case_medians = [
        result["execution"]["timing"]["cpu_ns_median"]
        for result in sample_results
        if result["execution"]["timing"]["cpu_ns_median"] is not None
    ]
    peak_values = [
        result["execution"]["timing"]["peak_rss_bytes"]
        for result in sample_results
        if result["execution"]["timing"]["peak_rss_bytes"] is not None
    ]
    model_load_values = [
        result["execution"]["timing"]["model_load_wall_ns"]
        for result in sample_results
        if result["execution"]["timing"]["model_load_wall_ns"] is not None
    ]
    image_load_values = [
        result["execution"]["timing"]["image_load_wall_ns"]
        for result in sample_results
        if result["execution"]["timing"]["image_load_wall_ns"] is not None
    ]
    image_first_prompt_values = [
        result["execution"]["timing"]["estimated_image_first_prompt_wall_ns"]
        for result in sample_results
        if result["execution"]["timing"]["estimated_image_first_prompt_wall_ns"]
        is not None
    ]
    cold_worker_first_prompt_values = [
        result["execution"]["timing"][
            "estimated_cold_worker_first_prompt_wall_ns"
        ]
        for result in sample_results
        if result["execution"]["timing"][
            "estimated_cold_worker_first_prompt_wall_ns"
        ]
        is not None
    ]
    distance_results = [
        result["metrics"]["distance"]
        for result in completed_results
        if result["metrics"]["distance"]["symmetric_mean"] is not None
    ]
    connectivity = [
        result["metrics"]["connectivity"]["summary"]
        for result in completed_results
    ]
    topology = [result["metrics"]["topology"] for result in completed_results]
    primary_record = next(
        record
        for record in tolerance_aggregates
        if float(record["tolerance"]) == float(primary)
    )
    return {
        "sample_count": total,
        "completed_count": completed,
        "failed_count": failed,
        "fallback_count": fallback_count,
        "nondeterministic_count": nondeterministic_count,
        "completion_rate": completed / total if total else 0.0,
        "eligible": (
            completed == total
            and fallback_count == 0
            and nondeterministic_count == 0
        ),
        "publication_ranking_eligible": False,
        "primary_tolerance_px": primary,
        "primary": primary_record,
        "by_tolerance": tolerance_aggregates,
        "completed_macro_cldice": _mean(
            result["metrics"]["cldice"] for result in completed_results
        ),
        "distance": {
            "valid_sample_count": len(distance_results),
            "valid_sample_fraction": len(distance_results) / total if total else 0.0,
            "macro_symmetric_mean_px": _mean(
                record["symmetric_mean"] for record in distance_results
            ),
            "macro_symmetric_p95_px": _mean(
                record["symmetric_p95"] for record in distance_results
            ),
        },
        "topology": {
            "breaks": sum(record["breaks"] for record in connectivity),
            "fragment_excess": sum(record["fragment_excess"] for record in connectivity),
            "missed_paths": sum(record["missed_paths"] for record in connectivity),
            "unmatched_prediction_branch_zones": sum(
                record["unmatched_prediction_branch_zones"] for record in topology
            ),
            "macro_coverage_ratio": _mean(record["coverage_ratio"] for record in connectivity),
            "macro_longest_fragment_ratio": _mean(
                record["longest_fragment_ratio"] for record in connectivity
            ),
        },
        "timing": {
            "observed_wall_sample_count": len(wall_case_medians),
            "observed_cpu_sample_count": len(cpu_case_medians),
            "case_wall_ns_median": _median(int(value) for value in wall_case_medians),
            "case_wall_ns_mean": _mean(wall_case_medians),
            "case_wall_ns_p95": _nearest_rank(wall_case_medians, 0.95),
            "estimated_dataset_pass_ns": sum(wall_case_medians),
            "case_prompt_wall_ns_median": _median(
                int(value) for value in wall_case_medians
            ),
            "case_prompt_wall_ns_p95": _nearest_rank(wall_case_medians, 0.95),
            "estimated_warm_prompt_pass_ns": sum(wall_case_medians),
            "case_cpu_ns_median": _median(int(value) for value in cpu_case_medians),
            "case_cpu_ns_mean": _mean(cpu_case_medians),
            "case_cpu_ns_p95": _nearest_rank(cpu_case_medians, 0.95),
            "model_load_wall_ns_max": max(model_load_values) if model_load_values else None,
            "observed_image_load_count": len(image_load_values),
            "estimated_image_load_pass_ns": (
                sum(image_load_values) if image_load_values else None
            ),
            "observed_image_first_prompt_count": len(image_first_prompt_values),
            "estimated_image_first_prompt_pass_ns": (
                sum(image_first_prompt_values) if image_first_prompt_values else None
            ),
            "observed_cold_worker_first_prompt_count": len(
                cold_worker_first_prompt_values
            ),
            "estimated_cold_worker_first_prompt_pass_ns": (
                sum(cold_worker_first_prompt_values)
                if cold_worker_first_prompt_values
                else None
            ),
            "worker_peak_rss_bytes_max": max(peak_values) if peak_values else None,
        },
        "strata": _strata_summary(sample_results, primary),
    }


def evaluate_benchmark(
    manifest_or_path: BenchmarkManifest | str | Path,
    method_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate selected precomputed methods and return a strict report object."""

    manifest = validate_benchmark(manifest_or_path, method_ids)
    methods = _selected_methods(manifest, method_ids)
    method_results = []
    for method in methods:
        samples = []
        for sample in manifest.samples:
            _reference, reference_raster = _load_raster(
                sample.reference_path,
                sample,
                f"Sample {sample.identifier} reference",
                sample.reference_sha256,
            )
            samples.append(
                _sample_result(
                    sample,
                    method,
                    reference_raster,
                    manifest.metric_config,
                )
            )
        method_results.append(
            {
                "method": _method_payload(method),
                "summary": _aggregate_method(samples, manifest.metric_config),
                "samples": samples,
            }
        )

    repository_root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "harness": _harness_identity(repository_root),
        "environment": _environment(repository_root),
        "timing_environment": _timing_environment(manifest),
        "provenance": {
            "level": "manifest_attested_precomputed",
            "publication_ranking_eligible": False,
            "note": (
                "The evaluator reads execution evidence from the manifest. "
                "Generated datasets preserve isolated-worker records, while "
                "manually imported records remain self-attested."
            ),
        },
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
        },
        "dataset": {
            "id": manifest.dataset.identifier,
            "version": manifest.dataset.version,
            "description": manifest.dataset.description,
            "license": manifest.dataset.license,
            "source": manifest.dataset.source,
            "sample_count": len(manifest.samples),
        },
        "metric_config": {
            "primary_tolerance_px": manifest.metric_config.primary_tolerance_px,
            "tolerances_px": list(manifest.metric_config.tolerances_px),
            "branch_tolerance_px": manifest.metric_config.branch_tolerance_px,
            "connectivity": manifest.metric_config.connectivity,
            "diagonal_rule": manifest.metric_config.diagonal_rule,
            "percentile": manifest.metric_config.percentile,
        },
        "methods": method_results,
    }
    json.dumps(report, allow_nan=False)
    return report


def _atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(temporary)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = _atomic_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if len(fieldnames) != len(set(fieldnames)):
        raise BenchmarkError("CSV field names must be unique.")
    temporary = _atomic_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(
                {
                    key: _csv_safe(value)
                    for key, value in row.items()
                }
                for row in rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_safe(value: Any) -> Any:
    """Prevent spreadsheet formula execution without changing numeric cells."""

    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def _sample_csv_rows(report: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    tolerance_fields = []
    for tolerance in report["metric_config"]["tolerances_px"]:
        label = _tolerance_label(tolerance)
        tolerance_fields.extend((f"precision_t{label}", f"recall_t{label}", f"f1_t{label}"))
    fields = [
        "dataset_id", "dataset_version", "method_id", "sample_id", "status",
        "actual_backend", "fallback_reason", "error", "width", "height",
        *REQUIRED_STRATA, "cldice", "symmetric_mean_px", "symmetric_p95_px",
        "prediction_components", "reference_components", "breaks", "fragment_excess",
        "missed_paths", "unmatched_prediction_branch_zones", "coverage_ratio",
        "longest_fragment_ratio", "wall_ns_median", "prompt_wall_ns_median",
        "prompt_wall_ns_p95", "image_load_wall_ns",
        "estimated_image_first_prompt_wall_ns",
        "estimated_cold_worker_first_prompt_wall_ns", "cpu_ns_median",
        "peak_rss_bytes", *tolerance_fields,
    ]
    rows = []
    for method_result in report["methods"]:
        method_id = method_result["method"]["id"]
        for sample in method_result["samples"]:
            execution = sample["execution"]
            timing = execution["timing"]
            row = {
                "dataset_id": report["dataset"]["id"],
                "dataset_version": report["dataset"]["version"],
                "method_id": method_id,
                "sample_id": sample["sample_id"],
                "status": execution["status"],
                "actual_backend": execution["actual_backend"],
                "fallback_reason": execution["fallback_reason"],
                "error": execution["error"],
                "width": sample["canvas"]["width"],
                "height": sample["canvas"]["height"],
                "wall_ns_median": timing["wall_ns_median"],
                "prompt_wall_ns_median": timing["prompt_wall_ns_median"],
                "prompt_wall_ns_p95": timing["prompt_wall_ns_p95"],
                "image_load_wall_ns": timing["image_load_wall_ns"],
                "estimated_image_first_prompt_wall_ns": timing[
                    "estimated_image_first_prompt_wall_ns"
                ],
                "estimated_cold_worker_first_prompt_wall_ns": timing[
                    "estimated_cold_worker_first_prompt_wall_ns"
                ],
                "cpu_ns_median": timing["cpu_ns_median"],
                "peak_rss_bytes": timing["peak_rss_bytes"],
                **sample["strata"],
            }
            metrics = sample["metrics"]
            if metrics is not None:
                topology = metrics["topology"]
                connectivity = metrics["connectivity"]["summary"]
                row.update(
                    {
                        "cldice": metrics["cldice"],
                        "symmetric_mean_px": metrics["distance"]["symmetric_mean"],
                        "symmetric_p95_px": metrics["distance"]["symmetric_p95"],
                        "prediction_components": topology["prediction"]["components"],
                        "reference_components": topology["reference"]["components"],
                        "breaks": connectivity["breaks"],
                        "fragment_excess": connectivity["fragment_excess"],
                        "missed_paths": connectivity["missed_paths"],
                        "unmatched_prediction_branch_zones": topology["unmatched_prediction_branch_zones"],
                        "coverage_ratio": connectivity["coverage_ratio"],
                        "longest_fragment_ratio": connectivity["longest_fragment_ratio"],
                    }
                )
                for tolerance in metrics["tolerance_metrics"]:
                    label = _tolerance_label(tolerance["tolerance"])
                    row[f"precision_t{label}"] = tolerance["precision"]
                    row[f"recall_t{label}"] = tolerance["recall"]
                    row[f"f1_t{label}"] = tolerance["f1"]
            rows.append(row)
    return fields, rows


def _summary_csv_rows(report: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "dataset_id", "dataset_version", "method_id", "label", "eligible",
        "publication_ranking_eligible",
        "sample_count", "completed_count", "failed_count", "fallback_count",
        "nondeterministic_count", "completion_rate", "primary_tolerance_px", "completed_macro_f1",
        "failure_adjusted_macro_f1", "completed_micro_precision",
        "completed_micro_recall", "completed_micro_f1", "completed_macro_cldice",
        "macro_symmetric_mean_px", "macro_symmetric_p95_px", "breaks",
        "fragment_excess", "missed_paths", "unmatched_prediction_branch_zones",
        "macro_coverage_ratio", "macro_longest_fragment_ratio",
        "case_wall_ns_median", "case_wall_ns_p95", "estimated_dataset_pass_ns",
        "case_prompt_wall_ns_median", "case_prompt_wall_ns_p95",
        "estimated_warm_prompt_pass_ns", "estimated_image_load_pass_ns",
        "estimated_image_first_prompt_pass_ns",
        "estimated_cold_worker_first_prompt_pass_ns",
        "worker_peak_rss_bytes_max",
    ]
    rows = []
    for method_result in report["methods"]:
        method = method_result["method"]
        summary = method_result["summary"]
        primary = summary["primary"]
        distance = summary["distance"]
        topology = summary["topology"]
        timing = summary["timing"]
        rows.append(
            {
                "dataset_id": report["dataset"]["id"],
                "dataset_version": report["dataset"]["version"],
                "method_id": method["id"],
                "label": method["label"],
                "eligible": summary["eligible"],
                "publication_ranking_eligible": summary["publication_ranking_eligible"],
                "sample_count": summary["sample_count"],
                "completed_count": summary["completed_count"],
                "failed_count": summary["failed_count"],
                "fallback_count": summary["fallback_count"],
                "nondeterministic_count": summary["nondeterministic_count"],
                "completion_rate": summary["completion_rate"],
                "primary_tolerance_px": summary["primary_tolerance_px"],
                "completed_macro_f1": primary["completed_macro_f1"],
                "failure_adjusted_macro_f1": primary["failure_adjusted_macro_f1"],
                "completed_micro_precision": primary["completed_micro_precision"],
                "completed_micro_recall": primary["completed_micro_recall"],
                "completed_micro_f1": primary["completed_micro_f1"],
                "completed_macro_cldice": summary["completed_macro_cldice"],
                "macro_symmetric_mean_px": distance["macro_symmetric_mean_px"],
                "macro_symmetric_p95_px": distance["macro_symmetric_p95_px"],
                **topology,
                "case_wall_ns_median": timing["case_wall_ns_median"],
                "case_wall_ns_p95": timing["case_wall_ns_p95"],
                "estimated_dataset_pass_ns": timing["estimated_dataset_pass_ns"],
                "case_prompt_wall_ns_median": timing[
                    "case_prompt_wall_ns_median"
                ],
                "case_prompt_wall_ns_p95": timing["case_prompt_wall_ns_p95"],
                "estimated_warm_prompt_pass_ns": timing[
                    "estimated_warm_prompt_pass_ns"
                ],
                "estimated_image_load_pass_ns": timing[
                    "estimated_image_load_pass_ns"
                ],
                "estimated_image_first_prompt_pass_ns": timing[
                    "estimated_image_first_prompt_pass_ns"
                ],
                "estimated_cold_worker_first_prompt_pass_ns": timing[
                    "estimated_cold_worker_first_prompt_pass_ns"
                ],
                "worker_peak_rss_bytes_max": timing["worker_peak_rss_bytes_max"],
            }
        )
    return fields, rows


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _prepare_runs_directory(directory: Path) -> Path:
    runs_directory = directory / "runs"
    if runs_directory.is_symlink():
        raise BenchmarkError("Report runs directory must not be a symbolic link.")
    if runs_directory.exists() and not runs_directory.is_dir():
        raise BenchmarkError("Report runs path must be a directory.")
    runs_directory.mkdir(exist_ok=True)
    if runs_directory.is_symlink() or runs_directory.resolve().parent != directory:
        raise BenchmarkError("Report runs directory escapes the output directory.")
    return runs_directory


def write_reports(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Atomically activate one immutable JSON/CSV report generation."""

    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    runs_directory = _prepare_runs_directory(directory)
    run_id = f"run-{uuid.uuid4().hex}"
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=runs_directory))
    if staging.resolve().parent != runs_directory:
        shutil.rmtree(staging)
        raise BenchmarkError("Report staging directory escapes the runs directory.")
    final_directory = runs_directory / run_id
    sample_fields, sample_rows = _sample_csv_rows(report)
    summary_fields, summary_rows = _summary_csv_rows(report)
    activated = False
    try:
        staged_samples = staging / SAMPLE_CSV_NAME
        staged_summary = staging / SUMMARY_CSV_NAME
        staged_json = staging / REPORT_JSON_NAME
        staged_commit = staging / COMMIT_JSON_NAME

        _atomic_csv(staged_samples, sample_fields, sample_rows)
        _atomic_csv(staged_summary, summary_fields, summary_rows)
        samples_sha256 = hashlib.sha256(staged_samples.read_bytes()).hexdigest()
        summary_sha256 = hashlib.sha256(staged_summary.read_bytes()).hexdigest()

        report_payload = dict(report)
        report_payload["report_files"] = {
            "samples_csv": {"name": SAMPLE_CSV_NAME, "sha256": samples_sha256},
            "summary_csv": {"name": SUMMARY_CSV_NAME, "sha256": summary_sha256},
        }
        _atomic_json(staged_json, report_payload)
        report_sha256 = hashlib.sha256(staged_json.read_bytes()).hexdigest()
        _atomic_json(
            staged_commit,
            {
                "schema_version": REPORT_SET_SCHEMA_VERSION,
                "run_id": run_id,
                "files": {
                    REPORT_JSON_NAME: report_sha256,
                    SAMPLE_CSV_NAME: samples_sha256,
                    SUMMARY_CSV_NAME: summary_sha256,
                },
            },
        )
        _fsync_directory(staging)
        os.rename(staging, final_directory)
        _fsync_directory(runs_directory)

        commit_sha256 = hashlib.sha256(
            (final_directory / COMMIT_JSON_NAME).read_bytes()
        ).hexdigest()
        latest_path = directory / LATEST_JSON_NAME
        _atomic_json(
            latest_path,
            {
                "schema_version": REPORT_POINTER_SCHEMA_VERSION,
                "active_run": run_id,
                "run_directory": f"runs/{run_id}",
                "commit": {
                    "name": COMMIT_JSON_NAME,
                    "sha256": commit_sha256,
                },
            },
        )
        activated = True
        _fsync_directory(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if final_directory.exists() and not activated:
            shutil.rmtree(final_directory)

    return {
        "json": final_directory / REPORT_JSON_NAME,
        "samples_csv": final_directory / SAMPLE_CSV_NAME,
        "summary_csv": final_directory / SUMMARY_CSV_NAME,
        "commit": final_directory / COMMIT_JSON_NAME,
        "latest": directory / LATEST_JSON_NAME,
        "run_dir": final_directory,
    }


__all__ = [
    "BenchmarkError",
    "evaluate_benchmark",
    "validate_benchmark",
    "write_reports",
]
