"""Command-line interface for the ArchaeoTrace contour benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .generate import GenerationError, generate_benchmark_dataset
from .geometry import CenterlineFormatError
from .manifest import ManifestError, load_manifest
from .runner import BenchmarkError, evaluate_benchmark, validate_benchmark, write_reports


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Evaluate one method id; repeat to select multiple methods.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m benchmarks",
        description="Validate and evaluate checksummed ordered contour traces.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="Validate manifest, checksums, dimensions, and centerline artifacts.",
    )
    validate.add_argument("manifest", type=Path)
    _add_selection(validate)

    generate = subparsers.add_parser(
        "generate",
        help="Run isolated CPU workers and atomically generate a measured dataset.",
    )
    generate.add_argument("template", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument(
        "--python-executable",
        help="Python interpreter containing NumPy, OpenCV, and scikit-image.",
    )
    generate.add_argument("--warmup-runs", type=int, default=1)
    generate.add_argument("--measurement-runs", type=int, default=3)
    generate.add_argument("--timeout-seconds", type=float, default=600.0)
    generate.add_argument(
        "--model-cache",
        type=Path,
        help="Verified local model cache (required by model-backed methods).",
    )

    model = subparsers.add_parser(
        "model",
        help="Inspect, fetch, or verify the pinned EfficientSAM-Ti split model.",
    )
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    for command, help_text in (
        ("status", "Inspect the local cache without network access."),
        ("fetch", "Explicitly fetch and verify missing pinned artifacts."),
        ("verify", "Re-hash both cached artifacts without network access."),
    ):
        command_parser = model_subparsers.add_parser(command, help=help_text)
        command_parser.add_argument("--model-cache", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate precomputed centerlines and write JSON/CSV reports.",
    )
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    _add_selection(evaluate)
    evaluate.add_argument(
        "--require-eligible",
        action="store_true",
        help=(
            "Return exit code 3 if a selected method failed, used a fallback, "
            "or produced non-deterministic outputs."
        ),
    )
    return parser


def _format_number(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _format_tolerance(value) -> str:
    return "n/a" if value is None else f"{float(value):g}"


def _run_validate(args) -> int:
    manifest = load_manifest(args.manifest)
    validate_benchmark(manifest, args.methods)
    selected = args.methods or [method.identifier for method in manifest.methods]
    print(
        f"VALID dataset={manifest.dataset.identifier}@{manifest.dataset.version} "
        f"samples={len(manifest.samples)} methods={','.join(selected)}"
    )
    return 0


def _run_evaluate(args) -> int:
    report = evaluate_benchmark(args.manifest, args.methods)
    paths = write_reports(report, args.output)
    for method_result in report["methods"]:
        method = method_result["method"]
        summary = method_result["summary"]
        primary = summary["primary"]
        print(
            f"{method['id']}: local_eligible={str(summary['eligible']).lower()} "
            "publication_eligible="
            f"{str(summary['publication_ranking_eligible']).lower()} "
            f"completion={summary['completion_rate']:.3f} "
            f"F1@{_format_tolerance(summary['primary_tolerance_px'])}px="
            f"{_format_number(primary['failure_adjusted_macro_f1'])}"
        )
    print(f"JSON {paths['json']}")
    print(f"SAMPLES_CSV {paths['samples_csv']}")
    print(f"SUMMARY_CSV {paths['summary_csv']}")
    print(f"COMMIT {paths['commit']}")
    print(f"LATEST {paths['latest']}")
    if args.require_eligible and any(
        not method_result["summary"]["eligible"]
        for method_result in report["methods"]
    ):
        return 3
    return 0


def _run_generate(args) -> int:
    options = {
        "python_executable": args.python_executable,
        "warmup_runs": args.warmup_runs,
        "measurement_runs": args.measurement_runs,
        "threads": 1,
        "timeout_seconds": args.timeout_seconds,
    }
    if args.model_cache is not None:
        options["model_cache"] = args.model_cache
    manifest = generate_benchmark_dataset(args.template, args.output, **options)
    print(
        f"GENERATED dataset={manifest.dataset.identifier}@{manifest.dataset.version} "
        f"samples={len(manifest.samples)} methods="
        f"{','.join(method.identifier for method in manifest.methods)}"
    )
    print(f"MANIFEST {manifest.path}")
    print(f"SHA256 {manifest.sha256}")
    return 0


def _run_model(args) -> int:
    try:
        from ai_vectorizer.core.efficientsam_spec import EFFICIENTSAM_TI_SPLIT
        from ai_vectorizer.core.model_store import (
            bundle_fingerprint,
            fetch_bundle,
            inspect_bundle,
            resolve_bundle,
        )

        cache = Path(os.path.abspath(os.fspath(args.model_cache)))
        fingerprint = bundle_fingerprint(EFFICIENTSAM_TI_SPLIT)
        if args.model_command == "fetch":
            bundle = fetch_bundle(cache, EFFICIENTSAM_TI_SPLIT)
            action = "FETCHED"
        elif args.model_command == "verify":
            bundle = resolve_bundle(cache, EFFICIENTSAM_TI_SPLIT)
            action = "VERIFIED"
        else:
            status = inspect_bundle(cache, EFFICIENTSAM_TI_SPLIT)
            if not status.ready:
                print(
                    f"MISSING model={EFFICIENTSAM_TI_SPLIT.id} "
                    f"cache={cache} fingerprint={fingerprint}"
                )
                return 3
            bundle = resolve_bundle(cache, EFFICIENTSAM_TI_SPLIT)
            action = "READY"
        artifact_text = ",".join(
            f"{artifact.id}:{artifact.sha256}"
            for artifact in EFFICIENTSAM_TI_SPLIT.artifacts
        )
        print(
            f"{action} model={EFFICIENTSAM_TI_SPLIT.id} "
            f"fingerprint={fingerprint} artifacts={artifact_text} cache={cache}"
        )
        # Keep the verified bundle alive until after all status evidence is
        # emitted; workers independently re-open and verify the same objects.
        del bundle
        return 0
    except Exception as exc:
        raise GenerationError(f"Model command failed: {exc}") from exc


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "generate":
            return _run_generate(args)
        if args.command == "model":
            return _run_model(args)
        return _run_evaluate(args)
    except (
        BenchmarkError,
        CenterlineFormatError,
        GenerationError,
        ManifestError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
