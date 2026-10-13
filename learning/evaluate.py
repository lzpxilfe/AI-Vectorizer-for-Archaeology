"""Frozen calibration and isolated CPU evaluation of the saved ordered route.

This is a separate, USGS-only pilot protocol. It does not relabel or replace
the existing mixed-feature benchmark and cannot certify Korean generalization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

from .dataset import DatasetError, canonical_bytes, checked_file, load_dataset_manifest, read_json, sha256_file
from .annotations import evaluation_reference, load_annotation, rasterize_annotation

SCHEMA = "archaeotrace-learned-evaluation/1"
METHODS = ("ink", "ink-recovery", "learned")
GRID = tuple((support, avoidance) for support in (0.4, 0.8) for avoidance in (2, 4, 6))
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def _write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evaluation-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic publication, never replace a run.
    finally:
        temporary.unlink(missing_ok=True)
    return path


def percentile95(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise DatasetError("timing samples must be nonnegative and finite")
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _sources_identity():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "learning/dataset.py", root / "learning/annotations.py",
             root / "learning/export.py", root / "learning/train.py",
             root / "benchmarks/worker.py", root / "benchmarks/metrics.py", root / "benchmarks/geometry.py"]
    paths += sorted((root / "ai_vectorizer/core").glob("*.py"))
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}


def _sample_annotation(dataset, sample):
    record = sample.get("annotation", {})
    annotation = load_annotation(checked_file(dataset.root, record.get("path"), record.get("sha256")), sample=sample, require_approved=True)
    evaluation_reference(annotation)  # Explicit, reviewed, visible target only.
    return annotation


def _bundle(dataset, bundle_path, *, synthetic_diagnostic=False):
    from ai_vectorizer.core.learned_model import inspect_bundle
    inspected = inspect_bundle(bundle_path)
    if not inspected.ready:
        raise DatasetError("model bundle is not ready: " + inspected.detail)
    document = inspected.manifest
    _guard_model_protocol(dataset, document, synthetic_diagnostic)
    return inspected, document


def _guard_model_protocol(dataset, document, synthetic_diagnostic):
    if (document["provenance"].get("allow_weak_labels") is True
            or document["provenance"].get("human_review_required") is True
            or document["provenance"].get("review_level") == "automated_weak_labels"):
        raise DatasetError("weak-label experiment cannot enter human-reviewed pilot evaluation or sealing")
    if document["provenance"]["dataset_manifest_sha256"] != dataset.manifest_sha256:
        raise DatasetError("model was exported against a different immutable dataset")
    diagnostic = any(document["provenance"].get(key) for key in ("synthetic_diagnostic", "diagnostic_only", "diagnostic"))
    if synthetic_diagnostic:
        if not diagnostic or not dataset.payload["dataset_id"].startswith("synthetic-"):
            raise DatasetError("diagnostic worker requires both a synthetic dataset and diagnostic model")
    elif diagnostic:
        raise DatasetError("synthetic diagnostic model cannot enter pilot calibration/holdout")
    else:
        _verify_validation_selection(document["provenance"])


def _verify_validation_selection(provenance):
    """Bind evaluation to the exporter's frozen validation-only seed winner.

    Checkpoint verification belongs to the explicit exporter; evaluation never
    opens pickle checkpoints or any training/validation raster or label bytes.
    """
    from .export import validate_validation_selection
    from .train import canonical_sha256
    selection = provenance.get("validation_selection")
    if not isinstance(selection, dict):
        raise DatasetError("real calibration requires frozen three-seed validation selection")
    content_hash = provenance.get("validation_selection_sha256")
    file_hash = provenance.get("validation_selection_file_sha256")
    if (not isinstance(file_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", file_hash)
            or canonical_sha256(selection) != content_hash):
        raise DatasetError("validation selection content/hash commitment changed")
    loss = provenance.get("best_validation_loss")
    if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
        raise DatasetError("exported validation loss must be finite and nonnegative")
    try:
        validate_validation_selection(selection, provenance)
    except (ValueError, KeyError, TypeError) as exc:
        raise DatasetError("invalid frozen validation selection: " + str(exc)) from exc


def _context_reader(dataset, sample):
    """Read native source context, never treat a crop edge as a map boundary."""
    import numpy as np
    source = dataset.sources[sample["source_id"]]
    path = checked_file(dataset.root, source.get("source_file"), source["source_sha256"])
    import rasterio
    from rasterio.windows import Window
    from types import SimpleNamespace
    raster = rasterio.open(path)
    try:
        if list((raster.width, raster.height)) != source["source_size"] or raster.dtypes != tuple("uint8" for _ in raster.dtypes) or raster.count not in (1, 3, 4):
            raise DatasetError("evaluation needs native Byte RGB/palette source context")
        palette = raster.colormap(1) if raster.count == 1 else None
        lookup = np.asarray([palette.get(i, (255, 255, 255, 255))[:3] for i in range(256)], dtype=np.uint8) if palette is not None else None
        def read(bounds):
            x0, y0, x1, y1 = bounds
            if not 0 <= x0 < x1 <= raster.width or not 0 <= y0 < y1 <= raster.height:
                raise DatasetError("source read leaves the true raster bounds")
            native = raster.read(window=Window(x0, y0, x1-x0, y1-y0))
            array = lookup[native[0]] if lookup is not None else np.moveaxis(native[:3], 0, -1)
            if array.dtype != np.uint8 or array.shape != (y1-y0, x1-x0, 3):
                raise DatasetError("source context is not native Byte RGB")
            return np.ascontiguousarray(array)
        return SimpleNamespace(size=(raster.width, raster.height), close=raster.close), read
    except Exception:
        raster.close()
        raise


def _native_ink_evidence(image, reader, sample, raster_size):
    """Use complete source-anchored Ink cores plus their true 31px halo."""
    import numpy as np
    from ai_vectorizer.core.edge_detector import EdgeDetector
    from ai_vectorizer.core.line_evidence import crop_line_evidence
    x, y, width, height = sample["source_crop_xywh"]
    if not np.array_equal(reader((x, y, x+width, y+height)), image["rgb"]):
        raise DatasetError("crop pixels differ from declared native source context")
    source_width, source_height = raster_size
    tile, halo = 128, 31
    x0, y0 = max(0, x//tile*tile-halo), max(0, y//tile*tile-halo)
    x1 = min(source_width, math.ceil((x+width)/tile)*tile+halo)
    y1 = min(source_height, math.ceil((y+height)/tile)*tile+halo)
    evidence = EdgeDetector.detect_ink_evidence(reader((x0, y0, x1, y1)), tile_origin=(x0, y0))
    evidence = crop_line_evidence(evidence, (x-x0, y-y0, x-x0+width, y-y0+height))
    image["evidence"] = evidence
    image["edges"] = evidence.centerline.astype(np.uint8) * 255
    return (x0, y0, x1, y1)


def _configuration():
    from ai_vectorizer.core.smart_recovery import DEFAULT_RECOVERY_CONFIG
    return {"edge_weight": 1.0, "livewire_window_px": 320, "target_snap_radius_px": 6,
            "smoothing_window_px": 5, "smoothing_profile": "smart-trace-livewire-v1",
            "recovery_policy_id": DEFAULT_RECOVERY_CONFIG.policy_id, "recovery_provisional": True,
            "recovery_thresholds": asdict(DEFAULT_RECOVERY_CONFIG),
            "recovery_configuration_sha256": DEFAULT_RECOVERY_CONFIG.sha256}


def _configuration_pair(support, avoidance):
    if type(support) not in (int, float) or type(avoidance) not in (int, float) or (support, avoidance) not in GRID:
        raise DatasetError("configuration must be one of the six frozen candidates")
    return float(support), int(avoidance)


def _worker(request):
    """One fresh CPU process owns one sample/method and its peak RSS."""
    import numpy as np
    from benchmarks import worker
    from benchmarks.geometry import CenterlineArtifact, CenterlinePath, rasterize_centerlines
    from benchmarks.metrics import compute_metrics
    split = request.get("split", "calibration")
    if split not in ("calibration", "holdout"):
        raise DatasetError("worker split is outside evaluation protocol")
    dataset = load_dataset_manifest(request["dataset"], verify_files=True, verify_splits={split})
    if request.get("dataset_sha256", dataset.manifest_sha256) != dataset.manifest_sha256:
        raise DatasetError("worker dataset changed from its request")
    diagnostic = request.get("synthetic_diagnostic", False)
    if type(diagnostic) is not bool:
        raise DatasetError("synthetic_diagnostic must be an explicit boolean")
    if request["method"] not in METHODS:
        raise DatasetError("unknown evaluation method")
    _configuration_pair(request["support"], request["avoidance"])
    if type(request["repetitions"]) is not int or not 3 <= request["repetitions"] <= 20:
        raise DatasetError("use 3–20 timing repetitions")
    if request["method"] == "learned":
        _bundle(dataset, request["bundle"], synthetic_diagnostic=diagnostic)
    else:
        # A control process must not load/parse an unused learned graph into
        # its peak RSS. The coordinator verifies that bundle before/after the
        # run; these controls need only its dataset/diagnostic protocol tag.
        _guard_model_protocol(dataset, read_json(Path(request["bundle"]) / "manifest.json"), diagnostic)
    sample = next(s for s in dataset.payload["samples"] if s["id"] == request["sample_id"])
    if sample["split"] != split:
        raise DatasetError("worker sample is outside its requested split")
    annotation = _sample_annotation(dataset, sample)
    reference = evaluation_reference(annotation)
    prompt_data = annotation["evaluation"]["prompt"]
    prompt = worker.TracePrompt(tuple(prompt_data["start_xy"]), tuple(prompt_data["end_xy"]),
                                previous_xy=tuple(prompt_data["previous_xy"]) if "previous_xy" in prompt_data else None)
    method = request["method"]
    backend = worker.INK_V2_EFFICIENTSAM_RECOVERY_BACKEND if method == "ink-recovery" else worker.INK_LIVEWIRE_V2_BACKEND
    pipeline = worker._load_pipeline(backend, 1, model_cache=Path(request["recovery_cache"]) if request.get("recovery_cache") else None)
    image = pipeline.load_image(checked_file(dataset.root, sample["image"], sample["image_sha256"]), 512, 512)
    image["source_tile_origin_xy"] = tuple(sample["source_tile_origin_xy"])
    config = _configuration()
    tile_ms, engine, prior, source = [], None, None, None
    source, reader = _context_reader(dataset, sample)
    ink_context_bounds = _native_ink_evidence(image, reader, sample, source.size)
    rss_before_model = worker._peak_rss_bytes()
    if method == "learned":
        from ai_vectorizer.core.learned_model import LearnedContourEngine, plan_source_tiles
        from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree
        from ai_vectorizer.core.trace_kernel import smooth_pixel_path
        engine = LearnedContourEngine.from_bundle(request["bundle"])
        x, y, w, h = sample["source_crop_xywh"]
        bounds, size = (x, y, x+w, y+h), source.size
        tiles = list(plan_source_tiles(bounds, size))
        snapshots = {}
        for tile in tiles:
            raw = reader(tile.read_bounds)
            snapshots[tuple(tile.read_bounds)] = raw
            padded = np.pad(raw, ((tile.padding[0], tile.padding[1]), (tile.padding[2], tile.padding[3]), (0,0)), mode="edge")
            engine.predict_tile(padded)  # Warm-up is excluded from prepared latency.
            for _ in range(request["repetitions"]):
                started = time.perf_counter()
                engine.predict_tile(padded)
                tile_ms.append((time.perf_counter()-started)*1000)
        prior = engine.predict_crop(lambda b: snapshots[tuple(b)], bounds, size)
        def predict():
            root = tuple(int(v) for v in prompt.start_xy)
            incoming = tuple(root[i]-int(prompt.previous_xy[i]) for i in (0,1)) if prompt.previous_xy else None
            tree = build_livewire_tree(image["rgb"], image["edges"], root, strength=1.0,
                    incoming_direction=incoming, evidence=image["evidence"], learned_prior=prior,
                    config=LiveWireConfig(max_window_size=320, target_snap_radius=6,
                        learned_support_cap=request["support"], learned_avoidance_cost_weight=request["avoidance"]))
            path = list(tree.trace(prompt.end_xy))
            if len(path) > 5:
                original = path
                path = list(smooth_pixel_path(path, window_size=5))
                path[0], path[-1] = original[0], original[-1]
            if len(path) < 2:
                raise DatasetError("learned Live-Wire did not return an ordered segment")
            path[0] = prompt.start_xy
            return path
    else:
        def predict():
            return list(pipeline.predict(image, prompt, config))
    try:
        predict()  # Same prepared-path measurement policy for all methods.
        durations, hashes, path = [], [], None
        for _ in range(request["repetitions"]):
            started = time.perf_counter()
            path = predict()
            durations.append((time.perf_counter()-started)*1000)
            hashes.append(hashlib.sha256(canonical_bytes(path)).hexdigest())
        if len(set(hashes)) != 1:
            raise DatasetError("repeated predictions changed the saved route")
        predicted_artifact = CenterlineArtifact(512,512,(CenterlinePath("prediction",tuple(tuple(p) for p in path)),),{},Path("prediction"),hashes[0])
        reference_artifact = CenterlineArtifact(512,512,tuple(CenterlinePath(p["id"],tuple(tuple(v) for v in p["points"])) for p in reference["paths"]),{},Path("reference"),annotation["revision"])
        predicted = rasterize_centerlines(predicted_artifact)
        truth = rasterize_centerlines(reference_artifact)
        valid = rasterize_annotation(annotation)["valid_mask"]
        unknown = int(np.count_nonzero(np.asarray(predicted.mask) & ~valid))
        measured = compute_metrics(predicted.mask, truth.mask, [(p.pixels, p.closed) for p in truth.paths], primary_tolerance=2)
        # Pixel heads are supplementary. Final-line metrics above decide gates.
        pixel_scores = None
        if prior is not None:
            targets = rasterize_annotation(annotation)
            pixel_scores = {}
            for name, probability, target in (("contour_support",prior.contour_score,targets["centerline"]), ("other_ink_risk",prior.avoidance_score,targets["other_ink"])):
                a, b = (probability >= .5) & valid, (target >= .5) & valid
                denominator = int(a.sum()+b.sum())
                pixel_scores[name] = {"dice_at_0_5": 2*int((a&b).sum())/denominator if denominator else 1.0, "reviewed_pixels": int(valid.sum())}
        recovery_evidence = pipeline.prediction_evidence() if method == "ink-recovery" else None
        status = "fallback" if recovery_evidence and recovery_evidence.get("fallback_reason") else "ok"
        return {"sample_id":sample["id"],"method":method,"status":status, "annotation_revision":annotation["revision"],
                "hard_case":sample.get("difficulty") == "hard", "path": [list(p) for p in path],"path_sha256":hashes[0],
                "metrics":measured,"pixel_scores":pixel_scores,"unknown_prediction_pixels":unknown,
                "prepared_path_ms":durations,"prepared_tile_ms":tile_ms,"peak_rss_bytes":worker._peak_rss_bytes(),
                "pre_model_peak_rss_bytes":rss_before_model,"provider":engine.metadata if engine else asdict(pipeline.info),
                "recovery_evidence":recovery_evidence,
                "source_context":"original_native_grid", "source_tile_origin_xy":sample["source_tile_origin_xy"],
                "ink_context_bounds":list(ink_context_bounds), "synthetic_diagnostic":diagnostic,
                "request_sha256":hashlib.sha256(canonical_bytes(request)).hexdigest()}
    finally:
        if source is not None:
            source.close()


def run_split(dataset_path, bundle_path, split, output_dir, *, support=.8, avoidance=6, recovery_cache=None, repetitions=3, methods=METHODS, write_report=True):
    if split not in ("calibration", "holdout"):
        raise DatasetError("evaluation split/configuration is outside the frozen protocol")
    support, avoidance = _configuration_pair(support, avoidance)
    if type(repetitions) is not int or not 3 <= repetitions <= 20:
        raise DatasetError("use 3–20 prepared timing repetitions")
    dataset = load_dataset_manifest(dataset_path, verify_splits={split})
    if dataset.payload["dataset_id"].startswith("synthetic-"):
        raise DatasetError("synthetic diagnostics cannot enter pilot split evaluation")
    inspected, _ = _bundle(dataset, bundle_path)
    selected = dataset.samples_for(split)
    if not selected:
        raise DatasetError("evaluation split contains no reviewed samples")
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise DatasetError("evaluation methods must be unique known methods")
    for sample in selected:
        _sample_annotation(dataset, sample)  # Fail before expensive partial runs.
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DatasetError("use a new empty directory for an immutable evaluation run")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    environment = dict(os.environ, **{key:"1" for key in THREAD_ENV})
    source_identity = _sources_identity()
    for sample in selected:
        for method in methods:
            if method not in METHODS:
                raise DatasetError("unknown evaluation method")
            request = {"dataset":str(dataset.path),"bundle":str(Path(bundle_path).resolve()),"sample_id":sample["id"],"method":method,
                       "support":support,"avoidance":avoidance,"repetitions":repetitions,
                       "split":split,"dataset_sha256":dataset.manifest_sha256,
                       "recovery_cache":str(Path(recovery_cache).resolve()) if recovery_cache else None}
            job = output_dir / (sample["id"]+"-"+method+".request.json")
            result_path = job.with_name(job.name.replace(".request.json", ".result.json"))
            _write(job, request)
            try:
                completed = subprocess.run([sys.executable,"-m","learning.evaluate","worker",str(job),str(result_path)],
                            env=environment, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True, timeout=300)
                if completed.returncode or not result_path.exists():
                    reason = completed.stderr[-2000:] or "worker returned no result"
                    raise DatasetError(reason)
                result = read_json(result_path)
                if result.get("sample_id") != sample["id"] or result.get("method") != method or result.get("request_sha256") != hashlib.sha256(canonical_bytes(request)).hexdigest():
                    raise DatasetError("worker output does not match its hash-bound request")
            except (DatasetError, subprocess.TimeoutExpired) as exc:
                result = {"sample_id":sample["id"],"method":method,"status":"failed","reason":str(exc)}
                _write(result_path, result, replace=True)
            finally:
                job.unlink(missing_ok=True)
            results.append(result)
    final_dataset = load_dataset_manifest(dataset_path, verify_splits={split})
    final_bundle, _ = _bundle(final_dataset, bundle_path)
    if (final_dataset.manifest_sha256, final_bundle.bundle_sha256, _sources_identity()) != (dataset.manifest_sha256, inspected.bundle_sha256, source_identity):
        raise DatasetError("dataset, bundle or evaluator changed during evaluation")
    report = {"schema_version":SCHEMA,"split":split,"dataset_id":dataset.payload["dataset_id"],"dataset_sha256":dataset.manifest_sha256,"model_sha256":inspected.model_sha256,
              "bundle_sha256":inspected.bundle_sha256,"configuration":{"support":support,"avoidance":avoidance},
              "methods":list(methods),"sample_ids":[s["id"] for s in selected],"results":results,
              "code_files_sha256":source_identity,"environment":{"platform":platform.platform(),"machine":platform.machine(),
                  "python":platform.python_version(),"processor":platform.processor(),"thread_env":{k:environment[k] for k in THREAD_ENV}},
              "review_level":"single_reviewer_pilot","publication_ranking_eligible":False}
    if write_report:
        _write(output_dir / "report.json", report)
    return report


def _summary(report, method):
    rows = [r for r in report["results"] if r["method"] == method]
    if not rows or len(rows) != len(report["sample_ids"]) or {r["sample_id"] for r in rows} != set(report["sample_ids"]) or any(r["status"] != "ok" or r.get("unknown_prediction_pixels",0) for r in rows):
        return None
    return {"mean_f1_2px":statistics.mean(next(t["f1"] for t in r["metrics"]["tolerance_metrics"] if t["tolerance"]==2) for r in rows),
            "breaks":sum(r["metrics"]["connectivity"]["summary"]["breaks"] for r in rows),
            "over_branches":sum(r["metrics"]["topology"]["unmatched_prediction_branch_zones"] for r in rows)}


def assess_holdout(report, model_size):
    """All incomplete/unknown/missing-hard-case gates fail explicitly."""
    baseline, learned = _summary(report,"ink"), _summary(report,"learned")
    recovered = _summary(report,"ink-recovery")
    gates = {"all_methods_complete":all(v is not None for v in (baseline,learned,recovered)),
             "f1_noninferior":False,"hard_p95_improvement_15pct":False,"topology_noninferior":False,
             "model_under_64mib":model_size<=64*1024**2,"extra_ram_under_512mib":False,"prepared_tile_p95_under_1s":False}
    gates["reviewed_real_holdout"] = bool(
        report.get("split") == "holdout"
        and report.get("dataset_id") == "usgs-contour-pilot-16x144-v1"
        and report.get("review_level") == "single_reviewer_pilot"
        and not report.get("synthetic_diagnostic", False)
        and not any(row.get("synthetic_diagnostic", False) for row in report["results"])
    )
    hard_ids = {r["sample_id"] for r in report["results"] if r.get("hard_case")}
    hard_medians = {}
    if baseline and learned:
        gates["f1_noninferior"] = learned["mean_f1_2px"] >= baseline["mean_f1_2px"]
        gates["topology_noninferior"] = learned["breaks"]<=baseline["breaks"] and learned["over_branches"]<=baseline["over_branches"]
        for method in ("ink","learned"):
            values=[r["metrics"]["distance"]["symmetric_p95"] for r in report["results"] if r["method"]==method and r["sample_id"] in hard_ids]
            if values and all(v is not None for v in values):
                hard_medians[method]=statistics.median(values)
        if len(hard_medians)==2 and hard_medians["ink"]>0:
            gates["hard_p95_improvement_15pct"] = hard_medians["learned"] <= .85*hard_medians["ink"]
        by_id={r["sample_id"]:r for r in report["results"] if r["method"]=="ink"}
        deltas=[max(0,r["peak_rss_bytes"]-by_id[r["sample_id"]]["peak_rss_bytes"]) for r in report["results"] if r["method"]=="learned"]
        tiles=[v for r in report["results"] if r["method"]=="learned" for v in r["prepared_tile_ms"]]
        gates["extra_ram_under_512mib"] = bool(deltas) and max(deltas)<=512*1024**2
        gates["prepared_tile_p95_under_1s"] = bool(tiles) and percentile95(tiles)<=1000
    return {"gates":gates,"passed":all(gates.values()),"summary":{"ink":baseline,"ink-recovery":recovered,"learned":learned},
            "hard_case_count":len(hard_ids),"hard_p95_medians":hard_medians,
            "limitations":["single reviewer pilot; no publication ranking", "CPU metrics apply only to the recorded host/runtime", "no Korean-map validation"]}


def calibrate(dataset, bundle, output_dir, *, recovery_cache=None, repetitions=3):
    output_dir=Path(output_dir)
    if (output_dir/"calibration.json").exists():
        raise DatasetError("calibration is already frozen; choose a new versioned experiment")
    candidates=[]
    for support, avoidance in GRID:
        report=run_split(dataset,bundle,"calibration",output_dir/("s%s-a%s"%(support,avoidance)),support=support,avoidance=avoidance,
                         repetitions=repetitions,methods=("ink","learned"))
        baseline, learned = _summary(report,"ink"), _summary(report,"learned")
        eligible = bool(baseline and learned and learned["mean_f1_2px"]>=baseline["mean_f1_2px"] and learned["breaks"]<=baseline["breaks"] and learned["over_branches"]<=baseline["over_branches"])
        distances=[r["metrics"]["distance"]["symmetric_p95"] for r in report["results"] if r["method"]=="learned" and r["status"]=="ok"]
        p95=statistics.median(distances) if distances and all(v is not None for v in distances) else None
        candidates.append({"configuration":report["configuration"],"eligible":eligible,"summary":learned,"median_p95":p95,
                           "report_path":"s%s-a%s/report.json"%(support,avoidance),
                           "report_sha256":sha256_file(output_dir/("s%s-a%s"%(support,avoidance))/"report.json")})
    eligible=[c for c in candidates if c["eligible"] and c["median_p95"] is not None]
    selected=min(eligible,key=lambda c:(c["median_p95"],-c["summary"]["mean_f1_2px"],c["configuration"]["support"],c["configuration"]["avoidance"])) if eligible else None
    freeze={"schema_version":"archaeotrace-calibration/1","dataset_sha256":report["dataset_sha256"],"model_sha256":report["model_sha256"],
            "bundle_sha256":report["bundle_sha256"],"split":"calibration","code_files_sha256":report["code_files_sha256"],
            "sample_ids":report["sample_ids"],"selected":selected,"candidates":candidates,"publication_ranking_eligible":False}
    _write(output_dir/"calibration.json",freeze)
    return freeze


def _verify_calibration(frozen, calibration_path, dataset, inspected):
    """Recompute six candidate eligibility/ranking from bound reports."""
    if frozen.get("schema_version") != "archaeotrace-calibration/1" or frozen.get("split") != "calibration" or not frozen.get("selected"):
        raise DatasetError("an eligible frozen calibration result is required")
    if (frozen["dataset_sha256"], frozen["model_sha256"], frozen["bundle_sha256"], frozen["code_files_sha256"]) != (dataset.manifest_sha256, inspected.model_sha256, inspected.bundle_sha256, _sources_identity()):
        raise DatasetError("dataset, model, bundle or evaluator changed after calibration")
    if frozen["sample_ids"] != list(dataset.sample_ids("calibration")) or len(frozen["candidates"]) != len(GRID):
        raise DatasetError("calibration candidate/sample set is incomplete")
    seen, eligible = set(), []
    for candidate in frozen["candidates"]:
        config = candidate["configuration"]
        pair = _configuration_pair(config["support"], config["avoidance"])
        if pair in seen:
            raise DatasetError("calibration configurations must be unique")
        seen.add(pair)
        path = checked_file(Path(calibration_path).parent, candidate["report_path"], candidate["report_sha256"])
        report = read_json(path)
        if report.get("split") != "calibration" or report.get("configuration") != config or report.get("methods") != ["ink", "learned"] or any(report.get(key) != frozen[key] for key in ("dataset_sha256", "model_sha256", "bundle_sha256", "code_files_sha256", "sample_ids")):
            raise DatasetError("calibration report identity does not match freeze")
        baseline, learned = _summary(report, "ink"), _summary(report, "learned")
        accepted = bool(baseline and learned and learned["mean_f1_2px"] >= baseline["mean_f1_2px"] and learned["breaks"] <= baseline["breaks"] and learned["over_branches"] <= baseline["over_branches"])
        distances = [row["metrics"]["distance"]["symmetric_p95"] for row in report["results"] if row["method"] == "learned" and row["status"] == "ok"]
        p95 = statistics.median(distances) if distances and all(value is not None for value in distances) else None
        if candidate["eligible"] is not accepted or candidate["summary"] != learned or candidate["median_p95"] != p95:
            raise DatasetError("calibration candidate summary or eligibility was altered")
        if accepted and p95 is not None:
            eligible.append(candidate)
    expected = min(eligible, key=lambda c: (c["median_p95"], -c["summary"]["mean_f1_2px"], c["configuration"]["support"], c["configuration"]["avoidance"])) if eligible else None
    if frozen["selected"] != expected:
        raise DatasetError("selected calibration configuration is not the eligible ranked candidate")
    return expected["configuration"]


def seal_bundle(bundle_path, report_path, output_dir, *, calibration_path, dataset_path=None):
    """Publish a new lineage-bound bundle only after real holdout gates pass."""
    from ai_vectorizer.core.learned_model import build_bundle_manifest, inspect_bundle
    original = inspect_bundle(bundle_path)
    if not original.ready:
        raise DatasetError("source bundle is not ready")
    report = read_json(report_path)
    assessment = assess_holdout(report, original.manifest["model"]["size_bytes"])
    if not assessment["passed"] or report.get("assessment") != assessment:
        raise DatasetError("only an unmodified passing real holdout report can seal a bundle")
    if report["model_sha256"] != original.model_sha256 or report["bundle_sha256"] != original.bundle_sha256 or report["dataset_sha256"] != original.manifest["provenance"]["dataset_manifest_sha256"]:
        raise DatasetError("holdout report does not belong to the source bundle")
    calibration = read_json(calibration_path)
    if sha256_file(calibration_path) != report.get("calibration_sha256") or calibration.get("selected", {}).get("configuration") != report["configuration"]:
        raise DatasetError("holdout configuration is not bound to its frozen calibration")
    if dataset_path is None:
        raise DatasetError("the immutable dataset is required to verify a pilot seal")
    # A direct library call must verify the same six-candidate freeze as the
    # holdout CLI. Reading only manifest metadata cannot inspect holdout labels.
    dataset = load_dataset_manifest(dataset_path, verify_splits=set())
    _guard_model_protocol(dataset, original.manifest, False)
    if report.get("sample_ids") != list(dataset.sample_ids("holdout")) or report.get("code_files_sha256") != _sources_identity():
        raise DatasetError("holdout sample set or evaluator identity changed before sealing")
    if _verify_calibration(calibration, calibration_path, dataset, original) != report["configuration"]:
        raise DatasetError("holdout configuration differs from recomputed calibration")
    support, avoidance = _configuration_pair(report["configuration"]["support"], report["configuration"]["avoidance"])
    provenance = dict(original.manifest["provenance"])
    if any(provenance.get(key) for key in ("synthetic_diagnostic", "diagnostic_only", "diagnostic")):
        raise DatasetError("a diagnostic bundle cannot be sealed as a pilot candidate")
    provenance["selected_guidance"] = {
        "support":support, "avoidance":avoidance,
        "calibration_sha256":report["calibration_sha256"],
        "holdout_report_sha256":sha256_file(report_path), "pilot_gates_passed":True,
        "source_bundle_sha256":original.bundle_sha256,
    }
    model_bytes = (Path(bundle_path)/"model.onnx").read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != original.model_sha256:
        raise DatasetError("source model changed before sealing")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output/"model.onnx").write_bytes(model_bytes)
        _write(output/"manifest.json", build_bundle_manifest(model_bytes, provenance, model_id=original.manifest["model_id"]))
        if not inspect_bundle(output).ready:
            raise DatasetError("sealed bundle verification failed")
        return output
    except Exception:
        import shutil
        shutil.rmtree(output)
        raise


def holdout(dataset_path,bundle_path,calibration_path,output_dir,*,recovery_cache,repetitions=3,seal_output=None):
    frozen=read_json(calibration_path)
    dataset=load_dataset_manifest(dataset_path, verify_splits={"holdout"})
    inspected, manifest=_bundle(dataset,bundle_path)
    config = _verify_calibration(frozen, calibration_path, dataset, inspected)
    if (Path(output_dir)/"report.json").exists():
        raise DatasetError("holdout report exists; never overwrite a final evaluation")
    report=run_split(dataset_path,bundle_path,"holdout",output_dir,support=config["support"],avoidance=config["avoidance"],
                     recovery_cache=recovery_cache,repetitions=repetitions,write_report=False)
    report["calibration_sha256"]=sha256_file(calibration_path)
    report["assessment"]=assess_holdout(report,manifest["model"]["size_bytes"])
    _write(Path(output_dir)/"report.json",report)
    if seal_output is not None:
        seal_bundle(bundle_path, Path(output_dir)/"report.json", seal_output,
                    calibration_path=calibration_path, dataset_path=dataset_path)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("calibrate","holdout"):
        p=sub.add_parser(name)
        p.add_argument("dataset");p.add_argument("bundle");p.add_argument("output_dir")
        p.add_argument("--recovery-cache",required=name=="holdout")
        p.add_argument("--repetitions",type=int,default=3)
        if name=="holdout":
            p.add_argument("--calibration",required=True)
            p.add_argument("--seal-output")
    p=sub.add_parser("worker");p.add_argument("request");p.add_argument("result")
    args=parser.parse_args(argv)
    try:
        if args.command=="worker":
            _write(args.result,_worker(read_json(args.request)))
        elif args.command=="calibrate":
            result=calibrate(args.dataset,args.bundle,args.output_dir,repetitions=args.repetitions)
            print("calibration frozen; eligible="+str(result["selected"] is not None))
        else:
            result=holdout(args.dataset,args.bundle,args.calibration,args.output_dir,recovery_cache=args.recovery_cache,repetitions=args.repetitions,seal_output=args.seal_output)
            print("holdout pilot gates passed="+str(result["assessment"]["passed"]))
        return 0
    except (DatasetError,ValueError,OSError,RuntimeError) as error:
        print(str(error),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
