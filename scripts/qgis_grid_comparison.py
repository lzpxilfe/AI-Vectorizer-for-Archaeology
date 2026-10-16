#!/usr/bin/env python3
"""Compare saved QGIS traces on frozen, visually inspected contour cases.

References are agent visual inspections, not human-approved benchmark labels.
Raw controls are versioned before their comparisons; post-output negative
corrections record their review history. Source pixels stay outside git;
image hashes, points and review notes make each case reviewable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# QGIS executes --code in its console globals, where __file__ can refer to
# qgis.utils rather than this script. The outer launcher binds the exact path.
SCRIPT_ENV = "ARCHAEOTRACE_GRID_COMPARISON_SCRIPT"
_script_path = os.environ.get(SCRIPT_ENV) or globals().get("__file__")
sys.path.insert(0, str(Path(_script_path).resolve().parent))
_bootstrap = os.environ.pop("ARCHAEOTRACE_CLEAN_PROFILE_BOOTSTRAP", None)
import qgis_clean_profile_smoke as smoke
if _bootstrap is not None:
    os.environ["ARCHAEOTRACE_CLEAN_PROFILE_BOOTSTRAP"] = _bootstrap

SPEC_ENV = "ARCHAEOTRACE_GRID_COMPARISON_SPEC"
CASE_AUDIT_PATH = Path(_script_path).resolve().parents[1] / "learning/data/korea-contour-case-audit-v1.json"


def native_pixel_extent(viewport_size, image_size, *, screen_scale=2):
    """Map native cell centres to integer screen positions without rounding.

    A full-raster fit uses a fractional map-units-per-screen-pixel scale.
    Rounding its events can change the nearest skeleton anchor between QGIS
    layouts. An integer zoom keeps the frozen source pixel prompts exact while
    retaining ordinary mouse event handlers. Two screen pixels per native
    pixel also keep short diagnostic spans outside the spot-height gesture.
    """
    viewport_width, viewport_height = viewport_size
    image_width, image_height = image_size
    values = (*viewport_size, *image_size, screen_scale)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values):
        raise ValueError("Viewport and image sizes must be positive integers")
    if viewport_width < image_width * screen_scale or viewport_height < image_height * screen_scale:
        raise ValueError("Comparison canvas must show the complete native raster")
    left = 0.5 - ((viewport_width - image_width * screen_scale) // 2) / screen_scale
    top = image_height - 0.5 + ((viewport_height - image_height * screen_scale) // 2) / screen_scale
    return left, top - viewport_height / screen_scale, left + viewport_width / screen_scale, top


def prepare_native_input(canvas, image_size, prompts):
    """Return a view preparation hook; never inject a path or move a prompt."""
    def view_coordinates():
        # QgsMapCanvas includes its frame; the map transform uses outputSize.
        viewport = canvas.mapSettings().outputSize()
        extent = native_pixel_extent((viewport.width(), viewport.height()), image_size)
        current = canvas.extent()
        actual = (current.xMinimum(), current.yMinimum(), current.xMaximum(), current.yMaximum())
        return extent, actual

    def validate(tool):
        from qgis.core import QgsPointXY

        extent, actual = view_coordinates()
        if any(abs(first - second) > 1e-8 for first, second in zip(actual, extent)):
            raise smoke.SmokeFailure("Native pixel input view changed during a trace")
        for prompt in prompts:
            point = QgsPointXY(*prompt)
            screen = tool.toCanvasCoordinates(point)
            mapped = tool.toMapCoordinates(screen)
            if math.hypot(mapped.x() - point.x(), mapped.y() - point.y()) > 1e-7:
                raise smoke.SmokeFailure("Integer mouse event changed a frozen source-pixel prompt")

    def prepare(tool):
        from qgis.core import QgsRectangle

        extent, actual = view_coordinates()
        if any(abs(first - second) > 1e-8 for first, second in zip(actual, extent)):
            canvas.setExtent(QgsRectangle(*extent))
            canvas.refresh()
            smoke._wait_until(lambda: not canvas.isDrawing(), "native pixel input view")
        validate(tool)
    prepare.validate = validate
    return prepare


def merge_archive_comparisons(before, after):
    """Bind product-version comparisons to matching real mouse coordinates."""
    if before.get("fixtures_sha256") != after.get("fixtures_sha256"):
        raise ValueError("Product comparisons require identical frozen fixtures")
    if before.get("qgis_version") != after.get("qgis_version"):
        raise ValueError("Product comparisons require the same QGIS runtime")
    if before.get("models") != after.get("models"):
        raise ValueError("Product comparisons require identical learned models")
    old_cases = {case["id"]: case for case in before["cases"]}
    new_cases = {case["id"]: case for case in after["cases"]}
    if old_cases.keys() != new_cases.keys():
        raise ValueError("Product comparisons require identical case IDs")
    for case_id, row in new_cases.items():
        previous = old_cases[case_id]["traces"]
        merged_traces = {}
        for method in ("ink", "usgs", "korea"):
            old_key, new_key = method + "_before", method + "_after"
            old_trace, new_trace = previous[old_key], row["traces"][new_key]
            old_input, new_input = old_trace.get("input_map_xy"), new_trace.get("input_map_xy")
            if not isinstance(old_input, dict) or not isinstance(new_input, dict):
                raise ValueError("Product comparisons require recorded real mouse inputs")
            for gesture in ("start", "end"):
                first, second = old_input.get(gesture), new_input.get(gesture)
                if (not isinstance(first, (tuple, list)) or len(first) != 2
                        or not isinstance(second, (tuple, list)) or len(second) != 2
                        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                               for value in (*first, *second))
                        or math.hypot(first[0] - second[0], first[1] - second[1]) > 1e-7):
                    raise ValueError("Product comparisons changed a real mouse input: " + case_id)
            merged_traces[old_key] = old_trace
            merged_traces[new_key] = new_trace
        row["traces"] = merged_traces
    after["baseline_archive"] = before["archive"]
    after["baseline_archive_sha256"] = before["archive_sha256"]
    after["baseline_qgis_exit_code"] = before["qgis_exit_code"]
    after["baseline_isolated_profile_removed_after_run"] = before.get("isolated_profile_removed_after_run")
    after["before_config"] = before["after_config"]
    after["input_policy"] = "integer mouse events at exact native cell centres; matching inputs verified"
    return after


def load_cases(fixtures, dataset, *, allow_withdrawn=False):
    document = json.loads(Path(fixtures).read_text())
    if (not isinstance(document, dict)
            or document.get("schema_version") not in (
                "archaeotrace-visual-contour-cases/1", "archaeotrace-label-gap-cases/1")
            or (document.get("frozen_before_model_outputs") is not True
                and not (document.get("classification_revision_after_outputs") is True
                         and document.get("unchanged_raw_controls_frozen_before_outputs") is True))
            or document.get("human_approved") is not False):
        raise ValueError("Cases require a frozen, explicitly non-human-approved visual review record")
    cases = document["cases"]
    gap_cases = document["schema_version"] == "archaeotrace-label-gap-cases/1"
    audit_paths = [CASE_AUDIT_PATH, CASE_AUDIT_PATH.with_name("korea-label-gap-case-audit-v1.json"),
                   CASE_AUDIT_PATH.with_name("korea-unseen-label-gap-case-audit-v1.json")]
    withdrawn = {row["case_id"] for path in audit_paths if path.exists()
                 for row in json.loads(path.read_text())["withdrawn"]}
    if not allow_withdrawn and any(case["id"] in withdrawn for case in cases):
        raise ValueError("Withdrawn same-contour case: use audited controls; historical replay requires --allow-withdrawn-cases")
    manifest = json.loads((Path(dataset) / "manifest.json").read_text())
    if document.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("Cases belong to a different dataset")
    samples = {row["id"]: row for row in manifest["samples"]}
    for case in cases:
        case["withdrawn_reference"] = case["id"] in withdrawn
        revised_negative = (
            gap_cases and document.get("classification_revision_after_outputs") is True
            and case.get("classification_changed_after_outputs") is True
            and case.get("expected_pair") == "different_contours"
            and (case.get("negative_review_provenance") or {}).get("source")
            == "user_reported_neighbor_hop_after_viewing_candidate"
        )
        if ((case.get("predictions_seen_before_freeze") is not False and not revised_negative)
                or (not gap_cases and case.get("continuous_same_contour") is not True)):
            raise ValueError("Cases must be inspected on one contour before viewing predictions")
        sample = samples[case.get("sample_id", case["id"])]
        if sample["split"] != "validation":
            raise ValueError("Only validation cases may be opened")
        image_path = Path(dataset) / sample["image"]
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != case["image_sha256"] or digest != sample["image_sha256"]:
            raise ValueError("Frozen case image changed: " + case["id"])
        if case.get("review_level") != "agent_visual_inspection":
            raise ValueError("Each case needs explicit visual inspection provenance")
        if gap_cases:
            if case.get("hidden_geometry_ground_truth") is not False:
                raise ValueError("Hidden gap geometry must be explicitly unknown")
            if case.get("expected_pair") not in ("plausible_same_contour", "different_contours", "ambiguous"):
                raise ValueError("Gap cases need explicit endpoint-pair uncertainty")
            points = case.get("prompt_rc")
            segments = case.get("visible_reference_segments_rc", [])
            regions = case.get("unknown_regions_xy")
            if (not isinstance(regions, list) or not regions
                    or any(not isinstance(box, list) or len(box) != 4
                           or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                  or not math.isfinite(v) or not 0 <= v <= 512 for v in box)
                           or box[0] >= box[2] or box[1] >= box[3] for box in regions)):
                raise ValueError("Hidden gap regions must be explicit; never score inferred hidden curves")
            if case["expected_pair"] == "plausible_same_contour" and (
                    not isinstance(segments, list) or len(segments) < 2):
                raise ValueError("Gap accuracy needs visible references on both sides")
            for segment in segments:
                _validate_points(segment)
            foreign = case.get("foreign_visible_control_segments_xy", [])
            if not isinstance(foreign, list):
                raise ValueError("Foreign visible controls must be native XY segments")
            for segment in foreign:
                _validate_points(segment)
            case["reference_rc"] = [p for segment in segments for p in segment] or points
            case["gap_diagnostic"] = True
        else:
            points = case.get("reference_rc")
        _validate_points(points)
        case["image"] = str(image_path)
    return cases


def _validate_points(points):
    if not isinstance(points, list) or len(points) < 2 or any(
            not isinstance(point, list) or len(point) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value < 512
                for value in point) for point in points):
        raise ValueError("References require finite native 512px crop coordinates")


def run_comparison():
    result_path = Path(os.environ[smoke.RESULT_PATH_ENV])
    try:
        import numpy as np
        from osgeo import gdal, osr
        from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsProject, QgsRasterLayer
        from qgis.PyQt.QtCore import QCoreApplication, QTimer
        import qgis.utils

        spec = json.loads(os.environ[SPEC_ENV])
        cases = load_cases(spec["fixtures"], spec["dataset"], allow_withdrawn=spec.get("allow_withdrawn", False))
        dependency_path = os.environ.get("ARCHAEOTRACE_SMOKE_LEARNED_DEPENDENCIES")
        if dependency_path:
            sys.path.append(dependency_path)
        if not qgis.utils.loadPlugin(smoke.PLUGIN_ID) or not qgis.utils.startPlugin(smoke.PLUGIN_ID):
            raise smoke.SmokeFailure("QGIS rejected the installed plugin")
        qgis.utils.plugins[smoke.PLUGIN_ID].run()
        from ai_vectorizer.core.learned_model import LearnedContourEngine, model_guidance_parameters

        engines = {name: LearnedContourEngine.from_bundle(path) for name, path in spec["models"].items()}
        project = QgsProject.instance()
        project.setCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
        canvas = qgis.utils.iface.mapCanvas()
        # Leave room for the plugin dock and a complete 512px native raster.
        qgis.utils.iface.mainWindow().resize(1800, 1480)
        results = []
        gdal.UseExceptions()
        for case in cases:
            pixels = np.moveaxis(gdal.Open(case["image"]).ReadAsArray()[:3], 0, -1)
            height, width = pixels.shape[:2]
            path = Path(os.environ[smoke.PROFILE_ROOT_ENV]) / (case["id"] + ".tif")
            ds = gdal.GetDriverByName("GTiff").Create(str(path), width, height, 3, gdal.GDT_Byte)
            ds.SetGeoTransform((0.0, 1.0, 0.0, float(height), 0.0, -1.0))
            crs = osr.SpatialReference()
            crs.ImportFromEPSG(3857)
            ds.SetProjection(crs.ExportToWkt())
            for band in range(3):
                ds.GetRasterBand(band + 1).WriteArray(pixels[:, :, band])
            ds.FlushCache()
            ds = None
            reference = case.get("prompt_rc", case["reference_rc"])
            # Retain the historical endpoint allowance for the product's
            # six-pixel target snap. Actual input coordinates are recorded
            # independently and must match exactly in version comparisons.
            fixture = {"curve_row": None, "endpoint_tolerance_px": 6.75,
                       "start": (reference[0][1] + 0.5, height - reference[0][0] - 0.5),
                       "end": (reference[-1][1] + 0.5, height - reference[-1][0] - 0.5)}
            fixture["prepare_input"] = prepare_native_input(
                canvas, (width, height), (fixture["start"], fixture["end"]))
            fixture["validate_input"] = fixture["prepare_input"].validate
            layer = QgsRasterLayer(str(path), case["id"])
            if not layer.isValid():
                raise smoke.SmokeFailure("Invalid fixture raster")
            project.addMapLayer(layer)
            canvas.setLayers([layer])
            canvas.setExtent(layer.extent())
            canvas.refresh()
            smoke._wait_until(lambda: not canvas.isDrawing(), "fixture render")
            row = {"id": case["id"], "traces": {}}
            for label, engine in [("ink", None), *engines.items()]:
                for after in (False, True):
                    if spec.get("single_phase") is not None and after != (spec["single_phase"] == "after"):
                        continue
                    key = label + ("_after" if after else "_before")
                    weights = model_guidance_parameters(engine.manifest if engine is not None else None)
                    if not after:
                        if not spec.get("product_comparison", False):
                            old = (engine.manifest["provenance"].get("selected_guidance") or {}) if engine is not None else {}
                            weights = (old.get("support", 0.8), old.get("avoidance", 6.0))
                        if spec.get("before_avoidance") is not None:
                            weights = (weights[0], spec["before_avoidance"])
                    elif spec.get("after_avoidance") is not None:
                        weights = (weights[0], spec["after_avoidance"])
                    row["traces"][key] = smoke._run_trace_scenario(
                        canvas=canvas, raster_layer=layer, fixture=fixture,
                        name=case["id"] + "-" + key, edge_weight=1.0,
                        freehand=False, smart_recovery=False,
                        learned_engine=engine, learned_guide=engine is not None,
                        learned_weights=weights,
                        grid_avoidance=after or spec.get("grid_in_both", False),
                        trace_diagnostics_dir=spec.get("trace_diagnostics_dir"),
                        livewire_config_overrides={
                            "learned_observation_gate": after and spec.get("observation_gate", False),
                            "learned_route_guard": after and spec.get("route_guard", False),
                            "observed_connections": (after or spec.get("compare_label_gaps", False) or spec.get("compare_occluded_labels", False)) and not spec.get("disable_observed", False),
                            "label_gap_connections": (spec.get("compare_occluded_labels", False) or (after and spec.get("compare_label_gaps", False))) and not spec.get("disable_label_gaps", False),
                            "automatic_label_occlusion": (after or spec.get("product_comparison", False)) and spec.get("compare_occluded_labels", False),
                        },
                    )
            results.append(row)
            project.removeMapLayer(layer.id())
        result = {"ok": True, "qgis_version": Qgis.QGIS_VERSION,
                  "review_level": "agent_visual_inspection", "human_approved": False,
                  "withdrawn_references": [case["id"] for case in cases if case["withdrawn_reference"]],
                  "fixtures_sha256": smoke.sha256_file(Path(spec["fixtures"])),
                  "input_policy": "integer mouse events at exact native cell centres; prompt round trips verified",
                  "after_config": {"grid_avoidance": True,
                                   "observed_connections": not spec.get("disable_observed", False),
                                   "label_gap_comparison": spec.get("compare_label_gaps", False),
                                   "label_gap_connections": (spec.get("compare_label_gaps", False) or spec.get("compare_occluded_labels", False)) and not spec.get("disable_label_gaps", False),
                                   "automatic_label_occlusion": spec.get("compare_occluded_labels", False),
                                   "before_avoidance_override": spec.get("before_avoidance"),
                                   "grid_in_both": spec.get("grid_in_both", False),
                                   "avoidance_override": spec.get("after_avoidance"),
                                   "unconfigured_model_avoidance": 0.0,
                                   "learned_observation_gate": spec.get("observation_gate", False),
                                   "learned_route_guard": spec.get("route_guard", False)},
                  "models": {name: engine.metadata for name, engine in engines.items()},
                  "cases": results}
    except BaseException as exc:
        import traceback
        result = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
    finally:
        result_path.write_text(json.dumps(result, indent=2))
        try:
            import qgis.utils
            from qgis.core import QgsProject
            if smoke.PLUGIN_ID in qgis.utils.plugins:
                qgis.utils.unloadPlugin(smoke.PLUGIN_ID)
            QgsProject.instance().clear()
        except Exception:
            pass
        from qgis.PyQt.QtCore import QCoreApplication, QTimer
        QTimer.singleShot(0, QCoreApplication.quit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="archive", type=Path, required=True)
    parser.add_argument("--baseline-zip", type=Path,
                        help="Compare an earlier installed product ZIP at identical real mouse inputs; automatic labels stay enabled in both")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--learned-dependencies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--grid-in-both", action="store_true",
                        help="Keep current grid costs in both runs to isolate the learned-cost change")
    parser.add_argument("--after-avoidance", type=float, choices=(0.0, 2.0, 4.0, 6.0),
                        help="Explicit diagnostic override; omit to use the model's cost policy")
    parser.add_argument("--before-avoidance", type=float, choices=(0.0, 2.0, 4.0, 6.0),
                        help="Explicit baseline override; use 0 to isolate observed connectivity")
    parser.add_argument("--no-observed-connections", action="store_true",
                        help="Disable the observed connection candidate for historical comparison")
    parser.add_argument("--compare-label-gaps", action="store_true",
                        help="Keep native connections in both runs and enable inferred gaps only after")
    parser.add_argument("--compare-occluded-labels", action="store_true",
                        help="Keep existing native/gap paths in both runs; enable detected-label reconnection after")
    parser.add_argument("--trace-diagnostics-dir", type=Path,
                        help="Opt in to local cached-pixel NPZ evidence and route diagnostics")
    parser.add_argument("--no-label-gap-connections", action="store_true",
                        help="Disable inferred gaps for explicit diagnostic ablation")
    parser.add_argument("--allow-withdrawn-cases", action="store_true",
                        help="Historical diagnostic only: include references withdrawn by raw review")
    parser.add_argument("--observation-gate", action="store_true",
                        help="Diagnostic only: restrict learned support to visible Ink (known regressions)")
    parser.add_argument("--route-guard", action="store_true",
                        help="Diagnostic only: prefer Ink over learned candidates (known regressions)")
    args = parser.parse_args()
    if args.compare_label_gaps and args.compare_occluded_labels:
        parser.error("Choose one path-change comparison at a time")
    if args.baseline_zip and not args.compare_occluded_labels:
        parser.error("--baseline-zip requires --compare-occluded-labels")
    if args.baseline_zip and not args.grid_in_both:
        parser.error("--baseline-zip requires --grid-in-both to retain the same grid policy")
    os.environ[SCRIPT_ENV] = str(Path(__file__).resolve())
    load_cases(args.fixtures, args.dataset, allow_withdrawn=args.allow_withdrawn_cases)
    comparison_spec = {
        "fixtures": str(args.fixtures.resolve()), "dataset": str(args.dataset.resolve()),
        "observation_gate": args.observation_gate, "route_guard": args.route_guard,
        "grid_in_both": args.grid_in_both, "after_avoidance": args.after_avoidance,
        "before_avoidance": args.before_avoidance, "disable_observed": args.no_observed_connections,
        "compare_label_gaps": args.compare_label_gaps, "disable_label_gaps": args.no_label_gap_connections,
        "compare_occluded_labels": args.compare_occluded_labels,
        "trace_diagnostics_dir": str(args.trace_diagnostics_dir.resolve()) if args.trace_diagnostics_dir else None,
        "allow_withdrawn": args.allow_withdrawn_cases,
        "models": {"usgs": str(args.old_model.resolve()), "korea": str(args.new_model.resolve())},
        "product_comparison": args.baseline_zip is not None}
    args.qgis_executable = Path("/Applications/QGIS.app/Contents/MacOS/QGIS")
    args.expected_version = smoke.DEFAULT_EXPECTED_VERSION
    if args.baseline_zip:
        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.archive = args.baseline_zip
        if args.profile_root is not None:
            baseline_args.profile_root = args.profile_root / "baseline"
            args.profile_root = args.profile_root / "current"
        os.environ[SPEC_ENV] = json.dumps({**comparison_spec, "single_phase": "before"})
        before = smoke.run_outer(baseline_args, bootstrap_script=Path(__file__))
        os.environ[SPEC_ENV] = json.dumps({**comparison_spec, "single_phase": "after"})
        result = merge_archive_comparisons(before, smoke.run_outer(args, bootstrap_script=Path(__file__)))
    else:
        os.environ[SPEC_ENV] = json.dumps(comparison_spec)
        result = smoke.run_outer(args, bootstrap_script=Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    score_and_render(result, load_cases(args.fixtures, args.dataset, allow_withdrawn=args.allow_withdrawn_cases), args.output.parent)
    print(json.dumps({"ok": result["ok"], "cases": len(result["cases"]), "output": str(args.output)}))


def minimum_polyline_distance(path, other_polylines):
    """Return the continuous segment distance without joining separate lines.

    Endpoint projections cover parallel, collinear and zero-length segments;
    an interior intersection contributes zero. Pairwise arrays are restricted
    to 128 by 128 segment chunks, independently of the saved route's length.
    All coordinates must already use the same native pixel-centre convention.
    """
    import numpy as np

    def points(value):
        array = np.asarray(value, dtype=np.float64)
        if (array.ndim != 2 or array.shape[1] != 2 or len(array) < 2
                or not np.isfinite(array).all()):
            raise ValueError("Polyline distances need finite XY lines with at least two points")
        return array

    route = points(path)
    others = [points(polyline) for polyline in other_polylines]
    if not others:
        raise ValueError("Polyline distances need at least one other line")
    # Concatenate real segment endpoints, not polyline vertices: no edge may
    # appear between the end of one inspected stroke and the next stroke.
    other_starts = np.concatenate([line[:-1] for line in others])
    other_ends = np.concatenate([line[1:] for line in others])

    def cross(first, last):
        return first[..., 0] * last[..., 1] - first[..., 1] * last[..., 0]

    def endpoint_squared(endpoint, start, vector):
        offset = endpoint - start
        length_squared = np.sum(vector * vector, axis=-1)
        projection = np.divide(
            np.sum(offset * vector, axis=-1), length_squared,
            out=np.zeros(offset.shape[:-1], dtype=np.float64),
            where=length_squared > 0.0,
        )
        residual = offset - np.clip(projection, 0.0, 1.0)[..., None] * vector
        return np.sum(residual * residual, axis=-1)

    best_squared = float("inf")
    chunk = 128
    for first_index in range(0, len(route) - 1, chunk):
        stop = min(len(route) - 1, first_index + chunk)
        first = route[first_index:stop, None, :]
        last = route[first_index + 1:stop + 1, None, :]
        direction = last - first
        for other_index in range(0, len(other_starts), chunk):
            other_first = other_starts[None, other_index:other_index + chunk, :]
            other_last = other_ends[None, other_index:other_index + chunk, :]
            other_direction = other_last - other_first
            offset = other_first - first
            denominator = cross(direction, other_direction)
            nonparallel = denominator != 0.0
            first_fraction = np.divide(
                cross(offset, other_direction), denominator,
                out=np.zeros_like(denominator), where=nonparallel,
            )
            other_fraction = np.divide(
                cross(offset, direction), denominator,
                out=np.zeros_like(denominator), where=nonparallel,
            )
            if np.any(nonparallel & (first_fraction >= 0.0) & (first_fraction <= 1.0)
                      & (other_fraction >= 0.0) & (other_fraction <= 1.0)):
                return 0.0
            distances = np.minimum.reduce((
                endpoint_squared(first, other_first, other_direction),
                endpoint_squared(last, other_first, other_direction),
                endpoint_squared(other_first, first, direction),
                endpoint_squared(other_last, first, direction),
            ))
            best_squared = min(best_squared, float(distances.min()))
            if best_squared == 0.0:
                return 0.0
    return math.sqrt(best_squared)


def score_and_render(result, cases, output):
    import numpy as np
    from PIL import Image, ImageDraw

    def dense(points):
        points = np.asarray(points, dtype=float)
        segments = [points[:1]]
        for first, last in zip(points[:-1], points[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(last - first) * 2)))
            segments.append(first + (last - first) * np.linspace(0, 1, count + 1)[1:, None])
        return np.concatenate(segments)

    def nearest(first, second):
        return np.concatenate([np.sqrt(np.sum((part[:, None] - second[None]) ** 2, axis=2)).min(axis=1)
                               for part in np.array_split(first, max(1, len(first) // 128))])

    fixtures = {case["id"]: case for case in cases}
    scored = []
    colours = {"ink": (205, 0, 175), "usgs": (220, 120, 0), "korea": (0, 150, 0)}
    for case in result["cases"]:
        fixture = fixtures[case["id"]]
        source = Image.open(fixture["image"]).convert("RGB")
        points = [(col + 0.5, row + 0.5) for row, col in fixture["reference_rc"]]
        is_gap = fixture.get("gap_diagnostic", False)
        visible = [[(col + 0.5, row + 0.5) for row, col in segment]
                   for segment in fixture.get("visible_reference_segments_rc", [])]
        reference = np.concatenate([dense(segment) for segment in visible]) if visible else dense(points)
        prompts = [(col + 0.5, row + 0.5) for row, col in fixture.get("prompt_rc", [])] or [points[0], points[-1]]
        # Keep failed detours visible too. Framing only the intended arms can
        # clip a route which leaves the contour and later returns to the click.
        displayed = np.concatenate([reference] + [
            np.asarray([(x, source.height - y) for x, y in trace["vertices_xy"]])
            for trace in case["traces"].values()
        ])
        minimum, maximum = displayed.min(axis=0) - 18, displayed.max(axis=0) + 18
        box = (max(0, int(minimum[0])), max(0, int(minimum[1])),
               min(source.width, int(maximum[0]) + 1), min(source.height, int(maximum[1]) + 1))
        scale = 3
        tile_width, tile_height = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
        panel = Image.new("RGB", (tile_width * 4, (tile_height + 24) * 2), "white")
        scores = {}
        foreign_controls = fixture.get("foreign_visible_control_segments_xy", [])
        foreign = [np.asarray(segment, dtype=np.float64) + 0.5
                   for segment in foreign_controls]
        foreign_contacts = {}
        for index, (label, trace) in enumerate(case["traces"].items()):
            saved = [(x, source.height - y) for x, y in trace["vertices_xy"]]
            prediction = dense(saved)
            if foreign:
                # These are separately inspected visible neighboring strokes.
                # A broad label box must not hide contact with known foreign
                # ink. This does not establish the missing contour's shape.
                distance = minimum_polyline_distance(saved, foreign)
                foreign_contacts[label] = {
                    "minimum_distance_px": round(distance, 3),
                    "contact_within_1px": distance <= 1.0,
                    "distance_method": "continuous_segments",
                }
            if is_gap:
                if fixture["expected_pair"] != "plausible_same_contour":
                    scores[label] = {"visible_f1_2px": None, "policy": "negative pair; no accuracy reference"}
                    continue
                unknown = np.zeros(len(prediction), dtype=bool)
                for x0, y0, x1, y1 in fixture["unknown_regions_xy"]:
                    unknown |= ((prediction[:, 0] >= x0) & (prediction[:, 0] <= x1)
                                & (prediction[:, 1] >= y0) & (prediction[:, 1] <= y1))
                prediction = prediction[~unknown]
                if not len(prediction):
                    scores[label] = {"visible_f1_2px": 0.0, "visible_p95_px": None}
                    continue
            precision = float(np.mean(nearest(prediction, reference) <= 2))
            recall = float(np.mean(nearest(reference, prediction) <= 2))
            distances = nearest(prediction, reference)
            scores[label] = {"median_px": round(float(np.median(distances)), 3),
                             "p95_px": round(float(np.percentile(distances, 95)), 3),
                             "precision_2px": round(precision, 3), "recall_2px": round(recall, 3),
                             "f1_2px": round(2 * precision * recall / max(1e-12, precision + recall), 3)}
            if is_gap:
                scores[label] = {"visible_" + key: value for key, value in scores[label].items()}
        for row, after in enumerate((False, True)):
            for col, method in enumerate(("reference", "ink", "usgs", "korea")):
                tile = source.crop(box).resize((tile_width, tile_height), Image.Resampling.NEAREST)
                draw = ImageDraw.Draw(tile)
                to_tile = lambda path: [((x - box[0]) * scale, (y - box[1]) * scale) for x, y in path]
                if method == "reference":
                    if row == 1:
                        for segment in (visible if is_gap else [points]):
                            draw.line(to_tile(segment), fill=(0, 120, 255), width=2)
                    for x, y in to_tile(prompts):
                        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(230, 0, 0), width=2)
                    label = "Raw + endpoints" if row == 0 else (
                        "WITHDRAWN reference" if fixture.get("withdrawn_reference") else "Inspected reference"
                    )
                    if is_gap and row == 1:
                        label = ("WITHDRAWN visible reference" if fixture.get("withdrawn_reference")
                                 else "Visible only; hidden curve unknown")
                else:
                    label = method + ("_after" if after else "_before")
                    saved = [(x, source.height - y) for x, y in case["traces"][label]["vertices_xy"]]
                    draw.line(to_tile(saved), fill=colours[method], width=2)
                    label += (" visible F1=" if is_gap else " F1=") + str(scores[label]["visible_f1_2px" if is_gap else "f1_2px"])
                panel.paste(tile, (col * tile_width, row * (tile_height + 24) + 24))
                ImageDraw.Draw(panel).text((col * tile_width + 4, row * (tile_height + 24) + 4), label, fill="black")
        panel.save(output / (case["id"] + "-comparison.png"))
        scored.append({"id": case["id"], "scores": scores,
                       **({"foreign_visible_contacts": foreign_contacts} if foreign_contacts else {}),
                       **({"hidden_geometry_scored": False, "expected_pair": fixture["expected_pair"]} if is_gap else {})})
    summary = {"scope": "agent visually inspected validation diagnostics; no human approval or holdout",
               "withdrawn_references": result.get("withdrawn_references", []),
               "cases": scored}
    (output / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")


if os.environ.get(smoke.BOOTSTRAP_ENV) == "1":
    from qgis.PyQt.QtCore import QTimer
    QTimer.singleShot(0, run_comparison)
elif __name__ == "__main__":
    main()
