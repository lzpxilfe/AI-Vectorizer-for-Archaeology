#!/usr/bin/env python3
"""Summarize opt-in QGIS decision logs without ranking unknown hidden curves.

Stage labels locate a decision in the pipeline; they do not establish that a
rejected candidate was correct or that a retained ordinary route was wrong.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def _text_stage(build, attempt):
    text = build.get("text_gap", {})
    if attempt.get("status") == "accepted":
        return "text_gap_accepted"
    if attempt.get("status") == "preserved":
        return "existing_route_preserved"
    if attempt.get("reason") == "healthy_outside_native_route":
        return "healthy_native_route_preserved"
    if text.get("status") in ("skipped", "error"):
        return "text_builder_" + text["status"]
    if text.get("reason") == "no_detected_regions":
        return "text_proposal_unavailable"
    if text.get("reason") == "connector_rejected":
        return "text_endpoint_or_curve_rejected"
    if attempt.get("reason") == "no_target_candidate":
        return "text_candidate_does_not_reach_target"
    if attempt.get("reason") == "candidate_display_rejected":
        return "text_candidate_display_rejected"
    return "text_not_selected"


def _finite_coordinates(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(not isinstance(item, bool) and isinstance(item, (int, float))
                    and math.isfinite(item) for item in value))


def _bbox_distance(first, last):
    dx = max(first[0] - last[2], last[0] - first[2], 0.0)
    dy = max(first[1] - last[3], last[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def _text_localization(route_diagnostics, build, fixture):
    """Localize bounded recorded proposal boxes, without inferring identity.

    Regions belong to the worker's local window. A box overlapping an unknown
    rectangle is only spatial context, not proof that its connector represents
    the queried glyph or either user-selected contour endpoint.
    """
    localization = {
        "status": "unavailable",
        "scope": "Recorded connection-selected window proposals versus queried unknown rectangles; no glyph or endpoint identity is established",
        "records_scope": "Completeness is relative to connection.region_count, not every detector proposal before selection",
        "coordinate_policy": "Absolute native XY bounds = local bbox + route origin; positive-area overlap; touching boxes have zero distance",
        "records_complete": None,
        "queried_glyph_endpoint_rejection_established": False,
        "accuracy_verdict": None,
    }
    origin = route_diagnostics.get("origin")
    if not _finite_coordinates(origin, 2):
        localization["reason"] = "missing_or_invalid_window_origin"
        return localization
    connection = build.get("text_gap", {}).get("connection")
    if not isinstance(connection, dict) or not isinstance(connection.get("regions"), list):
        localization["reason"] = "missing_connection_region_records"
        return localization
    unknown = fixture.get("unknown_regions_xy")
    if (not isinstance(unknown, list) or not unknown
            or any(not _finite_coordinates(box, 4) or box[0] >= box[2] or box[1] >= box[3]
                   for box in unknown)):
        localization["reason"] = "missing_or_invalid_unknown_rectangles"
        return localization
    records = connection["regions"]
    declared = connection.get("region_count")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        declared = None
    details, invalid = [], 0
    overlap_count, overlap_pairs, minimum = 0, 0, None
    for position, record in enumerate(records[:16]):
        box = record.get("bbox_xyxy") if isinstance(record, dict) else None
        if not _finite_coordinates(box, 4) or box[0] >= box[2] or box[1] >= box[3]:
            invalid += 1
            continue
        absolute = [float(box[0] + origin[0]), float(box[1] + origin[1]),
                    float(box[2] + origin[0]), float(box[3] + origin[1])]
        overlaps = sum(max(absolute[0], query[0]) < min(absolute[2], query[2])
                       and max(absolute[1], query[1]) < min(absolute[3], query[3])
                       for query in unknown)
        distance = min(_bbox_distance(absolute, query) for query in unknown)
        overlap_count += int(overlaps > 0)
        overlap_pairs += overlaps
        minimum = distance if minimum is None else min(minimum, distance)
        index = record.get("region_index", position)
        if isinstance(index, bool) or not isinstance(index, int):
            index = position
        details.append({"region_index": index, "absolute_bbox_xyxy": absolute,
                        "overlapping_unknown_rectangle_count": overlaps,
                        "minimum_bbox_distance_px": distance})
    complete = (declared == len(records) and len(records) <= 16 and not invalid) if declared is not None else None
    limitations = []
    if declared is None:
        limitations.append("declared_region_count_unavailable")
    elif declared != len(records):
        limitations.append("recorded_count_differs_from_declared_regions")
    if len(records) > 16:
        limitations.append("only_first_16_region_records_inspected")
    if invalid:
        limitations.append("invalid_recorded_region_bounds")
    if complete is not True:
        limitations.append("unrecorded_or_uninspected_proposals_may_overlap_query")
    localization.update(
        status="available", reason="recorded_bbox_localization",
        origin_xy=list(origin), unknown_rectangle_count=len(unknown),
        declared_region_count=declared, recorded_region_count=len(records),
        inspected_region_count=min(len(records), 16), valid_region_record_count=len(details),
        records_complete=complete, limitations=limitations,
        overlapping_recorded_region_count=overlap_count, bbox_overlap_pair_count=overlap_pairs,
        minimum_bbox_distance_px=minimum, regions=details,
        spatial_observation=("recorded_proposals_overlap_query" if overlap_count
                             else "no_recorded_proposal_overlaps_query"),
    )
    return localization


def summarize(result, scores, fixtures):
    if not result.get("ok"):
        raise ValueError("Only completed QGIS executions may be summarized")
    if fixtures.get("human_approved") is not False:
        raise ValueError("Development fixtures must declare their review scope")
    frozen = {row["id"]: row for row in fixtures["cases"]}
    results = {row["id"]: row for row in result["cases"]}
    score_rows = {row["id"]: row for row in scores["cases"]}
    if (len(frozen) != len(fixtures["cases"]) or frozen.keys() != results.keys()
            or frozen.keys() != score_rows.keys()):
        raise ValueError("Fixture, result and score case identities differ")
    rows, branches, stages = [], {}, {}
    for case_id, fixture in frozen.items():
        if (fixture.get("hidden_geometry_ground_truth") is not False
                or fixture.get("expected_pair") not in (
                    "plausible_same_contour", "different_contours", "ambiguous")):
            raise ValueError("Gap summaries require explicit pair and hidden-geometry uncertainty")
        case = {"id": case_id, "expected_pair": fixture["expected_pair"],
                "accuracy_reference_available": fixture["expected_pair"] == "plausible_same_contour",
                "hidden_geometry_scored": False, "methods": {}}
        for method in ("ink", "usgs", "korea"):
            key = method + "_after"
            trace = results[case_id]["traces"][key]
            route_diagnostics = trace.get("route_diagnostics") or {}
            decisions = route_diagnostics.get("decision_trace")
            if (not isinstance(decisions, dict)
                    or decisions.get("schema_version") != "livewire-trace-diagnostics/1"
                    or not isinstance(decisions.get("build"), dict)):
                raise ValueError("Missing decision logs; rerun QGIS with --trace-diagnostics-dir")
            build = decisions["build"]
            attempt = next((row for row in decisions.get("gap_attempts", [])
                            if row.get("kind") == "automatic-text-gap"), {})
            stage = _text_stage(build, attempt)
            selection = decisions.get("selection", {})
            branch = selection.get("branch", "unrecorded")
            score = score_rows[case_id]["scores"][key]
            if fixture["expected_pair"] != "plausible_same_contour" and score.get("visible_f1_2px") is not None:
                raise ValueError("Negative/ambiguous pairs cannot have accuracy scores")
            row = {
                "route_provenance": trace.get("route_provenance"),
                "selection": selection, "ordinary": decisions.get("ordinary", {}),
                "text_stage": stage, "text_build": build.get("text_gap", {}),
                "text_stage_scope": (
                    "Window-level builder stage; connector rejection does not establish rejection of the queried glyph endpoints"
                    if stage in ("text_endpoint_or_curve_rejected", "text_proposal_unavailable",
                                 "text_builder_skipped", "text_builder_error")
                    else "Target-level route decision; no glyph or contour identity accuracy verdict"
                ),
                "text_localization": _text_localization(route_diagnostics, build, fixture),
                "text_target_decision": attempt, "mixing": build.get("mixing", {}),
                "scores": score,
                "foreign_visible_contact": score_rows[case_id].get("foreign_visible_contacts", {}).get(key),
                "one_undo": trace.get("one_undo"),
            }
            case["methods"][method] = row
            branches.setdefault(method, Counter())[branch] += 1
            stages.setdefault(method, Counter())[stage] += 1
        rows.append(case)
    return {
        "schema_version": "archaeotrace-gap-decision-summary/1",
        "scope": "Agent-reviewed development diagnostics; stage labels are not accuracy verdicts",
        "hidden_geometry_scored": False, "human_approved": False,
        "model_method_mapping": {"ink": "no learned model", "usgs": "old-model argument",
                                 "korea": "new-model argument"},
        "models": result.get("models", {}), "case_count": len(rows),
        "pair_counts": dict(Counter(row["expected_pair"] for row in rows)),
        "selected_branch_counts": {key: dict(value) for key, value in branches.items()},
        "text_stage_counts": {key: dict(value) for key, value in stages.items()}, "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {key: getattr(args, key).read_bytes() for key in ("result", "scores", "fixtures")}
    result, scores, fixtures = (json.loads(inputs[key]) for key in ("result", "scores", "fixtures"))
    if result.get("fixtures_sha256") != hashlib.sha256(inputs["fixtures"]).hexdigest():
        raise ValueError("QGIS result is not bound to this frozen fixture")
    document = summarize(result, scores, fixtures)
    document["input_sha256"] = {key: hashlib.sha256(value).hexdigest() for key, value in inputs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"case_count": document["case_count"], "selected_branch_counts": document["selected_branch_counts"],
                      "text_stage_counts": document["text_stage_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
