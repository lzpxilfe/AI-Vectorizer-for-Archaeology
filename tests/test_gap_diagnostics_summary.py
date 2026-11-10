"""Decision stages must not become invented hidden-geometry accuracy labels."""
from copy import deepcopy

import pytest

from scripts.summarize_gap_diagnostics import summarize


def _inputs(pair="plausible_same_contour"):
    fixture = {"human_approved": False, "cases": [{"id": "gap", "expected_pair": pair,
                "hidden_geometry_ground_truth": False}]}
    decision = {"schema_version": "livewire-trace-diagnostics/1",
                "build": {"text_gap": {"status": "built", "reason": "connectors_built"}},
                "ordinary": {"branch": "observed_native", "reason": "connected_native"},
                "selection": {"branch": "ordinary_preserved", "reason": "visible_arms_preserved"},
                "gap_attempts": [{"kind": "automatic-text-gap", "status": "preserved",
                                  "reason": "ordinary_retains_visible_arms"}]}
    result = {"ok": True, "cases": [{"id": "gap", "traces": {
        method + "_after": {"route_provenance": "ink", "one_undo": True,
                            "route_diagnostics": {"decision_trace": deepcopy(decision)}}
        for method in ("ink", "usgs", "korea")}}]}
    scores = {"cases": [{"id": "gap", "scores": {method + "_after": {
        "visible_f1_2px": 1.0 if pair == "plausible_same_contour" else None}
        for method in ("ink", "usgs", "korea")}}]}
    return result, scores, fixture


def test_healthy_preservation_is_not_labeled_a_connection_failure():
    result, scores, fixture = _inputs()
    row = summarize(result, scores, fixture)["cases"][0]
    assert row["methods"]["ink"]["text_stage"] == "existing_route_preserved"
    assert row["hidden_geometry_scored"] is False
    assert "failure" not in row


@pytest.mark.parametrize("pair", ["different_contours", "ambiguous"])
def test_unscored_pairs_keep_uncertainty(pair):
    result, scores, fixture = _inputs(pair)
    row = summarize(result, scores, fixture)["cases"][0]
    assert not row["accuracy_reference_available"]
    assert row["methods"]["korea"]["scores"]["visible_f1_2px"] is None
    scores["cases"][0]["scores"]["ink_after"]["visible_f1_2px"] = 1.0
    with pytest.raises(ValueError, match="cannot have accuracy"):
        summarize(result, scores, fixture)


def test_detector_absence_and_connector_rejection_remain_distinct():
    result, scores, fixture = _inputs()
    for method, reason in (("ink", "no_detected_regions"), ("korea", "connector_rejected")):
        d = result["cases"][0]["traces"][method + "_after"]["route_diagnostics"]["decision_trace"]
        d["build"]["text_gap"] = {"status": "unavailable", "reason": reason}
        d["gap_attempts"][0].update(status="skipped", reason="connection_unavailable")
    rows = summarize(result, scores, fixture)["cases"][0]["methods"]
    assert rows["ink"]["text_stage"] == "text_proposal_unavailable"
    assert rows["korea"]["text_stage"] == "text_endpoint_or_curve_rejected"
    assert "Window-level" in rows["korea"]["text_stage_scope"]
    assert rows["korea"]["text_localization"]["status"] == "unavailable"


def test_legacy_or_unbound_decisions_are_not_silently_summarized():
    result, scores, fixture = _inputs()
    result["cases"][0]["traces"]["ink_after"]["route_diagnostics"] = {}
    with pytest.raises(ValueError, match="Missing decision logs"):
        summarize(result, scores, fixture)
    result, scores, fixture = _inputs()
    scores["cases"][0]["id"] = "other"
    with pytest.raises(ValueError, match="identities differ"):
        summarize(result, scores, fixture)


def _localized_inputs(records, count, origin=(100, 150)):
    result, scores, fixture = _inputs()
    fixture["cases"][0]["unknown_regions_xy"] = [[138, 178, 144, 184]]
    for trace in result["cases"][0]["traces"].values():
        trace["route_diagnostics"]["origin"] = list(origin)
        decisions = trace["route_diagnostics"]["decision_trace"]
        decisions["gap_attempts"][0].update(status="skipped", reason="connection_unavailable")
        decisions["build"]["text_gap"] = {
            "status": "unavailable", "reason": "connector_rejected",
            "connection": {"region_count": count, "regions": records},
        }
    return result, scores, fixture


def test_window_rejection_does_not_claim_queried_o_endpoints_were_examined():
    # These local bounds superficially match the source query. Adding the
    # worker origin places the recorded proposal elsewhere on the source.
    inputs = _localized_inputs([{"region_index": 0, "bbox_xyxy": [138, 178, 144, 184]}], 1)
    row = summarize(*inputs)["cases"][0]["methods"]["korea"]
    assert row["text_stage"] == "text_endpoint_or_curve_rejected"
    localization = row["text_localization"]
    assert localization["status"] == "available"
    assert localization["records_complete"] is True
    assert localization["regions"][0]["absolute_bbox_xyxy"] == [238.0, 328.0, 244.0, 334.0]
    assert localization["overlapping_recorded_region_count"] == 0
    assert localization["minimum_bbox_distance_px"] == pytest.approx((94 ** 2 + 144 ** 2) ** 0.5)
    assert localization["spatial_observation"] == "no_recorded_proposal_overlaps_query"
    assert localization["queried_glyph_endpoint_rejection_established"] is False
    assert localization["accuracy_verdict"] is None


def test_local_proposal_bbox_overlap_uses_origin_and_does_not_establish_identity():
    inputs = _localized_inputs([{"region_index": 4, "bbox_xyxy": [38, 28, 44, 34]}], 1)
    localization = summarize(*inputs)["cases"][0]["methods"]["ink"]["text_localization"]
    assert localization["regions"][0]["absolute_bbox_xyxy"] == [138.0, 178.0, 144.0, 184.0]
    assert localization["overlapping_recorded_region_count"] == 1
    assert localization["bbox_overlap_pair_count"] == 1
    assert localization["minimum_bbox_distance_px"] == 0.0
    assert localization["queried_glyph_endpoint_rejection_established"] is False


def test_partial_or_capped_region_records_cannot_establish_proposal_absence():
    inputs = _localized_inputs([{"bbox_xyxy": [0, 0, 10, 10]}], 17)
    localization = summarize(*inputs)["cases"][0]["methods"]["ink"]["text_localization"]
    assert localization["declared_region_count"] == 17
    assert localization["recorded_region_count"] == 1
    assert localization["records_complete"] is False
    assert localization["overlapping_recorded_region_count"] == 0
    assert "unrecorded_or_uninspected_proposals_may_overlap_query" in localization["limitations"]
    records = [{"bbox_xyxy": [0, 0, 10, 10]} for _ in range(17)]
    records[-1] = {"bbox_xyxy": [38, 28, 44, 34]}
    localization = summarize(*_localized_inputs(records, 17))["cases"][0]["methods"]["ink"]["text_localization"]
    assert localization["inspected_region_count"] == len(localization["regions"]) == 16
    assert localization["records_complete"] is False
    assert "only_first_16_region_records_inspected" in localization["limitations"]


def test_missing_legacy_region_records_are_unavailable_even_with_origin():
    result, scores, fixture = _inputs()
    fixture["cases"][0]["unknown_regions_xy"] = [[138, 178, 144, 184]]
    for trace in result["cases"][0]["traces"].values():
        trace["route_diagnostics"]["origin"] = [100, 150]
    localization = summarize(result, scores, fixture)["cases"][0]["methods"]["ink"]["text_localization"]
    assert localization["status"] == "unavailable"
    assert localization["reason"] == "missing_connection_region_records"
