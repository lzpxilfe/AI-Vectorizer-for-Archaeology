"""Product-kernel parity and honest evidence for manual contour gap checks."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from ai_vectorizer.core.edge_detector import EdgeDetector
from ai_vectorizer.core.livewire import blend_path_with_cursor
from ai_vectorizer.core.manual_gap_bridge import (
    build_manual_gap_bridge,
    sample_manual_gap_tangent,
)
from benchmarks.manual_gap_shadow import (
    build_manual_gap_shadow_cases,
    main,
    run_manual_gap_case,
    run_neutral_manual_gap_shadow,
)


def test_original_shadow_rejects_ambiguous_glyphs_and_keeps_same_segment_control():
    result = run_neutral_manual_gap_shadow()

    assert result["historical_map_evidence"] is False
    assert result["publication_ranking_eligible"] is False
    assert result["human_usability_study"] is False
    assert result["prompt"] == {
        "start_xy": [116.0, 118.0], "end_xy": [164.0, 113.0], "previous_xy": None,
    }
    control = result["ink_same_segment_control"]
    manual = result["manual_gap_bridge_product_kernel_v2"]
    assert control["score"]["candidate_smoke_gate"]["passed"] is False
    assert manual["requires_explicit_user_anchors"] is True
    assert manual["automatic_endpoint_selection"] is False
    assert manual["model_or_ocr_used"] is False
    assert manual["status"] == "ink_fallback"
    assert manual["reason"]
    assert manual["bridge_points_xy"] is None
    assert manual["effective_points_xy"] == control["points_xy"]
    assert manual["route_sha256"] == control["route_sha256"]
    assert result["full_trace_context"]["included_in_paired_comparison"] is False
    assert result["full_trace_context"]["prompt"]["start_xy"] != result["prompt"]["start_xy"]


@pytest.mark.parametrize("case_index", [4, 5])
def test_clear_gap_runs_same_sampled_tangent_and_blend_as_product(case_index):
    case = build_manual_gap_shadow_cases()[case_index]
    result = run_manual_gap_case(case)
    manual = result["manual_gap_bridge_product_kernel_v2"]
    evidence = EdgeDetector.detect_ink_evidence(case.image_rgb, tile_origin=(0, 0))
    start, end = result["prompt"]["start_xy"], result["prompt"]["end_xy"]
    start_tangent = sample_manual_gap_tangent(evidence, start, radius_pixels=3)
    end_tangent = sample_manual_gap_tangent(evidence, end, radius_pixels=3)
    bridge = build_manual_gap_bridge(start, end, start_tangent, end_tangent)
    expected = blend_path_with_cursor(bridge.points_xy, start, end, 1.0)

    assert manual["status"] == "preview"
    assert manual["sampled_start_tangent_xy"] == list(start_tangent)
    assert manual["sampled_end_tangent_xy"] == list(end_tangent)
    assert manual["effective_points_xy"] == [[float(x), float(y)] for x, y in expected]
    assert manual["score"]["endpoint_preserved"] is True
    assert manual["score"]["candidate_smoke_gate"]["passed"] is True
    assert manual["score"]["numeric_label_gap_coverage_within_4px"] == 1.0
    assert manual["score"]["parallel_switch_fraction"] == 0.0
    # The easy control already covers this gap; this is not a hard-case win.
    assert result["ink_same_segment_control"]["score"]["candidate_smoke_gate"]["passed"] is True


def test_reference_changes_cannot_change_generated_bridge_or_control():
    case = build_manual_gap_shadow_cases()[4]
    original = run_manual_gap_case(case)
    changed_reference = replace(
        case.reference_case,
        reference_xy=tuple((x, y + 30) for x, y in case.reference_case.reference_xy),
        parallel_xy=tuple((x, y + 30) for x, y in case.reference_case.parallel_xy),
    )
    changed = run_manual_gap_case(replace(case, reference_case=changed_reference))

    assert original["image_sha256"] == changed["image_sha256"]
    assert original["reference_sha256"] != changed["reference_sha256"]
    assert original["prompt_sha256"] == changed["prompt_sha256"]
    for method in ("ink_same_segment_control", "manual_gap_bridge_product_kernel_v2"):
        assert original[method]["route_sha256"] == changed[method]["route_sha256"]
        assert original[method]["score"]["target_p95_distance_px"] < 2.0
        assert changed[method]["score"]["target_p95_distance_px"] > 25.0


def test_gray_and_rotated_fixtures_preserve_lossless_prompt_coordinates():
    cases = build_manual_gap_shadow_cases()
    assert len(cases) == 6
    for case in cases:
        if case.grayscale:
            np.testing.assert_array_equal(case.image_rgb[..., 0], case.image_rgb[..., 1])
            np.testing.assert_array_equal(case.image_rgb[..., 1], case.image_rgb[..., 2])
        for point in (case.canonical_start_xy, case.canonical_end_xy):
            assert case.to_reference_xy(case.to_image_xy(point)) == point
    np.testing.assert_array_equal(cases[1].image_rgb, np.rot90(cases[0].image_rgb))
    np.testing.assert_array_equal(cases[3].image_rgb, np.rot90(cases[2].image_rgb))


def test_shadow_cli_writes_deterministic_routes_prompts_and_source_provenance(tmp_path, capsys):
    runs = []
    for index in range(2):
        output = tmp_path / ("shadow-{}".format(index))
        assert main(["--output-dir", str(output)]) == 0
        emitted = json.loads(capsys.readouterr().out)
        persisted = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert emitted == persisted
        assert len(list(output.glob("*.ppm"))) == 6
        for case in persisted["cases"]:
            assert (output / (case["challenge_id"] + ".ppm")).read_bytes().startswith(b"P6\n")
            for field, hash_field in (("prompt", "prompt_sha256"), ("configuration", "configuration_sha256")):
                encoded = json.dumps(case[field], sort_keys=True, separators=(",", ":"), allow_nan=False)
                assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == case[hash_field]
            assert all(len(digest) == 64 for digest in case["source_sha256"].values())
        runs.append(persisted)
    assert runs[0] == runs[1]
