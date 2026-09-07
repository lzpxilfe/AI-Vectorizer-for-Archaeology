"""Integration regression for the real product Ink tracing paths."""

import json

from benchmarks.complex_synthetic import (
    NUMERIC_LABEL_GAP_X,
    build_complex_trace_case,
    candidate_smoke_gate,
    main,
    run_complex_ink_challenge,
    score_complex_candidate_path,
)


def test_complex_challenge_exercises_real_ink_v1_and_v2_paths():
    result = run_complex_ink_challenge()

    assert result["historical_map_evidence"] is False
    assert result["publication_ranking_eligible"] is False
    assert result["input"]["width"] == 256
    assert result["input"]["height"] == 256
    assert len(result["input"]["failure_modes"]) >= 7

    v1 = result["ink_livewire_v1"]
    v2 = result["ink_livewire_v2"]
    assert v2["endpoint_preserved"] is True
    assert v1["edge_pixels"] > 200
    assert v2["evidence_centerline_pixels"] > 200
    assert v2["evidence_score_p95"] > 0.0

    # V2's anchored coloured-contour bridge must materially improve over v1
    # without turning the label gap into a parallel-contour switch.
    assert v2["target_p95_distance_px"] < v1["target_p95_distance_px"]
    assert (
        v2["reference_coverage_within_4px"]
        > v1["reference_coverage_within_4px"]
    )
    assert v2["parallel_switch_fraction"] < v1["parallel_switch_fraction"]
    assert v2["numeric_label_gap_coverage_within_4px"] >= 0.85
    assert v2["candidate_smoke_gate"]["passed"] is True


def test_complex_challenge_is_deterministic_and_cli_emits_inspectable_artifacts(tmp_path, capsys):
    first = run_complex_ink_challenge()
    second = run_complex_ink_challenge()
    assert first["image_sha256"] == second["image_sha256"]
    assert (
        first["ink_livewire_v1"]["path_sha256"]
        == second["ink_livewire_v1"]["path_sha256"]
    )
    assert (
        first["ink_livewire_v2"]["path_sha256"]
        == second["ink_livewire_v2"]["path_sha256"]
    )

    output = tmp_path / "challenge"
    assert main(["--output-dir", str(output)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    persisted = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert emitted == persisted
    assert (output / "complex-trace-challenge.ppm").read_bytes().startswith(b"P6\n")
    assert persisted["image_sha256"] == first["image_sha256"]


def test_complex_case_uses_an_independent_reference_not_an_evidence_mask():
    case = build_complex_trace_case()
    assert case.reference_xy[0] == case.start_xy
    assert case.reference_xy[-1] == case.end_xy
    assert len(case.reference_xy) == 225
    assert NUMERIC_LABEL_GAP_X == (118, 162)
    assert all(target[1] + 14 == parallel[1] for target, parallel in zip(case.reference_xy, case.parallel_xy))
    # This is an actual contour omission for a printed elevation label, not
    # simply a dark number painted over cyan contour pixels.
    label_columns = case.image_rgb[:, NUMERIC_LABEL_GAP_X[0] : NUMERIC_LABEL_GAP_X[1] + 1]
    assert not (label_columns == (185, 226, 226)).all(axis=2).any()


def test_candidate_score_is_bounded_and_uses_the_same_strict_gate():
    case = build_complex_trace_case()
    good = score_complex_candidate_path(case.reference_xy)
    assert good["endpoint_preserved"] is True
    assert good["candidate_smoke_gate"]["passed"] is True

    wrong_parallel = score_complex_candidate_path(
        (case.start_xy, *case.parallel_xy, case.end_xy)
    )
    assert wrong_parallel["candidate_smoke_gate"]["passed"] is False
    assert "parallel_switch_fraction_exceeds_0.03" in wrong_parallel["candidate_smoke_gate"]["reasons"]

    assert candidate_smoke_gate(
        {
            "endpoint_preserved": True,
            "target_p95_distance_px": 1.0,
                "target_coverage_within_4px": 1.0,
                "reference_coverage_within_4px": 1.0,
                "numeric_label_gap_coverage_within_4px": 1.0,
                "parallel_switch_fraction": 0.0,
        }
    )["passed"] is True
