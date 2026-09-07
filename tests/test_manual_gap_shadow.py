"""Integration evidence for explicit manual recovery of a neutral label gap."""

import json

from benchmarks.manual_gap_shadow import main, run_neutral_manual_gap_shadow


def test_neutral_manual_gap_shadow_is_explicit_and_beats_the_ink_control():
    result = run_neutral_manual_gap_shadow()

    assert result["historical_map_evidence"] is False
    assert result["publication_ranking_eligible"] is False
    control = result["ink_livewire_v2_neutral_control"]
    manual = result["manual_gap_bridge_shadow_v1"]
    assert control["candidate_smoke_gate"]["passed"] is False
    assert control["parallel_switch_fraction"] > 0.0
    assert manual["requires_explicit_user_anchors"] is True
    assert manual["automatic_endpoint_selection"] is False
    assert manual["model_or_ocr_used"] is False
    assert manual["score"]["endpoint_preserved"] is True
    assert manual["score"]["numeric_label_gap_coverage_within_4px"] >= 0.85
    assert manual["score"]["parallel_switch_fraction"] == 0.0
    assert manual["score"]["candidate_smoke_gate"]["passed"] is True


def test_neutral_manual_gap_shadow_cli_writes_inspectable_artifacts(tmp_path, capsys):
    output = tmp_path / "shadow"
    assert main(["--output-dir", str(output)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    persisted = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert emitted == persisted
    assert (output / "neutral-manual-gap-challenge.ppm").read_bytes().startswith(b"P6\n")
