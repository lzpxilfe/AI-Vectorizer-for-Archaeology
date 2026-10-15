"""The local visual diagnostic must reject changed pixels and protected splits."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def comparison():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qgis_grid_comparison.py"
    spec = importlib.util.spec_from_file_location("grid_comparison_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cohort(tmp_path, split="validation"):
    image = tmp_path / "crop.png"
    image.write_bytes(b"frozen local pixels")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset_id": "visual-test", "samples": [
        {"id": "crop-1", "image": "crop.png", "image_sha256": digest, "split": split}]}))
    document = {"schema_version": "archaeotrace-visual-contour-cases/1", "dataset_id": "visual-test",
                "frozen_before_model_outputs": True, "human_approved": False,
                "cases": [{"id": "case-1", "sample_id": "crop-1", "image_sha256": digest,
                           "continuous_same_contour": True, "predictions_seen_before_freeze": False,
                           "review_level": "agent_visual_inspection", "reference_rc": [[10, 10], [20, 20]]}]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(document))
    return path, image


@pytest.mark.parametrize("split", ["calibration", "holdout", "train"])
def test_protected_split_rejected_before_image_is_opened(comparison, tmp_path, split):
    path, image = _cohort(tmp_path, split)
    image.unlink()
    with pytest.raises(ValueError, match="Only validation"):
        comparison.load_cases(path, tmp_path)


def test_changed_source_pixels_and_unverified_pair_are_rejected(comparison, tmp_path):
    path, image = _cohort(tmp_path)
    assert comparison.load_cases(path, tmp_path)[0]["image"] == str(image)
    image.write_bytes(b"modified")
    with pytest.raises(ValueError, match="image changed"):
        comparison.load_cases(path, tmp_path)
    path, _ = _cohort(tmp_path)
    doc = json.loads(path.read_text())
    doc["cases"][0]["continuous_same_contour"] = False
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="one contour"):
        comparison.load_cases(path, tmp_path)


@pytest.mark.parametrize("withdrawn", ["kr-10-crop-01-clear-w", "kr-10-crop-12-label-500-diagonal"])
def test_withdrawn_click_pair_cannot_reenter_accuracy_comparison(comparison, tmp_path, withdrawn):
    path, image = _cohort(tmp_path)
    doc = json.loads(path.read_text())
    doc["cases"][0]["id"] = withdrawn
    path.write_text(json.dumps(doc))
    assert comparison.load_cases(path, tmp_path, allow_withdrawn=True)
    image.unlink()
    with pytest.raises(ValueError, match="Withdrawn same-contour"):
        comparison.load_cases(path, tmp_path)


def _gap_cohort(tmp_path):
    path, image = _cohort(tmp_path)
    doc = json.loads(path.read_text())
    doc["schema_version"] = "archaeotrace-label-gap-cases/1"
    doc["cases"][0].update(
        prompt_rc=[[30, 10], [30, 50]], expected_pair="plausible_same_contour",
        hidden_geometry_ground_truth=False, unknown_regions_xy=[[20, 20, 40, 45]],
        visible_reference_segments_rc=[[[30, 10], [30, 20]], [[30, 40], [30, 50]]],
    )
    path.write_text(json.dumps(doc))
    return path, image


def test_gap_review_requires_unknown_interval_and_never_recovers_hidden_truth(comparison, tmp_path):
    path, image = _gap_cohort(tmp_path)
    assert comparison.load_cases(path, tmp_path)[0]["gap_diagnostic"]
    doc = json.loads(path.read_text())
    doc["cases"][0]["unknown_regions_xy"] = []
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Hidden gap regions"):
        comparison.load_cases(path, tmp_path)


def test_hidden_curve_shape_is_excluded_from_visible_accuracy(comparison, tmp_path):
    from PIL import Image

    path, image = _gap_cohort(tmp_path)
    Image.new("RGB", (64, 64), "white").save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["samples"][0]["image_sha256"] = digest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    document = json.loads(path.read_text())
    document["cases"][0]["image_sha256"] = digest
    path.write_text(json.dumps(document))
    cases = comparison.load_cases(path, tmp_path)
    straight = [(10.5, 30.5), (20.5, 30.5), (40.5, 30.5), (50.5, 30.5)]
    bowed = [(10.5, 30.5), (20.5, 30.5), (30.5, 40.5), (40.5, 30.5), (50.5, 30.5)]
    traces = {method + suffix: {"vertices_xy": [(x, 64 - y) for x, y in points]}
              for method in ("ink", "usgs", "korea")
              for suffix, points in (("_before", straight), ("_after", bowed))}
    comparison.score_and_render({"cases": [{"id": "case-1", "traces": traces}]}, cases, tmp_path)
    scores = json.loads((tmp_path / "scores.json").read_text())["cases"][0]
    assert scores["hidden_geometry_scored"] is False
    assert scores["scores"]["ink_before"]["visible_f1_2px"] == 1.0
    assert scores["scores"]["ink_after"]["visible_f1_2px"] == 1.0
    assert "f1_2px" not in scores["scores"]["ink_after"]
