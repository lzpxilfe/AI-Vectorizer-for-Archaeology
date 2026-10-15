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


@pytest.mark.parametrize("withdrawn", ["kr-10-crop-01-clear-w", "kr-10-crop-12-label-500-diagonal", "kr-10-crop-12-label-600-diagonal"])
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


def test_post_output_user_correction_can_only_be_an_unscored_negative(comparison, tmp_path):
    path, _ = _gap_cohort(tmp_path)
    document = json.loads(path.read_text())
    document.update(frozen_before_model_outputs=False,
                    classification_revision_after_outputs=True,
                    unchanged_raw_controls_frozen_before_outputs=True)
    case = document["cases"][0]
    case.update(predictions_seen_before_freeze=True,
                classification_changed_after_outputs=True,
                expected_pair="different_contours",
                negative_review_provenance={"source": "user_reported_neighbor_hop_after_viewing_candidate"})
    path.write_text(json.dumps(document))
    assert comparison.load_cases(path, tmp_path)[0]["expected_pair"] == "different_contours"
    case["expected_pair"] = "plausible_same_contour"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="one contour"):
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


@pytest.mark.parametrize("viewport", [(1024, 1024), (1827, 1295), (1828, 1296)])
def test_native_view_keeps_all_frozen_pixel_centres_exact_integer_events(comparison, viewport):
    left, bottom, right, top = comparison.native_pixel_extent(viewport, (512, 512))
    assert right - left == viewport[0] / 2
    assert top - bottom == viewport[1] / 2
    # Include the neighbouring fractional-anchor boundary which previously
    # changed the 600 U route after QGIS rounded a fitted-raster mouse event.
    for col, row in [(0, 0), (511, 511), (225, 326), (252, 324), (153, 261)]:
        screen_x, screen_y = (col + 0.5 - left) * 2, (top - (511.5 - row)) * 2
        assert screen_x == round(screen_x)
        assert screen_y == round(screen_y)
        assert 0 <= screen_x < viewport[0]
        assert 0 <= screen_y < viewport[1]
        assert round(screen_x) / 2 + left == col + 0.5
        assert top - round(screen_y) / 2 == 511.5 - row


@pytest.mark.parametrize("viewport", [(1023, 1024), (1024, 1023), (0, 1200), (1200.0, 1200), (True, 1200)])
def test_native_view_refuses_cropped_or_invalid_canvas(comparison, viewport):
    with pytest.raises(ValueError):
        comparison.native_pixel_extent(viewport, (512, 512))


def _archive_comparison(phase):
    return {
        "ok": True, "fixtures_sha256": "same-frozen-pixels-and-prompts",
        "qgis_version": "same-real-runtime", "models": {"usgs": "same-model-sha"},
        "archive": phase + ".zip", "archive_sha256": phase + "-sha", "qgis_exit_code": 0,
        "after_config": {"automatic_label_occlusion": True},
        "cases": [{"id": "case-1", "traces": {
            method + "_" + phase: {
                "input_map_xy": {"start": [225.5, 185.5], "end": [252.5, 187.5]},
                "vertices_xy": [[225.5, 185.5], [252.5, 187.5]],
            } for method in ("ink", "usgs", "korea")}}],
    }


def test_product_zip_comparison_retains_baseline_and_binds_matching_mouse_events(comparison):
    before, after = _archive_comparison("before"), _archive_comparison("after")
    merged = comparison.merge_archive_comparisons(before, after)
    assert merged["baseline_archive_sha256"] == "before-sha"
    assert merged["archive_sha256"] == "after-sha"
    assert merged["before_config"]["automatic_label_occlusion"] is True
    assert len(merged["cases"][0]["traces"]) == 6


@pytest.mark.parametrize("gesture", ["start", "end"])
def test_product_zip_comparison_rejects_subpixel_anchor_changes(comparison, gesture):
    before, after = _archive_comparison("before"), _archive_comparison("after")
    after["cases"][0]["traces"]["korea_after"]["input_map_xy"][gesture][0] += 0.143
    with pytest.raises(ValueError, match="real mouse input"):
        comparison.merge_archive_comparisons(before, after)


@pytest.mark.parametrize("key", ["fixtures_sha256", "qgis_version", "models"])
def test_product_zip_comparison_rejects_context_changes(comparison, key):
    before, after = _archive_comparison("before"), _archive_comparison("after")
    after[key] = "changed"
    with pytest.raises(ValueError, match="same QGIS|identical"):
        comparison.merge_archive_comparisons(before, after)


def test_product_zip_comparison_rejects_missing_actual_mouse_input(comparison):
    before, after = _archive_comparison("before"), _archive_comparison("after")
    del before["cases"][0]["traces"]["ink_before"]["input_map_xy"]
    with pytest.raises(ValueError, match="recorded real mouse inputs"):
        comparison.merge_archive_comparisons(before, after)
