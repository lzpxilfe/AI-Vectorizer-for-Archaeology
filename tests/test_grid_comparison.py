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
