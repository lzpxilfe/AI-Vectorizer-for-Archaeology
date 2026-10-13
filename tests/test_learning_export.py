"""Optional integration test of trained synthetic weights and actual ONNX runtime."""

import hashlib
import json
import copy
from dataclasses import asdict

import pytest

pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
import numpy as np

from ai_vectorizer.core.learned_model import LearnedContourEngine, inspect_bundle
from learning.export import _load_selection_file, export_checkpoint, validate_validation_selection
from learning.model import ARCHITECTURE
from learning.train import CHECKPOINT_SCHEMA, TrainConfig, atomic_checkpoint, canonical_sha256, load_checkpoint, train_run
from tests.test_learning_training import SyntheticDataset, synthetic_manifest


@pytest.fixture(scope="module")
def trained_synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("learned-export")
    manifest = synthetic_manifest(root / "manifest.json")
    training, validation = SyntheticDataset(manifest, "train"), SyntheticDataset(manifest, "validation")
    result = train_run(manifest, root / "training", config=TrainConfig(max_epochs=1, batch_size=1,
        accumulation_steps=1, amp=False), device="cpu", synthetic_diagnostic=True,
        training_dataset=training, validation_dataset=validation)
    return root, manifest, result


def test_actual_fp32_onnx_export_has_verified_parity_and_runs_product_adapter(trained_synthetic):
    root, manifest, result = trained_synthetic
    output = root / "export"
    bundle = export_checkpoint(result["checkpoint"], output, manifest,
        license_spdx="CC0-1.0", synthetic_diagnostic=True)
    model = (output / "model.onnx").read_bytes()
    assert 0 < len(model) <= 64 * 1024 * 1024
    assert hashlib.sha256(model).hexdigest() == bundle["model"]["sha256"]
    assert json.loads((output / "manifest.json").read_text()) == bundle
    assert bundle["contract"]["heads"] == ["contour_support", "other_ink_risk"]
    assert bundle["contract"]["normalization"] == "imagenet_inside_graph"
    assert bundle["provenance"]["synthetic_diagnostic"] is True
    assert bundle["provenance"]["research_ready"] is False
    assert bundle["provenance"]["publication_ranking_eligible"] is False
    assert all(item["max_probability_abs_error"] <= 1e-4 for item in bundle["provenance"]["export_diagnostics"]["comparisons"])
    assert inspect_bundle(output).ready
    engine = LearnedContourEngine(output)
    prior = engine.predict_tile(np.full((512, 512, 3), 245, dtype=np.uint8))
    assert prior.contour_score.shape == (512, 512)
    assert prior.avoidance_score.shape == (512, 512)
    assert np.isfinite(prior.contour_score).all()
    assert not prior.contour_score.flags.writeable
    assert np.all((prior.contour_score >= 0) & (prior.contour_score <= 1))


def test_export_rejects_wrong_manifest_or_unmarked_synthetic_checkpoint(trained_synthetic):
    root, manifest, result = trained_synthetic
    with pytest.raises(ValueError, match="diagnostic"):
        export_checkpoint(result["checkpoint"], root / "unmarked", manifest)
    wrong = root / "wrong-manifest.json"
    wrong.write_text("{}\n")
    with pytest.raises(ValueError, match="differs"):
        export_checkpoint(result["checkpoint"], root / "wrong", wrong, synthetic_diagnostic=True)
    assert not (root / "wrong").exists()


def selection_fixture(root):
    """Small serialized metadata fixtures; no claim of domain training."""
    candidates, records = [], {}
    for seed, loss in ((3407, 0.4), (3408, 0.2), (3409, 0.3)):
        config = asdict(TrainConfig(seed=seed))
        payload = {
            "schema_version": CHECKPOINT_SCHEMA, "architecture": ARCHITECTURE,
            "gradient_accumulation_pending": 0,
            "config": config, "config_sha256": canonical_sha256(config),
            "dataset_manifest_sha256": "a" * 64,
            "code": {"commit": "b" * 40, "source_sha256": "c" * 64},
            "progress": {"epoch": 1, "best_validation": loss},
            "synthetic_diagnostic": False,
        }
        path = root / (str(seed) + ".pt")
        atomic_checkpoint(path, payload)
        record = load_checkpoint(path)
        records[seed] = record
        candidates.append({"seed": seed, "validation_loss": loss, "checkpoint": str(path),
            "checkpoint_sha256": record["checkpoint_file_sha256"],
            "training_config_sha256": record["config_sha256"],
            "training_source_sha256": record["code"]["source_sha256"]})
    document = {"schema_version": "archaeotrace-validation-selection/1",
        "selection_uses": "validation_only", "tie_break": "ascending_seed",
        "dataset_manifest_sha256": "a" * 64, "code_commit": "b" * 40,
        "selected": candidates[1], "candidates": candidates,
        "calibration_accessed": False, "holdout_accessed": False,
        "publication_ranking_eligible": False}
    return document, records[3408]


def test_weak_validation_selection_binds_scope_across_all_three_checkpoints(tmp_path):
    document, _selected = selection_fixture(tmp_path)
    for candidate in document["candidates"]:
        checkpoint = load_checkpoint(candidate["checkpoint"])
        checkpoint.update(allow_weak_labels=True, human_review_required=True,
                          review_level="automated_weak_labels")
        atomic_checkpoint(candidate["checkpoint"], checkpoint)
        candidate["checkpoint_sha256"] = load_checkpoint(candidate["checkpoint"])["checkpoint_file_sha256"]
    document.update(allow_weak_labels=True, human_review_required=True, review_level="automated_weak_labels")
    selected = load_checkpoint(document["selected"]["checkpoint"])
    path = tmp_path / "weak-selection.json"
    path.write_text(json.dumps(document))
    assert _load_selection_file(path, selected)["validation_selection"]["human_review_required"] is True
    document.pop("human_review_required")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="missing human review"):
        _load_selection_file(path, selected)
    document["human_review_required"] = True
    path.write_text(json.dumps(document))
    candidate = document["candidates"][0]
    changed = load_checkpoint(candidate["checkpoint"])
    changed.update(allow_weak_labels=False, human_review_required=False, review_level="single_reviewer_pilot")
    atomic_checkpoint(candidate["checkpoint"], changed)
    candidate["checkpoint_sha256"] = load_checkpoint(candidate["checkpoint"])["checkpoint_file_sha256"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="weak-label mode"):
        _load_selection_file(path, selected)


def test_weak_export_requires_matching_explicit_opt_in_before_model_execution(trained_synthetic, monkeypatch):
    root, manifest, result = trained_synthetic
    checkpoint = load_checkpoint(result["checkpoint"])
    checkpoint.update(allow_weak_labels=True, human_review_required=True,
                      review_level="automated_weak_labels", synthetic_diagnostic=False)
    weak_path = root / "weak-scope-fixture.pt"
    atomic_checkpoint(weak_path, checkpoint)
    with pytest.raises(ValueError, match="weak-label mode"):
        export_checkpoint(weak_path, root / "weak-without-opt-in", manifest)
    with pytest.raises(ValueError, match="selection-file"):
        export_checkpoint(weak_path, root / "weak-without-ranking", manifest, allow_weak_labels=True)


def test_actual_onnx_export_preserves_weak_scope_and_requires_human_review(trained_synthetic, monkeypatch):
    """Exercise actual ONNX bytes/parity; mock ranking separately tested above."""
    from learning import export as exporter
    root, manifest, result = trained_synthetic
    checkpoint = load_checkpoint(result["checkpoint"])
    checkpoint.update(allow_weak_labels=True, human_review_required=True,
                      review_level="automated_weak_labels", synthetic_diagnostic=False)
    weak_path = root / "weak-onnx-provenance-fixture.pt"
    atomic_checkpoint(weak_path, checkpoint)
    monkeypatch.setattr(exporter, "_load_selection_file", lambda *_args: {})
    output = root / "weak-onnx-provenance-fixture"
    bundle = export_checkpoint(weak_path, output, manifest, allow_weak_labels=True,
                               selection_file=root / "mock-ranking.json")
    assert bundle["model_id"] == "weak-label-contour-experiment-v1"
    assert bundle["provenance"]["allow_weak_labels"] is True
    assert bundle["provenance"]["human_review_required"] is True
    assert bundle["provenance"]["review_level"] == "automated_weak_labels"
    assert bundle["provenance"]["research_ready"] is False
    assert bundle["provenance"]["publication_ranking_eligible"] is False
    assert bundle["provenance"]["synthetic_diagnostic"] is False
    assert inspect_bundle(output).ready


def test_three_seed_selection_binds_actual_checkpoint_bytes_and_nonwinner_losses(tmp_path):
    document, selected = selection_fixture(tmp_path)
    path = tmp_path / "selected-model.json"
    path.write_text(json.dumps(document))
    proof = _load_selection_file(path, selected)
    assert proof["validation_selection"] == document
    assert proof["validation_selection_sha256"] == canonical_sha256(document)
    assert proof["validation_selection_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # A changed nonwinning loss still leaves the same apparent winner; simply
    # rehashing the JSON must not make that fabricated result acceptable.
    changed = copy.deepcopy(document)
    changed["candidates"][2]["validation_loss"] = 0.35
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="actual checkpoint bytes"):
        _load_selection_file(path, selected)
    path.write_text(json.dumps(document))
    altered_path = document["candidates"][0]["checkpoint"]
    altered = load_checkpoint(altered_path)
    altered["progress"]["best_validation"] = 0.41
    atomic_checkpoint(altered_path, altered)
    with pytest.raises(ValueError, match="actual checkpoint bytes"):
        _load_selection_file(path, selected)


def test_selection_rejects_wrong_winner_cohort_dataset_and_protected_access(tmp_path):
    document, selected = selection_fixture(tmp_path)
    context = {"dataset_manifest_sha256": selected["dataset_manifest_sha256"],
        "training_code_commit": selected["code"]["commit"], "seed": 3408,
        "best_validation_loss": 0.2, "training_checkpoint_sha256": selected["checkpoint_file_sha256"],
        "training_config_sha256": selected["config_sha256"], "training_source_sha256": "c" * 64}
    assert validate_validation_selection(document, context) == document
    for modify in (
        lambda value: value.update(selected=value["candidates"][0]),
        lambda value: value["candidates"][0].update(seed=3410),
        lambda value: value.update(dataset_manifest_sha256="d" * 64),
        lambda value: value.update(holdout_accessed=True),
        lambda value: value["candidates"][0].update(validation_loss=float("nan")),
    ):
        changed = copy.deepcopy(document)
        modify(changed)
        with pytest.raises(ValueError):
            validate_validation_selection(changed, context)

