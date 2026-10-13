"""Optional real-ML tests: masked gradients and exact optimizer-boundary resume."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
import numpy as np

from learning.augmentation import augment_sample, sample_rng
from learning.losses import contour_loss
from learning.model import build_model
from learning.train import TrainConfig, load_checkpoint, train_run, validate_checkpoint_label_policy


class SyntheticDataset:
    """Explicit diagnostic data; this bypass is unavailable to real training."""

    def __init__(self, manifest_path, split, count=1):
        self.manifest_sha256 = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        self.sample_ids = ["synthetic-{}-{}".format(split, index) for index in range(count)]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        image = np.full((512, 512, 3), 245, dtype=np.uint8)
        row = 210 + int(index) * 20
        image[row:row + 3, 50:462] = (156, 104, 60)
        image[60:460, 330:333] = 24
        center = np.zeros((512, 512), dtype=np.float32)
        risk = np.zeros_like(center)
        center[row:row + 3, 50:462] = 1.0
        risk[60:460, 330:333] = 1.0
        valid = np.ones((512, 512), dtype=bool)
        valid[190:260, 320:345] = False
        return {"image": image, "centerline": center, "other_ink": risk, "valid_mask": valid, "sample_id": self.sample_ids[index]}


def synthetic_manifest(path):
    path.write_text(json.dumps({"purpose": "synthetic gradient/resume/export diagnostic", "research_ready": False}) + "\n")
    return path


def test_optional_module_imports_neither_ml_runtime_nor_network():
    command = "import sys; import learning, learning.model, learning.train, learning.export, learning.losses; assert 'torch' not in sys.modules; assert 'segmentation_models_pytorch' not in sys.modules; assert 'onnxruntime' not in sys.modules"
    subprocess.run([sys.executable, "-c", command], check=True, cwd=Path(__file__).resolve().parents[1])


def test_model_returns_fixed_full_resolution_logits_without_downloading(monkeypatch):
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *_a, **_k: pytest.fail("unexpected weight download"))
    model = build_model(initialize_pretrained=False).eval()
    torch.set_num_threads(2)
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 512, 512))
    assert logits.shape == (1, 2, 512, 512)
    assert logits.dtype == torch.float32
    assert model.initialization_provenance["pretrained"] is False
    with pytest.raises(ValueError, match="512"):
        model(torch.zeros(1, 3, 256, 512))


def test_ignore_pixels_have_zero_gradient_and_do_not_change_topology_loss():
    logits = torch.randn(1, 2, 32, 32, requires_grad=True)
    center = torch.zeros(1, 32, 32)
    center[:, 16, 2:30] = 1
    risk = torch.zeros_like(center)
    valid = torch.ones_like(center, dtype=torch.bool)
    valid[:, 12:20, 12:20] = False
    base, _parts = contour_loss(logits, center, risk, valid)
    changed = logits.detach().clone()
    changed[:, :, 12:20, 12:20] = 20
    alternate, _parts = contour_loss(changed, center, risk, valid)
    torch.testing.assert_close(base, alternate, rtol=0, atol=0)
    base.backward()
    assert torch.count_nonzero(logits.grad[:, :, 12:20, 12:20]) == 0
    assert torch.count_nonzero(logits.grad[:, :, :10]) > 0
    blank = torch.randn(1, 2, 32, 32, requires_grad=True)
    loss, _parts = contour_loss(blank, center, risk, torch.zeros_like(valid))
    loss.backward()
    assert loss.item() == 0
    assert torch.count_nonzero(blank.grad) == 0


def test_train_augmentation_is_reproducible_and_preserves_ignore_geometry(tmp_path):
    manifest = synthetic_manifest(tmp_path / "manifest.json")
    sample = SyntheticDataset(manifest, "train")[0]
    first = augment_sample(sample, sample_rng(3407, 2, sample["sample_id"]))
    second = augment_sample(sample, sample_rng(3407, 2, sample["sample_id"]))
    for key in ("image", "centerline", "other_ink", "valid_mask"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["image"].dtype == np.uint8
    assert first["valid_mask"].sum() == sample["valid_mask"].sum()
    assert first["centerline"].sum() == sample["centerline"].sum()
    assert first["other_ink"].sum() == sample["other_ink"].sum()
    assert not np.shares_memory(first["image"], sample["image"])


def test_real_gradient_and_resume_match_uninterrupted_training_exactly(tmp_path):
    manifest = synthetic_manifest(tmp_path / "manifest.json")
    training = SyntheticDataset(manifest, "train", count=3)
    validation = SyntheticDataset(manifest, "validation")
    config = TrainConfig(max_epochs=1, batch_size=1, accumulation_steps=2, amp=False, checkpoint_seconds=0)
    common = {"config": config, "device": "cpu", "synthetic_diagnostic": True,
              "training_dataset": training, "validation_dataset": validation}
    baseline = train_run(manifest, tmp_path / "baseline", **common)
    interrupted = train_run(manifest, tmp_path / "resumed", max_optimizer_steps=1, **common)
    boundary = load_checkpoint(interrupted["checkpoint"])
    assert boundary["progress"]["position"] == 2
    assert boundary["gradient_accumulation_pending"] == 0
    assert boundary["optimizer"]["state"]
    resumed = train_run(manifest, tmp_path / "resumed", resume=interrupted["checkpoint"], **common)
    expected, actual = load_checkpoint(baseline["checkpoint"]), load_checkpoint(resumed["checkpoint"])
    assert expected["progress"] == actual["progress"]
    assert actual["progress"]["optimizer_steps"] == 2
    for name in expected["model"]:
        torch.testing.assert_close(expected["model"][name], actual["model"][name], rtol=0, atol=0)
    for key, values in expected["optimizer"]["state"].items():
        for name in values:
            torch.testing.assert_close(values[name], actual["optimizer"]["state"][key][name], rtol=0, atol=0)
    torch.testing.assert_close(expected["rng"]["torch"], actual["rng"]["torch"], rtol=0, atol=0)
    assert expected["rng"]["python"] == actual["rng"]["python"]
    assert expected["rng"]["numpy"] == actual["rng"]["numpy"]
    assert actual["selection"] == {"train": training.sample_ids, "validation": validation.sample_ids}


def test_real_training_rejects_unvalidated_injection_and_implicit_weights(tmp_path):
    manifest = synthetic_manifest(tmp_path / "manifest.json")
    training = SyntheticDataset(manifest, "train")
    with pytest.raises(ValueError, match="unvalidated"):
        train_run(manifest, tmp_path / "run", device="cpu", training_dataset=training)
    with pytest.raises(ValueError, match="explicit"):
        train_run(manifest, tmp_path / "run", device="cpu")


def test_weak_checkpoint_requires_opt_in_and_cannot_change_review_scope():
    checkpoint = {"allow_weak_labels": True, "human_review_required": True,
                  "review_level": "automated_weak_labels", "synthetic_diagnostic": False}
    with pytest.raises(ValueError, match="weak-label mode"):
        validate_checkpoint_label_policy(checkpoint)
    validate_checkpoint_label_policy(checkpoint, allow_weak_labels=True)
    for field, value in (("human_review_required", False), ("review_level", "single_reviewer_pilot"),
                         ("synthetic_diagnostic", True)):
        changed = dict(checkpoint)
        changed[field] = value
        with pytest.raises(ValueError, match="missing human review"):
            validate_checkpoint_label_policy(changed, allow_weak_labels=True)
    with pytest.raises(ValueError, match="mode"):
        validate_checkpoint_label_policy({"allow_weak_labels": False}, allow_weak_labels=True)


def test_weak_training_cannot_be_marked_synthetic(tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        train_run(tmp_path / "manifest.json", tmp_path / "run", allow_weak_labels=True, synthetic_diagnostic=True)


def test_actual_weak_training_checkpoint_and_resume_keep_experimental_scope(tmp_path, monkeypatch):
    """Small optimizer fixture with a frozen 144-record metadata cohort.

    Protected records deliberately have no readable files. Model initialization
    is replaced with offline random weights; this is execution evidence only.
    """
    import copy
    from learning import train as training_module
    from tests.test_learning_dataset import weak_fixture_manifest, write
    manifest, data = weak_fixture_manifest(tmp_path)
    prototype = data["samples"][-1]
    for index in range(140):
        protected = copy.deepcopy(prototype)
        protected["id"] = "protected-metadata-only-%03d" % index
        data["samples"].append(protected)
    data["split_policy"]["sample_counts"]["holdout"] = 141
    write(manifest, data)
    for sample in data["samples"][:4]:
        if sample["split"] in ("calibration", "holdout"):
            (tmp_path / sample["image"]).unlink()
            (tmp_path / sample["annotation"]["path"]).unlink()
    original_builder = training_module.build_model
    monkeypatch.setattr(training_module, "build_model", lambda **_kwargs: original_builder(initialize_pretrained=False))
    config = TrainConfig(max_epochs=1, batch_size=2, accumulation_steps=8,
                         amp=False, augment=False, checkpoint_seconds=0)
    common = {"config": config, "device": "cpu", "allow_weak_labels": True}
    result = train_run(manifest, tmp_path / "run", initialize_pretrained=True,
                       max_optimizer_steps=1, **common)
    boundary = load_checkpoint(result["checkpoint"])
    assert boundary["allow_weak_labels"] is True
    assert boundary["human_review_required"] is True
    assert boundary["review_level"] == "automated_weak_labels"
    assert boundary["synthetic_diagnostic"] is False
    completed = train_run(manifest, tmp_path / "run", resume=result["checkpoint"], **common)
    checkpoint = load_checkpoint(completed["checkpoint"])
    assert checkpoint["progress"]["epoch"] == 1
    assert checkpoint["progress"]["optimizer_steps"] == 1
    assert checkpoint["human_review_required"] is True
    assert checkpoint["selection"] == {"train": ["sheet-0-crop"], "validation": ["sheet-1-crop"]}

