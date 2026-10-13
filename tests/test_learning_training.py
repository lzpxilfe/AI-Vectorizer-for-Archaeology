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
from learning.train import TrainConfig, load_checkpoint, train_run


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

