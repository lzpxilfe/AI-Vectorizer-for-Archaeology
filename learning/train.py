"""Reproducible local training with checkpoints only at optimizer boundaries."""

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

from .augmentation import augment_sample, sample_rng
from .model import ARCHITECTURE, build_model


CHECKPOINT_SCHEMA = "archaeotrace-learning-checkpoint/1"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 3407
    learning_rate: float = 1e-4
    max_epochs: int = 60
    patience: int = 8
    batch_size: int = 2
    accumulation_steps: int = 8
    checkpoint_seconds: float = 300.0
    topology_weight: float = 0.2
    amp: bool = True
    augment: bool = True

    def validate(self):
        import math

        for name in ("seed", "max_epochs", "patience", "batch_size", "accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(name + " must be an integer")
        if self.seed < 0 or not 1 <= self.max_epochs <= 60 or self.patience < 1:
            raise ValueError("seed/epoch/patience range is invalid")
        if self.batch_size < 1 or self.accumulation_steps < 1:
            raise ValueError("batch and accumulation must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.checkpoint_seconds) or self.checkpoint_seconds < 0:
            raise ValueError("checkpoint_seconds must be finite and nonnegative")
        if not math.isfinite(self.topology_weight) or self.topology_weight < 0:
            raise ValueError("topology_weight must be finite and nonnegative")
        return self


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_identity():
    """Bind both Git base and actual local training bytes, including dirty code."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / "learning" / name for name in (
        "model.py", "losses.py", "augmentation.py", "train.py", "dataset.py", "annotations.py",
    )] + [root / "requirements-learning.txt"]
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", *files], cwd=root, text=True).strip())
    return {"commit": commit, "source_sha256": canonical_sha256(files), "source_files": files, "working_tree_dirty": dirty}


def runtime_environment(device):
    from importlib.metadata import version

    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": version("torch"), "torchvision": version("torchvision"),
        "segmentation_models_pytorch": version("segmentation-models-pytorch"),
        "numpy": version("numpy"), "Pillow": version("Pillow"), "device": str(device),
    }


def _seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state():
    import numpy as np
    import torch

    numpy = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), int(numpy[2]), int(numpy[3]), float(numpy[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("exact CUDA resume requires the recorded device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, payload):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("checkpoint destination cannot be a symlink")
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    import torch

    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("checkpoint must be a regular local file")
    snapshot = path.read_bytes()
    payload = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("unsupported training checkpoint")
    if payload.get("gradient_accumulation_pending") != 0:
        raise ValueError("resume requires a complete optimizer boundary")
    if canonical_sha256(payload["config"]) != payload.get("config_sha256"):
        raise ValueError("checkpoint configuration commitment does not match")
    payload["checkpoint_file_sha256"] = hashlib.sha256(snapshot).hexdigest()
    return payload


def validate_checkpoint_label_policy(checkpoint, *, allow_weak_labels=False):
    """An explicit experimental scope must survive resume and export unchanged."""
    if type(allow_weak_labels) is not bool:
        raise ValueError("allow_weak_labels must be an explicit boolean")
    recorded = checkpoint.get("allow_weak_labels", False)
    if type(recorded) is not bool or recorded != allow_weak_labels:
        raise ValueError("checkpoint weak-label mode changed; explicit opt-in must match")
    if recorded:
        if (checkpoint.get("human_review_required") is not True
                or checkpoint.get("review_level") != "automated_weak_labels"
                or checkpoint.get("synthetic_diagnostic") is not False):
            raise ValueError("weak-label checkpoint must preserve missing human review")
    elif checkpoint.get("human_review_required") is True or checkpoint.get("review_level") == "automated_weak_labels":
        raise ValueError("machine-label provenance cannot claim human-reviewed scope")


def _batch(dataset, indices, *, seed, epoch, augmentation, device):
    import numpy as np
    import torch

    samples = []
    for index in indices:
        sample = dataset[int(index)]
        if augmentation:
            sample = augment_sample(sample, sample_rng(seed, epoch, sample["sample_id"]))
        image = np.asarray(sample["image"])
        if image.dtype != np.uint8 or image.shape != (512, 512, 3):
            raise ValueError("dataset images must be native Byte RGB 512x512")
        for name in ("centerline", "other_ink", "valid_mask"):
            if np.asarray(sample[name]).shape != (512, 512):
                raise ValueError("dataset labels must share the 512x512 image grid")
        samples.append(sample)
    image = np.stack([np.moveaxis(value["image"], -1, 0) for value in samples]).astype(np.float32) / 255.0
    return {
        "image": torch.from_numpy(image).to(device),
        **{name: torch.from_numpy(np.stack([value[name] for value in samples])).to(device) for name in ("centerline", "other_ink", "valid_mask")},
    }


def _validation(model, dataset, config, device):
    import torch
    from .losses import contour_loss

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(dataset), config.batch_size):
            batch = _batch(dataset, range(start, min(start + config.batch_size, len(dataset))), seed=config.seed, epoch=0, augmentation=False, device=device)
            loss, _metrics = contour_loss(model(batch["image"]), batch["centerline"], batch["other_ink"], batch["valid_mask"], topology_weight=config.topology_weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("validation loss is not finite")
            total += float(loss)
            count += 1
    model.train()
    return total / count


def train_run(manifest_path, run_dir, *, config=None, resume=None,
              initialize_pretrained=False, encoder_weights_path=None, device="cuda",
              retry_oom=False, max_optimizer_steps=None, synthetic_diagnostic=False,
              training_dataset=None, validation_dataset=None, allow_weak_labels=False):
    """Train only validated train/validation selections, or explicit synthetic fixtures.

    Deliberate --retry-oom resumes a durable boundary at batch 1/accumulation
    16. An OOM never saves partial gradients/BatchNorm updates as a checkpoint.
    max_optimizer_steps stops at a boundary for functional diagnostics.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from .losses import contour_loss

    config = (config or TrainConfig()).validate()
    if type(allow_weak_labels) is not bool or (allow_weak_labels and synthetic_diagnostic):
        raise ValueError("weak-label opt-in must be boolean and cannot claim synthetic diagnostics")
    if retry_oom:
        if resume is None:
            raise ValueError("deliberate OOM retry needs an existing boundary checkpoint")
        config = replace(config, batch_size=1, accumulation_steps=16).validate()
    if (training_dataset is not None or validation_dataset is not None) and not synthetic_diagnostic:
        raise ValueError("unvalidated dataset injection is allowed only for synthetic diagnostics")
    if resume is None and not synthetic_diagnostic and not initialize_pretrained:
        raise ValueError("first real training requires explicit --initialize-pretrained ImageNet initialization")
    if training_dataset is None and validation_dataset is None:
        from .dataset import ContourDataset

        training_dataset = ContourDataset(manifest_path, "train", allow_weak_labels=allow_weak_labels)
        validation_dataset = ContourDataset(manifest_path, "validation", allow_weak_labels=allow_weak_labels)
    if training_dataset is None or validation_dataset is None or not len(training_dataset) or not len(validation_dataset):
        raise ValueError("training and validation each need approved labelled samples")
    if training_dataset.manifest_sha256 != validation_dataset.manifest_sha256:
        raise ValueError("train and validation must use one committed manifest")
    if set(training_dataset.sample_ids) & set(validation_dataset.sample_ids):
        raise ValueError("train and validation sample identities overlap")
    if not synthetic_diagnostic:
        # Inspect only declared global metadata here. Protected calibration/
        # holdout label bytes remain unopened until their evaluation stage.
        declared = training_dataset.manifest.payload["samples"]
        if allow_weak_labels:
            if len(declared) != 144 or any(sample.get("annotation", {}).get("status") != "machine_labeled" for sample in declared):
                raise ValueError("freeze all weak-label revisions before experimental training")
        elif any(sample.get("annotation", {}).get("status") != "approved"
               or (sample["split"] in ("calibration", "holdout") and (
                   sample.get("difficulty") not in ("standard", "hard")
                   or not sample.get("prompt") or not sample.get("reference_sha256")
               )) for sample in declared):
            raise ValueError("freeze all annotations and protected evaluation targets/difficulty before pilot training")
    data_digest = training_dataset.manifest_sha256
    selection = {"train": list(training_dataset.sample_ids), "validation": list(validation_dataset.sample_ids)}
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("an existing CUDA GPU runtime is required; select CPU explicitly for diagnostics")
    if target_device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    _seed_everything(config.seed)
    restored = load_checkpoint(resume) if resume is not None else None
    if restored is not None:
        validate_checkpoint_label_policy(restored, allow_weak_labels=allow_weak_labels)
    if restored is not None and initialize_pretrained:
        raise ValueError("resume loads checkpoint weights and cannot initialize pretrained weights")
    identity = code_identity()
    environment = runtime_environment(target_device)
    model = build_model(initialize_pretrained=initialize_pretrained, encoder_weights_path=encoder_weights_path).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    amp_enabled = bool(config.amp and target_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    progress = {
        "epoch": 0, "position": 0, "order": [], "optimizer_steps": 0,
        "epoch_loss_sum": 0.0, "epoch_batches": 0,
        "best_validation": None, "bad_epochs": 0, "history": [],
    }
    initialization = model.initialization_provenance
    if restored is not None:
        expected_config = dict(restored["config"])
        if retry_oom:
            expected_config.update(batch_size=1, accumulation_steps=16)
        if expected_config != asdict(config):
            raise ValueError("exact resume requires the recorded config; OOM retry changes only batch/accumulation")
        if restored["dataset_manifest_sha256"] != data_digest or restored["selection"] != selection:
            raise ValueError("resume dataset identity/selection changed")
        if restored["code"]["source_sha256"] != identity["source_sha256"]:
            raise ValueError("exact resume training source bytes changed")
        for key in ("python", "torch", "torchvision", "segmentation_models_pytorch", "numpy", "Pillow", "device"):
            if restored["environment"][key] != environment[key]:
                raise ValueError("exact resume runtime changed: " + key)
        if restored["synthetic_diagnostic"] != bool(synthetic_diagnostic):
            raise ValueError("checkpoint diagnostic scope changed")
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scaler.load_state_dict(restored["scaler"])
        progress = restored["progress"]
        initialization = restored["initialization"]
        _restore_rng(restored["rng"])
    run_dir = Path(run_dir)
    if run_dir.is_symlink():
        raise ValueError("run directory cannot be a symlink")
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume is None and (run_dir / "latest.pt").exists():
        raise ValueError("run already contains a checkpoint; resume it explicitly")
    config_values = asdict(config)
    last_checkpoint = time.monotonic()

    def checkpoint(name="latest.pt"):
        payload = {
            "schema_version": CHECKPOINT_SCHEMA, "architecture": ARCHITECTURE,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "config": config_values, "config_sha256": canonical_sha256(config_values),
            "dataset_manifest_sha256": data_digest, "selection": selection,
            "code": identity, "environment": environment, "initialization": initialization,
            "progress": progress, "rng": _rng_state(), "gradient_accumulation_pending": 0,
            "synthetic_diagnostic": bool(synthetic_diagnostic),
            "review_level": "synthetic_diagnostic" if synthetic_diagnostic else (
                "automated_weak_labels" if allow_weak_labels else "single_reviewer_pilot"),
            "allow_weak_labels": allow_weak_labels,
            "human_review_required": allow_weak_labels,
            "publication_ranking_eligible": False,
        }
        atomic_checkpoint(run_dir / name, payload)

    # Initial boundary is recoverable even if the first GPU group runs OOM.
    checkpoint()
    print("Training started: seed={} device={} scope={} resumed={}".format(
        config.seed, target_device, "automated_weak_labels" if allow_weak_labels else (
            "synthetic_diagnostic" if synthetic_diagnostic else "single_reviewer_pilot"), resume is not None), flush=True)
    model.train()
    stopped_reason = "early_stopping" if progress["bad_epochs"] >= config.patience else "max_epochs"
    while progress["epoch"] < config.max_epochs and progress["bad_epochs"] < config.patience:
        epoch = progress["epoch"]
        if not progress["order"]:
            progress["order"] = np.random.default_rng(config.seed + epoch).permutation(len(training_dataset)).tolist()
        order = progress["order"]
        if sorted(order) != list(range(len(training_dataset))) or not 0 <= progress["position"] <= len(order):
            raise ValueError("checkpoint sampler position/permutation is invalid")
        while progress["position"] < len(order):
            group_start = progress["position"]
            group_end = min(len(order), group_start + config.batch_size * config.accumulation_steps)
            batches = [order[start:min(start + config.batch_size, group_end)] for start in range(group_start, group_end, config.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            group_loss, group_count = 0.0, 0
            try:
                for indices in batches:
                    batch = _batch(training_dataset, indices, seed=config.seed, epoch=epoch, augmentation=config.augment, device=target_device)
                    if not bool(batch["valid_mask"].any()):
                        continue
                    with torch.amp.autocast(device_type=target_device.type, enabled=amp_enabled):
                        loss, _metrics = contour_loss(model(batch["image"]), batch["centerline"], batch["other_ink"], batch["valid_mask"], topology_weight=config.topology_weight)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("training loss is not finite")
                    scaler.scale(loss / len(batches)).backward()
                    group_loss += float(loss.detach())
                    group_count += 1
                stepped = False
                if group_count:
                    if group_count != len(batches):
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(len(batches) / group_count)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    stepped = scaler.get_scale() >= scale_before
                optimizer.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError as exc:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "GPU OOM: latest.pt remains the last complete optimizer boundary. "
                    "Resume it explicitly with --retry-oom for batch 1/accumulation 16."
                ) from exc
            progress["position"] = group_end
            progress["optimizer_steps"] += int(stepped)
            progress["epoch_loss_sum"] += group_loss
            progress["epoch_batches"] += group_count
            if time.monotonic() - last_checkpoint >= config.checkpoint_seconds:
                checkpoint()
                last_checkpoint = time.monotonic()
                print("Checkpoint: seed={} epoch={} processed={}/{} optimizer_steps={}".format(
                    config.seed, epoch + 1, group_end, len(order), progress["optimizer_steps"]), flush=True)
            if max_optimizer_steps is not None and progress["optimizer_steps"] >= int(max_optimizer_steps):
                checkpoint()
                return {"run_dir": str(run_dir), "checkpoint": str(run_dir / "latest.pt"), "progress": progress, "stopped_reason": "optimizer_step_limit", "synthetic_diagnostic": bool(synthetic_diagnostic)}
        validation = _validation(model, validation_dataset, config, target_device)
        training = progress["epoch_loss_sum"] / max(1, progress["epoch_batches"])
        improved = progress["best_validation"] is None or validation < progress["best_validation"]
        progress["best_validation"] = validation if improved else progress["best_validation"]
        progress["bad_epochs"] = 0 if improved else progress["bad_epochs"] + 1
        progress["history"].append({"epoch": epoch + 1, "training_loss": training, "validation_loss": validation})
        progress.update(epoch=epoch + 1, position=0, order=[], epoch_loss_sum=0.0, epoch_batches=0)
        checkpoint()
        if improved:
            checkpoint("best.pt")
        print("Epoch {}/{} seed={}: training_loss={:.6f} validation_loss={:.6f} best={:.6f}".format(
            epoch + 1, config.max_epochs, config.seed, training, validation, progress["best_validation"]), flush=True)
        last_checkpoint = time.monotonic()
        if progress["bad_epochs"] >= config.patience:
            stopped_reason = "early_stopping"
            break
    result = {"run_dir": str(run_dir), "checkpoint": str(run_dir / "latest.pt"), "best_checkpoint": str(run_dir / "best.pt"), "progress": progress, "stopped_reason": stopped_reason, "synthetic_diagnostic": bool(synthetic_diagnostic)}
    (run_dir / "training-result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("run_dir")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--initialize-pretrained", action="store_true")
    parser.add_argument("--encoder-weights")
    parser.add_argument("--retry-oom", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--allow-weak-labels", action="store_true", help="explicit experimental training; human review remains required")
    args = parser.parse_args(argv)
    config = TrainConfig(seed=args.seed, max_epochs=args.max_epochs)
    if args.resume:
        config = TrainConfig(**load_checkpoint(args.resume)["config"])
    print(json.dumps(train_run(args.manifest, args.run_dir, config=config, resume=args.resume,
        initialize_pretrained=args.initialize_pretrained, encoder_weights_path=args.encoder_weights,
        device=args.device, retry_oom=args.retry_oom, max_optimizer_steps=args.max_optimizer_steps,
        allow_weak_labels=args.allow_weak_labels), indent=2))


if __name__ == "__main__":
    main()
