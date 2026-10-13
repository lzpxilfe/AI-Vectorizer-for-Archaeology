"""Export one FP32 fixed-grid ONNX model with CPU parity and provenance."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

from .model import ARCHITECTURE, build_model
from .train import (canonical_sha256, code_identity, load_checkpoint,
                    runtime_environment, validate_checkpoint_label_policy)


PARITY_TOLERANCE = 1e-4


def validate_validation_selection(document, context):
    """Verify frozen three-seed ranking without opening held images or labels.

    context binds the selected candidate to independently supplied checkpoint
    facts, or to the exported provenance carrying those verified facts.
    """
    if not isinstance(document, dict) or document.get("schema_version") != "archaeotrace-validation-selection/1":
        raise ValueError("unsupported validation selection")
    if document.get("selection_uses") != "validation_only" or document.get("tie_break") != "ascending_seed":
        raise ValueError("selection must rank validation only with ascending-seed ties")
    if document.get("calibration_accessed") is not False or document.get("holdout_accessed") is not False:
        raise ValueError("selection must be frozen before calibration/holdout access")
    if document.get("publication_ranking_eligible") is not False:
        raise ValueError("single reviewer selection cannot claim publication ranking")
    weak = context.get("allow_weak_labels", False)
    if (type(document.get("allow_weak_labels", False)) is not bool
            or type(weak) is not bool or document.get("allow_weak_labels", False) != weak):
        raise ValueError("validation selection weak-label scope changed")
    if weak and (document.get("human_review_required") is not True
                 or document.get("review_level") != "automated_weak_labels"):
        raise ValueError("weak-label selection must preserve missing human review")
    if not weak and (document.get("human_review_required") is True
                     or document.get("review_level") == "automated_weak_labels"):
        raise ValueError("machine selection cannot claim human-reviewed scope")
    candidates = document.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("validation selection requires all three seeds")
    fields = {"seed", "validation_loss", "checkpoint", "checkpoint_sha256", "training_config_sha256", "training_source_sha256"}
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != fields:
            raise ValueError("candidate record is incomplete")
        loss = candidate["validation_loss"]
        if type(candidate["seed"]) is not int or type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError("candidate seed/loss must be a finite validation result")
        if not isinstance(candidate["checkpoint"], str) or not Path(candidate["checkpoint"]).is_absolute():
            raise ValueError("candidate checkpoints must have explicit absolute paths")
        for key in ("checkpoint_sha256", "training_config_sha256", "training_source_sha256"):
            if not isinstance(candidate[key], str) or not re.fullmatch(r"[0-9a-f]{64}", candidate[key]):
                raise ValueError("candidate commitment is malformed")
    if {value["seed"] for value in candidates} != {3407, 3408, 3409}:
        raise ValueError("validation selection seed cohort changed")
    selected = min(candidates, key=lambda value: (value["validation_loss"], value["seed"]))
    if document.get("selected") != selected:
        raise ValueError("selected candidate is not the validation-only winner")
    if document.get("dataset_manifest_sha256") != context["dataset_manifest_sha256"] or document.get("code_commit") != context["training_code_commit"]:
        raise ValueError("validation selection dataset/code identity changed")
    expected = {
        "seed": context["seed"], "validation_loss": context["best_validation_loss"],
        "checkpoint_sha256": context["training_checkpoint_sha256"],
        "training_config_sha256": context["training_config_sha256"],
        "training_source_sha256": context["training_source_sha256"],
    }
    if any(selected[key] != value for key, value in expected.items()):
        raise ValueError("selected checkpoint facts do not match the exported model")
    if any(candidate["training_source_sha256"] != context["training_source_sha256"] for candidate in candidates):
        raise ValueError("all candidate seeds must use the same frozen training source")
    return document


def _selection_context(checkpoint):
    return {
        "dataset_manifest_sha256": checkpoint["dataset_manifest_sha256"],
        "training_code_commit": checkpoint["code"]["commit"],
        "seed": checkpoint["config"]["seed"],
        "best_validation_loss": checkpoint["progress"]["best_validation"],
        "training_checkpoint_sha256": checkpoint["checkpoint_file_sha256"],
        "training_config_sha256": checkpoint["config_sha256"],
        "training_source_sha256": checkpoint["code"]["source_sha256"],
        "allow_weak_labels": checkpoint.get("allow_weak_labels", False),
    }


def _unique_selection_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in validation selection")
        result[key] = value
    return result


def _load_selection_file(path, selected_checkpoint):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("validation selection must be a regular local JSON file")
    snapshot = path.read_bytes()
    document = json.loads(snapshot, object_pairs_hook=_unique_selection_object)
    validate_validation_selection(document, _selection_context(selected_checkpoint))
    base_config = dict(selected_checkpoint["config"])
    for key in ("seed", "batch_size", "accumulation_steps"):
        base_config.pop(key)
    for candidate in document["candidates"]:
        checkpoint = load_checkpoint(candidate["checkpoint"])
        validate_checkpoint_label_policy(checkpoint, allow_weak_labels=selected_checkpoint.get("allow_weak_labels", False))
        facts = _selection_context(checkpoint)
        if checkpoint["synthetic_diagnostic"] or checkpoint["progress"]["epoch"] < 1:
            raise ValueError("candidate must be a real validation-selected completed-epoch checkpoint")
        if facts["dataset_manifest_sha256"] != document["dataset_manifest_sha256"] or facts["training_code_commit"] != document["code_commit"]:
            raise ValueError("candidate checkpoint dataset/code does not match selection")
        expected = {
            "seed": facts["seed"], "validation_loss": facts["best_validation_loss"],
            "checkpoint_sha256": facts["training_checkpoint_sha256"],
            "training_config_sha256": facts["training_config_sha256"],
            "training_source_sha256": facts["training_source_sha256"],
        }
        if any(candidate[key] != value for key, value in expected.items()):
            raise ValueError("candidate validation record does not match actual checkpoint bytes")
        candidate_config = dict(checkpoint["config"])
        pair = (candidate_config.pop("batch_size"), candidate_config.pop("accumulation_steps"))
        candidate_config.pop("seed")
        if candidate_config != base_config or pair not in ((2, 8), (1, 16)):
            raise ValueError("three candidate seeds must share one training policy and effective batch 16")
    return {
        "validation_selection": document,
        "validation_selection_sha256": canonical_sha256(document),
        "validation_selection_file_sha256": hashlib.sha256(snapshot).hexdigest(),
    }


def _parity_inputs():
    import numpy as np

    rng = np.random.default_rng(3407)
    random = rng.integers(0, 256, (1, 3, 512, 512), dtype=np.uint8)
    paper = np.full((1, 3, 512, 512), 245, dtype=np.uint8)
    paper[:, :, 200:204, 40:472] = np.asarray([156, 104, 60], dtype=np.uint8).reshape(1, 3, 1, 1)
    paper[:, :, 60:460, 325:328] = 24
    return [random.astype(np.float32) / 255.0, paper.astype(np.float32) / 255.0]


def export_checkpoint(checkpoint_path, output_directory, manifest_path, *,
                      license_spdx="GPL-2.0-only", synthetic_diagnostic=False,
                      model_id=None, selection_file=None, allow_weak_labels=False):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from importlib.metadata import version
    from ai_vectorizer.core.learned_model import build_bundle_manifest

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    validate_checkpoint_label_policy(checkpoint, allow_weak_labels=allow_weak_labels)
    if bool(checkpoint["synthetic_diagnostic"]) != bool(synthetic_diagnostic):
        raise ValueError("synthetic diagnostic checkpoints require an explicit diagnostic export")
    if not synthetic_diagnostic and selection_file is None:
        raise ValueError("pilot export requires the frozen three-seed --selection-file")
    selection_proof = _load_selection_file(selection_file, checkpoint) if selection_file is not None else {}
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("dataset manifest must be a regular local file")
    dataset_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if dataset_digest != checkpoint["dataset_manifest_sha256"]:
        raise ValueError("export dataset manifest differs from training")
    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("export output must be an absent directory")
    if not output.parent.is_dir():
        raise ValueError("export parent directory must already exist")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = build_model(initialize_pretrained=False).cpu().float()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    stage = Path(tempfile.mkdtemp(prefix=".learned-export-", dir=output.parent))
    published = False
    try:
        model_path = stage / "model.onnx"
        example = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
        with torch.no_grad():
            torch.onnx.export(
                model, (example,), str(model_path), input_names=["rgb"],
                output_names=["logits"], opset_version=18, dynamo=False,
                external_data=False, export_params=True, do_constant_folding=True,
            )
        graph = onnx.load(model_path, load_external_data=False)
        onnx.checker.check_model(graph, full_check=True)
        if graph.functions or any(node.domain not in ("", "ai.onnx") for node in graph.graph.node):
            raise ValueError("export must contain only standard ONNX operators")
        if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
            raise ValueError("export must contain all weights in one ONNX file")
        if any(tensor.data_type != onnx.TensorProto.FLOAT for tensor in graph.graph.initializer):
            raise ValueError("export weights must remain FP32")
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ValueError("export must expose one input and one dual-head output")
        input_spec, output_spec = session.get_inputs()[0], session.get_outputs()[0]
        if (input_spec.name, input_spec.shape, input_spec.type) != ("rgb", [1, 3, 512, 512], "tensor(float)"):
            raise ValueError("export input contract changed")
        if (output_spec.name, output_spec.shape, output_spec.type) != ("logits", [1, 2, 512, 512], "tensor(float)"):
            raise ValueError("export output contract changed")
        comparisons = []
        for rgb in _parity_inputs():
            with torch.no_grad():
                expected = torch.sigmoid(model(torch.from_numpy(rgb))).numpy()
            logits = session.run(["logits"], {"rgb": rgb})[0]
            if not np.isfinite(logits).all():
                raise ValueError("ONNX parity output contains nonfinite logits")
            actual = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
            difference = float(np.max(np.abs(expected - actual)))
            comparisons.append({"input_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "max_probability_abs_error": difference})
            if difference > PARITY_TOLERANCE:
                raise ValueError("CPU probability parity failed: {} > {}".format(difference, PARITY_TOLERANCE))
        identity = code_identity()
        provenance = {
            "dataset_manifest_sha256": dataset_digest,
            "training_code_commit": checkpoint["code"]["commit"],
            "export_code_commit": identity["commit"],
            "architecture": ARCHITECTURE, "seed": checkpoint["config"]["seed"],
            "license_spdx": license_spdx,
            "training_checkpoint_sha256": checkpoint["checkpoint_file_sha256"],
            "training_config_sha256": checkpoint["config_sha256"],
            "training_source_sha256": checkpoint["code"]["source_sha256"],
            "export_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_environment": checkpoint["environment"],
            "export_environment": {**runtime_environment("cpu"), "onnx": version("onnx"), "onnxruntime": version("onnxruntime")},
            "initialization": checkpoint["initialization"],
            "optimizer_steps": checkpoint["progress"]["optimizer_steps"],
            "epochs_completed": checkpoint["progress"]["epoch"],
            "best_validation_loss": checkpoint["progress"]["best_validation"],
            "review_level": checkpoint["review_level"],
            "allow_weak_labels": allow_weak_labels,
            "human_review_required": allow_weak_labels,
            "publication_ranking_eligible": False,
            "synthetic_diagnostic": bool(synthetic_diagnostic),
            "research_ready": False,
            "export_diagnostics": {"providers": ["CPUExecutionProvider"], "fp32": True, "probability_tolerance": PARITY_TOLERANCE, "comparisons": comparisons},
            **selection_proof,
        }
        model_bytes = model_path.read_bytes()
        bundle = build_bundle_manifest(model_bytes, provenance, model_id=model_id or (
            "synthetic-learned-guide-smoke" if synthetic_diagnostic else (
                "weak-label-contour-experiment-v1" if allow_weak_labels else "learned-contour-v1")
        ))
        (stage / "manifest.json").write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Reserve the destination exclusively; validated manifest is published
        # last, so a runtime never sees a usable partial model bundle.
        output.mkdir()
        published = True
        os.replace(model_path, output / "model.onnx")
        os.replace(stage / "manifest.json", output / "manifest.json")
        return bundle
    except BaseException:
        if published:
            shutil.rmtree(output)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output_directory")
    parser.add_argument("manifest")
    parser.add_argument("--license", default="GPL-2.0-only")
    parser.add_argument("--synthetic-diagnostic", action="store_true")
    parser.add_argument("--selection-file")
    parser.add_argument("--allow-weak-labels", action="store_true", help="export explicitly experimental weak-label weights")
    args = parser.parse_args(argv)
    print(json.dumps(export_checkpoint(args.checkpoint, args.output_directory, args.manifest,
        license_spdx=args.license, synthetic_diagnostic=args.synthetic_diagnostic,
        selection_file=args.selection_file, allow_weak_labels=args.allow_weak_labels), indent=2))


if __name__ == "__main__":
    main()
