"""Real tiny ONNX diagnostics, independent of trained model accuracy."""

import hashlib
import json
import math

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper

from ai_vectorizer.core.edge_detector import EdgeDetector
from ai_vectorizer.core.learned_model import (
    LearnedContourEngine, LearnedModelCancelled, LearnedModelError,
    build_bundle_manifest, import_local_bundle, inspect_bundle, plan_source_tiles,
    model_guidance_parameters,
)
from ai_vectorizer.core.livewire import LiveWireConfig, build_livewire_tree, is_livewire_available


def _model_bytes(*, raw_weights=None, raw_bias=None):
    # Diagnostic colour arithmetic with the required in-graph ImageNet
    # normalization; this graph is deliberately not a trained map model.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if raw_weights is None:
        raw_weights = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
    raw_weights = np.asarray(raw_weights, dtype=np.float32)
    raw_bias = np.zeros(2, dtype=np.float32) if raw_bias is None else np.asarray(raw_bias, dtype=np.float32)
    initializers = [
        numpy_helper.from_array(mean.reshape(1, 3, 1, 1), "mean"),
        numpy_helper.from_array(std.reshape(1, 3, 1, 1), "std"),
        numpy_helper.from_array((raw_weights * std).reshape(2, 3, 1, 1), "weights"),
        numpy_helper.from_array(raw_bias + raw_weights @ mean, "bias"),
    ]
    nodes = [
        helper.make_node("Sub", ["rgb", "mean"], ["centered"]),
        helper.make_node("Div", ["centered", "std"], ["normalized"]),
        helper.make_node("Conv", ["normalized", "weights", "bias"], ["logits"]),
    ]
    graph = helper.make_graph(nodes, "diagnostic-no-trained-accuracy", [helper.make_tensor_value_info("rgb", TensorProto.FLOAT, [1, 3, 512, 512])], [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2, 512, 512])], initializers)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 17)])
    return model.SerializeToString()


def _write_bundle(path, model_bytes=None, *, selected_guidance=None):
    path.mkdir(parents=True, exist_ok=True)
    model_bytes = _model_bytes() if model_bytes is None else model_bytes
    provenance = {
        "dataset_manifest_sha256": "a" * 64,
        "training_code_commit": "b" * 40,
        "export_code_commit": "c" * 40,
        "architecture": "diagnostic-color-conv-not-trained",
        "seed": 0,
        "license_spdx": "MIT",
        "diagnostic": True,
    }
    if selected_guidance is not None:
        provenance["selected_guidance"] = selected_guidance
    manifest = build_bundle_manifest(model_bytes, provenance, model_id="diagnostic-v1")
    (path / "model.onnx").write_bytes(model_bytes)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def engine(tmp_path):
    return LearnedContourEngine.from_bundle(_write_bundle(tmp_path / "source"))


def test_real_cpu_onnx_has_fixed_shapes_and_stable_sigmoid(engine):
    image = np.full((512, 512, 3), (255, 70, 0), dtype=np.uint8)
    prior = engine.predict_tile(image)
    np.testing.assert_allclose(prior.contour_score, 1.0 / (1.0 + math.exp(-1.0)), atol=1e-7)
    np.testing.assert_allclose(prior.avoidance_score, 0.5, atol=1e-7)
    assert not prior.contour_score.flags.writeable
    assert engine.metadata["provider"] == "CPUExecutionProvider"
    assert engine.metadata["intra_op_threads"] == 1
    assert len(engine.model_sha256) == 64
    with pytest.raises(LearnedModelError, match="Byte RGB"):
        engine.predict_tile(image.astype(np.float32))
    with pytest.raises(LearnedModelError, match="Byte RGB"):
        engine.predict_tile(image[:, :, 0])


def test_large_logits_are_stable_without_overflow(tmp_path):
    model = _model_bytes(raw_weights=np.zeros((2, 3)), raw_bias=[1000, -1000])
    runtime = LearnedContourEngine(_write_bundle(tmp_path / "extreme", model))
    with np.errstate(over="raise", invalid="raise"):
        prior = runtime.predict_tile(np.zeros((512, 512, 3), dtype=np.uint8))
    assert np.all(prior.contour_score == 1.0)
    assert np.all(prior.avoidance_score == 0.0)


def test_source_tiles_repeat_overlapping_pans_and_true_edge_padding(engine):
    y, x = np.mgrid[:810, :830]
    image = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    reads = []
    def read(bounds):
        reads.append(bounds)
        left, top, right, bottom = bounds
        return image[top:bottom, left:right]
    first = engine.predict_crop(read, (0, 0, 520, 510), (830, 810))
    second = engine.predict_crop(read, (310, 300, 830, 810), (830, 810))
    np.testing.assert_array_equal(first.contour_score[300:510, 310:520], second.contour_score[:210, :210])
    np.testing.assert_array_equal(first.avoidance_score[300:510, 310:520], second.avoidance_score[:210, :210])
    assert all(0 <= left < right <= 830 and 0 <= top < bottom <= 810 for left, top, right, bottom in reads)
    plan = plan_source_tiles((0, 0, 520, 510), (830, 810))
    assert plan[0].core_origin == (0, 0)
    assert plan[0].read_bounds == (0, 0, 448, 448)
    assert plan[0].padding == (64, 0, 64, 0)
    tiny = np.full((3, 5, 3), (255, 0, 0), dtype=np.uint8)
    prior = engine.predict_crop(lambda bounds: tiny, (0, 0, 5, 3), (5, 3))
    assert prior.shape == (3, 5)
    np.testing.assert_allclose(prior.contour_score, 1.0 / (1.0 + math.exp(-1.0)), atol=1e-7)


def test_cancelled_or_malformed_source_read_cannot_publish_a_crop(engine):
    reads = []
    with pytest.raises(LearnedModelCancelled):
        engine.predict_crop(lambda bounds: reads.append(bounds), (0, 0, 10, 10), (10, 10), cancel_check=lambda: True)
    assert not reads
    with pytest.raises(LearnedModelError, match="dimensions"):
        engine.predict_crop(lambda bounds: np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 10, 10), (800, 800))
    for bounds in ((0, 0, 1025, 5), (0.0, 0, 5, 5), (0, 0, 0, 1), (-1, 0, 5, 5)):
        with pytest.raises(LearnedModelError):
            plan_source_tiles(bounds, (2000, 2000))


def test_local_import_is_hash_bound_atomic_and_offline(tmp_path, monkeypatch):
    source = _write_bundle(tmp_path / "source")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network used"))
    assert inspect_bundle(source).ready
    cache = tmp_path / "cache"
    imported = import_local_bundle(source, cache)
    assert imported.parent == cache / "learned-contours"
    assert import_local_bundle(source, cache) == imported
    assert inspect_bundle(imported).model_sha256 == hashlib.sha256((source / "model.onnx").read_bytes()).hexdigest()
    assert not list(imported.parent.glob(".import-*"))
    original = (imported / "model.onnx").read_bytes()
    (source / "model.onnx").write_bytes(b"corrupt")
    assert not inspect_bundle(source).ready
    with pytest.raises(LearnedModelError, match="SHA-256"):
        import_local_bundle(source, cache)
    assert (imported / "model.onnx").read_bytes() == original


def test_unconfigured_import_uses_support_without_an_arbitrary_avoidance_cost(tmp_path):
    source = _write_bundle(tmp_path / "source")
    imported = import_local_bundle(source, tmp_path / "cache")
    inspection = inspect_bundle(imported)
    assert inspection.ready
    assert "selected_guidance" not in inspection.manifest["provenance"]
    assert model_guidance_parameters(None) == (0.8, 0.0)
    assert model_guidance_parameters(inspection.manifest) == (0.8, 0.0)
    runtime = LearnedContourEngine(imported)
    assert runtime.guidance_parameters == (0.8, 0.0)
    assert not runtime.has_selected_guidance
    # The policy does not alter either exported model head or bundle bytes.
    prior = runtime.predict_tile(np.zeros((512, 512, 3), dtype=np.uint8))
    assert np.all(prior.avoidance_score == 0.5)
    assert (source / "model.onnx").read_bytes() == (imported / "model.onnx").read_bytes()


@pytest.mark.parametrize("support,avoidance", [(0.4, 2), (0.8, 4), (0.4, 6)])
def test_report_bound_selected_guidance_preserves_its_explicit_costs(tmp_path, support, avoidance):
    selected = {
        "support": support, "avoidance": avoidance,
        "calibration_sha256": "a" * 64, "holdout_report_sha256": "b" * 64,
        "source_bundle_sha256": "c" * 64, "pilot_gates_passed": True,
    }
    source = _write_bundle(tmp_path / "configured", selected_guidance=selected)
    imported = import_local_bundle(source, tmp_path / "cache")
    runtime = LearnedContourEngine(imported)
    assert runtime.has_selected_guidance
    assert runtime.guidance_parameters == (support, avoidance)
    assert model_guidance_parameters(runtime.manifest) == (support, avoidance)
    assert runtime.manifest["provenance"]["selected_guidance"] == selected


def test_import_never_replaces_a_concurrent_empty_cache_directory(tmp_path, monkeypatch):
    import ai_vectorizer.core.learned_model as module
    source = _write_bundle(tmp_path / "source")
    original_publish = module._publish_directory_no_replace
    destinations = []
    def race(staging, destination, descriptor):
        destination.mkdir()
        destinations.append(destination)
        original_publish(staging, destination, descriptor)
    monkeypatch.setattr(module, "_publish_directory_no_replace", race)
    with pytest.raises((FileNotFoundError, LearnedModelError)):
        import_local_bundle(source, tmp_path / "cache")
    assert destinations[0].is_dir()
    assert not list(destinations[0].iterdir())
    assert not list(destinations[0].parent.glob(".import-*"))


def test_symlink_files_and_directory_components_are_refused(tmp_path):
    source = _write_bundle(tmp_path / "source")
    linked = tmp_path / "link"
    linked.symlink_to(source, target_is_directory=True)
    assert not inspect_bundle(linked).ready
    with pytest.raises((LearnedModelError, OSError)):
        import_local_bundle(linked, tmp_path / "cache")
    target = source / "real.onnx"
    (source / "model.onnx").rename(target)
    (source / "model.onnx").symlink_to(target)
    assert not inspect_bundle(source).ready
    safe = _write_bundle(tmp_path / "safe")
    cache_link = tmp_path / "cache-link"
    cache_link.symlink_to(tmp_path / "safe", target_is_directory=True)
    with pytest.raises((LearnedModelError, OSError)):
        import_local_bundle(safe, cache_link)


def test_duplicate_manifest_fields_and_modified_contract_are_refused(tmp_path):
    source = _write_bundle(tmp_path / "source")
    original = (source / "manifest.json").read_text()
    (source / "manifest.json").write_text(original[:-1] + ',"model_id":"other"}')
    assert "duplicate" in inspect_bundle(source).detail
    manifest = json.loads(original)
    manifest["contract"]["heads"] = ["contour_support", "corridor"]
    (source / "manifest.json").write_text(json.dumps(manifest))
    assert "contract" in inspect_bundle(source).detail


@pytest.mark.parametrize("change",["weight","hash","approval"])
def test_sealed_guidance_metadata_rejects_invalid_weights_or_report_binding(tmp_path,change):
    source=_write_bundle(tmp_path/"source")
    path=source/"manifest.json"
    manifest=json.loads(path.read_text())
    selected={"support":.4,"avoidance":2,"calibration_sha256":"a"*64,
              "holdout_report_sha256":"b"*64,"pilot_gates_passed":True}
    if change=="weight":selected["support"]=.6
    elif change=="hash":selected["holdout_report_sha256"]="not-a-hash"
    else:selected["pilot_gates_passed"]=False
    manifest["provenance"]["selected_guidance"]=selected
    path.write_text(json.dumps(manifest))
    assert not inspect_bundle(source).ready
    with pytest.raises(LearnedModelError, match="selected guidance"):
        model_guidance_parameters(manifest)


@pytest.mark.parametrize("unsafe", ["external", "custom", "shape"])
def test_unsafe_graphs_are_refused_before_session_or_cache(tmp_path, unsafe):
    model = onnx.load_model_from_string(_model_bytes())
    if unsafe == "external":
        tensor = model.graph.initializer[0]
        tensor.data_location = TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="outside.bin")
    elif unsafe == "custom":
        model.graph.node[0].domain = "custom.local"
        model.opset_import.add(domain="custom.local", version=1)
    else:
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 256
    source = _write_bundle(tmp_path / unsafe, model.SerializeToString())
    assert not inspect_bundle(source).ready
    cache = tmp_path / "cache"
    with pytest.raises(LearnedModelError):
        import_local_bundle(source, cache)
    assert not cache.exists()
    with pytest.raises(LearnedModelError):
        LearnedContourEngine(source)


def test_nonfinite_inference_is_rejected(tmp_path):
    model = _model_bytes(raw_weights=np.zeros((2, 3)), raw_bias=[np.nan, 0])
    runtime = LearnedContourEngine(_write_bundle(tmp_path / "nan", model))
    with pytest.raises(LearnedModelError, match="finite"):
        runtime.predict_tile(np.zeros((512, 512, 3), dtype=np.uint8))


@pytest.mark.skipif(not is_livewire_available(), reason="SciPy is not installed")
def test_explicit_diagnostic_costs_and_actual_onnx_prefer_contour_over_black_shortcut(tmp_path):
    model = _model_bytes(raw_weights=[[15, 0, -15], [-6, -6, -6]], raw_bias=[-2, 6])
    # The hand-built diagnostic depends on its avoidance head. Declare its
    # fixture-only selected costs instead of silently enabling arbitrary
    # avoidance for every imported, uncalibrated model.
    selected = {
        "support": 0.8, "avoidance": 6,
        "calibration_sha256": "a" * 64, "holdout_report_sha256": "b" * 64,
        "pilot_gates_passed": True,
    }
    runtime = LearnedContourEngine(_write_bundle(
        tmp_path / "color-diagnostic", model, selected_guidance=selected,
    ))
    height, width = 82, 145
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    desired = []
    for x in range(8, 137):
        y = int(round(52 - 27 * math.sin(math.pi * (x - 8) / 128)))
        desired.append((x, y))
        image[y, x] = (156, 104, 60)
    image[52, 8:137] = (24, 24, 24)
    image[18:60, 82] = (24, 24, 24)
    evidence = EdgeDetector.detect_ink_evidence(image)
    observed = evidence.centerline.copy()
    prior = runtime.predict_crop(lambda bounds: image, (0, 0, width, height), (width, height))
    edges = np.zeros(evidence.shape, dtype=np.uint8)
    support_weight, avoidance_weight = runtime.guidance_parameters
    config = LiveWireConfig(
        max_window_size=160, target_snap_radius=0,
        learned_support_cap=support_weight,
        learned_avoidance_cost_weight=avoidance_weight,
    )
    baseline = build_livewire_tree(image, edges, desired[0], evidence=evidence, config=config)
    guided = build_livewire_tree(image, edges, desired[0], evidence=evidence, learned_prior=prior, config=config)
    reference = np.asarray(desired)
    def worst_error(tree):
        path = np.asarray(tree.trace(desired[-1]))
        return np.sqrt(((path[:, None] - reference[None]) ** 2).sum(axis=2)).min(axis=1).max()
    assert worst_error(baseline) > 20
    assert worst_error(guided) <= 2
    np.testing.assert_array_equal(evidence.centerline, observed)
    assert np.isfinite(guided.distances).all()
