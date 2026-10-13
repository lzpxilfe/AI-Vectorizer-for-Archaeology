"""Evaluator integration diagnostics and frozen-config integrity checks.

Synthetic fixtures here test execution/contracts only and cannot pass or seal
the historical-map pilot gate.
"""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from learning import evaluate
from learning.annotations import annotation_revision
from learning.dataset import DatasetError, canonical_bytes, sha256_file
from test_learning_dataset import fixture_manifest, write


def _selection_provenance():
    """Validation-ranking metadata fixture, without actual model/checkpoints."""
    from learning.train import canonical_sha256
    candidates=[{"seed":seed,"validation_loss":loss,"checkpoint":"/protocol-fixture/seed-%d/best.pt"%seed,
                 "checkpoint_sha256":str(index+1)*64,"training_config_sha256":str(index+4)*64,
                 "training_source_sha256":"d"*64}
                for index,(seed,loss) in enumerate(((3407,.4),(3408,.3),(3409,.3)))]
    selected=candidates[1]
    selection={"schema_version":"archaeotrace-validation-selection/1",
               "selection_uses":"validation_only","tie_break":"ascending_seed",
               "dataset_manifest_sha256":"a"*64,"code_commit":"e"*40,
               "selected":selected,"candidates":candidates,
               "calibration_accessed":False,"holdout_accessed":False,"publication_ranking_eligible":False}
    return {"dataset_manifest_sha256":"a"*64,"training_code_commit":"e"*40,
            "seed":selected["seed"],"best_validation_loss":selected["validation_loss"],
            "training_checkpoint_sha256":selected["checkpoint_sha256"],
            "training_config_sha256":selected["training_config_sha256"],"training_source_sha256":"d"*64,
            "validation_selection":selection,"validation_selection_sha256":canonical_sha256(selection),
            "validation_selection_file_sha256":"f"*64}


def test_real_model_protocol_rechecks_frozen_validation_selection_without_training_deps():
    dataset=SimpleNamespace(manifest_sha256="a"*64,payload={"dataset_id":"protocol-unit-fixture"})
    provenance=_selection_provenance()
    evaluate._guard_model_protocol(dataset,{"provenance":provenance},False)
    provenance.pop("validation_selection")
    with pytest.raises(DatasetError,match="three-seed"):
        evaluate._guard_model_protocol(dataset,{"provenance":provenance},False)


@pytest.mark.parametrize("marker", [
    {"allow_weak_labels": True}, {"human_review_required": True},
    {"review_level": "automated_weak_labels"},
])
def test_weak_model_cannot_enter_pilot_evaluation_or_sealing(marker):
    dataset = SimpleNamespace(manifest_sha256="a" * 64, payload={"dataset_id": "protocol-unit-fixture"})
    provenance = _selection_provenance()
    provenance.update(marker)
    with pytest.raises(DatasetError, match="weak-label experiment"):
        evaluate._guard_model_protocol(dataset, {"provenance": provenance}, False)


@pytest.mark.parametrize("change",["content_hash","file_hash","calibration_access","holdout_access",
                                    "selection_scope","winner","cohort","seed","checkpoint",
                                    "config","source","validation_loss"])
def test_validation_selection_hash_scope_ranking_and_export_identity_tampering_are_refused(change):
    from learning.train import canonical_sha256
    provenance=_selection_provenance()
    selection=provenance["validation_selection"]
    if change=="content_hash":provenance["validation_selection_sha256"]="0"*64
    elif change=="file_hash":provenance["validation_selection_file_sha256"]="not-a-sha"
    elif change=="calibration_access":selection["calibration_accessed"]=True
    elif change=="holdout_access":selection["holdout_accessed"]=True
    elif change=="selection_scope":selection["selection_uses"]="holdout"
    elif change=="winner":selection["selected"]=selection["candidates"][2]
    elif change=="cohort":selection["candidates"][2]["seed"]=3410
    elif change=="seed":provenance["seed"]=3407
    elif change=="checkpoint":provenance["training_checkpoint_sha256"]="0"*64
    elif change=="config":provenance["training_config_sha256"]="0"*64
    elif change=="source":provenance["training_source_sha256"]="0"*64
    else:provenance["best_validation_loss"]=False
    if change!="content_hash":provenance["validation_selection_sha256"]=canonical_sha256(selection)
    with pytest.raises(DatasetError):evaluate._verify_validation_selection(provenance)


def _metrics(f1=.9, p95=3):
    return {"tolerance_metrics":[{"tolerance":2,"f1":f1}],
            "connectivity":{"summary":{"breaks":0}},
            "topology":{"unmatched_prediction_branch_zones":0},
            "distance":{"symmetric_p95":p95}}


def _report(configuration, *, methods=("ink", "learned"), split="calibration"):
    rows = []
    for method in methods:
        rows.append({"sample_id":"cal-1","method":method,"status":"ok","metrics":_metrics(p95=10-configuration["support"]*configuration["avoidance"] if method=="learned" else 10),
                     "unknown_prediction_pixels":0,"hard_case":True,"peak_rss_bytes":1000,
                     "prepared_tile_ms":[10,11,12],"synthetic_diagnostic":False})
    return {"schema_version":evaluate.SCHEMA,"split":split,"dataset_id":"synthetic-contract-fixture",
            "dataset_sha256":"a"*64,"model_sha256":"b"*64,"bundle_sha256":"c"*64,
            "code_files_sha256":{"fixture":"d"*64},"sample_ids":["cal-1"],"methods":list(methods),
            "configuration":configuration,"results":rows,"publication_ranking_eligible":False,
            "review_level":"single_reviewer_pilot"}


def test_summary_uses_existing_metrics_keys_and_rejects_failed_or_duplicate_rows():
    report = _report({"support":.8,"avoidance":6})
    assert evaluate._summary(report,"learned") == {"mean_f1_2px":.9,"breaks":0,"over_branches":0}
    broken = copy.deepcopy(report)
    broken["results"][1]["status"]="fallback"
    assert evaluate._summary(broken,"learned") is None
    broken = copy.deepcopy(report)
    broken["results"][1]["unknown_prediction_pixels"]=1
    assert evaluate._summary(broken,"learned") is None
    report["results"].append(copy.deepcopy(report["results"][1]))
    assert evaluate._summary(report,"learned") is None


def test_synthetic_complete_reports_never_pass_pilot_holdout_gate():
    report = _report({"support":.8,"avoidance":6}, methods=evaluate.METHODS, split="holdout")
    assessment = evaluate.assess_holdout(report, 1000)
    assert assessment["gates"]["all_methods_complete"]
    assert not assessment["gates"]["reviewed_real_holdout"]
    assert not assessment["passed"]
    report["dataset_id"]="usgs-contour-pilot-16x144-v1"
    report["results"][2]["synthetic_diagnostic"]=True
    assert not evaluate.assess_holdout(report,1000)["passed"]


def _calibration_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate,"_sources_identity",lambda:{"fixture":"d"*64})
    def run(dataset,bundle,split,output_dir,**kwargs):
        assert split=="calibration"
        report=_report({"support":kwargs["support"],"avoidance":kwargs["avoidance"]})
        evaluate._write(Path(output_dir)/"report.json",report)
        return report
    monkeypatch.setattr(evaluate,"run_split",run)
    root=tmp_path/"calibration"
    frozen=evaluate.calibrate("unused-data","unused-model",root)
    dataset=SimpleNamespace(manifest_sha256="a"*64,sample_ids=lambda split:("cal-1",))
    inspected=SimpleNamespace(model_sha256="b"*64,bundle_sha256="c"*64)
    return root,frozen,dataset,inspected


def test_six_candidate_freeze_recomputes_eligible_selection(tmp_path, monkeypatch):
    root,frozen,dataset,inspected=_calibration_fixture(tmp_path,monkeypatch)
    assert len(frozen["candidates"])==6
    config=evaluate._verify_calibration(frozen,root/"calibration.json",dataset,inspected)
    assert config=={"support":.8,"avoidance":6}
    with pytest.raises(DatasetError,match="already frozen"):
        evaluate.calibrate("unused-data","unused-model",root)


@pytest.mark.parametrize("change",["selected","eligibility","report_hash","duplicate","outside_grid","absolute_path"])
def test_calibration_selection_and_report_tampering_are_refused(tmp_path,monkeypatch,change):
    root,frozen,dataset,inspected=_calibration_fixture(tmp_path,monkeypatch)
    bad=copy.deepcopy(frozen)
    if change=="selected":bad["selected"]=copy.deepcopy(bad["candidates"][0])
    elif change=="eligibility":bad["candidates"][0]["eligible"]=False
    elif change=="report_hash":bad["candidates"][0]["report_sha256"]="0"*64
    elif change=="duplicate":bad["candidates"][1]=copy.deepcopy(bad["candidates"][0])
    elif change=="outside_grid":bad["candidates"][0]["configuration"]["support"]=.6
    else:bad["candidates"][0]["report_path"]=str(root/"s0.4-a2/report.json")
    with pytest.raises(DatasetError):
        evaluate._verify_calibration(bad,root/"calibration.json",dataset,inspected)


def test_timeout_records_failed_methods_and_removes_private_requests(tmp_path,monkeypatch):
    sample={"id":"cal-1"}
    dataset=SimpleNamespace(path=tmp_path/"dataset.json",manifest_sha256="a"*64,payload={"dataset_id":"reviewed-protocol-unit"},samples_for=lambda split:(sample,))
    inspected=SimpleNamespace(model_sha256="b"*64,bundle_sha256="c"*64)
    verified_splits=[]
    def load(path,**kwargs):
        verified_splits.append(kwargs["verify_splits"])
        return dataset
    monkeypatch.setattr(evaluate,"load_dataset_manifest",load)
    monkeypatch.setattr(evaluate,"_bundle",lambda *args,**kwargs:(inspected,{}))
    monkeypatch.setattr(evaluate,"_sample_annotation",lambda *args:None)
    monkeypatch.setattr(evaluate,"_sources_identity",lambda:{"fixture":"d"*64})
    original_run=evaluate.subprocess.run
    def timeout_worker(command,**kwargs):
        if "learning.evaluate" in command:
            raise subprocess.TimeoutExpired("diagnostic-worker",300)
        return original_run(command,**kwargs)
    monkeypatch.setattr(evaluate.subprocess,"run",timeout_worker)
    report=evaluate.run_split(dataset.path,tmp_path/"bundle","calibration",tmp_path/"run",methods=("ink","learned"))
    assert all(row["status"]=="failed" for row in report["results"])
    assert not list((tmp_path/"run").glob("*.request.json"))
    assert verified_splits==[{"calibration"},{"calibration"}]
    assert evaluate._summary(report,"ink") is None


def _diagnostic_dataset(tmp_path):
    path,payload=fixture_manifest(tmp_path)
    sample=payload["samples"][2]
    source=payload["sources"][2]
    raster=np.full((1024,1024,3),255,dtype=np.uint8)
    with Image.open(tmp_path/sample["image"]) as crop:raster[:512,:512]=np.asarray(crop)
    source_path=tmp_path/"native.tif"
    Image.fromarray(raster).save(source_path)
    source["source_file"]=source_path.name
    source["source_sha256"]=sha256_file(source_path)
    annotation_path=tmp_path/sample["annotation"]["path"]
    annotation=json.loads(annotation_path.read_text())
    annotation["evaluation"]={"target_contour_id":"contour","explicit_target_selection":True,
                              "prompt":{"start_xy":[2,8],"end_xy":[20,8],"previous_xy":[1,8]}}
    annotation["revision"]=annotation_revision(annotation)
    write(annotation_path,annotation)
    sample["annotation"].update(sha256=sha256_file(annotation_path),revision=annotation["revision"])
    write(path,payload)
    from test_learned_model import _write_bundle
    bundle=_write_bundle(tmp_path/"bundle")
    manifest=json.loads((bundle/"manifest.json").read_text())
    manifest["provenance"].update(dataset_manifest_sha256=sha256_file(path),synthetic_diagnostic=True)
    write(bundle/"manifest.json",manifest)
    return path,sample,bundle


@pytest.mark.filterwarnings("ignore:Dataset has no geotransform.*")
@pytest.mark.parametrize("method",["ink","learned"])
def test_real_512_source_worker_records_geometry_and_actual_cpu_metadata(tmp_path,method):
    pytest.importorskip("onnxruntime");pytest.importorskip("onnx")
    pytest.importorskip("rasterio");pytest.importorskip("cv2")
    path,sample,bundle=_diagnostic_dataset(tmp_path)
    request={"dataset":str(path),"bundle":str(bundle),"sample_id":sample["id"],"split":"calibration",
             "method":method,"support":.8,"avoidance":6,"repetitions":3,
             "recovery_cache":None,"synthetic_diagnostic":True}
    result=evaluate._worker(request)
    assert result["status"]=="ok" and result["synthetic_diagnostic"] is True
    assert result["source_context"]=="original_native_grid"
    assert result["ink_context_bounds"]==[0,0,543,543]
    assert len(result["prepared_path_ms"])==3
    assert result["peak_rss_bytes"]>0
    assert result["metrics"]["connectivity"]["summary"]["breaks"]==0
    assert result["request_sha256"]==hashlib.sha256(canonical_bytes(request)).hexdigest()
    if method=="learned":
        assert result["provider"]["provider"]=="CPUExecutionProvider"
        assert result["provider"]["intra_op_threads"]==1
        assert len(result["prepared_tile_ms"])>=3
    else:assert result["provider"]["actual_backend"]=="ink-livewire-v2"
    request.pop("synthetic_diagnostic")
    with pytest.raises(DatasetError,match="diagnostic"):
        evaluate._worker(request)


def test_atomic_reports_cannot_overwrite_a_published_result(tmp_path):
    target=tmp_path/"report.json"
    evaluate._write(target,{"fixture":1})
    with pytest.raises(FileExistsError):evaluate._write(target,{"fixture":2})
    assert json.loads(target.read_text())=={"fixture":1}


def test_synthetic_complete_report_cannot_seal_a_model(tmp_path):
    pytest.importorskip("onnx")
    from test_learned_model import _write_bundle
    bundle=_write_bundle(tmp_path/"diagnostic-model")
    report=_report({"support":.8,"avoidance":6},methods=evaluate.METHODS,split="holdout")
    report["assessment"]=evaluate.assess_holdout(report,1000)
    path=evaluate._write(tmp_path/"diagnostic-report.json",report)
    output=tmp_path/"sealed"
    with pytest.raises(DatasetError,match="passing real holdout"):
        evaluate.seal_bundle(bundle,path,output,calibration_path=tmp_path/"unused-calibration.json")
    assert not output.exists()


def test_direct_seal_requires_dataset_and_rechecks_calibration(tmp_path,monkeypatch):
    """Protocol stubs never create a model or a historical-map result."""
    # Resolve the same module that the function's lazy import will use. Other
    # optional-runtime tests temporarily restore sys.modules; a package's
    # cached attribute may then refer to an earlier module object.
    import importlib
    learned_model = importlib.import_module("ai_vectorizer.core.learned_model")
    inspected=SimpleNamespace(ready=True,model_sha256="b"*64,bundle_sha256="c"*64,
                              manifest={"model":{"size_bytes":1000},
                                        "provenance":{"dataset_manifest_sha256":"a"*64}})
    monkeypatch.setattr(learned_model,"inspect_bundle",lambda path:inspected)
    report=_report({"support":.8,"avoidance":6},methods=evaluate.METHODS,split="holdout")
    report["dataset_id"]="usgs-contour-pilot-16x144-v1"
    report["assessment"]=evaluate.assess_holdout(report,1000)
    assert report["assessment"]["passed"]  # Simulated gate only, no source data.
    calibration=evaluate._write(tmp_path/"calibration.json",{"selected":{"configuration":report["configuration"]}})
    report["calibration_sha256"]=sha256_file(calibration)
    path=evaluate._write(tmp_path/"stub-report.json",report)
    output=tmp_path/"never-sealed"
    with pytest.raises(DatasetError,match="immutable dataset"):
        evaluate.seal_bundle("stub-model",path,output,calibration_path=calibration)
    dataset=SimpleNamespace(sample_ids=lambda split:("cal-1",),manifest_sha256="a"*64)
    verified=[]
    def load(path,**kwargs):
        verified.append(kwargs["verify_splits"])
        return dataset
    monkeypatch.setattr(evaluate,"load_dataset_manifest",load)
    monkeypatch.setattr(evaluate,"_guard_model_protocol",lambda *args:None)
    monkeypatch.setattr(evaluate,"_sources_identity",lambda:{"fixture":"d"*64})
    def refuse(*args):raise DatasetError("six-candidate calibration incomplete")
    monkeypatch.setattr(evaluate,"_verify_calibration",refuse)
    with pytest.raises(DatasetError,match="six-candidate"):
        evaluate.seal_bundle("stub-model",path,output,calibration_path=calibration,dataset_path="stub-dataset")
    assert verified==[set()]  # Sealing cannot inspect withheld raster/label bytes.
    assert not output.exists()
