"""Split leakage, crop identity and explicit approval integration tests."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from learning.annotations import annotation_revision
from learning.data_cli import attach_annotation, crop_coordinates, create_colab_archive, mark_case_difficulty
from learning.dataset import (ContourDataset, DatasetError, image_pixel_sha256,
                              load_dataset_manifest, sha256_file)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+"\n")


def fixture_manifest(root, approved=True):
    (root/"rights.txt").write_text("Synthetic rights fixture, not public map evidence.\n")
    sources=[];samples=[]
    splits=("train","validation","calibration","holdout")
    for i,split in enumerate(splits):
        sid=f"sheet-{i}";name=f"Quadrangle {i}";family=f"xx-quadrangle-{i}"
        sources.append({"id":sid,"split":split,"family_id":family,"quadrangle_name":name,"state":"XX","aliases":[family],
                        "source_url":f"https://example.test/{i}.tif","source_sha256":str(i+1)*64,
                        "source_size":[1024,1024],"source_geotransform":[0,1,0,0,0,-1],
                        "provenance_id":{"authority":"example.test","namespace":"fixture","value":sid},
                        "rights_status":"public_domain","license":"Synthetic fixture only","rights_statement_url":"https://example.test/rights",
                        "rights_snapshot":"rights.txt","rights_snapshot_sha256":sha256_file(root/"rights.txt"),"source_file":None})
        image=root/"images"/(sid+".png");image.parent.mkdir(exist_ok=True)
        pixels=np.full((512,512,3),250-i,dtype=np.uint8);pixels[8:10,2:20]=30+i
        Image.fromarray(pixels).save(image)
        sample={"id":sid+"-crop","source_id":sid,"split":split,"source_crop_xywh":[0,0,512,512],"source_tile_origin_xy":[0,0],
                "geotransform":[0,1,0,0,0,-1],"image":str(image.relative_to(root)),"image_sha256":sha256_file(image),"image_pixel_sha256":image_pixel_sha256(image)}
        label={"schema_version":"archaeotrace-learning-annotation/1","coordinate_space":"pixel_xy","sample_id":sample["id"],"image_sha256":sample["image_sha256"],
               "width":512,"height":512,"annotation_status":"approved" if approved else "draft","annotator_id":"user","reviewer_id":"user" if approved else None,
               "review_level":"single_reviewer_pilot" if approved else "unreviewed","all_visible_contours_confirmed":approved,"publication_ranking_eligible":False,
               "parent_revision":None,"contours":[{"id":"contour","points":[[2,8],[20,8]]}],"other_ink":[],"other_ink_regions":[],
               "reviewed":[[[-.5,-.5],[511.5,-.5],[511.5,511.5],[-.5,511.5]]] if approved else [],"ignore":[],"evaluation":None}
        label["revision"]=annotation_revision(label);labelpath=root/"annotations"/(sid+".json");write(labelpath,label)
        sample["annotation"]={"path":str(labelpath.relative_to(root)),"sha256":sha256_file(labelpath),"status":label["annotation_status"],"revision":label["revision"]}
        samples.append(sample)
    payload={"schema_version":"archaeotrace-learning-dataset/1","dataset_id":"synthetic-unit-fixture","seed":"fixture-only","publication_ranking_eligible":False,
             "split_policy":{"assignment_unit":"sheet_family","frozen":True,"source_counts":{s:1 for s in splits},"sample_counts":{s:1 for s in splits}},"sources":sources,"samples":samples}
    path=root/"manifest.json";write(path,payload);return path,payload


def test_approved_dataset_yields_correct_masks_and_train_only(tmp_path):
    path,_=fixture_manifest(tmp_path)
    dataset=ContourDataset(path,"train");item=dataset[0]
    assert dataset.sample_ids==("sheet-0-crop",)
    assert item["image"].shape==(512,512,3) and item["image"].dtype==np.uint8
    assert item["centerline"].shape==item["valid_mask"].shape==(512,512)
    assert item["centerline"][8,10]==1
    with pytest.raises(DatasetError):ContourDataset(path,"holdout")


def test_all_drafts_block_training_without_auto_approval(tmp_path):
    path,_=fixture_manifest(tmp_path,False)
    load_dataset_manifest(path)
    with pytest.raises(DatasetError,match="manual review"):ContourDataset(path,"train")
    with pytest.raises(DatasetError):load_dataset_manifest(path,require_approved=True)


@pytest.mark.parametrize("leak",["alias","source_hash","source_url","pixel_content"])
def test_cross_split_aliases_and_reencoded_identical_pixels_fail(tmp_path,leak):
    path,data=fixture_manifest(tmp_path)
    if leak=="alias":data["sources"][1]["aliases"].append(data["sources"][0]["family_id"].upper())
    elif leak=="source_hash":data["sources"][1]["source_sha256"]=data["sources"][0]["source_sha256"]
    elif leak=="source_url":data["sources"][1]["source_url"]=data["sources"][0]["source_url"]
    else:
        other=tmp_path/data["samples"][1]["image"]
        with Image.open(tmp_path/data["samples"][0]["image"]) as image:image.save(other,compress_level=1)
        data["samples"][1]["image_sha256"]=sha256_file(other)
        data["samples"][1]["image_pixel_sha256"]=image_pixel_sha256(other)
    write(path,data)
    with pytest.raises(DatasetError,match="leak"):load_dataset_manifest(path)


def test_source_origin_and_affine_cannot_be_silently_changed(tmp_path):
    path,data=fixture_manifest(tmp_path)
    data["samples"][0]["source_tile_origin_xy"]=[1,0];write(path,data)
    with pytest.raises(DatasetError,match="origin"):load_dataset_manifest(path)
    data["samples"][0]["source_tile_origin_xy"]=[0,0];data["samples"][0]["geotransform"][0]=100;write(path,data)
    with pytest.raises(DatasetError,match="affine"):load_dataset_manifest(path)


def test_hash_tampering_and_symlink_escape_are_rejected(tmp_path):
    path,data=fixture_manifest(tmp_path)
    image=tmp_path/data["samples"][0]["image"]
    original=image.read_bytes();image.write_bytes(original+b"changed")
    with pytest.raises(DatasetError,match="checksum"):load_dataset_manifest(path)
    image.unlink();target=tmp_path/"external.png";target.write_bytes(original);image.symlink_to(target)
    with pytest.raises(DatasetError,match="symlink"):load_dataset_manifest(path)


def test_wholly_ignored_approved_tile_is_refused(tmp_path):
    path,data=fixture_manifest(tmp_path);info=data["samples"][0]["annotation"];p=tmp_path/info["path"]
    label=json.loads(p.read_text());label["ignore"]=label["reviewed"];label["revision"]=annotation_revision(label);write(p,label)
    info["sha256"]=sha256_file(p);write(path,data)
    with pytest.raises(DatasetError,match="manual review"):ContourDataset(path,"train")


def test_annotation_attachment_creates_revision_without_rewriting_previous(tmp_path):
    path,data=fixture_manifest(tmp_path)
    old=path.read_bytes();label=tmp_path/data["samples"][0]["annotation"]["path"]
    result=attach_annotation(path,"sheet-0-crop",label,tmp_path/"manifest-reviewed.json")
    assert path.read_bytes()==old
    assert result.payload["parent_manifest_sha256"]==sha256_file(path)
    assert result.payload["samples"][0]["annotation"]["status"]=="approved"
    pointer=json.loads((tmp_path/"dataset_latest.json").read_text())
    assert pointer["manifest"]=="manifest-reviewed.json"
    assert pointer["manifest_sha256"]==result.manifest_sha256


def test_crop_selection_is_seeded_before_image_analysis():
    assert crop_coordinates(5000,6000,12,"seed","a")==crop_coordinates(5000,6000,12,"seed","a")
    assert crop_coordinates(5000,6000,12,"seed","a")!=crop_coordinates(5000,6000,12,"seed","b")


def test_training_constructor_never_reads_holdout_images_or_annotations(tmp_path,monkeypatch):
    path,data=fixture_manifest(tmp_path)
    protected={str((tmp_path/s[key]).resolve()) for s in data["samples"] if s["split"] in ("calibration","holdout") for key in ("image",)}
    protected.update(str((tmp_path/s["annotation"]["path"]).resolve()) for s in data["samples"] if s["split"] in ("calibration","holdout"))
    original=Path.open
    def guarded(self,*args,**kwargs):
        if str(self.resolve()) in protected:raise AssertionError("training accessed reserved evaluation bytes")
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    item=ContourDataset(path,"train")[0]
    assert item["sample_id"]=="sheet-0-crop"


def test_manual_difficulty_is_revisioned_and_does_not_approve_drafts(tmp_path):
    path,data=fixture_manifest(tmp_path,False)
    result=mark_case_difficulty(path,"sheet-3-crop","hard",tmp_path/"manifest-hard.json")
    assert result.payload["samples"][3]["difficulty"]=="hard"
    assert result.payload["samples"][3]["annotation"]["status"]=="draft"
    assert "difficulty" not in json.loads(path.read_text())["samples"][3]
    with pytest.raises(DatasetError):mark_case_difficulty(path,"sheet-3-crop","inferred",tmp_path/"invalid.json")
