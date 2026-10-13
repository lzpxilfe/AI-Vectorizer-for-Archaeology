"""Explicit public-source preparation and manual annotation attachment CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import urllib.request
import uuid
import zipfile

from .dataset import (SCHEMA_VERSION, SPLITS, DatasetError, canonical_bytes,
                      checked_file, image_pixel_sha256, load_dataset_manifest,
                      read_json, safe_path, sha256_file)
from .annotations import (ANNOTATION_SCHEMA, annotation_revision,
                          evaluation_reference, load_annotation)

DEFAULT_PLAN = Path(__file__).parent / "data" / "usgs-contour-pilot-plan.json"


def _write_new(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_bytes(value));handle.flush();os.fsync(handle.fileno())


def crop_coordinates(width, height, count, seed, source_id):
    """Stratified interior cells with fixed jitter; no model-based cherry-pick."""
    if width<1024 or height<1024 or count not in (6,12):raise DatasetError("source must support six/twelve native 512px interior crops")
    random_seed=int(hashlib.sha256((seed+"|"+source_id+"|crop-grid-v1").encode()).hexdigest(),16)
    rng=random.Random(random_seed);rows=count//3
    left,top=int(width*.18),int(height*.18)
    right,bottom=int(width*.82),int(height*.82)
    cell_w,cell_h=(right-left)/3,(bottom-top)/rows
    boxes=[]
    for row in range(rows):
        for col in range(3):
            cx=left+(col+.5)*cell_w+rng.uniform(-.12,.12)*cell_w
            cy=top+(row+.5)*cell_h+rng.uniform(-.12,.12)*cell_h
            x=max(0,min(width-512,int(cx-256)));y=max(0,min(height-512,int(cy-256)))
            box=[x,y,512,512]
            if box in boxes:raise DatasetError("source is too small for unique stratified crops")
            boxes.append(box)
    return boxes


def _fetch_source(url, destination, expected_hash, expected_bytes):
    if destination.exists():
        if sha256_file(destination)!=expected_hash:raise DatasetError("existing source checksum mismatch")
        return
    if not url.startswith("https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/HistoricalTopo/GeoTIFF/"):
        raise DatasetError("preparation downloads only selected official USGS historical GeoTIFFs")
    temporary=destination.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url,timeout=60) as response, temporary.open("xb") as handle:
            if response.url!=url:raise DatasetError("unexpected source redirect")
            digest=hashlib.sha256();size=0
            for chunk in iter(lambda:response.read(1024*1024),b""):
                size+=len(chunk)
                if size>min(400*1024*1024,expected_bytes):raise DatasetError("source exceeded pinned byte count")
                handle.write(chunk);digest.update(chunk)
            handle.flush();os.fsync(handle.fileno())
        if size!=expected_bytes or digest.hexdigest()!=expected_hash:raise DatasetError("downloaded public source does not match frozen artifact")
        temporary.rename(destination)
    finally:
        if temporary.exists():temporary.unlink()


def prepare_pilot(output_dir, plan_path=DEFAULT_PLAN, *, use_existing_sources=False):
    """Freeze source plan first, then fetch pinned rasters and exact source crops.

    All annotations are intentionally empty drafts. No prepared crop is training
    ready until the user marks a reviewed area and explicitly approves labels.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    from PIL import Image
    plan_path=Path(plan_path).resolve();plan=read_json(plan_path)
    root=Path(output_dir).absolute()
    if root.is_symlink():raise DatasetError("output cannot be a symlink")
    root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists():raise DatasetError("use a new dataset output directory")
    for name in ("sources","images","annotations","rights","evidence"): (root/name).mkdir(exist_ok=True)
    selection=root/"source-selection.json"
    if selection.exists():
        if sha256_file(selection)!=plan["source_selection_sha256"]:raise DatasetError("existing split freeze differs")
    else:shutil.copyfile(plan_path.parent/plan["source_selection"],selection)
    # Rights text is a complete source snapshot, rather than invented item terms.
    rights_name="rights/usgs-topographic-rights.txt"
    shutil.copyfile(checked_file(plan_path.parent,plan["rights_snapshot"],plan["rights_snapshot_sha256"]),root/rights_name)
    sources=[];samples=[]
    for source in plan["sources"]:
        record=dict(source);source_path=root/"sources"/(record["id"]+".tif")
        if use_existing_sources:
            if not source_path.is_file() or sha256_file(source_path)!=record["source_sha256"]:raise DatasetError("pinned local source missing or changed")
        else:_fetch_source(record["source_url"],source_path,record["source_sha256"],record["source_bytes"])
        record.update(rights_snapshot=rights_name,rights_snapshot_sha256=plan["rights_snapshot_sha256"],source_file=str(source_path.relative_to(root)))
        with rasterio.open(source_path) as raster:
            if raster.dtypes!=tuple("uint8" for _ in raster.dtypes) or raster.count not in (1,3,4):raise DatasetError("pilot source must be native Byte RGB/palette")
            record["source_size"]=[raster.width,raster.height]
            record["raster_size"]={"width":raster.width,"height":raster.height}
            record["source_geotransform"]=list(raster.transform.to_gdal())
            record["crs_wkt"]=raster.crs.to_wkt() if raster.crs else ""
            palette=raster.colormap(1) if raster.count==1 else None
            for index,box in enumerate(crop_coordinates(raster.width,raster.height,record["crop_count"],plan["seed"],record["id"]),1):
                sid=record["id"]+"-crop-"+str(index).zfill(2)
                window=Window(*box);native=raster.read(window=window)
                if palette is not None:
                    lookup=np.asarray([palette.get(i,(255,255,255,255))[:3] for i in range(256)],dtype=np.uint8)
                    image=lookup[native[0]]
                else:image=np.moveaxis(native[:3],0,-1)
                image_path=root/"images"/(sid+".png");Image.fromarray(image,mode="RGB").save(image_path,compress_level=9)
                transform=list(raster.window_transform(window).to_gdal())
                # QGIS reads PNG world files without resampling; center convention
                # matches pixel_to_map in the standalone annotation service.
                g=transform;world=[g[1],g[4],g[2],g[5],g[0]+.5*g[1]+.5*g[2],g[3]+.5*g[4]+.5*g[5]]
                image_path.with_suffix(".pgw").write_text("\n".join(format(v,".17g") for v in world)+"\n")
                image_path.with_suffix(".prj").write_text(record["crs_wkt"]+"\n")
                sample={"id":sid,"source_id":record["id"],"split":record["split"],"source_crop_xywh":box,"source_tile_origin_xy":box[:2],
                        "image":str(image_path.relative_to(root)),"image_sha256":sha256_file(image_path),"image_pixel_sha256":image_pixel_sha256(image_path),
                        "geotransform":transform,"crs_wkt":record["crs_wkt"]}
                annotation={"schema_version":ANNOTATION_SCHEMA,"coordinate_space":"pixel_xy","sample_id":sid,"image_sha256":sample["image_sha256"],
                            "width":512,"height":512,"annotation_status":"draft","annotator_id":"unannotated-preparation","reviewer_id":None,
                            "review_level":"unreviewed","all_visible_contours_confirmed":False,"publication_ranking_eligible":False,
                            "parent_revision":None,"contours":[],"other_ink":[],"other_ink_regions":[],"reviewed":[],"ignore":[],"evaluation":None}
                annotation["revision"]=annotation_revision(annotation)
                annotation_path=root/"annotations"/(sid+"-draft.json");_write_new(annotation_path,annotation)
                sample["annotation"]={"path":str(annotation_path.relative_to(root)),"sha256":sha256_file(annotation_path),"status":"draft","revision":annotation["revision"]}
                samples.append(sample)
        sources.append(record)
        print("Prepared",record["id"],record["crop_count"],"draft crops",flush=True)
    data={"schema_version":SCHEMA_VERSION,"dataset_id":"usgs-contour-pilot-16x144-v1","seed":plan["seed"],
          "publication_ranking_eligible":False,"review_status":"drafts_need_user_review",
          "source_selection":{"path":"source-selection.json","sha256":plan["source_selection_sha256"]},
          "split_policy":{"assignment_unit":"sheet_family","frozen":True,"source_counts":{s:sum(x["split"]==s for x in sources) for s in SPLITS},
                          "sample_counts":{s:sum(x["split"]==s for x in samples) for s in SPLITS}},
          "crop_config":{"size":512,"selector":"stratified-grid-jitter-v1","seed":plan["seed"],"interior_fraction":[.18,.82],"jitter_fraction":.12,
                         "rgb_interpretation":"native RGB bands or embedded palette lookup; no resampling/stretch","lossless":True},
          "sources":sources,"samples":samples}
    _write_new(root/"manifest.json",data)
    result=load_dataset_manifest(root/"manifest.json")
    print("144 public crops are drafts; manual review is required before training.",flush=True)
    return result


def attach_annotation(manifest_path, sample_id, annotation_path, output_path):
    manifest=load_dataset_manifest(manifest_path)
    samples=[s for s in manifest.payload["samples"] if s["id"]==sample_id]
    if len(samples)!=1:raise DatasetError("sample not found")
    sample=samples[0];annotation=load_annotation(annotation_path,sample=sample)
    # Canonical immutable revisions are copied into the dataset, never aliased
    # to an editable user file. A new manifest revision preserves prior state.
    destination=manifest.root/"annotations"/(sample_id+"-"+annotation["revision"]+".json")
    if destination.exists():
        if read_json(destination)!=annotation:raise DatasetError("annotation revision collision")
    else:_write_new(destination,annotation)
    sample["annotation"]={"path":str(destination.relative_to(manifest.root)),"sha256":sha256_file(destination),"status":annotation["annotation_status"],"revision":annotation["revision"]}
    sample["difficulty"]=annotation.get("case_difficulty","unassessed")
    if annotation.get("evaluation") is not None:
        reference=evaluation_reference(annotation);reference_path=manifest.root/"references"/(sample_id+"-"+annotation["revision"]+".json")
        if not reference_path.exists():_write_new(reference_path,reference)
        sample.update(reference_file=str(reference_path.relative_to(manifest.root)),reference_sha256=sha256_file(reference_path),prompt=annotation["evaluation"]["prompt"])
    else:
        for key in ("reference_file","reference_sha256","prompt"):sample.pop(key,None)
    output_path=Path(output_path).absolute()
    if output_path.parent.resolve()!=manifest.root:raise DatasetError("new manifest revision must stay beside dataset artifacts")
    manifest.payload["parent_manifest_sha256"]=manifest.manifest_sha256
    _write_new(output_path,manifest.payload)
    result=load_dataset_manifest(output_path)
    pointer=manifest.root/"dataset_latest.json"
    if pointer.is_symlink():raise DatasetError("latest dataset pointer cannot be a symlink")
    temporary=manifest.root/(".dataset-latest-"+uuid.uuid4().hex+".json")
    _write_new(temporary,{"schema_version":"archaeotrace-learning-dataset-pointer/1","manifest":output_path.name,"manifest_sha256":result.manifest_sha256})
    os.replace(temporary,pointer)
    return result


def mark_case_difficulty(manifest_path, sample_id, difficulty, output_path):
    """Explicit manual case classification creates a new annotation revision."""
    if difficulty not in ("standard","hard"):
        raise DatasetError("manual case difficulty must be standard or hard")
    manifest=load_dataset_manifest(manifest_path)
    samples=[s for s in manifest.payload["samples"] if s["id"]==sample_id]
    if len(samples)!=1:raise DatasetError("sample not found")
    sample=samples[0];info=sample["annotation"]
    annotation=dict(load_annotation(checked_file(manifest.root,info["path"],info["sha256"]),sample=sample))
    annotation["parent_revision"]=annotation["revision"]
    annotation["case_difficulty"]=difficulty
    annotation["revision"]=annotation_revision(annotation)
    label_path=manifest.root/"annotations"/(sample_id+"-"+annotation["revision"]+".json")
    if not label_path.exists():_write_new(label_path,annotation)
    return attach_annotation(manifest_path,sample_id,label_path,output_path)


def verify_prepared_crops(manifest_path):
    """Prove decoded PNG pixels equal native source windows, not resized views."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window
    from PIL import Image
    manifest=load_dataset_manifest(manifest_path);verified=0;drafts=0
    for source in manifest.sources.values():
        if source.get("source_file") is None:raise DatasetError("native source rasters are required for pixel/context verification")
        with rasterio.open(checked_file(manifest.root,source["source_file"],source["source_sha256"])) as raster:
            palette=raster.colormap(1) if raster.count==1 else None
            for sample in (s for s in manifest.payload["samples"] if s["source_id"]==source["id"]):
                native=raster.read(window=Window(*sample["source_crop_xywh"]))
                if palette is not None:
                    lookup=np.asarray([palette.get(i,(255,255,255,255))[:3] for i in range(256)],dtype=np.uint8)
                    expected=lookup[native[0]]
                else:expected=np.moveaxis(native[:3],0,-1)
                with Image.open(manifest.root/sample["image"]) as image:actual=np.asarray(image)
                if actual.shape!=expected.shape or not np.array_equal(actual,expected):raise DatasetError("crop is not lossless native source pixels: "+sample["id"])
                info=sample["annotation"]
                annotation=load_annotation(checked_file(manifest.root,info["path"],info["sha256"]),sample=sample)
                drafts+=annotation["annotation_status"]=="draft";verified+=1
    return {"schema_version":"archaeotrace-learning-data-verification/1","manifest_sha256":manifest.manifest_sha256,
            "sources":len(manifest.sources),"verified_native_crops":verified,"draft_annotations":drafts,
            "publication_ranking_eligible":False,"historical_map_accuracy_evidence":False}


def create_colab_archive(manifest_path, output_path, *, include_source_rasters=True):
    """Portable public crops/labels plus native context for honest evaluation."""
    manifest=load_dataset_manifest(manifest_path);data=json.loads(json.dumps(manifest.payload));files=set()
    for source in data["sources"]:
        if include_source_rasters and source.get("source_file"):
            files.add(source["source_file"])
        else:source["source_file"]=None
        files.add(source["rights_snapshot"])
    files.add(data["source_selection"]["path"])
    for sample in data["samples"]:
        files.update([sample["image"],sample["annotation"]["path"]])
        if sample.get("reference_file"):files.add(sample["reference_file"])
    target=Path(output_path).absolute()
    if target.exists():raise DatasetError("archive already exists")
    with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):archive.write(safe_path(manifest.root,name),name)
        archive.writestr("manifest.json",canonical_bytes(data))
        archive.writestr("REVIEW_REQUIRED.txt","Public images are staged. Draft annotations are NOT training labels. Mark reviewed/ignore areas and explicitly approve every training/validation sample. Holdout is for frozen evaluation only.\n")
    return target


def add_data_commands(parser):
    sub=parser.add_subparsers(dest="data_command",required=True)
    prepare=sub.add_parser("prepare",help="explicitly fetch pinned public USGS rasters and produce draft crops")
    prepare.add_argument("--plan",default=str(DEFAULT_PLAN));prepare.add_argument("--output",required=True);prepare.add_argument("--existing-sources",action="store_true")
    validate=sub.add_parser("validate",help="check rights, split isolation and hashes")
    validate.add_argument("manifest");validate.add_argument("--require-approved",action="store_true")
    verify=sub.add_parser("verify-pixels",help="verify lossless crops against their pinned native rasters")
    verify.add_argument("manifest")
    attach=sub.add_parser("attach",help="attach an explicitly exported annotation revision")
    attach.add_argument("manifest");attach.add_argument("--sample",required=True);attach.add_argument("--annotation",required=True);attach.add_argument("--output",required=True)
    difficulty=sub.add_parser("difficulty",help="manually classify a case before freezing evaluation; never auto-label")
    difficulty.add_argument("manifest");difficulty.add_argument("sample_id");difficulty.add_argument("difficulty",choices=("standard","hard"));difficulty.add_argument("--output",required=True)
    archive=sub.add_parser("archive",help="create portable public crop/annotation archive")
    archive.add_argument("manifest");archive.add_argument("--output",required=True)
    archive.add_argument("--crops-only",action="store_true",help="omit native rasters; this archive cannot run source-context evaluation")


def dispatch_data(args):
    try:
        if args.data_command=="prepare":prepare_pilot(args.output,args.plan,use_existing_sources=args.existing_sources)
        elif args.data_command=="validate":
            manifest=load_dataset_manifest(args.manifest,require_approved=args.require_approved)
            print(json.dumps({"manifest_sha256":manifest.manifest_sha256,"samples":{s:len(manifest.samples_for(s)) for s in SPLITS},"publication_ranking_eligible":False}))
        elif args.data_command=="attach":attach_annotation(args.manifest,args.sample,args.annotation,args.output)
        elif args.data_command=="difficulty":mark_case_difficulty(args.manifest,args.sample_id,args.difficulty,args.output)
        elif args.data_command=="verify-pixels":print(json.dumps(verify_prepared_crops(args.manifest)))
        elif args.data_command=="archive":print(create_colab_archive(args.manifest,args.output,include_source_rasters=not args.crops_only))
        return 0
    except (DatasetError,OSError,ValueError) as exc:
        print("Dataset action refused:",str(exc));return 2
