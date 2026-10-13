"""Local QGIS annotation sessions with native affine and explicit pilot review.

No training or network code is imported. Creating a session never approves
model proposals. Manual changes stay in ordinary QGIS edit buffers.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid


class AnnotationProjectError(ValueError):
    pass


def pixel_to_map(point, geotransform):
    """GDAL affine maps corners; annotation coordinates denote pixel centres."""
    x,y=point;g=geotransform
    return (g[0]+(x+0.5)*g[1]+(y+0.5)*g[2], g[3]+(x+0.5)*g[4]+(y+0.5)*g[5])


def map_to_pixel(point, geotransform):
    x,y=point;g=geotransform
    determinant=g[1]*g[5]-g[2]*g[4]
    if not math.isfinite(determinant) or abs(determinant)<=1e-24:
        raise AnnotationProjectError("native affine is singular")
    dx,dy=x-g[0],y-g[3]
    return ((dx*g[5]-dy*g[2])/determinant-0.5, (dy*g[1]-dx*g[4])/determinant-0.5)


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda:stream.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def _canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")


@dataclass
class AnnotationSession:
    sample_id: str
    image_path: Path
    image_sha256: str
    project_dir: Path
    gpkg_path: Path
    geotransform: tuple
    crs_wkt: str
    width: int
    height: int
    raster: object
    layers: dict
    parent_revision: object = None


def _safe_directory(path):
    path=Path(path).absolute()
    for parent in (path,*path.parents):
        if parent.is_symlink():raise AnnotationProjectError("annotation directory cannot traverse symlinks")
    path.mkdir(parents=True,exist_ok=True)
    return path


def _dataset_crop_metadata(path, sample_id):
    """Resume a sibling manifest safely without importing the training package."""
    root=path.parent.parent;manifest=root/"manifest.json"
    pointer=root/"dataset_latest.json"
    if pointer.is_file():
        if pointer.is_symlink() or pointer.stat().st_size>4096:raise AnnotationProjectError("unsafe latest dataset pointer")
        latest=json.loads(pointer.read_text(encoding="utf-8"));relative=Path(latest.get("manifest",""))
        if relative.is_absolute() or len(relative.parts)!=1:raise AnnotationProjectError("latest dataset pointer leaves dataset")
        manifest=root/relative
        if manifest.is_symlink() or not manifest.is_file() or _hash(manifest)!=latest.get("manifest_sha256"):
            raise AnnotationProjectError("latest dataset pointer hash mismatch")
    if not manifest.is_file():return None,None
    if manifest.is_symlink() or manifest.stat().st_size>8*1024*1024:
        raise AnnotationProjectError("unsafe sibling dataset manifest")
    data=json.loads(manifest.read_text(encoding="utf-8"))
    matches=[s for s in data.get("samples",[]) if s.get("id")==sample_id]
    if not matches:return None,None
    if len(matches)!=1:raise AnnotationProjectError("duplicate sample ID in sibling manifest")
    sample=matches[0]
    if Path(sample["image"]).is_absolute() or ".." in Path(sample["image"]).parts or (root/sample["image"]).resolve()!=path.resolve() or _hash(path)!=sample["image_sha256"]:
        raise AnnotationProjectError("crop does not match the sibling immutable dataset")
    info=sample.get("annotation",{});annotation=None
    if info.get("path"):
        relative=Path(info["path"])
        if relative.is_absolute() or ".." in relative.parts:raise AnnotationProjectError("annotation leaves dataset")
        candidate=root/relative
        if any(p.is_symlink() for p in (candidate,*candidate.parents)) or not candidate.is_file() or _hash(candidate)!=info.get("sha256"):
            raise AnnotationProjectError("resume annotation checksum/path mismatch")
        if candidate.stat().st_size>8*1024*1024:raise AnnotationProjectError("resume annotation too large")
        annotation=json.loads(candidate.read_text(encoding="utf-8"))
        if annotation.get("image_sha256")!=sample["image_sha256"] or annotation.get("sample_id")!=sample_id:
            raise AnnotationProjectError("resume annotation belongs to another crop")
        if hashlib.sha256(_canonical({k:v for k,v in annotation.items() if k!="revision"})).hexdigest()!=annotation.get("revision"):
            raise AnnotationProjectError("resume annotation revision mismatch")
    return sample,annotation


def _draft_paths(image, source_tile_origin_xy=(0,0)):
    """Walk Ink skeleton edges as drafts; no semantics or missing-ink bridges."""
    import numpy as np
    from .edge_detector import EdgeDetector
    evidence=EdgeDetector.detect_ink_evidence(image, tile_origin=source_tile_origin_xy)
    points=set((int(x),int(y)) for y,x in zip(*np.nonzero(evidence.centerline)))
    neighbors={p:sorted((p[0]+dx,p[1]+dy) for dx in (-1,0,1) for dy in (-1,0,1)
                        if (dx or dy) and (p[0]+dx,p[1]+dy) in points) for p in points}
    visited=set();paths=[]
    for start in sorted(points,key=lambda p:(len(neighbors[p])==2,p)):
        for other in neighbors[start]:
            edge=tuple(sorted((start,other)))
            if edge in visited:continue
            visited.add(edge);path=[start,other];previous,current=start,other
            while len(neighbors[current])==2 and len(path)<100000:
                following=next(p for p in neighbors[current] if p!=previous)
                edge=tuple(sorted((current,following)))
                if edge in visited:break
                visited.add(edge);path.append(following);previous,current=current,following
            if len(path)>=4:paths.append([[float(x),float(y)] for x,y in path])
            if len(paths)>=4096:return paths
    return paths


def create_annotation_project(crop_image_path, project_dir, sample_id,
                              geotransform=None, crs_wkt="", draft_contours=None,
                              source_tile_origin_xy=None):
    """Create four editable GPKG layers and a native-affine raster copy.

    The caller shows the layers. ``reviewed`` and ``ignore`` start empty;
    existing Ink proposals are drafts and are never automatic labels.
    """
    from qgis.core import (QgsVectorLayer,QgsRasterLayer,QgsVectorFileWriter,
                           QgsProject,QgsField,QgsFeature,QgsGeometry,QgsPointXY,
                           QgsCoordinateReferenceSystem)
    from qgis.PyQt.QtCore import QVariant,QMetaType
    from osgeo import gdal
    import numpy as np
    path=Path(crop_image_path).absolute()
    if path.is_symlink() or not path.is_file():raise AnnotationProjectError("regular crop file required")
    source=gdal.Open(str(path),gdal.GA_ReadOnly)
    if source is None or source.GetDriver().ShortName!="PNG" or source.RasterCount!=3 or (source.RasterXSize,source.RasterYSize)!=(512,512) or any(source.GetRasterBand(i).DataType!=gdal.GDT_Byte for i in range(1,4)):
        raise AnnotationProjectError("annotation requires a native 512x512 RGB PNG crop")
    pixels=np.moveaxis(source.ReadAsArray(),0,-1).copy();source=None
    sample,resume=_dataset_crop_metadata(path,sample_id)
    if sample is not None:
        if geotransform is None:geotransform=sample["geotransform"]
        if not crs_wkt:crs_wkt=sample.get("crs_wkt","")
        if source_tile_origin_xy is None:source_tile_origin_xy=sample["source_tile_origin_xy"]
    source_tile_origin_xy=source_tile_origin_xy or (0,0)
    transform=tuple(geotransform or (0.,1.,0.,0.,0.,-1.))
    if len(transform)!=6 or not all(type(v) in (int,float) and math.isfinite(v) for v in transform):
        raise AnnotationProjectError("finite six-value native affine required")
    map_to_pixel(pixel_to_map((0,0),transform),transform)
    if not isinstance(sample_id,str) or not sample_id.strip():raise AnnotationProjectError("sample ID required")
    base=_safe_directory(project_dir)
    session_dir=base/(re.sub(r"[^a-zA-Z0-9_-]","-",sample_id)+"-"+uuid.uuid4().hex[:12]);session_dir.mkdir()
    raster_path=session_dir/"crop.tif"
    dataset=gdal.GetDriverByName("GTiff").Create(str(raster_path),512,512,3,gdal.GDT_Byte,options=["COMPRESS=DEFLATE"])
    if dataset is None:raise AnnotationProjectError("could not create native crop raster")
    dataset.SetGeoTransform(transform)
    if crs_wkt:dataset.SetProjection(crs_wkt)
    for channel in range(3):dataset.GetRasterBand(channel+1).WriteArray(pixels[:,:,channel])
    dataset.FlushCache();dataset=None
    raster=QgsRasterLayer(str(raster_path),sample_id+" native crop")
    if not raster.isValid():raise AnnotationProjectError("native crop raster is invalid")
    crs=QgsCoordinateReferenceSystem()
    if crs_wkt:crs.createFromWkt(crs_wkt)
    gpkg=session_dir/"annotations.gpkg";layers={}
    string_type=getattr(QVariant,"String",None)
    if string_type is None:string_type=QMetaType.Type.QString
    for index,name in enumerate(("contours","other_ink","other_ink_regions","reviewed","ignore")):
        geometry="LineString" if name in ("contours","other_ink") else "Polygon"
        memory=QgsVectorLayer(geometry,name,"memory")
        if crs.isValid():memory.setCrs(crs)
        memory.dataProvider().addAttributes([QgsField("label_state",string_type),QgsField("sample_id",string_type)])
        memory.updateFields()
        options=QgsVectorFileWriter.SaveVectorOptions();options.driverName="GPKG";options.layerName=name
        options.actionOnExistingFile=(QgsVectorFileWriter.CreateOrOverwriteFile if index==0 else QgsVectorFileWriter.CreateOrOverwriteLayer)
        result=QgsVectorFileWriter.writeAsVectorFormatV3(memory,str(gpkg),QgsProject.instance().transformContext(),options)
        if result[0]!=QgsVectorFileWriter.NoError:raise AnnotationProjectError("could not create blank annotation layer: "+name)
        layer=QgsVectorLayer(str(gpkg)+"|layername="+name,name+" — "+sample_id,"ogr")
        if not layer.isValid() or not layer.startEditing():raise AnnotationProjectError("annotation edit buffer unavailable")
        layers[name]=layer
    proposed={name:[] for name in layers}
    if resume is not None:
        for name in layers:proposed[name]=resume.get(name,[])
    if not proposed["contours"] and not proposed["reviewed"]:
        proposals=_draft_paths(pixels,source_tile_origin_xy) if draft_contours is None else draft_contours
        proposed["contours"]=[{"points":points} for points in proposals]
    for name,layer in layers.items():
        layer.beginEditCommand("Draft annotations for manual review")
        try:
            for record in proposed[name]:
                points=record["points"] if name in ("contours","other_ink") else record
                native=[QgsPointXY(*pixel_to_map(point,transform)) for point in points]
                if len(native)<(2 if name in ("contours","other_ink") else 3):continue
                feature=QgsFeature(layer.fields())
                feature.setGeometry(QgsGeometry.fromPolylineXY(native) if name in ("contours","other_ink") else QgsGeometry.fromPolygonXY([native]))
                feature["label_state"]="draft";feature["sample_id"]=sample_id
                if not layer.addFeature(feature):raise AnnotationProjectError("draft geometry could not enter edit buffer")
        except Exception:
            layer.destroyEditCommand();raise
        layer.endEditCommand()
    return AnnotationSession(sample_id,path,_hash(path),session_dir,gpkg,transform,crs_wkt,512,512,raster,layers,resume.get("revision") if resume else None)


def export_annotation_project(session, output_path, annotator_id, reviewer_id=None,
                              approve=False, all_visible_contours_confirmed=False,
                              evaluation_target_id=None, case_difficulty="unassessed"):
    """User-authorized export; approval never follows from an Ink proposal."""
    if _hash(session.image_path)!=session.image_sha256:
        raise AnnotationProjectError("crop changed during annotation")
    if not isinstance(annotator_id,str) or not annotator_id.strip():raise AnnotationProjectError("annotator identity required")
    if case_difficulty not in ("unassessed","standard","hard"):
        raise AnnotationProjectError("difficulty must be manually marked standard/hard or remain unassessed")
    if approve:
        if not isinstance(reviewer_id,str) or not reviewer_id.strip() or reviewer_id in ("ink-draft-v2","automatic","model"):
            raise AnnotationProjectError("a named human reviewer must explicitly approve")
        if all_visible_contours_confirmed is not True:raise AnnotationProjectError("confirm all visible contours in reviewed areas")
        if any(layer.isModified() for layer in session.layers.values()):
            raise AnnotationProjectError("save QGIS layer edits before approving annotations")
    records={};selected_ids=[]
    for name,layer in session.layers.items():
        if not layer.isValid():raise AnnotationProjectError("annotation layer is no longer valid")
        records[name]=[]
        for feature in layer.getFeatures():
            geometry=feature.geometry()
            if geometry.isEmpty() or not geometry.isGeosValid():raise AnnotationProjectError("invalid annotation geometry")
            if name in ("contours","other_ink"):
                parts=geometry.asMultiPolyline() if geometry.isMultipart() else [geometry.asPolyline()]
                for part_index,part in enumerate(parts):
                    identity=name+"-"+str(feature.id())+"-"+str(part_index)
                    points=[list(map_to_pixel((p.x(),p.y()),session.geotransform)) for p in part]
                    if len(points)<2:raise AnnotationProjectError("visible ink needs an observed line segment")
                    if any(not(-1e-7<=x<=511.5+1e-7 and -1e-7<=y<=511.5+1e-7) for x,y in points):raise AnnotationProjectError("line leaves the crop")
                    points=[[max(0.0,min(511.5,x)),max(0.0,min(511.5,y))] for x,y in points]
                    records[name].append({"id":identity,"points":points})
                    if name=="contours" and feature.id() in layer.selectedFeatureIds():selected_ids.append(identity)
            else:
                polygons=geometry.asMultiPolygon() if geometry.isMultipart() else [geometry.asPolygon()]
                for polygon in polygons:
                    if len(polygon)!=1:raise AnnotationProjectError("polygon holes must be explicit ignore polygons")
                    ring=[list(map_to_pixel((p.x(),p.y()),session.geotransform)) for p in polygon[0]]
                    if any(not(-.5-1e-7<=x<=511.5+1e-7 and -.5-1e-7<=y<=511.5+1e-7) for x,y in ring):raise AnnotationProjectError("review/ignore polygon leaves crop")
                    records[name].append([[max(-.5,min(511.5,x)),max(-.5,min(511.5,y))] for x,y in ring])
    if approve and not records["reviewed"]:raise AnnotationProjectError("draw and save a reviewed area before approval")
    target_id=evaluation_target_id
    if approve and target_id is None and len(selected_ids)==1:target_id=selected_ids[0]
    evaluation=None
    if target_id is not None:
        if not approve:raise AnnotationProjectError("evaluation target requires an approved annotation")
        selected=[line for line in records["contours"] if line["id"]==target_id]
        if len(selected)!=1:raise AnnotationProjectError("explicit evaluation contour ID not found")
        points=selected[0]["points"]
        if points[0]==points[-1]:raise AnnotationProjectError("select an open evaluation segment")
        evaluation={"target_contour_id":target_id,"explicit_target_selection":True,
                    "prompt":{"schema_version":"archaeotrace-trace-prompt/2","start_xy":points[0],"end_xy":points[-1],"positive_xy":[],"negative_xy":[]}}
    annotation={"schema_version":"archaeotrace-learning-annotation/1","coordinate_space":"pixel_xy",
                "sample_id":session.sample_id,"image_sha256":session.image_sha256,"width":512,"height":512,
                "annotation_status":"approved" if approve else "draft","annotator_id":annotator_id.strip(),
                "reviewer_id":reviewer_id.strip() if isinstance(reviewer_id,str) else None,
                "review_level":"single_reviewer_pilot" if approve else "unreviewed",
                "all_visible_contours_confirmed":bool(approve and all_visible_contours_confirmed),
                "publication_ranking_eligible":False,"parent_revision":session.parent_revision,
                "geotransform":list(session.geotransform),"crs_wkt":session.crs_wkt,"case_difficulty":case_difficulty,"evaluation":evaluation,**records}
    annotation["revision"]=hashlib.sha256(_canonical(annotation)).hexdigest()
    target=Path(output_path).absolute();_safe_directory(target.parent)
    if target.exists():raise AnnotationProjectError("use a new annotation revision path; never overwrite labels")
    with target.open("xb") as handle:
        handle.write(_canonical(annotation));handle.flush();os.fsync(handle.fileno())
    session.parent_revision=annotation["revision"]
    return annotation
