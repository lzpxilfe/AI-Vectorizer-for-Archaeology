"""Explicit human review and masked visible-ink annotation targets."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .dataset import DatasetError, canonical_bytes, read_json, sha256_file

ANNOTATION_SCHEMA = "archaeotrace-learning-annotation/1"


def annotation_revision(annotation):
    return hashlib.sha256(canonical_bytes({k: v for k, v in annotation.items() if k != "revision"})).hexdigest()


def _point(value, width, height, *, polygon=False):
    lower = -0.5 if polygon else 0.0
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in value):
        raise DatasetError("annotation coordinates must be finite pixel [x,y]")
    x, y = value
    if not lower <= x <= width - 0.5 or not lower <= y <= height - 0.5:
        raise DatasetError("annotation coordinate outside crop")
    return float(x), float(y)


def validate_annotation(annotation, *, sample=None, require_approved=False):
    if annotation.get("schema_version") != ANNOTATION_SCHEMA or annotation.get("coordinate_space") != "pixel_xy":
        raise DatasetError("unsupported annotation schema/coordinates")
    width, height = annotation.get("width"), annotation.get("height")
    if type(width) is not int or type(height) is not int or not 1 <= width <= 1024 or not 1 <= height <= 1024:
        raise DatasetError("annotation dimensions must be bounded")
    if annotation.get("revision") != annotation_revision(annotation):
        raise DatasetError("annotation revision hash mismatch")
    if annotation.get("case_difficulty", "unassessed") not in ("unassessed", "standard", "hard"):
        raise DatasetError("case difficulty must be explicitly marked or unassessed")
    if sample is not None and (annotation.get("sample_id") != sample["id"] or annotation.get("image_sha256") != sample["image_sha256"] or [width, height] != sample["source_crop_xywh"][2:]):
        raise DatasetError("annotation does not belong to this immutable crop")
    ids = set()
    for key in ("contours", "other_ink"):
        records = annotation.get(key)
        if not isinstance(records, list) or len(records) > 4096:
            raise DatasetError("annotation paths must be bounded lists")
        for path in records:
            if not isinstance(path, dict) or not isinstance(path.get("id"), str) or path["id"] in ids:
                raise DatasetError("annotation path IDs must be unique")
            ids.add(path["id"])
            points = path.get("points", [])
            if not 2 <= len(points) <= 100000:
                raise DatasetError("a visible segment needs at least two points")
            for point in points:
                _point(point, width, height)
            if not any(a != b for a, b in zip(points, points[1:])):
                raise DatasetError("visible segment has zero length")
    for key in ("reviewed", "ignore", "other_ink_regions"):
        polygons = annotation.get(key, [] if key == "other_ink_regions" else None)
        if not isinstance(polygons, list) or len(polygons) > 4096:
            raise DatasetError("reviewed/ignore must be polygon lists")
        for polygon in polygons:
            if not isinstance(polygon, list) or not 3 <= len(polygon) <= 100000:
                raise DatasetError("reviewed/ignore polygon needs three vertices")
            points = [_point(p, width, height, polygon=True) for p in polygon]
            area = abs(sum(a[0]*b[1] - b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))
            if area <= 1e-8:
                raise DatasetError("reviewed/ignore polygon has zero area")
    approved = annotation.get("annotation_status") == "approved"
    if approved:
        if not isinstance(annotation.get("reviewer_id"), str) or not annotation["reviewer_id"].strip() or annotation["reviewer_id"] in ("ink-draft-v2", "automatic", "model"):
            raise DatasetError("approval requires a named human reviewer")
        if annotation.get("all_visible_contours_confirmed") is not True or not annotation["reviewed"]:
            raise DatasetError("reviewed area and explicit all-visible-contours confirmation required")
        if annotation.get("review_level") != "single_reviewer_pilot" or annotation.get("publication_ranking_eligible") is not False:
            raise DatasetError("one-user review must remain a non-publication pilot")
    elif annotation.get("annotation_status") != "draft":
        raise DatasetError("annotation status must be draft or approved")
    if require_approved and not approved:
        raise DatasetError("draft labels cannot enter training")
    evaluation = annotation.get("evaluation")
    if evaluation is not None:
        if not approved:
            raise DatasetError("evaluation target requires manual approval")
        selected = [p for p in annotation["contours"] if p["id"] == evaluation.get("target_contour_id")]
        if len(selected) != 1 or evaluation.get("explicit_target_selection") is not True:
            raise DatasetError("evaluation contour must be selected explicitly")
        prompt = evaluation.get("prompt", {})
        if prompt.get("start_xy") != selected[0]["points"][0] or prompt.get("end_xy") != selected[0]["points"][-1]:
            raise DatasetError("evaluation prompt must match selected endpoints")
        for key in ("start_xy", "end_xy", "previous_xy"):
            if key in prompt:
                _point(prompt[key], width, height)
        if prompt["start_xy"] == prompt["end_xy"]:
            raise DatasetError("evaluation trace endpoints must differ")
    return annotation


def load_annotation(path, *, sample=None, require_approved=False):
    return validate_annotation(read_json(path), sample=sample, require_approved=require_approved)


def rasterize_annotation(annotation, *, sigma_px=1.5, other_sigma_px=2.0):
    """Gaussian visible-segment targets; unreviewed/ignored pixels stay invalid.

    Separate polyline records never connect, even where an elevation label has
    removed ink. The model cannot infer an unannotated occluded bridge here.
    """
    import numpy as np
    from PIL import Image, ImageDraw
    validate_annotation(annotation)
    for value in (sigma_px, other_sigma_px):
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0.1 <= value <= 8:
            raise DatasetError("target sigma must be finite and bounded")
    width, height = annotation["width"], annotation["height"]
    grid_y, grid_x = np.mgrid[:height, :width].astype(np.float32)
    def polygons_mask(polygons):
        # Rasterize at pixel centres; PIL edge inclusivity is made explicit by
        # cropping a 2x canvas at odd (pixel-centre) samples.
        image = Image.new("1", (width*2+2, height*2+2), 0)
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            draw.polygon([(round(2*x+1), round(2*y+1)) for x,y in polygon], fill=1)
        return np.asarray(image, dtype=bool)[1:height*2:2, 1:width*2:2].copy()
    reviewed = polygons_mask(annotation["reviewed"])
    ignored = polygons_mask(annotation["ignore"])
    valid = reviewed & ~ignored
    def target(paths, sigma):
        minimum = np.full((height, width), np.inf, dtype=np.float32)
        for path in paths:
            for a,b in zip(path["points"],path["points"][1:]):
                ax,ay = a; bx,by = b
                dx,dy = bx-ax,by-ay
                denominator = dx*dx+dy*dy
                if denominator <= 1e-12:
                    continue
                # Limit each segment's work to a 4-sigma halo without filling
                # gaps between separately observed fragments.
                halo = int(math.ceil(4*sigma))
                x0,x1=max(0,math.floor(min(ax,bx))-halo),min(width,math.ceil(max(ax,bx))+halo+1)
                y0,y1=max(0,math.floor(min(ay,by))-halo),min(height,math.ceil(max(ay,by))+halo+1)
                gx,gy=grid_x[y0:y1,x0:x1],grid_y[y0:y1,x0:x1]
                projection=((gx-ax)*dx+(gy-ay)*dy)/denominator
                t=np.clip(projection,0,1)
                distance=(gx-ax-t*dx)**2+(gy-ay-t*dy)**2
                # Gaussian width is perpendicular to observed segments. Do
                # not extend end caps into an unobserved occluded gap.
                distance[(projection<0)|(projection>1)]=np.inf
                np.minimum(minimum[y0:y1,x0:x1],distance,out=minimum[y0:y1,x0:x1])
        score=np.exp(-minimum/(2*sigma*sigma)).astype(np.float32)
        score[~valid]=0
        return score
    other=target(annotation["other_ink"], other_sigma_px)
    regions=polygons_mask(annotation.get("other_ink_regions", [])) & valid
    other[regions]=1.0
    return {"centerline": target(annotation["contours"], sigma_px),
            "other_ink": other,
            "valid_mask": valid, "ignore_mask": ~valid}


def evaluation_reference(annotation):
    validate_annotation(annotation, require_approved=True)
    evaluation=annotation.get("evaluation")
    if evaluation is None:
        raise DatasetError("select an evaluation contour explicitly; no automatic target")
    path=next(p for p in annotation["contours"] if p["id"]==evaluation["target_contour_id"])
    masks=rasterize_annotation(annotation)
    # Final-line v1 has no ignore semantics: refuse target touching an unknown
    # pixel rather than misreporting a partly reviewed trace as ground truth.
    valid=masks["valid_mask"]
    for a,b in zip(path["points"],path["points"][1:]):
        steps=max(1,int(math.ceil(math.dist(a,b)*2)))
        for i in range(steps+1):
            x=a[0]+(b[0]-a[0])*i/steps;y=a[1]+(b[1]-a[1])*i/steps
            if not valid[min(annotation["height"]-1,int(y+0.5)),min(annotation["width"]-1,int(x+0.5))]:
                raise DatasetError("evaluation target crosses unreviewed/ignored pixels")
    return {"schema_version":"archaeotrace-centerline/1","coordinate_space":"pixel_xy",
            "image_size":{"width":annotation["width"],"height":annotation["height"]},
            "paths":[{"id":path["id"],"closed":False,"points":path["points"]}],
            "metadata":{"annotation_revision":annotation["revision"],"review_level":annotation["review_level"],"publication_ranking_eligible":False}}
