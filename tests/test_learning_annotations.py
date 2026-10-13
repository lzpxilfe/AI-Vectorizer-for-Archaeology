"""Unknown pixels and single-user review must never become implicit labels."""
import copy

import numpy as np
import pytest

from learning.annotations import (annotation_revision, evaluation_reference,
                                 rasterize_annotation, validate_annotation)
from learning.dataset import DatasetError


def annotation(approved=True):
    value={"schema_version":"archaeotrace-learning-annotation/1","coordinate_space":"pixel_xy",
           "sample_id":"fixture","image_sha256":"a"*64,"width":32,"height":32,
           "annotation_status":"approved" if approved else "draft","annotator_id":"researcher",
           "reviewer_id":"researcher" if approved else None,"review_level":"single_reviewer_pilot" if approved else "unreviewed",
           "all_visible_contours_confirmed":approved,"publication_ranking_eligible":False,"parent_revision":None,
           "contours":[{"id":"left","points":[[2,8],[8,8]]},{"id":"right","points":[[22,8],[28,8]]}],
           "other_ink":[{"id":"road","points":[[3,15],[20,15]]}],"other_ink_regions":[],
           "reviewed":[[[-.5,-.5],[20,-.5],[20,31.5],[-.5,31.5]]],
           "ignore":[[[4,6],[7,6],[7,10],[4,10]]],"evaluation":None}
    value["revision"]=annotation_revision(value)
    return value


def reseal(value):
    value["revision"]=annotation_revision(value)
    return value


def machine_annotation():
    value = annotation(False)
    value.update(annotation_status="machine_labeled", review_level="automated_weak_labels",
                 reviewer_id=None, all_visible_contours_confirmed=False,
                 human_review_required=True,
                 weak_label_provenance={"method": "test-only", "code_sha256": "a" * 64, "run_id": "fixture"})
    return reseal(value)


def test_machine_labels_are_structurally_valid_but_never_human_approval():
    value = machine_annotation()
    validate_annotation(value)
    with pytest.raises(DatasetError, match="human-reviewed"):
        validate_annotation(value, require_approved=True)
    assert rasterize_annotation(value)["valid_mask"].any()
    with pytest.raises(DatasetError):
        evaluation_reference(value)


@pytest.mark.parametrize("change", [
    {"reviewer_id": "person"}, {"all_visible_contours_confirmed": True},
    {"human_review_required": False}, {"review_level": "single_reviewer_pilot"},
    {"weak_label_provenance": {"method": "fixture", "run_id": "fixture"}},
    {"annotation_status": "approved", "reviewer_id": "person", "all_visible_contours_confirmed": True},
])
def test_machine_labels_cannot_forge_human_review_or_omit_provenance(change):
    value = machine_annotation()
    value.update(change)
    with pytest.raises(DatasetError):
        validate_annotation(reseal(value))


def test_unreviewed_and_explicit_ignore_have_no_supervision():
    result=rasterize_annotation(annotation())
    assert not result["valid_mask"][8,5]
    assert not result["valid_mask"][8,25]
    assert result["centerline"][8,5]==result["centerline"][8,25]==0
    assert result["other_ink"][15,10]==1
    assert result["centerline"].dtype==np.float32
    assert result["valid_mask"].dtype==bool


def test_separate_visible_segments_do_not_fill_a_label_gap():
    value=annotation();value["reviewed"]=[[[-.5,-.5],[31.5,-.5],[31.5,31.5],[-.5,31.5]]];value["ignore"]=[]
    result=rasterize_annotation(reseal(value),sigma_px=1)
    assert result["centerline"][8,8]==result["centerline"][8,22]==1
    assert result["centerline"][8,15]==0
    value["contours"][1]["points"]=[[12,8],[28,8]]
    narrow=rasterize_annotation(reseal(value),sigma_px=1.5)
    assert narrow["centerline"][8,10]==0


def test_explicit_other_ink_region_does_not_override_ignore():
    value=annotation();value["other_ink_regions"]=[[[2,2],[12,2],[12,12],[2,12]]]
    result=rasterize_annotation(reseal(value))
    assert result["other_ink"][3,3]==1
    assert result["other_ink"][8,5]==0
    assert result["other_ink"][3,25]==0


def test_single_user_pilot_approval_is_allowed_but_never_publication():
    validate_annotation(annotation(),require_approved=True)
    bad=annotation();bad["publication_ranking_eligible"]=True
    with pytest.raises(DatasetError):validate_annotation(reseal(bad),require_approved=True)
    with pytest.raises(DatasetError):validate_annotation(annotation(False),require_approved=True)
    bad=annotation();bad["reviewer_id"]="ink-draft-v2"
    with pytest.raises(DatasetError):validate_annotation(reseal(bad),require_approved=True)


def test_revision_tampering_and_nonfinite_coordinates_are_rejected():
    bad=annotation();bad["contours"][0]["points"][0][0]=3
    with pytest.raises(DatasetError,match="revision"):validate_annotation(bad)
    bad=annotation();bad["contours"][0]["points"][0][0]=float("nan")
    with pytest.raises(ValueError):reseal(bad)


def test_evaluation_needs_explicit_fully_reviewed_target():
    value=annotation()
    with pytest.raises(DatasetError,match="explicitly"):evaluation_reference(value)
    value["evaluation"]={"target_contour_id":"left","explicit_target_selection":True,"prompt":{"start_xy":[2,8],"end_xy":[8,8]}}
    with pytest.raises(DatasetError,match="ignored"):evaluation_reference(reseal(value))
    value["ignore"]=[]
    reference=evaluation_reference(reseal(value))
    assert reference["paths"][0]["points"]==[[2,8],[8,8]]
    assert reference["metadata"]["publication_ranking_eligible"] is False
