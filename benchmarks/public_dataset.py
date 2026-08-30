"""Offline validator for the planned 8-sheet/48-crop public benchmark.

This module performs no network I/O.  The checked-in template reserves a
balanced, sheet-isolated evaluation plan with unresolved fields.  A later
materialization pass must provide rights snapshots, raster/reference bytes,
prompts, crop coordinates, and completed human review before the same plan can
pass ``require_materialized=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .evidence import PROMPT_EVIDENCE_SCHEMA_VERSION, canonical_prompt
from .geometry import CenterlineFormatError, load_centerline_artifact
from .manifest import ManifestError, image_dimensions


PUBLIC_DATASET_SCHEMA_VERSION = "archaeotrace-public-dataset-plan/1"
PUBLIC_DATASET_SHEET_COUNT = 8
PUBLIC_DATASET_CROPS_PER_SHEET = 6
PUBLIC_DATASET_CROP_COUNT = 48
PUBLIC_DATASET_SPLITS = ("calibration", "holdout")
PUBLIC_DATASET_ORIGIN_GROUPS = ("usgs_htmc", "korea_rights_cleared")
PUBLIC_DATASET_STRATA = (
    "clean_dark_curve",
    "thick_or_scale",
    "faded_or_broken",
    "colored_line",
    "text_or_number_crossing",
    "dense_parallel",
    "stain_fold_bleed",
    "straight_grid_distractor",
)
_RIGHTS_STATUSES = frozenset({"unresolved", "public_domain", "open_license"})
_MATERIALIZED_RIGHTS = frozenset({"public_domain", "open_license"})
_MAX_PLAN_BYTES = 4 * 1024 * 1024


class PublicDatasetError(ValueError):
    """Raised when a dataset plan is ambiguous or unverifiable."""


@dataclass(frozen=True)
class PublicDatasetReport:
    path: Path
    dataset_id: str
    sheet_count: int
    crop_count: int
    split_sheet_counts: dict[str, int]
    split_crop_counts: dict[str, int]
    origin_group_counts: dict[str, dict[str, int]]
    stratum_counts: dict[str, dict[str, int]]
    materialized: bool
    publication_ranking_eligible: bool


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PublicDatasetError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise PublicDatasetError(f"non-standard JSON number is forbidden: {value}")


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicDatasetError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PublicDatasetError(f"{label} must be an array")
    return value


def _text(value: Any, label: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PublicDatasetError(f"{label} must be a non-empty string")
    return value.strip()


def _url(value: Any, label: str, *, required: bool) -> str | None:
    text = _text(value, label, required=required)
    if text is None:
        return None
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PublicDatasetError(f"{label} must be an absolute HTTP(S) URL")
    return text


def _sha256(value: Any, label: str, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicDatasetError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(root: Path, value: Any, digest: Any, label: str) -> Path:
    relative = Path(_text(value, label) or "")
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise PublicDatasetError(f"{label} must be a safe relative path")
    expected = _sha256(digest, f"{label}_sha256", required=True)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PublicDatasetError(f"{label} must not traverse a symbolic link")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PublicDatasetError(f"{label} escapes the plan directory") from exc
    if not resolved.is_file():
        raise PublicDatasetError(f"{label} does not exist: {relative.as_posix()}")
    actual = _file_sha256(resolved)
    if actual != expected:
        raise PublicDatasetError(
            f"{label} checksum mismatch: expected {expected}, got {actual}"
        )
    return resolved


def _load(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PublicDatasetError(f"could not read dataset plan: {exc}") from exc
    if len(raw) > _MAX_PLAN_BYTES:
        raise PublicDatasetError(f"dataset plan exceeds {_MAX_PLAN_BYTES} bytes")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PublicDatasetError(f"invalid dataset plan JSON: {exc}") from exc
    return _object(value, "dataset plan")


def _point(value: Any, width: int, height: int, label: str) -> tuple[float, float]:
    point = _array(value, label)
    if len(point) != 2:
        raise PublicDatasetError(f"{label} must contain x and y")
    numbers: list[float] = []
    for index, item in enumerate(point):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PublicDatasetError(f"{label}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise PublicDatasetError(f"{label}[{index}] must be finite")
        numbers.append(number)
    if not 0.0 <= numbers[0] < width or not 0.0 <= numbers[1] < height:
        raise PublicDatasetError(f"{label} lies outside the crop")
    return numbers[0], numbers[1]


def _validate_prompt(value: Any, width: int, height: int, label: str) -> None:
    prompt = _object(value, label)
    required_fields = {
        "schema_version",
        "start_xy",
        "end_xy",
        "positive_xy",
        "negative_xy",
    }
    allowed_fields = {*required_fields, "previous_xy"}
    if not required_fields.issubset(prompt) or not set(prompt).issubset(
        allowed_fields
    ):
        raise PublicDatasetError(f"{label} has unsupported or missing fields")
    if prompt.get("schema_version") != PROMPT_EVIDENCE_SCHEMA_VERSION:
        raise PublicDatasetError(
            f"{label}.schema_version must be {PROMPT_EVIDENCE_SCHEMA_VERSION!r}"
        )
    _point(prompt.get("start_xy"), width, height, f"{label}.start_xy")
    _point(prompt.get("end_xy"), width, height, f"{label}.end_xy")
    previous = prompt.get("previous_xy")
    if previous is not None:
        _point(previous, width, height, f"{label}.previous_xy")
    guides = 0
    for name in ("positive_xy", "negative_xy"):
        values = _array(prompt.get(name), f"{label}.{name}")
        guides += len(values)
        for index, point in enumerate(values):
            _point(point, width, height, f"{label}.{name}[{index}]")
    if guides > 4:
        raise PublicDatasetError(f"{label} may contain at most four guide points")
    try:
        canonical_prompt(
            prompt,
            schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION,
        )
    except ValueError as exc:
        raise PublicDatasetError(f"{label} is not a valid prompt-v2 record: {exc}") from exc


def validate_public_dataset_plan(
    path: str | Path,
    *,
    require_materialized: bool = False,
) -> PublicDatasetReport:
    """Validate split isolation, fixed strata, provenance, and local evidence."""

    plan_path = Path(path).resolve()
    root = plan_path.parent.resolve()
    plan = _load(plan_path)
    required_top = {
        "schema_version",
        "dataset_id",
        "title",
        "publication_ranking_eligible",
        "threshold_status",
        "split_policy",
        "difficulty_strata",
        "sheets",
    }
    if set(plan) != required_top:
        raise PublicDatasetError("dataset plan has unsupported or missing top-level fields")
    if plan.get("schema_version") != PUBLIC_DATASET_SCHEMA_VERSION:
        raise PublicDatasetError("unsupported public dataset plan schema")
    dataset_id = _text(plan.get("dataset_id"), "dataset_id") or ""
    _text(plan.get("title"), "title")
    if plan.get("publication_ranking_eligible") is not False:
        raise PublicDatasetError(
            "publication_ranking_eligible must remain false before public calibration"
        )
    if plan.get("threshold_status") != "provisional":
        raise PublicDatasetError("threshold_status must remain 'provisional'")
    if _array(plan.get("difficulty_strata"), "difficulty_strata") != list(
        PUBLIC_DATASET_STRATA
    ):
        raise PublicDatasetError("difficulty_strata must match the fixed v1 order")

    policy = _object(plan.get("split_policy"), "split_policy")
    if set(policy) != {
        "assignment_unit",
        "calibration_sheet_ids",
        "holdout_sheet_ids",
        "selection_seed",
        "frozen_holdout",
    }:
        raise PublicDatasetError("split_policy has unsupported or missing fields")
    if policy.get("assignment_unit") != "sheet" or policy.get("frozen_holdout") is not True:
        raise PublicDatasetError("split assignment must be sheet-level with a frozen holdout")
    _text(policy.get("selection_seed"), "split_policy.selection_seed")
    declared_ids: dict[str, tuple[str, ...]] = {}
    for split in PUBLIC_DATASET_SPLITS:
        declared_ids[split] = tuple(
            _text(value, f"split_policy.{split}_sheet_ids") or ""
            for value in _array(
                policy.get(f"{split}_sheet_ids"),
                f"split_policy.{split}_sheet_ids",
            )
        )
        if len(declared_ids[split]) != 4 or len(set(declared_ids[split])) != 4:
            raise PublicDatasetError(f"{split} must declare four unique sheet ids")
    if set(declared_ids["calibration"]) & set(declared_ids["holdout"]):
        raise PublicDatasetError("calibration and holdout sheet ids overlap")

    sheets = _array(plan.get("sheets"), "sheets")
    if len(sheets) != PUBLIC_DATASET_SHEET_COUNT:
        raise PublicDatasetError("dataset plan must contain exactly eight sheets")
    sheet_ids: set[str] = set()
    source_ids: set[str] = set()
    crop_ids: set[str] = set()
    materialized = True
    actual_ids = {split: set() for split in PUBLIC_DATASET_SPLITS}
    split_sheet_counts = {split: 0 for split in PUBLIC_DATASET_SPLITS}
    split_crop_counts = {split: 0 for split in PUBLIC_DATASET_SPLITS}
    origin_group_counts = {
        split: {group: 0 for group in PUBLIC_DATASET_ORIGIN_GROUPS}
        for split in PUBLIC_DATASET_SPLITS
    }
    stratum_counts = {
        split: {stratum: 0 for stratum in PUBLIC_DATASET_STRATA}
        for split in PUBLIC_DATASET_SPLITS
    }
    source_urls_by_split = {split: set() for split in PUBLIC_DATASET_SPLITS}
    source_raster_hashes_by_split = {
        split: set() for split in PUBLIC_DATASET_SPLITS
    }

    for sheet_index, raw_sheet in enumerate(sheets):
        label = f"sheets[{sheet_index}]"
        sheet = _object(raw_sheet, label)
        if set(sheet) != {
            "id",
            "source_id",
            "split",
            "origin_group",
            "source",
            "crops",
        }:
            raise PublicDatasetError(f"{label} has unsupported or missing fields")
        sheet_id = _text(sheet.get("id"), f"{label}.id") or ""
        source_id = _text(sheet.get("source_id"), f"{label}.source_id") or ""
        split = sheet.get("split")
        origin_group = sheet.get("origin_group")
        if split not in PUBLIC_DATASET_SPLITS:
            raise PublicDatasetError(f"{label}.split is invalid")
        if origin_group not in PUBLIC_DATASET_ORIGIN_GROUPS:
            raise PublicDatasetError(f"{label}.origin_group is invalid")
        if sheet_id in sheet_ids or source_id in source_ids:
            raise PublicDatasetError("sheet ids and source ids must be globally unique")
        sheet_ids.add(sheet_id)
        source_ids.add(source_id)
        actual_ids[split].add(sheet_id)
        split_sheet_counts[split] += 1
        origin_group_counts[split][origin_group] += 1

        source = _object(sheet.get("source"), f"{label}.source")
        source_keys = {
            "title",
            "publisher",
            "date_or_sheet",
            "source_url",
            "license",
            "rights_status",
            "rights_statement_url",
            "text_snapshot",
            "text_snapshot_sha256",
            "source_raster",
            "source_raster_sha256",
        }
        if set(source) != source_keys:
            raise PublicDatasetError(f"{label}.source has unsupported or missing fields")
        _text(source.get("title"), f"{label}.source.title")
        rights = source.get("rights_status")
        if rights not in _RIGHTS_STATUSES:
            raise PublicDatasetError(f"{label}.source.rights_status is invalid")
        is_open = rights in _MATERIALIZED_RIGHTS
        for field in ("publisher", "date_or_sheet", "license"):
            _text(source.get(field), f"{label}.source.{field}", required=is_open)
        source_url = _url(
            source.get("source_url"), f"{label}.source.source_url", required=is_open
        )
        _url(
            source.get("rights_statement_url"),
            f"{label}.source.rights_statement_url",
            required=is_open,
        )
        if source_url is not None:
            source_urls_by_split[split].add(source_url)
        if not is_open:
            materialized = False
        source_assets = (
            source.get("text_snapshot"),
            source.get("text_snapshot_sha256"),
            source.get("source_raster"),
            source.get("source_raster_sha256"),
        )
        if all(value is None for value in source_assets):
            materialized = False
        elif any(value is None for value in source_assets):
            raise PublicDatasetError(f"{label}.source asset paths/hashes are incomplete")
        else:
            _verified_file(
                root,
                source["text_snapshot"],
                source["text_snapshot_sha256"],
                f"{label}.source.text_snapshot",
            )
            _verified_file(
                root,
                source["source_raster"],
                source["source_raster_sha256"],
                f"{label}.source.source_raster",
            )
            raster_digest = source["source_raster_sha256"]
            if any(
                raster_digest in hashes
                for hashes in source_raster_hashes_by_split.values()
            ):
                raise PublicDatasetError(
                    "source_raster_sha256 must identify one unique source sheet"
                )
            source_raster_hashes_by_split[split].add(raster_digest)

        crops = _array(sheet.get("crops"), f"{label}.crops")
        if len(crops) != PUBLIC_DATASET_CROPS_PER_SHEET:
            raise PublicDatasetError(f"{label} must reserve exactly six crops")
        slots: set[int] = set()
        for crop_index, raw_crop in enumerate(crops):
            crop_label = f"{label}.crops[{crop_index}]"
            crop = _object(raw_crop, crop_label)
            crop_keys = {
                "id",
                "slot",
                "difficulty_stratum",
                "source_crop_xywh",
                "source_tile_origin_xy",
                "image",
                "image_sha256",
                "prompt",
                "ordered_reference",
                "ordered_reference_sha256",
                "annotation",
                "notes",
            }
            if set(crop) != crop_keys:
                raise PublicDatasetError(f"{crop_label} has unsupported or missing fields")
            crop_id = _text(crop.get("id"), f"{crop_label}.id") or ""
            slot = crop.get("slot")
            stratum = crop.get("difficulty_stratum")
            if isinstance(slot, bool) or not isinstance(slot, int) or not 1 <= slot <= 6:
                raise PublicDatasetError(f"{crop_label}.slot must be 1..6")
            if crop_id in crop_ids or slot in slots:
                raise PublicDatasetError("crop ids and per-sheet slots must be unique")
            if stratum not in PUBLIC_DATASET_STRATA:
                raise PublicDatasetError(f"{crop_label}.difficulty_stratum is invalid")
            crop_ids.add(crop_id)
            slots.add(slot)
            stratum_counts[split][stratum] += 1
            _text(crop.get("notes"), f"{crop_label}.notes", required=False)

            xywh = crop.get("source_crop_xywh")
            source_tile_origin = crop.get("source_tile_origin_xy")
            width = height = None
            if xywh is None:
                materialized = False
            else:
                values = _array(xywh, f"{crop_label}.source_crop_xywh")
                if (
                    len(values) != 4
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
                    or values[0] < 0
                    or values[1] < 0
                    or values[2] < 1
                    or values[3] < 1
                ):
                    raise PublicDatasetError(
                        f"{crop_label}.source_crop_xywh must be [x,y,width,height] integers"
                    )
                width, height = values[2], values[3]

            if source_tile_origin is None:
                materialized = False
            else:
                origin_values = _array(
                    source_tile_origin,
                    f"{crop_label}.source_tile_origin_xy",
                )
                if (
                    len(origin_values) != 2
                    or any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in origin_values
                    )
                    or any(value < 0 for value in origin_values)
                ):
                    raise PublicDatasetError(
                        f"{crop_label}.source_tile_origin_xy must be "
                        "non-negative [x,y] integers"
                    )
                if xywh is not None and origin_values != values[:2]:
                    raise PublicDatasetError(
                        f"{crop_label}.source_tile_origin_xy must equal the "
                        "source_crop_xywh origin"
                    )

            crop_assets = (
                crop.get("image"),
                crop.get("image_sha256"),
                crop.get("ordered_reference"),
                crop.get("ordered_reference_sha256"),
            )
            if all(value is None for value in crop_assets):
                materialized = False
            elif any(value is None for value in crop_assets):
                raise PublicDatasetError(f"{crop_label} asset paths/hashes are incomplete")
            else:
                image_path = _verified_file(
                    root, crop["image"], crop["image_sha256"], f"{crop_label}.image"
                )
                reference_path = _verified_file(
                    root,
                    crop["ordered_reference"],
                    crop["ordered_reference_sha256"],
                    f"{crop_label}.ordered_reference",
                )
                if width is None or height is None:
                    raise PublicDatasetError(
                        f"{crop_label} assets require source_crop_xywh"
                    )
                if image_path.suffix.lower() != ".png":
                    raise PublicDatasetError(
                        f"{crop_label}.image must be a lossless .png file"
                    )
                try:
                    observed_dimensions = image_dimensions(image_path)
                except (OSError, ManifestError, ValueError) as exc:
                    raise PublicDatasetError(
                        f"{crop_label}.image is not a valid bounded PNG: {exc}"
                    ) from exc
                if observed_dimensions != (width, height):
                    raise PublicDatasetError(
                        f"{crop_label}.image dimensions must match source_crop_xywh"
                    )
                try:
                    reference = load_centerline_artifact(reference_path)
                except (OSError, CenterlineFormatError, ValueError) as exc:
                    raise PublicDatasetError(
                        f"{crop_label}.ordered_reference is invalid: {exc}"
                    ) from exc
                if (reference.width, reference.height) != (width, height):
                    raise PublicDatasetError(
                        f"{crop_label}.ordered_reference image_size must match the crop"
                    )
                if not reference.paths:
                    raise PublicDatasetError(
                        f"{crop_label}.ordered_reference must contain an ordered centerline"
                    )

            prompt = crop.get("prompt")
            if prompt is None:
                materialized = False
            elif width is None or height is None:
                raise PublicDatasetError(f"{crop_label}.prompt requires source_crop_xywh")
            else:
                _validate_prompt(prompt, width, height, f"{crop_label}.prompt")

            annotation = _object(crop.get("annotation"), f"{crop_label}.annotation")
            if set(annotation) != {
                "reviewer_id",
                "review_status",
                "adjudicator_id",
                "adjudication_status",
            }:
                raise PublicDatasetError(f"{crop_label}.annotation fields are invalid")
            reviewer = _text(
                annotation.get("reviewer_id"),
                f"{crop_label}.annotation.reviewer_id",
                required=False,
            )
            adjudicator = _text(
                annotation.get("adjudicator_id"),
                f"{crop_label}.annotation.adjudicator_id",
                required=False,
            )
            review_status = annotation.get("review_status")
            adjudication_status = annotation.get("adjudication_status")
            if review_status not in {"unreviewed", "approved", "rejected"}:
                raise PublicDatasetError(f"{crop_label}.annotation.review_status is invalid")
            if adjudication_status not in {"pending", "accepted", "rejected"}:
                raise PublicDatasetError(
                    f"{crop_label}.annotation.adjudication_status is invalid"
                )
            if reviewer is not None and reviewer == adjudicator:
                raise PublicDatasetError(
                    f"{crop_label} reviewer_id and adjudicator_id must be independent"
                )
            if not (
                reviewer
                and adjudicator
                and review_status == "approved"
                and adjudication_status == "accepted"
            ):
                materialized = False
            split_crop_counts[split] += 1

    if actual_ids != {split: set(ids) for split, ids in declared_ids.items()}:
        raise PublicDatasetError("split_policy sheet ids do not match the sheet records")
    if source_urls_by_split["calibration"] & source_urls_by_split["holdout"]:
        raise PublicDatasetError("a source URL appears in both calibration and holdout")
    if (
        source_raster_hashes_by_split["calibration"]
        & source_raster_hashes_by_split["holdout"]
    ):
        raise PublicDatasetError(
            "a source raster hash appears in both calibration and holdout"
        )
    for split in PUBLIC_DATASET_SPLITS:
        if split_sheet_counts[split] != 4 or split_crop_counts[split] != 24:
            raise PublicDatasetError(f"{split} must contain four sheets and 24 crops")
        if origin_group_counts[split] != {group: 2 for group in PUBLIC_DATASET_ORIGIN_GROUPS}:
            raise PublicDatasetError(f"{split} must contain two sheets from each origin group")
        for stratum, count in stratum_counts[split].items():
            if count < 3:
                raise PublicDatasetError(
                    f"{split} difficulty stratum {stratum!r} needs at least three crops"
                )
    total_strata = {
        stratum: sum(stratum_counts[split][stratum] for split in PUBLIC_DATASET_SPLITS)
        for stratum in PUBLIC_DATASET_STRATA
    }
    if any(count != 6 for count in total_strata.values()):
        raise PublicDatasetError("each difficulty stratum must contain exactly six crops")
    if len(sheet_ids) != 8 or len(source_ids) != 8 or len(crop_ids) != 48:
        raise PublicDatasetError("dataset must contain 8 unique sources and 48 unique crops")
    if require_materialized and not materialized:
        raise PublicDatasetError(
            "plan is structurally valid but rights/assets/prompts/review remain unresolved"
        )
    return PublicDatasetReport(
        path=plan_path,
        dataset_id=dataset_id,
        sheet_count=8,
        crop_count=48,
        split_sheet_counts=split_sheet_counts,
        split_crop_counts=split_crop_counts,
        origin_group_counts=origin_group_counts,
        stratum_counts=stratum_counts,
        materialized=materialized,
        publication_ranking_eligible=False,
    )


__all__ = [
    "PUBLIC_DATASET_CROP_COUNT",
    "PUBLIC_DATASET_CROPS_PER_SHEET",
    "PUBLIC_DATASET_ORIGIN_GROUPS",
    "PUBLIC_DATASET_SCHEMA_VERSION",
    "PUBLIC_DATASET_SHEET_COUNT",
    "PUBLIC_DATASET_SPLITS",
    "PUBLIC_DATASET_STRATA",
    "PublicDatasetError",
    "PublicDatasetReport",
    "validate_public_dataset_plan",
]
