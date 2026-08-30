"""Offline verification for locally staged public-benchmark source crops.

The plan validator proves that declared files and hashes are internally
consistent without depending on an image codec.  This optional Pillow-backed
pass adds the provenance check that a checked-in lossless PNG is pixel-for-
pixel equal to the declared rectangle in its immutable source raster.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .geometry import CenterlineArtifact, CenterlinePath, load_centerline_artifact
from .public_dataset import (
    is_independently_accepted_annotation,
    validate_public_dataset_plan,
)


class PublicAssetVerificationError(ValueError):
    """Raised when staged public assets violate their declared evidence."""


@dataclass(frozen=True)
class PublicAssetVerificationReport:
    path: Path
    staged_sheet_count: int
    verified_crop_count: int
    draft_reference_count: int


def _pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency-free QGIS path
        raise PublicAssetVerificationError(
            "Pillow is required for source-raster pixel verification"
        ) from exc
    return Image


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise PublicAssetVerificationError(f"{label} must be an [x, y] pair")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise PublicAssetVerificationError(f"{label} must be numeric") from exc


def verify_lossless_source_crop(
    source_path: str | Path,
    crop_path: str | Path,
    source_crop_xywh: Sequence[int],
) -> None:
    """Require a PNG to contain exactly the declared source-raster pixels."""

    if (
        isinstance(source_crop_xywh, (str, bytes))
        or not isinstance(source_crop_xywh, Sequence)
        or len(source_crop_xywh) != 4
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in source_crop_xywh
        )
    ):
        raise PublicAssetVerificationError(
            "source_crop_xywh must be four integer values"
        )
    x, y, width, height = source_crop_xywh
    if x < 0 or y < 0 or width < 1 or height < 1:
        raise PublicAssetVerificationError(
            "source_crop_xywh must describe a positive bounded rectangle"
        )

    Image = _pillow_image()
    try:
        with Image.open(Path(source_path)) as source:
            if x + width > source.width or y + height > source.height:
                raise PublicAssetVerificationError(
                    "source_crop_xywh extends beyond the source raster"
                )
            expected = source.crop((x, y, x + width, y + height)).convert("RGBA")
        with Image.open(Path(crop_path)) as crop:
            if crop.format != "PNG":
                raise PublicAssetVerificationError("staged crop must be a PNG")
            actual = crop.convert("RGBA")
    except PublicAssetVerificationError:
        raise
    except (OSError, ValueError) as exc:
        raise PublicAssetVerificationError(
            f"could not decode staged source/crop images: {exc}"
        ) from exc

    if expected.size != actual.size or expected.tobytes() != actual.tobytes():
        raise PublicAssetVerificationError(
            "staged PNG pixels do not equal the declared source-raster crop"
        )


def _load_plan(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicAssetVerificationError(
            f"could not read dataset plan: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PublicAssetVerificationError("dataset plan root must be an object")
    return payload


def _same_point(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return (
        abs(first[0] - second[0]) <= 1e-9
        and abs(first[1] - second[1]) <= 1e-9
    )


def _prompted_open_reference_path(
    reference: CenterlineArtifact,
    label: str,
) -> CenterlinePath:
    if len(reference.paths) != 1:
        raise PublicAssetVerificationError(
            f"{label} ordered reference must contain exactly one prompted path"
        )
    path_record = reference.paths[0]
    if path_record.closed:
        raise PublicAssetVerificationError(
            f"{label} ordered reference path must be open"
        )
    if not any(
        math.hypot(second[0] - first[0], second[1] - first[1]) > 0.0
        for first, second in zip(path_record.points, path_record.points[1:])
    ):
        raise PublicAssetVerificationError(
            f"{label} ordered reference path must have positive geometric length"
        )
    for point_index, (x, y) in enumerate(path_record.points):
        if not 0.0 <= x < reference.width or not 0.0 <= y < reference.height:
            raise PublicAssetVerificationError(
                f"{label} ordered reference point {point_index} lies outside the crop"
            )
    return path_record


def verify_public_dataset_assets(
    path: str | Path,
) -> PublicAssetVerificationReport:
    """Verify every locally staged sheet without performing network I/O."""

    plan_path = Path(path).resolve()
    # This verifies every path/hash and all structural constraints before the
    # payload below is allowed to name local files.
    validate_public_dataset_plan(plan_path)
    payload = _load_plan(plan_path)
    root = plan_path.parent
    staged_sheet_count = 0
    verified_crop_count = 0
    draft_reference_count = 0

    for sheet_index, sheet in enumerate(payload["sheets"]):
        source_value = sheet["source"]["source_raster"]
        staged_crops = [
            (crop_index, crop)
            for crop_index, crop in enumerate(sheet["crops"])
            if crop["image"] is not None
        ]
        if source_value is None:
            if staged_crops:
                raise PublicAssetVerificationError(
                    f"sheets[{sheet_index}] has staged crops without a source raster"
                )
            continue

        source_path = root / source_value
        for crop_index, crop in staged_crops:
            label = f"sheets[{sheet_index}].crops[{crop_index}]"
            verify_lossless_source_crop(
                source_path,
                root / crop["image"],
                crop["source_crop_xywh"],
            )
            reference = load_centerline_artifact(root / crop["ordered_reference"])
            path_record = _prompted_open_reference_path(reference, label)
            start = _point(crop["prompt"]["start_xy"], f"{label}.prompt.start_xy")
            end = _point(crop["prompt"]["end_xy"], f"{label}.prompt.end_xy")
            if _same_point(start, end):
                raise PublicAssetVerificationError(
                    f"{label} prompt start and end must differ"
                )
            if not (
                _same_point(path_record.points[0], start)
                and _same_point(path_record.points[-1], end)
            ):
                raise PublicAssetVerificationError(
                    f"{label} ordered reference must run from prompt start to end"
                )

            annotation = crop["annotation"]
            independently_accepted = is_independently_accepted_annotation(
                annotation
            )
            if not independently_accepted:
                if reference.metadata.get("annotation_status") != "draft_unreviewed":
                    raise PublicAssetVerificationError(
                        f"{label} unreviewed reference must identify itself as a draft"
                    )
                draft_reference_count += 1
            verified_crop_count += 1

        if len(staged_crops) == len(sheet["crops"]):
            staged_sheet_count += 1

    return PublicAssetVerificationReport(
        path=plan_path,
        staged_sheet_count=staged_sheet_count,
        verified_crop_count=verified_crop_count,
        draft_reference_count=draft_reference_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify checked-in public benchmark crops against source rasters."
    )
    parser.add_argument("plan", type=Path)
    arguments = parser.parse_args(argv)
    print(verify_public_dataset_assets(arguments.plan))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a command
    raise SystemExit(main())


__all__ = [
    "PublicAssetVerificationError",
    "PublicAssetVerificationReport",
    "verify_lossless_source_crop",
    "verify_public_dataset_assets",
]
