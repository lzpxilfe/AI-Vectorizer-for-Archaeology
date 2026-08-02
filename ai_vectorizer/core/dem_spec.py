"""Pure-Python specifications and guards for terrain raster generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import uuid


DEFAULT_TARGET_LONG_SIDE = 1024
DEFAULT_MAX_GRID_CELLS = 25_000_000
INTERPOLATION_ITEM_SEPARATOR = "::|::"
INTERPOLATION_VALUE_SEPARATOR = "::~::"


class DemSpecificationError(ValueError):
    """Raised when a DEM grid or output specification is unsafe."""


@dataclass(frozen=True)
class GridEstimate:
    """Estimated raster dimensions for a requested extent and pixel size."""

    columns: int
    rows: int

    @property
    def cells(self) -> int:
        return self.columns * self.rows


def _positive_finite(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DemSpecificationError(f"{label} must be a number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise DemSpecificationError(f"{label} must be greater than zero.")
    return number


def estimate_grid(
    width: float,
    height: float,
    pixel_size: float,
    max_cells: int = DEFAULT_MAX_GRID_CELLS,
) -> GridEstimate:
    """Return the output grid size, rejecting grids which are too large."""

    width = _positive_finite(width, "Extent width")
    height = _positive_finite(height, "Extent height")
    pixel_size = _positive_finite(pixel_size, "Pixel size")
    if max_cells <= 0:
        raise DemSpecificationError("Maximum cell count must be greater than zero.")

    estimate = GridEstimate(
        columns=max(1, math.ceil(width / pixel_size)),
        rows=max(1, math.ceil(height / pixel_size)),
    )
    if estimate.cells > max_cells:
        raise DemSpecificationError(
            "Requested DEM is too large: "
            f"{estimate.columns:,} x {estimate.rows:,} "
            f"({estimate.cells:,} cells; limit {max_cells:,})."
        )
    return estimate


def suggest_pixel_size(
    width: float,
    height: float,
    target_long_side: int = DEFAULT_TARGET_LONG_SIDE,
) -> float:
    """Suggest a readable 1/2/5-scaled pixel size for an extent."""

    width = _positive_finite(width, "Extent width")
    height = _positive_finite(height, "Extent height")
    if target_long_side <= 0:
        raise DemSpecificationError("Target raster size must be greater than zero.")

    raw = max(width, height) / target_long_side
    exponent = math.floor(math.log10(raw))
    scale = 10**exponent
    fraction = raw / scale
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return float(nice_fraction * scale)


def normalize_tiff_path(path: str) -> str:
    """Normalize a GeoTIFF path while rejecting unrelated extensions."""

    raw = str(path or "").strip()
    if not raw:
        raise DemSpecificationError("An output GeoTIFF path is required.")

    candidate = Path(raw).expanduser()
    suffix = candidate.suffix.lower()
    if not suffix:
        candidate = candidate.with_suffix(".tif")
    elif suffix not in (".tif", ".tiff"):
        raise DemSpecificationError("Terrain outputs must use .tif or .tiff.")
    return os.path.abspath(str(candidate))


def default_hillshade_path(dem_path: str) -> str:
    """Return a hillshade filename beside a DEM output."""

    normalized = Path(normalize_tiff_path(dem_path))
    return str(normalized.with_name(f"{normalized.stem}_hillshade.tif"))


def interpolation_entry(
    layer_reference: str,
    value_source: int,
    field_index: int,
    source_type: int,
) -> str:
    """Encode one QGIS interpolation input row."""

    layer_reference = str(layer_reference or "").strip()
    if not layer_reference:
        raise DemSpecificationError("Interpolation layer reference is empty.")
    if (
        INTERPOLATION_ITEM_SEPARATOR in layer_reference
        or INTERPOLATION_VALUE_SEPARATOR in layer_reference
    ):
        raise DemSpecificationError("Interpolation layer reference contains a reserved separator.")
    if int(field_index) < 0:
        raise DemSpecificationError("A numeric elevation field is required.")
    return INTERPOLATION_VALUE_SEPARATOR.join(
        (
            layer_reference,
            str(int(value_source)),
            str(int(field_index)),
            str(int(source_type)),
        )
    )


def interpolation_data(entries: list[str] | tuple[str, ...]) -> str:
    """Join one or more encoded QGIS interpolation inputs."""

    cleaned = [str(entry).strip() for entry in entries if str(entry).strip()]
    if not cleaned:
        raise DemSpecificationError("At least one interpolation input is required.")
    return INTERPOLATION_ITEM_SEPARATOR.join(cleaned)


def extent_parameter(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    crs_authid: str,
) -> str:
    """Encode a CRS-aware QGIS Processing extent parameter."""

    values = tuple(float(value) for value in (x_min, x_max, y_min, y_max))
    if not all(math.isfinite(value) for value in values):
        raise DemSpecificationError("DEM extent contains a non-finite coordinate.")
    if x_max <= x_min or y_max <= y_min:
        raise DemSpecificationError("DEM extent must have positive width and height.")
    crs_authid = str(crs_authid or "").strip()
    if not crs_authid:
        raise DemSpecificationError("A valid output CRS is required.")
    coordinates = ",".join(f"{value:.15g}" for value in values)
    return f"{coordinates} [{crs_authid}]"


def is_tiff_file(path: str) -> bool:
    """Return whether a file starts with a classic TIFF or BigTIFF signature."""

    try:
        with open(path, "rb") as handle:
            signature = handle.read(4)
    except OSError:
        return False
    return signature in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def paths_refer_to_same_file(first_path: str, second_path: str) -> bool:
    """Compare output paths safely across aliases and case-insensitive filesystems.

    Existing files use the filesystem's own identity check.  New outputs are
    compared by their physical parent directory and a conservative case-folded
    filename so that e.g. ``Terrain.tif`` and ``terrain.tif`` cannot overwrite
    one another on common macOS and Windows filesystems.
    """

    first = os.path.abspath(os.path.expanduser(str(first_path)))
    second = os.path.abspath(os.path.expanduser(str(second_path)))
    try:
        return os.path.samefile(first, second)
    except OSError:
        pass

    first_parent = os.path.realpath(os.path.dirname(first))
    second_parent = os.path.realpath(os.path.dirname(second))
    try:
        same_parent = os.path.samefile(first_parent, second_parent)
    except OSError:
        same_parent = os.path.normcase(first_parent) == os.path.normcase(second_parent)
    return same_parent and os.path.basename(first).casefold() == os.path.basename(second).casefold()


def publish_output_files(pairs: tuple[tuple[str, str], ...]) -> None:
    """Atomically replace output files and restore prior files on failure."""

    if not pairs:
        raise DemSpecificationError("At least one output file is required.")
    final_paths = [final_path for _, final_path in pairs]
    for index, final_path in enumerate(final_paths):
        if any(
            paths_refer_to_same_file(final_path, other_path)
            for other_path in final_paths[index + 1 :]
        ):
            raise DemSpecificationError(
                "Terrain outputs must use distinct destination files."
            )
    token = uuid.uuid4().hex
    backups = {}
    published = []
    try:
        for work_path, final_path in pairs:
            if not os.path.exists(work_path):
                raise OSError(f"Processing output is missing: {work_path}")
            for existing_path in (
                final_path,
                f"{final_path}.aux.xml",
                f"{final_path}.ovr",
            ):
                if not os.path.exists(existing_path):
                    continue
                final = Path(existing_path)
                backup = str(
                    final.with_name(f".{final.name}.archaeotrace-{token}.backup")
                )
                os.replace(existing_path, backup)
                backups[existing_path] = backup
            os.replace(work_path, final_path)
            published.append(final_path)
    except Exception as exc:
        restore_errors = []
        for final_path in reversed(published):
            try:
                if os.path.exists(final_path):
                    os.remove(final_path)
            except OSError as restore_exc:
                restore_errors.append(str(restore_exc))
        for final_path, backup in backups.items():
            try:
                if os.path.exists(backup):
                    os.replace(backup, final_path)
            except OSError as restore_exc:
                restore_errors.append(str(restore_exc))
        detail = ""
        if restore_errors:
            detail = (
                " Could not fully restore old outputs: "
                f"{'; '.join(restore_errors)}"
            )
        raise DemSpecificationError(
            f"Could not publish terrain outputs: {exc}.{detail}"
        ) from exc
    else:
        for backup in backups.values():
            try:
                os.remove(backup)
            except OSError:
                pass
