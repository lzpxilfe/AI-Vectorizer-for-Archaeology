"""Pure-Python specifications and guards for terrain raster generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat
import uuid


DEFAULT_TARGET_LONG_SIDE = 1024
DEFAULT_MAX_GRID_CELLS = 25_000_000
INTERPOLATION_ITEM_SEPARATOR = "::|::"
INTERPOLATION_VALUE_SEPARATOR = "::~::"
RASTER_SIDECAR_SUFFIXES = (".aux.xml", ".ovr", ".msk")


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
    except (TypeError, ValueError, OverflowError) as exc:
        raise DemSpecificationError(f"{label} must be a number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise DemSpecificationError(f"{label} must be greater than zero.")
    return number


def _positive_integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise DemSpecificationError(f"{label} must be a positive integer.")
    number = _positive_finite(value, label)
    if not number.is_integer():
        raise DemSpecificationError(f"{label} must be a positive integer.")
    return int(number)


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
    max_cells = _positive_integer(max_cells, "Maximum cell count")

    column_ratio = width / pixel_size
    row_ratio = height / pixel_size
    if (
        not math.isfinite(column_ratio)
        or not math.isfinite(row_ratio)
        or column_ratio > max_cells
        or row_ratio > max_cells
    ):
        raise DemSpecificationError(
            f"Requested DEM is too large (limit {max_cells:,} cells)."
        )

    estimate = GridEstimate(
        columns=max(1, math.ceil(column_ratio)),
        rows=max(1, math.ceil(row_ratio)),
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
    target_long_side = _positive_integer(target_long_side, "Target raster size")

    raw = max(width, height) / target_long_side
    if not math.isfinite(raw) or raw <= 0:
        raise DemSpecificationError("Suggested pixel size is outside the supported range.")
    exponent = math.floor(math.log10(raw))
    scale = 10**exponent
    if not scale:
        return raw
    fraction = raw / scale
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    try:
        suggestion = float(nice_fraction * scale)
    except OverflowError:
        return raw
    return suggestion if math.isfinite(suggestion) and suggestion > 0 else raw


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
    try:
        value_source = int(value_source)
        field_index = int(field_index)
        source_type = int(source_type)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DemSpecificationError(
            "Interpolation parameters must be integers."
        ) from exc
    if field_index < 0:
        raise DemSpecificationError("A numeric elevation field is required.")
    return INTERPOLATION_VALUE_SEPARATOR.join(
        (
            layer_reference,
            str(value_source),
            str(field_index),
            str(source_type),
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

    try:
        values = tuple(float(value) for value in (x_min, x_max, y_min, y_max))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DemSpecificationError("DEM extent coordinates must be numbers.") from exc
    if not all(math.isfinite(value) for value in values):
        raise DemSpecificationError("DEM extent contains a non-finite coordinate.")
    x_min, x_max, y_min, y_max = values
    if x_max <= x_min or y_max <= y_min:
        raise DemSpecificationError("DEM extent must have positive width and height.")
    crs_authid = str(crs_authid or "").strip()
    if not crs_authid:
        raise DemSpecificationError("A valid output CRS is required.")
    coordinates = ",".join(f"{value:.15g}" for value in values)
    return f"{coordinates} [{crs_authid}]"


def is_tiff_file(path: str) -> bool:
    """Return whether a file starts with a classic TIFF or BigTIFF signature."""

    descriptor = None
    try:
        initial = os.lstat(path)
        if not stat.S_ISREG(initial.st_mode):
            return False
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(initial, opened):
            return False
        signature = os.read(descriptor, 4)
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
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


def _same_file_snapshot(expected: os.stat_result, current: os.stat_result) -> bool:
    """Return whether a path still names the unchanged preflight file."""

    return (
        os.path.samestat(expected, current)
        and expected.st_mode == current.st_mode
        and expected.st_size == current.st_size
        and getattr(expected, "st_mtime_ns", None)
        == getattr(current, "st_mtime_ns", None)
    )


def publish_output_files(pairs: tuple[tuple[str, str], ...]) -> None:
    """Atomically replace output files and restore prior files on failure."""

    try:
        pairs = tuple(
            (os.fsdecode(os.fspath(work_path)), os.fsdecode(os.fspath(final_path)))
            for work_path, final_path in pairs
        )
    except (TypeError, ValueError) as exc:
        raise DemSpecificationError(
            "Terrain output pairs must contain source and destination paths."
        ) from exc
    if not pairs:
        raise DemSpecificationError("At least one output file is required.")

    source_paths = []
    destination_paths = []
    for work_path, final_path in pairs:
        source_paths.extend(
            (work_path,) + tuple(
                f"{work_path}{suffix}" for suffix in RASTER_SIDECAR_SUFFIXES
            )
        )
        destination_paths.extend(
            (final_path,) + tuple(
                f"{final_path}{suffix}" for suffix in RASTER_SIDECAR_SUFFIXES
            )
        )

    for paths, label in (
        (source_paths, "source"),
        (destination_paths, "destination"),
    ):
        for index, path in enumerate(paths):
            if any(
                paths_refer_to_same_file(path, other_path)
                for other_path in paths[index + 1 :]
            ):
                raise DemSpecificationError(
                    f"Terrain outputs must use distinct {label} files."
                )
    for source_path in source_paths:
        if any(
            paths_refer_to_same_file(source_path, destination_path)
            for destination_path in destination_paths
        ):
            raise DemSpecificationError(
                "Terrain source and destination paths must not overlap."
            )

    source_snapshots = {}
    for work_path, _final_path in pairs:
        try:
            source_information = os.lstat(work_path)
        except FileNotFoundError as exc:
            raise DemSpecificationError(
                f"Could not publish terrain outputs: Processing output is missing: "
                f"{work_path}."
            ) from exc
        except OSError as exc:
            raise DemSpecificationError(
                f"Could not inspect processing output {work_path}: {exc}"
            ) from exc
        if not stat.S_ISREG(source_information.st_mode):
            raise DemSpecificationError(
                f"Processing output must be a regular file: {work_path}"
            )
        source_snapshots[work_path] = source_information

    primary_sources = {work_path for work_path, _final_path in pairs}
    for source_path in source_paths:
        if source_path in primary_sources or not os.path.lexists(source_path):
            continue
        sidecar_information = os.lstat(source_path)
        if not stat.S_ISREG(sidecar_information.st_mode):
            raise DemSpecificationError(
                f"Processing sidecar must be a regular file: {source_path}"
            )
        source_snapshots[source_path] = sidecar_information
    destination_snapshots = {}
    for destination_path in destination_paths:
        if not os.path.lexists(destination_path):
            continue
        destination_information = os.lstat(destination_path)
        if not stat.S_ISREG(destination_information.st_mode):
            raise DemSpecificationError(
                f"Existing terrain output must be a regular file: {destination_path}"
            )
        destination_snapshots[destination_path] = destination_information

    token = uuid.uuid4().hex
    backups = {}
    published = []
    try:
        for work_path, final_path in pairs:
            if not os.path.lexists(work_path):
                raise OSError(f"Processing output is missing: {work_path}")
            destinations = (final_path,) + tuple(
                f"{final_path}{suffix}" for suffix in RASTER_SIDECAR_SUFFIXES
            )
            for existing_path in destinations:
                expected_destination = destination_snapshots.get(existing_path)
                if expected_destination is None:
                    if os.path.lexists(existing_path):
                        raise OSError(
                            f"Destination appeared after preflight: {existing_path}"
                        )
                    continue
                current_destination = os.lstat(existing_path)
                if not _same_file_snapshot(
                    expected_destination,
                    current_destination,
                ):
                    raise OSError(
                        f"Destination changed after preflight: {existing_path}"
                    )
                final = Path(existing_path)
                backup = str(
                    final.with_name(f".{final.name}.archaeotrace-{token}.backup")
                )
                if os.path.lexists(backup):
                    raise OSError(f"Recovery backup path already exists: {backup}")
                os.replace(existing_path, backup)
                backups[existing_path] = backup
                if not _same_file_snapshot(
                    expected_destination,
                    os.lstat(backup),
                ):
                    raise OSError(
                        f"Destination changed while it was backed up: {existing_path}"
                    )
            expected_source = source_snapshots[work_path]
            if not _same_file_snapshot(expected_source, os.lstat(work_path)):
                raise OSError(f"Processing output changed after preflight: {work_path}")
            os.replace(work_path, final_path)
            published.append(final_path)
            if not _same_file_snapshot(expected_source, os.lstat(final_path)):
                raise OSError(f"Processing output changed while publishing: {work_path}")
            for suffix in RASTER_SIDECAR_SUFFIXES:
                work_sidecar = f"{work_path}{suffix}"
                expected_sidecar = source_snapshots.get(work_sidecar)
                if expected_sidecar is None:
                    if os.path.lexists(work_sidecar):
                        raise OSError(
                            f"Processing sidecar appeared after preflight: {work_sidecar}"
                        )
                    continue
                if not _same_file_snapshot(
                    expected_sidecar,
                    os.lstat(work_sidecar),
                ):
                    raise OSError(
                        f"Processing sidecar changed after preflight: {work_sidecar}"
                    )
                final_sidecar = f"{final_path}{suffix}"
                os.replace(work_sidecar, final_sidecar)
                published.append(final_sidecar)
                if not _same_file_snapshot(
                    expected_sidecar,
                    os.lstat(final_sidecar),
                ):
                    raise OSError(
                        f"Processing sidecar changed while publishing: {work_sidecar}"
                    )
    except Exception as exc:
        restore_errors = []
        for final_path in reversed(published):
            try:
                if os.path.lexists(final_path):
                    os.remove(final_path)
            except OSError as restore_exc:
                restore_errors.append(str(restore_exc))
        for final_path, backup in backups.items():
            try:
                if os.path.lexists(backup):
                    os.replace(backup, final_path)
            except OSError as restore_exc:
                restore_errors.append(str(restore_exc))
        detail = ""
        if restore_errors:
            detail = (
                " Could not fully restore old outputs: "
                f"{'; '.join(restore_errors)}"
            )
            recovery_paths = [
                backup
                for backup in backups.values()
                if os.path.lexists(backup)
            ]
            if recovery_paths:
                detail += (
                    " Recovery backups preserved at: "
                    f"{', '.join(recovery_paths)}"
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
