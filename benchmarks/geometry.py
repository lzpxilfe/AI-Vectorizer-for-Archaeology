"""Canonical ordered-centerline artifacts and deterministic rasterization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
from typing import Any


ARTIFACT_SCHEMA_VERSION = "archaeotrace-centerline/1"
COORDINATE_SPACE = "pixel_xy"
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_CANVAS_DIMENSION = 1024
MAX_CANVAS_PIXELS = 1024 * 1024
MAX_PATHS = 10_000
MAX_POINTS = 100_000
MAX_ORDERED_PIXELS = 100_000
MAX_ABSOLUTE_COORDINATE = 1_000_000_000.0


class CenterlineFormatError(ValueError):
    """Raised when a benchmark centerline artifact is invalid."""


@dataclass(frozen=True)
class CenterlinePath:
    identifier: str
    points: tuple[tuple[float, float], ...]
    closed: bool = False


@dataclass(frozen=True)
class CenterlineArtifact:
    width: int
    height: int
    paths: tuple[CenterlinePath, ...]
    metadata: dict[str, Any]
    source_path: Path
    sha256: str


@dataclass(frozen=True)
class RasterizedPath:
    identifier: str
    pixels: tuple[tuple[int, int], ...]
    closed: bool


@dataclass(frozen=True)
class RasterizedCenterlines:
    width: int
    height: int
    mask: tuple[tuple[bool, ...], ...]
    paths: tuple[RasterizedPath, ...]


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CenterlineFormatError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise CenterlineFormatError(
        f"Non-finite or non-standard JSON number is not allowed: {value}."
    )


def _json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise CenterlineFormatError(f"JSON number must be finite: {value}.")
    return number


def _json_integer(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise CenterlineFormatError("JSON integer exceeds the signed 64-bit limit.")
    number = int(value)
    if abs(number) > (1 << 63) - 1:
        raise CenterlineFormatError("JSON integer exceeds the signed 64-bit limit.")
    return number


def _read_limited_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise CenterlineFormatError(f"Centerline artifact exceeds {limit} bytes.")
    return raw


def _validate_json_numbers(value: Any, label: str) -> None:
    """Reject exponent-overflow floats in every passthrough JSON field."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise CenterlineFormatError(f"{label} must contain only finite numbers.")
        return
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise CenterlineFormatError(
                f"{label} must not contain Unicode surrogate code points."
            )
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise CenterlineFormatError(
                f"{label} must not contain ASCII control characters."
            )
        return
    if value is None or isinstance(value, (int, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_numbers(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_numbers(key, f"{label} key")
            _validate_json_numbers(item, f"{label}.{key}")
        return
    raise CenterlineFormatError(f"{label} contains an unsupported JSON value.")


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CenterlineFormatError(f"{label} must be a positive integer.")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CenterlineFormatError(f"{label} must be a non-empty string.")
    identifier = value.strip()
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in identifier):
        raise CenterlineFormatError(
            f"{label} must not contain ASCII control characters."
        )
    if any(0xD800 <= ord(character) <= 0xDFFF for character in identifier):
        raise CenterlineFormatError(
            f"{label} must not contain Unicode surrogate code points."
        )
    return identifier


def _point(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise CenterlineFormatError(f"{label} must be an [x, y] pair.")
    coordinates = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise CenterlineFormatError(f"{label} coordinates must be numbers.")
        try:
            number = float(coordinate)
        except (OverflowError, ValueError) as exc:
            raise CenterlineFormatError(
                f"{label} coordinates must be finite numbers."
            ) from exc
        if not math.isfinite(number):
            raise CenterlineFormatError(f"{label} coordinates must be finite.")
        if abs(number) > MAX_ABSOLUTE_COORDINATE:
            raise CenterlineFormatError(
                f"{label} coordinates exceed the supported magnitude."
            )
        coordinates.append(number)
    return coordinates[0], coordinates[1]


def load_centerline_artifact(path: str | Path) -> CenterlineArtifact:
    """Load and validate the canonical ordered-centerline JSON format."""

    source_path = Path(path).resolve()
    try:
        raw = _read_limited_bytes(source_path, MAX_ARTIFACT_BYTES)
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_invalid_json_constant,
            parse_float=_json_float,
            parse_int=_json_integer,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CenterlineFormatError(
            f"Could not read centerline artifact {source_path}: {exc}"
        ) from exc

    try:
        _validate_json_numbers(payload, "artifact")
    except RecursionError as exc:
        raise CenterlineFormatError("Centerline artifact nesting is too deep.") from exc

    if not isinstance(payload, dict):
        raise CenterlineFormatError("Centerline artifact root must be an object.")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise CenterlineFormatError(
            f"Unsupported centerline schema: {payload.get('schema_version')!r}."
        )
    if payload.get("coordinate_space") != COORDINATE_SPACE:
        raise CenterlineFormatError(
            f"coordinate_space must be {COORDINATE_SPACE!r}."
        )

    image_size = payload.get("image_size")
    if not isinstance(image_size, dict):
        raise CenterlineFormatError("image_size must be an object.")
    width = _positive_integer(image_size.get("width"), "image_size.width")
    height = _positive_integer(image_size.get("height"), "image_size.height")
    if width > MAX_CANVAS_DIMENSION or height > MAX_CANVAS_DIMENSION:
        raise CenterlineFormatError(
            f"Centerline dimensions may not exceed {MAX_CANVAS_DIMENSION} pixels."
        )
    if width * height > MAX_CANVAS_PIXELS:
        raise CenterlineFormatError(
            f"Centerline canvas exceeds {MAX_CANVAS_PIXELS} pixels."
        )

    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, list):
        raise CenterlineFormatError("paths must be an array.")
    if len(raw_paths) > MAX_PATHS:
        raise CenterlineFormatError(f"paths may contain at most {MAX_PATHS} entries.")
    paths = []
    identifiers = set()
    total_points = 0
    for path_index, raw_path in enumerate(raw_paths):
        label = f"paths[{path_index}]"
        if not isinstance(raw_path, dict):
            raise CenterlineFormatError(f"{label} must be an object.")
        identifier = _identifier(raw_path.get("id"), f"{label}.id")
        if identifier in identifiers:
            raise CenterlineFormatError(f"Duplicate path id: {identifier}.")
        identifiers.add(identifier)

        raw_points = raw_path.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise CenterlineFormatError(f"{label}.points needs at least two points.")
        total_points += len(raw_points)
        if total_points > MAX_POINTS:
            raise CenterlineFormatError(
                f"Centerline artifacts may contain at most {MAX_POINTS} points."
            )
        points = tuple(
            _point(raw_point, f"{label}.points[{point_index}]")
            for point_index, raw_point in enumerate(raw_points)
        )
        closed = raw_path.get("closed", False)
        if not isinstance(closed, bool):
            raise CenterlineFormatError(f"{label}.closed must be boolean.")
        paths.append(CenterlinePath(identifier, points, closed))

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CenterlineFormatError("metadata must be an object when present.")
    return CenterlineArtifact(
        width=width,
        height=height,
        paths=tuple(paths),
        metadata=dict(metadata),
        source_path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _clip_segment(
    first: tuple[float, float],
    second: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a segment to pixel-center bounds using Liang-Barsky."""

    x_min, x_max = -0.5, width - 0.5
    y_min, y_max = -0.5, height - 0.5
    x0, y0 = first
    x1, y1 = second
    dx, dy = x1 - x0, y1 - y0
    lower, upper = 0.0, 1.0

    for p, q in (
        (-dx, x0 - x_min),
        (dx, x_max - x0),
        (-dy, y0 - y_min),
        (dy, y_max - y0),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        ratio = q / p
        if p < 0:
            if ratio > upper:
                return None
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return None
            upper = min(upper, ratio)

    return (
        (x0 + lower * dx, y0 + lower * dy),
        (x0 + upper * dx, y0 + upper * dy),
    )


def _round_pixel(value: float, maximum: int) -> int:
    return min(maximum, max(0, int(math.floor(value + 0.5))))


def _bresenham(
    first: tuple[int, int],
    second: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return inclusive integer pixels for a segment."""

    if second < first:
        return list(reversed(_bresenham(second, first)))

    x0, y0 = first
    x1, y1 = second
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    pixels = []
    while True:
        pixels.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return pixels
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _adjacent(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1])) <= 1


def _rasterize_path(
    path: CenterlinePath,
    width: int,
    height: int,
    pixel_budget: int,
) -> list[RasterizedPath]:
    source_segments = zip(path.points, path.points[1:])
    if path.closed and path.points[-1] != path.points[0]:
        source_segments = chain(
            source_segments,
            ((path.points[-1], path.points[0]),),
        )

    pieces: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    emitted_pixels = 0
    source_was_fully_visible = True
    for first, second in source_segments:
        if not (
            -0.5 <= first[0] <= width - 0.5
            and -0.5 <= first[1] <= height - 0.5
            and -0.5 <= second[0] <= width - 0.5
            and -0.5 <= second[1] <= height - 0.5
        ):
            source_was_fully_visible = False
        clipped = _clip_segment(first, second, width, height)
        if clipped is None:
            source_was_fully_visible = False
            if current:
                pieces.append(current)
                current = []
            continue
        rounded_first = (
            _round_pixel(clipped[0][0], width - 1),
            _round_pixel(clipped[0][1], height - 1),
        )
        rounded_second = (
            _round_pixel(clipped[1][0], width - 1),
            _round_pixel(clipped[1][1], height - 1),
        )
        segment_pixels = _bresenham(rounded_first, rounded_second)
        if current and _adjacent(current[-1], segment_pixels[0]):
            if current[-1] == segment_pixels[0]:
                emitted_pixels += len(segment_pixels) - 1
                current.extend(segment_pixels[1:])
            else:
                emitted_pixels += len(segment_pixels)
                current.extend(segment_pixels)
        else:
            if current:
                pieces.append(current)
            emitted_pixels += len(segment_pixels)
            current = segment_pixels
        if emitted_pixels > pixel_budget:
            raise CenterlineFormatError(
                "Rasterized ordered paths exceed the canvas pixel budget."
            )
    if current:
        pieces.append(current)

    if path.closed and len(pieces) > 1 and _adjacent(pieces[-1][-1], pieces[0][0]):
        merged = pieces[-1] + pieces[0]
        pieces = [merged] + pieces[1:-1]

    rasterized = []
    for index, pixels in enumerate(pieces):
        cleaned = []
        for pixel in pixels:
            if not cleaned or pixel != cleaned[-1]:
                cleaned.append(pixel)
        is_closed = (
            path.closed
            and source_was_fully_visible
            and len(pieces) == 1
            and len(cleaned) > 1
            and _adjacent(cleaned[0], cleaned[-1])
        )
        if is_closed and len(cleaned) > 1 and cleaned[-1] == cleaned[0]:
            cleaned.pop()
        if not cleaned:
            continue
        identifier = path.identifier if len(pieces) == 1 else f"{path.identifier}.part-{index + 1}"
        rasterized.append(RasterizedPath(identifier, tuple(cleaned), is_closed))
    return rasterized


def rasterize_centerlines(artifact: CenterlineArtifact) -> RasterizedCenterlines:
    """Rasterize ordered paths onto their declared canvas without antialiasing."""

    rows = [[False] * artifact.width for _ in range(artifact.height)]
    paths = []
    remaining_pixel_budget = min(
        artifact.width * artifact.height,
        MAX_ORDERED_PIXELS,
    )
    for path in artifact.paths:
        rasterized_paths = _rasterize_path(
            path,
            artifact.width,
            artifact.height,
            remaining_pixel_budget,
        )
        for rasterized_path in rasterized_paths:
            paths.append(rasterized_path)
            remaining_pixel_budget -= len(rasterized_path.pixels)
            for x, y in rasterized_path.pixels:
                rows[y][x] = True
    return RasterizedCenterlines(
        width=artifact.width,
        height=artifact.height,
        mask=tuple(tuple(row) for row in rows),
        paths=tuple(paths),
    )
