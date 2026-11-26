"""Named tracing presets, independent of QGIS.

A preset records the dock settings that change from one map series to
another, so a user who works through several sheets printed in the same
style does not have to rediscover them. It holds no image data: the ink
colour is only the sampled RGB triple.

Presets are versioned JSON (``archaeotrace-trace-preset/1``). Reading is
strict. A malformed preset, an unknown field or a newer schema version is
rejected with a :class:`PresetError` that names the problem; nothing is
guessed or silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .ink_color import InkColorProfile


SCHEMA_VERSION = "archaeotrace-trace-preset/1"
LIBRARY_SCHEMA_VERSION = "archaeotrace-trace-preset-library/1"
_SCHEMA_PREFIX = "archaeotrace-trace-preset/"
_LIBRARY_PREFIX = "archaeotrace-trace-preset-library/"

METHODS = ("ink", "lsd", "hed", "mobile_sam", "sam", "legacy_canny")
SAM_METHODS = ("mobile_sam", "sam")
OPENCV_METHODS = ("lsd", "hed")

MAX_NAME_LENGTH = 80
MAX_CONTOUR_INTERVAL = 10000.0
MAX_LIBRARY_SIZE = 200

_FIELDS = (
    "schema_version",
    "name",
    "method",
    "assist_strength",
    "smart_recovery",
    "freehand",
    "auto_path",
    "learned_guide",
    "ink_color_rgb",
    "contour_interval",
)
_REQUIRED = (
    "schema_version",
    "name",
    "method",
    "assist_strength",
    "smart_recovery",
    "freehand",
    "auto_path",
    "learned_guide",
)


class PresetError(ValueError):
    """A preset that cannot be read or stored as written.

    ``code`` is stable and is what the UI translates; ``detail`` names the
    field or value involved.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def normalize_name(name: Any) -> str:
    """Return the stored form of a preset name, or raise ``PresetError``."""

    if not isinstance(name, str):
        raise PresetError("invalid_name", "name must be text")
    text = " ".join(name.split())
    if not text:
        raise PresetError("invalid_name", "name is empty")
    if len(text) > MAX_NAME_LENGTH:
        raise PresetError("invalid_name", f"name is longer than {MAX_NAME_LENGTH} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise PresetError("invalid_name", "name contains control characters")
    return text


def name_key(name: str) -> str:
    """Key used to compare names: case and surrounding spaces do not count."""

    return normalize_name(name).casefold()


def _strict_bool(payload: Mapping[str, Any], field: str) -> bool:
    value = payload[field]
    if not isinstance(value, bool):
        raise PresetError("invalid_field", f"{field} must be true or false")
    return value


@dataclass(frozen=True)
class TracePreset:
    """Dock settings for one map series."""

    name: str
    method: str = "ink"
    assist_strength: int = 30
    smart_recovery: bool = False
    freehand: bool = False
    auto_path: bool = False
    learned_guide: bool = False
    ink_color_rgb: Optional[Tuple[int, int, int]] = None
    contour_interval: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", normalize_name(self.name))
        if self.method not in METHODS:
            raise PresetError("invalid_field", f"method must be one of {', '.join(METHODS)}")
        strength = self.assist_strength
        if isinstance(strength, bool) or not isinstance(strength, int) or not 0 <= strength <= 100:
            raise PresetError("invalid_field", "assist_strength must be an integer from 0 to 100")
        for field in ("smart_recovery", "freehand", "auto_path", "learned_guide"):
            if not isinstance(getattr(self, field), bool):
                raise PresetError("invalid_field", f"{field} must be true or false")
        if self.ink_color_rgb is not None:
            rgb = self.ink_color_rgb
            if (
                isinstance(rgb, (str, bytes))
                or not isinstance(rgb, Sequence)
                or len(rgb) != 3
                or any(isinstance(v, bool) or not isinstance(v, int) for v in rgb)
            ):
                raise PresetError("invalid_field", "ink_color_rgb must be three integers")
            try:
                InkColorProfile(tuple(rgb))
            except ValueError as exc:
                raise PresetError("invalid_field", f"ink_color_rgb: {exc}") from exc
            object.__setattr__(self, "ink_color_rgb", tuple(int(v) for v in rgb))
        interval = self.contour_interval
        if interval is not None:
            if isinstance(interval, bool) or not isinstance(interval, (int, float)):
                raise PresetError("invalid_field", "contour_interval must be a number")
            interval = float(interval)
            if not math.isfinite(interval) or not 0.0 < interval <= MAX_CONTOUR_INTERVAL:
                raise PresetError(
                    "invalid_field",
                    f"contour_interval must be above 0 and at most {MAX_CONTOUR_INTERVAL:g}",
                )
            object.__setattr__(self, "contour_interval", interval)
        # The dock never produces these combinations; a hand-edited file
        # that does is refused rather than reinterpreted.
        if self.freehand and (self.smart_recovery or self.learned_guide or self.auto_path):
            raise PresetError(
                "conflicting_fields",
                "freehand cannot be combined with smart_recovery, learned_guide or auto_path",
            )
        if self.method != "ink" and (self.smart_recovery or self.learned_guide):
            raise PresetError(
                "conflicting_fields", "smart_recovery and learned_guide require method ink",
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "method": self.method,
            "assist_strength": self.assist_strength,
            "smart_recovery": self.smart_recovery,
            "freehand": self.freehand,
            "auto_path": self.auto_path,
            "learned_guide": self.learned_guide,
            "ink_color_rgb": list(self.ink_color_rgb) if self.ink_color_rgb else None,
            "contour_interval": self.contour_interval,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    def renamed(self, name: str) -> "TracePreset":
        values = self.to_dict()
        values.pop("schema_version")
        values["name"] = name
        values["ink_color_rgb"] = self.ink_color_rgb
        return TracePreset(**values)

    def same_settings(self, other: "TracePreset") -> bool:
        """True when everything but the name is equal."""

        mine = self.to_dict()
        theirs = other.to_dict()
        mine.pop("name")
        theirs.pop("name")
        return mine == theirs


def _check_schema(value: Any, current: str, prefix: str) -> None:
    if value == current:
        return
    if isinstance(value, str) and value.startswith(prefix):
        version = value[len(prefix):]
        if version.isdigit() and int(version) > int(current[len(prefix):]):
            raise PresetError("newer_version", value)
    raise PresetError("unknown_schema", repr(value))


def preset_from_dict(payload: Any) -> TracePreset:
    """Read one preset, rejecting anything not written by this schema."""

    if not isinstance(payload, Mapping):
        raise PresetError("malformed", "a preset must be a JSON object")
    _check_schema(payload.get("schema_version"), SCHEMA_VERSION, _SCHEMA_PREFIX)
    unknown = sorted(set(payload) - set(_FIELDS))
    if unknown:
        raise PresetError("unknown_field", ", ".join(str(field) for field in unknown))
    missing = [field for field in _REQUIRED if field not in payload]
    if missing:
        raise PresetError("missing_field", ", ".join(missing))
    rgb = payload.get("ink_color_rgb")
    if rgb is not None and not isinstance(rgb, list):
        raise PresetError("invalid_field", "ink_color_rgb must be a list of three integers or null")
    return TracePreset(
        name=payload["name"],
        method=payload["method"],
        assist_strength=payload["assist_strength"],
        smart_recovery=_strict_bool(payload, "smart_recovery"),
        freehand=_strict_bool(payload, "freehand"),
        auto_path=_strict_bool(payload, "auto_path"),
        learned_guide=_strict_bool(payload, "learned_guide"),
        ink_color_rgb=tuple(rgb) if rgb is not None else None,
        contour_interval=payload.get("contour_interval"),
    )


def preset_from_json(text: str) -> TracePreset:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise PresetError("malformed", f"not valid JSON: {exc}") from exc
    return preset_from_dict(payload)


def library_to_json(presets: Sequence[TracePreset]) -> str:
    """Serialize a list of presets with unique names, in the given order."""

    keys = [name_key(preset.name) for preset in presets]
    if len(set(keys)) != len(keys):
        raise PresetError("duplicate_name", "preset names must be unique")
    if len(presets) > MAX_LIBRARY_SIZE:
        raise PresetError("too_many", f"at most {MAX_LIBRARY_SIZE} presets")
    return json.dumps(
        {
            "schema_version": LIBRARY_SCHEMA_VERSION,
            "presets": [preset.to_dict() for preset in presets],
        },
        ensure_ascii=False,
        sort_keys=False,
    )


def library_from_json(text: Optional[str]) -> List[TracePreset]:
    """Read the stored list. An empty or missing value is an empty list."""

    if text is None or (isinstance(text, str) and not text.strip()):
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise PresetError("malformed", f"not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PresetError("malformed", "the preset list must be a JSON object")
    _check_schema(payload.get("schema_version"), LIBRARY_SCHEMA_VERSION, _LIBRARY_PREFIX)
    if set(payload) != {"schema_version", "presets"}:
        raise PresetError("unknown_field", ", ".join(sorted(set(payload) - {"schema_version"})))
    items = payload["presets"]
    if not isinstance(items, list):
        raise PresetError("malformed", "presets must be a list")
    presets = [preset_from_dict(item) for item in items]
    keys = [name_key(preset.name) for preset in presets]
    if len(set(keys)) != len(keys):
        raise PresetError("duplicate_name", "preset names must be unique")
    return presets


def upsert(presets: Sequence[TracePreset], preset: TracePreset) -> List[TracePreset]:
    """Replace the preset with the same name, or append a new one."""

    key = name_key(preset.name)
    result = list(presets)
    for index, existing in enumerate(result):
        if name_key(existing.name) == key:
            result[index] = preset
            return result
    if len(result) >= MAX_LIBRARY_SIZE:
        raise PresetError("too_many", f"at most {MAX_LIBRARY_SIZE} presets")
    result.append(preset)
    return result


def rename(presets: Sequence[TracePreset], old_name: str, new_name: str) -> List[TracePreset]:
    old_key = name_key(old_name)
    new_key = name_key(new_name)
    if new_key != old_key and any(name_key(p.name) == new_key for p in presets):
        raise PresetError("duplicate_name", normalize_name(new_name))
    result = []
    found = False
    for preset in presets:
        if name_key(preset.name) == old_key:
            result.append(preset.renamed(new_name))
            found = True
        else:
            result.append(preset)
    if not found:
        raise PresetError("not_found", normalize_name(old_name))
    return result


def remove(presets: Sequence[TracePreset], name: str) -> List[TracePreset]:
    key = name_key(name)
    result = [preset for preset in presets if name_key(preset.name) != key]
    if len(result) == len(presets):
        raise PresetError("not_found", normalize_name(name))
    return result


def find(presets: Sequence[TracePreset], name: str) -> Optional[TracePreset]:
    key = name_key(name)
    for preset in presets:
        if name_key(preset.name) == key:
            return preset
    return None


# Reasons an option cannot take effect on this computer.
SKIP_OPENCV = "opencv_missing"
SKIP_HED_MODEL = "hed_model_missing"
SKIP_SAM_BACKEND = "sam_backend_missing"
SKIP_RECOVERY_RUNTIME = "recovery_runtime_missing"
SKIP_LEARNED_MODEL = "learned_model_missing"


@dataclass(frozen=True)
class PresetApplication:
    """What to set in the dock, and what was left out and why."""

    settings: TracePreset
    skipped: Tuple[Tuple[str, str], ...]


def plan_application(preset: TracePreset, availability: Mapping[str, bool]) -> PresetApplication:
    """Decide which preset options can be applied here.

    ``availability`` keys: ``opencv``, ``hed_model``, ``mobile_sam_backend``,
    ``sam_backend``, ``recovery_runtime``, ``learned_model``. A missing key
    counts as unavailable. Options whose runtime is missing are left at their
    off/default value and reported, so tracing never starts in a state that
    the dock would refuse or silently replace.
    """

    method = preset.method
    auto_path = preset.auto_path
    smart_recovery = preset.smart_recovery
    learned_guide = preset.learned_guide
    skipped: List[Tuple[str, str]] = []

    if method in OPENCV_METHODS and not availability.get("opencv", False):
        skipped.append(("method", SKIP_OPENCV))
        method = "ink"
    elif method == "hed" and not availability.get("hed_model", False):
        skipped.append(("method", SKIP_HED_MODEL))
        method = "ink"
    if auto_path and method in SAM_METHODS and not availability.get(f"{method}_backend", False):
        skipped.append(("auto_path", SKIP_SAM_BACKEND))
        auto_path = False
    if smart_recovery and not availability.get("recovery_runtime", False):
        skipped.append(("smart_recovery", SKIP_RECOVERY_RUNTIME))
        smart_recovery = False
    if learned_guide and not availability.get("learned_model", False):
        skipped.append(("learned_guide", SKIP_LEARNED_MODEL))
        learned_guide = False

    settings = TracePreset(
        name=preset.name,
        method=method,
        assist_strength=preset.assist_strength,
        smart_recovery=smart_recovery,
        freehand=preset.freehand,
        auto_path=auto_path,
        learned_guide=learned_guide,
        ink_color_rgb=preset.ink_color_rgb,
        contour_interval=preset.contour_interval,
    )
    return PresetApplication(settings=settings, skipped=tuple(skipped))


def suggest_contour_elevation(history: Sequence[float], interval: Optional[float]) -> Optional[float]:
    """Suggest the elevation of the next new contour, or ``None``.

    ``history`` holds the elevations the user confirmed for earlier new
    contours, oldest first. With one value the suggestion steps up by the
    interval; with two or more it keeps the direction of the last step
    (down if the last confirmed value was lower than the one before). The
    suggestion is only a dialog default; the user always confirms it.
    """

    if interval is None or not history:
        return None
    try:
        step = float(interval)
        last = float(history[-1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(step) and step > 0.0 and math.isfinite(last)):
        return None
    direction = 1.0
    if len(history) >= 2:
        try:
            before = float(history[-2])
        except (TypeError, ValueError):
            before = last
        if math.isfinite(before) and last < before:
            direction = -1.0
    return last + direction * step
