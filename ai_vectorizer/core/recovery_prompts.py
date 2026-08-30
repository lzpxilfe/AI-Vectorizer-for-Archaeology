"""Deterministic, QGIS-free Smart Recovery prompt construction.

The product and public benchmark must send the same point tensors to
EfficientSAM.  In particular, an earlier confirmed vertex is Live-Wire
direction evidence only; it is never a segmentation-model prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence, Tuple


RECOVERY_PROMPT_SCHEMA_VERSION = (
    "archaeotrace-efficientsam-recovery-prompt-tensors/1"
)
RECOVERY_NEGATIVE_DISTANCE_PX = 10.0
RECOVERY_PROMPT_MIN_DISTANCE_PX = 3.0
RECOVERY_PROMPT_MAX_POINTS = 6


class RecoveryPromptError(ValueError):
    """Raised when a recovery segment cannot form the frozen prompt tensor."""


def _finite_point(value: Sequence[float], label: str) -> Tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise RecoveryPromptError(f"{label} must be an [x, y] pair")
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError) as exc:
        raise RecoveryPromptError(f"{label} must be numeric") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise RecoveryPromptError(f"{label} must be finite")
    return x, y


def _positive_dimension(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RecoveryPromptError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class RecoveryPromptTensors:
    """Exact float32-compatible coordinates and labels sent to EfficientSAM."""

    points_xy: Tuple[Tuple[float, float], ...]
    labels: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.points_xy) != len(self.labels):
            raise RecoveryPromptError("recovery prompt coordinates and labels disagree")
        if not 2 <= len(self.points_xy) <= RECOVERY_PROMPT_MAX_POINTS:
            raise RecoveryPromptError("recovery prompt must contain two to six points")
        if self.labels[:2] != (1, 1) or any(
            label != 0 for label in self.labels[2:]
        ):
            raise RecoveryPromptError(
                "recovery prompt must contain two positives followed by negatives"
            )

    def as_numpy(self, np_module: Any) -> Tuple[Any, Any]:
        """Return contiguous arrays using the product inference dtypes."""

        return (
            np_module.ascontiguousarray(self.points_xy, dtype=np_module.float32),
            np_module.ascontiguousarray(self.labels, dtype=np_module.int32),
        )

    def canonical_document(self) -> dict[str, Any]:
        point_count = len(self.points_xy)
        return {
            "schema_version": RECOVERY_PROMPT_SCHEMA_VERSION,
            "batched_point_coords": {
                "dtype": "float32-le",
                "shape": [1, 1, point_count, 2],
                "values": [list(point) for point in self.points_xy],
            },
            "batched_point_labels": {
                "dtype": "float32-le",
                "shape": [1, 1, point_count],
                "values": [float(label) for label in self.labels],
            },
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.canonical_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def build_recovery_prompt_tensors(
    start_xy: Sequence[float],
    end_xy: Sequence[float],
    *,
    width: int,
    height: int,
) -> RecoveryPromptTensors:
    """Build the frozen anchor/target + perpendicular-negative prompt.

    Coordinates are truncated with :class:`int`, matching QGIS
    ``SmartTraceTool.map_to_pixel``.  No history/``previous_xy`` parameter is
    accepted by design: incoming direction belongs only to Live-Wire.
    """

    width = _positive_dimension(width, "width")
    height = _positive_dimension(height, "height")
    raw_start = _finite_point(start_xy, "start_xy")
    raw_end = _finite_point(end_xy, "end_xy")
    start = (int(raw_start[0]), int(raw_start[1]))
    end = (int(raw_end[0]), int(raw_end[1]))

    def in_bounds(point: Tuple[int, int]) -> bool:
        return 0 <= point[0] < width and 0 <= point[1] < height

    if not in_bounds(start) or not in_bounds(end):
        raise RecoveryPromptError("recovery endpoints must lie inside the image")

    points: list[Tuple[int, int]] = []
    labels: list[int] = []
    minimum_distance_sq = RECOVERY_PROMPT_MIN_DISTANCE_PX ** 2

    def append_if_distinct(point: Tuple[int, int], label: int) -> None:
        if any(
            (existing[0] - point[0]) ** 2 + (existing[1] - point[1]) ** 2
            < minimum_distance_sq
            for existing in points
        ):
            return
        points.append(point)
        labels.append(label)

    append_if_distinct(start, 1)
    append_if_distinct(end, 1)
    if len(points) != 2:
        raise RecoveryPromptError(
            "recovery endpoints must be at least three pixels apart"
        )

    direction_x = float(end[0] - start[0])
    direction_y = float(end[1] - start[1])
    norm = math.hypot(direction_x, direction_y)
    if norm > 0.0:
        perpendicular_x = -direction_y / norm
        perpendicular_y = direction_x / norm
        for base_x, base_y in (start, end):
            for sign in (-1, 1):
                negative = (
                    int(round(
                        base_x
                        + perpendicular_x * RECOVERY_NEGATIVE_DISTANCE_PX * sign
                    )),
                    int(round(
                        base_y
                        + perpendicular_y * RECOVERY_NEGATIVE_DISTANCE_PX * sign
                    )),
                )
                if in_bounds(negative):
                    append_if_distinct(negative, 0)

    if len(points) > RECOVERY_PROMPT_MAX_POINTS:  # Defensive contract guard.
        raise RecoveryPromptError("recovery prompt exceeded six points")
    return RecoveryPromptTensors(
        points_xy=tuple((float(x), float(y)) for x, y in points),
        labels=tuple(labels),
    )


__all__ = [
    "RECOVERY_NEGATIVE_DISTANCE_PX",
    "RECOVERY_PROMPT_MAX_POINTS",
    "RECOVERY_PROMPT_MIN_DISTANCE_PX",
    "RECOVERY_PROMPT_SCHEMA_VERSION",
    "RecoveryPromptError",
    "RecoveryPromptTensors",
    "build_recovery_prompt_tensors",
]
