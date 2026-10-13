"""Immutable learned preferences, separate from observed Ink evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar, Optional, Tuple

import numpy as np

from .trace_guidance import TraceGuidance


@dataclass(frozen=True)
class LearnedLinePrior:
    """Uncalibrated continuous contour support and soft avoidance scores.

    Neither head is observed raster ink or a final vector.  Keeping this
    snapshot separate preserves the original centreline, tangent field and
    snapping evidence when the optional provider is absent or fails.
    """

    SCHEMA_VERSION: ClassVar[str] = "archaeotrace-learned-line-prior/1"

    contour_score: np.ndarray
    avoidance_score: np.ndarray

    def __post_init__(self) -> None:
        contour = _score_array(self.contour_score, "contour_score")
        avoidance = _score_array(self.avoidance_score, "avoidance_score")
        if contour.shape != avoidance.shape:
            raise ValueError("avoidance_score must match contour_score dimensions")
        contour.setflags(write=False)
        avoidance.setflags(write=False)
        object.__setattr__(self, "contour_score", contour)
        object.__setattr__(self, "avoidance_score", avoidance)

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(int(value) for value in self.contour_score.shape)


def crop_learned_prior(
    prior: LearnedLinePrior,
    bounds: Tuple[int, int, int, int],
) -> LearnedLinePrior:
    """Copy an end-exclusive ``(x0, y0, x1, y1)`` crop."""

    if not isinstance(prior, LearnedLinePrior):
        raise TypeError("prior must be a LearnedLinePrior instance")
    if (
        isinstance(bounds, (str, bytes))
        or not hasattr(bounds, "__len__")
        or len(bounds) != 4
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in bounds
        )
    ):
        raise ValueError("bounds must contain four integer pixel coordinates")
    x0, y0, x1, y1 = (int(value) for value in bounds)
    height, width = prior.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("bounds must be a non-empty rectangle inside prior")
    selection = np.s_[y0:y1, x0:x1]
    return LearnedLinePrior(
        prior.contour_score[selection], prior.avoidance_score[selection]
    )


def compose_learned_guidance(
    guidance: Optional[TraceGuidance],
    prior: Optional[LearnedLinePrior],
    *,
    manual_weight: float = 6.0,
    learned_weight: float = 6.0,
) -> Tuple[Optional[TraceGuidance], float]:
    """Return normalized weighted avoidance and its effective cost weight.

    Costs are max-composed per pixel, so lowering the learned weight cannot
    weaken a manual selection.  Consumers such as Smart Recovery use the
    returned weight with the returned ``TraceGuidance`` snapshot.
    """

    manual_weight = _weight(manual_weight)
    learned_weight = _weight(learned_weight)
    if guidance is not None and not isinstance(guidance, TraceGuidance):
        raise TypeError("guidance must be a TraceGuidance instance or None")
    if prior is None:
        return guidance, manual_weight
    if not isinstance(prior, LearnedLinePrior):
        raise TypeError("prior must be a LearnedLinePrior instance or None")
    if guidance is not None and guidance.shape != prior.shape:
        raise ValueError("guidance dimensions must match prior")
    if learned_weight == 0.0 or not np.any(prior.avoidance_score > 0.0):
        return guidance, manual_weight
    effective_weight = max(manual_weight, learned_weight)
    if effective_weight <= 0.0:
        return None, 0.0
    score = prior.avoidance_score * np.float32(learned_weight / effective_weight)
    if guidance is not None:
        score = np.maximum(
            guidance.avoidance_score * np.float32(manual_weight / effective_weight),
            score,
        )
    return TraceGuidance(score), effective_weight


def _weight(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("avoidance weights must be finite non-negative numbers")
    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("avoidance weights must be finite non-negative numbers") from exc
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("avoidance weights must be finite non-negative numbers")
    return weight


def _score_array(value, label: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.float32, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite numeric array") from exc
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise ValueError(f"{label} must be a non-empty 2D array")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{label} must contain finite values between zero and one")
    return array


__all__ = ["LearnedLinePrior", "crop_learned_prior", "compose_learned_guidance"]
