"""Typed, QGIS-independent evidence for line-centre tracing.

``LineEvidence`` is deliberately a small NumPy boundary.  Detectors may use
very different internal representations, but Live-Wire only consumes a
continuous centre score plus optional axial direction evidence.  The arrays
are copied, normalised, and made read-only at construction time so background
tasks can safely share one immutable snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class LineEvidence:
    """Immutable line-centre evidence over one raster-sized pixel grid.

    Args:
        center_score: Continuous centre support in the closed interval
            ``[0, 1]``.  It is a guide score, not a calibrated probability.
        centerline: A directly usable boolean centreline mask.
        tangent_x/tangent_y: Optional axial unit tangent components.  Their
            sign is immaterial; ``(1, 0)`` and ``(-1, 0)`` describe the same
            line direction.
        coherence: Optional direction confidence in ``[0, 1]``.
        scale_px: Optional detector scale, in pixels, which supplied the
            strongest response at each location.  Zero means no scale.

    Missing optional arrays are materialised as read-only zero arrays.  This
    keeps consumers simple while allowing score-only evidence producers.
    """

    SCHEMA_VERSION: ClassVar[str] = "archaeotrace-line-evidence/1"

    center_score: np.ndarray
    centerline: np.ndarray
    tangent_x: Optional[np.ndarray] = None
    tangent_y: Optional[np.ndarray] = None
    coherence: Optional[np.ndarray] = None
    scale_px: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        score = self._float_array(self.center_score, "center_score")
        if score.ndim != 2 or min(score.shape, default=0) < 1:
            raise ValueError("center_score must be a non-empty 2D array")
        if np.any((score < 0.0) | (score > 1.0)):
            raise ValueError("center_score values must be between zero and one")

        centerline = np.array(self.centerline, dtype=bool, order="C", copy=True)
        self._require_shape(centerline, score.shape, "centerline")

        tangent_x = self._optional_float_array(
            self.tangent_x,
            score.shape,
            "tangent_x",
        )
        tangent_y = self._optional_float_array(
            self.tangent_y,
            score.shape,
            "tangent_y",
        )
        if (self.tangent_x is None) != (self.tangent_y is None):
            raise ValueError("tangent_x and tangent_y must be supplied together")

        coherence = self._optional_float_array(
            self.coherence,
            score.shape,
            "coherence",
        )
        if np.any((coherence < 0.0) | (coherence > 1.0)):
            raise ValueError("coherence values must be between zero and one")

        scale_px = self._optional_float_array(
            self.scale_px,
            score.shape,
            "scale_px",
        )
        if np.any(scale_px < 0.0):
            raise ValueError("scale_px values cannot be negative")

        # Direction is axial, so normalising its magnitude is safe.  A zero
        # vector cannot express an orientation and therefore has zero
        # coherence even if a producer accidentally supplied otherwise.
        # Compute the norm in float64.  Finite float32 components near their
        # maximum otherwise overflow ``hypot`` to infinity, producing a zero
        # tangent paired with non-zero coherence instead of a unit direction.
        tangent_norm = np.hypot(
            tangent_x.astype(np.float64, copy=False),
            tangent_y.astype(np.float64, copy=False),
        )
        oriented = tangent_norm > np.float32(1e-6)
        normalized_x = np.zeros(score.shape, dtype=np.float32)
        normalized_y = np.zeros(score.shape, dtype=np.float32)
        normalized_x[oriented] = (
            tangent_x[oriented].astype(np.float64, copy=False)
            / tangent_norm[oriented]
        ).astype(np.float32)
        normalized_y[oriented] = (
            tangent_y[oriented].astype(np.float64, copy=False)
            / tangent_norm[oriented]
        ).astype(np.float32)
        tangent_x = normalized_x
        tangent_y = normalized_y
        coherence[~oriented] = 0.0

        arrays = (score, centerline, tangent_x, tangent_y, coherence, scale_px)
        for array in arrays:
            array.setflags(write=False)

        object.__setattr__(self, "center_score", score)
        object.__setattr__(self, "centerline", centerline)
        object.__setattr__(self, "tangent_x", tangent_x)
        object.__setattr__(self, "tangent_y", tangent_y)
        object.__setattr__(self, "coherence", coherence)
        object.__setattr__(self, "scale_px", scale_px)

    @property
    def score(self) -> np.ndarray:
        """Read-only compatibility alias for :attr:`center_score`."""

        return self.center_score

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(int(value) for value in self.center_score.shape)

    @staticmethod
    def _require_shape(
        array: np.ndarray,
        shape: Tuple[int, ...],
        label: str,
    ) -> None:
        if array.shape != shape:
            raise ValueError(f"{label} must have shape {shape}, got {array.shape}")

    @classmethod
    def _float_array(cls, value, label: str) -> np.ndarray:
        try:
            array = np.array(value, dtype=np.float32, order="C", copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} must be a finite numeric array") from exc
        if not np.isfinite(array).all():
            raise ValueError(f"{label} must contain only finite values")
        return array

    @classmethod
    def _optional_float_array(
        cls,
        value,
        shape: Tuple[int, ...],
        label: str,
    ) -> np.ndarray:
        if value is None:
            return np.zeros(shape, dtype=np.float32)
        array = cls._float_array(value, label)
        cls._require_shape(array, shape, label)
        return array


def crop_line_evidence(
    evidence: LineEvidence,
    bounds: Tuple[int, int, int, int],
) -> LineEvidence:
    """Return an immutable end-exclusive crop in the same pixel grid.

    ``bounds`` is ``(x0, y0, x1, y1)``.  It is intentionally a plain tuple so
    background recovery tasks can share the exact Live-Wire window without a
    QGIS geometry dependency.
    """

    if not isinstance(evidence, LineEvidence):
        raise TypeError("evidence must be a LineEvidence instance")
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
        raise ValueError("bounds must be an integer (x0, y0, x1, y1) tuple")
    x0, y0, x1, y1 = (int(value) for value in bounds)
    height, width = evidence.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("bounds must be a non-empty rectangle inside evidence")
    selection = np.s_[y0:y1, x0:x1]
    return LineEvidence(
        center_score=evidence.center_score[selection],
        centerline=evidence.centerline[selection],
        tangent_x=evidence.tangent_x[selection],
        tangent_y=evidence.tangent_y[selection],
        coherence=evidence.coherence[selection],
        scale_px=evidence.scale_px[selection],
    )


__all__ = ["LineEvidence", "crop_line_evidence"]
