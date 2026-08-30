"""Conservative, QGIS-free policy for optional Smart Recovery routes.

Ink is always the champion.  This module only decides whether the champion is
weak enough to justify running a recovery provider, builds a continuous cost
map from an optional semantic corridor, and arbitrates a resulting challenger.
It deliberately performs no model loading and opens no network connections.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


RECOVERY_POLICY_ID = "smart-recovery-gate-v1-provisional"


@dataclass(frozen=True)
class RecoveryConfig:
    """Versioned conservative defaults, pending calibration on public data."""

    policy_id: str = RECOVERY_POLICY_ID
    low_support_threshold: float = 0.28
    trigger_support_quantile: float = 0.22
    trigger_longest_unsupported_run: int = 9
    trigger_mean_coherence: float = 0.16
    trigger_branch_density: float = 0.20
    maximum_detour_ratio: float = 3.0
    endpoint_tolerance_pixels: float = 2.0
    strong_ink_threshold: float = 0.68
    minimum_strong_ink_retention: float = 0.90
    maximum_route_separation_p95: float = 8.0
    minimum_support_quantile_gain: float = 0.03
    minimum_gap_reduction_pixels: int = 2
    maximum_mean_support_regret: float = 0.01
    maximum_direction_consistency_regret: float = 0.05
    maximum_branch_density_gain: float = 0.05
    corridor_outside_penalty: float = 2.0
    ink_missing_penalty: float = 4.5

    def validate(self) -> "RecoveryConfig":
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id must be a non-empty string")

        def finite_real(value: Any) -> bool:
            return (
                not isinstance(value, bool)
                and isinstance(value, Real)
                and math.isfinite(value)
            )

        unit_values = (
            self.low_support_threshold,
            self.trigger_support_quantile,
            self.trigger_mean_coherence,
            self.trigger_branch_density,
            self.strong_ink_threshold,
            self.minimum_strong_ink_retention,
            self.minimum_support_quantile_gain,
            self.maximum_mean_support_regret,
            self.maximum_direction_consistency_regret,
            self.maximum_branch_density_gain,
        )
        if any(
            not finite_real(value) or not 0.0 <= value <= 1.0
            for value in unit_values
        ):
            raise ValueError("score thresholds must be finite values in [0, 1]")
        nonnegative_values = (
            self.endpoint_tolerance_pixels,
            self.maximum_route_separation_p95,
            self.corridor_outside_penalty,
            self.ink_missing_penalty,
        )
        if any(
            not finite_real(value) or value < 0.0
            for value in nonnegative_values
        ):
            raise ValueError("recovery tuning values must be finite and non-negative")
        if (
            not finite_real(self.maximum_detour_ratio)
            or self.maximum_detour_ratio < 1.0
        ):
            raise ValueError("maximum_detour_ratio must be at least one")
        integer_values = (
            self.trigger_longest_unsupported_run,
            self.minimum_gap_reduction_pixels,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_values
        ):
            raise ValueError("recovery run thresholds must be non-negative integers")
        return self

    @property
    def sha256(self) -> str:
        """Return a stable identity for benchmark and diagnostic evidence."""

        payload = json.dumps(
            asdict(self),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


DEFAULT_RECOVERY_CONFIG = RecoveryConfig().validate()


@dataclass(frozen=True)
class RouteQuality:
    support_q10: float
    mean_support: float
    longest_unsupported_run: int
    mean_coherence: float
    detour_ratio: float
    branch_density: float
    endpoint_error: float
    sample_count: int

    def as_dict(self) -> Dict[str, Union[float, int]]:
        return asdict(self)

    @property
    def direction_consistency(self) -> float:
        """Axial route/tangent agreement under the stable evidence key.

        ``mean_coherence`` remains the serialized field name for benchmark
        compatibility, but its value measures route direction alignment
        multiplied by local evidence coherence.
        """

        return self.mean_coherence


@dataclass(frozen=True)
class RecoveryGateDecision:
    trigger: bool
    reason: str
    quality: RouteQuality
    policy_id: str
    configuration_sha256: str


@dataclass(frozen=True)
class RecoverySelection:
    selected: str
    reason: str
    champion_quality: RouteQuality
    challenger_quality: Optional[RouteQuality]
    strong_ink_retention: float
    route_separation_p95: float
    policy_id: str
    configuration_sha256: str

    @property
    def accepted(self) -> bool:
        return self.selected == "challenger"


def _score_array(evidence: Any) -> np.ndarray:
    score = np.asarray(getattr(evidence, "center_score", None), dtype=np.float32)
    if score.ndim != 2 or min(score.shape) < 1:
        raise ValueError("evidence.center_score must be a non-empty 2D array")
    if not np.isfinite(score).all() or np.any(score < 0.0) or np.any(score > 1.0):
        raise ValueError("evidence.center_score must contain finite values in [0, 1]")
    return score


def _coherence_array(evidence: Any, shape: Tuple[int, int]) -> np.ndarray:
    coherence = np.asarray(getattr(evidence, "coherence", None), dtype=np.float32)
    if coherence.shape != shape:
        raise ValueError("evidence.coherence must match center_score")
    if not np.isfinite(coherence).all() or np.any(coherence < 0.0) or np.any(coherence > 1.0):
        raise ValueError("evidence.coherence must contain finite values in [0, 1]")
    return coherence


def _path_array(path: Sequence[Sequence[float]], shape: Tuple[int, int]) -> np.ndarray:
    maximum_points = int(shape[0]) * int(shape[1])
    try:
        point_count = len(path)
    except TypeError as exc:
        raise ValueError("path must be a finite sequence of (x, y) points") from exc
    if point_count > maximum_points:
        raise ValueError("path has more vertices than the evidence grid can contain")
    try:
        points = np.asarray(path, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("path must contain finite (x, y) points") from exc
    if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 2:
        raise ValueError("path must contain at least two (x, y) points")
    if not np.isfinite(points).all():
        raise ValueError("path points must be finite")
    height, width = shape
    rounded = np.rint(points).astype(np.int64)
    if (
        np.any(rounded[:, 0] < 0)
        or np.any(rounded[:, 0] >= width)
        or np.any(rounded[:, 1] < 0)
        or np.any(rounded[:, 1] >= height)
    ):
        raise ValueError("path leaves the evidence bounds")
    return points


def _rasterized_path(
    path: np.ndarray,
    maximum_samples: Optional[int] = None,
) -> np.ndarray:
    """Sample every segment at no more than one pixel spacing."""

    samples: List[Tuple[float, float]] = []
    sample_count = 1
    for start, end in zip(path, path[1:]):
        distance = float(np.hypot(*(end - start)))
        steps = max(1, int(math.ceil(distance)))
        sample_count += steps
        if maximum_samples is not None and sample_count > maximum_samples:
            raise ValueError("rasterized path exceeds the evidence grid capacity")
        for index in range(steps):
            fraction = index / steps
            samples.append(tuple(start + (end - start) * fraction))
    samples.append(tuple(path[-1]))
    rounded = np.rint(np.asarray(samples, dtype=np.float64)).astype(np.int64)
    if len(rounded) <= 1:
        return rounded
    keep = np.ones(len(rounded), dtype=bool)
    keep[1:] = np.any(rounded[1:] != rounded[:-1], axis=1)
    return rounded[keep]


def _route_length(path: np.ndarray) -> float:
    delta = np.diff(path, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _branch_density(centerline: np.ndarray, samples: np.ndarray) -> float:
    active = np.asarray(centerline) > 0
    if active.ndim != 2:
        return 0.0
    height, width = active.shape
    branches = 0
    for x, y in samples:
        x0, x1 = max(0, x - 1), min(width, x + 2)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        neighborhood = int(np.count_nonzero(active[y0:y1, x0:x1]))
        if neighborhood >= 4:
            branches += 1
    return float(branches / max(1, len(samples)))


def _direction_consistency(
    evidence: Any,
    samples: np.ndarray,
    route_coherence: np.ndarray,
    shape: Tuple[int, int],
) -> float:
    """Return mean axial route/tangent alignment weighted by coherence.

    Older duck-typed evidence providers did not expose tangents.  They retain
    the previous mean-coherence behaviour; ``LineEvidence`` and all Ink v2
    producers take the direction-aware branch.
    """

    tangent_x_value = getattr(evidence, "tangent_x", None)
    tangent_y_value = getattr(evidence, "tangent_y", None)
    if tangent_x_value is None and tangent_y_value is None:
        return float(np.mean(route_coherence))
    if tangent_x_value is None or tangent_y_value is None:
        raise ValueError("evidence tangent_x and tangent_y must be supplied together")
    tangent_x = np.asarray(tangent_x_value, dtype=np.float32)
    tangent_y = np.asarray(tangent_y_value, dtype=np.float32)
    if tangent_x.shape != shape or tangent_y.shape != shape:
        raise ValueError("evidence tangents must match center_score")
    if not np.isfinite(tangent_x).all() or not np.isfinite(tangent_y).all():
        raise ValueError("evidence tangents must contain only finite values")

    route_vectors = np.zeros((len(samples), 2), dtype=np.float64)
    sample_points = samples.astype(np.float64, copy=False)
    if len(samples) > 1:
        route_vectors[0] = sample_points[1] - sample_points[0]
        route_vectors[-1] = sample_points[-1] - sample_points[-2]
    if len(samples) > 2:
        route_vectors[1:-1] = sample_points[2:] - sample_points[:-2]

    route_norm = np.hypot(route_vectors[:, 0], route_vectors[:, 1])
    sampled_tangent_x = tangent_x[samples[:, 1], samples[:, 0]].astype(
        np.float64,
        copy=False,
    )
    sampled_tangent_y = tangent_y[samples[:, 1], samples[:, 0]].astype(
        np.float64,
        copy=False,
    )
    evidence_norm = np.hypot(sampled_tangent_x, sampled_tangent_y)
    denominator = route_norm * evidence_norm
    dot = (
        route_vectors[:, 0] * sampled_tangent_x
        + route_vectors[:, 1] * sampled_tangent_y
    )
    alignment = np.divide(
        np.abs(dot),
        denominator,
        out=np.zeros(len(samples), dtype=np.float64),
        where=denominator > 1e-9,
    )
    weighted = np.clip(alignment, 0.0, 1.0) * route_coherence
    return float(np.mean(weighted))


def evaluate_route(
    path: Sequence[Sequence[float]],
    evidence: Any,
    *,
    expected_start: Optional[Sequence[float]] = None,
    expected_end: Optional[Sequence[float]] = None,
    config: RecoveryConfig = DEFAULT_RECOVERY_CONFIG,
) -> RouteQuality:
    """Measure an ordered path only against immutable Ink evidence."""

    config = config.validate()
    score = _score_array(evidence)
    coherence = _coherence_array(evidence, score.shape)
    points = _path_array(path, score.shape)
    samples = _rasterized_path(points, maximum_samples=int(score.size))
    support = score[samples[:, 1], samples[:, 0]]
    route_coherence = coherence[samples[:, 1], samples[:, 0]]
    unsupported = support < config.low_support_threshold
    direct = float(np.hypot(*(points[-1] - points[0])))
    detour = _route_length(points) / max(direct, 1.0)

    endpoint_error = 0.0
    for actual, expected in ((points[0], expected_start), (points[-1], expected_end)):
        if expected is None:
            continue
        candidate = np.asarray(expected, dtype=np.float64)
        if candidate.shape != (2,) or not np.isfinite(candidate).all():
            raise ValueError("expected endpoints must be finite (x, y) points")
        endpoint_error = max(endpoint_error, float(np.hypot(*(actual - candidate))))

    centerline = getattr(evidence, "centerline", np.zeros(score.shape, dtype=np.uint8))
    return RouteQuality(
        support_q10=float(np.quantile(support, 0.10)),
        mean_support=float(np.mean(support)),
        longest_unsupported_run=_longest_true_run(unsupported),
        mean_coherence=_direction_consistency(
            evidence,
            samples,
            route_coherence,
            score.shape,
        ),
        detour_ratio=float(detour),
        branch_density=_branch_density(centerline, samples),
        endpoint_error=float(endpoint_error),
        sample_count=int(len(samples)),
    )


def recovery_gate(
    champion_path: Sequence[Sequence[float]],
    evidence: Any,
    *,
    expected_start: Optional[Sequence[float]] = None,
    expected_end: Optional[Sequence[float]] = None,
    force: bool = False,
    config: RecoveryConfig = DEFAULT_RECOVERY_CONFIG,
) -> RecoveryGateDecision:
    """Return whether an optional provider may run; it never selects a route."""

    config = config.validate()
    quality = evaluate_route(
        champion_path,
        evidence,
        expected_start=expected_start,
        expected_end=expected_end,
        config=config,
    )
    if force:
        trigger, reason = True, "manual_request"
    elif quality.endpoint_error > config.endpoint_tolerance_pixels:
        trigger, reason = True, "endpoint_snap"
    elif quality.detour_ratio > config.maximum_detour_ratio:
        trigger, reason = True, "excessive_detour"
    elif quality.branch_density > config.trigger_branch_density:
        trigger, reason = True, "branch_ambiguity"
    elif quality.longest_unsupported_run >= config.trigger_longest_unsupported_run:
        trigger, reason = True, "unsupported_gap"
    elif quality.support_q10 < config.trigger_support_quantile:
        trigger, reason = True, "low_support"
    elif quality.mean_coherence < config.trigger_mean_coherence:
        trigger, reason = True, "low_coherence"
    else:
        trigger, reason = False, "ink_confident"
    return RecoveryGateDecision(
        trigger=trigger,
        reason=reason,
        quality=quality,
        policy_id=config.policy_id,
        configuration_sha256=config.sha256,
    )


def build_corridor_cost_map(
    evidence: Any,
    corridor_score: Any,
    *,
    config: RecoveryConfig = DEFAULT_RECOVERY_CONFIG,
) -> np.ndarray:
    """Fuse a semantic corridor with continuous Ink evidence for A*.

    A boolean mask is accepted, but it is interpreted as a corridor score and
    never ORed into the centerline.  Strong Ink remains cheap even outside the
    corridor so a semantic mask cannot erase a reliable traced section.
    """

    config = config.validate()
    ink = _score_array(evidence)
    corridor = np.asarray(corridor_score, dtype=np.float32)
    if corridor.shape != ink.shape:
        raise ValueError("corridor_score must match evidence.center_score")
    if (
        not np.isfinite(corridor).all()
        or np.any(corridor < 0.0)
        or np.any(corridor > 1.0)
    ):
        raise ValueError("corridor_score must contain finite values in [0, 1]")
    ink64 = ink.astype(np.float64, copy=False)
    corridor64 = corridor.astype(np.float64, copy=False)
    outside_weight = np.where(
        ink64 >= config.strong_ink_threshold,
        0.0,
        1.0,
    )
    cost = (
        1.0
        + config.ink_missing_penalty * (1.0 - ink64)
        + config.corridor_outside_penalty * (1.0 - corridor64) * outside_weight
    )
    float32_max = float(np.finfo(np.float32).max)
    if not np.isfinite(cost).all() or np.any(cost > float32_max):
        raise ValueError("recovery penalties overflow the float32 cost-map contract")
    return np.ascontiguousarray(np.maximum(cost, 1.0), dtype=np.float32)


def _nearest_distances(
    source: np.ndarray,
    target: np.ndarray,
    shape: Tuple[int, int],
    maximum_relevant_distance: float,
) -> np.ndarray:
    """Return exact nearby distances without allocating an N×M matrix.

    The arbiter only distinguishes a retained Ink pixel (<=2px) and a route
    separation above its configured safety limit. Values beyond that limit
    are therefore safely capped. SciPy supplies a linear-size distance field
    when available; the bounded NumPy/Python fallback searches only the small
    relevant neighbourhood and cannot amplify a hostile path into quadratic
    memory use.
    """

    source_pixels = np.asarray(source, dtype=np.int64)
    target_pixels = np.asarray(target, dtype=np.int64)
    target_mask = np.zeros(shape, dtype=bool)
    target_mask[target_pixels[:, 1], target_pixels[:, 0]] = True
    cap = max(2, int(math.ceil(maximum_relevant_distance)))
    try:
        from scipy import ndimage

        distance = ndimage.distance_transform_edt(~target_mask)
        return np.minimum(
            distance[source_pixels[:, 1], source_pixels[:, 0]],
            float(cap + 1),
        ).astype(np.float64, copy=False)
    except Exception:
        target_points = {
            (int(point[0]), int(point[1]))
            for point in target_pixels
        }
        result = np.full(len(source), float(cap + 1), dtype=np.float64)
        offsets = sorted(
            (
                (dx * dx + dy * dy, dx, dy)
                for dy in range(-cap, cap + 1)
                for dx in range(-cap, cap + 1)
                if dx * dx + dy * dy <= cap * cap
            ),
            key=lambda item: item[0],
        )
        for index, (x, y) in enumerate(source_pixels):
            for squared, dx, dy in offsets:
                if (int(x + dx), int(y + dy)) in target_points:
                    result[index] = math.sqrt(squared)
                    break
        return result


def arbitrate_routes(
    champion_path: Sequence[Sequence[float]],
    challenger_path: Sequence[Sequence[float]],
    evidence: Any,
    *,
    expected_start: Optional[Sequence[float]] = None,
    expected_end: Optional[Sequence[float]] = None,
    config: RecoveryConfig = DEFAULT_RECOVERY_CONFIG,
) -> RecoverySelection:
    """Accept a challenger only when it safely improves weak Ink sections."""

    config = config.validate()
    score = _score_array(evidence)
    champion = _path_array(champion_path, score.shape)
    if expected_start is None:
        expected_start = tuple(champion[0])
    if expected_end is None:
        expected_end = tuple(champion[-1])
    champion_quality = evaluate_route(
        champion,
        evidence,
        expected_start=expected_start,
        expected_end=expected_end,
        config=config,
    )
    try:
        challenger = _path_array(challenger_path, score.shape)
        challenger_quality = evaluate_route(
            challenger,
            evidence,
            expected_start=expected_start,
            expected_end=expected_end,
            config=config,
        )
    except (TypeError, ValueError):
        return RecoverySelection(
            "champion",
            "invalid_challenger",
            champion_quality,
            None,
            0.0,
            math.inf,
            config.policy_id,
            config.sha256,
        )

    champion_samples = _rasterized_path(
        champion,
        maximum_samples=int(score.size),
    )
    challenger_samples = _rasterized_path(
        challenger,
        maximum_samples=int(score.size),
    )
    challenger_float = challenger_samples.astype(np.float64)
    champion_to_challenger = _nearest_distances(
        champion_samples.astype(np.float64),
        challenger_float,
        score.shape,
        config.maximum_route_separation_p95,
    )
    challenger_to_champion = _nearest_distances(
        challenger_float,
        champion_samples.astype(np.float64),
        score.shape,
        config.maximum_route_separation_p95,
    )
    champion_support = score[champion_samples[:, 1], champion_samples[:, 0]]
    strong = champion_support >= config.strong_ink_threshold
    if np.any(strong):
        retained = float(np.mean(champion_to_challenger[strong] <= 2.0))
    else:
        retained = 1.0
    separation_p95 = max(
        float(np.quantile(champion_to_challenger, 0.95)),
        float(np.quantile(challenger_to_champion, 0.95)),
    )

    reason = "accepted"
    if challenger_quality.endpoint_error > config.endpoint_tolerance_pixels:
        reason = "endpoint_mismatch"
    elif challenger_quality.detour_ratio > config.maximum_detour_ratio:
        reason = "excessive_detour"
    elif retained < config.minimum_strong_ink_retention:
        reason = "strong_ink_replaced"
    elif separation_p95 > config.maximum_route_separation_p95:
        reason = "possible_parallel_switch"
    elif (
        challenger_quality.branch_density
        > champion_quality.branch_density + config.maximum_branch_density_gain
    ):
        reason = "branch_density_increase"
    elif (
        challenger_quality.direction_consistency
        < champion_quality.direction_consistency
        - config.maximum_direction_consistency_regret
    ):
        reason = "direction_consistency_regression"
    elif (
        challenger_quality.mean_support
        < champion_quality.mean_support - config.maximum_mean_support_regret
    ):
        reason = "mean_support_regression"
    else:
        support_gain = challenger_quality.support_q10 - champion_quality.support_q10
        gap_reduction = (
            champion_quality.longest_unsupported_run
            - challenger_quality.longest_unsupported_run
        )
        if (
            support_gain < config.minimum_support_quantile_gain
            and gap_reduction < config.minimum_gap_reduction_pixels
        ):
            reason = "no_material_improvement"

    selected = "challenger" if reason == "accepted" else "champion"
    return RecoverySelection(
        selected,
        reason,
        champion_quality,
        challenger_quality,
        retained,
        separation_p95,
        config.policy_id,
        config.sha256,
    )


__all__ = [
    "DEFAULT_RECOVERY_CONFIG",
    "RECOVERY_POLICY_ID",
    "RecoveryConfig",
    "RecoveryGateDecision",
    "RecoverySelection",
    "RouteQuality",
    "arbitrate_routes",
    "build_corridor_cost_map",
    "evaluate_route",
    "recovery_gate",
]
