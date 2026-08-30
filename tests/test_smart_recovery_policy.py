from types import SimpleNamespace

import numpy as np
import pytest

from ai_vectorizer.core.smart_recovery import (
    DEFAULT_RECOVERY_CONFIG,
    RecoveryConfig,
    _nearest_distances,
    arbitrate_routes,
    build_corridor_cost_map,
    evaluate_route,
    recovery_gate,
)


def _evidence(score):
    score = np.asarray(score, dtype=np.float32)
    return SimpleNamespace(
        center_score=score,
        coherence=np.where(score > 0.2, 0.8, 0.05).astype(np.float32),
        centerline=(score >= 0.5).astype(np.uint8) * 255,
    )


def _horizontal(y, x0=2, x1=27):
    return [(x, y) for x in range(x0, x1 + 1)]


def test_recovery_configuration_has_stable_identity():
    assert len(DEFAULT_RECOVERY_CONFIG.sha256) == 64
    assert DEFAULT_RECOVERY_CONFIG.sha256 == RecoveryConfig().sha256
    assert "provisional" in DEFAULT_RECOVERY_CONFIG.policy_id


def test_route_quality_and_gate_use_continuous_ink_support():
    score = np.zeros((20, 30), dtype=np.float32)
    score[10, 2:28] = 0.9
    confident = _evidence(score)

    quality = evaluate_route(_horizontal(10), confident)
    assert quality.support_q10 == pytest.approx(0.9)
    assert quality.longest_unsupported_run == 0
    assert recovery_gate(_horizontal(10), confident).trigger is False

    score[10, 11:20] = 0.02
    weak = _evidence(score)
    decision = recovery_gate(_horizontal(10), weak)
    assert decision.trigger is True
    assert decision.reason in {"unsupported_gap", "low_support", "low_coherence"}
    assert decision.configuration_sha256 == DEFAULT_RECOVERY_CONFIG.sha256


def test_route_quality_measures_axial_route_direction_not_only_coherence():
    score = np.ones((16, 16), dtype=np.float32)
    horizontal_tangent = np.ones_like(score)
    evidence = SimpleNamespace(
        center_score=score,
        centerline=np.zeros_like(score, dtype=np.uint8),
        tangent_x=horizontal_tangent,
        tangent_y=np.zeros_like(score),
        coherence=np.ones_like(score),
    )

    parallel = evaluate_route([(2, 5), (13, 5)], evidence)
    perpendicular = evaluate_route([(7, 2), (7, 13)], evidence)

    assert parallel.direction_consistency == pytest.approx(1.0)
    assert parallel.mean_coherence == parallel.direction_consistency
    assert perpendicular.direction_consistency == pytest.approx(0.0)


def test_manual_gate_does_not_claim_ink_is_bad():
    score = np.zeros((12, 20), dtype=np.float32)
    score[6, 1:19] = 1.0
    decision = recovery_gate(_horizontal(6, 1, 18), _evidence(score), force=True)
    assert decision.trigger is True
    assert decision.reason == "manual_request"


def test_corridor_is_a_soft_prior_and_cannot_erase_strong_ink():
    score = np.zeros((7, 9), dtype=np.float32)
    score[3, 1:8] = 0.95
    evidence = _evidence(score)
    corridor = np.zeros_like(score)
    corridor[2, 1:8] = 1.0

    cost = build_corridor_cost_map(evidence, corridor)
    assert cost.dtype == np.float32
    assert np.all(np.isfinite(cost))
    assert np.all(cost >= 1.0)
    # Strong Ink outside SAM's corridor remains cheaper than a weak pixel in it.
    assert cost[3, 4] < cost[2, 4]


def test_recovery_configuration_and_cost_map_fail_closed_on_unsafe_numbers():
    with pytest.raises(ValueError, match="maximum_detour_ratio"):
        RecoveryConfig(maximum_detour_ratio=True).validate()
    with pytest.raises(ValueError, match="score thresholds"):
        RecoveryConfig(maximum_mean_support_regret=1.1).validate()

    evidence = _evidence(np.zeros((4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_corridor_cost_map(
            evidence,
            np.full((4, 5), 2.0, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="overflow"):
        build_corridor_cost_map(
            evidence,
            np.zeros((4, 5), dtype=np.float32),
            config=RecoveryConfig(ink_missing_penalty=1e308),
        )


def test_arbiter_accepts_only_a_nearby_material_improvement():
    score = np.zeros((20, 30), dtype=np.float32)
    score[10, 2:28] = 0.9
    score[10, 11:20] = 0.02
    score[9, 2:28] = 0.92
    evidence = _evidence(score)
    evidence.centerline = np.zeros_like(score, dtype=np.uint8)

    selection = arbitrate_routes(
        _horizontal(10),
        _horizontal(9),
        evidence,
        expected_start=(2, 10),
        expected_end=(27, 10),
    )
    assert selection.accepted is True
    assert selection.reason == "accepted"
    assert selection.challenger_quality is not None
    assert (
        selection.challenger_quality.support_q10
        > selection.champion_quality.support_q10
    )


def test_gate_triggers_on_ambiguous_branch_density():
    score = np.zeros((12, 20), dtype=np.float32)
    score[5, 1:19] = 0.95
    evidence = _evidence(score)
    evidence.centerline[4:7, 1:19] = 255

    decision = recovery_gate(_horizontal(5, 1, 18), evidence)
    assert decision.trigger is True
    assert decision.reason == "branch_ambiguity"
    assert decision.quality.branch_density > DEFAULT_RECOVERY_CONFIG.trigger_branch_density


def test_arbiter_rejects_a_challenger_that_adds_branch_ambiguity():
    score = np.zeros((20, 30), dtype=np.float32)
    score[10, 2:28] = 0.9
    score[10, 11:20] = 0.02
    score[7, 2:28] = 0.95
    score[7:11, 2] = 0.95
    score[7:11, 27] = 0.95
    evidence = _evidence(score)
    evidence.centerline = np.zeros_like(score, dtype=np.uint8)
    evidence.centerline[6:9, 2:28] = 255
    challenger = [(2, 10), (2, 7), (27, 7), (27, 10)]

    selection = arbitrate_routes(
        _horizontal(10),
        challenger,
        evidence,
        expected_start=(2, 10),
        expected_end=(27, 10),
        config=RecoveryConfig(minimum_strong_ink_retention=0.0),
    )
    assert selection.accepted is False
    assert selection.reason == "branch_density_increase"
    assert selection.challenger_quality is not None
    assert (
        selection.challenger_quality.branch_density
        > selection.champion_quality.branch_density
    )


def test_arbiter_rejects_direction_consistency_regression():
    score = np.zeros((20, 30), dtype=np.float32)
    score[10, 2:28] = 0.9
    score[10, 11:20] = 0.02
    score[8, 2:28] = 0.95
    score[8:11, 2] = 0.95
    score[8:11, 27] = 0.95
    tangent_x = np.ones_like(score)
    tangent_y = np.zeros_like(score)
    tangent_x[8, 2:28] = 0.0
    tangent_y[8, 2:28] = 1.0
    evidence = SimpleNamespace(
        center_score=score,
        centerline=np.zeros_like(score, dtype=np.uint8),
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        coherence=np.ones_like(score),
    )
    challenger = [(2, 10), (2, 8), (27, 8), (27, 10)]

    selection = arbitrate_routes(
        _horizontal(10),
        challenger,
        evidence,
        expected_start=(2, 10),
        expected_end=(27, 10),
        config=RecoveryConfig(
            minimum_strong_ink_retention=0.0,
            maximum_route_separation_p95=10.0,
        ),
    )

    assert selection.accepted is False
    assert selection.reason == "direction_consistency_regression"
    assert selection.challenger_quality is not None
    assert (
        selection.challenger_quality.direction_consistency
        < selection.champion_quality.direction_consistency
    )


def test_arbiter_uses_symmetric_separation_for_far_excursions():
    score = np.zeros((45, 30), dtype=np.float32)
    score[30, 2:28] = 0.9
    score[30, 11:20] = 0.02
    score[6:31, 10] = 0.95
    score[6, 10:21] = 0.95
    score[6:31, 20] = 0.95
    evidence = _evidence(score)
    champion = _horizontal(30)
    challenger = [(2, 30), (10, 30), (10, 6), (20, 6), (20, 30), (27, 30)]

    selection = arbitrate_routes(
        champion,
        challenger,
        evidence,
        expected_start=(2, 30),
        expected_end=(27, 30),
    )
    assert selection.accepted is False
    assert selection.reason == "possible_parallel_switch"
    assert selection.route_separation_p95 > DEFAULT_RECOVERY_CONFIG.maximum_route_separation_p95


def test_arbiter_keeps_champion_on_parallel_switch_or_invalid_output():
    score = np.zeros((20, 30), dtype=np.float32)
    score[10, 2:28] = 0.9
    score[2, 2:28] = 1.0
    evidence = _evidence(score)
    parallel_detour = [(2, 10), (2, 2), (27, 2), (27, 10)]

    rejected = arbitrate_routes(
        _horizontal(10),
        parallel_detour,
        evidence,
        expected_start=(2, 10),
        expected_end=(27, 10),
    )
    assert rejected.accepted is False
    assert rejected.reason in {
        "strong_ink_replaced",
        "possible_parallel_switch",
        "excessive_detour",
    }

    invalid = arbitrate_routes(_horizontal(10), [(2, 10), (99, 99)], evidence)
    assert invalid.accepted is False
    assert invalid.reason == "invalid_challenger"

    shifted_endpoints = arbitrate_routes(_horizontal(10), _horizontal(2), evidence)
    assert shifted_endpoints.accepted is False
    assert shifted_endpoints.reason == "endpoint_mismatch"


def test_nonfinite_evidence_is_rejected_before_model_use():
    score = np.ones((8, 8), dtype=np.float32)
    score[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        recovery_gate([(1, 1), (6, 6)], _evidence(score))


def test_route_separation_uses_bounded_memory_for_long_paths():
    x = np.arange(1, 10_001, dtype=np.int64)
    source = np.column_stack((x, np.full_like(x, 4)))
    target = np.column_stack((x, np.full_like(x, 5)))
    distances = _nearest_distances(source, target, (10, 10_002), 8.0)
    np.testing.assert_allclose(distances, 1.0)


def test_arbiter_rejects_paths_whose_rasterization_exceeds_the_grid():
    score = np.ones((10, 10), dtype=np.float32)
    evidence = _evidence(score)
    champion = [(0, 0), (9, 9)]
    challenger = [
        (0 if index % 2 == 0 else 9, index % 10)
        for index in range(12)
    ]

    selection = arbitrate_routes(champion, challenger, evidence)
    assert selection.accepted is False
    assert selection.reason == "invalid_challenger"
