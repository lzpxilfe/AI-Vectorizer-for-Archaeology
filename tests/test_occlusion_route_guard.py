"""Visible-arm preservation never promotes an unobserved label as truth."""

import numpy as np
import pytest

from ai_vectorizer.core.confirmed_gap_curve import (
    ConfirmedGapCurveConfig, build_confirmed_gap_curve,
)
from ai_vectorizer.core.occlusion_route_guard import preserves_visible_route


def _fixture():
    source, target = (35.0, 60.0), (65.0, 60.0)
    ordinary = [(35.0, 40.0), source]
    inferred = list(ordinary)
    ordinary += list(build_confirmed_gap_curve(
        source, target, (0, 1), (0, 1),
        config=ConfirmedGapCurveConfig(handle_ratio=0.4),
    ).points_xy)[1:]
    inferred += list(build_confirmed_gap_curve(
        source, target, (0, 1), (0, 1),
        config=ConfirmedGapCurveConfig(handle_ratio=0.55),
    ).points_xy)[1:]
    ordinary.append((65.0, 40.0))
    inferred.append((65.0, 40.0))
    mask = np.zeros((96, 96), dtype=bool)
    mask[61:85, 34:67] = True
    return ordinary, inferred, dict(mask=mask, origin=(0, 0), source_xy=source, target_xy=target)


def test_matching_visible_arms_keep_the_existing_route_without_changing_inputs():
    old, new, kwargs = _fixture()
    before_old, before_new = list(old), list(new)
    before_mask = kwargs["mask"].copy()
    assert preserves_visible_route(old, new, **kwargs)
    assert old == before_old and new == before_new
    np.testing.assert_array_equal(kwargs["mask"], before_mask)


@pytest.mark.parametrize("rotation", [0, 1, 2, 3])
def test_translation_and_rotation_do_not_change_visible_arm_authority(rotation):
    old, new, kwargs = _fixture()
    mask = kwargs["mask"]
    endpoints = [kwargs["source_xy"], kwargs["target_xy"]]
    for _ in range(rotation):
        mask = np.rot90(mask)
        old = [(y, 95 - x) for x, y in old]
        new = [(y, 95 - x) for x, y in new]
        endpoints = [(y, 95 - x) for x, y in endpoints]
    origin = np.array((1200.0, 3000.0))
    assert preserves_visible_route(
        np.asarray(old) + origin, np.asarray(new) + origin,
        mask=mask, origin=origin,
        source_xy=np.asarray(endpoints[0]) + origin,
        target_xy=np.asarray(endpoints[1]) + origin,
    )


def test_neighbouring_visible_arm_is_rejected_even_when_endpoints_are_the_same():
    old, new, kwargs = _fixture()
    old.insert(1, (40.0, 52.0))
    assert not preserves_visible_route(old, new, **kwargs)


def test_shortcut_skipping_one_true_terminal_is_rejected():
    old, new, kwargs = _fixture()
    old = [old[0], (40.0, 60.0), *old[3:]]
    assert not preserves_visible_route(old, new, **kwargs)


def test_straight_shortcut_that_never_enters_label_is_rejected():
    old, new, kwargs = _fixture()
    shortcut = [old[0], kwargs["source_xy"], kwargs["target_xy"], old[-1]]
    assert not preserves_visible_route(shortcut, new, **kwargs)


def test_glyph_loop_is_rejected_despite_exact_visible_agreement():
    old, new, kwargs = _fixture()
    old = [old[0], kwargs["source_xy"], (42, 68), (50, 76), (57, 68), (45, 76), (55, 74), kwargs["target_xy"], old[-1]]
    assert not preserves_visible_route(old, new, **kwargs)


def test_non_crossing_excessive_glyph_detour_is_rejected():
    old, new, kwargs = _fixture()
    old = [old[0], kwargs["source_xy"], (35, 83), (65, 83), kwargs["target_xy"], old[-1]]
    assert not preserves_visible_route(old, new, **kwargs)


def test_two_separate_passages_through_text_are_rejected():
    old, new, kwargs = _fixture()
    mask = kwargs["mask"].copy()
    mask[48:51, 34:37] = True
    kwargs["mask"] = mask
    assert not preserves_visible_route(old, new, **kwargs)


def test_changing_clicked_endpoint_or_reversing_gap_authority_is_rejected():
    old, new, kwargs = _fixture()
    changed = list(old)
    changed[-1] = (66.0, 40.0)
    assert not preserves_visible_route(changed, new, **kwargs)
    kwargs["source_xy"], kwargs["target_xy"] = kwargs["target_xy"], kwargs["source_xy"]
    assert not preserves_visible_route(old, new, **kwargs)


def test_inferred_route_cannot_claim_an_endpoint_it_does_not_contain():
    old, new, kwargs = _fixture()
    kwargs["source_xy"] = (35.1, 60.0)
    assert not preserves_visible_route(old, new, **kwargs)


@pytest.mark.parametrize("value", [[], [[0, 0]], [[0, 0], [float("nan"), 1]], "invalid"])
def test_invalid_geometry_declines_without_mutating_fallback(value):
    old, new, kwargs = _fixture()
    assert not preserves_visible_route(value, new, **kwargs)
    assert not preserves_visible_route(old, value, **kwargs)


def test_bad_mask_and_out_of_window_geometry_fail_closed():
    old, new, kwargs = _fixture()
    kwargs["mask"] = kwargs["mask"].astype(np.uint8)
    assert not preserves_visible_route(old, new, **kwargs)
    old, new, kwargs = _fixture()
    kwargs["origin"] = (1000, 1000)
    assert not preserves_visible_route(old, new, **kwargs)


def test_vertex_and_resampling_work_budgets_decline_before_large_work():
    old, new, kwargs = _fixture()
    assert not preserves_visible_route(np.zeros((8193, 2)), new, **kwargs)
    kwargs["mask"] = np.ones((96, 96), dtype=bool)
    far = [old[0], (-1e6, -1e6), *old[1:]]
    assert not preserves_visible_route(far, new, **kwargs)
