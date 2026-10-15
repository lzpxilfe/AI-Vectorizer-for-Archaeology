"""Local endpoint direction must survive a later bend on the same stroke."""

import numpy as np
import pytest

from ai_vectorizer.core.occluded_connection import _arm


def _mask(points):
    native = np.zeros((64, 64), dtype=bool)
    for x, y in points:
        native[y, x] = True
    return native


def _turn(point, rotation, reflection=False):
    x, y = point
    if reflection:
        x = 63 - x
    for _ in range(rotation):
        x, y = y, 63 - x
    return x, y


@pytest.mark.parametrize("rotation", range(4))
@pytest.mark.parametrize("reflection", [False, True])
def test_later_tight_bend_keeps_local_signed_endpoint_tangent(rotation, reflection):
    # A long observed hook has stable first 4--6 pixels, then turns back.
    # Its full-context chord is too short for the original straight-arm fit.
    points = [(20, 20), (21, 20), (22, 20), (23, 21), (24, 21),
              (25, 22), (26, 22), (27, 22), (28, 22), (28, 23),
              (27, 24), (26, 24), (25, 24)]
    points = [_turn(p, rotation, reflection) for p in points]
    native = _mask(points)
    before = native.copy()
    arm = _arm(native, points[0], np.full(native.shape, 3.0), (8, 8, 55, 55))
    assert arm is not None
    context = np.asarray(arm.context)
    length = np.linalg.norm(np.diff(context, axis=0), axis=1).sum()
    assert length >= 12.0
    assert np.linalg.norm(context[-1] - context[0]) < 0.70 * length
    # An outward endpoint derivative points away from its own visible arm.
    expected = np.asarray(points[0], float) - points[1]
    expected /= np.linalg.norm(expected)
    assert np.dot(arm.outward, expected) > 0.99
    assert arm.width == 3.0
    assert arm.pixel == points[0]
    np.testing.assert_array_equal(native, before)


@pytest.mark.parametrize("rotation", range(4))
def test_unstable_local_hook_still_declines(rotation):
    # This shorter-radius hook turns within the tangent fit itself. A local
    # direction that depends on whether four or six pixels are used is unsafe.
    points = [(20, 20), (21, 20), (22, 20), (23, 21), (24, 22),
              (25, 23), (25, 24), (25, 25), (24, 26), (23, 27),
              (22, 27), (21, 27), (20, 27)]
    points = [_turn(p, rotation) for p in points]
    assert _arm(_mask(points), points[0], None, (8, 8, 55, 55)) is None


@pytest.mark.parametrize("rotation", range(4))
def test_short_glyph_fragment_cannot_use_local_fit_to_claim_identity(rotation):
    points = [_turn((20 + i, 20), rotation) for i in range(8)]
    assert _arm(_mask(points), points[0], None, (8, 8, 55, 55)) is None


@pytest.mark.parametrize("rotation", range(4))
def test_early_branch_and_nonendpoint_are_still_rejected(rotation):
    points = [(20 + i, 20) for i in range(20)] + [(24, 21), (24, 22)]
    points = [_turn(p, rotation) for p in points]
    native = _mask(points)
    assert _arm(native, points[0], None, (8, 8, 55, 55)) is None
    assert _arm(native, _turn((30, 20), rotation), None, (8, 8, 55, 55)) is None


@pytest.mark.parametrize("points,expected,context_length", [
    ([(20, y) for y in range(20, 40)], (0.0, -1.0), 13),
    ([(20 + i, 20 + i) for i in range(20)],
     (-0.7071067811865476, -0.7071067811865476), 10),
    ([(20, 20), (20, 21), (20, 22), (20, 23), (20, 24),
      (21, 25), (21, 26), (21, 27), (22, 28), (22, 29),
      (23, 30), (23, 31), (24, 32), (24, 33), (25, 34), (25, 35)],
     (-0.0008086112252324106, -0.9999996730738898), 12),
])
def test_successful_previous_fits_keep_exact_direction_and_context(points, expected, context_length):
    # Frozen successful outputs from the previous sustained quadratic fit.
    # Local fitting is a fallback, so even a tiny direction change is a bug.
    native = _mask(points)
    arm = _arm(native, points[0], np.full(native.shape, 3.0), (10, 10, 45, 45))
    assert arm is not None
    np.testing.assert_array_equal(arm.outward, expected)
    assert arm.context == tuple(points[:context_length])
    assert arm.width == 3.0
    assert arm.perimeter == 10
