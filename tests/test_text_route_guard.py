"""Only sustained following of the selected source glyph can veto preservation."""

import numpy as np
import pytest

from ai_vectorizer.core.text_route_guard import follows_text_ink


def _fixture():
    core = np.zeros((96, 96), dtype=bool)
    core[41:44, 24:57] = True
    old = [(20, 30), (25, 42), (55, 42), (60, 30)]
    new = [(20, 30), (60, 30)]
    return core, old, new


def _check(core, old, new, **kwargs):
    return follows_text_ink(old, new, core=core, origin=(0, 0),
                            source_xy=(20, 30), target_xy=(60, 30), **kwargs)


def test_sustained_selected_glyph_following_has_a_materially_better_bridge():
    core, old, new = _fixture()
    assert _check(core, old, new)


@pytest.mark.parametrize("kind", ["transverse", "both-follow", "paper", "other-glyph", "equal-path"])
def test_brief_crossings_and_equally_good_or_unrelated_paths_keep_priority(kind):
    core, old, new = _fixture()
    if kind == "transverse":
        core[:] = False
        core[:, 39:42] = True
    elif kind == "both-follow":
        core[:] = True
    elif kind in ("paper", "other-glyph"):
        core[:] = False
        if kind == "other-glyph":
            core[20:23, 24:57] = True
    else:
        new = old
    assert not _check(core, old, new)


def test_measurement_is_invariant_to_vertex_subdivision_and_duplicates():
    core, old, new = _fixture()
    def subdivide(points):
        return np.concatenate([np.linspace(a, b, 29) for a, b in zip(points, points[1:])])
    assert _check(core, subdivide(old), subdivide(new))
    assert _check(core, old, new)


def test_contacts_outside_the_selected_genuine_ends_do_not_veto_preservation():
    core = np.zeros((96, 96), dtype=bool)
    core[11:28, 19:22] = True
    old = [(20, 10), (20, 30), (60, 30)]
    new = [(20, 10), (20, 30), (60, 30)]
    assert not _check(core, old, new)


@pytest.mark.parametrize("bad", [None, [(1, 2)], [(float("nan"), 3), (60, 30)],
                                 [(1e300, 30), (60, 30)]])
def test_malformed_and_unbounded_geometry_keeps_priority(bad):
    core, _old, new = _fixture()
    assert not _check(core, bad, new)


def test_nonzero_origin_uses_original_native_pixel_coordinates():
    core, old, new = _fixture()
    offset = np.asarray((120, 230))
    assert follows_text_ink(np.asarray(old) + offset, np.asarray(new) + offset,
                            core=core, origin=offset, source_xy=np.asarray(old[0]) + offset,
                            target_xy=np.asarray(old[-1]) + offset)
