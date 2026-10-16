"""Source QA must catch paper controls before a route becomes an eval target."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _audit():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_visual_controls.py"
    spec = importlib.util.spec_from_file_location("visual_control_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_case


def test_dark_prompts_do_not_hide_a_visible_control_on_paper():
    image = np.full((40, 40, 3), 240, np.uint8)
    image[20, 5:35] = 40
    case = {"id": "raw", "prompt_rc": [[20, 5], [20, 34]],
            "visible_control_segments_xy": [[[5, 20], [20, 22], [34, 20]]]}
    result = _audit()(image, case)
    assert not result["all_controls_on_dark_pixels"]
    assert [c["xy"] for c in result["checks"] if not c["dark_source_pixel"]] == [[20, 22]]


def test_source_contrast_is_not_a_claim_of_contour_identity():
    image = np.full((40, 40, 3), 240, np.uint8)
    image[15, 5:35] = image[23, 5:35] = 40
    result = _audit()(image, {"id": "two-lines", "prompt_rc": [[15, 5], [23, 34]]})
    assert result["all_controls_on_dark_pixels"]
    assert not result["contour_identity_established"]


def test_foreign_visible_controls_need_source_ink_too():
    image = np.full((40, 40, 3), 240, np.uint8)
    image[15, 5:35] = 40
    result = _audit()(image, {"id": "foreign", "prompt_rc": [[15, 5], [15, 34]],
                            "foreign_visible_control_segments_xy": [[[20, 23], [21, 23]]]})
    assert not result["all_controls_on_dark_pixels"]


@pytest.mark.parametrize("point", [[40, 10], [1, float("nan")]])
def test_invalid_source_controls_are_not_clipped_to_convenient_ink(point):
    with pytest.raises(ValueError):
        _audit()(np.full((40, 40, 3), 240, np.uint8),
                 {"id": "invalid", "prompt_rc": [[5, 5]],
                  "visible_control_segments_xy": [[point]]})
