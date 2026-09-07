"""Unit tests for the QGIS-free soft route-guidance contract."""

import unittest

import numpy as np

from ai_vectorizer.core.trace_guidance import (
    TraceGuidance,
    crop_trace_guidance,
    guidance_from_boxes,
)


class TraceGuidanceContractTests(unittest.TestCase):
    def test_array_is_copied_read_only_and_range_checked(self):
        source = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float64)
        guidance = TraceGuidance(source)
        source[0, 1] = 0.0

        self.assertEqual(guidance.shape, (2, 2))
        self.assertEqual(float(guidance.avoidance_score[0, 1]), 0.5)
        self.assertFalse(guidance.avoidance_score.flags.writeable)
        with self.assertRaises(ValueError):
            guidance.avoidance_score[0, 0] = 1.0

        for invalid in (
            np.array([[np.nan]], dtype=np.float32),
            np.array([[1.01]], dtype=np.float32),
            np.array([[-0.01]], dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "avoidance_score"):
                    TraceGuidance(invalid)

    def test_boxes_are_soft_max_composed_and_accept_reversed_corners(self):
        guidance = guidance_from_boxes(
            (10, 12),
            [
                (8, 7, 4, 3),
                (1, 1, 2, 2),
            ],
            feather_pixels=2.0,
        )

        score = guidance.avoidance_score
        self.assertEqual(float(score[5, 6]), 1.0)
        self.assertEqual(float(score[1, 1]), 1.0)
        self.assertGreater(float(score[2, 6]), 0.0)
        self.assertEqual(float(score[0, 11]), 0.0)
        self.assertFalse(score.flags.writeable)

    def test_boxes_can_be_hard_and_outside_the_current_cache(self):
        guidance = guidance_from_boxes(
            (5, 5),
            [(-2, -2, 1, 1), (9, 9, 12, 12)],
            feather_pixels=0,
        )
        np.testing.assert_array_equal(
            guidance.avoidance_score,
            np.array(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
        )

    def test_crop_preserves_the_immutable_contract(self):
        guidance = guidance_from_boxes((8, 9), [(2, 1, 7, 6)])
        cropped = crop_trace_guidance(guidance, (2, 1, 6, 5))

        self.assertEqual(cropped.shape, (4, 4))
        np.testing.assert_array_equal(
            cropped.avoidance_score,
            guidance.avoidance_score[1:5, 2:6],
        )
        self.assertFalse(cropped.avoidance_score.flags.writeable)
        with self.assertRaisesRegex(ValueError, "inside guidance"):
            crop_trace_guidance(guidance, (2, 1, 10, 5))
        with self.assertRaisesRegex(ValueError, "integer"):
            crop_trace_guidance(guidance, (2.0, 1, 6, 5))

    def test_invalid_box_and_shape_inputs_are_rejected(self):
        for shape in ((0, 1), (1, 0), (2.0, 2), "2,2"):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    guidance_from_boxes(shape, [])

        for boxes in (None, "not-a-box-list", [(1, 2, 3)], [(1, 2, 3, np.nan)]):
            with self.subTest(boxes=boxes):
                with self.assertRaisesRegex(ValueError, "box|boxes"):
                    guidance_from_boxes((4, 4), boxes)
        with self.assertRaisesRegex(ValueError, "feather_pixels"):
            guidance_from_boxes((4, 4), [], feather_pixels=-1)
