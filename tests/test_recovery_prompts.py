"""Shared product/benchmark Smart Recovery prompt contract tests."""

import unittest

import numpy as np

from ai_vectorizer.core.recovery_prompts import (
    RECOVERY_PROMPT_SCHEMA_VERSION,
    RecoveryPromptError,
    build_recovery_prompt_tensors,
)


class RecoveryPromptTests(unittest.TestCase):
    def test_anchor_target_and_four_perpendicular_negatives_are_frozen(self):
        tensors = build_recovery_prompt_tensors(
            (10.9, 20.9),
            (50.9, 20.1),
            width=100,
            height=100,
        )

        self.assertEqual(
            tensors.points_xy,
            (
                (10.0, 20.0),
                (50.0, 20.0),
                (10.0, 10.0),
                (10.0, 30.0),
                (50.0, 10.0),
                (50.0, 30.0),
            ),
        )
        self.assertEqual(tensors.labels, (1, 1, 0, 0, 0, 0))
        document = tensors.canonical_document()
        self.assertEqual(document["schema_version"], RECOVERY_PROMPT_SCHEMA_VERSION)
        self.assertEqual(document["batched_point_coords"]["shape"], [1, 1, 6, 2])

        points, labels = tensors.as_numpy(np)
        self.assertEqual(points.dtype, np.float32)
        self.assertEqual(labels.dtype, np.int32)
        self.assertTrue(points.flags.c_contiguous)
        self.assertTrue(labels.flags.c_contiguous)

    def test_out_of_bounds_negatives_are_omitted_without_clamping(self):
        tensors = build_recovery_prompt_tensors(
            (10, 2),
            (50, 2),
            width=100,
            height=100,
        )
        self.assertEqual(
            tensors.points_xy,
            ((10.0, 2.0), (50.0, 2.0), (10.0, 12.0), (50.0, 12.0)),
        )
        self.assertEqual(tensors.labels, (1, 1, 0, 0))

    def test_near_duplicate_endpoints_are_rejected(self):
        with self.assertRaisesRegex(RecoveryPromptError, "three pixels"):
            build_recovery_prompt_tensors(
                (10, 10),
                (12, 10),
                width=100,
                height=100,
            )


if __name__ == "__main__":
    unittest.main()
