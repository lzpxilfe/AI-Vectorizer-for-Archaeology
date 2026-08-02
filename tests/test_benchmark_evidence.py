"""Tests for prompt and EfficientSAM tensor evidence hashes."""

from types import SimpleNamespace
import unittest

from benchmarks.evidence import (
    canonical_sam_prompt_tensors,
    prompt_sha256,
    sam_prompt_tensor_sha256,
)


class PromptEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.prompt = {
            "start_xy": [140, 512],
            "end_xy": [880, 512],
            "positive_xy": [[320.25, 400]],
            "negative_xy": [[512, 512]],
        }

    def test_semantic_and_tensor_hashes_are_stable_across_prompt_types(self):
        prompt_object = SimpleNamespace(**self.prompt)

        self.assertEqual(
            prompt_sha256(self.prompt),
            "389ff2b139e47e083cb75c05a05571a19a6967b0cb2f079de0c5db7217c79ad0",
        )
        self.assertEqual(prompt_sha256(prompt_object), prompt_sha256(self.prompt))
        self.assertEqual(
            sam_prompt_tensor_sha256(self.prompt),
            "b22cae3c17ba0dd65ab1b48670bc1a0325bc8af275d85ceffd513d1a107b38df",
        )
        self.assertEqual(
            sam_prompt_tensor_sha256(prompt_object),
            sam_prompt_tensor_sha256(self.prompt),
        )

    def test_tensor_contract_preserves_adapter_order_labels_and_shapes(self):
        tensors = canonical_sam_prompt_tensors(self.prompt)

        self.assertEqual(
            tensors["batched_point_coords"]["values"],
            [
                [140.0, 512.0],
                [320.25, 400.0],
                [880.0, 512.0],
                [512.0, 512.0],
            ],
        )
        self.assertEqual(
            tensors["batched_point_coords"]["shape"],
            [1, 1, 4, 2],
        )
        self.assertEqual(
            tensors["batched_point_labels"]["values"],
            [1.0, 1.0, 1.0, 0.0],
        )

    def test_any_semantic_prompt_change_changes_both_hashes(self):
        changed = dict(self.prompt)
        changed["negative_xy"] = [[513, 512]]

        self.assertNotEqual(prompt_sha256(changed), prompt_sha256(self.prompt))
        self.assertNotEqual(
            sam_prompt_tensor_sha256(changed),
            sam_prompt_tensor_sha256(self.prompt),
        )


if __name__ == "__main__":
    unittest.main()
