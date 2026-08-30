"""Tests for prompt and EfficientSAM tensor evidence hashes."""

from types import SimpleNamespace
import unittest

from benchmarks.evidence import (
    PROMPT_EVIDENCE_SCHEMA_VERSION,
    PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
    canonical_prompt,
    canonical_recovery_prompt_tensors,
    canonical_sam_prompt_tensors,
    canonical_source_grid_input,
    prompt_sha256,
    recovery_prompt_tensor_sha256,
    sam_prompt_tensor_sha256,
    source_grid_input_sha256,
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

    def test_optional_previous_point_upgrades_only_the_semantic_prompt(self):
        legacy = canonical_prompt(self.prompt)
        with_previous = dict(self.prompt, previous_xy=[100, 500])
        upgraded = canonical_prompt(with_previous)

        self.assertEqual(legacy["schema_version"], PROMPT_EVIDENCE_SCHEMA_VERSION_V1)
        self.assertNotIn("previous_xy", legacy)
        self.assertEqual(upgraded["schema_version"], PROMPT_EVIDENCE_SCHEMA_VERSION)
        self.assertEqual(upgraded["previous_xy"], [100.0, 500.0])
        self.assertNotEqual(prompt_sha256(with_previous), prompt_sha256(self.prompt))
        self.assertEqual(
            sam_prompt_tensor_sha256(with_previous),
            sam_prompt_tensor_sha256(self.prompt),
        )

    def test_explicit_v2_without_previous_has_distinct_schema_bound_hash(self):
        legacy = canonical_prompt(
            self.prompt,
            schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
        )
        current = canonical_prompt(
            self.prompt,
            schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION,
        )

        self.assertEqual(legacy["schema_version"], PROMPT_EVIDENCE_SCHEMA_VERSION_V1)
        self.assertEqual(current["schema_version"], PROMPT_EVIDENCE_SCHEMA_VERSION)
        self.assertNotIn("previous_xy", current)
        self.assertEqual(
            prompt_sha256(
                self.prompt,
                schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
            ),
            "389ff2b139e47e083cb75c05a05571a19a6967b0cb2f079de0c5db7217c79ad0",
        )
        self.assertNotEqual(
            prompt_sha256(
                self.prompt,
                schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
            ),
            prompt_sha256(
                self.prompt,
                schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION,
            ),
        )

    def test_prompt_object_preserves_schema_provenance(self):
        prompt_object = SimpleNamespace(
            **self.prompt,
            previous_xy=None,
            schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION,
        )

        self.assertEqual(
            canonical_prompt(prompt_object)["schema_version"],
            PROMPT_EVIDENCE_SCHEMA_VERSION,
        )
        with self.assertRaisesRegex(ValueError, "v1 does not support"):
            canonical_prompt(
                dict(self.prompt, previous_xy=[100, 500]),
                schema_version=PROMPT_EVIDENCE_SCHEMA_VERSION_V1,
            )

    def test_recovery_tensor_hash_is_the_actual_product_prompt(self):
        with_previous_and_guides = dict(
            self.prompt,
            previous_xy=[100, 500],
            positive_xy=[[300, 300]],
            negative_xy=[[400, 400]],
        )
        product_only = dict(
            self.prompt,
            positive_xy=[],
            negative_xy=[],
        )

        observed = canonical_recovery_prompt_tensors(
            with_previous_and_guides,
            width=1024,
            height=1024,
        )
        self.assertEqual(
            observed["batched_point_coords"]["values"],
            [
                [140.0, 512.0],
                [880.0, 512.0],
                [140.0, 502.0],
                [140.0, 522.0],
                [880.0, 502.0],
                [880.0, 522.0],
            ],
        )
        self.assertEqual(
            recovery_prompt_tensor_sha256(
                with_previous_and_guides,
                width=1024,
                height=1024,
            ),
            recovery_prompt_tensor_sha256(
                product_only,
                width=1024,
                height=1024,
            ),
        )

    def test_source_grid_hash_binds_origin_without_changing_prompt_hash(self):
        image_sha256 = "a" * 64
        first = canonical_source_grid_input(image_sha256, (0, 0))
        shifted = canonical_source_grid_input(image_sha256, (128, 64))

        self.assertEqual(first["source_tile_origin_xy"], [0, 0])
        self.assertEqual(shifted["source_tile_origin_xy"], [128, 64])
        self.assertNotEqual(
            source_grid_input_sha256(image_sha256, (0, 0)),
            source_grid_input_sha256(image_sha256, (128, 64)),
        )
        self.assertEqual(prompt_sha256(self.prompt), prompt_sha256(dict(self.prompt)))


if __name__ == "__main__":
    unittest.main()
