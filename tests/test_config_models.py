"""QGIS-free tests for tracing mode compatibility and defaults."""

import unittest

from ai_vectorizer import config


class TracingModeConfigTests(unittest.TestCase):
    def test_existing_default_index_now_selects_ink_centerline(self):
        self.assertEqual(config.MODEL_IDX_INK, 0)
        self.assertEqual(config.DEFAULT_EDGE_METHOD, "ink")
        self.assertEqual(config.EDGE_METHOD_BY_MODEL[0], "ink")

    def test_legacy_canny_remains_an_explicit_option(self):
        self.assertNotEqual(config.MODEL_IDX_LEGACY_CANNY, config.MODEL_IDX_INK)
        self.assertEqual(
            config.EDGE_METHOD_BY_MODEL[config.MODEL_IDX_LEGACY_CANNY],
            "canny",
        )
        self.assertIn("Legacy Canny", config.MODE_NAME_BY_MODEL.values())

    def test_existing_model_indices_are_stable(self):
        self.assertEqual(config.MODEL_IDX_LSD, 1)
        self.assertEqual(config.MODEL_IDX_HED, 2)
        self.assertEqual(config.MODEL_IDX_MOBILE_SAM, 3)
        self.assertEqual(config.MODEL_IDX_SAM, 4)


if __name__ == "__main__":
    unittest.main()
