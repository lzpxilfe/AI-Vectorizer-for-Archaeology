"""Regression tests for the dependency-light default Human Assist path."""

import unittest
from unittest.mock import patch

import numpy as np

import ai_vectorizer.core.edge_detector as edge_detector_module


class NumpyCannyFallbackTests(unittest.TestCase):
    def test_canny_detector_works_without_cv2(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[:, 32:] = 220

        with patch.object(edge_detector_module, "get_cv2", return_value=None):
            detector = edge_detector_module.EdgeDetector(
                method=edge_detector_module.EdgeDetector.METHOD_CANNY,
            )
            edges = detector.detect_edges(image)

        self.assertIsNone(detector.cv2)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertEqual(edges.shape, image.shape)
        self.assertGreater(int(np.count_nonzero(edges)), 0)

    def test_rgb_input_is_normalized_without_cv2(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:, 16:, 0] = 255

        with patch.object(edge_detector_module, "get_cv2", return_value=None):
            detector = edge_detector_module.EdgeDetector(
                method=edge_detector_module.EdgeDetector.METHOD_CANNY,
            )
            edges = detector.detect_edges(image)

        self.assertEqual(edges.shape, image.shape[:2])
        self.assertGreater(int(np.count_nonzero(edges)), 0)


if __name__ == "__main__":
    unittest.main()
