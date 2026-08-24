"""Regression tests for dependency-light Human Assist detectors."""

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


class InkCenterlineTests(unittest.TestCase):
    def test_dark_stroke_becomes_one_centerline_without_cv2(self):
        image = np.full((64, 64), 235, dtype=np.uint8)
        image[8:56, 30:35] = 25

        with patch.object(edge_detector_module, "get_cv2", return_value=None):
            detector = edge_detector_module.EdgeDetector(
                method=edge_detector_module.EdgeDetector.METHOD_INK,
            )
            edges = detector.detect_edges(image)

        active = edges > 0
        self.assertIsNone(detector.cv2)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertEqual(edges.shape, image.shape)
        self.assertGreater(int(active.sum()), 35)
        self.assertLess(int(active.sum()), 65)
        self.assertLessEqual(int(active[12:52].sum(axis=1).max()), 1)
        central_columns = np.flatnonzero(active[12:52].any(axis=0))
        np.testing.assert_array_equal(central_columns, np.array([32]))

    def test_uniform_paper_has_no_false_centerline(self):
        image = np.full((48, 48), 180, dtype=np.uint8)
        detector = edge_detector_module.EdgeDetector(
            method=edge_detector_module.EdgeDetector.METHOD_INK,
        )

        edges = detector.detect_edges(image)

        self.assertEqual(int(np.count_nonzero(edges)), 0)

    def test_tiny_scan_speckles_are_removed(self):
        image = np.full((64, 64), 235, dtype=np.uint8)
        image[12:52, 31:34] = 25
        image[4, 4] = 0
        image[59, 54] = 0
        detector = edge_detector_module.EdgeDetector(
            method=edge_detector_module.EdgeDetector.METHOD_INK,
        )

        edges = detector.detect_edges(image)

        self.assertEqual(int(edges[4, 4]), 0)
        self.assertEqual(int(edges[59, 54]), 0)
        self.assertGreater(int(np.count_nonzero(edges)), 30)

    def test_zoomed_thick_stroke_still_has_one_centerline(self):
        image = np.full((96, 96), 235, dtype=np.uint8)
        image[12:84, 42:53] = 25
        detector = edge_detector_module.EdgeDetector(
            method=edge_detector_module.EdgeDetector.METHOD_INK,
        )

        edges = detector.detect_edges(image)

        active = edges > 0
        self.assertGreater(int(active.sum()), 55)
        self.assertLess(int(active.sum()), 90)
        self.assertLessEqual(int(active[18:78].sum(axis=1).max()), 1)
        center = float(np.median(np.argwhere(active)[..., 1]))
        self.assertAlmostEqual(center, 47.0, delta=1.0)

    def test_missing_optional_dependencies_still_produces_one_centerline(self):
        image = np.full((64, 64), 235, dtype=np.uint8)
        image[8:56, 30:35] = 25

        with patch.object(edge_detector_module, "_scipy_ndimage", None):
            with patch.object(
                edge_detector_module,
                "_skimage_threshold_otsu",
                None,
            ):
                with patch.object(
                    edge_detector_module,
                    "_skimage_skeletonize",
                    None,
                ):
                    with patch.object(
                        edge_detector_module,
                        "get_cv2",
                        return_value=None,
                    ):
                        detector = edge_detector_module.EdgeDetector(
                            method=edge_detector_module.EdgeDetector.METHOD_INK,
                        )
                        edges = detector.detect_edges(image)
                        status = detector.get_ink_runtime_status()

        active = edges > 0
        self.assertEqual(active.shape, image.shape)
        self.assertGreater(int(active.sum()), 35)
        self.assertLess(int(active.sum()), 65)
        self.assertLessEqual(int(active[12:52].sum(axis=1).max()), 1)
        self.assertEqual(status["reason"], "numpy_fallback")
        self.assertEqual(status["background_backend"], "numpy")
        self.assertEqual(status["thinning_backend"], "numpy")

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
