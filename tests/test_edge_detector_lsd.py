# -*- coding: utf-8 -*-
"""QGIS-free regression tests for OpenCV LSD output compatibility."""

import unittest


try:
    import numpy as np
except ModuleNotFoundError:
    np = None


if np is not None:
    from ai_vectorizer.core.edge_detector import EdgeDetector
else:
    EdgeDetector = None


@unittest.skipIf(np is None, "NumPy is an optional plugin runtime dependency")
class TestLsdLineNormalization(unittest.TestCase):
    def test_normalizes_opencv_4_layout(self):
        lines = np.array(
            [[[1.25, 2.5, 3.75, 4.0]], [[5.0, 6.0, 7.0, 8.0]]],
            dtype=np.float32,
        )

        normalized = EdgeDetector._normalize_lsd_lines(lines)

        self.assertEqual(normalized.shape, (2, 4))
        np.testing.assert_array_equal(normalized, lines[:, 0, :])

    def test_normalizes_opencv_5_layout(self):
        lines = np.array(
            [[1.25, 2.5, 3.75, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=np.float32,
        )

        normalized = EdgeDetector._normalize_lsd_lines(lines)

        self.assertEqual(normalized.shape, (2, 4))
        np.testing.assert_array_equal(normalized, lines)

    def test_none_means_no_detected_segments(self):
        normalized = EdgeDetector._normalize_lsd_lines(None)

        self.assertEqual(normalized.shape, (0, 4))
        self.assertEqual(normalized.dtype, np.float32)

    def test_rejects_unknown_layout(self):
        with self.assertRaisesRegex(ValueError, "expected .*N, 1, 4.*N, 4"):
            EdgeDetector._normalize_lsd_lines(np.zeros((2, 2, 2), dtype=np.float32))

    def test_rejects_non_finite_coordinates(self):
        lines = np.array([[1.0, 2.0, np.nan, 4.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "must be finite"):
            EdgeDetector._normalize_lsd_lines(lines)


@unittest.skipIf(np is None, "NumPy is an optional plugin runtime dependency")
class TestLsdFailureContract(unittest.TestCase):
    def test_lsd_failure_is_not_hidden_by_canny_fallback(self):
        class FailingLsdDetector(EdgeDetector):
            def __init__(self):
                self.method = self.METHOD_LSD
                self.canny_called = False

            @staticmethod
            def _prepare_input_images(image):
                return image, np.stack([image, image, image], axis=-1)

            def _detect_lsd(self, _gray):
                raise RuntimeError("invalid LSD runtime output")

            def _detect_canny(self, _gray, _low_threshold, _high_threshold):
                self.canny_called = True
                return np.zeros_like(_gray)

        detector = FailingLsdDetector()

        with self.assertRaisesRegex(RuntimeError, "invalid LSD runtime output"):
            detector.detect_edges(np.zeros((8, 8), dtype=np.uint8))
        self.assertFalse(detector.canny_called)


try:
    if np is None:
        raise ModuleNotFoundError
    import cv2
except (ImportError, ModuleNotFoundError):
    cv2 = None


@unittest.skipUnless(
    np is not None and cv2 is not None,
    "OpenCV integration test requires optional NumPy and cv2 dependencies",
)
class TestRealOpenCvLsd(unittest.TestCase):
    def test_detects_a_line_with_installed_opencv_layout(self):
        image = np.full((64, 96), 240, dtype=np.uint8)
        cv2.line(image, (8, 32), (88, 32), 30, 2, cv2.LINE_AA)

        detector = EdgeDetector(method=EdgeDetector.METHOD_LSD)
        edges = detector.detect_edges(image)

        self.assertEqual(edges.shape, image.shape)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(edges)), 0)


if __name__ == "__main__":
    unittest.main()
