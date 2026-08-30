"""Unit tests for the QGIS-independent Ink v2 evidence boundary."""

import unittest
from unittest.mock import patch

import numpy as np

import ai_vectorizer.core.edge_detector as edge_detector_module
from ai_vectorizer.core.line_evidence import LineEvidence, crop_line_evidence


class LineEvidenceContractTests(unittest.TestCase):
    def test_arrays_are_copied_normalized_and_read_only(self):
        score = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float64)
        centerline = score == 1.0
        tangent_x = np.full(score.shape, 2.0)
        tangent_y = np.zeros(score.shape)
        coherence = np.full(score.shape, 0.6)

        evidence = LineEvidence(
            center_score=score,
            centerline=centerline,
            tangent_x=tangent_x,
            tangent_y=tangent_y,
            coherence=coherence,
        )
        score[0, 1] = 0.0

        self.assertEqual(evidence.shape, (2, 2))
        self.assertIs(evidence.score, evidence.center_score)
        self.assertEqual(float(evidence.center_score[0, 1]), 0.5)
        np.testing.assert_allclose(evidence.tangent_x, 1.0)
        np.testing.assert_allclose(evidence.tangent_y, 0.0)
        for array in (
            evidence.center_score,
            evidence.centerline,
            evidence.tangent_x,
            evidence.tangent_y,
            evidence.coherence,
            evidence.scale_px,
        ):
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError):
            evidence.center_score[0, 0] = 1.0

    def test_invalid_scores_and_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            LineEvidence(
                center_score=np.array([[1.1]], dtype=np.float32),
                centerline=np.array([[True]]),
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            LineEvidence(
                center_score=np.zeros((2, 2), dtype=np.float32),
                centerline=np.zeros((2, 3), dtype=bool),
            )
        with self.assertRaisesRegex(ValueError, "supplied together"):
            LineEvidence(
                center_score=np.zeros((2, 2), dtype=np.float32),
                centerline=np.zeros((2, 2), dtype=bool),
                tangent_x=np.ones((2, 2), dtype=np.float32),
            )

        for field in ("center_score", "tangent_x", "coherence", "scale_px"):
            values = {
                "center_score": np.zeros((2, 2), dtype=np.float32),
                "centerline": np.zeros((2, 2), dtype=bool),
                "tangent_x": np.ones((2, 2), dtype=np.float32),
                "tangent_y": np.zeros((2, 2), dtype=np.float32),
                "coherence": np.ones((2, 2), dtype=np.float32),
                "scale_px": np.ones((2, 2), dtype=np.float32),
            }
            values[field][0, 0] = np.nan
            with self.subTest(non_finite_field=field):
                with self.assertRaisesRegex(ValueError, "finite"):
                    LineEvidence(**values)

    def test_extreme_finite_tangent_is_normalized_without_overflow(self):
        maximum = np.finfo(np.float32).max
        evidence = LineEvidence(
            center_score=np.zeros((2, 2), dtype=np.float32),
            centerline=np.zeros((2, 2), dtype=bool),
            tangent_x=np.full((2, 2), maximum, dtype=np.float32),
            tangent_y=np.full((2, 2), maximum, dtype=np.float32),
            coherence=np.ones((2, 2), dtype=np.float32),
        )

        expected = np.float32(1.0 / np.sqrt(2.0))
        np.testing.assert_allclose(evidence.tangent_x, expected, atol=1e-6)
        np.testing.assert_allclose(evidence.tangent_y, expected, atol=1e-6)
        np.testing.assert_array_equal(evidence.coherence, 1.0)

    def test_crop_uses_end_exclusive_integer_bounds_and_stays_immutable(self):
        score = np.arange(30, dtype=np.float32).reshape(5, 6) / 30.0
        evidence = LineEvidence(
            center_score=score,
            centerline=score > 0.5,
            tangent_x=np.ones_like(score),
            tangent_y=np.zeros_like(score),
            coherence=np.full_like(score, 0.8),
            scale_px=np.full_like(score, 15.0),
        )

        cropped = crop_line_evidence(evidence, (2, 1, 5, 4))
        self.assertEqual(cropped.shape, (3, 3))
        np.testing.assert_array_equal(cropped.center_score, score[1:4, 2:5])
        self.assertFalse(cropped.center_score.flags.writeable)
        with self.assertRaisesRegex(ValueError, "inside evidence"):
            crop_line_evidence(evidence, (2, 1, 7, 4))
        with self.assertRaisesRegex(ValueError, "integer"):
            crop_line_evidence(evidence, (2.0, 1, 5, 4))


class InkEvidenceTests(unittest.TestCase):
    def test_v2_is_opt_in_and_does_not_change_v1_binary_output(self):
        image = np.full((64, 64), 235, dtype=np.uint8)
        image[8:56, 30:35] = 25
        detector = edge_detector_module.EdgeDetector(
            method=edge_detector_module.EdgeDetector.METHOD_INK,
        )

        before = detector.detect_edges(image)
        evidence = detector.detect_ink_evidence(image)
        after = detector.detect_edges(image)

        np.testing.assert_array_equal(before, after)
        self.assertEqual(before.dtype, np.uint8)
        self.assertEqual(evidence.center_score.dtype, np.float32)
        self.assertGreater(int(evidence.centerline.sum()), 35)

    def test_multiscale_evidence_centres_a_wide_stroke(self):
        image = np.full((96, 96), 235, dtype=np.uint8)
        image[10:86, 38:59] = 25

        evidence = edge_detector_module.EdgeDetector.detect_ink_evidence(image)

        self.assertGreater(int(evidence.centerline.sum()), 45)
        centre_columns = np.argwhere(evidence.centerline)[..., 1]
        self.assertAlmostEqual(float(np.median(centre_columns)), 48.0, delta=1.0)
        self.assertTrue(
            np.all(evidence.scale_px[evidence.centerline] == 31.0)
        )

    def test_isoluminant_colour_stroke_is_detected_without_opencv(self):
        image = np.full((64, 64, 3), 180, dtype=np.uint8)
        # This magenta has approximately the same standard luminance as the
        # gray paper, but one colour channel is substantially darker.
        image[8:56, 30:35] = np.array([255, 127, 255], dtype=np.uint8)

        with patch.object(edge_detector_module, "get_cv2", return_value=None):
            evidence = edge_detector_module.EdgeDetector.detect_ink_evidence(image)

        self.assertGreater(int(evidence.centerline.sum()), 35)
        centre_columns = np.flatnonzero(evidence.centerline[12:52].any(axis=0))
        np.testing.assert_array_equal(centre_columns, np.array([32]))

    def test_rgb_sources_are_luminance_plus_each_normalized_channel(self):
        image = np.array(
            [
                [[0, 64, 255], [255, 128, 0]],
                [[32, 96, 192], [16, 48, 160]],
            ],
            dtype=np.uint8,
        )
        sources, shape = edge_detector_module.EdgeDetector._ink_evidence_sources(image)

        self.assertEqual(shape, (2, 2))
        self.assertEqual(len(sources), 4)
        red, green, blue = (
            image[..., channel].astype(np.float32) / np.float32(255.0)
            for channel in range(3)
        )
        expected_luminance = red * 0.299 + green * 0.587 + blue * 0.114
        np.testing.assert_allclose(sources[0], expected_luminance, atol=1e-6)
        np.testing.assert_allclose(sources[1], red, atol=1e-6)
        np.testing.assert_allclose(sources[2], green, atol=1e-6)
        np.testing.assert_allclose(sources[3], blue, atol=1e-6)

    def test_isoluminant_red_and_blue_strokes_keep_channel_evidence(self):
        cases = (
            (np.array([255, 0, 0], dtype=np.uint8), 76),
            (np.array([0, 0, 255], dtype=np.uint8), 29),
        )
        for stroke, paper_level in cases:
            with self.subTest(stroke=stroke.tolist()):
                image = np.full((64, 64, 3), paper_level, dtype=np.uint8)
                image[8:56, 30:35] = stroke
                evidence = edge_detector_module.EdgeDetector.detect_ink_evidence(image)
                self.assertGreater(int(evidence.centerline.sum()), 35)
                centre_columns = np.flatnonzero(
                    evidence.centerline[12:52].any(axis=0)
                )
                np.testing.assert_array_equal(centre_columns, np.array([32]))

    def test_small_response_remains_continuous_but_not_binary_centerline(self):
        image = np.full((48, 48), 235, dtype=np.uint8)
        image[24, 24] = 15

        evidence = edge_detector_module.EdgeDetector.detect_ink_evidence(image)

        self.assertGreater(float(evidence.center_score[24, 24]), 0.0)
        self.assertFalse(bool(evidence.centerline[24, 24]))
        self.assertEqual(int(evidence.centerline.sum()), 0)

    def test_spur_pruning_never_erodes_straight_line_endpoints(self):
        centerline = np.zeros((11, 15), dtype=bool)
        centerline[5, 3:12] = True

        pruned = edge_detector_module.EdgeDetector._prune_short_ink_spurs(
            centerline,
        )

        np.testing.assert_array_equal(pruned, centerline)

    def test_spur_pruning_preserves_legitimate_t_and_x_junctions(self):
        tee = np.zeros((15, 15), dtype=bool)
        tee[8, 2:13] = True
        tee[3:9, 7] = True
        cross = np.zeros((15, 15), dtype=bool)
        for offset in range(-4, 5):
            cross[7 + offset, 7 + offset] = True
            cross[7 + offset, 7 - offset] = True

        np.testing.assert_array_equal(
            edge_detector_module.EdgeDetector._prune_short_ink_spurs(tee),
            tee,
        )
        np.testing.assert_array_equal(
            edge_detector_module.EdgeDetector._prune_short_ink_spurs(cross),
            cross,
        )

    def test_only_short_junction_spur_is_pruned_and_score_is_preserved(self):
        centerline = np.zeros((13, 17), dtype=bool)
        centerline[8, 2:15] = True
        centerline[6:8, 8] = True
        continuous_score = centerline.astype(np.float32) * np.float32(0.35)

        pruned = edge_detector_module.EdgeDetector._prune_short_ink_spurs(
            centerline,
        )
        evidence = LineEvidence(
            center_score=continuous_score,
            centerline=pruned,
        )

        self.assertFalse(bool(pruned[6, 8]))
        self.assertFalse(bool(pruned[7, 8]))
        np.testing.assert_array_equal(pruned[8, 2:15], centerline[8, 2:15])
        self.assertGreater(float(evidence.center_score[6, 8]), 0.0)
        self.assertGreater(float(evidence.center_score[7, 8]), 0.0)

    def test_detector_prunes_only_binary_spur_after_continuous_scoring(self):
        image = np.full((64, 64), 235, dtype=np.uint8)
        image[30:35, 8:56] = 25
        image[25:35, 30:35] = 25
        compatibility_centerline = np.zeros((64, 64), dtype=bool)
        compatibility_centerline[32, 8:56] = True
        compatibility_centerline[30:32, 32] = True

        with patch.object(
            edge_detector_module.EdgeDetector,
            "_tiled_ink_centerline",
            return_value=compatibility_centerline,
        ):
            evidence = edge_detector_module.EdgeDetector.detect_ink_evidence(image)

        self.assertFalse(bool(evidence.centerline[30, 32]))
        self.assertFalse(bool(evidence.centerline[31, 32]))
        self.assertTrue(bool(evidence.centerline[32, 32]))
        self.assertGreater(float(evidence.center_score[30, 32]), 0.0)
        self.assertGreater(float(evidence.center_score[31, 32]), 0.0)
        self.assertEqual(float(evidence.center_score[32, 32]), 1.0)

    def test_branch_longer_than_explicit_spur_bound_is_preserved(self):
        self.assertEqual(
            edge_detector_module.EdgeDetector.INK_EVIDENCE_MAX_SPUR_LENGTH_PX,
            2.0,
        )
        centerline = np.zeros((13, 17), dtype=bool)
        centerline[9, 2:15] = True
        centerline[6:9, 8] = True

        pruned = edge_detector_module.EdgeDetector._prune_short_ink_spurs(
            centerline,
        )

        np.testing.assert_array_equal(pruned, centerline)

    def test_spur_pruning_is_stable_across_overlapping_tile_crops(self):
        centerline = np.zeros((32, 320), dtype=bool)
        centerline[20, :] = True
        centerline[18:20, 150] = True

        full = edge_detector_module.EdgeDetector._prune_short_ink_spurs(
            centerline,
        )
        cropped = edge_detector_module.EdgeDetector._prune_short_ink_spurs(
            centerline[:, 96:288],
        )

        np.testing.assert_array_equal(
            full[4:28, 128:256],
            cropped[4:28, 32:160],
        )

    def test_detector_tile_seam_spur_is_conservatively_retained(self):
        centerline = np.zeros((16, 150), dtype=bool)
        centerline[10, 120:137] = True
        centerline[8:10, 128] = True

        unrestricted = edge_detector_module.EdgeDetector._prune_short_ink_spurs(
            centerline,
        )
        tile_safe = edge_detector_module.EdgeDetector._prune_short_ink_spurs(
            centerline,
            tile_origin=(0, 0),
        )

        self.assertFalse(bool(unrestricted[8, 128]))
        self.assertTrue(bool(tile_safe[8, 128]))
        self.assertTrue(bool(tile_safe[9, 128]))

    def test_spur_pruning_observes_cancellation_during_endpoint_walk(self):
        centerline = np.zeros((32, 64), dtype=bool)
        centerline[20, 2:62] = True
        centerline[18:20, 12] = True
        centerline[18:20, 24] = True
        calls = {"count": 0}

        def cancel_during_walk():
            calls["count"] += 1
            return calls["count"] >= 5

        with self.assertRaises(edge_detector_module.InkEvidenceCancelled):
            edge_detector_module.EdgeDetector._prune_short_ink_spurs(
                centerline,
                cancel_check=cancel_during_walk,
            )
        self.assertGreaterEqual(calls["count"], 5)

    def test_evidence_detection_observes_cancellation_between_bounded_steps(self):
        image = np.full((96, 96, 3), 235, dtype=np.uint8)
        image[8:88, 44:51] = np.array([20, 40, 60], dtype=np.uint8)
        calls = {"count": 0}

        def cancel_after_several_boundaries():
            calls["count"] += 1
            return calls["count"] >= 5

        with self.assertRaises(edge_detector_module.InkEvidenceCancelled):
            edge_detector_module.EdgeDetector.detect_ink_evidence(
                image,
                cancel_check=cancel_after_several_boundaries,
            )
        self.assertGreaterEqual(calls["count"], 5)

    def test_continuous_score_and_horizontal_tangent_are_exposed(self):
        image = np.full((96, 96), 240, dtype=np.uint8)
        levels = (220, 190, 150, 90, 30, 30, 90, 150, 190, 220)
        for row, value in zip(range(43, 53), levels):
            image[row, 12:84] = value

        evidence = edge_detector_module.EdgeDetector.detect_ink_evidence(image)

        self.assertGreater(len(np.unique(evidence.center_score)), 4)
        oriented = evidence.coherence > 0.05
        self.assertGreater(int(oriented.sum()), 100)
        self.assertGreater(
            float(np.median(np.abs(evidence.tangent_x[oriented]))),
            0.9,
        )
        self.assertLess(
            float(np.median(np.abs(evidence.tangent_y[oriented]))),
            0.1,
        )

    def test_source_anchored_tiles_match_across_overlapping_crops(self):
        image = np.full((128, 320), 235, dtype=np.uint8)
        for x in range(20, 300):
            y = 60 + ((x // 40) % 2)
            image[y - 2:y + 3, x] = 30 + (x % 80)
        image[20:100, 245:248] = 0

        full = edge_detector_module.EdgeDetector.detect_ink_evidence(
            image,
            tile_origin=(0, 0),
        )
        cropped = edge_detector_module.EdgeDetector.detect_ink_evidence(
            image[:, 96:288],
            tile_origin=(96, 0),
        )

        # Global tile 128..255 has its complete 16px halo in both inputs.
        np.testing.assert_array_equal(
            full.center_score[16:112, 128:256],
            cropped.center_score[16:112, 32:160],
        )
        np.testing.assert_array_equal(
            full.centerline[16:112, 128:256],
            cropped.centerline[16:112, 32:160],
        )

    def test_combined_filter_and_tile_halo_is_exact_on_noisy_overlap(self):
        rng = np.random.default_rng(9)
        image = np.clip(
            rng.normal(220, 18, (128, 384)),
            0,
            255,
        ).astype(np.uint8)
        image[20:110, 190:196] = rng.integers(
            5,
            60,
            size=(90, 6),
            dtype=np.uint8,
        )
        image[40:44, 80:310] = 30
        image[10:118, 110] = 0
        image[5:123, 280] = 255

        full = edge_detector_module.EdgeDetector.detect_ink_evidence(
            image,
            tile_origin=(0, 0),
        )
        # Tile 128..255 needs the detector's 16px response context plus the
        # 31px filter's 15px radius: exactly 31 source pixels on each side.
        cropped = edge_detector_module.EdgeDetector.detect_ink_evidence(
            image[:, 97:287],
            tile_origin=(97, 0),
        )

        for field in (
            "center_score",
            "centerline",
            "tangent_x",
            "tangent_y",
            "coherence",
            "scale_px",
        ):
            with self.subTest(field=field):
                np.testing.assert_array_equal(
                    getattr(full, field)[:, 128:256],
                    getattr(cropped, field)[:, 31:159],
                )

    def test_two_dimensional_tile_neighborhood_is_pan_invariant(self):
        rng = np.random.default_rng(1)
        image = np.clip(
            rng.normal(220, 18, (384, 384)),
            0,
            255,
        ).astype(np.uint8)
        image[20:350, 190:196] = rng.integers(
            5,
            60,
            size=(330, 6),
            dtype=np.uint8,
        )
        image[180:184, 20:360] = 30

        full = edge_detector_module.EdgeDetector.detect_ink_evidence(
            image,
            tile_origin=(0, 0),
        )
        # Complete global tile 128..255 plus the exact 31px source context.
        cropped = edge_detector_module.EdgeDetector.detect_ink_evidence(
            image[97:287, 97:287],
            tile_origin=(97, 97),
        )

        for field in (
            "center_score",
            "centerline",
            "tangent_x",
            "tangent_y",
            "coherence",
            "scale_px",
        ):
            with self.subTest(field=field):
                np.testing.assert_array_equal(
                    getattr(full, field)[128:256, 128:256],
                    getattr(cropped, field)[31:159, 31:159],
                )

    def test_numpy_only_v2_and_tile_origin_validation(self):
        image = np.full((48, 48), 235, dtype=np.uint8)
        image[6:42, 22:27] = 25
        with patch.object(edge_detector_module, "_scipy_ndimage", None):
            with patch.object(edge_detector_module, "_skimage_threshold_otsu", None):
                with patch.object(edge_detector_module, "_skimage_skeletonize", None):
                    with patch.object(edge_detector_module, "get_cv2", return_value=None):
                        evidence = (
                            edge_detector_module.EdgeDetector.detect_ink_evidence(
                                image,
                            )
                        )
        self.assertGreater(int(evidence.centerline.sum()), 25)

        with self.assertRaisesRegex(ValueError, "integer"):
            edge_detector_module.EdgeDetector.detect_ink_evidence(
                image,
                tile_origin=(0.5, 0),
            )


if __name__ == "__main__":
    unittest.main()
