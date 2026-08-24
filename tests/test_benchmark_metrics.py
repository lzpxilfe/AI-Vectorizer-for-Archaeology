import json
import unittest

from benchmarks.metrics import _matched_branch_zone_count, compute_metrics


def mask(height, width, pixels=()):
    result = [[False] * width for _ in range(height)]
    for row, column in pixels:
        result[row][column] = True
    return result


class BenchmarkMetricsTests(unittest.TestCase):
    def test_perfect_closed_path(self):
        path_xy = (
            [(column, 1) for column in range(1, 6)]
            + [(5, row) for row in range(2, 6)]
            + [(column, 5) for column in range(4, 0, -1)]
            + [(1, row) for row in range(4, 1, -1)]
        )
        centerline = mask(7, 7, [(y, x) for x, y in path_xy])

        result = compute_metrics(
            centerline,
            centerline,
            (path_xy + [path_xy[0]], True),
            tolerances=(0, 1),
            primary_tolerance=0,
        )

        self.assertEqual(result["cldice"], 1.0)
        self.assertEqual(result["distance"]["symmetric_mean"], 0.0)
        self.assertEqual(result["distance"]["symmetric_p95"], 0.0)
        self.assertEqual(result["tolerance_metrics"][0]["f1"], 1.0)
        self.assertEqual(result["topology"]["prediction"]["components"], 1)
        path_result = result["connectivity"]["paths"][0]
        self.assertTrue(path_result["closed"])
        self.assertEqual(path_result["total_pixels"], len(path_xy))
        self.assertEqual(path_result["fragments"], 1)
        self.assertEqual(path_result["breaks"], 0)
        self.assertEqual(path_result["coverage_ratio"], 1.0)
        self.assertEqual(path_result["longest_fragment_ratio"], 1.0)

    def test_one_pixel_offset_has_exact_euclidean_distance(self):
        reference_pixels = [(2, column) for column in range(1, 6)]
        prediction_pixels = [(3, column) for column in range(1, 6)]
        reference_path_xy = [(column, 2) for column in range(1, 6)]

        result = compute_metrics(
            mask(6, 7, prediction_pixels),
            mask(6, 7, reference_pixels),
            [(reference_path_xy, False)],
            tolerances=(0, 1),
            primary_tolerance=1,
        )

        self.assertEqual(result["cldice"], 0.0)
        distance = result["distance"]
        self.assertEqual(distance["prediction_to_reference"]["mean"], 1.0)
        self.assertEqual(distance["prediction_to_reference"]["p95"], 1.0)
        self.assertEqual(distance["reference_to_prediction"]["mean"], 1.0)
        self.assertEqual(distance["symmetric_mean"], 1.0)
        self.assertEqual(distance["symmetric_p95"], 1.0)
        self.assertEqual(result["tolerance_metrics"][0]["f1"], 0.0)
        self.assertEqual(result["tolerance_metrics"][1]["f1"], 1.0)
        self.assertEqual(
            result["tolerance_metrics"][1]["matched_prediction_pixels"], 5
        )
        connectivity = result["connectivity"]["paths"][0]
        self.assertEqual(connectivity["coverage_ratio"], 1.0)
        self.assertEqual(connectivity["fragments"], 1)
        self.assertEqual(connectivity["breaks"], 0)

    def test_broken_line_reports_components_fragments_and_break(self):
        reference_pixels = [(2, column) for column in range(1, 6)]
        prediction_pixels = [(2, column) for column in (1, 2, 4, 5)]
        reference_path_xy = [(column, 2) for column in range(1, 6)]

        result = compute_metrics(
            mask(5, 7, prediction_pixels),
            mask(5, 7, reference_pixels),
            [(reference_path_xy, False)],
            tolerances=(0, 1),
            primary_tolerance=0,
        )

        self.assertAlmostEqual(result["cldice"], 8.0 / 9.0)
        self.assertAlmostEqual(result["distance"]["symmetric_mean"], 0.1)
        self.assertEqual(result["distance"]["symmetric_p95"], 1.0)
        exact = result["tolerance_metrics"][0]
        self.assertEqual(exact["matched_prediction_pixels"], 4)
        self.assertEqual(exact["matched_reference_pixels"], 4)
        self.assertEqual(exact["precision"], 1.0)
        self.assertEqual(exact["recall"], 0.8)
        self.assertAlmostEqual(exact["f1"], 8.0 / 9.0)
        self.assertEqual(result["topology"]["prediction"]["components"], 2)
        connectivity = result["connectivity"]["paths"][0]
        self.assertEqual(connectivity["matched_pixels"], 4)
        self.assertEqual(connectivity["fragments"], 2)
        self.assertEqual(connectivity["fragment_excess"], 1)
        self.assertEqual(connectivity["breaks"], 1)
        self.assertEqual(connectivity["coverage_ratio"], 0.8)
        self.assertEqual(connectivity["longest_fragment_ratio"], 0.4)

    def test_spurious_branch_is_an_unmatched_branch_zone(self):
        reference_pixels = [(3, column) for column in range(1, 6)]
        prediction_pixels = reference_pixels + [(2, 3), (1, 3)]
        reference_path_xy = [(column, 3) for column in range(1, 6)]

        result = compute_metrics(
            mask(7, 7, prediction_pixels),
            mask(7, 7, reference_pixels),
            [(reference_path_xy, False)],
            tolerances=(0,),
            primary_tolerance=0,
        )

        prediction_topology = result["topology"]["prediction"]
        reference_topology = result["topology"]["reference"]
        self.assertEqual(prediction_topology["components"], 1)
        self.assertEqual(prediction_topology["branch_zones"], 1)
        self.assertEqual(prediction_topology["branch_pixels"], 1)
        self.assertEqual(prediction_topology["endpoints"], 3)
        self.assertEqual(reference_topology["branch_zones"], 0)
        self.assertEqual(reference_topology["endpoints"], 2)
        self.assertEqual(result["topology"]["matched_prediction_branch_zones"], 0)
        self.assertEqual(
            result["topology"]["unmatched_prediction_branch_zones"], 1
        )
        self.assertEqual(
            result["topology"]["unmatched_reference_branch_zones"], 0
        )

    def test_missing_reference_branch_is_an_unmatched_branch_zone(self):
        reference_pixels = [
            (3, column) for column in range(1, 6)
        ] + [(2, 3), (1, 3)]
        prediction_pixels = [(3, column) for column in range(1, 6)]
        reference_path_xy = [(column, 3) for column in range(1, 6)]

        result = compute_metrics(
            mask(7, 7, prediction_pixels),
            mask(7, 7, reference_pixels),
            [(reference_path_xy, False)],
            tolerances=(0,),
            primary_tolerance=0,
        )

        self.assertEqual(result["topology"]["prediction"]["branch_zones"], 0)
        self.assertEqual(result["topology"]["reference"]["branch_zones"], 1)
        self.assertEqual(result["topology"]["matched_prediction_branch_zones"], 0)
        self.assertEqual(
            result["topology"]["unmatched_prediction_branch_zones"], 0
        )
        self.assertEqual(
            result["topology"]["unmatched_reference_branch_zones"], 1
        )

    def test_closed_path_partial_omission_is_one_cyclic_break(self):
        path_xy = (
            [(column, 1) for column in range(1, 6)]
            + [(5, row) for row in range(2, 6)]
            + [(column, 5) for column in range(4, 0, -1)]
            + [(1, row) for row in range(4, 1, -1)]
        )
        reference_pixels = [(y, x) for x, y in path_xy]
        prediction_pixels = [
            pixel for pixel in reference_pixels if pixel not in {(1, 2), (1, 3)}
        ]

        result = compute_metrics(
            mask(7, 7, prediction_pixels),
            mask(7, 7, reference_pixels),
            [(path_xy, True)],
            tolerances=(0,),
            primary_tolerance=0,
        )

        connectivity = result["connectivity"]["paths"][0]
        self.assertEqual(connectivity["fragments"], 1)
        self.assertEqual(connectivity["fragment_excess"], 0)
        self.assertEqual(connectivity["breaks"], 1)
        self.assertAlmostEqual(
            connectivity["coverage_ratio"],
            (len(path_xy) - 2) / len(path_xy),
        )

    def test_empty_masks_and_one_empty_mask_are_json_safe(self):
        empty = mask(3, 5)
        both_empty = compute_metrics(
            empty,
            empty,
            tolerances=(0,),
            primary_tolerance=0,
        )
        self.assertEqual(both_empty["cldice"], 1.0)
        self.assertIsNone(both_empty["distance"]["symmetric_mean"])
        self.assertIsNone(both_empty["distance"]["symmetric_p95"])
        self.assertIsNone(
            both_empty["distance"]["prediction_to_reference"]["mean"]
        )
        self.assertEqual(both_empty["tolerance_metrics"][0]["f1"], 1.0)
        self.assertEqual(both_empty["topology"]["prediction"]["components"], 0)
        self.assertIsNone(
            both_empty["connectivity"]["summary"]["coverage_ratio"]
        )
        json.dumps(both_empty, allow_nan=False)

        reference_pixels = [(1, column) for column in range(1, 4)]
        reference_path_xy = [(column, 1) for column in range(1, 4)]
        one_empty = compute_metrics(
            empty,
            mask(3, 5, reference_pixels),
            [(reference_path_xy, False)],
            tolerances=(0, 2),
            primary_tolerance=1,
        )
        self.assertEqual(one_empty["cldice"], 0.0)
        self.assertIsNone(one_empty["distance"]["symmetric_mean"])
        self.assertIsNone(one_empty["distance"]["symmetric_p95"])
        self.assertEqual(one_empty["tolerance_metrics"][0]["precision"], 0.0)
        self.assertEqual(one_empty["tolerance_metrics"][0]["recall"], 0.0)
        self.assertEqual(one_empty["tolerance_metrics"][0]["f1"], 0.0)
        connectivity = one_empty["connectivity"]["paths"][0]
        self.assertEqual(connectivity["coverage_ratio"], 0.0)
        self.assertEqual(connectivity["fragments"], 0)
        self.assertTrue(connectivity["missed"])
        self.assertEqual(connectivity["breaks"], 0)
        self.assertEqual(connectivity["longest_fragment_ratio"], 0.0)
        self.assertEqual(one_empty["connectivity"]["summary"]["missed_paths"], 1)
        json.dumps(one_empty, allow_nan=False)

    def test_open_endpoint_omission_is_not_an_internal_break(self):
        reference_pixels = [(2, column) for column in range(1, 6)]
        prediction_pixels = [(2, column) for column in range(3, 6)]
        reference_path_xy = [(column, 2) for column in range(1, 6)]

        result = compute_metrics(
            mask(5, 7, prediction_pixels),
            mask(5, 7, reference_pixels),
            [(reference_path_xy, False)],
            tolerances=(0,),
            primary_tolerance=0,
        )

        connectivity = result["connectivity"]["paths"][0]
        self.assertFalse(connectivity["missed"])
        self.assertEqual(connectivity["coverage_ratio"], 0.6)
        self.assertEqual(connectivity["fragments"], 1)
        self.assertEqual(connectivity["fragment_excess"], 0)
        self.assertEqual(connectivity["breaks"], 0)
        self.assertEqual(result["connectivity"]["summary"]["missed_paths"], 0)

    def test_no_corner_cut_keeps_l_and_diagonal_lines_unbranched(self):
        for pixels in (
            [(1, 1), (2, 1), (2, 2), (2, 3)],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
        ):
            with self.subTest(pixels=pixels):
                centerline = mask(6, 6, pixels)
                path_xy = [(column, row) for row, column in pixels]
                result = compute_metrics(
                    centerline,
                    centerline,
                    [(path_xy, False)],
                    tolerances=(0,),
                    primary_tolerance=0,
                )
                topology = result["topology"]["prediction"]
                self.assertEqual(topology["components"], 1)
                self.assertEqual(topology["branch_zones"], 0)
                self.assertEqual(topology["endpoints"], 2)

    def test_branch_zone_matching_is_one_to_one(self):
        reference_pixels = [(5, 5), (4, 5), (5, 4), (5, 6)]
        prediction_pixels = [
            (5, 2),
            (4, 2),
            (5, 1),
            (5, 3),
            (5, 8),
            (4, 8),
            (5, 7),
            (5, 9),
        ]

        result = compute_metrics(
            mask(11, 11, prediction_pixels),
            mask(11, 11, reference_pixels),
            tolerances=(3,),
            primary_tolerance=3,
        )

        self.assertEqual(result["topology"]["prediction"]["branch_zones"], 2)
        self.assertEqual(result["topology"]["reference"]["branch_zones"], 1)
        self.assertEqual(result["topology"]["matched_prediction_branch_zones"], 1)
        self.assertEqual(
            result["topology"]["unmatched_prediction_branch_zones"], 1
        )
        self.assertEqual(
            result["topology"]["unmatched_reference_branch_zones"], 0
        )

    def test_large_identical_branch_zones_do_not_use_cartesian_pixel_pairs(self):
        zone = [(index // 100, index % 100) for index in range(5_000)]

        self.assertEqual(_matched_branch_zone_count([zone], [zone], 3), 1)


if __name__ == "__main__":
    unittest.main()
