import heapq
import json
import math
import random
import unittest

from ai_vectorizer.core.trace_kernel import (
    TraceConfig,
    TraceInputError,
    centerline_points,
    chaikin_smooth_path,
    find_path,
    quantize_pixel_point,
    smooth_pixel_path,
    trace_path,
)


class ShapeBackedMap:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        return self.rows[index]


class TraceKernelTests(unittest.TestCase):
    def test_straight_path_matches_product_segment_semantics(self):
        result = find_path([[1.0, 1.0, 1.0]], (0, 0), (2, 0))

        self.assertEqual(result.path, ((1, 0), (2, 0)))
        self.assertEqual(result.start, (0, 0))
        self.assertEqual(result.endpoint, (2, 0))
        self.assertTrue(result.reached_target)
        self.assertFalse(result.used_partial)
        self.assertEqual(result.total_cost, 2.0)

    def test_diagonal_path_uses_product_neighbor_order_and_cost(self):
        result = find_path([[1.0] * 3 for _ in range(3)], (0, 0), (2, 2))

        self.assertEqual(result.path, ((1, 1), (2, 2)))
        self.assertAlmostEqual(result.total_cost, 2 * 1.41421356237)

    def test_cost_corridor_can_outweigh_shortest_geometric_route(self):
        costs = [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 25.0, 25.0, 25.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        result = find_path(costs, (0, 1), (4, 1))

        self.assertTrue(result.reached_target)
        self.assertTrue(any(y != 1 for _x, y in result.path[:-1]))
        self.assertLess(result.total_cost, 20.0)

    def test_accumulated_path_cost_overflow_is_rejected(self):
        with self.assertRaisesRegex(TraceInputError, "accumulated path cost overflowed"):
            find_path(
                [[1.0, 1.7e308, 1.7e308]],
                (0, 0),
                (2, 0),
                allow_partial=False,
            )

    def test_shape_backed_cost_map_and_coordinate_clamping(self):
        result = find_path(ShapeBackedMap([[1, 1], [1, 1]]), (-12.2, -4), (99, 88))

        self.assertEqual(result.start, (0, 0))
        self.assertEqual(result.target, (1, 1))
        self.assertEqual(result.path, ((1, 1),))

    def test_equal_start_and_target_is_a_success_with_empty_extension(self):
        result = find_path([[1]], (0, 0), (0, 0))

        self.assertTrue(result.reached_target)
        self.assertFalse(result.used_partial)
        self.assertEqual(result.path, ())
        self.assertEqual(result.total_cost, 0.0)
        self.assertEqual(centerline_points(result), ((0.0, 0.0),))

    def test_iteration_limit_returns_nearest_partial_path(self):
        config = TraceConfig(
            max_iterations_base=2,
            max_iterations_distance_factor=0,
            max_width=10,
            max_height=10,
            max_cells=100,
        )
        result = find_path([[1.0] * 5], (0, 0), (4, 0), config=config)

        self.assertFalse(result.reached_target)
        self.assertTrue(result.used_partial)
        self.assertEqual(result.path, ((1, 0),))
        self.assertEqual(result.endpoint, (1, 0))
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.status, "partial_iteration_limit")
        self.assertTrue(result.limit_hit)

    def test_iteration_limit_can_forbid_partial_path(self):
        config = TraceConfig(
            max_iterations_base=2,
            max_iterations_distance_factor=0,
            max_width=10,
            max_height=10,
            max_cells=100,
        )
        result = find_path(
            [[1.0] * 5],
            (0, 0),
            (4, 0),
            config=config,
            allow_partial=False,
        )

        self.assertFalse(result.reached_target)
        self.assertFalse(result.used_partial)
        self.assertEqual(result.path, ())
        self.assertIsNone(result.total_cost)

    def test_stale_heap_entries_do_not_consume_iteration_budget(self):
        # This weighted map queues two nodes first through expensive routes and
        # later through cheaper ones.  The target needs nine real expansions;
        # the historical implementation re-expanded the stale entries and hit
        # the limit before reaching it.
        costs = [
            [2.0, 1.0, 5.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 20.0],
        ]
        config = TraceConfig(
            max_iterations_base=9,
            max_iterations_distance_factor=0,
            max_width=3,
            max_height=3,
            max_cells=9,
        )

        result = find_path(
            costs,
            (0, 0),
            (2, 2),
            allow_partial=False,
            config=config,
        )

        self.assertTrue(result.reached_target)
        self.assertEqual(result.iterations, 9)
        self.assertEqual(result.endpoint, (2, 2))

    def test_smoothing_matches_historical_centered_window(self):
        points = [(0, 0), (1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]

        smoothed = smooth_pixel_path(points)

        self.assertEqual(smoothed[0], (1.0, 2.0))
        self.assertEqual(smoothed[1], (1.5, 3.0))
        self.assertEqual(smoothed[2], (2.0, 4.0))
        self.assertEqual(smoothed[-1], (4.0, 8.0))

    def test_short_path_is_normalized_but_not_smoothed(self):
        self.assertEqual(
            smooth_pixel_path([(0, 0), (1, 3)]),
            ((0.0, 0.0), (1.0, 3.0)),
        )

    def test_centerline_includes_start_once(self):
        result = find_path([[1, 1, 1]], (0, 0), (2, 0))

        self.assertEqual(centerline_points(result), ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))

    def test_strict_worker_api_rejects_oob_instead_of_clamping(self):
        with self.assertRaisesRegex(TraceInputError, "outside"):
            trace_path([[1, 1]], (-0.1, 0), (1, 0))

    def test_quantization_policies_are_explicit(self):
        self.assertEqual(quantize_pixel_point((1.6, 2.5), mode="truncate"), (1, 2))
        self.assertEqual(quantize_pixel_point((1.6, 2.5), mode="round"), (2, 2))
        with self.assertRaisesRegex(TraceInputError, "integer"):
            quantize_pixel_point((1.1, 2), mode="reject_noninteger")

    def test_trace_path_uses_product_map_to_pixel_truncation(self):
        result = trace_path([[1, 1, 1]], (0.9, 0), (2, 0))
        self.assertEqual(result.start, (0, 0))
        self.assertEqual(result.points_xy, ((0, 0), (1, 0), (2, 0)))
        self.assertEqual(
            centerline_points(result, segment_start_xy=(0.9, 0)),
            ((0.9, 0.0), (1.0, 0.0), (2.0, 0.0)),
        )

    def test_same_quantized_pixel_uses_untouched_product_target_fallback(self):
        result = trace_path([[1, 1, 1]], (1.1, 0), (1.9, 0))

        self.assertEqual(result.path, ())
        self.assertEqual(
            centerline_points(
                result,
                segment_start_xy=(1.1, 0),
                segment_target_xy=(1.9, 0),
            ),
            ((1.1, 0.0), (1.9, 0.0)),
        )

    def test_chaikin_open_path_preserves_endpoints_and_count(self):
        points = ((1, 0), (2, 2), (3, 0))
        smoothed = chaikin_smooth_path(points, iterations=1)
        self.assertEqual(smoothed[0], (1.0, 0.0))
        self.assertEqual(smoothed[-1], (3.0, 0.0))
        self.assertEqual(len(smoothed), 6)

    def test_chaikin_preserves_float64_map_coordinate_precision(self):
        points = (
            (14_000_000.25, 3_700_000.25),
            (14_000_001.25, 3_700_001.25),
            (14_000_002.25, 3_700_000.25),
        )

        unchanged = chaikin_smooth_path(points, iterations=0)
        smoothed = chaikin_smooth_path(points, iterations=1)

        self.assertEqual(unchanged, points)
        self.assertEqual(smoothed[0], points[0])
        self.assertEqual(smoothed[-1], points[-1])
        self.assertEqual(smoothed[1], (14_000_000.5, 3_700_000.5))

    def test_product_centerline_reproduces_historical_endpoint_shift(self):
        result = trace_path([[1] * 7], (0, 0), (6, 0))
        points = centerline_points(result)

        self.assertEqual(points[0], (0.0, 0.0))
        self.assertEqual(points[1], (2.0, 0.0))
        self.assertEqual(points[-1], (5.0, 0.0))
        self.assertEqual(len(points), 49)

    def test_config_rejects_an_inadmissible_product_heuristic(self):
        with self.assertRaisesRegex(ValueError, "admissible"):
            TraceConfig(minimum_pixel_cost=0.5)
        with self.assertRaisesRegex(ValueError, "8-neighbor"):
            TraceConfig(neighbors=((0, 0),))

    def test_weighted_random_maps_match_a_dijkstra_cost_oracle(self):
        rng = random.Random(20260802)
        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )

        for case_index in range(20):
            costs = [[float(rng.randint(1, 9)) for _x in range(5)] for _y in range(5)]
            start = (rng.randrange(5), rng.randrange(5))
            target = (rng.randrange(5), rng.randrange(5))

            queue = [(0.0, start)]
            best = {start: 0.0}
            while queue:
                current_cost, current = heapq.heappop(queue)
                if current_cost != best[current]:
                    continue
                if current == target:
                    break
                for dx, dy in neighbors:
                    next_point = (current[0] + dx, current[1] + dy)
                    if not (0 <= next_point[0] < 5 and 0 <= next_point[1] < 5):
                        continue
                    move_cost = 1.41421356237 if dx and dy else 1.0
                    candidate = current_cost + costs[next_point[1]][next_point[0]] * move_cost
                    if candidate < best.get(next_point, math.inf):
                        best[next_point] = candidate
                        heapq.heappush(queue, (candidate, next_point))

            with self.subTest(case_index=case_index, start=start, target=target):
                result = trace_path(costs, start, target)
                self.assertTrue(result.reached_target)
                self.assertAlmostEqual(result.total_cost, best[target], places=10)

    def test_tie_break_and_final_centerline_are_byte_deterministic(self):
        costs = [[1.0] * 5 for _ in range(5)]
        serialized = []
        paths = []
        for _ in range(3):
            result = trace_path(costs, (0, 3), (4, 0))
            paths.append(result.points_xy)
            serialized.append(
                json.dumps(centerline_points(result), separators=(",", ":")).encode("ascii")
            )

        self.assertEqual(
            paths[0],
            ((0, 3), (1, 2), (2, 1), (3, 0), (4, 0)),
        )
        self.assertEqual(len(set(serialized)), 1)

    def test_trace_does_not_mutate_the_cost_map(self):
        costs = [[1.0, 2.0], [3.0, 4.0]]
        before = [row[:] for row in costs]
        trace_path(costs, (0, 0), (1, 1))
        self.assertEqual(costs, before)

    def test_invalid_cost_maps_fail_closed(self):
        invalid_maps = (
            [],
            [[]],
            [[1, 1], [1]],
            [[0]],
            [[math.nan]],
            [[math.inf]],
            [["not-a-cost"]],
        )
        for cost_map in invalid_maps:
            with self.subTest(cost_map=cost_map):
                with self.assertRaises(TraceInputError):
                    find_path(cost_map, (0, 0), (0, 0))

    def test_resource_limits_are_enforced_before_search(self):
        config = TraceConfig(max_width=2, max_height=2, max_cells=3)
        with self.assertRaisesRegex(TraceInputError, "cells"):
            find_path([[1, 1], [1, 1]], (0, 0), (1, 1), config=config)

    def test_non_finite_coordinates_are_rejected(self):
        with self.assertRaises(TraceInputError):
            find_path([[1]], (math.inf, 0), (0, 0))


if __name__ == "__main__":
    unittest.main()
