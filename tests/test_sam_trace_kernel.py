# -*- coding: utf-8 -*-
"""QGIS-free regression tests for the shared SAM mask trace boundary."""

from __future__ import annotations

import subprocess
import sys
import unittest


try:
    import numpy as np
except ModuleNotFoundError:
    np = None


from ai_vectorizer.core import trace_kernel as product_trace_kernel
from ai_vectorizer.core.sam_trace_kernel import (
    SamTraceConfig,
    build_cost_map,
    nearest_active_pixel,
    postprocess_mask,
    trace_mask,
    trace_mask_centerline,
)


class FakeCV2:
    MORPH_CLOSE = 3
    DIST_L2 = 2

    def __init__(self, *, morphology_outputs=(), distance_map=None):
        self.morphology_outputs = list(morphology_outputs)
        self.distance_map = distance_map
        self.morphology_calls = []
        self.distance_calls = []

    def morphologyEx(self, image, operation, kernel):
        self.morphology_calls.append((image.copy(), operation, kernel.copy()))
        if self.morphology_outputs:
            return np.asarray(self.morphology_outputs.pop(0), dtype=np.uint8).copy()
        return image.copy()

    def distanceTransform(self, image, distance_type, mask_size):
        self.distance_calls.append((image.copy(), distance_type, mask_size))
        if self.distance_map is not None:
            return np.asarray(self.distance_map, dtype=np.float32).copy()
        return image.astype(np.float32)


class SpyTraceKernel:
    TraceConfig = product_trace_kernel.TraceConfig
    quantize_pixel_point = staticmethod(product_trace_kernel.quantize_pixel_point)
    centerline_points = staticmethod(product_trace_kernel.centerline_points)

    def __init__(self):
        self.calls = []

    def trace_path(self, cost_map, start_xy, end_xy, **kwargs):
        self.calls.append((cost_map, start_xy, end_xy, dict(kwargs)))
        return product_trace_kernel.trace_path(cost_map, start_xy, end_xy, **kwargs)


class SamTraceImportTests(unittest.TestCase):
    def test_import_does_not_eagerly_load_optional_runtime_dependencies(self):
        script = """
import sys
before = set(sys.modules)
import ai_vectorizer.core.sam_trace_kernel
loaded = set(sys.modules) - before
blocked = {'numpy', 'cv2', 'skimage'}
assert not any(name.split('.')[0] in blocked for name in loaded), sorted(loaded)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipIf(np is None, "NumPy is an optional plugin runtime dependency")
class SamMaskPostprocessTests(unittest.TestCase):
    def test_threshold_close_and_guard_return_boolean_mask(self):
        raw = np.zeros((4, 5), dtype=np.float32)
        raw[1, 1] = 0.1
        raw[1, 2] = -0.1
        raw[2, 3] = 2.0
        cv2 = FakeCV2()
        config = SamTraceConfig(mask_min_pixels=2, mask_max_area_ratio=1.0)

        processed = postprocess_mask(
            raw,
            cv2_module=cv2,
            np_module=np,
            config=config,
        )

        self.assertEqual(processed.dtype, np.bool_)
        self.assertEqual(int(np.count_nonzero(processed)), 2)
        source, operation, kernel = cv2.morphology_calls[0]
        self.assertEqual(operation, cv2.MORPH_CLOSE)
        self.assertEqual(source.dtype, np.uint8)
        self.assertEqual(int(source[1, 1]), 255)
        self.assertEqual(int(source[1, 2]), 0)
        np.testing.assert_array_equal(kernel, np.ones((3, 3), dtype=np.uint8))

    def test_area_guards_run_after_the_first_close(self):
        expanded = np.zeros((4, 4), dtype=np.uint8)
        expanded.flat[:5] = 255
        cv2 = FakeCV2(morphology_outputs=(expanded,))
        config = SamTraceConfig(mask_min_pixels=1, mask_max_area_ratio=0.25)

        processed = postprocess_mask(
            np.ones((4, 4), dtype=np.float32),
            cv2_module=cv2,
            np_module=np,
            config=config,
        )

        self.assertIsNone(processed)

    def test_minimum_area_and_two_dimensional_guard_match_product_signal(self):
        config = SamTraceConfig(mask_min_pixels=3, mask_max_area_ratio=1.0)
        cv2 = FakeCV2()
        too_small = np.zeros((3, 3), dtype=np.float32)
        too_small[0, :2] = 1.0

        self.assertIsNone(
            postprocess_mask(
                too_small,
                cv2_module=cv2,
                np_module=np,
                config=config,
            )
        )
        self.assertIsNone(
            postprocess_mask(
                np.zeros((1, 3, 3)),
                cv2_module=object(),
                np_module=np,
                config=config,
            )
        )


@unittest.skipIf(np is None, "NumPy is an optional plugin runtime dependency")
class SamCostMapTests(unittest.TestCase):
    def test_cost_order_matches_product_edges_distance_bonus_and_skeleton(self):
        mask = np.zeros((3, 5), dtype=bool)
        mask[1, 1:4] = True
        edges = np.zeros(mask.shape, dtype=np.uint8)
        edges[1, 3] = 255
        distance = np.zeros(mask.shape, dtype=np.float32)
        distance[1, 1:4] = (1.0, 2.0, 1.0)
        skeleton = np.zeros(mask.shape, dtype=bool)
        skeleton[1, 2] = True
        cv2 = FakeCV2(distance_map=distance)

        closed, observed_skeleton, costs = build_cost_map(
            mask,
            edges,
            cv2_module=cv2,
            np_module=np,
            thin_binary_mask=lambda _mask: skeleton,
        )

        np.testing.assert_array_equal(closed, mask)
        np.testing.assert_array_equal(observed_skeleton, skeleton)
        self.assertEqual(costs.dtype, np.float32)
        self.assertEqual(float(costs[0, 0]), 12.0)
        self.assertEqual(float(costs[1, 2]), 1.0)
        self.assertAlmostEqual(float(costs[1, 1]), 2.125)
        self.assertAlmostEqual(float(costs[1, 3]), 1.225)
        self.assertEqual(len(cv2.morphology_calls), 1)

    def test_edges_and_thinner_must_preserve_mask_shape(self):
        mask = np.ones((3, 3), dtype=bool)
        cv2 = FakeCV2()
        with self.assertRaisesRegex(ValueError, "edges must have the same shape"):
            build_cost_map(
                mask,
                np.ones((2, 2), dtype=np.uint8),
                cv2_module=cv2,
                np_module=np,
                thin_binary_mask=lambda value: value,
            )
        with self.assertRaisesRegex(ValueError, "preserve the mask shape"):
            build_cost_map(
                mask,
                None,
                cv2_module=cv2,
                np_module=np,
                thin_binary_mask=lambda _value: np.ones((2, 2), dtype=bool),
            )


@unittest.skipIf(np is None, "NumPy is an optional plugin runtime dependency")
class SamEndpointAndTraceTests(unittest.TestCase):
    def test_nearest_active_pixel_preserves_ring_scan_tie_break(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1, 1] = True
        mask[1, 3] = True

        self.assertEqual(nearest_active_pixel(mask, 2, 2), (1, 1))

    def test_trace_uses_truncated_prompts_skeleton_snaps_and_strict_api(self):
        mask = np.zeros((9, 9), dtype=bool)
        mask[4, 1:8] = True
        cv2 = FakeCV2()
        spy = SpyTraceKernel()
        config = SamTraceConfig(mask_min_pixels=1, mask_max_area_ratio=1.0)

        traced = trace_mask(
            mask,
            None,
            (1.9, 2.2),
            (7.8, 6.7),
            trace_kernel=spy,
            cv2_module=cv2,
            np_module=np,
            thin_binary_mask=lambda value: value,
            config=config,
        )

        self.assertEqual(traced.snapped_start, (1, 4))
        self.assertEqual(traced.snapped_end, (7, 4))
        self.assertEqual(traced.path, tuple((x, 4) for x in range(2, 8)))
        self.assertEqual(len(spy.calls), 1)
        _costs, start, end, kwargs = spy.calls[0]
        self.assertEqual(start, (1, 4))
        self.assertEqual(end, (7, 4))
        self.assertIs(kwargs["allow_partial"], False)

    def test_one_failed_skeleton_snap_recomputes_both_against_mask(self):
        mask = np.zeros((5, 7), dtype=bool)
        mask[0, 1] = True
        mask[1, 1] = True
        mask[3, 5] = True
        skeleton = np.zeros_like(mask)
        skeleton[0, 1] = True
        config = SamTraceConfig(
            mask_min_pixels=1,
            mask_max_area_ratio=1.0,
            nearest_active_radius=1,
        )

        traced = trace_mask(
            mask,
            None,
            (1, 1),
            (5, 3),
            cv2_module=FakeCV2(),
            np_module=np,
            thin_binary_mask=lambda _value: skeleton,
            config=config,
        )

        self.assertEqual(traced.snapped_start, (1, 1))
        self.assertEqual(traced.snapped_end, (5, 3))

    def test_worker_centerline_matches_shared_historical_smoothing_profile(self):
        raw = np.zeros((9, 9), dtype=np.float32)
        raw[4, 1:8] = 1.0
        cv2 = FakeCV2()
        config = SamTraceConfig(mask_min_pixels=1, mask_max_area_ratio=1.0)
        processed = postprocess_mask(
            raw,
            cv2_module=cv2,
            np_module=np,
            config=config,
        )

        observed = trace_mask_centerline(
            processed,
            None,
            (1.9, 2.2),
            (7.8, 6.7),
            cv2_module=cv2,
            np_module=np,
            thin_binary_mask=lambda value: value,
            config=config,
        )

        self.assertEqual(len(cv2.morphology_calls), 2)
        traced = trace_mask(
            processed,
            None,
            (1.9, 2.2),
            (7.8, 6.7),
            cv2_module=FakeCV2(),
            np_module=np,
            thin_binary_mask=lambda value: value,
            config=config,
        )
        expected = product_trace_kernel.centerline_points(
            traced.trace_result,
            smooth=True,
            window_size=5,
            chaikin_iterations=3,
            segment_start_xy=(1.9, 2.2),
            segment_target_xy=(7.8, 6.7),
        )
        self.assertEqual(observed, expected)
        self.assertEqual(observed[0], (1.9, 2.2))

    def test_empty_extension_stays_empty_for_product_edge_fallback(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        points = trace_mask_centerline(
            mask,
            None,
            (1.1, 1.1),
            (1.9, 1.9),
            cv2_module=FakeCV2(),
            np_module=np,
            thin_binary_mask=lambda value: value,
            config=SamTraceConfig(mask_min_pixels=1, mask_max_area_ratio=1.0),
        )
        self.assertEqual(points, ())


if __name__ == "__main__":
    unittest.main()
