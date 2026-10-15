"""Real QGIS automatic label routing, preview and edit lifecycles.

The synthetic image has two printed counters hiding the bottom of a broad
U and an uninterrupted outer contour. RGB detection, NumPy thinning, the
background task and ordinary mouse acceptance all run without a shortcut.
Hidden historical geometry is not asserted by this controlled construction.
"""

import os
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    from qgis.core import (
        QgsApplication, QgsCoordinateReferenceSystem, QgsPointXY, QgsProject,
        QgsRectangle, QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QPointF

    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError("QGIS bindings are required for automatic occlusion tests") from exc
    HAS_QGIS = False


class _TaskDispatcher:
    def __init__(self, immediate=True):
        self.immediate = immediate
        self.tasks = []

    def addTask(self, task):
        self.tasks.append(task)
        if self.immediate:
            task.finished(bool(task.run()))
        return True


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class OccludedConnectionQgisTests(unittest.TestCase):
    SIZE = 192

    @classmethod
    def setUpClass(cls):
        cls.owns_app = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        if cls.owns_app:
            cls.application.initQgis()
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool

        cls.Tool = SmartTraceTool

    @classmethod
    def tearDownClass(cls):
        if cls.owns_app:
            cls.application.exitQgis()

    def setUp(self):
        import numpy as np

        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.core.livewire import is_livewire_available

        if not is_livewire_available():
            self.skipTest("Automatic label connections require optional SciPy")
        self.np = np
        self.crs = QgsCoordinateReferenceSystem("EPSG:3857")
        self.canvas = QgsMapCanvas()
        self.canvas.setDestinationCrs(self.crs)
        self.canvas.setExtent(QgsRectangle(0, 0, self.SIZE, self.SIZE))
        self.layer = QgsVectorLayer("LineString?crs=EPSG:3857", "automatic label preview", "memory")
        self.tool = self.Tool(
            self.canvas, SimpleNamespace(crs=lambda: self.crs), self.layer,
            freehand=True, language="en",
        )
        self.tool.freehand = False
        self.tool.edge_method = EdgeDetector.METHOD_INK
        self.tool.edge_weight = 1.0
        self.tool._push_message = mock.Mock()
        self.dispatcher = _TaskDispatcher()
        self.task_patch = mock.patch.object(QgsApplication, "taskManager", return_value=self.dispatcher)
        self.task_patch.start()
        self._publish_scene()

    def tearDown(self):
        if hasattr(self, "tool"):
            self.tool.dispose()
            self.tool.deleteLater()
            self.canvas.deleteLater()
            self.application.processEvents()
        if hasattr(self, "task_patch"):
            self.task_patch.stop()

    def _publish_scene(self, *, angle=0.0, inner=False, intact_outer=False):
        from scipy.ndimage import rotate
        from ai_vectorizer.core import edge_detector

        np = self.np
        image = np.full((self.SIZE, self.SIZE, 3), 245, dtype=np.uint8)
        image[60:101, 74] = 35
        image[60:101, 100] = 35
        for x in (78, 90):
            image[100:114, x:x + 7] = 35
            image[102:112, x + 2:x + 5] = 245
        # A nearby continuous contour is cheaper than much of the blank,
        # but cannot become the far fragment of the chosen index contour.
        image[55:130, 65] = 35
        image[55:130, 109] = 35
        image[129, 65:110] = 35
        if intact_outer:
            # The intact outer U and both label fragments share a native
            # network far from the text. An inferred alternative therefore
            # exists, but must never replace that visible outside route.
            image[60, 65:75] = 35
            image[60, 100:110] = 35
        if inner:
            image[70:97, 78] = 35
            image[70:97, 96] = 35
            image[96, 78:97] = 35
        radians = np.deg2rad(angle)
        self.rotation = np.asarray((
            (np.cos(radians), np.sin(radians)),
            (-np.sin(radians), np.cos(radians)),
        ))
        self.rotation_center = np.asarray(((self.SIZE - 1) / 2,) * 2)
        if angle:
            image = rotate(image, angle, reshape=False, order=0, mode="constant", cval=245)
        self.root = self._rotate_point((65, 84) if intact_outer else (74, 84))
        self.target = self._rotate_point((109, 84) if intact_outer else (100, 84))
        self.neighbour = self._rotate_point((109, 84))
        # Exercise the actual portable Ink implementation, including the
        # thinning backend shipped in the current macOS QGIS environment.
        with mock.patch.object(edge_detector, "_skimage_skeletonize", None), mock.patch.object(
            edge_detector, "get_cv2", return_value=None,
        ):
            self.evidence = edge_detector.EdgeDetector.detect_ink_evidence(image)
        image.setflags(write=False)
        self.image = image
        self.tool._publish_edge_cache(
            rgb_image=image, edges=self.evidence.centerline.astype(np.uint8) * 255,
            evidence=self.evidence, cost_map=None,
            read_extent=QgsRectangle(0, 0, self.SIZE, self.SIZE),
            output_size=(self.SIZE, self.SIZE), tile_origin=(0, 0),
            cache_identity=("printed counters and broad contour", angle, inner, intact_outer),
            recovery_compatible=True,
        )

    def _rotate_point(self, point):
        value = self.rotation_center + self.rotation @ (self.np.asarray(point) - self.rotation_center)
        return tuple(int(component) for component in self.np.rint(value))

    def _unrotate(self, route):
        points = self.np.asarray(route, dtype=float)
        return (points - self.rotation_center) @ self.rotation + self.rotation_center

    @staticmethod
    def _coordinates(points):
        return tuple((point.x(), point.y()) for point in points)

    def _start(self, *, dispatch=True):
        anchor = self.tool.pixel_to_map(*self.root)
        self.tool.is_tracing = True
        self.tool.start_point = QgsPointXY(anchor)
        self.tool.path_points = [QgsPointXY(anchor)]
        self.tool.checkpoints = [0]
        self.tool.last_map_point = QgsPointXY(anchor)
        self.tool.last_input_point = QgsPointXY(anchor)
        if dispatch:
            self.assertTrue(self.tool._request_livewire_tree(force=True))
        return anchor

    def _target_map(self):
        return self.tool.pixel_to_map(*self.target)

    def _assert_broad_route(self, route):
        self.assertEqual(tuple(route[0]), tuple(map(float, self.root)))
        self.assertEqual(tuple(route[-1]), tuple(map(float, self.target)))
        points = self._unrotate(route)
        middle = points[(points[:, 0] >= 81) & (points[:, 0] <= 94)]
        self.assertGreater(len(middle), 20)
        # The synthetic broad curve must pass below the glyph's top edge,
        # and remain far inside the independent outer contour at y=129.
        self.assertGreater(float(middle[:, 1].max()), 109.0)
        self.assertLess(float(points[:, 1].max()), 119.0)
        self.assertGreaterEqual(float(points[:, 0].min()), 72.0)
        self.assertLessEqual(float(points[:, 0].max()), 102.0)

    def test_unmocked_rgb_to_normal_click_save_and_one_undo(self):
        from ai_vectorizer.core.label_occlusion import detect_label_occlusions
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        self.assertTrue(detect_label_occlusions(self.image))
        self._start()
        tree = self.tool._livewire_tree
        self.assertIsNotNone(tree.occluded_connection)
        route, provenance = tree.trace_with_provenance(self.target)
        self.assertEqual(provenance, "automatic-label-gap")
        self._assert_broad_route(route)
        self.assertLess(max(y for _x, y in tree._trace_without_occlusion(self.target)), 102)
        # Requesting the actual neighbouring contour cannot silently use
        # the inferred far fragment as a way to hop to that target.
        self.assertIsNone(tree.occluded_connection.trace(self.neighbour))
        self.tool._present_livewire_cursor_preview(
            self._target_map(), request_tree=False, schedule_recovery=False,
        )
        preview = self._coordinates(self.tool.preview_path)
        self.assertGreater(len(preview), 20)
        expected = self._coordinates(self.tool.path_points) + preview
        self.tool.toMapCoordinates = lambda _point: QgsPointXY(self._target_map())
        self.tool.canvasPressEvent(SimpleNamespace(
            pos=lambda: QPointF(*self.target),
            button=lambda: _qt_value("LeftButton", "MouseButton"), modifiers=lambda: 0,
        ))
        self.assertEqual(self._coordinates(self.tool.path_points), expected)
        before = self.layer.undoStack().index()
        self.assertTrue(self.tool.save_to_layer(closed=False, elevation=600.0))
        self.assertEqual(self.layer.featureCount(), 1)
        feature = next(self.layer.getFeatures())
        self.assertEqual(self._coordinates(feature.geometry().asPolyline()), expected)
        self.assertEqual(self.layer.dataProvider().featureCount(), 0)
        self.assertEqual(self.layer.undoStack().index(), before + 1)
        self.layer.undoStack().undo()
        self.assertEqual(self.layer.featureCount(), 0)

    def test_diagonal_rgb_counters_keep_the_broad_contour(self):
        self._publish_scene(angle=30.0)
        self._start()
        tree = self.tool._livewire_tree
        self.assertIsNotNone(tree.occluded_connection)
        route, provenance = tree.trace_with_provenance(self.target)
        self.assertEqual(provenance, "automatic-label-gap")
        self._assert_broad_route(route)
        self.assertIsNone(tree.occluded_connection.trace(self.neighbour))
        self.tool._present_livewire_cursor_preview(
            self._target_map(), request_tree=False, schedule_recovery=False,
        )
        self.assertGreater(len(self.tool.preview_path), 20)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_detection_runs_in_worker_and_hover_does_no_image_preparation(self):
        from ai_vectorizer.core import label_occlusion, livewire, occluded_connection
        from ai_vectorizer.tools import smart_trace_tool

        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        main_thread = threading.get_ident()
        threads = []
        detect = label_occlusion.detect_label_occlusions

        def record(*args, **kwargs):
            threads.append(threading.get_ident())
            return detect(*args, **kwargs)

        outcome = []
        with mock.patch.object(label_occlusion, "detect_label_occlusions", side_effect=record):
            worker = threading.Thread(target=lambda: outcome.append(task.run()))
            worker.start()
            worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome, [True], task.error)
        self.assertTrue(threads)
        self.assertNotIn(main_thread, threads)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertIsNotNone(task.tree.occluded_connection)
        task.finished(True)
        with mock.patch.object(
            label_occlusion, "detect_label_occlusions", side_effect=AssertionError("detected on hover"),
        ), mock.patch.object(
            occluded_connection, "build_occluded_connection", side_effect=AssertionError("rebuilt on hover"),
        ), mock.patch.object(
            smart_trace_tool, "build_livewire_tree", side_effect=AssertionError("prepared on hover"),
        ), mock.patch.object(
            livewire, "_line_features", side_effect=AssertionError("filtered on hover"),
        ):
            for _ in range(3):
                self.tool._present_livewire_cursor_preview(
                    self._target_map(), request_tree=False, schedule_recovery=False,
                )
        self.assertGreater(len(self.tool.preview_path), 20)
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_manual_and_active_learned_avoidance_keep_existing_costs(self):
        from dataclasses import replace
        from ai_vectorizer.core.learned_prior import LearnedLinePrior
        from ai_vectorizer.core.livewire import build_livewire_tree
        from ai_vectorizer.core.trace_guidance import TraceGuidance

        self._start()
        self.assertIsNotNone(self.tool._livewire_tree.occluded_connection)
        self.dispatcher.immediate = False
        for mode in ("manual", "learned"):
            with self.subTest(mode=mode):
                avoid = self.np.zeros(self.evidence.shape, dtype=self.np.float32)
                avoid[98:118, 74:101] = 1.0
                self.tool.cached_trace_guidance = TraceGuidance(avoid) if mode == "manual" else None
                self.assertTrue(self.tool._request_livewire_tree(force=True))
                task = self.dispatcher.tasks[-1]
                if mode == "learned":
                    task.learned_prior = LearnedLinePrior(self.np.zeros_like(avoid), avoid)
                    task.config = replace(task.config, learned_avoidance_cost_weight=6.0)
                self.assertTrue(task.run(), task.error)
                self.assertIsNone(task.tree.occluded_connection)
                previous = build_livewire_tree(
                    task.image, task.edges, task.anchor_pixel, strength=task.strength,
                    incoming_direction=task.incoming_direction, evidence=task.evidence,
                    guidance=task.guidance, learned_prior=task.learned_prior,
                    grid_prior=task.grid_prior,
                    config=replace(task.config, automatic_label_occlusion=False),
                )
                self.assertEqual(task.tree.trace(self.target), previous.trace(self.target))
        self.assertEqual(len(self.tool.path_points), 1)

    def test_cancel_and_stale_raster_results_preserve_confirmed_path(self):
        self.dispatcher.immediate = False
        self._start()
        cancelled = self.dispatcher.tasks[-1]
        cancelled.cancel()
        self.assertFalse(cancelled.run())
        cancelled.finished(False)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        stale = self.dispatcher.tasks[-1]
        self.assertTrue(stale.run(), stale.error)
        self.assertIsNotNone(stale.tree.occluded_connection)
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        current = self.dispatcher.tasks[-1]
        confirmed = self._coordinates(self.tool.path_points)
        stale.finished(True)
        self.assertIs(self.tool._livewire_task, current)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.tool._on_raster_data_changed()
        self.tool._edge_cache_timer.stop()
        self.assertFalse(current.run())
        current.finished(True)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_rapid_click_waits_and_reset_discards_late_result(self):
        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        self.assertTrue(self.tool._defer_click_until_livewire_ready(self._target_map()))
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertTrue(task.run(), task.error)
        task.finished(True)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        pixels = [self.tool.map_to_pixel(point) for point in self.tool.path_points]
        self.assertEqual(pixels[-1], self.target)
        self.assertGreater(max(y for _x, y in pixels), 109)
        accepted = self._coordinates(self.tool.path_points)
        task.finished(True)
        self.assertEqual(self._coordinates(self.tool.path_points), accepted)
        pending = self.dispatcher.tasks[-1]
        self.tool.reset_tracing()
        pending.finished(True)
        self.assertFalse(self.tool.path_points)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_cancellation_during_detection_publishes_no_partial_tree(self):
        from ai_vectorizer.core import label_occlusion

        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        detect = label_occlusion.detect_label_occlusions

        def cancel_after_detection(*args, **kwargs):
            regions = detect(*args, **kwargs)
            self.assertTrue(regions)
            task.cancel()
            return regions

        with mock.patch.object(label_occlusion, "detect_label_occlusions", side_effect=cancel_after_detection):
            self.assertFalse(task.run())
        task.finished(False)
        self.assertIsNone(task.tree)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_zero_assist_skips_detection_and_keeps_literal_cursor(self):
        from ai_vectorizer.core import label_occlusion
        from ai_vectorizer.tools import smart_trace_tool

        self.tool.edge_weight = 0.0
        self._start(dispatch=False)
        with mock.patch.object(label_occlusion, "detect_label_occlusions") as detect, mock.patch.object(
            smart_trace_tool, "build_livewire_tree",
        ) as build:
            self.assertFalse(self.tool._request_livewire_tree(force=True))
            self.tool._present_livewire_cursor_preview(
                self._target_map(), request_tree=False, schedule_recovery=False,
            )
        detect.assert_not_called()
        build.assert_not_called()
        self.assertFalse(self.dispatcher.tasks)
        self.assertEqual(self._coordinates(self.tool.preview_path), self._coordinates([self._target_map()]))

    def test_layer_removal_and_dispose_discard_completed_results(self):
        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        self.assertTrue(task.run(), task.error)
        self.assertIsNotNone(task.tree.occluded_connection)
        self.tool._set_source_lifecycle_listeners(True)
        project = QgsProject.instance()
        project.addMapLayer(self.layer)
        confirmed = self._coordinates(self.tool.path_points)
        project.removeMapLayer(self.layer.id())
        self.application.processEvents()
        task.finished(True)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.tool.dispose()
        task.finished(True)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        self.assertFalse(self.tool.path_points)

    def test_dispose_before_callback_discards_preview_and_pending_click(self):
        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        self.assertTrue(self.tool._defer_click_until_livewire_ready(self._target_map()))
        self.assertTrue(task.run(), task.error)
        self.tool.dispose()
        task.finished(True)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        self.assertIsNone(self.tool.preview_band)
        self.assertFalse(self.tool.path_points)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_fractional_assist_blends_once_and_checks_rendered_neighbour_crossing(self):
        from dataclasses import replace
        from ai_vectorizer.core.livewire import blend_path_with_cursor

        self._start()
        full = self.tool._livewire_tree.trace(self.target)
        self._assert_broad_route(full)
        self.tool.edge_weight = 0.95
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        fractional = self.tool._livewire_tree
        route, provenance = fractional.trace_with_provenance(self.target)
        self.assertEqual(provenance, "automatic-label-gap")
        expected = blend_path_with_cursor(full, self.root, self.target, 0.95)
        self.np.testing.assert_allclose(route, expected, atol=1e-7)

        self.tool.reset_tracing()
        self.tool.edge_weight = 1.0
        self._publish_scene(inner=True)
        self._start()
        connection = self.tool._livewire_tree.occluded_connection
        self.assertIsNotNone(connection)
        candidate = connection.trace(self.target)
        self.assertIsNotNone(candidate)
        self.assertTrue(connection.accepts_display(candidate))
        half = blend_path_with_cursor(candidate, self.root, self.target, 0.5)
        # The full curve encloses the inner native U. Applying the user's
        # assist percentage pulls its displayed curve through that U.
        self.assertFalse(connection.accepts_display(half))
        self.tool.edge_weight = 0.5
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        tree = self.tool._livewire_tree
        route, provenance = tree.trace_with_provenance(self.target)
        self.assertEqual(provenance, "ink")
        self.assertEqual(route, tree._trace_without_occlusion(self.target))
        self.tool._present_livewire_cursor_preview(
            self._target_map(), request_tree=False, schedule_recovery=False,
        )
        guarded_preview = self._coordinates(self.tool.preview_path)
        # The UI's normal resampling and float conversion must produce the
        # exact same fallback preview as a tree with this feature disabled.
        self.tool._livewire_tree = replace(tree, occluded_connection=None)
        self.tool._present_livewire_cursor_preview(
            self._target_map(), request_tree=False, schedule_recovery=False,
        )
        self.assertEqual(self._coordinates(self.tool.preview_path), guarded_preview)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_intact_outside_contour_survives_partial_blend_crossing_the_label(self):
        from dataclasses import replace
        from ai_vectorizer.core.livewire import blend_path_with_cursor

        self._publish_scene(intact_outer=True)
        self.tool.edge_weight = 0.5
        self._start()
        tree = self.tool._livewire_tree
        connection = tree.occluded_connection
        self.assertIsNotNone(connection)
        outside = connection.outside_route(self.target)
        self.assertIsNotNone(outside)
        self.assertGreater(max(y for _x, y in outside), 125)
        ordinary = tree._trace_without_occlusion(self.target)
        # At 50% the cursor blend of a healthy native U passes over the text.
        # Testing the displayed path's mask intersection alone used to allow
        # an entirely different inferred U to replace this native contour.
        self.assertTrue(connection.intersects(ordinary))
        alternative = connection.trace(self.target)
        self.assertIsNotNone(alternative)
        alternative = blend_path_with_cursor(alternative, self.root, self.target, 0.5)
        self.assertTrue(connection.accepts_display(alternative))
        self.assertNotEqual(ordinary, alternative)
        route, provenance = tree.trace_with_provenance(self.target)
        self.assertEqual(provenance, "ink")
        self.assertEqual(route, ordinary)
        self.tool._present_livewire_cursor_preview(
            self._target_map(), request_tree=False, schedule_recovery=False,
        )
        protected_preview = self._coordinates(self.tool.preview_path)
        self.tool._livewire_tree = replace(tree, occluded_connection=None)
        self.tool._present_livewire_cursor_preview(
            self._target_map(), request_tree=False, schedule_recovery=False,
        )
        self.assertEqual(self._coordinates(self.tool.preview_path), protected_preview)
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertEqual(self.layer.featureCount(), 0)


if __name__ == "__main__":
    unittest.main()
