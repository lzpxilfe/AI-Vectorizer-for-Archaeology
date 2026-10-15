"""Real QGIS worker and edit lifecycles for native observed connections.

The small synthetic scene contains an unbroken deep U and a separate nearby
stroke. Its pixel graph prefers a shortcut; the observed graph must preserve
that U without changing deferred clicks, manual costs or saved geometry.
"""

import os
import threading
from types import SimpleNamespace
import unittest
from unittest import mock


try:
    from qgis.core import (
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsPointXY,
        QgsRectangle,
        QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QPointF

    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError("QGIS bindings are required for observed connection tests") from exc
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
class ObservedConnectionQgisTests(unittest.TestCase):
    SIZE = 256
    ROOT = (80, 60)
    TARGET = (140, 60)

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
        from ai_vectorizer.core.line_evidence import LineEvidence
        from ai_vectorizer.core.livewire import is_livewire_available

        if not is_livewire_available():
            self.skipTest("Observed connections require optional SciPy")
        self.np = np
        self.crs = QgsCoordinateReferenceSystem("EPSG:3857")
        self.canvas = QgsMapCanvas()
        self.canvas.setDestinationCrs(self.crs)
        self.canvas.setExtent(QgsRectangle(0, 0, self.SIZE, self.SIZE))
        self.layer = QgsVectorLayer("LineString?crs=EPSG:3857", "deep observed bend", "memory")
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
        # Exact native observed pixels deliberately avoid detector variability;
        # the actual production QgsTask still builds both graphs in run().
        mask = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        mask[60:161, 80] = True
        mask[160, 80:141] = True
        mask[60:161, 140] = True
        # A separate inner U is only 5px away at the upper anchors. No closing
        # or dilation may turn it into a connection to the selected contour.
        mask[60:136, 85] = True
        mask[135, 85:136] = True
        mask[60:136, 135] = True
        self.evidence = LineEvidence(mask.astype(np.float32), mask)
        self.image = np.full((self.SIZE, self.SIZE, 3), 245, dtype=np.uint8)
        self.image[mask] = 35
        self.image.setflags(write=False)
        self.tool._publish_edge_cache(
            rgb_image=self.image, edges=mask.astype(np.uint8) * 255,
            evidence=self.evidence, cost_map=None,
            read_extent=QgsRectangle(0, 0, self.SIZE, self.SIZE),
            output_size=(self.SIZE, self.SIZE), tile_origin=(0, 0),
            cache_identity=("native observed U",), recovery_compatible=True,
        )

    def tearDown(self):
        if hasattr(self, "tool"):
            self.tool.dispose()
            self.tool.deleteLater()
            self.canvas.deleteLater()
            self.application.processEvents()
        if hasattr(self, "task_patch"):
            self.task_patch.stop()

    @staticmethod
    def _coordinates(points):
        return tuple((point.x(), point.y()) for point in points)

    def _start(self, *, dispatch=True):
        anchor = self.tool.pixel_to_map(*self.ROOT)
        self.tool.is_tracing = True
        self.tool.start_point = QgsPointXY(anchor)
        self.tool.path_points = [QgsPointXY(anchor)]
        self.tool.checkpoints = [0]
        self.tool.last_map_point = QgsPointXY(anchor)
        self.tool.last_input_point = QgsPointXY(anchor)
        if dispatch:
            self.assertTrue(self.tool._request_livewire_tree(force=True))
        return anchor

    def _target(self):
        return self.tool.pixel_to_map(*self.TARGET)

    def _pixels(self, points):
        return [self.tool.map_to_pixel(point) for point in points]

    def test_worker_builds_deep_connection_once_and_hover_uses_cached_tree(self):
        from ai_vectorizer.core import livewire
        from ai_vectorizer.tools import smart_trace_tool

        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        main_thread = threading.get_ident()
        threads = []
        build = livewire.build_observed_connection

        def record(*args, **kwargs):
            threads.append(threading.get_ident())
            return build(*args, **kwargs)

        outcome = []
        with mock.patch.object(livewire, "build_observed_connection", side_effect=record):
            worker = threading.Thread(target=lambda: outcome.append(task.run()))
            worker.start()
            worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome, [True], task.error)
        self.assertTrue(threads)
        self.assertNotIn(main_thread, threads)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertIsNotNone(task.tree.observed_connection)
        task.finished(True)
        self.assertIs(self.tool._livewire_tree, task.tree)
        route = task.tree.trace(self.TARGET)
        length = sum(self.np.linalg.norm(self.np.subtract(b, a)) for a, b in zip(route, route[1:]))
        self.assertGreater(length, 3 * self.np.linalg.norm(self.np.subtract(self.TARGET, self.ROOT)))
        self.assertEqual(max(y for _x, y in route), 160)
        with mock.patch.object(
            smart_trace_tool, "build_livewire_tree", side_effect=AssertionError("rebuilt on hover"),
        ), mock.patch.object(
            livewire, "build_observed_connection", side_effect=AssertionError("rebuilt observed graph on hover"),
        ), mock.patch.object(
            livewire, "_line_features", side_effect=AssertionError("filtered pixels on hover"),
        ):
            for _ in range(3):
                self.tool._present_livewire_cursor_preview(
                    self._target(), request_tree=False, schedule_recovery=False,
                )
        self.assertGreater(max(y for _x, y in self._pixels(self.tool.preview_path)), 158)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_manual_and_active_learned_avoidance_keep_pixel_cost_fallback(self):
        from ai_vectorizer.core.learned_prior import LearnedLinePrior
        from ai_vectorizer.core.trace_guidance import TraceGuidance

        self._start()
        self.assertIsNotNone(self.tool._livewire_tree.observed_connection)
        baseline = self.tool._livewire_tree.trace(self.TARGET)
        self.dispatcher.immediate = False
        for mode in ("manual", "learned"):
            with self.subTest(mode=mode):
                avoid = self.np.zeros(self.evidence.shape, dtype=self.np.float32)
                avoid[145:175, 70:151] = 1.0
                self.tool.cached_trace_guidance = TraceGuidance(avoid) if mode == "manual" else None
                self.assertTrue(self.tool._request_livewire_tree(force=True))
                task = self.dispatcher.tasks[-1]
                if mode == "learned":
                    # Actual bundles must have a selected cost before this
                    # head is enabled. Test the explicit active-cost case.
                    from dataclasses import replace
                    task.learned_prior = LearnedLinePrior(self.np.zeros_like(avoid), avoid)
                    task.config = replace(task.config, learned_avoidance_cost_weight=6.0)
                self.assertTrue(task.run(), task.error)
                self.assertIsNone(task.tree.observed_connection)
                self.assertNotEqual(task.tree.trace(self.TARGET), baseline)
                self.assertLess(max(y for _x, y in task.tree.trace(self.TARGET)), 145)
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_cancelled_and_stale_trees_cannot_publish_or_change_confirmed_path(self):
        self.dispatcher.immediate = False
        self._start()
        cancelled = self.dispatcher.tasks[-1]
        cancelled.cancel()
        self.assertFalse(cancelled.run())
        cancelled.finished(False)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        stale = self.dispatcher.tasks[-1]
        self.assertTrue(stale.run(), stale.error)
        self.assertIsNotNone(stale.tree.observed_connection)
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

    def test_rapid_pending_click_waits_for_connection_and_cancel_discards_it(self):
        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        self.assertTrue(self.tool._defer_click_until_livewire_ready(self._target()))
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertEqual(self.tool._pending_livewire_accept_point, self._target())
        self.assertTrue(task.run(), task.error)
        task.finished(True)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        pixels = self._pixels(self.tool.path_points)
        self.assertGreater(max(y for _x, y in pixels), 158)
        self.assertEqual(pixels[-1], self.TARGET)
        self.assertEqual(self.layer.featureCount(), 0)

        pending = self.dispatcher.tasks[-1]
        self.tool.reset_tracing()
        pending.finished(True)
        self.assertFalse(self.tool.path_points)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_zero_assist_does_not_dispatch_connection_and_uses_literal_cursor(self):
        from ai_vectorizer.tools import smart_trace_tool

        self.tool.edge_weight = 0.0
        self._start(dispatch=False)
        with mock.patch.object(smart_trace_tool, "build_livewire_tree") as build:
            self.assertFalse(self.tool._request_livewire_tree(force=True))
            self.tool._present_livewire_cursor_preview(
                self._target(), request_tree=False, schedule_recovery=False,
            )
        build.assert_not_called()
        self.assertFalse(self.dispatcher.tasks)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(self._coordinates(self.tool.preview_path), self._coordinates([self._target()]))

    def test_fractional_strength_blends_same_connection_once(self):
        from ai_vectorizer.core.livewire import blend_path_with_cursor

        self._start()
        full = self.tool._livewire_tree.trace(self.TARGET)
        self.tool.edge_weight = 0.5
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        half = self.tool._livewire_tree
        self.assertIsNotNone(half.observed_connection)
        expected = blend_path_with_cursor(full, self.ROOT, self.TARGET, 0.5)
        self.np.testing.assert_allclose(half.trace(self.TARGET), expected, atol=1e-7)
        self.tool._present_livewire_cursor_preview(
            self._target(), request_tree=False, schedule_recovery=False,
        )
        # Pixel-to-map smoothing may round, but half strength must retain a
        # substantial bend and cannot apply the factor a second time.
        maximum = max(y for _x, y in self._pixels(self.tool.preview_path))
        self.assertGreater(maximum, 108)
        self.assertLess(maximum, 112)

    def test_preview_is_saved_exactly_with_one_undo_for_deep_connection(self):
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        self._start()
        self.tool._present_livewire_cursor_preview(
            self._target(), request_tree=False, schedule_recovery=False,
        )
        preview = self._coordinates(self.tool.preview_path)
        self.assertGreater(len(preview), 2)
        expected = self._coordinates(self.tool.path_points) + preview
        self.tool.toMapCoordinates = lambda _point: QgsPointXY(self._target())
        self.tool.canvasPressEvent(SimpleNamespace(
            pos=lambda: QPointF(*self.TARGET),
            button=lambda: _qt_value("LeftButton", "MouseButton"), modifiers=lambda: 0,
        ))
        self.assertEqual(self._coordinates(self.tool.path_points), expected)
        before = self.layer.undoStack().index()
        self.assertTrue(self.tool.save_to_layer(closed=False, elevation=100.0))
        self.assertEqual(self.layer.featureCount(), 1)
        saved = next(self.layer.getFeatures())
        self.assertEqual(self._coordinates(saved.geometry().asPolyline()), expected)
        self.assertEqual(self.layer.dataProvider().featureCount(), 0)
        self.assertEqual(self.layer.undoStack().index(), before + 1)
        self.layer.undoStack().undo()
        self.assertEqual(self.layer.featureCount(), 0)


if __name__ == "__main__":
    unittest.main()
