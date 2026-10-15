"""Real QGIS lifecycles for one inferred gap in a shallow native contour.

The hidden synthetic arc is a test construction, not a reference for an
occluded historical map. Exact cached fragments isolate connection geometry;
the production background task, map-tool preview and edits remain real.
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
        raise RuntimeError("QGIS bindings are required for label-gap tests") from exc
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
class LabelGapQgisTests(unittest.TestCase):
    SIZE = 256
    ROOT = (40, 71)
    TARGET = (160, 71)

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

    @staticmethod
    def _arc_y(x):
        return 60 + 0.003 * (x - 100) ** 2

    def setUp(self):
        import numpy as np

        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.core.line_evidence import LineEvidence
        from ai_vectorizer.core.livewire import is_livewire_available

        if not is_livewire_available():
            self.skipTest("Label-gap connections require optional SciPy")
        self.np = np
        self.crs = QgsCoordinateReferenceSystem("EPSG:3857")
        self.canvas = QgsMapCanvas()
        self.canvas.setDestinationCrs(self.crs)
        self.canvas.setExtent(QgsRectangle(0, 0, self.SIZE, self.SIZE))
        self.layer = QgsVectorLayer("LineString?crs=EPSG:3857", "label-gap preview", "memory")
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

        mask = np.zeros((self.SIZE, self.SIZE), dtype=bool)
        for x in range(40, 161):
            if x <= 74 or x >= 126:
                mask[int(round(self._arc_y(x))), x] = True
        # An uninterrupted neighbouring contour must remain separate, even
        # when its ink is cheaper than the missing middle of the selected arc.
        for x in range(30, 171):
            mask[int(round(self._arc_y(x))) + 14, x] = True
        # Compact printed glyphs occupy the blank. Their short terminal
        # strokes and closed ring cannot become continuation endpoints.
        mask[56:65, 98] = True
        mask[64, 96:101] = True
        mask[55:67, 108] = True
        mask[55:67, 114] = True
        mask[55, 108:115] = True
        mask[66, 108:115] = True
        self.evidence = LineEvidence(mask.astype(np.float32), mask)
        self.image = np.full((self.SIZE, self.SIZE, 3), 245, dtype=np.uint8)
        self.image[mask] = 35
        self.image.setflags(write=False)
        self.tool._publish_edge_cache(
            rgb_image=self.image, edges=mask.astype(np.uint8) * 255,
            evidence=self.evidence, cost_map=None,
            read_extent=QgsRectangle(0, 0, self.SIZE, self.SIZE),
            output_size=(self.SIZE, self.SIZE), tile_origin=(0, 0),
            cache_identity=("shallow arc with printed label",), recovery_compatible=True,
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

    def _assert_arc_route(self, route):
        self.assertEqual(tuple(route[0]), tuple(map(float, self.ROOT)))
        self.assertEqual(tuple(route[-1]), tuple(map(float, self.TARGET)))
        middle = [(x, y) for x, y in route if 80 <= x <= 120]
        self.assertGreater(len(middle), 20)
        # This tests the known synthetic construction. A straight chord at
        # y=71, the +14px neighbouring contour and glyph loops all fail it.
        self.assertLess(max(abs(y - self._arc_y(x)) for x, y in route), 3.0)
        self.assertLess(min(y for _x, y in middle), 62.0)

    def test_worker_builds_gap_once_and_hover_does_no_image_or_graph_work(self):
        from ai_vectorizer.core import label_gap, livewire
        from ai_vectorizer.tools import smart_trace_tool

        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        main_thread = threading.get_ident()
        threads = []
        build = label_gap.build_label_gap_connectors

        def record(*args, **kwargs):
            threads.append(threading.get_ident())
            return build(*args, **kwargs)

        outcome = []
        with mock.patch.object(label_gap, "build_label_gap_connectors", side_effect=record):
            worker = threading.Thread(target=lambda: outcome.append(task.run()))
            worker.start()
            worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome, [True], task.error)
        self.assertTrue(threads)
        self.assertNotIn(main_thread, threads)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertIsNotNone(task.tree.label_gap_connection)
        self.assertIsNone(task.tree.observed_connection.trace(self.TARGET))
        self._assert_arc_route(task.tree.trace(self.TARGET))
        task.finished(True)
        with mock.patch.object(
            smart_trace_tool, "build_livewire_tree", side_effect=AssertionError("rebuilt on hover"),
        ), mock.patch.object(
            label_gap, "build_label_gap_connectors", side_effect=AssertionError("paired gaps on hover"),
        ), mock.patch.object(
            livewire, "build_label_gap_connection", side_effect=AssertionError("rebuilt gap tree on hover"),
        ), mock.patch.object(
            livewire, "_line_features", side_effect=AssertionError("filtered pixels on hover"),
        ):
            for _ in range(3):
                self.tool._present_livewire_cursor_preview(
                    self._target(), request_tree=False, schedule_recovery=False,
                )
        self.assertGreater(len(self.tool.preview_path), 2)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_manual_and_active_learned_avoidance_keep_existing_pixel_mode(self):
        from dataclasses import replace
        from ai_vectorizer.core.learned_prior import LearnedLinePrior
        from ai_vectorizer.core.livewire import build_livewire_tree
        from ai_vectorizer.core.trace_guidance import TraceGuidance

        self._start()
        self.assertIsNotNone(self.tool._livewire_tree.label_gap_connection)
        self.dispatcher.immediate = False
        for mode in ("manual", "learned"):
            with self.subTest(mode=mode):
                avoid = self.np.zeros(self.evidence.shape, dtype=self.np.float32)
                avoid[50:75, 75:126] = 1.0
                self.tool.cached_trace_guidance = TraceGuidance(avoid) if mode == "manual" else None
                self.assertTrue(self.tool._request_livewire_tree(force=True))
                task = self.dispatcher.tasks[-1]
                if mode == "learned":
                    task.learned_prior = LearnedLinePrior(self.np.zeros_like(avoid), avoid)
                    task.config = replace(task.config, learned_avoidance_cost_weight=6.0)
                self.assertTrue(task.run(), task.error)
                self.assertIsNone(task.tree.observed_connection)
                self.assertIsNone(task.tree.label_gap_connection)
                previous = build_livewire_tree(
                    task.image, task.edges, task.anchor_pixel,
                    strength=task.strength, incoming_direction=task.incoming_direction,
                    evidence=task.evidence, guidance=task.guidance,
                    learned_prior=task.learned_prior, grid_prior=task.grid_prior,
                    config=replace(task.config, label_gap_connections=False),
                )
                self.assertEqual(task.tree.trace(self.TARGET), previous.trace(self.TARGET))
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_cancelled_and_stale_gap_trees_do_not_change_confirmed_path(self):
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
        self.assertIsNotNone(stale.tree.label_gap_connection)
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

    def test_rapid_click_waits_for_gap_and_reset_discards_pending_result(self):
        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        self.assertTrue(self.tool._defer_click_until_livewire_ready(self._target()))
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertTrue(task.run(), task.error)
        task.finished(True)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        pixels = [self.tool.map_to_pixel(point) for point in self.tool.path_points]
        self.assertEqual(pixels[-1], self.TARGET)
        self.assertLess(min(y for x, y in pixels if 80 <= x <= 120), 63)
        pending = self.dispatcher.tasks[-1]
        self.tool.reset_tracing()
        pending.finished(True)
        self.assertFalse(self.tool.path_points)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_cancellation_during_gap_preparation_does_not_publish_a_tree(self):
        from ai_vectorizer.core import label_gap

        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        build = label_gap.build_label_gap_connectors

        def cancel_after_pairing(*args, **kwargs):
            connectors = build(*args, **kwargs)
            self.assertTrue(connectors)
            task.cancel()
            return connectors

        with mock.patch.object(label_gap, "build_label_gap_connectors", side_effect=cancel_after_pairing):
            self.assertFalse(task.run())
        task.finished(False)
        self.assertIsNone(task.tree)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(len(self.tool.path_points), 1)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_zero_assist_does_not_dispatch_and_keeps_literal_cursor(self):
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

    def test_layer_removal_and_dispose_discard_completed_gap_results(self):
        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        self.assertTrue(task.run(), task.error)
        self.assertIsNotNone(task.tree.label_gap_connection)
        self.tool._set_source_lifecycle_listeners(True)
        project = QgsProject.instance()
        project.addMapLayer(self.layer)
        layer_id = self.layer.id()
        confirmed = self._coordinates(self.tool.path_points)
        project.removeMapLayer(layer_id)
        self.application.processEvents()
        task.finished(True)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.tool.dispose()
        task.finished(True)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertFalse(self.tool.path_points)

    def test_dispose_discards_a_completed_gap_tree_before_its_callback(self):
        self.dispatcher.immediate = False
        self._start()
        task = self.dispatcher.tasks[-1]
        self.assertTrue(task.run(), task.error)
        self.assertIsNotNone(task.tree.label_gap_connection)
        self.tool.dispose()
        task.finished(True)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertIsNone(self.tool._pending_livewire_accept_point)
        self.assertFalse(self.tool.path_points)
        self.assertIsNone(self.tool.preview_band)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_fractional_strength_blends_the_single_gap_once(self):
        from ai_vectorizer.core.livewire import blend_path_with_cursor

        self._start()
        full = self.tool._livewire_tree.trace(self.TARGET)
        self._assert_arc_route(full)
        self.tool.edge_weight = 0.5
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        half = self.tool._livewire_tree
        self.assertIsNotNone(half.label_gap_connection)
        expected = blend_path_with_cursor(full, self.ROOT, self.TARGET, 0.5)
        self.np.testing.assert_allclose(half.trace(self.TARGET), expected, atol=1e-7)

    def test_gap_preview_is_saved_exactly_and_one_undo_removes_it(self):
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        self._start()
        self._assert_arc_route(self.tool._livewire_tree.trace(self.TARGET))
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
