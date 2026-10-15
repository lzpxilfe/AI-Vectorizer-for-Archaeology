"""Exercise automatic grid avoidance through real QGIS task/cache lifecycles.

Run with QGIS' Python and a disposable QGIS_CUSTOM_CONFIG_PATH.  The tests
retain actual QgsTask bodies, canvas items and memory-layer edits; task
dispatch is deterministic so a completed-but-stale result can be delivered.
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
        raise RuntimeError("QGIS bindings are required for grid lifecycle tests") from exc
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
class MapGridQgisTests(unittest.TestCase):
    SIZE = 256

    @classmethod
    def setUpClass(cls):
        cls.owns_app = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        if cls.owns_app:
            cls.application.initQgis()
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool, _InkEvidenceTask

        cls.Tool = SmartTraceTool
        cls.InkTask = _InkEvidenceTask

    @classmethod
    def tearDownClass(cls):
        if cls.owns_app:
            cls.application.exitQgis()

    def setUp(self):
        import numpy as np

        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.core.livewire import is_livewire_available

        if not is_livewire_available():
            self.skipTest("Grid-guided Live-Wire requires optional SciPy")
        self.np = np
        self.crs = QgsCoordinateReferenceSystem("EPSG:3857")
        self.canvas = QgsMapCanvas()
        self.canvas.setDestinationCrs(self.crs)
        self.canvas.setExtent(QgsRectangle(0, 0, self.SIZE, self.SIZE))
        self.layer = QgsVectorLayer("LineString?crs=EPSG:3857", "grid regression", "memory")
        self.tool = self.Tool(
            self.canvas,
            SimpleNamespace(crs=lambda: self.crs),
            self.layer,
            freehand=True,
            language="en",
        )
        self.tool.freehand = False
        self.tool.edge_method = EdgeDetector.METHOD_INK
        self.tool.edge_weight = 1.0
        self.tool._push_message = mock.Mock()
        self.detector = EdgeDetector(method=EdgeDetector.METHOD_INK)
        self.image = np.full((self.SIZE, self.SIZE, 3), 245, dtype=np.uint8)
        for pixel in (48, 128, 208):
            self.image[:, pixel] = (35, 35, 35)
            self.image[pixel, :] = (35, 35, 35)
        # This curved brown line crosses the grid but is not itself a grid
        # candidate.  The same visible contour owns both evaluation clicks.
        for x in range(8, self.SIZE - 8):
            y = self._contour_y(x)
            self.image[y - 1:y + 2, x] = (145, 95, 55)
        self.image.setflags(write=False)
        self.dispatcher = _TaskDispatcher()
        self.task_patch = mock.patch.object(
            QgsApplication, "taskManager", return_value=self.dispatcher,
        )
        self.task_patch.start()

    def tearDown(self):
        if hasattr(self, "tool"):
            self.tool.dispose()
            self.tool.deleteLater()
            self.canvas.deleteLater()
            self.application.processEvents()
        if hasattr(self, "task_patch"):
            self.task_patch.stop()

    @staticmethod
    def _contour_y(x):
        import math

        return int(round(89 + 24 * math.sin((x - 8) * math.pi / 240)))

    @staticmethod
    def _coordinates(points):
        return tuple((point.x(), point.y()) for point in points)

    def _new_ink_task(self, **overrides):
        arguments = dict(
            detector=self.detector,
            fallback_image=self.image,
            fallback_rgb_image=self.image,
            fallback_cache_identity=("visible",),
            fallback_cache_extent=(0, 0, self.SIZE, self.SIZE),
            fallback_output_size=(self.SIZE, self.SIZE),
            evidence_image=self.image,
            recovery_image=self.image,
            recovery_compatible=True,
            recovery_disabled_reason="",
            tile_origin=(0, 0),
            enable_evidence=True,
            evidence_disabled_reason="",
            build_cost_map=False,
            edge_weight=self.tool.edge_weight,
            generation=self.tool._ink_evidence_generation,
            cache_identity=("native source grid", self.tool._cache_generation),
            cache_extent=(0, 0, self.SIZE, self.SIZE),
            output_size=(self.SIZE, self.SIZE),
            callback=self.tool._on_ink_evidence_finished,
        )
        arguments.update(overrides)
        task = self.InkTask(**arguments)
        self.tool._ink_evidence_task = task
        self.tool._pending_cache_identity = task.cache_identity
        self.tool._current_ink_cache_identity = lambda: task.cache_identity
        return task

    def _publish_current_grid(self):
        task = self._new_ink_task()
        self.assertTrue(task.run(), task.error)
        self.assertIsNotNone(task.evidence, task.evidence_error)
        self.assertIsNotNone(task.map_grid)
        self.assertGreater(float(self.np.max(task.map_grid.score)), 0.0)
        task.finished(True)
        self.assertIs(self.tool.cached_map_grid, task.map_grid)
        return task

    def _start_on_contour(self, x=16):
        anchor = self.tool.pixel_to_map(x, self._contour_y(x))
        self.tool.is_tracing = True
        self.tool.start_point = QgsPointXY(anchor)
        self.tool.path_points = [QgsPointXY(anchor)]
        self.tool.checkpoints = [0]
        self.tool.last_map_point = QgsPointXY(anchor)
        self.tool.last_input_point = QgsPointXY(anchor)
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        self.assertIsNotNone(self.tool._livewire_tree)

    def test_background_detection_is_published_only_by_current_finished_callback(self):
        from ai_vectorizer.tools import smart_trace_tool

        main_thread = threading.get_ident()
        threads = []
        original = smart_trace_tool.detect_map_grid

        def detect(*args, **kwargs):
            threads.append(threading.get_ident())
            return original(*args, **kwargs)

        task = self._new_ink_task()
        outcome = []
        with mock.patch.object(smart_trace_tool, "detect_map_grid", side_effect=detect):
            worker = threading.Thread(target=lambda: outcome.append(task.run()))
            worker.start()
            worker.join(timeout=20)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome, [True], task.error)
        self.assertTrue(threads)
        self.assertNotIn(main_thread, threads)
        self.assertIsNone(self.tool.cached_map_grid)
        self.assertEqual(self.layer.featureCount(), 0)
        task.finished(True)
        self.assertIs(self.tool.cached_map_grid, task.map_grid)
        self.assertEqual(self.tool.cached_map_grid.shape, (self.SIZE, self.SIZE))
        self.assertGreater(float(self.np.max(self.tool.cached_map_grid.score)), 0.0)
        self._start_on_contour()
        self.assertIs(self.dispatcher.tasks[-1].grid_prior, task.map_grid)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_cancelled_and_obsolete_cache_callbacks_cannot_publish_grid(self):
        from ai_vectorizer.tools import smart_trace_tool

        cancelled = self._new_ink_task()
        cancelled.cancel()
        with mock.patch.object(smart_trace_tool, "detect_map_grid") as detect:
            self.assertFalse(cancelled.run())
        detect.assert_not_called()
        self.tool._clear_edge_cache()
        cancelled.finished(False)
        self.assertIsNone(self.tool.cached_map_grid)

        late_cancel = self._new_ink_task()
        original = smart_trace_tool.detect_map_grid

        def cancel_after_detection(*args, **kwargs):
            prior = original(*args, **kwargs)
            late_cancel.cancel()
            return prior

        with mock.patch.object(
            smart_trace_tool, "detect_map_grid", side_effect=cancel_after_detection,
        ):
            self.assertFalse(late_cancel.run())
        late_cancel.finished(False)
        self.assertIsNone(self.tool.cached_map_grid)

        stale = self._new_ink_task()
        self.assertTrue(stale.run(), stale.error)
        self.assertIsNotNone(stale.map_grid)
        self.tool._clear_edge_cache()
        current = self._new_ink_task()
        stale.finished(True)
        self.assertIs(self.tool._ink_evidence_task, current)
        self.assertIsNone(self.tool.cached_map_grid)
        self.assertFalse(self.tool.path_points)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_zero_assist_skips_grid_detection_and_clears_old_snapshot(self):
        from ai_vectorizer.tools import smart_trace_tool

        self._publish_current_grid()
        self.tool.edge_weight = 0.0
        self.tool.cache_dirty = True
        with mock.patch.object(smart_trace_tool, "detect_map_grid") as detect, mock.patch.object(
            self.detector, "detect_edges",
        ) as detect_edges:
            self.tool.edge_detector = self.detector
            self.tool.update_edge_cache()
        detect.assert_not_called()
        detect_edges.assert_not_called()
        self.assertIsNone(self.tool.cached_map_grid)
        self.assertIsNone(self.tool._ink_evidence_task)

    def test_source_change_clears_grid_and_queued_tree_cannot_restore_it(self):
        self._publish_current_grid()
        self._start_on_contour()
        confirmed = self._coordinates(self.tool.path_points)
        self.dispatcher.immediate = False
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        stale = self.dispatcher.tasks[-1]
        self.assertTrue(stale.run(), stale.error)

        self.tool._on_raster_data_changed()
        self.tool._edge_cache_timer.stop()
        stale.finished(True)

        self.assertIsNone(self.tool.cached_map_grid)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertFalse(self.tool.preview_path)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_recovery_keeps_grid_manual_and_learned_avoidance_in_same_crop(self):
        from ai_vectorizer.core.learned_prior import LearnedLinePrior
        from ai_vectorizer.core.map_grid import MapGridPrior
        from ai_vectorizer.core.trace_guidance import TraceGuidance
        from ai_vectorizer.tools import smart_trace_tool

        self._publish_current_grid()
        self._start_on_contour()
        manual = self.np.zeros((self.SIZE, self.SIZE), dtype=self.np.float32)
        manual[100, 85] = 0.8
        learned = self.np.zeros_like(manual)
        learned[110, 90] = 0.9
        self.tool.cached_trace_guidance = TraceGuidance(manual)
        self.tool._tree_learned_prior = LearnedLinePrior(
            self.np.zeros_like(manual), learned,
        )
        # Explicitly enable the learned avoidance head for this composition
        # test; bundles without selected costs now default to support only.
        self.tool.learned_avoidance_weight = 6.0
        self.tool.smart_recovery_enabled = True
        self.tool.recovery_engine = SimpleNamespace(
            encode=lambda _image: "encoding",
            predict=lambda _encoding, _points, _labels: SimpleNamespace(
                mask=self.np.ones_like(manual),
            ),
        )
        self.dispatcher.immediate = False
        request = dict(
            trigger="synthetic grid crossing",
            champion_path=((40.0, 100.0), (100.0, 100.0)),
            start_pixel=(40.0, 100.0),
            target_pixel=(100.0, 100.0),
            prompt_points=self.np.asarray([[40, 100], [100, 100]], dtype=self.np.float32),
            prompt_labels=self.np.asarray([1, 1], dtype=self.np.int32),
            window_bounds=(32, 64, 128, 144),
            cache_generation=self.tool._cache_generation,
        )
        self.assertTrue(self.tool._start_recovery_request(request))
        recovery = self.dispatcher.tasks[-1]
        self.assertIs(recovery.grid_prior, self.tool.cached_map_grid)
        self.assertGreater(float(recovery.guidance.avoidance_score[100, 85]), 0)
        self.assertGreater(float(recovery.guidance.avoidance_score[110, 90]), 0)
        with mock.patch.object(
            smart_trace_tool, "build_corridor_cost_map",
            wraps=smart_trace_tool.build_corridor_cost_map,
        ) as build_cost:
            self.assertTrue(recovery.run(), recovery.error)
        bounded = build_cost.call_args.kwargs
        self.assertEqual(bounded["grid_prior"].shape, (80, 96))
        self.assertEqual(bounded["guidance"].shape, (80, 96))
        self.assertGreater(float(bounded["guidance"].avoidance_score[36, 53]), 0)
        self.assertGreater(float(bounded["guidance"].avoidance_score[46, 58]), 0)
        self.assertEqual(self.layer.featureCount(), 0)
        # A prior from a different full cache must not pass just because its
        # crop happens to fit the Recovery window.
        wrong_grid = self.np.zeros((self.SIZE + 8, self.SIZE + 8), dtype=self.np.float32)
        recovery.grid_prior = MapGridPrior(wrong_grid, wrong_grid)
        self.assertFalse(recovery.run())
        self.assertIsInstance(recovery.error, ValueError)
        self.assertIn("full Ink evidence grid", str(recovery.error))

    def test_grid_detector_failure_preserves_ink_and_disabled_native_cache_skips_it(self):
        from ai_vectorizer.tools import smart_trace_tool

        task = self._new_ink_task()
        with mock.patch.object(
            smart_trace_tool, "detect_map_grid", side_effect=ValueError("injected detector failure"),
        ):
            self.assertTrue(task.run(), task.error)
        self.assertIsNotNone(task.evidence)
        self.assertIsNone(task.map_grid)
        task.finished(True)
        self.assertIsNotNone(self.tool.cached_ink_evidence)
        self.assertIsNone(self.tool.cached_map_grid)

        for settings in ({"detect_grid": False}, {"recovery_compatible": False}):
            with self.subTest(settings=settings):
                task = self._new_ink_task(**settings)
                with mock.patch.object(smart_trace_tool, "detect_map_grid") as detect:
                    self.assertTrue(task.run(), task.error)
                detect.assert_not_called()
                self.assertIsNone(task.map_grid)

    def test_grid_preview_is_exact_saved_geometry_and_single_undo(self):
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        self._publish_current_grid()
        self._start_on_contour()
        target = self.tool.pixel_to_map(220, self._contour_y(220))
        self.tool._present_livewire_cursor_preview(
            target, request_tree=False, schedule_recovery=False,
        )
        preview = self._coordinates(self.tool.preview_path)
        self.assertGreater(len(preview), 2)
        expected = self._coordinates(self.tool.path_points) + preview
        self.tool.toMapCoordinates = lambda _point: QgsPointXY(target)
        self.tool.canvasPressEvent(SimpleNamespace(
            pos=lambda: QPointF(220, self._contour_y(220)),
            button=lambda: _qt_value("LeftButton", "MouseButton"),
            modifiers=lambda: 0,
        ))
        self.assertEqual(self._coordinates(self.tool.path_points), expected)
        undo_before = self.layer.undoStack().index()
        self.assertTrue(self.tool.save_to_layer(closed=False, elevation=100.0))
        self.assertEqual(self.layer.featureCount(), 1)
        saved = next(self.layer.getFeatures())
        self.assertEqual(self._coordinates(saved.geometry().asPolyline()), expected)
        self.assertEqual(self.layer.dataProvider().featureCount(), 0)
        self.assertEqual(self.layer.undoStack().index(), undo_before + 1)
        self.layer.undoStack().undo()
        self.assertEqual(self.layer.featureCount(), 0)


if __name__ == "__main__":
    unittest.main()
