"""Real-QGIS regressions for explicit ink colour and preview ownership.

Use QGIS' Python and a disposable QGIS_CUSTOM_CONFIG_PATH. The fixture owns
real canvas items and an editable memory layer; only task dispatch is made
synchronous so the real Live-Wire task and finished callback are deterministic.
"""

import os
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
    from qgis.PyQt.QtCore import QEvent, QPointF, Qt
    from qgis.PyQt.QtGui import QKeyEvent

    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError("QGIS bindings are required for this job") from exc
    HAS_QGIS = False


def _qt_value(name, group):
    value = getattr(Qt, name, None)
    return value if value is not None else getattr(getattr(Qt, group), name)


class _TaskDispatcher:
    """Dispatch the actual QgsTask body now or retain it for a late callback."""

    def __init__(self, immediate=True):
        self.immediate = immediate
        self.tasks = []

    def addTask(self, task):
        self.tasks.append(task)
        if self.immediate:
            task.finished(bool(task.run()))
        return True


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class InkColorQgisTests(unittest.TestCase):
    BROWN = (156, 104, 60)
    BLACK = (24, 24, 24)
    BLUE = (45, 105, 170)
    HEIGHT = 64
    WIDTH = 96

    @classmethod
    def setUpClass(cls):
        cls._owns_application = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        if cls._owns_application:
            cls.application.initQgis()
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool

        cls.SmartTraceTool = SmartTraceTool

    @classmethod
    def tearDownClass(cls):
        if cls._owns_application:
            cls.application.exitQgis()

    def setUp(self):
        import numpy as np

        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.core.line_evidence import LineEvidence
        from ai_vectorizer.core.livewire import is_livewire_available

        if not is_livewire_available():
            self.skipTest("Colour-guided Live-Wire requires optional SciPy")
        self.np = np
        self.crs = QgsCoordinateReferenceSystem("EPSG:3857")
        self.canvas = QgsMapCanvas()
        self.canvas.setDestinationCrs(self.crs)
        self.canvas.setExtent(QgsRectangle(0, 0, self.WIDTH, self.HEIGHT))
        self.layer = QgsVectorLayer("LineString?crs=EPSG:3857", "ink colour", "memory")
        self.states = []
        self.tool = self.SmartTraceTool(
            self.canvas,
            SimpleNamespace(crs=lambda: self.crs),
            self.layer,
            freehand=True,
            language="en",
            ink_color_callback=lambda rgb, available: self.states.append((rgb, available)),
        )
        tool = self.tool
        tool.freehand = False
        tool.edge_method = EdgeDetector.METHOD_INK
        tool.edge_weight = 1.0
        tool.is_tracing = True
        # This exact source-grid transform uses the production coordinate
        # conversion, rather than replacing map/pixel methods with lambdas.
        tool.cache_transform = {
            "x_min": 0.0, "y_max": float(self.HEIGHT), "px_w": 1.0, "px_h": 1.0,
        }
        anchor = tool.pixel_to_map(8, 20)
        tool.start_point = QgsPointXY(anchor)
        tool.path_points = [QgsPointXY(anchor)]
        tool.checkpoints = [0]
        tool.last_map_point = QgsPointXY(anchor)
        tool.last_input_point = QgsPointXY(anchor)
        tool._push_message = mock.Mock()
        self.dispatcher = _TaskDispatcher()
        self.task_patch = mock.patch.object(
            QgsApplication, "taskManager", return_value=self.dispatcher,
        )
        self.task_patch.start()

        self.image = np.full((self.HEIGHT, self.WIDTH, 3), 245, dtype=np.uint8)
        mask = np.zeros((self.HEIGHT, self.WIDTH), dtype=bool)
        for y, color in ((20, self.BROWN), (32, self.BLACK), (44, self.BLUE)):
            mask[y, 8:88] = True
            self.image[y, 8:88] = color
        score = mask.astype(np.float32)
        self.evidence = LineEvidence(
            center_score=score,
            centerline=mask,
            tangent_x=score,
            tangent_y=np.zeros_like(score),
            coherence=score,
        )
        self.edges = np.where(mask, 255, 0).astype(np.uint8)
        self._publish_cache()
        self.assertIsNotNone(tool._livewire_tree)
        self.assertTrue(self.layer.isEditable())

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

    def _publish_cache(self):
        self.tool._publish_edge_cache(
            rgb_image=self.image,
            edges=self.edges,
            evidence=self.evidence,
            cost_map=None,
            read_extent=QgsRectangle(0, 0, self.WIDTH, self.HEIGHT),
            output_size=(self.WIDTH, self.HEIGHT),
            tile_origin=(0, 0),
            cache_identity=("synthetic native Byte RGB", self.tool._cache_generation),
            recovery_compatible=True,
        )

    def _hover(self, x=18, y=20):
        self.tool.last_hover_pos = self.tool.pixel_to_map(x, y)

    def _sample(self):
        self._hover()
        self.assertTrue(self.tool.sample_ink_color_from_hover())
        self.assertEqual(self.tool._ink_color_profile.sample_rgb, self.BROWN)

    def _show_preview(self, x=78, y=20):
        target = self.tool.pixel_to_map(x, y)
        self.tool._present_livewire_cursor_preview(
            target, request_tree=False, schedule_recovery=False,
        )
        self.assertGreater(len(self.tool.preview_path), 2)
        self.assertGreater(self.tool.preview_band.numberOfVertices(), 2)
        return target, self._coordinates(self.tool.preview_path)

    def _key(self, name, modifiers=None):
        kind = getattr(QEvent, "KeyPress", None)
        if kind is None:
            kind = QEvent.Type.KeyPress
        if modifiers is None:
            modifiers = _qt_value("NoModifier", "KeyboardModifier")
        event = QKeyEvent(kind, _qt_value(name, "Key"), modifiers)
        self.tool.keyPressEvent(event)
        return event

    def _add_avoidance(self):
        self.assertTrue(self.tool._handle_manual_avoidance_click(self.tool.pixel_to_map(40, 16)))
        self.assertTrue(self.tool._handle_manual_avoidance_click(self.tool.pixel_to_map(45, 24)))
        self.assertEqual(len(self.tool._manual_avoidance_regions), 1)
        self.assertGreater(self.tool.manual_avoidance_band.numberOfVertices(), 0)

    def test_real_c_and_shift_c_events_change_only_optional_guidance(self):
        confirmed = self._coordinates(self.tool.path_points)
        self._hover()
        self.assertTrue(self._key("Key_C").isAccepted())
        self.assertEqual(self.tool._ink_color_profile.sample_rgb, self.BROWN)
        self.assertIsNotNone(self.tool.cached_trace_guidance)
        self.assertEqual(self.states[-1], (self.BROWN, True))

        self.assertTrue(self._key("Key_C", _qt_value("ShiftModifier", "KeyboardModifier")).isAccepted())

        self.assertIsNone(self.tool._ink_color_profile)
        self.assertIsNone(self.tool.cached_trace_guidance)
        self.assertEqual(self.states[-1], (None, True))
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_paper_and_gray_samples_keep_profile_preview_and_anchors(self):
        self._sample()
        profile = self.tool._ink_color_profile
        _target, preview = self._show_preview()
        confirmed = self._coordinates(self.tool.path_points)
        for x, y in ((35, 8), (35, 32)):
            with self.subTest(pixel=(x, y)):
                self._hover(x, y)
                self.assertFalse(self.tool.sample_ink_color_from_hover())
                self.assertIs(self.tool._ink_color_profile, profile)
                self.assertEqual(self._coordinates(self.tool.preview_path), preview)
                self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
                self.assertEqual(self.states[-1], (self.BROWN, True))

    def test_manual_avoidance_max_composes_and_survives_color_clear(self):
        from ai_vectorizer.core.ink_color import guidance_from_ink_color
        from ai_vectorizer.core.trace_guidance import guidance_from_boxes

        self._sample()
        color = guidance_from_ink_color(self.image, self.evidence, self.tool._ink_color_profile)
        confirmed = self._coordinates(self.tool.path_points)
        self._add_avoidance()
        avoidance = guidance_from_boxes(
            self.evidence.shape,
            self.tool._manual_avoidance_cache_boxes(),
            feather_pixels=self.tool.MANUAL_AVOIDANCE_FEATHER_PIXELS,
        )
        self.np.testing.assert_array_equal(
            self.tool.cached_trace_guidance.avoidance_score,
            self.np.maximum(color.avoidance_score, avoidance.avoidance_score),
        )
        self.assertTrue(self.tool.clear_ink_color())
        self.np.testing.assert_array_equal(
            self.tool.cached_trace_guidance.avoidance_score, avoidance.avoidance_score,
        )
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertEqual(len(self.tool._manual_avoidance_regions), 1)
        self.assertGreater(self.tool.manual_avoidance_band.numberOfVertices(), 0)

    def test_pan_rebuild_preserves_profile_and_reports_waiting_then_active(self):
        self._sample()
        profile = self.tool._ink_color_profile
        old_score = self.tool.cached_trace_guidance.avoidance_score.copy()
        _target, _preview = self._show_preview()
        confirmed = self._coordinates(self.tool.path_points)

        self.tool._schedule_edge_cache_update()
        self.tool._edge_cache_timer.stop()

        self.assertIs(self.tool._ink_color_profile, profile)
        self.assertIsNone(self.tool.cached_trace_guidance)
        self.assertFalse(self.tool.preview_path)
        self.assertTrue(self.tool.cache_dirty)
        self.assertEqual(self.states[-1], (self.BROWN, False))
        self._publish_cache()
        self.assertFalse(self.tool.cache_dirty)
        self.assertEqual(self.states[-1], (self.BROWN, True))
        self.np.testing.assert_array_equal(self.tool.cached_trace_guidance.avoidance_score, old_score)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)

    def test_reset_and_source_pixel_change_clear_explicit_profile(self):
        self._sample()
        self.tool._on_raster_data_changed()
        self.assertIsNone(self.tool._ink_color_profile)
        self.assertEqual(self.states[-1], (None, False))
        self._publish_cache()
        self._sample()

        self.tool.reset_tracing()

        self.assertIsNone(self.tool._ink_color_profile)
        self.assertIsNone(self.tool.cached_trace_guidance)
        self.assertFalse(self.tool.path_points)
        self.assertEqual(self.states[-1], (None, False))

    def test_pending_clicked_target_refuses_sample_and_clear_until_real_callback(self):
        self._sample()
        profile = self.tool._ink_color_profile
        confirmed = self._coordinates(self.tool.path_points)
        self.dispatcher.immediate = False
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        task = self.dispatcher.tasks[-1]
        target = self.tool.pixel_to_map(78, 20)
        self.assertTrue(self.tool._defer_click_until_livewire_ready(target))
        preview = self._coordinates(self.tool.preview_path)
        self._hover(18, 44)

        self.assertFalse(self.tool.sample_ink_color_from_hover())
        self.assertFalse(self.tool.clear_ink_color())
        self.assertIs(self.tool._ink_color_profile, profile)
        self.assertEqual(self.tool._pending_livewire_accept_point, target)
        self.assertEqual(self._coordinates(self.tool.preview_path), preview)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)

        self.assertTrue(task.run(), task.error)
        task.finished(True)

        self.assertIsNone(self.tool._pending_livewire_accept_point)
        self.assertIs(self.tool._ink_color_profile, profile)
        self.assertGreater(len(self.tool.path_points), len(confirmed))
        self.assertEqual(self.tool.path_points[-1], target)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_clearing_profile_discards_completed_but_stale_tree_callback(self):
        self._sample()
        self._show_preview()
        confirmed = self._coordinates(self.tool.path_points)
        self.dispatcher.immediate = False
        self.assertTrue(self.tool._request_livewire_tree(force=True))
        stale = self.dispatcher.tasks[-1]
        self.assertTrue(stale.run(), stale.error)

        self.assertTrue(self.tool.clear_ink_color())
        current = self.dispatcher.tasks[-1]
        self.assertIsNot(stale, current)
        self.assertFalse(self.tool.preview_path)
        stale.finished(True)

        self.assertIs(self.tool._livewire_task, current)
        self.assertIsNone(self.tool._livewire_tree)
        self.assertFalse(self.tool.preview_path)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertTrue(current.run(), current.error)
        current.finished(True)
        self.assertIsNotNone(self.tool._livewire_tree)
        self.assertGreater(len(self.tool.preview_path), 2)

    def test_guided_preview_is_exact_accepted_saved_geometry_and_one_undo(self):
        self._sample()
        target, preview = self._show_preview()
        confirmed = self._coordinates(self.tool.path_points)
        # Route the actual map click through the production event handler;
        # only the event's screen-to-map conversion is deterministic here.
        self.tool.toMapCoordinates = lambda _pos: QgsPointXY(target)
        event = SimpleNamespace(
            pos=lambda: QPointF(78, 20),
            button=lambda: _qt_value("LeftButton", "MouseButton"),
            modifiers=lambda: 0,
        )
        self.tool.canvasPressEvent(event)
        expected = confirmed + preview
        self.assertEqual(self._coordinates(self.tool.path_points), expected)
        self.assertGreater(self.tool.confirm_band.numberOfVertices(), 2)
        self.assertEqual(self.layer.featureCount(), 0)
        undo_before = self.layer.undoStack().index()

        self.assertTrue(self.tool.save_to_layer(closed=False, elevation=100.0))

        self.assertTrue(self.layer.isEditable())
        self.assertTrue(self.layer.isModified())
        self.assertEqual(self.layer.featureCount(), 1)
        self.assertEqual(self.layer.dataProvider().featureCount(), 0)
        saved = next(self.layer.getFeatures())
        self.assertEqual(self._coordinates(saved.geometry().asPolyline()), expected)
        self.assertEqual(saved["elevation"], 100.0)
        self.assertEqual(self.layer.undoStack().index(), undo_before + 1)
        self.layer.undoStack().undo()
        self.assertEqual(self.layer.featureCount(), 0)
        self.assertEqual(self.layer.dataProvider().featureCount(), 0)

    def test_zero_assist_never_samples_or_builds_optional_guidance(self):
        self.tool.edge_weight = 0.0
        self._hover()
        with mock.patch(
            "ai_vectorizer.tools.smart_trace_tool.sample_ink_color",
            side_effect=AssertionError("0% must not read sample pixels"),
        ), mock.patch(
            "ai_vectorizer.tools.smart_trace_tool.guidance_from_ink_color",
            side_effect=AssertionError("0% must not build colour guidance"),
        ):
            self.assertFalse(self.tool.sample_ink_color_from_hover())
            self.assertIsNone(self.tool._refresh_trace_guidance())
        self.assertIsNone(self.tool._ink_color_profile)
