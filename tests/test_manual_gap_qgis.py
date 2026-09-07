"""Real-QGIS regressions for explicit label-gap preview ownership.

Run with QGIS' Python and a disposable QGIS_CUSTOM_CONFIG_PATH. Ordinary
Python test jobs skip this module; no model download is needed.
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
        QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QPointF, Qt

    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError("QGIS bindings are required for this job") from exc
    HAS_QGIS = False


def _qt_value(name, group):
    value = getattr(Qt, name, None)
    return value if value is not None else getattr(getattr(Qt, group), name)


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class ManualGapQgisTests(unittest.TestCase):
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

        crs = QgsCoordinateReferenceSystem("EPSG:3857")
        self.canvas = QgsMapCanvas()
        self.canvas.setDestinationCrs(crs)
        self.layer = QgsVectorLayer("LineString?crs=EPSG:3857", "bridge", "memory")
        self.tool = self.SmartTraceTool(
            self.canvas,
            SimpleNamespace(crs=lambda: crs),
            self.layer,
            freehand=True,
            language="en",
        )
        tool = self.tool
        tool.freehand = False
        tool.edge_method = EdgeDetector.METHOD_INK
        tool.edge_weight = 1.0
        tool.is_tracing = True
        tool.start_point = QgsPointXY(10, 31)
        tool.path_points = [tool.start_point, QgsPointXY(60, 31)]
        tool.checkpoints = [0, 1]
        tool.last_map_point = tool.path_points[-1]
        tool.last_input_point = tool.path_points[-1]
        self.target = QgsPointXY(109, 31)
        tool.last_hover_pos = QgsPointXY(self.target)
        tool._livewire_request_point = QgsPointXY(self.target)
        tool._cache_generation = 7
        tool.cache_transform = {"x_min": 0.0, "y_max": 72.0, "px_w": 1.0, "px_h": 1.0}
        centerline = np.zeros((72, 168), dtype=bool)
        centerline[31, 8:61] = True
        centerline[31, 109:160] = True
        score = centerline.astype(np.float32)
        tool.cached_ink_evidence = LineEvidence(
            center_score=score,
            centerline=centerline,
            tangent_x=score,
            tangent_y=np.zeros_like(score),
            coherence=score,
        )
        tool.cached_edges = np.where(centerline, 255, 0).astype(np.uint8)
        tool.cached_rgb_image = np.zeros((72, 168, 3), dtype=np.uint8)
        tool.map_to_pixel_float = lambda point: (point.x(), point.y())
        tool.map_to_pixel = lambda point: (round(point.x()), round(point.y()))
        tool.pixel_to_map = lambda x, y: QgsPointXY(x, y)
        tool.toMapCoordinates = lambda point: QgsPointXY(point.x(), point.y())
        self.canvas.mapUnitsPerPixel = lambda: 1.0
        tool._request_livewire_tree = mock.Mock(return_value=False)
        tool._push_message = mock.Mock()

    def tearDown(self):
        self.tool.reset_tracing()
        self.tool.deleteLater()
        self.canvas.deleteLater()
        self.application.processEvents()

    def _mouse_event(self, point=None, modifiers=0):
        point = point or self.target
        return SimpleNamespace(
            pos=lambda: QPointF(point.x(), point.y()),
            button=lambda: _qt_value("LeftButton", "MouseButton"),
            buttons=lambda: 0,
            modifiers=lambda: modifiers,
        )

    @staticmethod
    def _coordinates(points):
        return tuple((point.x(), point.y()) for point in points)

    def _preview(self):
        self.assertTrue(self.tool.preview_manual_gap_bridge())
        return self._coordinates(self.tool.preview_path)

    def test_late_livewire_result_cannot_replace_visible_bridge(self):
        tool = self.tool
        bridge = self._preview()
        task = SimpleNamespace(
            generation=tool._livewire_generation,
            anchor_pixel=(60, 31),
        )
        tool._livewire_task = task
        tool._livewire_request_point = QgsPointXY(self.target)
        present = mock.Mock(side_effect=lambda *_a, **_k: tool.preview_path.clear())
        tool._present_livewire_cursor_preview = present

        tool._on_livewire_tree_finished(task, True, SimpleNamespace(root=(60, 31)), None)

        present.assert_not_called()
        self.assertEqual(self._coordinates(tool.preview_path), bridge)
        self.assertTrue(tool._manual_gap_bridge_preview_matches(self.target))

    def test_hover_back_from_dock_preserves_bridge_until_explicit_action(self):
        bridge = self._preview()
        confirmed = self._coordinates(self.tool.path_points)

        self.tool.canvasMoveEvent(self._mouse_event(QgsPointXY(155, 60)))

        self.assertEqual(self._coordinates(self.tool.preview_path), bridge)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertTrue(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_parallel_contour_click_is_not_a_bridge_confirmation(self):
        self._preview()
        self.assertFalse(
            self.tool._manual_gap_bridge_preview_matches(QgsPointXY(109, 36))
        )

    def test_zoomed_out_confirmation_is_bounded_in_source_pixels(self):
        self._preview()
        self.canvas.mapUnitsPerPixel = lambda: 10.0
        self.assertFalse(
            self.tool._manual_gap_bridge_preview_matches(QgsPointXY(109, 36))
        )

    def test_acceptance_commits_exact_preview_and_stored_endpoint(self):
        bridge = self._preview()
        confirmed = self._coordinates(self.tool.path_points)

        self.tool.canvasPressEvent(self._mouse_event(QgsPointXY(109.5, 31)))

        self.assertEqual(self._coordinates(self.tool.path_points), confirmed + bridge)
        self.assertEqual(self.tool.last_input_point, self.target)
        self.assertEqual(self.tool.checkpoints, [0, 1, len(confirmed + bridge) - 1])
        self.assertFalse(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_modifier_clicks_do_not_accept_bridge(self):
        for modifier_name in ("AltModifier", "ShiftModifier", "ControlModifier"):
            with self.subTest(modifier=modifier_name):
                bridge = self._preview()
                confirmed = self._coordinates(self.tool.path_points)
                self.tool._defer_click_until_livewire_ready = mock.Mock(return_value=True)
                self.tool._handle_manual_avoidance_click = mock.Mock()
                self.tool.canvasPressEvent(
                    self._mouse_event(modifiers=_qt_value(modifier_name, "KeyboardModifier"))
                )
                self.assertNotEqual(
                    self._coordinates(self.tool.path_points), confirmed + bridge
                )

    def test_stale_cache_anchor_or_mutated_preview_cannot_be_confirmed(self):
        for mutation in ("cache", "anchor", "preview"):
            with self.subTest(mutation=mutation):
                self.tool.path_points[-1] = QgsPointXY(60, 31)
                self._preview()
                if mutation == "cache":
                    self.tool._cache_generation += 1
                elif mutation == "anchor":
                    self.tool.path_points[-1] = QgsPointXY(59, 31)
                else:
                    self.tool.preview_path[0] = QgsPointXY(90, 60)
                self.assertFalse(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_clearing_cache_discards_bridge_geometry_without_rewinding_trace(self):
        self._preview()
        confirmed = self._coordinates(self.tool.path_points)

        self.tool._clear_edge_cache()

        self.assertFalse(self.tool.preview_path)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertFalse(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_escape_discards_bridge_but_keeps_confirmed_trace(self):
        bridge = self._preview()
        confirmed = self._coordinates(self.tool.path_points)
        event = SimpleNamespace(
            key=lambda: _qt_value("Key_Escape", "Key"),
            modifiers=lambda: 0,
            accept=mock.Mock(),
            ignore=mock.Mock(),
        )

        self.tool.keyPressEvent(event)

        self.assertTrue(self.tool.is_tracing)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertNotEqual(self._coordinates(self.tool.preview_path), bridge)
        self.assertFalse(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_g_shortcut_previews_current_hover_without_moving_to_dock(self):
        confirmed = self._coordinates(self.tool.path_points)
        event = SimpleNamespace(
            key=lambda: _qt_value("Key_G", "Key"),
            modifiers=lambda: 0,
            isAutoRepeat=lambda: False,
            accept=mock.Mock(),
            ignore=mock.Mock(),
        )

        self.tool.keyPressEvent(event)

        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertTrue(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_zero_assist_never_reads_tangents_or_builds_a_bridge(self):
        self.tool.edge_weight = 0.0
        self.tool._manual_gap_bridge_tangent = mock.Mock(side_effect=AssertionError("unexpected evidence read"))
        self.assertFalse(self.tool.preview_manual_gap_bridge())
        self.assertFalse(self.tool.preview_path)

    def test_recovery_retry_cannot_replace_bridge(self):
        bridge = self._preview()
        self.tool.smart_recovery_requested = True
        self.tool.smart_recovery_enabled = True
        self.tool._schedule_smart_recovery = mock.Mock(return_value=True)

        self.assertFalse(self.tool.retry_current_segment())

        self.tool._schedule_smart_recovery.assert_not_called()
        self.assertEqual(self._coordinates(self.tool.preview_path), bridge)

    def test_partial_assist_scales_curvature_without_moving_endpoints(self):
        self.tool._manual_gap_bridge_tangent = lambda pixel: (
            1.0, 0.3 if pixel[0] < 80.0 else -0.3
        )
        full = self._preview()
        self.tool.edge_weight = 0.5
        partial = self._preview()
        self.assertEqual(len(full), len(partial))
        self.assertEqual(partial[-1], (109.0, 31.0))
        self.assertGreater(max(y - 31.0 for _x, y in full), 1.0)
        for (_full_x, full_y), (_half_x, half_y) in zip(full, partial):
            self.assertAlmostEqual(half_y - 31.0, (full_y - 31.0) * 0.5)

    def test_failed_map_conversion_preserves_existing_preview(self):
        bridge = self._preview()
        confirmed = self._coordinates(self.tool.path_points)
        self.tool.pixel_to_map = mock.Mock(side_effect=RuntimeError("CRS changed"))

        self.assertFalse(self.tool.preview_manual_gap_bridge())

        self.assertEqual(self._coordinates(self.tool.preview_path), bridge)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertTrue(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_closing_bridge_preserves_preview_on_cancel_or_failed_save(self):
        tool = self.tool
        tool.start_point = QgsPointXY(self.target)
        tool.path_points = [tool.start_point, QgsPointXY(40, 15), QgsPointXY(60, 31)]
        tool.checkpoints = [0, 2]
        bridge = self._preview()
        confirmed = self._coordinates(tool.path_points)
        tool.ask_elevation = mock.Mock(return_value=None)

        tool.canvasPressEvent(self._mouse_event())

        self.assertEqual(self._coordinates(tool.path_points), confirmed)
        self.assertEqual(self._coordinates(tool.preview_path), bridge)
        self.assertTrue(tool._manual_gap_bridge_preview_matches(self.target))
        tool.ask_elevation.return_value = 100.0
        tool.save_to_layer = mock.Mock(return_value=False)

        tool.canvasPressEvent(self._mouse_event())

        tool.save_to_layer.assert_called_once_with(closed=True, elevation=100.0)
        self.assertEqual(self._coordinates(tool.path_points), confirmed)
        self.assertEqual(self._coordinates(tool.preview_path), bridge)
        self.assertTrue(tool._manual_gap_bridge_preview_matches(self.target))

    def _prepare_detector_driven_case(self, case):
        import numpy as np

        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.core.livewire import build_livewire_tree, is_livewire_available
        from benchmarks.manual_gap_shadow import LIVEWIRE_CONFIG

        if not is_livewire_available():
            self.skipTest("Detector-to-LiveWire parity requires optional SciPy")
        tool = self.tool
        start = case.to_image_xy(case.canonical_start_xy)
        end = case.to_image_xy(case.canonical_end_xy)
        tool.cached_rgb_image = case.image_rgb
        tool.cached_ink_evidence = EdgeDetector.detect_ink_evidence(
            case.image_rgb, tile_origin=(0, 0)
        )
        tool.cached_edges = np.where(
            tool.cached_ink_evidence.centerline, 255, 0
        ).astype(np.uint8)
        tool.path_points = [QgsPointXY(*start)]
        tool.checkpoints = [0]
        tool.start_point = QgsPointXY(*start)
        tool.last_map_point = QgsPointXY(*start)
        tool.last_input_point = QgsPointXY(*start)
        self.target = QgsPointXY(*end)
        tool.last_hover_pos = QgsPointXY(self.target)
        tool._livewire_request_point = QgsPointXY(self.target)
        tool._livewire_tree = build_livewire_tree(
            case.image_rgb,
            tool.cached_edges,
            start,
            evidence=tool.cached_ink_evidence,
            strength=tool.edge_weight,
            config=LIVEWIRE_CONFIG,
        )
        tool._present_livewire_cursor_preview(
            self.target, request_tree=False, schedule_recovery=False
        )

    def test_detector_driven_glyph_adjacent_endpoint_uses_outward_contour_support(self):
        from benchmarks.manual_gap_shadow import build_manual_gap_shadow_cases

        self._prepare_detector_driven_case(build_manual_gap_shadow_cases()[0])
        champion = self._coordinates(self.tool.preview_path)
        confirmed = self._coordinates(self.tool.path_points)
        self.assertTrue(champion)

        bridge = self._preview()

        self.assertNotEqual(bridge, champion)
        self.assertGreater(len(bridge), 2)
        self.assertEqual(bridge[-1], (164.0, 113.0))
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertTrue(self.tool._manual_gap_bridge_preview_matches(self.target))

        self.tool.canvasPressEvent(self._mouse_event())

        self.assertEqual(self._coordinates(self.tool.path_points), confirmed + bridge)

    def test_detector_driven_glyph_parallel_negative_control_preserves_ink(self):
        from benchmarks.manual_gap_shadow import build_manual_gap_shadow_cases

        self._prepare_detector_driven_case(build_manual_gap_shadow_cases()[6])
        champion = self._coordinates(self.tool.preview_path)
        confirmed = self._coordinates(self.tool.path_points)
        self.assertTrue(champion)

        self.assertFalse(self.tool.preview_manual_gap_bridge())

        self.assertEqual(self._coordinates(self.tool.preview_path), champion)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertFalse(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_detector_driven_clear_gap_previews_and_accepts_exact_route(self):
        from benchmarks.manual_gap_shadow import build_manual_gap_shadow_cases

        self._prepare_detector_driven_case(build_manual_gap_shadow_cases()[4])
        confirmed = self._coordinates(self.tool.path_points)
        bridge = self._preview()
        self.assertGreater(len(bridge), 2)
        self.assertTrue(all(abs(y - 128.0) < 2.0 for _x, y in bridge))

        self.tool.canvasPressEvent(self._mouse_event())

        self.assertEqual(self._coordinates(self.tool.path_points), confirmed + bridge)
        self.assertEqual(self.tool.last_input_point, self.target)
        self.assertFalse(self.tool._manual_gap_bridge_preview_matches(self.target))
