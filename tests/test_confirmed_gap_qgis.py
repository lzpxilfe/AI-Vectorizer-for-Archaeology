"""Actual QGIS explicit broad-turn previews and their existing edit boundary."""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

from tests import test_manual_gap_qgis as _manual


HAS_QGIS = _manual.HAS_QGIS
if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1" and not HAS_QGIS:
    raise RuntimeError("QGIS bindings are required for confirmed-gap tests")


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class ConfirmedGapQgisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _manual.ManualGapQgisTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        _manual.ManualGapQgisTests.tearDownClass.__func__(cls)

    def setUp(self):
        # Share the real canvas/layer and exact-coordinate manual-G harness;
        # do not inherit its tests or substitute the production map-tool.
        _manual.ManualGapQgisTests.setUp(self)
        import numpy as np
        from qgis.core import QgsPointXY
        from ai_vectorizer.core.line_evidence import LineEvidence

        self.np = np
        self.Point = QgsPointXY
        self.mask = np.zeros((96, 96), dtype=bool)
        self.mask[20:51, 30] = True
        self.mask[20:51, 60] = True
        self.tool.path_points = [QgsPointXY(30, 25), QgsPointXY(30, 50)]
        self.tool.checkpoints = [0, 1]
        self.tool.start_point = QgsPointXY(self.tool.path_points[0])
        self.tool.last_map_point = QgsPointXY(self.tool.path_points[-1])
        self.tool.last_input_point = QgsPointXY(self.tool.path_points[-1])
        self._target((60, 50))
        self._publish_mask()

    def tearDown(self):
        self.tool.dispose()
        self.tool.deleteLater()
        self.canvas.deleteLater()
        self.application.processEvents()

    _mouse_event = _manual.ManualGapQgisTests._mouse_event
    _coordinates = staticmethod(_manual.ManualGapQgisTests._coordinates)

    def _target(self, pixel):
        self.target = self.Point(*pixel)
        self.tool.last_hover_pos = self.Point(self.target)
        self.tool._livewire_request_point = self.Point(self.target)

    def _publish_mask(self):
        from ai_vectorizer.core.line_evidence import LineEvidence

        score = self.mask.astype(self.np.float32)
        self.tool.cached_ink_evidence = LineEvidence(
            score, self.mask, self.np.zeros_like(score), score, score,
        )
        self.tool.cached_edges = self.mask.astype(self.np.uint8) * 255
        grey = self.np.where(self.mask, 35, 245).astype(self.np.uint8)
        self.tool.cached_rgb_image = self.np.repeat(grey[:, :, None], 3, axis=2)

    def _preview(self):
        self.assertTrue(self.tool.preview_manual_gap_bridge())
        points = self._coordinates(self.tool.preview_path)
        self.assertGreater(max(y for _x, y in points), 60)
        self.assertEqual(points[-1], (60.0, 50.0))
        return points

    def test_g_falls_back_to_signed_broad_turn_without_moving_confirmed_anchors(self):
        confirmed = self._coordinates(self.tool.path_points)
        event = SimpleNamespace(
            key=lambda: _manual._qt_value("Key_G", "Key"),
            modifiers=lambda: 0, isAutoRepeat=lambda: False,
            accept=mock.Mock(), ignore=mock.Mock(),
        )
        with mock.patch.object(
            self.tool, "_confirmed_manual_gap_curve", wraps=self.tool._confirmed_manual_gap_curve,
        ) as build:
            self.tool.keyPressEvent(event)
        build.assert_called_once_with((30.0, 50.0), (60.0, 50.0))
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertTrue(self.tool._manual_gap_bridge_preview_matches(self.target))
        self.assertGreater(max(point.y() for point in self.tool.preview_path), 60)
        self.assertEqual(self.layer.featureCount(), 0)

    def test_saved_broad_turn_is_exact_preview_and_one_undo_removes_it(self):
        preview = self._preview()
        confirmed = self._coordinates(self.tool.path_points)
        self.tool.canvasPressEvent(self._mouse_event(self.Point(60.5, 50)))
        expected = confirmed + preview
        self.assertEqual(self._coordinates(self.tool.path_points), expected)
        self.assertEqual(self.tool.last_input_point, self.target)
        before = self.layer.undoStack().index()
        self.assertTrue(self.tool.save_to_layer(closed=False, elevation=700.0))
        self.assertEqual(self.layer.featureCount(), 1)
        saved = next(self.layer.getFeatures())
        self.assertEqual(self._coordinates(saved.geometry().asPolyline()), expected)
        self.assertEqual(self.layer.undoStack().index(), before + 1)
        self.layer.undoStack().undo()
        self.assertEqual(self.layer.featureCount(), 0)

    def test_first_anchor_short_gap_accepts_open_curve_before_explicit_finish(self):
        self.mask[:, 60] = False
        self.mask[20:51, 50] = True
        self._publish_mask()
        self.tool.path_points = [self.Point(30, 50)]
        self.tool.start_point = self.Point(30, 50)
        self.tool.last_map_point = self.Point(30, 50)
        self.tool.last_input_point = self.Point(30, 50)
        self.tool.checkpoints = [0]
        self._target((50, 50))
        self.assertTrue(self.tool.is_near_start(self.target))
        self.assertTrue(self.tool.preview_manual_gap_bridge())
        preview = self._coordinates(self.tool.preview_path)
        expected = ((30.0, 50.0),) + preview
        self.assertGreater(max(y for _x, y in preview), 58)
        self.tool.ask_elevation = mock.Mock(side_effect=AssertionError("short gap consumed by spot height"))
        self.tool.create_spot_height = mock.Mock(side_effect=AssertionError("created spot from gap"))
        self.tool.canvasPressEvent(self._mouse_event())
        self.tool.ask_elevation.assert_not_called()
        self.tool.create_spot_height.assert_not_called()
        self.assertTrue(self.tool.is_tracing)
        self.assertEqual(self._coordinates(self.tool.path_points), expected)
        self.assertEqual(self.layer.featureCount(), 0)
        self.tool.ask_elevation = mock.Mock(return_value=600.0)
        before = self.layer.undoStack().index()
        event = self._mouse_event()
        event.button = lambda: _manual._qt_value("RightButton", "MouseButton")
        self.tool.canvasPressEvent(event)
        self.tool.ask_elevation.assert_called_once_with()
        self.tool.create_spot_height.assert_not_called()
        self.assertEqual(self.layer.featureCount(), 1)
        saved = next(self.layer.getFeatures())
        self.assertEqual(self._coordinates(saved.geometry().asPolyline()), expected)
        self.assertEqual(self.layer.undoStack().index(), before + 1)
        self.assertFalse(self.tool.is_tracing)
        self.layer.undoStack().undo()
        self.assertEqual(self.layer.featureCount(), 0)

    def test_neighbour_endpoint_pair_is_refused_without_replacing_existing_preview(self):
        self.mask[20:51, 66] = True
        self._publish_mask()
        self._target((66, 50))
        self.tool.preview_path = [self.Point(47, 47), self.Point(66, 50)]
        preview = self._coordinates(self.tool.preview_path)
        confirmed = self._coordinates(self.tool.path_points)
        self.assertFalse(self.tool.preview_manual_gap_bridge())
        self.assertEqual(self._coordinates(self.tool.preview_path), preview)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertFalse(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_partial_display_crossing_is_refused_even_when_full_curve_is_clear(self):
        from ai_vectorizer.core.confirmed_gap_curve import build_confirmed_gap_curve
        from ai_vectorizer.core.confirmed_gap_evidence import confirmed_gap_curve_is_clear
        from ai_vectorizer.core.livewire import blend_path_with_cursor

        # The full U clears this short but sustained vertical fragment. Its
        # half-strength preview passes through it; shorter full candidates
        # either intersect the fragment themselves or do so after blending.
        self.mask[51:60, 45] = True
        self._publish_mask()
        curve = build_confirmed_gap_curve((30, 50), (60, 50), (0, 1), (0, 1))
        self.assertTrue(confirmed_gap_curve_is_clear(self.tool.cached_ink_evidence, curve.points_xy))
        displayed = blend_path_with_cursor(curve.points_xy, (30, 50), (60, 50), 0.5)
        self.assertFalse(confirmed_gap_curve_is_clear(self.tool.cached_ink_evidence, displayed))
        self.tool.edge_weight = 0.5
        self.tool.preview_path = [self.Point(45, 48), self.Point(60, 50)]
        preview = self._coordinates(self.tool.preview_path)
        confirmed = self._coordinates(self.tool.path_points)
        self.assertFalse(self.tool.preview_manual_gap_bridge())
        self.assertEqual(self._coordinates(self.tool.preview_path), preview)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)

    def test_existing_compatible_hermite_path_stays_first(self):
        self.tool._manual_gap_bridge_tangent = lambda _pixel: (1.0, 0.0)
        with mock.patch.object(
            self.tool, "_confirmed_manual_gap_curve", side_effect=AssertionError("replaced existing Hermite"),
        ):
            self.assertTrue(self.tool.preview_manual_gap_bridge())
        self.assertTrue(all(abs(point.y() - 50.0) < 1e-8 for point in self.tool.preview_path))

    def test_late_livewire_callback_cannot_replace_broad_turn_preview(self):
        preview = self._preview()
        task = SimpleNamespace(generation=self.tool._livewire_generation, anchor_pixel=(30, 50))
        self.tool._livewire_task = task
        present = mock.Mock(side_effect=lambda *_a, **_k: self.tool.preview_path.clear())
        self.tool._present_livewire_cursor_preview = present
        self.tool._on_livewire_tree_finished(task, True, SimpleNamespace(root=(30, 50)), None)
        present.assert_not_called()
        self.assertEqual(self._coordinates(self.tool.preview_path), preview)
        self.assertTrue(self.tool._manual_gap_bridge_preview_matches(self.target))

    def test_escape_and_cache_change_discard_only_the_broad_proposal(self):
        confirmed = self._coordinates(self.tool.path_points)
        self._preview()
        event = SimpleNamespace(
            key=lambda: _manual._qt_value("Key_Escape", "Key"),
            modifiers=lambda: 0, accept=mock.Mock(), ignore=mock.Mock(),
        )
        self.tool.keyPressEvent(event)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)
        self.assertFalse(self.tool._manual_gap_bridge_preview_matches(self.target))
        self._preview()
        self.tool._clear_edge_cache()
        self.assertFalse(self.tool.preview_path)
        self.assertEqual(self._coordinates(self.tool.path_points), confirmed)

    def test_zero_assist_skips_broad_curve_evidence_and_geometry(self):
        self.tool.edge_weight = 0.0
        with mock.patch.object(
            self.tool, "_confirmed_manual_gap_curve", side_effect=AssertionError("read broad evidence at0%"),
        ):
            self.assertFalse(self.tool.preview_manual_gap_bridge())
        self.assertFalse(self.tool.preview_path)


if __name__ == "__main__":
    unittest.main()
