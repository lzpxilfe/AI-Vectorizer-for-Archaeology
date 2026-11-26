"""Real-QGIS regressions for the contour-interval elevation suggestion.

Kept in its own module: like the other tool modules it creates map tools
outside a project, and CI runs each QGIS module in a separate process.
"""

import importlib.util
import os
from types import SimpleNamespace
import unittest
from unittest import mock


HAS_QGIS = importlib.util.find_spec("qgis") is not None
if not HAS_QGIS and os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
    raise RuntimeError("QGIS bindings are required for this job")


@unittest.skipUnless(HAS_QGIS, "native QGIS unavailable")
class ContourElevationHintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qgis.core import QgsApplication
        cls.owned = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        if cls.owned:
            cls.application.initQgis()

    @classmethod
    def tearDownClass(cls):
        if cls.owned:
            cls.application.exitQgis()

    def _tool(self, history, interval=20.0):
        from qgis.core import QgsCoordinateReferenceSystem, QgsVectorLayer
        from qgis.gui import QgsMapCanvas
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool
        crs = QgsCoordinateReferenceSystem("EPSG:3857")
        layer = QgsVectorLayer("LineString?crs=EPSG:3857&field=elevation:double", "c", "memory")
        canvas = QgsMapCanvas()
        canvas.setDestinationCrs(crs)
        tool = SmartTraceTool(
            canvas, SimpleNamespace(crs=lambda: crs), layer, freehand=True,
            language="en", contour_interval=interval, elevation_history=history,
        )
        self._owned.append((tool, canvas, layer))
        return tool

    def setUp(self):
        self._owned = []

    def tearDown(self):
        for tool, canvas, _layer in self._owned:
            tool.dispose()
            tool.deleteLater()
            canvas.deleteLater()
        self.application.processEvents()
        self._owned = []

    def test_new_contour_starts_from_the_suggestion_and_records_the_answer(self):
        from qgis.PyQt.QtWidgets import QInputDialog
        history = [100.0]
        tool = self._tool(history)
        with mock.patch.object(QInputDialog, "getDouble", return_value=(120.0, True)) as dialog:
            self.assertEqual(tool.ask_elevation(), 120.0)
        self.assertEqual(dialog.call_args.args[3], 120.0)
        self.assertIn("Suggested from the previous 100 m", dialog.call_args.args[2])
        self.assertEqual(history, [100.0, 120.0])

        with mock.patch.object(QInputDialog, "getDouble", return_value=(100.0, True)) as dialog:
            tool.ask_elevation()
        self.assertEqual(dialog.call_args.args[3], 140.0)
        with mock.patch.object(QInputDialog, "getDouble", return_value=(80.0, True)) as dialog:
            tool.ask_elevation()
        self.assertEqual(dialog.call_args.args[3], 80.0)
        self.assertEqual(history, [100.0, 120.0, 100.0, 80.0])

    def test_spot_heights_cancel_and_no_interval_leave_the_history_alone(self):
        from qgis.PyQt.QtWidgets import QInputDialog
        history = [100.0]
        tool = self._tool(history)
        with mock.patch.object(QInputDialog, "getDouble", return_value=(311.0, True)) as dialog:
            tool.ask_elevation(spot_height=True)
        self.assertEqual(dialog.call_args.args[3], tool.ELEVATION_DEFAULT)
        with mock.patch.object(QInputDialog, "getDouble", return_value=(0.0, False)):
            self.assertIsNone(tool.ask_elevation())
        self.assertEqual(history, [100.0])

        plain = self._tool([100.0], interval=None)
        with mock.patch.object(QInputDialog, "getDouble", return_value=(5.0, True)) as dialog:
            plain.ask_elevation()
        self.assertEqual(dialog.call_args.args[3], plain.ELEVATION_DEFAULT)
        self.assertEqual(dialog.call_args.args[2], "Elevation (m):")


if __name__ == "__main__":
    unittest.main()
