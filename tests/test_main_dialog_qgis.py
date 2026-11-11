"""Real-QGIS dock sizing, input readiness and save guidance regressions."""

import gc
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HAS_QGIS = importlib.util.find_spec("qgis") is not None
if not HAS_QGIS and os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
    raise RuntimeError("QGIS bindings are required for this job")


@unittest.skipUnless(HAS_QGIS, "native QGIS unavailable")
class MainDialogQgisTests(unittest.TestCase):
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

    def setUp(self):
        from qgis.core import QgsProject
        from qgis.gui import QgsMapCanvas
        from qgis.PyQt.QtWidgets import QMainWindow
        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock, _qt_value

        self.temp = tempfile.TemporaryDirectory()
        self.project = QgsProject.instance()
        self.project.clear()
        self.window = QMainWindow()
        self.canvas = QgsMapCanvas(self.window)
        self.window.setCentralWidget(self.canvas)
        self.iface = SimpleNamespace(
            mapCanvas=lambda: self.canvas, mainWindow=lambda: self.window,
            activeLayer=lambda: None, setActiveLayer=lambda _layer: None,
        )
        self.models = mock.patch.object(AIVectorizerDock, "_sam_models_dir", return_value=self.temp.name)
        self.recovery = mock.patch.object(AIVectorizerDock, "_refresh_recovery_availability")
        self.models.start()
        self.recovery.start()
        self.addCleanup(self.models.stop)
        self.addCleanup(self.recovery.stop)
        hed_storage = mock.patch.multiple(EdgeDetector, **{
            name: getattr(EdgeDetector, name) for name in (
                "HED_MODEL_DIR", "HED_PROTOTXT", "HED_CAFFEMODEL",
                "_hed_runtime_status_cache", "_hed_runtime_status_signature",
            )
        })
        hed_storage.start()
        self.addCleanup(hed_storage.stop)
        self.dock = AIVectorizerDock(self.iface, self.window)
        self.dock.current_language = "en"
        self.dock.apply_language()
        self.window.addDockWidget(_qt_value("LeftDockWidgetArea", "DockWidgetArea"), self.dock)

    def tearDown(self):
        self.dock.cleanup(permanent=True)
        self.canvas.setLayers([])
        self.project.clear()
        self.dock.deleteLater()
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.dock = self.canvas = self.window = None
        gc.collect()
        self.temp.cleanup()

    def _raster(self):
        from osgeo import gdal
        from qgis.core import QgsRasterLayer
        path = str(Path(self.temp.name) / "map.tif")
        data = gdal.GetDriverByName("GTiff").Create(path, 16, 16, 1, gdal.GDT_Byte)
        data.SetGeoTransform((0, 1, 0, 16, 0, -1))
        data.GetRasterBand(1).Fill(250)
        data = None
        raster = QgsRasterLayer(path, "map", "gdal")
        self.assertTrue(raster.isValid())
        self.project.addMapLayer(raster)
        return raster

    def _output(self):
        from qgis.core import QgsVectorLayer
        output = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        self.assertTrue(output.isValid())
        self.project.addMapLayer(output)
        return output

    def test_small_window_can_reach_start_dem_and_help_in_both_languages(self):
        from ai_vectorizer.ui.main_dialog import _qt_value
        for language in ("ko", "en"):
            self.dock.current_language = language
            self.dock.apply_language()
            for advanced in (False, True):
                with self.subTest(language=language, advanced=advanced):
                    self.dock.advanced_check.setChecked(advanced)
                    self.window.resize(1024, 700)
                    self.window.show()
                    self.application.processEvents()
                    self.window.resizeDocks([self.dock], [360], _qt_value("Horizontal", "Orientation"))
                    self.application.processEvents()
                    self.assertLessEqual(self.window.height(), 700)
                    self.assertGreater(self.dock.scroll_area.verticalScrollBar().maximum(), 0)
                    self.assertEqual(self.dock.scroll_area.horizontalScrollBar().maximum(), 0)
                    for control in (self.dock.trace_btn, self.dock.dem_btn, self.dock.help_btn):
                        self.dock.scroll_area.ensureWidgetVisible(control)
                        self.application.processEvents()
                        top = control.mapTo(self.dock.scroll_area.viewport(), control.rect().topLeft())
                        self.assertGreaterEqual(top.y(), 0)
                        self.assertLessEqual(top.y() + control.height(),
                                             self.dock.scroll_area.viewport().height())

    def test_start_requires_raster_and_output_and_updates_when_removed(self):
        self.assertFalse(self.dock.trace_btn.isEnabled())
        self.assertIn("Step 1", self.dock.status_label.text())
        self.assertIn("Step 2", self.dock.status_label.text())
        output = self._output()
        self.dock.vector_combo.setLayer(output)
        self.assertFalse(self.dock.trace_btn.isEnabled())
        self.assertIn("raster map", self.dock.status_label.text())
        self.assertTrue(self.dock.dem_btn.isEnabled())
        raster = self._raster()
        self.dock.layer_combo.setLayer(raster)
        self.assertTrue(self.dock.trace_btn.isEnabled())
        self.assertIn("Ready", self.dock.status_label.text())
        self.project.removeMapLayer(output.id())
        self.assertFalse(self.dock.trace_btn.isEnabled())
        self.assertFalse(self.dock.dem_btn.isEnabled())
        self.assertIn("output line", self.dock.status_label.text())
        self.dock.current_language = "ko"
        self.dock.apply_language()
        self.assertFalse(self.dock.trace_btn.isEnabled())
        self.assertIn("출력 선", self.dock.status_label.text())

    def test_removing_raster_retains_dem_but_disables_tracing(self):
        raster = self._raster()
        output = self._output()
        self.dock.vector_combo.setLayer(output)
        self.dock.layer_combo.setLayer(raster)
        self.assertTrue(self.dock.trace_btn.isEnabled())
        self.project.removeMapLayer(raster.id())
        self.assertFalse(self.dock.trace_btn.isEnabled())
        self.assertTrue(self.dock.dem_btn.isEnabled())
        self.assertIn("raster map", self.dock.status_label.text())

    def test_controls_and_help_distinguish_finishing_from_file_save(self):
        for language, finish, save in (("en", "finish", "Save Layer Edits"),
                                        ("ko", "선 확정", "레이어 편집 저장")):
            with self.subTest(language=language):
                self.dock.current_language = language
                self.dock.apply_language()
                self.assertIn(finish, self.dock.controls_label.text())
                self.assertIn(save, self.dock.controls_label.text())
                self.assertIn(save, self.dock._help_text())
                self.assertNotIn("AI", self.dock.edge_strength_label.text())


if __name__ == "__main__":
    unittest.main()
