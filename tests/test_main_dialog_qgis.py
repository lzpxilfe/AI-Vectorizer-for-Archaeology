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
        from qgis.PyQt.QtCore import QSettings
        from qgis.PyQt.QtWidgets import QMainWindow
        from ai_vectorizer.config import SETTINGS_LANG_KEY
        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock, _qt_value

        self.temp = tempfile.TemporaryDirectory()
        ini_format = getattr(QSettings, "IniFormat", None)
        if ini_format is None:
            ini_format = QSettings.Format.IniFormat
        self.settings = QSettings(str(Path(self.temp.name) / "settings.ini"), ini_format)
        self.settings.setValue(SETTINGS_LANG_KEY, "en")
        settings_patch = mock.patch("ai_vectorizer.ui.main_dialog.QSettings", return_value=self.settings)
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
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

    def _reopen_dock(self):
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock, _qt_value
        self.dock.cleanup(permanent=True)
        self.window.removeDockWidget(self.dock)
        self.dock.deleteLater()
        self.application.processEvents()
        self.dock = AIVectorizerDock(self.iface, self.window)
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

    def _output(self, name="contours"):
        from qgis.core import QgsVectorLayer
        output = QgsVectorLayer("LineString?crs=EPSG:3857", name, "memory")
        self.assertTrue(output.isValid())
        self.project.addMapLayer(output)
        return output

    def _input_state_snapshot(self):
        def identity(layer):
            try:
                return layer.id() if layer is not None else None
            except RuntimeError:
                return "deleted layer wrapper"

        def selection(combo):
            return {"count": combo.count(), "index": combo.currentIndex(),
                    "layer_id": identity(combo.currentLayer())}

        return {"raster": selection(self.dock.layer_combo),
                "output": selection(self.dock.vector_combo),
                "output_layer_id": identity(self.dock.output_layer),
                "trace_enabled": self.dock.trace_btn.isEnabled(),
                "dem_enabled": self.dock.dem_btn.isEnabled(),
                "status": self.dock.status_label.text()}

    def _process_layer_events(self):
        # Record both sides of one normal UI event-loop iteration. This exposes
        # whether a QGIS version delays model selection or leaves stale UI state.
        before = self._input_state_snapshot()
        self.application.processEvents()
        return {"before_events": before, "after_events": self._input_state_snapshot()}

    def _start_freehand_session(self):
        raster = self._raster()
        output = self._output("original contours")
        self.dock.layer_combo.setLayer(raster)
        self.dock.vector_combo.setLayer(output)
        self._process_layer_events()
        self.canvas.setLayers([raster])
        self.canvas.setExtent(raster.extent())
        self.dock.freehand_check.setChecked(True)
        self.dock.trace_btn.click()
        self.assertIsNotNone(self.dock.active_tool)
        self.assertIs(self.canvas.mapTool(), self.dock.active_tool)
        return raster, output, self.dock.active_tool

    def test_first_auto_selection_without_layer_changed_updates_ready_state(self):
        # QGIS 4's first row can be selected without this higher-level signal.
        # Keep native index/model signals active to reproduce the observed state.
        self.dock.vector_combo.layerChanged.disconnect(self.dock.on_layer_selected)
        self.dock.layer_combo.layerChanged.disconnect(self.dock.on_raster_layer_selected)
        output = self._output()
        raster = self._raster()
        self.assertEqual(self.dock.vector_combo.currentLayer().id(), output.id())
        self.assertIsNone(self.dock.output_layer)
        self.assertTrue(self.dock._layer_selection_timer.isActive())
        details = self._process_layer_events()
        self.assertEqual(self.dock.output_layer.id(), output.id(), details)
        self.assertEqual(self.dock.layer_combo.currentLayer().id(), raster.id(), details)
        self.assertTrue(self.dock.trace_btn.isEnabled(), details)
        self.assertTrue(self.dock.dem_btn.isEnabled(), details)
        self.assertIn("Ready", self.dock.status_label.text(), details)

    def test_model_changes_preserve_same_tool_and_stop_it_on_output_switch(self):
        _raster, output, tool = self._start_freehand_session()
        deactivations = []
        tool.deactivated.connect(lambda: deactivations.append(True))
        replacement = self._output("replacement contours")
        self._process_layer_events()
        self.assertIs(self.dock.active_tool, tool)
        self.assertTrue(self.dock.trace_btn.isChecked())
        self.assertFalse(self.dock.dem_btn.isEnabled())
        self.assertEqual(deactivations, [])
        self.assertEqual(self.dock.output_layer.id(), output.id())
        self.dock.vector_combo.layerChanged.disconnect(self.dock.on_layer_selected)
        self.dock.vector_combo.setLayer(replacement)
        self._process_layer_events()
        self.assertIsNone(self.dock.active_tool)
        self.assertFalse(self.dock.trace_btn.isChecked())
        self.assertTrue(self.dock.dem_btn.isEnabled())
        self.assertEqual(self.dock.output_layer.id(), replacement.id())
        self.assertEqual(deactivations, [True])

    def test_active_raster_removal_does_not_leave_a_checked_start_button(self):
        raster, _output, tool = self._start_freehand_session()
        deactivations = []
        tool.deactivated.connect(lambda: deactivations.append(True))
        self.dock.layer_combo.layerChanged.disconnect(self.dock.on_raster_layer_selected)
        self.project.removeMapLayer(raster.id())
        details = self._process_layer_events()
        self.assertIsNone(self.dock.active_tool)
        self.assertFalse(self.dock.trace_btn.isChecked(), details)
        self.assertFalse(self.dock.trace_btn.isEnabled(), details)
        self.assertTrue(self.dock.dem_btn.isEnabled(), details)
        self.assertEqual(deactivations, [True])

    def test_cleanup_cancels_pending_sync_and_ignores_late_layer_signals(self):
        self.assertIs(self.dock._layer_selection_timer.parent(), self.dock)
        self._output()
        self.assertTrue(self.dock._layer_selection_timer.isActive())
        self.dock.cleanup(permanent=True)
        status = self.dock.status_label.text()
        self.assertFalse(self.dock._layer_selection_timer.isActive())
        raster = self._raster()
        self.dock.on_raster_layer_selected(raster)
        self.dock._sync_selected_layers()
        self.application.processEvents()
        self.assertFalse(self.dock._layer_selection_timer.isActive())
        self.assertFalse(self.dock.trace_btn.isEnabled())
        self.assertEqual(self.dock.status_label.text(), status)

    def test_closed_dock_resynchronizes_inputs_when_reopened(self):
        self.dock.cleanup()
        output = self._output()
        self._raster()
        self.application.processEvents()
        self.assertFalse(self.dock._layer_selection_timer.isActive())
        self.assertIsNone(self.dock.output_layer)
        self.window.show()
        details = self._process_layer_events()
        self.assertEqual(self.dock.output_layer.id(), output.id(), details)
        self.assertTrue(self.dock.trace_btn.isEnabled(), details)

    def test_small_window_can_reach_start_dem_and_help_in_both_languages(self):
        self._assert_small_window_controls_reachable()

    def test_small_window_with_larger_system_font_and_fusion_style(self):
        from qgis.PyQt.QtGui import QFont
        from qgis.PyQt.QtWidgets import QStyleFactory, QWidget
        style = QStyleFactory.create("Fusion")
        style.setParent(self.window)
        widgets = [self.window] + self.window.findChildren(QWidget)
        for widget in widgets:
            widget.setStyle(style)
        font = QFont(self.dock.font())
        # Pixels keep this stress case independent of platform DPI. The
        # default-font case above still runs unchanged on every QGIS image.
        font.setPixelSize(18)
        self.dock.setFont(font)
        try:
            self._assert_small_window_controls_reachable()
        finally:
            for widget in widgets:
                widget.setStyle(None)

    def test_optional_model_controls_fit_small_window(self):
        from ai_vectorizer.config import MODEL_IDX_HED, MODEL_IDX_MOBILE_SAM, MODEL_IDX_SAM
        from ai_vectorizer.core.edge_detector import EdgeDetector
        from qgis.PyQt.QtGui import QFont
        font = QFont(self.dock.font())
        font.setPixelSize(18)
        self.dock.setFont(font)
        self.dock.auto_path_check.setChecked(True)
        for model, reason in ((MODEL_IDX_HED, "missing_weights"),
                              (MODEL_IDX_HED, "missing_opencv"),
                              (MODEL_IDX_MOBILE_SAM, None), (MODEL_IDX_SAM, None)):
            with self.subTest(model=model, reason=reason), \
                 mock.patch.object(EdgeDetector, "get_hed_runtime_status",
                                   return_value={"ok": False, "reason": reason}), \
                 mock.patch.object(self.dock, "_sam_backend_available", return_value=False):
                self.dock.model_combo.setCurrentIndex(model)
                self.dock.on_model_changed(model)
                if model != MODEL_IDX_HED:
                    self.dock.init_sam_engine()
                controls = tuple(widget for widget in (
                    self.dock.sam_check_btn, self.dock.sam_report_btn,
                    self.dock.sam_download_btn, self.dock.install_cmd,
                ) if not widget.isHidden())
                self.assertTrue(controls)
                self._assert_small_window_controls_reachable(controls)

    def _assert_small_window_controls_reachable(self, extra_controls=()):
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
                    controls = (self.dock.trace_btn, self.dock.dem_btn, self.dock.help_btn)
                    if advanced:
                        controls += extra_controls
                    for control in controls:
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
        details = self._process_layer_events()
        self.assertEqual(details["after_events"]["output"]["layer_id"], output.id(), details)
        self.assertFalse(self.dock.trace_btn.isEnabled(), details)
        self.assertIn("raster map", self.dock.status_label.text(), details)
        self.assertTrue(self.dock.dem_btn.isEnabled(), details)
        raster = self._raster()
        self.dock.layer_combo.setLayer(raster)
        details = self._process_layer_events()
        self.assertEqual(details["after_events"]["raster"]["layer_id"], raster.id(), details)
        self.assertTrue(self.dock.trace_btn.isEnabled(), details)
        self.assertIn("Ready", self.dock.status_label.text(), details)
        self.project.removeMapLayer(output.id())
        details = self._process_layer_events()
        self.assertFalse(self.dock.trace_btn.isEnabled(), details)
        self.assertFalse(self.dock.dem_btn.isEnabled(), details)
        self.assertIn("output line", self.dock.status_label.text(), details)
        self.dock.current_language = "ko"
        self.dock.apply_language()
        self.assertFalse(self.dock.trace_btn.isEnabled())
        self.assertIn("출력 선", self.dock.status_label.text())

    def test_removing_raster_retains_dem_but_disables_tracing(self):
        raster = self._raster()
        output = self._output()
        self.dock.vector_combo.setLayer(output)
        self.dock.layer_combo.setLayer(raster)
        details = self._process_layer_events()
        self.assertEqual(details["after_events"]["output"]["layer_id"], output.id(), details)
        self.assertEqual(details["after_events"]["raster"]["layer_id"], raster.id(), details)
        self.assertTrue(self.dock.trace_btn.isEnabled(), details)
        self.project.removeMapLayer(raster.id())
        details = self._process_layer_events()
        self.assertFalse(self.dock.trace_btn.isEnabled(), details)
        self.assertTrue(self.dock.dem_btn.isEnabled(), details)
        self.assertIn("raster map", self.dock.status_label.text(), details)

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

    def test_first_run_uses_qgis_locale_and_english_for_other_locales(self):
        from qgis.core import QgsApplication
        from ai_vectorizer.config import SETTINGS_LANG_KEY
        for locale, expected in (("ko", "ko"), ("ko_KR", "ko"), ("en_GB", "en"),
                                 ("de_DE", "en"), ("fr", "en"), ("", "en")):
            with self.subTest(locale=locale), mock.patch.object(QgsApplication, "locale", return_value=locale):
                self.settings.remove(SETTINGS_LANG_KEY)
                # QGIS may retain a userLocale while its override is disabled.
                self.settings.setValue("locale/userLocale", "ko_KR")
                self.settings.setValue("locale/overrideFlag", False)
                self._reopen_dock()
                self.assertEqual(self.dock.current_language, expected)
                self.assertEqual(self.dock.lang_combo.currentData(), expected)
                self.assertFalse(self.settings.contains(SETTINGS_LANG_KEY))

    def test_language_choice_persists_across_reopening_and_overrides_locale(self):
        from qgis.core import QgsApplication
        from ai_vectorizer.config import SETTINGS_LANG_KEY
        for chosen, locale in (("ko", "en_US"), ("en", "ko_KR")):
            with self.subTest(chosen=chosen):
                self.dock.lang_combo.setCurrentIndex(self.dock.lang_combo.findData(chosen))
                self.assertEqual(self.dock.current_language, chosen)
                self.assertEqual(self.settings.value(SETTINGS_LANG_KEY), chosen)
                self.settings.sync()
                with mock.patch.object(QgsApplication, "locale", return_value=locale):
                    self._reopen_dock()
                self.assertEqual(self.dock.current_language, chosen)
                self.assertEqual(self.dock.lang_combo.currentData(), chosen)

    def test_invalid_saved_language_uses_current_qgis_locale(self):
        from qgis.core import QgsApplication
        from ai_vectorizer.config import SETTINGS_LANG_KEY
        self.settings.setValue(SETTINGS_LANG_KEY, "unsupported")
        with mock.patch.object(QgsApplication, "locale", return_value="en_US"):
            self._reopen_dock()
        self.assertEqual(self.dock.lang_combo.currentData(), "en")

    def test_english_controls_help_and_model_menus_have_no_korean_text(self):
        from qgis.PyQt.QtWidgets import QWidget
        self.dock.lang_combo.setCurrentIndex(self.dock.lang_combo.findData("ko"))
        self.dock.lang_combo.setCurrentIndex(self.dock.lang_combo.findData("en"))
        for name, widget in vars(self.dock).items():
            if not isinstance(widget, QWidget) or widget is self.dock.lang_combo:
                continue
            for accessor in ("text", "title", "toolTip", "placeholderText", "accessibleName"):
                read = getattr(widget, accessor, None)
                if callable(read):
                    with self.subTest(widget=name, property=accessor):
                        self.assertNotRegex(str(read()), "[가-힣]")
        self.assertNotRegex(self.dock._help_text(), "[가-힣]")
        for index in range(self.dock.model_combo.count()):
            self.assertNotRegex(self.dock.model_combo.itemText(index), "[가-힣]")
        self.assertFalse(self.dock.smart_recovery_check.isChecked())
        self.assertFalse(self.dock.learned_guide_check.isChecked())
        self.assertIn("EfficientSAM", self.dock.smart_recovery_check.toolTip())

    def test_language_switch_updates_open_dem_dialog_and_preserves_output_paths(self):
        self._output()
        self._process_layer_events()
        self.dock.open_dem_dialog()
        dem = self.dock.dem_dialog
        self.assertIsNotNone(dem)
        output_path = str(Path(self.temp.name) / "chosen_terrain.tif")
        dem.dem_path.setText(output_path)
        for language, label in (("ko", "DEM 생성"), ("en", "Build DEM")):
            self.dock.lang_combo.setCurrentIndex(self.dock.lang_combo.findData(language))
            self.assertEqual(dem.language, language)
            self.assertEqual(dem.build_button.text(), label)
            self.assertEqual(dem.dem_path.text(), output_path)
        self.assertEqual(dem.dem_browse.accessibleName(), "Choose DEM output file")

    def test_language_switch_retranslates_existing_local_model_guidance(self):
        self.dock.lang_combo.setCurrentIndex(self.dock.lang_combo.findData("ko"))
        self.dock.learned_guide_check.setChecked(True)
        self.assertIn("모델 폴더를 가져오세요", self.dock.learned_status.text())
        self.dock.lang_combo.setCurrentIndex(self.dock.lang_combo.findData("en"))
        self.assertIn("Import a local bundle", self.dock.learned_status.text())
        self.assertNotRegex(self.dock.learned_status.text(), "[가-힣]")
        self.dock.lang_combo.setCurrentIndex(self.dock.lang_combo.findData("ko"))
        self.assertIn("모델 폴더를 가져오세요", self.dock.learned_status.text())


if __name__ == "__main__":
    unittest.main()
