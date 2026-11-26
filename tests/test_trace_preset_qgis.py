"""Real-QGIS regressions for tracing presets, project memory and the elevation hint."""

import gc
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HAS_QGIS = importlib.util.find_spec("qgis") is not None
if not HAS_QGIS and os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
    raise RuntimeError("QGIS bindings are required for this job")


ALL_AVAILABLE = {
    "opencv": True, "hed_model": True, "mobile_sam_backend": True,
    "sam_backend": True, "recovery_runtime": True, "learned_model": True,
}


@unittest.skipUnless(HAS_QGIS, "native QGIS unavailable")
class TracePresetDockTests(unittest.TestCase):
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
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        ini_format = getattr(QSettings, "IniFormat", None)
        if ini_format is None:
            ini_format = QSettings.Format.IniFormat
        self.settings = QSettings(str(Path(self.temp.name) / "settings.ini"), ini_format)
        self.settings.setValue(SETTINGS_LANG_KEY, "en")
        for target in ("ai_vectorizer.ui.main_dialog.QSettings", "ai_vectorizer.ui.trace_presets_panel.QSettings"):
            patcher = mock.patch(target, return_value=self.settings)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.project = QgsProject.instance()
        self.project.clear()
        self.window = QMainWindow()
        self.canvas = QgsMapCanvas(self.window)
        self.window.setCentralWidget(self.canvas)
        self.iface = SimpleNamespace(
            mapCanvas=lambda: self.canvas, mainWindow=lambda: self.window,
            activeLayer=lambda: None, setActiveLayer=lambda _layer: None,
        )
        for name, value in (
            ("_sam_models_dir", self.temp.name),
            ("_refresh_recovery_availability", None),
        ):
            patcher = mock.patch.object(AIVectorizerDock, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.availability = dict(ALL_AVAILABLE)
        patcher = mock.patch.object(
            AIVectorizerDock, "_preset_availability", side_effect=lambda: dict(self.availability),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.dock = self._open_dock()

    def _open_dock(self):
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock
        dock = AIVectorizerDock(self.iface, self.window)
        dock.current_language = "en"
        dock.apply_language()
        return dock

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

    def _save(self, name):
        from qgis.PyQt.QtWidgets import QInputDialog, QMessageBox
        yes = getattr(QMessageBox, "Yes", None) or QMessageBox.StandardButton.Yes
        with mock.patch.object(QInputDialog, "getText", return_value=(name, True)), \
                mock.patch.object(QMessageBox, "question", return_value=yes):
            return self.dock.save_preset()

    def _set_series_settings(self):
        from ai_vectorizer.config import MODEL_IDX_INK
        self.dock.model_combo.setCurrentIndex(MODEL_IDX_INK)
        self.dock.freedom_slider.setValue(55)
        self.dock.smart_recovery_check.setChecked(True)
        self.dock._series_ink_rgb = (150, 90, 50)
        self.dock.contour_interval_spin.setValue(20.0)

    def _reset_widgets(self):
        from ai_vectorizer.config import MODEL_IDX_LEGACY_CANNY
        self.dock.smart_recovery_check.setChecked(False)
        self.dock.model_combo.setCurrentIndex(MODEL_IDX_LEGACY_CANNY)
        self.dock.freedom_slider.setValue(10)
        self.dock._series_ink_rgb = None
        self.dock.contour_interval_spin.setValue(0.0)

    def _assert_series_settings(self):
        from ai_vectorizer.config import MODEL_IDX_INK
        self.assertEqual(self.dock.model_combo.currentIndex(), MODEL_IDX_INK)
        self.assertEqual(self.dock.freedom_slider.value(), 55)
        self.assertTrue(self.dock.smart_recovery_check.isChecked())
        self.assertEqual(self.dock._series_ink_rgb, (150, 90, 50))
        self.assertEqual(self.dock.contour_interval_spin.value(), 20.0)

    def test_save_and_apply_round_trip_through_qgis_settings(self):
        from ai_vectorizer.ui.trace_presets_panel import PRESET_SETTINGS_KEY
        self._set_series_settings()
        self.assertTrue(self._save("AMS Korea L751"))
        stored = json.loads(self.settings.value(PRESET_SETTINGS_KEY))
        self.assertEqual(stored["schema_version"], "archaeotrace-trace-preset-library/1")
        self.assertEqual(stored["presets"][0]["ink_color_rgb"], [150, 90, 50])
        self.assertEqual(self.dock.preset_combo.currentData(), "AMS Korea L751")

        self._reset_widgets()
        self.assertTrue(self.dock.apply_selected_preset())
        self._assert_series_settings()
        self.assertIn("Applied preset 'AMS Korea L751'", self.dock.preset_status.text())

        # A new dock reads the same stored preset.
        self.dock.cleanup(permanent=True)
        self.dock.deleteLater()
        self.dock = self._open_dock()
        self.assertGreaterEqual(self.dock.preset_combo.findData("AMS Korea L751"), 1)

    def test_unavailable_options_are_skipped_and_named(self):
        self._set_series_settings()
        self._save("Series")
        self._reset_widgets()
        self.availability["recovery_runtime"] = False
        self.dock.preset_combo.setCurrentIndex(self.dock.preset_combo.findData("Series"))
        self.assertTrue(self.dock.apply_selected_preset())
        self.assertFalse(self.dock.smart_recovery_check.isChecked())
        self.assertEqual(self.dock.freedom_slider.value(), 55)
        self.assertEqual(self.dock._series_ink_rgb, (150, 90, 50))
        self.assertIn("Smart Recovery (ONNX Runtime is not installed)", self.dock.preset_status.text())

    def test_rename_delete_and_refused_newer_import(self):
        from qgis.PyQt.QtWidgets import QFileDialog, QInputDialog, QMessageBox
        self._save("First")
        with mock.patch.object(QInputDialog, "getText", return_value=("Second", True)):
            self.assertTrue(self.dock.rename_preset())
        self.assertEqual(self.dock.preset_combo.currentData(), "Second")

        newer = Path(self.temp.name) / "newer.json"
        payload = {"schema_version": "archaeotrace-trace-preset/2", "name": "Future"}
        newer.write_text(json.dumps(payload), encoding="utf-8")
        with mock.patch.object(QFileDialog, "getOpenFileName", return_value=(str(newer), "")), \
                mock.patch.object(QMessageBox, "warning") as warning:
            self.assertFalse(self.dock.import_preset())
        self.assertIn("newer ArchaeoTrace", warning.call_args.args[2])
        self.assertEqual(self.dock.preset_combo.findData("Future"), -1)

        exported = Path(self.temp.name) / "second.json"
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=(str(exported), "")):
            self.assertTrue(self.dock.export_preset())
        self.assertEqual(json.loads(exported.read_text(encoding="utf-8"))["name"], "Second")

        yes = getattr(QMessageBox, "Yes", None) or QMessageBox.StandardButton.Yes
        with mock.patch.object(QMessageBox, "question", return_value=yes):
            self.assertTrue(self.dock.delete_preset())
        self.assertEqual(self.dock.preset_combo.count(), 1)

        with mock.patch.object(QFileDialog, "getOpenFileName", return_value=(str(exported), "")):
            self.assertTrue(self.dock.import_preset())
        self.assertEqual(self.dock.preset_combo.currentData(), "Second")

    def test_unreadable_stored_presets_are_kept_as_a_backup(self):
        from ai_vectorizer.ui.trace_presets_panel import (
            PRESET_BACKUP_SETTINGS_KEY,
            PRESET_SETTINGS_KEY,
        )
        self.settings.setValue(PRESET_SETTINGS_KEY, "{broken")
        self.dock.cleanup(permanent=True)
        self.dock.deleteLater()
        self.dock = self._open_dock()
        self.assertIn("could not be read", self.dock.preset_status.text())
        self.assertEqual(self.dock.preset_combo.count(), 1)
        self.assertTrue(self._save("Fresh"))
        self.assertEqual(self.settings.value(PRESET_BACKUP_SETTINGS_KEY), "{broken")
        self.assertEqual(json.loads(self.settings.value(PRESET_SETTINGS_KEY))["presets"][0]["name"], "Fresh")

    def test_project_remembers_settings_per_raster_and_survives_save(self):
        map_a = SimpleNamespace(id=lambda: "sheet_a_6627III")
        map_b = SimpleNamespace(id=lambda: "sheet b with spaces")
        self._set_series_settings()
        self.assertTrue(self.dock._remember_project_preset(map_a))
        self._reset_widgets()
        self.dock._on_preset_raster_changed(map_b)
        self.assertEqual(self.dock.freedom_slider.value(), 10)
        self.dock._on_preset_raster_changed(map_a)
        self._assert_series_settings()
        self.assertIn("saved with this map", self.dock.preset_status.text())

        path = str(Path(self.temp.name) / "memory.qgz")
        self.assertTrue(self.project.write(path))
        self.project.clear()
        self.assertTrue(self.project.read(path))
        self._reset_widgets()
        self.dock._preset_raster_id = None
        self.dock._on_preset_raster_changed(map_a)
        self._assert_series_settings()

    def test_preset_controls_are_locked_while_tracing(self):
        self._save("Locked")
        self.dock._set_trace_configuration_enabled(False)
        for widget in self.dock._preset_widgets():
            self.assertFalse(widget.isEnabled())
        self.dock._set_trace_configuration_enabled(True)
        self.assertTrue(self.dock.preset_apply_btn.isEnabled())

    def test_idle_clear_forgets_the_remembered_ink_colour(self):
        self.dock._on_ink_color_choice((150, 90, 50))
        self.dock._on_ink_color_changed(None, False)
        self.assertIn("150, 90, 50", self.dock.ink_color_status.text())
        self.assertTrue(self.dock.ink_color_clear_btn.isEnabled())
        self.assertTrue(self.dock.clear_ink_color())
        self.assertIsNone(self.dock._series_ink_rgb)
        self.assertIn("all ink", self.dock.ink_color_status.text())


if __name__ == "__main__":
    unittest.main()
