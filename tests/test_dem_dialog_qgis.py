"""Real-QGIS spot-height field initialization and terrain input regressions."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    from qgis.core import (
        QgsApplication, QgsFeature, QgsField, QgsGeometry, QgsProject,
        QgsRasterLayer, QgsVectorLayer,
    )
    from qgis.PyQt.QtCore import QEventLoop, QMetaType, QTimer
    try:
        from qgis.PyQt.QtCore import QVariant
    except ImportError:
        QVariant = None
    from qgis.PyQt.QtWidgets import QMainWindow
    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError("QGIS bindings are required for this job") from exc
    HAS_QGIS = False


def _field_type(name):
    if QVariant is not None and hasattr(QVariant, name):
        return getattr(QVariant, name)
    return getattr(QMetaType.Type, "QString" if name == "String" else name)


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class DemDialogQgisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._owns_application = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        if cls._owns_application:
            cls.application.initQgis()
        from ai_vectorizer.core.dem_pipeline import build_dem_request, DemInputError
        from ai_vectorizer.ui.dem_dialog import DemBuildDialog
        cls.build_request = staticmethod(build_dem_request)
        cls.InputError = DemInputError
        cls.Dialog = DemBuildDialog

    @classmethod
    def tearDownClass(cls):
        QgsProject.instance().clear()
        if cls._owns_application:
            cls.application.exitQgis()

    def setUp(self):
        QgsProject.instance().clear()
        self.temp = tempfile.TemporaryDirectory()
        self.window = QMainWindow()
        self.dialogs = []
        self.contours = self._layer(
            "LineString", "contours", [("tag", "String"), ("elevation", "Double")],
            [("LINESTRING(0 0,100 0)", ["lower", 100.0]),
             ("LINESTRING(0 100,100 100)", ["upper", 120.0])],
        )

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.close()
            dialog.deleteLater()
        QgsApplication.processEvents()
        QgsProject.instance().clear()
        self.window.close()
        self.temp.cleanup()

    def _layer(self, geometry, name, fields, rows, crs="EPSG:3857"):
        layer = QgsVectorLayer(geometry + "?crs=" + crs, name, "memory")
        self.assertTrue(layer.isValid())
        layer.dataProvider().addAttributes([QgsField(n, _field_type(t)) for n, t in fields])
        layer.updateFields()
        features = []
        for wkt, values in rows:
            feature = QgsFeature(layer.fields())
            feature.setGeometry(QgsGeometry.fromWkt(wkt))
            feature.setAttributes(values)
            features.append(feature)
        self.assertTrue(layer.dataProvider().addFeatures(features)[0])
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)
        return layer

    def _spots(self, *, name="spots", field="elevation", value=130.0, crs="EPSG:3857", wkt="POINT(50 50)", geometry="Point"):
        return self._layer(geometry, name,
                           [("id", "Int"), ("label", "String"), (field, "Double")],
                           [(wkt, [123456, "height", value])], crs=crs)

    def _dialog(self):
        dialog = self.Dialog(SimpleNamespace(mainWindow=lambda: self.window),
                             contour_layer=self.contours, language="en")
        self.dialogs.append(dialog)
        dialog.pixel_size.setValue(10.0)
        return dialog

    def _request(self, spot=None, field="elevation", filename="dem.tif"):
        return self.build_request(self.contours, "elevation", 10.0,
                                  str(Path(self.temp.name) / filename),
                                  spot_layer=spot, spot_field=field)

    def test_preexisting_selected_spot_has_bound_elevation_on_first_open(self):
        spots = self._spots()
        dialog = self._dialog()
        self.assertEqual(dialog.spot_combo.currentLayer(), spots)
        self.assertEqual(dialog.spot_field.currentField(), "elevation")
        request = self._request(dialog.spot_combo.currentLayer(), dialog.spot_field.currentField())
        self.assertIn(spots.id() + "::~::0::~::2::~::0", request.interpolation_data)
        self.assertEqual(request.spot_stats.unique_values, (130.0,))

    def test_custom_contour_height_requires_choice_and_preserves_it_on_reopen(self):
        self.contours = self._layer(
            "LineString", "survey contours", [("id", "Int"), ("height_m", "Double")],
            [("LINESTRING(0 0,100 0)", [1, 100.0]),
             ("LINESTRING(0 100,100 100)", [2, 120.0])],
        )
        dialog = self._dialog()
        self.assertEqual(dialog.contour_field.currentField(), "")
        self.assertFalse(dialog.build_button.isEnabled())
        with self.assertRaisesRegex(self.InputError, "Elevation field"):
            self.build_request(self.contours, dialog.contour_field.currentField(), 10.0,
                               str(Path(self.temp.name) / "survey.tif"))
        dialog.contour_field.setField("height_m")
        self.assertTrue(dialog.build_button.isEnabled())
        request = self.build_request(self.contours, dialog.contour_field.currentField(), 10.0,
                                     str(Path(self.temp.name) / "survey.tif"))
        self.assertEqual(request.contour_stats.unique_values, (100.0, 120.0))
        dialog.set_inputs(self.contours)
        self.assertEqual(dialog.contour_field.currentField(), "height_m")
        dialog.contour_field.setField("")
        dialog.set_inputs(self.contours)
        self.assertEqual(dialog.contour_field.currentField(), "")
        self.assertFalse(dialog.build_button.isEnabled())

    def test_switching_contour_layer_does_not_inherit_numeric_id_or_height(self):
        dialog = self._dialog()
        self.assertEqual(dialog.contour_field.currentField(), "elevation")
        self.assertTrue(dialog.build_button.isEnabled())
        custom = self._layer(
            "LineString", "external", [("id", "Int"), ("height_m", "Double")],
            [("LINESTRING(0 0,100 0)", [1, 100.0]),
             ("LINESTRING(0 100,100 100)", [2, 120.0])],
        )
        dialog.contour_combo.setLayer(custom)
        self.assertEqual(dialog.contour_field.currentField(), "")
        self.assertFalse(dialog.build_button.isEnabled())
        dialog.contour_field.setField("height_m")
        dialog.set_inputs(self.contours)
        self.assertEqual(dialog.contour_field.currentField(), "elevation")
        dialog.set_inputs(custom)
        self.assertEqual(dialog.contour_field.currentField(), "")
        self.assertFalse(dialog.build_button.isEnabled())

    def test_nonnumeric_named_elevation_does_not_fall_back_to_id(self):
        self.contours = self._layer(
            "LineString", "text heights", [("id", "Int"), ("elevation", "String")],
            [("LINESTRING(0 0,100 0)", [1, "100m"]),
             ("LINESTRING(0 100,100 100)", [2, "120m"])],
        )
        dialog = self._dialog()
        self.assertEqual(dialog.contour_field.currentField(), "")
        self.assertFalse(dialog.build_button.isEnabled())

    def test_no_points_is_optional_and_field_stays_empty(self):
        dialog = self._dialog()
        self.assertIsNone(dialog.spot_combo.currentLayer())
        self.assertEqual(dialog.spot_field.currentField(), "")
        self.assertFalse(dialog.spot_field.isEnabled())
        dialog._set_running(True)
        dialog._set_running(False)
        self.assertFalse(dialog.spot_field.isEnabled())
        self.assertIsNone(self._request().spot_stats)

    def test_custom_height_field_requires_choice_and_never_uses_numeric_id(self):
        spots = self._spots(field="height_m")
        dialog = self._dialog()
        self.assertEqual(dialog.spot_combo.currentLayer(), spots)
        self.assertEqual(dialog.spot_field.currentField(), "")
        self.assertFalse(dialog.build_button.isEnabled())
        with self.assertRaisesRegex(self.InputError, "Select a numeric spot-height"):
            self._request(spots, dialog.spot_field.currentField())
        dialog.spot_field.setField("height_m")
        self.assertTrue(dialog.build_button.isEnabled())
        self.assertEqual(self._request(spots, dialog.spot_field.currentField()).spot_stats.maximum, 130.0)
        dialog.set_inputs(self.contours)
        self.assertEqual(dialog.spot_field.currentField(), "height_m")

    def test_switching_to_custom_schema_clears_previous_height_choice(self):
        self._spots()
        dialog = self._dialog()
        custom = self._spots(name="survey", field="height_m")
        dialog.spot_combo.setLayer(custom)
        self.assertEqual(dialog.spot_field.currentField(), "")
        dialog.spot_combo.setLayer(None)
        self.assertEqual(dialog.spot_field.currentField(), "")
        self.assertFalse(dialog.spot_field.isEnabled())

    def test_selected_point_extent_matches_build_request_grid(self):
        self._spots(wkt="POINT(1000 1000)")
        dialog = self._dialog()
        request = self._request(dialog.spot_combo.currentLayer())
        self.assertEqual((request.grid.columns, request.grid.rows), (100, 100))
        self.assertIn("100 × 100 (10,000 cells)", dialog.grid_summary.text())
        dialog.spot_combo.setLayer(None)
        self.assertIn("10 × 10 (100 cells)", dialog.grid_summary.text())

    def test_mismatched_crs_is_rejected_without_transforming_coordinates(self):
        spots = self._spots(crs="EPSG:4326", wkt="POINT(127 36)")
        dialog = self._dialog()
        self.assertFalse(dialog.build_button.isEnabled())
        self.assertIn("same CRS", dialog.grid_summary.text())
        with self.assertRaisesRegex(self.InputError, "same CRS"):
            self._request(spots)

    def test_missing_nonnumeric_and_null_elevations_are_rejected(self):
        spots = self._spots()
        with self.assertRaisesRegex(self.InputError, "does not exist"):
            self._request(spots, "missing")
        with self.assertRaisesRegex(self.InputError, "must be numeric"):
            self._request(spots, "label")
        null_spot = self._spots(name="null", value=None)
        with self.assertRaisesRegex(self.InputError, "missing or invalid elevations"):
            self._request(null_spot)

    def test_unsaved_point_edits_are_rejected_without_committing_them(self):
        spots = self._spots()
        spots.startEditing()
        identity = next(spots.getFeatures()).id()
        spots.changeAttributeValue(identity, 2, 150.0)
        with self.assertRaisesRegex(self.InputError, "Save or roll back"):
            self._request(spots)
        self.assertTrue(spots.isModified())
        self.assertEqual(next(spots.getFeatures())[2], 150.0)
        spots.rollBack()

    def test_point_and_multipoint_run_real_tin_translate_and_hillshade(self):
        import qgis
        from ai_vectorizer.core.dem_pipeline import DemPipelineRunner
        executable_directory = Path(sys.executable).resolve().parent
        if (executable_directory / "gdal_translate").is_file():
            entries = os.environ.get("PATH", "").split(os.pathsep)
            if str(executable_directory) not in entries:
                os.environ["PATH"] = os.pathsep.join((str(executable_directory), *entries))
        for candidate in (
            Path(qgis.__file__).resolve().parent.parent / "plugins",
            Path(QgsApplication.pkgDataPath()) / "python/plugins",
            Path(QgsApplication.prefixPath()) / "share/qgis/python/plugins",
            Path(sys.executable).resolve().parent.parent / "Resources/qgis/python/plugins",
        ):
            if (candidate / "processing").is_dir():
                sys.path.insert(0, str(candidate))
                break
        from processing.core.Processing import Processing
        Processing.initialize()
        for geometry, wkt in (("Point", "POINT(50 50)"),
                              ("MultiPoint", "MULTIPOINT((30 30),(70 70))")):
            with self.subTest(geometry=geometry):
                spots = self._spots(name=geometry, geometry=geometry, wkt=wkt)
                request = self._request(spots, filename=geometry + ".tif")
                runner = DemPipelineRunner()
                event_loop, terminal = QEventLoop(), []
                def finish(kind, *details):
                    terminal.append((kind, details))
                    event_loop.quit()
                runner.succeeded.connect(lambda *paths: finish("succeeded", *paths))
                runner.failed.connect(lambda message: finish("failed", message))
                runner.canceled.connect(lambda: finish("canceled"))
                timer = QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(lambda: finish("timeout"))
                timer.start(30_000)
                runner.start(request)
                execute = getattr(event_loop, "exec", None) or event_loop.exec_
                execute()
                timer.stop()
                self.assertEqual(terminal[0][0], "succeeded", terminal)
                self.assertFalse(runner.is_running)
                self.assertEqual(runner._resources, [])
                for path in (request.dem_path, request.hillshade_path):
                    self.assertTrue(Path(path).is_file())
                    self.assertTrue(QgsRasterLayer(path, "output", "gdal").isValid())
                QgsProject.instance().removeMapLayer(spots.id())


if __name__ == "__main__":
    unittest.main()
