"""Native annotation UI regressions with real crop/GPKG data and Qt widgets.

Only user dialogs are replaced. All approval fixtures are synthetic and never
change the review status of a public pilot crop.
"""

import gc
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


@unittest.skipUnless(importlib.util.find_spec("qgis") is not None,
                     "native QGIS unavailable")
class NativeAnnotationUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qgis.core import QgsApplication
        cls.application = QgsApplication.instance()
        cls.owned = cls.application is None
        if cls.owned:
            cls.application = QgsApplication([], False)
        cls.application.initQgis()
        from osgeo import gdal
        gdal.AllRegister()

    @classmethod
    def tearDownClass(cls):
        if cls.owned:
            from qgis.core import QgsProject
            QgsProject.instance().clear()
            gc.collect()
            cls.application.exitQgis()

    def setUp(self):
        from qgis.core import QgsCoordinateReferenceSystem, QgsProject, QgsRectangle
        from qgis.gui import QgsLayerTreeMapCanvasBridge, QgsMapCanvas
        from qgis.PyQt.QtWidgets import QMainWindow
        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock
        from osgeo import gdal

        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.project = QgsProject.instance()
        self.project.clear()
        self.window = QMainWindow()
        self.canvas = QgsMapCanvas(self.window)
        self.window.setCentralWidget(self.canvas)
        self.window.resize(512, 512)
        self.window.show()
        self.application.processEvents()
        self.canvas.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
        self.canvas.setExtent(QgsRectangle(10000, 10000, 10512, 10512))
        self.bridge = QgsLayerTreeMapCanvasBridge(self.project.layerTreeRoot(), self.canvas)

        class Iface:
            def __init__(self, canvas, window):
                self.canvas, self.window = canvas, window
                self.current_layer = None

            def mapCanvas(self): return self.canvas
            def mainWindow(self): return self.window
            def setActiveLayer(self, layer): self.current_layer = layer
            def activeLayer(self): return self.current_layer

        self.iface = Iface(self.canvas, self.window)
        # Keep optional runtime/cache discovery inside the synthetic fixture.
        self.models_patch = mock.patch.object(AIVectorizerDock, "_sam_models_dir",
                                             return_value=str(self.root / "models"))
        self.models_patch.start()
        self.hed_patch = mock.patch.multiple(EdgeDetector, **{
            name: getattr(EdgeDetector, name) for name in (
                "HED_MODEL_DIR", "HED_PROTOTXT", "HED_CAFFEMODEL",
                "_hed_runtime_status_cache", "_hed_runtime_status_signature",
            )
        })
        self.hed_patch.start()
        self.dock = AIVectorizerDock(self.iface, self.window)
        self.image = self.root / "synthetic-ui.png"
        memory = gdal.GetDriverByName("MEM").Create("", 512, 512, 3, gdal.GDT_Byte)
        for index, value in enumerate((250, 245, 235), 1):
            memory.GetRasterBand(index).Fill(value)
        created = gdal.GetDriverByName("PNG").CreateCopy(str(self.image), memory)
        created = None
        memory = None
        self.sessions = self.root / "sessions"
        self.sessions.mkdir()

    def tearDown(self):
        session = self.dock._annotation_session
        if session is not None:
            for layer in session.layers.values():
                if layer.isEditable():
                    layer.rollBack()
        self.dock.cleanup(permanent=True)
        self.dock._annotation_session = None
        self.canvas.setLayers([])
        self.project.clear()
        if session is not None:
            session.layers.clear()
            session.raster = None
        self.iface.current_layer = None
        self.dock.deleteLater()
        self.bridge.deleteLater()
        self.window.close()
        self.window.deleteLater()
        self.application.processEvents()
        self.models_patch.stop()
        self.hed_patch.stop()
        self.dock = self.bridge = self.canvas = self.window = None
        gc.collect()
        self.temporary.cleanup()

    def _open_crop(self):
        from ai_vectorizer.ui import main_dialog
        with mock.patch.object(main_dialog.QFileDialog, "getOpenFileName",
                               return_value=(str(self.image), "PNG (*.png)")), \
             mock.patch.object(main_dialog.QFileDialog, "getExistingDirectory",
                               return_value=str(self.sessions)), \
             mock.patch.object(main_dialog.QMessageBox, "warning") as warning:
            self.assertTrue(self.dock.open_annotation_project())
            warning.assert_not_called()
        self.application.processEvents()
        return self.dock._annotation_session

    def _add_reviewed_reference(self, session):
        from qgis.core import QgsFeature, QgsGeometry, QgsPointXY
        from ai_vectorizer.core.annotation_project import pixel_to_map

        contours = session.layers["contours"]
        # The fixture is blank, so this explicit synthetic reference is the
        # only contour. Keep endpoint coordinates away from crop boundaries.
        points = ((20, 80), (80, 80), (140, 80))
        line = QgsFeature(contours.fields())
        line.setGeometry(QgsGeometry.fromPolylineXY([
            QgsPointXY(*pixel_to_map(point, session.geotransform)) for point in points
        ]))
        line["label_state"], line["sample_id"] = "manual", session.sample_id
        self.assertTrue(contours.addFeature(line))
        reviewed = session.layers["reviewed"]
        area = QgsFeature(reviewed.fields())
        ring = ((-.5, -.5), (511.5, -.5), (511.5, 511.5), (-.5, 511.5), (-.5, -.5))
        area.setGeometry(QgsGeometry.fromPolygonXY([[
            QgsPointXY(*pixel_to_map(point, session.geotransform)) for point in ring
        ]]))
        area["label_state"], area["sample_id"] = "manual", session.sample_id
        self.assertTrue(reviewed.addFeature(area))
        for layer in session.layers.values():
            self.assertTrue(layer.commitChanges())
        contours.selectAll()
        self.assertEqual(len(contours.selectedFeatureIds()), 1)
        return contours.selectedFeatureIds()[0], points

    def _export(self, destination, answer, difficulty="Hard"):
        from ai_vectorizer.ui import main_dialog
        with mock.patch.object(main_dialog.QFileDialog, "getSaveFileName",
                               return_value=(str(destination), "JSON (*.json)")), \
             mock.patch.object(main_dialog.QInputDialog, "getText",
                               return_value=("synthetic-human-fixture", True)), \
             mock.patch.object(main_dialog.QMessageBox, "question", return_value=answer), \
             mock.patch.object(main_dialog.QInputDialog, "getItem",
                               side_effect=lambda _parent, _title, _label, choices, *_args:
                               (choices[1] if difficulty == "Hard" else choices[0], True)) as item, \
             mock.patch.object(main_dialog.QMessageBox, "warning") as warning:
            self.assertTrue(self.dock.export_annotation_project())
            warning.assert_not_called()
        return json.loads(destination.read_text(encoding="utf-8")), item

    def test_open_registers_crop_and_five_editable_layers_and_fits_canvas(self):
        session = self._open_crop()
        expected = {session.raster.id(), *(layer.id() for layer in session.layers.values())}
        self.assertEqual(len(expected), 6)
        self.assertEqual(set(self.project.mapLayers()), expected)
        self.assertEqual({layer.id() for layer in self.canvas.layers()}, expected)
        self.assertEqual(set(session.layers), {
            "contours", "other_ink", "other_ink_regions", "reviewed", "ignore",
        })
        for layer in [session.raster, *session.layers.values()]:
            node = self.project.layerTreeRoot().findLayer(layer.id())
            self.assertIsNotNone(node)
            self.assertTrue(node.isVisible())
        self.assertTrue(all(layer.isEditable() for layer in session.layers.values()))
        self.assertTrue(session.gpkg_path.is_file())
        self.assertIs(self.iface.current_layer, session.layers["contours"])
        canvas_extent, crop_extent = self.canvas.extent(), session.raster.extent()
        self.assertTrue(canvas_extent.contains(crop_extent))
        self.assertAlmostEqual(canvas_extent.center().x(), crop_extent.center().x(), places=6)
        self.assertAlmostEqual(canvas_extent.center().y(), crop_extent.center().y(), places=6)
        self.assertLessEqual(canvas_extent.width(), crop_extent.width() * 1.5)
        self.assertLessEqual(canvas_extent.height(), crop_extent.height() * 1.5)
        self.assertTrue(self.dock.annotation_export_btn.isEnabled())

    def test_yes_exports_canonical_selected_target_and_exact_pixel_endpoints(self):
        from ai_vectorizer.ui.main_dialog import _message_box_button
        session = self._open_crop()
        feature_id, points = self._add_reviewed_reference(session)
        label, difficulty_dialog = self._export(self.root / "approved.json",
                                                _message_box_button("Yes"))
        self.assertEqual(label["annotation_status"], "approved")
        self.assertEqual(label["review_level"], "single_reviewer_pilot")
        self.assertTrue(label["all_visible_contours_confirmed"])
        self.assertFalse(label["publication_ranking_eligible"])
        self.assertEqual(label["case_difficulty"], "hard")
        target = label["evaluation"]["target_contour_id"]
        self.assertEqual(target, "contours-{}-0".format(feature_id))
        self.assertIsInstance(target, str)
        self.assertEqual(label["evaluation"]["prompt"]["start_xy"], list(points[0]))
        self.assertEqual(label["evaluation"]["prompt"]["end_xy"], list(points[-1]))
        difficulty_dialog.assert_called_once()

    def test_no_exports_unapproved_draft_even_with_a_selected_contour(self):
        from ai_vectorizer.ui.main_dialog import _message_box_button
        session = self._open_crop()
        self._add_reviewed_reference(session)
        label, difficulty_dialog = self._export(self.root / "draft.json",
                                                _message_box_button("No"))
        self.assertEqual(label["annotation_status"], "draft")
        self.assertEqual(label["review_level"], "unreviewed")
        self.assertFalse(label["all_visible_contours_confirmed"])
        self.assertFalse(label["publication_ranking_eligible"])
        self.assertIsNone(label["evaluation"])
        self.assertEqual(label["case_difficulty"], "unassessed")
        self.assertEqual(len(label["contours"]), 1)
        difficulty_dialog.assert_not_called()


if __name__ == "__main__":
    unittest.main()
