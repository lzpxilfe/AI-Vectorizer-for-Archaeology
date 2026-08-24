"""Runtime QGIS regressions for edit and plugin lifecycle safety.

The suite skips cleanly in ordinary Python and runs in the QGIS CI image (or
with QGIS' bundled Python locally).
"""

from pathlib import Path
import gc
import os
import sys
import tempfile
import unittest
from unittest import mock


try:
    from qgis.core import (
        QgsApplication,
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsDefaultValue,
        QgsFeature,
        QgsField,
        QgsFieldConstraints,
        QgsGeometry,
        QgsPointXY,
        QgsProject,
        QgsRasterLayer,
        QgsVectorLayer,
    )
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QEventLoop, QMetaType, QTimer, QUrl
    try:
        from qgis.PyQt.QtCore import QVariant
    except ImportError:
        QVariant = None
    from qgis.PyQt.QtWidgets import QMainWindow, QToolBar

    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError(
            "QGIS bindings are required for this runtime safety job"
        ) from exc
    HAS_QGIS = False


def _field_type(name):
    meta_name = "QString" if name == "String" else name
    meta_types = getattr(QMetaType, "Type", None)
    if meta_types is not None and hasattr(meta_types, meta_name):
        return getattr(meta_types, meta_name)
    if QVariant is not None and hasattr(QVariant, name):
        return getattr(QVariant, name)
    raise RuntimeError(name)


def _field_constraint(name):
    legacy = getattr(QgsFieldConstraints, name, None)
    if legacy is not None:
        return legacy
    return getattr(QgsFieldConstraints.Constraint, name)


def _constraint_strength(name):
    legacy = getattr(QgsFieldConstraints, name, None)
    if legacy is not None:
        return legacy
    return getattr(QgsFieldConstraints.ConstraintStrength, name)


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class QgisRuntimeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._owns_application = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        cls._temporary_directories = []
        if cls._owns_application:
            cls.application.initQgis()
        from ai_vectorizer.plugin import AIVectorizer
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool

        cls.AIVectorizer = AIVectorizer
        cls.SmartTraceTool = SmartTraceTool

    @classmethod
    def tearDownClass(cls):
        QgsProject.instance().clear()
        if cls._owns_application:
            cls.application.exitQgis()
        for temporary_directory in cls._temporary_directories:
            temporary_directory.cleanup()

    def setUp(self):
        QgsProject.instance().clear()
        self.crs = QgsCoordinateReferenceSystem("EPSG:3857")
        self.canvas = QgsMapCanvas()
        self.canvas.setDestinationCrs(self.crs)

    def _tool(self, layer, raster_crs=None):
        crs = raster_crs or self.crs

        class Raster:
            def crs(self):
                return crs

        return self.SmartTraceTool(
            self.canvas,
            Raster(),
            layer,
            freehand=True,
            language="en",
        )

    def test_resume_updates_geometry_and_elevation_together(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        feature = QgsFeature(layer.fields())
        feature.setGeometry(
            QgsGeometry.fromPolylineXY([QgsPointXY(0, 0), QgsPointXY(1, 0)])
        )
        ok, added = layer.dataProvider().addFeatures([feature])
        self.assertTrue(ok)

        tool = self._tool(layer)
        tool.path_points = [QgsPointXY(1, 0), QgsPointXY(2, 0)]
        tool.resume_feature_id = added[0].id()
        tool.resume_at_start = False

        self.assertTrue(tool.save_to_layer(closed=False, elevation=20.0))
        self.assertTrue(layer.isEditable())
        elevation_index = layer.fields().indexOf("elevation")
        self.assertGreaterEqual(elevation_index, 0)
        updated = layer.getFeature(added[0].id())
        self.assertEqual(updated[elevation_index], 20.0)
        self.assertEqual(updated.geometry().asPolyline()[-1], QgsPointXY(2, 0))

        # Field creation, geometry and elevation are one QGIS edit command.
        layer.undoStack().undo()
        self.assertEqual(layer.fields().indexOf("elevation"), -1)
        restored = layer.getFeature(added[0].id())
        self.assertEqual(restored.geometry().asPolyline()[-1], QgsPointXY(1, 0))

    def test_new_feature_is_added_as_one_undoable_edit_command(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(2, 0)]
        )

        self.assertTrue(tool.save_geometry(geometry, elevation=12.0))
        self.assertTrue(layer.isEditable())
        self.assertEqual(layer.featureCount(), 1)
        self.assertGreaterEqual(layer.fields().indexOf("elevation"), 0)

        layer.undoStack().undo()
        self.assertEqual(layer.featureCount(), 0)
        self.assertEqual(layer.fields().indexOf("elevation"), -1)

    def test_failed_feature_add_rolls_back_new_elevation_field(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)

        class RejectingLayer:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getattr__(self, name):
                return getattr(self.wrapped, name)

            def addFeature(self, _feature):
                return False

        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(2, 0)]
        )
        rejecting = RejectingLayer(layer)

        self.assertFalse(
            tool._add_geometry_feature(rejecting, geometry, elevation=12.0)
        )
        layer.updateFields()
        self.assertEqual(layer.fields().indexOf("elevation"), -1)
        self.assertEqual(layer.featureCount(), 0)

    def test_idle_ctrl_z_is_delegated_to_qgis_edit_stack(self):
        from ai_vectorizer.tools.smart_trace_tool import _qt_value

        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(2, 0)]
        )
        self.assertTrue(tool.save_geometry(geometry, elevation=12.0))
        self.assertEqual(layer.featureCount(), 1)

        class Event:
            ignored = False
            accepted = False

            def key(self):
                return _qt_value("Key_Z", "Key")

            def modifiers(self):
                return _qt_value("ControlModifier", "KeyboardModifier")

            def ignore(self):
                self.ignored = True

            def accept(self):
                self.accepted = True

        event = Event()
        self.assertFalse(tool.is_tracing)
        tool.keyPressEvent(event)
        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)

        # QGIS receives the ignored shortcut and invokes this layer edit stack.
        layer.undoStack().undo()
        self.assertEqual(layer.featureCount(), 0)

    def test_file_layer_write_stays_uncommitted_until_qgis_save(self):
        from ai_vectorizer.ui.main_dialog import _write_vector_layer, _writer_no_error

        source = QgsVectorLayer("LineString?crs=EPSG:3857", "source", "memory")
        source.dataProvider().addAttributes(
            [QgsField("elevation", _field_type("Double"))]
        )
        source.updateFields()
        temporary_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(temporary_directory)
        path = str(Path(temporary_directory.name) / "contours.shp")
        self.assertEqual(
            _write_vector_layer(source, path, self.crs)[0],
            _writer_no_error(),
        )
        layer = QgsVectorLayer(path, "contours", "ogr")
        self.assertTrue(layer.isValid())
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        self.assertTrue(tool._ensure_edit_session(layer))
        feature = QgsFeature(layer.fields())
        feature.setGeometry(
            QgsGeometry.fromPolylineXY([QgsPointXY(0, 0), QgsPointXY(3, 0)])
        )
        feature[0] = 20.0

        self.assertTrue(tool._add_feature(layer, feature))
        self.assertEqual(layer.featureCount(), 1)
        on_disk = QgsVectorLayer(path, "on-disk", "ogr")
        self.assertTrue(on_disk.isValid())
        self.assertEqual(on_disk.featureCount(), 0)

        layer.undoStack().undo()
        self.assertEqual(layer.featureCount(), 0)
        del on_disk
        del tool
        del layer
        gc.collect()
        QgsApplication.processEvents()

    def test_feature_defaults_and_zm_guard(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "defaults", "memory")
        layer.dataProvider().addAttributes(
            [
                QgsField("elevation", _field_type("Double")),
                QgsField("note", _field_type("String")),
                QgsField("id", _field_type("Int")),
            ]
        )
        layer.updateFields()
        note_index = layer.fields().indexOf("note")
        id_index = layer.fields().indexOf("id")
        layer.setDefaultValueDefinition(note_index, QgsDefaultValue("'default-note'"))
        layer.setDefaultValueDefinition(id_index, QgsDefaultValue("1000"))
        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.language = "en"
        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(1, 1)]
        )

        feature = tool._build_feature(layer, geometry, 7.0)
        self.assertEqual(feature[note_index], "default-note")
        self.assertEqual(feature[id_index], 1000)

        multi_layer = QgsVectorLayer(
            "MultiLineString?crs=EPSG:3857",
            "multi",
            "memory",
        )
        multi_feature = tool._build_feature(multi_layer, geometry)
        self.assertTrue(multi_feature.geometry().isMultipart())

        z_layer = QgsVectorLayer("LineStringZ?crs=EPSG:3857", "z", "memory")
        m_layer = QgsVectorLayer("LineStringM?crs=EPSG:3857", "m", "memory")
        self.assertEqual(tool.unsupported_output_reason(z_layer), "z_or_m")
        self.assertEqual(tool.unsupported_output_reason(m_layer), "z_or_m")

    def test_hard_field_constraints_reject_atomic_feature_add(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "required", "memory")
        layer.dataProvider().addAttributes(
            [QgsField("required", _field_type("String"))]
        )
        layer.updateFields()
        required_index = layer.fields().indexOf("required")
        layer.setFieldConstraint(
            required_index,
            _field_constraint("ConstraintNotNull"),
            _constraint_strength("ConstraintStrengthHard"),
        )
        tool = self._tool(layer)
        geometry = QgsGeometry.fromPolylineXY(
            [QgsPointXY(0, 0), QgsPointXY(2, 0)]
        )

        # Adding the elevation schema and feature is one edit command. A hard
        # failure on another required field must roll both changes back.
        self.assertFalse(tool.save_geometry(geometry, elevation=12.0))
        layer.updateFields()
        self.assertEqual(layer.featureCount(), 0)
        self.assertEqual(layer.fields().indexOf("elevation"), -1)

    def test_spot_layer_does_not_adopt_same_named_user_layer(self):
        from ai_vectorizer.config import DEFAULT_SPOT_LAYER_NAME

        project = QgsProject.instance()
        user_layer = QgsVectorLayer(
            "Point?crs=EPSG:3857",
            DEFAULT_SPOT_LAYER_NAME,
            "memory",
        )
        contour_layer = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "contours",
            "memory",
        )
        project.addMapLayer(user_layer)
        project.addMapLayer(contour_layer)

        class Settings:
            def destinationCrs(self):
                return contour_layer.crs()

        class Canvas:
            def mapSettings(self):
                return Settings()

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.spot_height_layer = None
        tool.vector_layer = contour_layer
        tool.canvas = Canvas()
        owned_layer = tool.get_or_create_spot_layer()

        self.assertNotEqual(owned_layer.id(), user_layer.id())
        self.assertTrue(
            owned_layer.customProperty(tool.SPOT_LAYER_OWNERSHIP_PROPERTY, False)
        )

    def test_removed_owned_spot_layer_is_recreated_without_dead_wrapper(self):
        project = QgsProject.instance()
        contour_layer = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "contours",
            "memory",
        )
        project.addMapLayer(contour_layer)

        class Settings:
            def destinationCrs(self):
                return contour_layer.crs()

        class Canvas:
            def mapSettings(self):
                return Settings()

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        tool.spot_height_layer = None
        tool.vector_layer = contour_layer
        tool.canvas = Canvas()
        first = tool.get_or_create_spot_layer()
        first_id = first.id()
        project.removeMapLayer(first_id)
        gc.collect()
        QgsApplication.processEvents()

        second = tool.get_or_create_spot_layer()
        self.assertTrue(second.isValid())
        self.assertNotEqual(second.id(), first_id)

    def test_non_editable_layer_starts_one_atomic_edit_command(self):
        class Layer:
            def __init__(self):
                self.editable = False
                self.commands = []
                self.attributes = {}

            def isEditable(self):
                return self.editable

            def startEditing(self):
                self.editable = True
                return True

            def beginEditCommand(self, _label):
                self.commands.append("begin")

            def changeGeometry(self, _feature_id, _geometry):
                return True

            def changeAttributeValue(self, _feature_id, index, value):
                self.attributes[index] = value
                return True

            def endEditCommand(self):
                self.commands.append("end")

            def destroyEditCommand(self):
                self.commands.append("destroy")

            def updateExtents(self):
                pass

            def fields(self):
                return []

            class _Feature:
                @staticmethod
                def isValid():
                    return True

            def getFeature(self, _feature_id):
                return self._Feature()

        tool = self.SmartTraceTool.__new__(self.SmartTraceTool)
        layer = Layer()

        self.assertTrue(tool._update_feature(layer, 4, object(), {2: 55.0}))
        self.assertTrue(layer.editable)
        self.assertEqual(layer.commands, ["begin", "end"])
        self.assertEqual(layer.attributes, {2: 55.0})

    def test_canvas_crs_change_resets_trace_and_refreshes_transforms(self):
        raster_crs = QgsCoordinateReferenceSystem("EPSG:4326")
        layer = QgsVectorLayer("LineString?crs=EPSG:4326", "contours", "memory")
        tool = self._tool(layer, raster_crs=raster_crs)
        tool.activate()
        tool.is_tracing = True
        tool.path_points = [QgsPointXY(0, 0), QgsPointXY(1, 1)]

        new_canvas_crs = QgsCoordinateReferenceSystem("EPSG:32652")
        self.canvas.setDestinationCrs(new_canvas_crs)
        self.assertFalse(tool.is_tracing)
        self.assertEqual(tool.path_points, [])

        source_point = QgsPointXY(500000, 4000000)
        actual = tool._map_point_to_raster(source_point)
        expected_transform = QgsCoordinateTransform(
            new_canvas_crs,
            raster_crs,
            QgsProject.instance(),
        )
        expected = expected_transform.transform(source_point)
        self.assertAlmostEqual(actual.x(), expected.x(), places=8)
        self.assertAlmostEqual(actual.y(), expected.y(), places=8)
        tool.deactivate()

    def test_active_tool_stops_when_output_source_is_replaced(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        emissions = []
        tool.deactivated.connect(lambda: emissions.append(True))
        self.canvas.setMapTool(tool)
        self.assertIs(self.canvas.mapTool(), tool)

        layer.dataSourceChanged.emit()

        self.assertIsNot(self.canvas.mapTool(), tool)
        self.assertEqual(emissions, [True])

    def test_loaded_output_path_detection_decodes_uri_and_aliases(self):
        from ai_vectorizer.ui import main_dialog
        from ai_vectorizer.ui.main_dialog import (
            AIVectorizerDock,
            _layer_file_path,
            _project_layers_using_path,
            _write_vector_layer,
        )

        source = QgsVectorLayer("LineString?crs=EPSG:3857", "source", "memory")
        temporary_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(temporary_directory)
        path = str(Path(temporary_directory.name) / "contours.shp")
        alias = str(Path(temporary_directory.name) / "contours-alias.shp")
        self.assertEqual(_write_vector_layer(source, path, self.crs)[0], 0)
        os.symlink(path, alias)

        loaded = QgsVectorLayer(f"{path}|layerid=0", "loaded contours", "ogr")
        unrelated = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "unrelated",
            "memory",
        )
        self.assertTrue(loaded.isValid())
        QgsProject.instance().addMapLayer(loaded)
        QgsProject.instance().addMapLayer(unrelated)

        matches = _project_layers_using_path(alias)
        self.assertEqual([layer.id() for layer in matches], [loaded.id()])

        class ProviderUriLayer:
            @staticmethod
            def providerType():
                return "ogr"

            @staticmethod
            def source():
                return f"{QUrl.fromLocalFile(path).toString()}|layerid=0"

        self.assertEqual(_layer_file_path(ProviderUriLayer()), path)

        warnings = []

        class PathEdit:
            @staticmethod
            def text():
                return alias

        class DockHarness:
            shp_path = PathEdit()

            @staticmethod
            def _tr(_ko, en):
                return en

        class MessageBoxHarness:
            @staticmethod
            def warning(*args):
                warnings.append(args)

        with mock.patch.object(
            main_dialog,
            "QMessageBox",
            MessageBoxHarness,
        ), mock.patch.object(main_dialog, "_write_vector_layer") as writer:
            AIVectorizerDock.create_shp_layer(DockHarness())
        writer.assert_not_called()
        self.assertEqual(len(warnings), 1)
        self.assertIn("loaded contours", warnings[0][-1])

    def test_preview_temp_directory_tracks_project_layer_lifetime(self):
        from ai_vectorizer.ui.main_dialog import _TemporaryPreviewStore

        store = _TemporaryPreviewStore(QgsProject.instance())
        preview_directory = tempfile.TemporaryDirectory()
        preview_path = Path(preview_directory.name) / "preview.tif"
        preview_path.write_bytes(b"preview")
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "preview", "memory")
        QgsProject.instance().addMapLayer(layer)
        layer_id = layer.id()
        store.track(layer, preview_directory)

        QgsProject.instance().removeMapLayer(layer_id)
        QgsApplication.processEvents()
        self.assertFalse(preview_path.exists())
        self.assertEqual(store._directories, {})

        shutdown_directory = tempfile.TemporaryDirectory()
        shutdown_path = Path(shutdown_directory.name) / "preview.tif"
        shutdown_path.write_bytes(b"preview")
        shutdown_layer = QgsVectorLayer(
            "LineString?crs=EPSG:3857",
            "preview shutdown",
            "memory",
        )
        QgsProject.instance().addMapLayer(shutdown_layer)
        shutdown_id = shutdown_layer.id()
        store.track(shutdown_layer, shutdown_directory)
        store.shutdown()
        self.assertIsNone(QgsProject.instance().mapLayer(shutdown_id))
        self.assertFalse(shutdown_path.exists())
        self.assertFalse(store._connected)

        retry_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(retry_directory)
        retry_path = Path(retry_directory.name) / "preview.tif"
        retry_path.write_bytes(b"preview")

        class InitiallyLockedDirectory:
            name = retry_directory.name

            @staticmethod
            def cleanup():
                raise OSError("provider handle is still open")

        _TemporaryPreviewStore._cleanup_directory(InitiallyLockedDirectory())
        event_loop = QEventLoop()
        QTimer.singleShot(0, event_loop.quit)
        execute = getattr(event_loop, "exec", None) or event_loop.exec_
        execute()
        self.assertFalse(retry_path.exists())

    def test_dem_pipeline_runs_all_processing_stages_and_releases_resources(self):
        import qgis

        # The macOS app bundle keeps GDAL executables beside its Python
        # launcher. A shell-driven offscreen run does not inherit the PATH
        # adjustment made by the normal QGIS application launcher.
        executable_directory = Path(sys.executable).resolve().parent
        if (executable_directory / "gdal_translate").is_file():
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if str(executable_directory) not in path_entries:
                os.environ["PATH"] = os.pathsep.join(
                    (str(executable_directory), *path_entries)
                )

        processing_paths = (
            Path(qgis.__file__).resolve().parent.parent / "plugins",
            Path(QgsApplication.pkgDataPath()) / "python" / "plugins",
            Path(QgsApplication.prefixPath()) / "share" / "qgis" / "python" / "plugins",
            Path(sys.executable).resolve().parent.parent
            / "Resources"
            / "qgis"
            / "python"
            / "plugins",
        )
        for processing_path in processing_paths:
            if (processing_path / "processing").is_dir():
                path_text = str(processing_path)
                if path_text not in sys.path:
                    sys.path.insert(0, path_text)
                break
        from processing.core.Processing import Processing
        from ai_vectorizer.core.dem_pipeline import (
            DemPipelineRunner,
            build_dem_request,
        )

        Processing.initialize()
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "terrain", "memory")
        layer.dataProvider().addAttributes(
            [QgsField("elevation", _field_type("Double"))]
        )
        layer.updateFields()
        features = []
        for elevation, y_coordinate in ((100.0, 0.0), (110.0, 50.0), (120.0, 100.0)):
            feature = QgsFeature(layer.fields())
            feature[0] = elevation
            feature.setGeometry(
                QgsGeometry.fromPolylineXY(
                    [
                        QgsPointXY(0.0, y_coordinate),
                        QgsPointXY(100.0, y_coordinate),
                    ]
                )
            )
            features.append(feature)
        self.assertTrue(layer.dataProvider().addFeatures(features)[0])
        QgsProject.instance().addMapLayer(layer)

        temporary_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(temporary_directory)
        dem_path = str(Path(temporary_directory.name) / "terrain-dem.tif")
        hillshade_path = str(Path(temporary_directory.name) / "terrain-hillshade.tif")
        request = build_dem_request(
            contour_layer=layer,
            contour_field="elevation",
            pixel_size=10.0,
            dem_path=dem_path,
            hillshade_path=hillshade_path,
        )
        runner = DemPipelineRunner()
        event_loop = QEventLoop()
        terminal = []

        def finish(kind, *details):
            terminal.append((kind, details))
            event_loop.quit()

        runner.succeeded.connect(lambda *paths: finish("succeeded", *paths))
        runner.failed.connect(lambda message: finish("failed", message))
        runner.canceled.connect(lambda: finish("canceled"))
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(lambda: finish("timeout"))
        timeout.start(30_000)

        runner.start(request)
        execute = getattr(event_loop, "exec", None) or event_loop.exec_
        execute()
        timeout.stop()

        self.assertEqual(terminal[0][0], "succeeded", terminal)
        self.assertTrue(Path(dem_path).is_file())
        self.assertTrue(Path(hillshade_path).is_file())
        self.assertFalse(runner.is_running)
        self.assertEqual(runner._resources, [])

        loaded_dem = QgsRasterLayer(dem_path, "loaded terrain", "gdal")
        self.assertTrue(loaded_dem.isValid())
        QgsProject.instance().addMapLayer(loaded_dem)
        dem_alias = str(Path(temporary_directory.name) / "terrain-dem-alias.tif")
        os.symlink(dem_path, dem_alias)
        from ai_vectorizer.core.dem_pipeline import loaded_project_paths

        self.assertEqual(loaded_project_paths([dem_alias]), [dem_alias])

    def test_dem_publish_rechecks_outputs_loaded_during_async_run(self):
        from ai_vectorizer.core import dem_pipeline

        temporary_directory = tempfile.TemporaryDirectory()
        self.__class__._temporary_directories.append(temporary_directory)
        dem_path = str(Path(temporary_directory.name) / "terrain-dem.tif")
        hillshade_path = str(
            Path(temporary_directory.name) / "terrain-hillshade.tif"
        )
        dem_work_path = str(Path(temporary_directory.name) / ".dem-work.tif")
        hillshade_work_path = str(
            Path(temporary_directory.name) / ".hillshade-work.tif"
        )
        Path(dem_work_path).write_bytes(b"staged dem")
        Path(hillshade_work_path).write_bytes(b"staged hillshade")

        runner = dem_pipeline.DemPipelineRunner()
        runner._request = mock.Mock(
            dem_path=dem_path,
            hillshade_path=hillshade_path,
            crs_authid="EPSG:3857",
        )
        runner._raw_dem_path = None
        runner._dem_work_path = dem_work_path
        runner._hillshade_work_path = hillshade_work_path
        runner._dem_dimensions = (10, 10)
        failures = []
        runner.failed.connect(failures.append)

        with mock.patch.object(
            dem_pipeline,
            "loaded_project_paths",
            return_value=[dem_path],
        ), mock.patch.object(
            runner,
            "_validate_raster",
            return_value=(10, 10),
        ), mock.patch.object(runner, "_publish_outputs") as publish:
            runner._on_hillshade_finished(True, {}, mock.Mock(errors=[]))

        publish.assert_not_called()
        self.assertEqual(len(failures), 1)
        self.assertIn("loaded in QGIS", failures[0])
        self.assertFalse(Path(dem_work_path).exists())
        self.assertFalse(Path(hillshade_work_path).exists())

    def test_timers_are_tool_owned_and_deactivated_emits_once(self):
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        emissions = []
        tool.deactivated.connect(lambda: emissions.append(True))

        self.assertIs(tool._edge_cache_timer.parent(), tool)
        self.assertIs(tool._proposal_timer.parent(), tool)
        tool.activate()
        tool.deactivate()
        self.assertEqual(emissions, [True])

    def test_deactivated_canvas_owned_tool_is_scheduled_for_deletion(self):
        from ai_vectorizer.ui.main_dialog import AIVectorizerDock

        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "contours", "memory")
        tool = self._tool(layer)
        self.assertIs(tool.parent(), self.canvas)
        bands = [
            tool.preview_band,
            tool.confirm_band,
            tool.start_marker,
            tool.close_indicator,
            tool.checkpoint_markers,
            tool.snap_marker,
        ]
        scene_items = self.canvas.scene().items()
        self.assertTrue(all(band in scene_items for band in bands))
        destroyed = []
        tool.destroyed.connect(lambda *_args: destroyed.append(True))

        class DockHarness:
            def __init__(self, active_tool):
                self.active_tool = active_tool
                self._sender = active_tool
                self.idle = False

            def sender(self):
                return self._sender

            def _set_idle_ui(self):
                self.idle = True

        dock = DockHarness(tool)
        tool.deactivated.connect(
            lambda: AIVectorizerDock.on_tool_deactivated(dock)
        )
        self.canvas.setMapTool(tool)
        self.canvas.unsetMapTool(tool)
        self.assertIsNone(dock.active_tool)
        self.assertTrue(dock.idle)
        scene_items = self.canvas.scene().items()
        self.assertTrue(all(band not in scene_items for band in bands))

        event_loop = QEventLoop()
        QTimer.singleShot(0, event_loop.quit)
        execute = getattr(event_loop, "exec", None) or event_loop.exec_
        execute()
        self.assertEqual(destroyed, [True])

    def test_custom_toolbar_is_removed_on_unload(self):
        class Iface:
            def __init__(self):
                self.window = QMainWindow()
                self.removed_actions = []

            def mainWindow(self):
                return self.window

            def addToolBar(self, name):
                toolbar = QToolBar(name, self.window)
                self.window.addToolBar(toolbar)
                return toolbar

            def addPluginToVectorMenu(self, _menu, _action):
                pass

            def removePluginVectorMenu(self, _menu, action):
                self.removed_actions.append(action)

            def removeToolBarIcon(self, _action):
                pass

        iface = Iface()
        plugin = self.AIVectorizer(iface)
        plugin.initGui()
        toolbar = plugin.toolbar
        action = plugin.actions[0]
        self.assertIn(action, toolbar.actions())

        plugin.unload()

        self.assertIsNone(plugin.toolbar)
        self.assertEqual(plugin.actions, [])
        self.assertNotIn(action, toolbar.actions())

    def test_vector_writer_and_standard_path_compatibility(self):
        from qgis.PyQt.QtCore import QStandardPaths
        from ai_vectorizer.ui.main_dialog import (
            _standard_location,
            _write_vector_layer,
            _writer_no_error,
        )

        location = _standard_location("AppDataLocation")
        self.assertTrue(QStandardPaths.writableLocation(location))
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "output", "memory")
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "output.shp")
            result = _write_vector_layer(layer, path, self.crs)

            self.assertEqual(result[0], _writer_no_error())
            self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
