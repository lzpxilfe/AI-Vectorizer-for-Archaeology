"""Actual QGIS block/alpha-provider regressions; no live WMS service required."""
from pathlib import Path
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    from qgis.core import Qgis, QgsApplication, QgsRasterBlock, QgsRasterLayer
    from qgis.PyQt.QtGui import QColor, QImage, QPainter
    from osgeo import gdal, osr
    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError("QGIS bindings are required for raster boundary tests") from exc
    HAS_QGIS = False


def _qgis_type(name):
    legacy = getattr(Qgis, name, None)
    return legacy if legacy is not None else getattr(Qgis.DataType, name)


def _image_format(name):
    legacy = getattr(QImage, name, None)
    return legacy if legacy is not None else getattr(QImage.Format, name)


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class RasterBoundaryQgisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owns_app = QgsApplication.instance() is None
        cls.app = QgsApplication.instance() or QgsApplication([], False)
        if cls.owns_app:
            cls.app.initQgis()

    @classmethod
    def tearDownClass(cls):
        if cls.owns_app:
            cls.app.exitQgis()

    def test_actual_argb_block_matches_qt_white_background_composition(self):
        import numpy as np
        from ai_vectorizer.core.raster_utils import raster_block_to_rgb_uint8

        image = QImage(4, 1, _image_format("Format_ARGB32"))
        for x, rgba in enumerate(((0, 0, 0, 0), (0, 128, 32, 0), (10, 20, 200, 128), (13, 27, 204, 255))):
            image.setPixelColor(x, 0, QColor(*rgba))
        for name, qt_format in (("ARGB32", "Format_ARGB32"),
                                ("ARGB32_Premultiplied", "Format_ARGB32_Premultiplied")):
            with self.subTest(data_type=name):
                source = image.convertToFormat(_image_format(qt_format))
                block = QgsRasterBlock(_qgis_type(name), 4, 1)
                self.assertTrue(block.setImage(source))
                expected = QImage(4, 1, _image_format("Format_RGB32"))
                expected.fill(QColor(255, 255, 255))
                painter = QPainter(expected)
                painter.drawImage(0, 0, source)
                painter.end()
                reference = np.array([[[expected.pixelColor(x, 0).red(), expected.pixelColor(x, 0).green(),
                                        expected.pixelColor(x, 0).blue()] for x in range(4)]], dtype=np.uint8)
                np.testing.assert_array_equal(raster_block_to_rgb_uint8(block, 4, 1), reference)

    def test_actual_qgis_nodata_bitmap_is_paper_white(self):
        import numpy as np
        from ai_vectorizer.core.raster_utils import raster_block_to_uint8

        block = QgsRasterBlock(_qgis_type("Byte"), 3, 1)
        block.setData(np.array([0, 10, 20], dtype=np.uint8).tobytes())
        self.assertTrue(block.setIsNoData(0))
        np.testing.assert_array_equal(raster_block_to_uint8(block, 3, 1), [[255, 10, 20]])

    def _dataset(self, path, arrays, dtype, interpretations=None, nodata=None):
        dataset = gdal.GetDriverByName("GTiff").Create(str(path), arrays[0].shape[1], arrays[0].shape[0], len(arrays), dtype)
        dataset.SetGeoTransform([0, 1, 0, arrays[0].shape[0], 0, -1])
        crs = osr.SpatialReference()
        crs.ImportFromEPSG(3857)
        dataset.SetProjection(crs.ExportToWkt())
        for i, values in enumerate(arrays, 1):
            band = dataset.GetRasterBand(i)
            band.WriteArray(values)
            if interpretations:
                band.SetColorInterpretation(interpretations[i - 1])
            if nodata is not None:
                band.SetNoDataValue(nodata)
        dataset.FlushCache()
        dataset = None

    def test_real_rgba_provider_uses_its_declared_alpha_band(self):
        import numpy as np
        from ai_vectorizer.core.raster_utils import read_raster_bands_with_native

        with tempfile.TemporaryDirectory(prefix="archaeotrace-rgba-") as folder:
            path = Path(folder) / "rgba.tif"
            self._dataset(path, [np.array([row], dtype=np.uint8) for row in
                                ([0, 10, 13], [0, 20, 27], [0, 200, 204], [0, 128, 255])], gdal.GDT_Byte,
                          [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand, gdal.GCI_AlphaBand])
            layer = QgsRasterLayer(str(path), "synthetic RGBA", "gdal")
            self.assertTrue(layer.isValid())
            display, native, stable = read_raster_bands_with_native(layer.dataProvider(), layer.extent(), 3, 1)
            np.testing.assert_array_equal(np.stack(display, axis=-1), [[[255, 255, 255], [132, 137, 227], [13, 27, 204]]])
            self.assertEqual(native, [])
            self.assertFalse(stable)
            layer = None

    def test_real_float_provider_missing_values_remain_white(self):
        import numpy as np
        from ai_vectorizer.core.raster_utils import read_raster_bands_with_native

        with tempfile.TemporaryDirectory(prefix="archaeotrace-nodata-") as folder:
            path = Path(folder) / "float.tif"
            self._dataset(path, [np.array([[0., 1., np.nan, -9999.]], dtype=np.float32)], gdal.GDT_Float32, nodata=-9999)
            layer = QgsRasterLayer(str(path), "synthetic NoData", "gdal")
            self.assertTrue(layer.isValid())
            display, native, stable = read_raster_bands_with_native(layer.dataProvider(), layer.extent(), 4, 1)
            np.testing.assert_array_equal(display[0], [[0, 255, 255, 255]])
            self.assertTrue(np.isnan(native[0][0, 2:]).all())
            self.assertFalse(stable)
            layer = None


    def _rendered_tool(self, source_size=0, provider_type="wms", immediate=True):
        import numpy as np
        from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle, QgsVectorLayer
        from qgis.gui import QgsMapCanvas
        from ai_vectorizer.tools.smart_trace_tool import SmartTraceTool

        class Provider:
            def __init__(self):
                self.calls = []

            def bandCount(self):
                return 1

            def dataType(self, _band):
                return _qgis_type("ARGB32")

            def dataSourceUri(self):
                return "synthetic-rendered-raster"

            def block(self, band, extent, width, height):
                self.calls.append((band, width, height))
                values = np.zeros((height, width), dtype=np.uint32)
                values[max(0, height // 2 - 1):height // 2 + 2, 4:-4] = 0xFF232323
                block = QgsRasterBlock(_qgis_type("ARGB32"), width, height)
                block.setData(values.tobytes())
                return block

        class Dispatcher:
            def __init__(self):
                self.tasks = []

            def addTask(self, task):
                self.tasks.append(task)
                if immediate:
                    task.finished(bool(task.run()))
                return True

        canvas = QgsMapCanvas()
        canvas.resize(256, 256)
        canvas.show()
        crs = QgsCoordinateReferenceSystem("EPSG:3857")
        canvas.setDestinationCrs(crs)
        canvas.setExtent(QgsRectangle(0, 0, 256, 256))
        self.app.processEvents()
        provider = Provider()
        raster_extent = QgsRectangle(canvas.extent())
        raster = SimpleNamespace(
            crs=lambda: crs, extent=lambda: QgsRectangle(raster_extent),
            width=lambda: source_size, height=lambda: source_size,
            providerType=lambda: provider_type, dataProvider=lambda: provider,
            isValid=lambda: True, id=lambda: "synthetic-rendered-layer",
        )
        layer = QgsVectorLayer("LineString?crs=EPSG:3857", "rendered raster test", "memory")
        tool = SmartTraceTool(canvas, raster, layer, freehand=True, language="en")
        from ai_vectorizer.core.edge_detector import EdgeDetector
        tool.freehand = False
        tool.edge_method = EdgeDetector.METHOD_INK
        tool.edge_detector = EdgeDetector(method=EdgeDetector.METHOD_INK)
        tool.edge_weight = 1.0
        tool._is_active = True
        tool._push_message = mock.Mock()
        return canvas, tool, provider, Dispatcher(), layer

    def _close_rendered_tool(self, canvas, tool):
        tool.dispose()
        tool.deleteLater()
        canvas.deleteLater()
        self.app.processEvents()

    def test_zero_size_rendered_provider_builds_cache_and_livewire_path(self):
        import numpy as np
        from qgis.core import QgsPointXY
        from ai_vectorizer.core.livewire import is_livewire_available

        self.assertTrue(is_livewire_available(), "Actual QGIS regression needs SciPy")
        for source_size, provider_type in ((0, "wms"), (256, "wms"), (0, "synthetic")):
            with self.subTest(source_size=source_size, provider_type=provider_type):
                canvas, tool, provider, dispatcher, layer = self._rendered_tool(source_size, provider_type)
                try:
                    with mock.patch.object(QgsApplication, "taskManager", return_value=dispatcher):
                        tool.update_edge_cache()
                        self.assertIsNotNone(tool.cached_edges)
                        self.assertGreater(int(np.count_nonzero(tool.cached_edges)), 0)
                        height, width = tool.cached_edges.shape
                        self.assertTrue(10 <= min(width, height) <= max(width, height) <= 1000)
                        self.assertEqual(provider.calls, [(1, width, height)])
                        self.assertEqual(tool.cached_rgb_image[0, 0].tolist(), [255, 255, 255])
                        self.assertEqual(tool.cached_rgb_image[height // 2, width // 2].tolist(), [35, 35, 35])
                        self.assertIsNone(tool.cached_ink_evidence)
                        self.assertFalse(tool._recovery_cache_compatible)
                        self.assertFalse(tool._learned_cache_ready())
                        self.assertEqual(tool.cache_tile_origin, (0, 0))
                        start = (max(5, width // 2 - 30), height // 2)
                        target = (min(width - 5, start[0] + 60), height // 2)
                        anchor = tool.pixel_to_map(*start)
                        tool.is_tracing = True
                        tool.start_point = QgsPointXY(anchor)
                        tool.path_points = [QgsPointXY(anchor)]
                        tool.checkpoints = [0]
                        self.assertTrue(tool._request_livewire_tree(force=True))
                        self.assertIsNotNone(tool._livewire_tree)
                        route = tool._livewire_tree.trace(target)
                        self.assertIsNotNone(route)
                        self.assertGreater(len(route), 2)
                        self.assertTrue(np.isfinite(route).all())
                        self.assertLessEqual(float(np.abs(np.asarray(route)[:, 1] - height // 2).max()), 5.)
                        self.assertEqual(layer.featureCount(), 0)
                finally:
                    self._close_rendered_tool(canvas, tool)

    def test_rendered_read_is_bounded_by_visible_fraction_and_resize_identity(self):
        from qgis.core import QgsRectangle
        from ai_vectorizer.core.raster_utils import compute_resampled_dimensions

        canvas, tool, _provider, _dispatcher, _layer = self._rendered_tool()
        try:
            full = tool._canvas_extent_in_raster_crs()
            half = QgsRectangle(full.xMinimum(), full.yMinimum(), full.center().x(), full.yMaximum())
            size = canvas.mapSettings().outputSize()
            expected = compute_resampled_dimensions(full.width(), full.height(), size.width(), size.height(),
                                                    half.width(), half.height(), 1000)
            self.assertEqual(tool._visible_raster_read_size(half, full, full), expected)
            read, origin = tool._ink_evidence_extent_and_origin(half, full)
            self.assertEqual(read, half)
            self.assertEqual(origin, (0, 0))
            _size, enabled, reason = tool._ink_evidence_sampling_plan(read, full)
            self.assertFalse(enabled)
            self.assertIn("Rendered", reason)
            identity = tool._current_ink_cache_identity()
            canvas.resize(512, 384)
            self.app.processEvents()
            self.assertNotEqual(identity, tool._current_ink_cache_identity())
        finally:
            self._close_rendered_tool(canvas, tool)

    def test_local_native_sampling_keeps_previous_source_size_contract(self):
        from qgis.core import QgsRectangle
        from ai_vectorizer.core.raster_utils import compute_resampled_dimensions

        canvas, tool, _provider, _dispatcher, _layer = self._rendered_tool(256, "gdal")
        try:
            full = tool.raster_layer.extent()
            read = QgsRectangle(full.xMinimum(), full.yMinimum(), full.center().x(), full.yMaximum())
            expected = compute_resampled_dimensions(full.width(), full.height(), 256, 256,
                                                    read.width(), read.height(), 1000)
            self.assertTrue(tool._raster_has_native_grid())
            self.assertEqual(tool._visible_raster_read_size(read, full), expected)
            self.assertEqual(tool._ink_evidence_sampling_plan(read, full), (expected, True, ""))
        finally:
            self._close_rendered_tool(canvas, tool)


    def test_concrete_qgis_sentinel_mask_matches_native_api(self):
        import numpy as np
        from ai_vectorizer.core.raster_utils import _raster_nodata_mask

        values = np.array([[0., 1e-16, 1e-14, np.nan, 1.]], dtype=np.float64)
        block = QgsRasterBlock(_qgis_type("Float64"), 5, 1)
        block.setData(values.tobytes())
        block.setIsNoData(4)
        block.setNoDataValue(0.)
        expected = np.array([[block.isNoData(i) for i in range(5)]], dtype=bool)
        np.testing.assert_array_equal(_raster_nodata_mask(block, values), expected)

    def test_rendered_edge_preview_reads_view_pixels_and_preserves_white_paper(self):
        import numpy as np
        from qgis.core import QgsProject
        from ai_vectorizer.core.edge_detector import EdgeDetector
        from ai_vectorizer.config import MODEL_IDX_INK, PREVIEW_EDGE_MAX_DIMENSION
        from ai_vectorizer.ui import main_dialog

        for source_size, provider_type in ((0, "wms"), (256, "gdal")):
            with self.subTest(provider_type=provider_type):
                canvas, tool, provider, _dispatcher, _layer = self._rendered_tool(source_size, provider_type)
                project = QgsProject.instance()
                store = main_dialog._TemporaryPreviewStore(project)
                images = []
                original = EdgeDetector.detect_edges

                def inspect(detector, image):
                    images.append(np.array(image, copy=True))
                    return original(detector, image)

                dock = SimpleNamespace(
                    layer_combo=SimpleNamespace(currentLayer=lambda: tool.raster_layer),
                    model_combo=SimpleNamespace(currentIndex=lambda: MODEL_IDX_INK),
                    _is_sam_model=lambda _index: False,
                    _canvas_extent_in_layer_crs=lambda _raster: tool._canvas_extent_in_raster_crs(),
                    _tr=lambda ko, en: en, _preview_store=store,
                    iface=SimpleNamespace(mapCanvas=lambda: canvas),
                )
                try:
                    with mock.patch.object(EdgeDetector, "detect_edges", autospec=True, side_effect=inspect), \
                            mock.patch.object(main_dialog.QMessageBox, "information") as information, \
                            mock.patch.object(main_dialog.QMessageBox, "critical") as critical, \
                            mock.patch.object(main_dialog.QMessageBox, "warning") as warning:
                        main_dialog.AIVectorizerDock.preview_edges(dock)
                    critical.assert_not_called()
                    warning.assert_not_called()
                    information.assert_called_once()
                    self.assertEqual(len(images), 1)
                    image = images[0]
                    self.assertTrue(10 <= min(image.shape[:2]) <= max(image.shape[:2]) <= PREVIEW_EDGE_MAX_DIMENSION)
                    self.assertEqual(image[0, 0].tolist(), [255, 255, 255])
                    self.assertEqual(image[image.shape[0] // 2, image.shape[1] // 2].tolist(), [35, 35, 35])
                    self.assertEqual(provider.calls, [(1, image.shape[1], image.shape[0])])
                    self.assertEqual(len(store._directories), 1)
                    preview = project.mapLayer(next(iter(store._directories)))
                    self.assertEqual((preview.height(), preview.width()), image.shape[:2])
                    dataset = gdal.Open(preview.source())
                    edges = dataset.ReadAsArray()
                    dataset = None
                    self.assertGreater(int(np.count_nonzero(edges)), 0)
                    self.assertEqual(int(np.count_nonzero(edges[0])), 0)
                    self.assertFalse(tool._learned_cache_ready())
                    self.assertFalse(tool._recovery_cache_compatible)
                finally:
                    store.shutdown()
                    self._close_rendered_tool(canvas, tool)


if __name__ == "__main__":
    unittest.main()
