"""Actual-QGIS terrain input validation before asynchronous Processing."""

import math
import os
from pathlib import Path
import tempfile
import unittest

try:
    from qgis.core import (
        QgsApplication, QgsFeature, QgsField, QgsGeometry, QgsPoint,
        QgsPointXY, QgsProject, QgsVectorLayer,
    )
    from qgis.PyQt.QtCore import QMetaType
    try:
        from qgis.PyQt.QtCore import QVariant
    except ImportError:
        QVariant = None
    HAS_QGIS = True
except ImportError as exc:
    if os.environ.get("ARCHAEOTRACE_REQUIRE_QGIS") == "1":
        raise RuntimeError("QGIS bindings are required for this job") from exc
    HAS_QGIS = False


@unittest.skipUnless(HAS_QGIS, "QGIS Python bindings are not installed")
class DemPipelineQgisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._owns_application = QgsApplication.instance() is None
        cls.application = QgsApplication.instance() or QgsApplication([], False)
        if cls._owns_application:
            cls.application.initQgis()
        from ai_vectorizer.core.dem_pipeline import (
            DemInputError, build_dem_request, inspect_elevation_layer,
        )
        cls.InputError = DemInputError
        cls.build_request = staticmethod(build_dem_request)
        cls.inspect = staticmethod(inspect_elevation_layer)

    @classmethod
    def tearDownClass(cls):
        QgsProject.instance().clear()
        if cls._owns_application:
            cls.application.exitQgis()

    def setUp(self):
        QgsProject.instance().clear()
        self.temp = tempfile.TemporaryDirectory()
        self.contours = self._layer("LineString", "contours", [
            (QgsGeometry.fromWkt("LINESTRING(0 0,100 0)"), 100.0),
            (QgsGeometry.fromWkt("LINESTRING(0 100,100 100)"), 120.0),
        ])

    def tearDown(self):
        QgsProject.instance().clear()
        self.temp.cleanup()

    def _layer(self, kind, name, rows):
        layer = QgsVectorLayer(kind + "?crs=EPSG:3857", name, "memory")
        self.assertTrue(layer.isValid())
        double_type = (
            QVariant.Double if QVariant is not None and hasattr(QVariant, "Double")
            else QMetaType.Type.Double
        )
        layer.dataProvider().addAttributes([QgsField("elevation", double_type)])
        layer.updateFields()
        self._add_rows(layer, rows)
        QgsProject.instance().addMapLayer(layer)
        return layer

    def _add_rows(self, layer, rows):
        features = []
        for geometry, elevation in rows:
            feature = QgsFeature(layer.fields())
            feature.setGeometry(geometry)
            feature.setAttributes([elevation])
            features.append(feature)
        self.assertTrue(layer.dataProvider().addFeatures(features)[0])
        layer.updateExtents()

    def _assert_rejected_without_changing_outputs(self, *, spot=None):
        dem = Path(self.temp.name) / "existing.tif"
        hillshade = Path(self.temp.name) / "existing_hillshade.tif"
        dem.write_bytes(b"previous terrain")
        hillshade.write_bytes(b"previous hillshade")
        with self.assertRaisesRegex(self.InputError, "XY coordinates must be finite"):
            self.build_request(
                self.contours, "elevation", 10.0, str(dem),
                spot_layer=spot, spot_field="elevation",
            )
        self.assertEqual(dem.read_bytes(), b"previous terrain")
        self.assertEqual(hillshade.read_bytes(), b"previous hillshade")
        self.assertEqual(set(Path(self.temp.name).iterdir()), {dem, hillshade})

    def test_nonfinite_contour_xy_rejects_the_entire_partial_line(self):
        for axis in (0, 1):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(axis=axis, value=value):
                    coordinate = [75.0, 50.0]
                    coordinate[axis] = value
                    geometry = QgsGeometry.fromPolylineXY([
                        QgsPointXY(50.0, 50.0), QgsPointXY(*coordinate),
                    ])
                    self._add_rows(self.contours, [(geometry, 140.0)])
                    stats = self.inspect(self.contours, "elevation")
                    self.assertEqual(stats.invalid_geometry_count, 1)
                    self.assertEqual(stats.vertex_count, 4)
                    self.assertEqual(stats.unique_values, (100.0, 120.0))
                    self._assert_rejected_without_changing_outputs()
                    invalid_ids = [
                        feature.id() for feature in self.contours.getFeatures()
                        if feature["elevation"] == 140.0
                    ]
                    self.assertTrue(self.contours.dataProvider().deleteFeatures(invalid_ids))
                    self.contours.updateExtents()

    def test_nonfinite_spot_xy_is_not_used_as_an_elevation_sample(self):
        for axis in (0, 1):
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(axis=axis, value=value):
                    coordinate = [50.0, 50.0]
                    coordinate[axis] = value
                    spot = self._layer("Point", "spots", [
                        (QgsGeometry.fromPointXY(QgsPointXY(*coordinate)), 140.0),
                    ])
                    stats = self.inspect(spot, "elevation")
                    self.assertEqual(stats.invalid_geometry_count, 1)
                    self.assertEqual(stats.vertex_count, 0)
                    self._assert_rejected_without_changing_outputs(spot=spot)
                    QgsProject.instance().removeMapLayer(spot.id())

    def test_attribute_elevations_do_not_require_geometry_z(self):
        contours = self._layer("LineStringZ", "z contours", [
            (QgsGeometry.fromPolyline([
                QgsPoint(0.0, 0.0, math.nan), QgsPoint(100.0, 0.0, math.nan),
            ]), 100.0),
            (QgsGeometry.fromPolyline([
                QgsPoint(0.0, 100.0, math.nan), QgsPoint(100.0, 100.0, math.nan),
            ]), 120.0),
        ])
        request = self.build_request(
            contours, "elevation", 10.0, str(Path(self.temp.name) / "finite-xy.tif"),
        )
        self.assertEqual(request.contour_stats.invalid_geometry_count, 0)
        self.assertEqual(request.contour_stats.vertex_count, 4)
        self.assertEqual(request.contour_stats.unique_values, (100.0, 120.0))


if __name__ == "__main__":
    unittest.main()
