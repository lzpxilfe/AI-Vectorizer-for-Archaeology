"""Annotation transactions with a native QGIS suite using only unittest."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tempfile
import unittest

from ai_vectorizer.core.annotation_project import (
    AnnotationProjectError, AnnotationSession, map_to_pixel, pixel_to_map,
    create_annotation_project, export_annotation_project,
    _dataset_crop_metadata,
)


class Point:
    def __init__(self, point): self.point = point
    def x(self): return self.point[0]
    def y(self): return self.point[1]


class Geometry:
    def __init__(self, points): self.points = [Point(p) for p in points]
    def isEmpty(self): return False
    def isGeosValid(self): return True
    def isMultipart(self): return False
    def asPolyline(self): return self.points
    def asPolygon(self): return [self.points]


class Feature:
    def __init__(self, points): self.geo = Geometry(points)
    def id(self): return 1
    def geometry(self): return self.geo


class Layer:
    def __init__(self, features=(), modified=False, selected=False):
        self.features, self.modified, self.selected = features, modified, selected
    def isModified(self): return self.modified
    def isValid(self): return True
    def getFeatures(self): return iter(self.features)
    def selectedFeatureIds(self): return [1] if self.selected else []


def fixture_session(root, modified=False, selected=False):
    image = root / "crop.png"
    image.write_bytes(b"hash-bound crop fixture")
    transform = (100, 2, .2, 200, -.1, -3)
    line = [pixel_to_map(p, transform) for p in ((2, 8), (20, 8))]
    reviewed = [pixel_to_map(p, transform) for p in
                ((-.5, -.5), (511.5, -.5), (511.5, 511.5), (-.5, 511.5), (-.5, -.5))]
    layers = {"contours": Layer([Feature(line)], modified, selected),
              "other_ink": Layer(), "other_ink_regions": Layer(),
              "reviewed": Layer([Feature(reviewed)]), "ignore": Layer()}
    return AnnotationSession("fixture", image, hashlib.sha256(image.read_bytes()).hexdigest(),
        root, root / "annotations.gpkg", transform, "", 512, 512, None, layers)


class AnnotationProjectTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
    def tearDown(self):
        self.temporary.cleanup()

    def test_pixel_centres_and_rotated_affine(self):
        transform = (100, 2, .2, 200, -.1, -3)
        for actual, expected in zip(pixel_to_map((0, 0), transform), (101.1, 198.45)):
            self.assertAlmostEqual(actual, expected)
        for point in ((0, 0), (511, 511), (17.25, 200.75), (-.5, -.5)):
            for actual, expected in zip(map_to_pixel(pixel_to_map(point, transform), transform), point):
                self.assertAlmostEqual(actual, expected)
        with self.assertRaises(AnnotationProjectError): map_to_pixel((0, 0), (0, 1, 2, 0, 2, 4))

    def test_drafts_are_never_approved(self):
        value = export_annotation_project(fixture_session(self.root, True), self.root / "draft.json", "ink-draft-v2")
        self.assertEqual(value["annotation_status"], "draft")
        self.assertEqual(value["review_level"], "unreviewed")
        self.assertIsNone(value["evaluation"])
        self.assertFalse(value["publication_ranking_eligible"])

    def test_single_user_review_requires_saved_edits_and_confirmation(self):
        active = fixture_session(self.root, True)
        with self.assertRaisesRegex(AnnotationProjectError, "save"):
            export_annotation_project(active, self.root / "approved.json", "user", "user", True, True)
        active.layers["contours"].modified = False
        with self.assertRaisesRegex(AnnotationProjectError, "confirm"):
            export_annotation_project(active, self.root / "approved.json", "user", "user", True, False)
        value = export_annotation_project(active, self.root / "approved.json", "user", "user", True, True)
        self.assertEqual(value["review_level"], "single_reviewer_pilot")
        self.assertEqual(value["reviewer_id"], value["annotator_id"])

    def test_target_selection_and_revision_preservation(self):
        active = fixture_session(self.root, False, True)
        first = export_annotation_project(active, self.root / "review1.json", "user", "user", True, True)
        self.assertEqual(first["evaluation"]["target_contour_id"], "contours-1-0")
        second = export_annotation_project(active, self.root / "review2.json", "user", "user", True, True)
        self.assertEqual(second["parent_revision"], first["revision"])
        with self.assertRaisesRegex(AnnotationProjectError, "overwrite"):
            export_annotation_project(active, self.root / "review1.json", "user", "user", True, True)

    def test_changed_crop_invalidates_session(self):
        active = fixture_session(self.root)
        active.image_path.write_bytes(b"changed crop")
        with self.assertRaisesRegex(AnnotationProjectError, "changed"):
            export_annotation_project(active, self.root / "bad.json", "user")

    def test_latest_manifest_resume_pointer_is_hash_bound(self):
        image = self.root / "images" / "fixture.png"
        image.parent.mkdir()
        image.write_bytes(b"immutable crop")
        sample = {"id": "fixture", "image": "images/fixture.png",
                  "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                  "geotransform": [100, 2, 0, 200, 0, -2], "annotation": {}}
        target = self.root / "manifest-reviewed.json"
        target.write_text(json.dumps({"samples": [sample]}))
        pointer = self.root / "dataset_latest.json"
        pointer.write_text(json.dumps({"manifest": target.name,
            "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
        observed, annotation = _dataset_crop_metadata(image, "fixture")
        self.assertEqual(observed["geotransform"], sample["geotransform"])
        self.assertIsNone(annotation)
        target.write_text(target.read_text() + " ")
        with self.assertRaisesRegex(AnnotationProjectError, "hash"):
            _dataset_crop_metadata(image, "fixture")


@unittest.skipUnless(importlib.util.find_spec("qgis") is not None, "native QGIS unavailable")
class NativeAnnotationProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qgis.core import QgsApplication
        cls.application = QgsApplication.instance()
        cls.owned = cls.application is None
        if cls.owned:
            cls.application = QgsApplication([], False)
        # Another native suite may have called exitQgis while the QApplication
        # wrapper still exists. Reopen providers and GDAL for this suite.
        cls.application.initQgis()
        from osgeo import gdal
        gdal.AllRegister()

    @classmethod
    def tearDownClass(cls):
        if cls.owned:
            from qgis.core import QgsProject
            QgsProject.instance().clear()
            import gc
            gc.collect()
            cls.application.exitQgis()

    def test_native_save_export_and_resume(self):
        from qgis.core import QgsProject, QgsFeature, QgsGeometry, QgsPointXY
        from osgeo import gdal
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            image = root / "images" / "native.png"
            image.parent.mkdir()
            memory = gdal.GetDriverByName("MEM").Create("", 512, 512, 3, gdal.GDT_Byte)
            for index, value in enumerate((250, 245, 235), 1): memory.GetRasterBand(index).Fill(value)
            created = gdal.GetDriverByName("PNG").CreateCopy(str(image), memory)
            created = None
            memory = None
            transform = (100, 2, .2, 200, -.1, -3)
            active = create_annotation_project(image, root / "sessions", "native", transform,
                draft_contours=[[[2, 8], [20, 8]]])
            QgsProject.instance().addMapLayers([active.raster, *active.layers.values()])
            resumed = None
            try:
                self.assertEqual(set(active.layers), {"contours", "other_ink", "other_ink_regions", "reviewed", "ignore"})
                self.assertTrue(active.layers["contours"].isModified())
                with self.assertRaises(AnnotationProjectError):
                    export_annotation_project(active, root / "blocked.json", "user", "user", True, True)
                layer = active.layers["reviewed"]
                feature = QgsFeature(layer.fields())
                points = [QgsPointXY(*pixel_to_map(p, transform)) for p in
                    ((-.5, -.5), (511.5, -.5), (511.5, 511.5), (-.5, 511.5), (-.5, -.5))]
                feature.setGeometry(QgsGeometry.fromPolygonXY([points]))
                feature["label_state"], feature["sample_id"] = "manual", "native"
                self.assertTrue(layer.addFeature(feature))
                # Explicit test user Save Layer Edits; production cannot commit.
                for layer in active.layers.values(): self.assertTrue(layer.commitChanges())
                active.layers["contours"].selectAll()
                target = root / "reviewed.json"
                label = export_annotation_project(active, target, "user", "user", True, True)
                for actual, expected in zip(label["contours"][0]["points"][0], (2, 8)):
                    self.assertAlmostEqual(actual, expected)
                self.assertIsNotNone(label["evaluation"])
                manifest = {"samples": [{"id": "native", "image": "images/native.png",
                    "image_sha256": active.image_sha256, "geotransform": list(transform),
                    "source_tile_origin_xy": [64, 96], "crs_wkt": "",
                    "annotation": {"path": "reviewed.json", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}}]}
                (root / "manifest.json").write_text(json.dumps(manifest))
                resumed = create_annotation_project(image, root / "sessions", "native")
                QgsProject.instance().addMapLayers([resumed.raster, *resumed.layers.values()])
                self.assertEqual(resumed.geotransform, transform)
                self.assertEqual(resumed.parent_revision, label["revision"])
                self.assertEqual(resumed.layers["contours"].featureCount(), 1)
                self.assertEqual(resumed.layers["reviewed"].featureCount(), 1)
                self.assertTrue(resumed.layers["contours"].isModified())
            finally:
                for session in (active, resumed):
                    if session is not None:
                        for item in session.layers.values():
                            if item.isEditable(): item.rollBack()
                QgsProject.instance().clear()
                for session in (active, resumed):
                    if session is not None:
                        session.layers.clear()
                        session.raster = None
                layer = None
                item = None
                feature = None
                points = None


if __name__ == "__main__":
    unittest.main()
