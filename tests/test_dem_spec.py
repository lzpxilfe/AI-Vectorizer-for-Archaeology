"""Tests for QGIS-independent DEM specifications."""

from pathlib import Path
import tempfile
import unittest

from ai_vectorizer.core.dem_spec import (
    DemSpecificationError,
    default_hillshade_path,
    estimate_grid,
    extent_parameter,
    interpolation_data,
    interpolation_entry,
    is_tiff_file,
    normalize_tiff_path,
    paths_refer_to_same_file,
    publish_output_files,
    suggest_pixel_size,
)


class DemSpecificationTests(unittest.TestCase):
    def test_estimate_grid_rounds_up(self):
        estimate = estimate_grid(100.1, 50.1, 10)

        self.assertEqual(estimate.columns, 11)
        self.assertEqual(estimate.rows, 6)
        self.assertEqual(estimate.cells, 66)

    def test_estimate_grid_rejects_unsafe_cell_count(self):
        with self.assertRaisesRegex(DemSpecificationError, "too large"):
            estimate_grid(1000, 1000, 1, max_cells=999_999)

    def test_suggest_pixel_size_uses_one_two_five_scale(self):
        self.assertEqual(suggest_pixel_size(10_000, 5_000, target_long_side=1000), 10.0)
        self.assertEqual(suggest_pixel_size(13_000, 5_000, target_long_side=1000), 20.0)

    def test_tiff_paths_are_normalized(self):
        self.assertEqual(normalize_tiff_path("/tmp/terrain"), "/tmp/terrain.tif")
        self.assertEqual(normalize_tiff_path("/tmp/terrain.TIFF"), "/tmp/terrain.TIFF")
        with self.assertRaises(DemSpecificationError):
            normalize_tiff_path("/tmp/terrain.png")

    def test_default_hillshade_path_is_paired_with_dem(self):
        result = default_hillshade_path("/tmp/site_dem.tif")

        self.assertEqual(Path(result).name, "site_dem_hillshade.tif")

    def test_interpolation_data_matches_qgis_encoding(self):
        contour = interpolation_entry("contours", 0, 2, 1)
        spots = interpolation_entry("spots", 0, 1, 0)

        self.assertEqual(
            interpolation_data([contour, spots]),
            "contours::~::0::~::2::~::1::|::spots::~::0::~::1::~::0",
        )

    def test_extent_parameter_preserves_qgis_coordinate_order(self):
        result = extent_parameter(1, 11, 2, 12, "EPSG:5186")

        self.assertEqual(result, "1,11,2,12 [EPSG:5186]")

    def test_tiff_signature_rejects_ascii_grid_with_tif_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            fake_tiff = Path(folder) / "terrain.tif"
            fake_tiff.write_bytes(b"NCOLS 10\nNROWS 10\n")
            real_tiff = Path(folder) / "real.tif"
            real_tiff.write_bytes(b"II*\x00payload")

            self.assertFalse(is_tiff_file(str(fake_tiff)))
            self.assertTrue(is_tiff_file(str(real_tiff)))

    def test_output_identity_conservatively_case_folds_new_names(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "Terrain.tif"
            second = Path(folder) / "terrain.tif"

            self.assertTrue(paths_refer_to_same_file(str(first), str(second)))

    def test_output_identity_detects_existing_file_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Path(folder) / "terrain.tif"
            alias = Path(folder) / "alias.tif"
            original.write_bytes(b"terrain")
            alias.symlink_to(original)

            self.assertTrue(paths_refer_to_same_file(str(original), str(alias)))

    def test_publish_output_files_replaces_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work = folder_path / "work.tif"
            final = folder_path / "final.tif"
            old_sidecar = folder_path / "final.tif.aux.xml"
            work.write_bytes(b"new")
            final.write_bytes(b"old")
            old_sidecar.write_bytes(b"old statistics")

            publish_output_files(((str(work), str(final)),))

            self.assertEqual(final.read_bytes(), b"new")
            self.assertFalse(work.exists())
            self.assertFalse(old_sidecar.exists())

    def test_publish_output_files_rolls_back_pair_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work_dem = folder_path / "work_dem.tif"
            missing_hillshade = folder_path / "missing_hillshade.tif"
            final_dem = folder_path / "dem.tif"
            final_hillshade = folder_path / "hillshade.tif"
            work_dem.write_bytes(b"new dem")
            final_dem.write_bytes(b"old dem")
            final_hillshade.write_bytes(b"old hillshade")

            with self.assertRaisesRegex(DemSpecificationError, "Could not publish"):
                publish_output_files(
                    (
                        (str(work_dem), str(final_dem)),
                        (str(missing_hillshade), str(final_hillshade)),
                    )
                )

            self.assertEqual(final_dem.read_bytes(), b"old dem")
            self.assertEqual(final_hillshade.read_bytes(), b"old hillshade")

    def test_publish_rejects_case_alias_destinations_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work_dem = folder_path / "work_dem.tif"
            work_hillshade = folder_path / "work_hillshade.tif"
            work_dem.write_bytes(b"dem")
            work_hillshade.write_bytes(b"hillshade")

            with self.assertRaisesRegex(DemSpecificationError, "distinct"):
                publish_output_files(
                    (
                        (str(work_dem), str(folder_path / "Terrain.tif")),
                        (str(work_hillshade), str(folder_path / "terrain.tif")),
                    )
                )

            self.assertTrue(work_dem.exists())
            self.assertTrue(work_hillshade.exists())


if __name__ == "__main__":
    unittest.main()
