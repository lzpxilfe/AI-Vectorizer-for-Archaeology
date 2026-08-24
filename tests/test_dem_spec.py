"""Tests for QGIS-independent DEM specifications."""

from pathlib import Path
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import ai_vectorizer.core.dem_spec as dem_spec_module
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

    def test_estimate_grid_rejects_non_integer_limit_and_overflow(self):
        for invalid_limit in (True, 1.5, math.nan, math.inf):
            with self.subTest(limit=invalid_limit):
                with self.assertRaises(DemSpecificationError):
                    estimate_grid(10, 10, 1, max_cells=invalid_limit)

        with self.assertRaisesRegex(DemSpecificationError, "too large"):
            estimate_grid(1e308, 1e308, 1e-308)

    def test_suggest_pixel_size_uses_one_two_five_scale(self):
        self.assertEqual(suggest_pixel_size(10_000, 5_000, target_long_side=1000), 10.0)
        self.assertEqual(suggest_pixel_size(13_000, 5_000, target_long_side=1000), 20.0)

    def test_suggest_pixel_size_rejects_malformed_target(self):
        for invalid_target in (True, 1.5, math.nan, math.inf):
            with self.subTest(target=invalid_target):
                with self.assertRaises(DemSpecificationError):
                    suggest_pixel_size(100, 50, target_long_side=invalid_target)

    def test_suggest_pixel_size_keeps_extreme_finite_result_representable(self):
        suggestion = suggest_pixel_size(1.7e308, 1e308, target_long_side=1)

        self.assertTrue(math.isfinite(suggestion))
        self.assertGreater(suggestion, 0)

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

    def test_extent_parameter_compares_converted_numeric_values(self):
        result = extent_parameter("2", "11", "3", "12", "EPSG:5186")

        self.assertEqual(result, "2,11,3,12 [EPSG:5186]")
        with self.assertRaises(DemSpecificationError):
            extent_parameter("invalid", 11, 3, 12, "EPSG:5186")

    def test_tiff_signature_rejects_ascii_grid_with_tif_extension(self):
        with tempfile.TemporaryDirectory() as folder:
            fake_tiff = Path(folder) / "terrain.tif"
            fake_tiff.write_bytes(b"NCOLS 10\nNROWS 10\n")
            real_tiff = Path(folder) / "real.tif"
            real_tiff.write_bytes(b"II*\x00payload")

            self.assertFalse(is_tiff_file(str(fake_tiff)))
            self.assertTrue(is_tiff_file(str(real_tiff)))

    def test_tiff_signature_rejects_symlinked_input(self):
        with tempfile.TemporaryDirectory() as folder:
            real_tiff = Path(folder) / "real.tif"
            linked_tiff = Path(folder) / "linked.tif"
            real_tiff.write_bytes(b"II*\x00payload")
            try:
                os.symlink(real_tiff, linked_tiff)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            self.assertFalse(is_tiff_file(str(linked_tiff)))

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

    def test_publish_output_files_removes_stale_mask_and_moves_new_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work = folder_path / "work.tif"
            final = folder_path / "final.tif"
            work_mask = folder_path / "work.tif.msk"
            final_mask = folder_path / "final.tif.msk"
            work.write_bytes(b"new")
            work_mask.write_bytes(b"new mask")
            final.write_bytes(b"old")
            final_mask.write_bytes(b"stale mask")

            publish_output_files(((str(work), str(final)),))

            self.assertEqual(final.read_bytes(), b"new")
            self.assertEqual(final_mask.read_bytes(), b"new mask")
            self.assertFalse(work_mask.exists())

    def test_publish_output_files_drops_stale_mask_without_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work = folder_path / "work.tif"
            final = folder_path / "final.tif"
            final_mask = folder_path / "final.tif.msk"
            work.write_bytes(b"new")
            final.write_bytes(b"old")
            final_mask.write_bytes(b"stale mask")

            publish_output_files(((str(work), str(final)),))

            self.assertEqual(final.read_bytes(), b"new")
            self.assertFalse(final_mask.exists())

    def test_publish_output_files_rolls_back_pair_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work_dem = folder_path / "work_dem.tif"
            missing_hillshade = folder_path / "missing_hillshade.tif"
            final_dem = folder_path / "dem.tif"
            final_hillshade = folder_path / "hillshade.tif"
            work_dem.write_bytes(b"new dem")
            work_dem_mask = folder_path / "work_dem.tif.msk"
            work_dem_mask.write_bytes(b"new mask")
            final_dem.write_bytes(b"old dem")
            final_dem_mask = folder_path / "dem.tif.msk"
            final_dem_mask.write_bytes(b"old mask")
            final_hillshade.write_bytes(b"old hillshade")

            with self.assertRaisesRegex(DemSpecificationError, "Could not publish"):
                publish_output_files(
                    (
                        (str(work_dem), str(final_dem)),
                        (str(missing_hillshade), str(final_hillshade)),
                    )
                )

            self.assertEqual(final_dem.read_bytes(), b"old dem")
            self.assertEqual(final_dem_mask.read_bytes(), b"old mask")
            self.assertEqual(final_hillshade.read_bytes(), b"old hillshade")

    def test_publish_rolls_back_existing_pair_after_second_replace_error(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work_dem = folder_path / "work_dem.tif"
            work_hillshade = folder_path / "work_hillshade.tif"
            final_dem = folder_path / "dem.tif"
            final_hillshade = folder_path / "hillshade.tif"
            work_dem.write_bytes(b"new dem")
            work_hillshade.write_bytes(b"new hillshade")
            final_dem.write_bytes(b"old dem")
            final_hillshade.write_bytes(b"old hillshade")
            real_replace = os.replace

            def fail_hillshade_publish(source, destination):
                if Path(source) == work_hillshade and Path(destination) == final_hillshade:
                    raise OSError("injected hillshade publish failure")
                return real_replace(source, destination)

            with patch.object(
                dem_spec_module.os,
                "replace",
                side_effect=fail_hillshade_publish,
            ):
                with self.assertRaisesRegex(
                    DemSpecificationError,
                    "injected hillshade publish failure",
                ):
                    publish_output_files(
                        (
                            (str(work_dem), str(final_dem)),
                            (str(work_hillshade), str(final_hillshade)),
                        )
                    )

            self.assertEqual(final_dem.read_bytes(), b"old dem")
            self.assertEqual(final_hillshade.read_bytes(), b"old hillshade")

    def test_failed_restore_reports_and_preserves_recovery_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work_dem = folder_path / "work_dem.tif"
            work_hillshade = folder_path / "work_hillshade.tif"
            final_dem = folder_path / "dem.tif"
            final_hillshade = folder_path / "hillshade.tif"
            work_dem.write_bytes(b"new dem")
            work_hillshade.write_bytes(b"new hillshade")
            final_dem.write_bytes(b"old dem")
            final_hillshade.write_bytes(b"old hillshade")
            real_replace = os.replace

            def fail_publish_and_dem_restore(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if source_path == work_hillshade and destination_path == final_hillshade:
                    raise OSError("injected hillshade publish failure")
                if (
                    destination_path == final_dem
                    and source_path.name.endswith(".backup")
                ):
                    raise OSError("injected DEM restore failure")
                return real_replace(source, destination)

            with patch.object(
                dem_spec_module.os,
                "replace",
                side_effect=fail_publish_and_dem_restore,
            ):
                with self.assertRaisesRegex(
                    DemSpecificationError,
                    "Recovery backups preserved at",
                ) as raised:
                    publish_output_files(
                        (
                            (str(work_dem), str(final_dem)),
                            (str(work_hillshade), str(final_hillshade)),
                        )
                    )

            recovery_copies = list(
                folder_path.glob(f".{final_dem.name}.archaeotrace-*.backup")
            )
            self.assertEqual(len(recovery_copies), 1)
            self.assertEqual(recovery_copies[0].read_bytes(), b"old dem")
            self.assertIn(str(recovery_copies[0]), str(raised.exception))
            self.assertFalse(final_dem.exists())
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

    def test_publish_rejects_source_aliasing_another_destination_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work_dem = folder_path / "work_dem.tif"
            final_dem = folder_path / "dem.tif"
            final_hillshade = folder_path / "hillshade.tif"
            work_dem.write_bytes(b"new dem")
            final_dem.write_bytes(b"old dem")
            final_hillshade.write_bytes(b"old hillshade")

            with self.assertRaisesRegex(DemSpecificationError, "overlap"):
                publish_output_files(
                    (
                        (str(work_dem), str(final_dem)),
                        (str(final_dem), str(final_hillshade)),
                    )
                )

            self.assertEqual(work_dem.read_bytes(), b"new dem")
            self.assertEqual(final_dem.read_bytes(), b"old dem")
            self.assertEqual(final_hillshade.read_bytes(), b"old hillshade")

    def test_publish_materializes_generator_before_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work = folder_path / "work.tif"
            final = folder_path / "final.tif"
            work.write_bytes(b"new")

            publish_output_files((pair for pair in ((str(work), str(final)),)))

            self.assertEqual(final.read_bytes(), b"new")
            self.assertFalse(work.exists())

    def test_publish_rejects_non_regular_work_file(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            real_work = folder_path / "real_work.tif"
            linked_work = folder_path / "linked_work.tif"
            final = folder_path / "final.tif"
            real_work.write_bytes(b"new")
            try:
                os.symlink(real_work, linked_work)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(DemSpecificationError, "regular file"):
                publish_output_files(((str(linked_work), str(final)),))

            self.assertTrue(linked_work.is_symlink())
            self.assertFalse(final.exists())

    def test_publish_detects_source_swap_and_restores_previous_output(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work = folder_path / "work.tif"
            final = folder_path / "final.tif"
            work.write_bytes(b"trusted processing output")
            final.write_bytes(b"previous output")
            real_replace = os.replace
            swapped = False

            def swap_source_during_publish(source, destination):
                nonlocal swapped
                if not swapped and Path(source) == work and Path(destination) == final:
                    swapped = True
                    work.unlink()
                    work.write_bytes(b"untrusted replacement")
                return real_replace(source, destination)

            with patch.object(
                dem_spec_module.os,
                "replace",
                side_effect=swap_source_during_publish,
            ):
                with self.assertRaisesRegex(
                    DemSpecificationError,
                    "changed while publishing",
                ):
                    publish_output_files(((str(work), str(final)),))

            self.assertTrue(swapped)
            self.assertEqual(final.read_bytes(), b"previous output")
            self.assertEqual(
                list(folder_path.glob(".*.archaeotrace-*.backup")),
                [],
            )

    def test_publish_rejects_sidecar_that_appears_after_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            work = folder_path / "work.tif"
            final = folder_path / "final.tif"
            late_sidecar = folder_path / "work.tif.msk"
            work.write_bytes(b"trusted processing output")
            final.write_bytes(b"previous output")
            real_replace = os.replace

            def create_sidecar_during_publish(source, destination):
                if Path(source) == work and Path(destination) == final:
                    late_sidecar.write_bytes(b"unexpected sidecar")
                return real_replace(source, destination)

            with patch.object(
                dem_spec_module.os,
                "replace",
                side_effect=create_sidecar_during_publish,
            ):
                with self.assertRaisesRegex(
                    DemSpecificationError,
                    "appeared after preflight",
                ):
                    publish_output_files(((str(work), str(final)),))

            self.assertEqual(final.read_bytes(), b"previous output")
            self.assertEqual(late_sidecar.read_bytes(), b"unexpected sidecar")
            self.assertFalse((folder_path / "final.tif.msk").exists())


if __name__ == "__main__":
    unittest.main()
