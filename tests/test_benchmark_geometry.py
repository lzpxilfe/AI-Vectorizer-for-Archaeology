"""Tests for canonical centerline loading and rasterization."""

import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.geometry import (
    CenterlineFormatError,
    load_centerline_artifact,
    rasterize_centerlines,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "benchmarks" / "data" / "synthetic-smoke"


class CenterlineGeometryTests(unittest.TestCase):
    def _write_artifact(self, folder, paths, width=5, height=5):
        path = Path(folder) / "artifact.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "archaeotrace-centerline/1",
                    "coordinate_space": "pixel_xy",
                    "image_size": {"width": width, "height": height},
                    "paths": paths,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_fixture_rasterizes_ordered_inclusive_pixels(self):
        artifact = load_centerline_artifact(
            FIXTURE_ROOT / "references" / "straight-line.json"
        )

        raster = rasterize_centerlines(artifact)

        self.assertEqual(
            raster.paths[0].pixels,
            tuple((x, 4) for x in range(1, 8)),
        )
        self.assertEqual(sum(sum(row) for row in raster.mask), 7)

    def test_forward_and_reverse_segments_have_the_same_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            forward = self._write_artifact(
                folder,
                [{"id": "line", "points": [[0, 0], [4, 3]]}],
            )
            forward_mask = rasterize_centerlines(
                load_centerline_artifact(forward)
            ).mask
            reverse = self._write_artifact(
                folder,
                [{"id": "line", "points": [[4, 3], [0, 0]]}],
            )
            reverse_mask = rasterize_centerlines(
                load_centerline_artifact(reverse)
            ).mask

        self.assertEqual(forward_mask, reverse_mask)

    def test_clips_and_rounds_half_up_at_canvas_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_artifact(
                folder,
                [{"id": "line", "points": [[-2, 1.5], [6, 1.5]]}],
            )

            raster = rasterize_centerlines(load_centerline_artifact(path))

        self.assertEqual(raster.paths[0].pixels[0], (0, 2))
        self.assertEqual(raster.paths[0].pixels[-1], (4, 2))
        self.assertTrue(all(0 <= x < 5 and 0 <= y < 5 for x, y in raster.paths[0].pixels))

    def test_completely_external_segment_produces_an_empty_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_artifact(
                folder,
                [{"id": "line", "points": [[-5, -5], [-3, -3]]}],
            )

            raster = rasterize_centerlines(load_centerline_artifact(path))

        self.assertFalse(raster.paths)
        self.assertEqual(sum(sum(row) for row in raster.mask), 0)

    def test_rejects_non_finite_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_artifact(
                folder,
                [{"id": "line", "points": [[0, 0], [float("nan"), 2]]}],
            )

            with self.assertRaisesRegex(CenterlineFormatError, "finite"):
                load_centerline_artifact(path)


if __name__ == "__main__":
    unittest.main()
