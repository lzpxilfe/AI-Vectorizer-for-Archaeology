import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock
import zlib

from benchmarks.public_dataset import (
    PUBLIC_DATASET_STRATA,
    PublicDatasetError,
    validate_public_dataset_plan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    REPOSITORY_ROOT
    / "benchmarks"
    / "data"
    / "public-8x6-template"
    / "dataset-plan.json"
)


def _template_payload():
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _gray_png(width, height):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + b"\x7f" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


class PublicDatasetPlanTests(unittest.TestCase):
    def _write(self, root, payload):
        path = Path(root) / "dataset-plan.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def test_checked_in_template_is_balanced_but_not_materialized(self):
        report = validate_public_dataset_plan(TEMPLATE)
        self.assertEqual(report.sheet_count, 8)
        self.assertEqual(report.crop_count, 48)
        self.assertEqual(report.split_sheet_counts, {"calibration": 4, "holdout": 4})
        self.assertEqual(report.split_crop_counts, {"calibration": 24, "holdout": 24})
        self.assertFalse(report.materialized)
        self.assertFalse(report.publication_ranking_eligible)
        for split in ("calibration", "holdout"):
            self.assertEqual(
                report.origin_group_counts[split],
                {"usgs_htmc": 2, "korea_rights_cleared": 2},
            )
            self.assertEqual(
                report.stratum_counts[split],
                {stratum: 3 for stratum in PUBLIC_DATASET_STRATA},
            )

    def test_materialized_gate_rejects_unresolved_rights_and_assets(self):
        with self.assertRaisesRegex(PublicDatasetError, "remain unresolved"):
            validate_public_dataset_plan(TEMPLATE, require_materialized=True)

    def test_split_policy_rejects_sheet_leakage(self):
        payload = _template_payload()
        payload["split_policy"]["holdout_sheet_ids"][0] = "cal-01"
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(PublicDatasetError, "overlap"):
                validate_public_dataset_plan(path)

    def test_each_split_requires_three_of_every_fixed_difficulty(self):
        payload = _template_payload()
        payload["sheets"][0]["crops"][0]["difficulty_stratum"] = "thick_or_scale"
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(PublicDatasetError, "at least three"):
                validate_public_dataset_plan(path)

    def test_each_split_requires_two_sheets_from_each_origin_group(self):
        payload = _template_payload()
        payload["sheets"][0]["origin_group"] = "korea_rights_cleared"
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(PublicDatasetError, "two sheets"):
                validate_public_dataset_plan(path)

    def test_populated_asset_hash_is_verified_without_network_access(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        crop.update(
            image="crop.png",
            image_sha256="0" * 64,
            ordered_reference="reference.json",
            ordered_reference_sha256="0" * 64,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "crop.png").write_bytes(b"local crop")
            (root / "reference.json").write_bytes(b"local reference")
            path = self._write(root, payload)
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("validator attempted network I/O"),
            ):
                with self.assertRaisesRegex(PublicDatasetError, "checksum mismatch"):
                    validate_public_dataset_plan(path)

    def test_crop_bytes_must_be_a_structurally_valid_png(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        image = b"not a PNG"
        reference = b"not a reference"
        crop.update(
            source_crop_xywh=[0, 0, 1, 1],
            image="crop.png",
            image_sha256=hashlib.sha256(image).hexdigest(),
            ordered_reference="reference.json",
            ordered_reference_sha256=hashlib.sha256(reference).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "crop.png").write_bytes(image)
            (root / "reference.json").write_bytes(reference)
            path = self._write(root, payload)
            with self.assertRaisesRegex(PublicDatasetError, "valid bounded PNG"):
                validate_public_dataset_plan(path)

    def test_ordered_reference_must_parse_and_match_crop_dimensions(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        image = _gray_png(1, 1)
        reference = b"not a centerline artifact"
        crop.update(
            source_crop_xywh=[0, 0, 1, 1],
            image="crop.png",
            image_sha256=hashlib.sha256(image).hexdigest(),
            ordered_reference="reference.json",
            ordered_reference_sha256=hashlib.sha256(reference).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "crop.png").write_bytes(image)
            (root / "reference.json").write_bytes(reference)
            path = self._write(root, payload)
            with self.assertRaisesRegex(PublicDatasetError, "ordered_reference is invalid"):
                validate_public_dataset_plan(path)

    def test_png_dimensions_must_match_source_crop_xywh(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        image = _gray_png(1, 1)
        reference = b"not reached"
        crop.update(
            source_crop_xywh=[10, 20, 2, 2],
            image="crop.png",
            image_sha256=hashlib.sha256(image).hexdigest(),
            ordered_reference="reference.json",
            ordered_reference_sha256=hashlib.sha256(reference).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "crop.png").write_bytes(image)
            (root / "reference.json").write_bytes(reference)
            path = self._write(root, payload)
            with self.assertRaisesRegex(PublicDatasetError, "dimensions must match"):
                validate_public_dataset_plan(path)

    def test_source_tile_origin_must_match_public_crop_origin(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        crop["source_crop_xywh"] = [128, 64, 10, 10]
        crop["source_tile_origin_xy"] = [0, 0]

        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(
                PublicDatasetError,
                "must equal the source_crop_xywh origin",
            ):
                validate_public_dataset_plan(path)

    def test_ordered_reference_cannot_be_an_empty_centerline(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        image = _gray_png(1, 1)
        reference = (
            json.dumps(
                {
                    "schema_version": "archaeotrace-centerline/1",
                    "coordinate_space": "pixel_xy",
                    "image_size": {"width": 1, "height": 1},
                    "paths": [],
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        crop.update(
            source_crop_xywh=[0, 0, 1, 1],
            image="crop.png",
            image_sha256=hashlib.sha256(image).hexdigest(),
            ordered_reference="reference.json",
            ordered_reference_sha256=hashlib.sha256(reference).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "crop.png").write_bytes(image)
            (root / "reference.json").write_bytes(reference)
            path = self._write(root, payload)
            with self.assertRaisesRegex(PublicDatasetError, "ordered centerline"):
                validate_public_dataset_plan(path)

    def test_reviewer_and_adjudicator_must_be_independent(self):
        payload = _template_payload()
        annotation = payload["sheets"][0]["crops"][0]["annotation"]
        annotation.update(
            reviewer_id="reviewer-1",
            review_status="approved",
            adjudicator_id="reviewer-1",
            adjudication_status="accepted",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(PublicDatasetError, "must be independent"):
                validate_public_dataset_plan(path)

    def test_publication_eligibility_cannot_be_enabled_in_plan(self):
        payload = _template_payload()
        payload["publication_ranking_eligible"] = True
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(PublicDatasetError, "must remain false"):
                validate_public_dataset_plan(path)

    def test_materialized_prompt_accepts_optional_previous_xy(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        prompt = {
            "schema_version": "archaeotrace-trace-prompt/2",
            "start_xy": [1, 1],
            "end_xy": [8, 8],
            "positive_xy": [],
            "negative_xy": [],
        }
        crop["source_crop_xywh"] = [0, 0, 10, 10]
        crop["prompt"] = dict(prompt)
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            validate_public_dataset_plan(path)

            crop["prompt"]["previous_xy"] = None
            path = self._write(folder, payload)
            validate_public_dataset_plan(path)

            crop["prompt"]["schema_version"] = "archaeotrace-trace-prompt/1"
            path = self._write(folder, payload)
            with self.assertRaisesRegex(
                PublicDatasetError,
                "schema_version must be",
            ):
                validate_public_dataset_plan(path)

            crop["prompt"]["schema_version"] = "archaeotrace-trace-prompt/2"
            crop["prompt"]["unexpected"] = []
            path = self._write(folder, payload)
            with self.assertRaisesRegex(
                PublicDatasetError,
                "unsupported or missing fields",
            ):
                validate_public_dataset_plan(path)


if __name__ == "__main__":
    unittest.main()
