import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest import mock
import zlib

try:
    from PIL import Image as PillowImage
except ImportError:  # pragma: no cover - dependency-free QGIS compatibility run
    PillowImage = None

from benchmarks.public_assets import (
    PublicAssetVerificationError,
    verify_lossless_source_crop,
    verify_public_dataset_assets,
)
from benchmarks.public_dataset import (
    PUBLIC_DATASET_STRATA,
    PublicDatasetError,
    is_independently_accepted_annotation,
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


def _checked_in_payload():
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _template_payload():
    """Return the unresolved contract fixture used by isolated mutation tests."""

    payload = _checked_in_payload()
    sheet = payload["sheets"][0]
    sheet["source"] = {
        "title": "Unresolved rights-cleared source slot cal-01",
        "publisher": None,
        "date_or_sheet": None,
        "source_url": None,
        "provenance_id": None,
        "license": None,
        "rights_status": "unresolved",
        "rights_statement_url": None,
        "text_snapshot": None,
        "text_snapshot_sha256": None,
        "source_raster": None,
        "source_raster_sha256": None,
    }
    for crop in sheet["crops"]:
        for field in (
            "source_crop_xywh",
            "source_tile_origin_xy",
            "image",
            "image_sha256",
            "prompt",
            "ordered_reference",
            "ordered_reference_sha256",
        ):
            crop[field] = None
        crop["notes"] = None
    return payload


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

    def _copy_checked_in_dataset(self, root):
        target = Path(root) / "public-8x6-template"
        shutil.copytree(TEMPLATE.parent, target)
        return target / "dataset-plan.json"

    def _write_reference_mutation(self, plan_path, reference_payload):
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        crop = payload["sheets"][0]["crops"][0]
        reference_path = plan_path.parent / crop["ordered_reference"]
        reference_path.write_text(
            json.dumps(reference_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        crop["ordered_reference_sha256"] = hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest()
        plan_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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

    def test_first_usgs_sheet_is_staged_but_references_remain_unreviewed(self):
        payload = _checked_in_payload()
        sheet = payload["sheets"][0]
        source = sheet["source"]

        self.assertEqual(sheet["id"], "cal-01")
        self.assertEqual(sheet["origin_group"], "usgs_htmc")
        self.assertEqual(source["rights_status"], "public_domain")
        self.assertEqual(source["publisher"], "U.S. Geological Survey")
        self.assertEqual(
            source["provenance_id"],
            {
                "authority": "sciencebase.gov",
                "namespace": "catalog-item",
                "value": "604ea84ad34eb12031203797",
            },
        )
        self.assertTrue(source["source_raster"].endswith(".tif"))
        self.assertIsNotNone(source["text_snapshot"])
        for crop in sheet["crops"]:
            self.assertTrue(crop["image"].endswith(".png"))
            self.assertIsNotNone(crop["prompt"])
            self.assertIsNotNone(crop["ordered_reference"])
            self.assertEqual(crop["annotation"]["review_status"], "unreviewed")
            self.assertEqual(crop["annotation"]["adjudication_status"], "pending")

        report = validate_public_dataset_plan(TEMPLATE)
        self.assertFalse(report.materialized)
        self.assertFalse(report.publication_ranking_eligible)

    @unittest.skipIf(PillowImage is None, "Pillow is not installed")
    def test_checked_in_usgs_crops_are_exact_source_pixels(self):
        report = verify_public_dataset_assets(TEMPLATE)
        self.assertEqual(report.staged_sheet_count, 1)
        self.assertEqual(report.verified_crop_count, 6)
        self.assertEqual(report.draft_reference_count, 6)

    @unittest.skipIf(PillowImage is None, "Pillow is not installed")
    def test_lossless_crop_verifier_rejects_pixel_substitution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_path = root / "source.png"
            crop_path = root / "crop.png"
            source = PillowImage.new("RGB", (4, 4), (240, 220, 190))
            source.putpixel((2, 2), (40, 90, 120))
            source.save(source_path)
            source.crop((1, 1, 3, 3)).save(crop_path)

            verify_lossless_source_crop(source_path, crop_path, [1, 1, 2, 2])
            changed = PillowImage.open(crop_path).convert("RGB")
            changed.putpixel((1, 1), (41, 90, 120))
            changed.save(crop_path)
            with self.assertRaisesRegex(
                PublicAssetVerificationError,
                "do not equal",
            ):
                verify_lossless_source_crop(source_path, crop_path, [1, 1, 2, 2])

    @unittest.skipIf(PillowImage is None, "Pillow is not installed")
    def test_public_reference_must_be_one_open_in_bounds_prompted_path(self):
        with tempfile.TemporaryDirectory() as folder:
            plan_path = self._copy_checked_in_dataset(folder)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            crop = payload["sheets"][0]["crops"][0]
            reference_path = plan_path.parent / crop["ordered_reference"]
            original = json.loads(reference_path.read_text(encoding="utf-8"))

            extra_path = copy.deepcopy(original)
            extra_path["paths"].append(
                {
                    "id": "unexpected-extra-path",
                    "closed": False,
                    "points": [[0, 0], [1, 1]],
                }
            )
            self._write_reference_mutation(plan_path, extra_path)
            with self.assertRaisesRegex(
                PublicAssetVerificationError,
                "exactly one prompted path",
            ):
                verify_public_dataset_assets(plan_path)

            closed_path = copy.deepcopy(original)
            closed_path["paths"][0]["closed"] = True
            self._write_reference_mutation(plan_path, closed_path)
            with self.assertRaisesRegex(
                PublicAssetVerificationError,
                "path must be open",
            ):
                verify_public_dataset_assets(plan_path)

            outside_path = copy.deepcopy(original)
            outside_path["paths"][0]["points"][1][0] = -1
            self._write_reference_mutation(plan_path, outside_path)
            with self.assertRaisesRegex(
                PublicAssetVerificationError,
                "lies outside the crop",
            ):
                verify_public_dataset_assets(plan_path)

            zero_length = copy.deepcopy(original)
            repeated_point = zero_length["paths"][0]["points"][0]
            zero_length["paths"][0]["points"] = [
                repeated_point,
                repeated_point,
                repeated_point,
            ]
            self._write_reference_mutation(plan_path, zero_length)
            with mock.patch(
                "benchmarks.public_assets.validate_public_dataset_plan"
            ):
                with self.assertRaisesRegex(
                    PublicAssetVerificationError,
                    "positive geometric length",
                ):
                    verify_public_dataset_assets(plan_path)

    @unittest.skipIf(PillowImage is None, "Pillow is not installed")
    def test_asset_verifier_rejects_identical_prompt_endpoints(self):
        with tempfile.TemporaryDirectory() as folder:
            plan_path = self._copy_checked_in_dataset(folder)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            prompt = payload["sheets"][0]["crops"][0]["prompt"]
            prompt["end_xy"] = list(prompt["start_xy"])
            plan_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            with mock.patch(
                "benchmarks.public_assets.validate_public_dataset_plan"
            ):
                with self.assertRaisesRegex(
                    PublicAssetVerificationError,
                    "prompt start and end must differ",
                ):
                    verify_public_dataset_assets(plan_path)

    @unittest.skipIf(PillowImage is None, "Pillow is not installed")
    def test_accepted_status_without_reviewer_id_remains_a_draft(self):
        with tempfile.TemporaryDirectory() as folder:
            plan_path = self._copy_checked_in_dataset(folder)
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            crop = payload["sheets"][0]["crops"][0]
            reference_path = plan_path.parent / crop["ordered_reference"]
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            reference["metadata"]["annotation_status"] = "not_a_draft"
            reference_path.write_text(
                json.dumps(reference, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            crop["ordered_reference_sha256"] = hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest()
            crop["annotation"].update(
                reviewer_id=None,
                review_status="approved",
                adjudicator_id=None,
                adjudication_status="accepted",
            )
            plan_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            self.assertFalse(validate_public_dataset_plan(plan_path).materialized)
            with self.assertRaisesRegex(
                PublicAssetVerificationError,
                "must identify itself as a draft",
            ):
                verify_public_dataset_assets(plan_path)

    def test_independent_acceptance_requires_two_distinct_named_identities(self):
        accepted = {
            "reviewer_id": "reviewer-1",
            "review_status": "approved",
            "adjudicator_id": "adjudicator-1",
            "adjudication_status": "accepted",
        }
        self.assertTrue(is_independently_accepted_annotation(accepted))

        invalid_updates = (
            {"reviewer_id": None},
            {"adjudicator_id": ""},
            {"adjudicator_id": " reviewer-1 "},
            {"review_status": "unreviewed"},
            {"adjudication_status": "pending"},
        )
        for update in invalid_updates:
            with self.subTest(update=update):
                candidate = dict(accepted)
                candidate.update(update)
                self.assertFalse(is_independently_accepted_annotation(candidate))

    def test_materialized_source_requires_immutable_provenance_id(self):
        payload = _checked_in_payload()
        payload["sheets"][0]["source"]["provenance_id"] = None
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(
                PublicDatasetError,
                "provenance_id must be an object",
            ):
                validate_public_dataset_plan(path)

    def test_provenance_id_is_globally_unique_across_splits(self):
        payload = _template_payload()
        provenance_id = {
            "authority": "example.gov",
            "namespace": "catalog-item",
            "value": "same-upstream-sheet",
        }
        payload["sheets"][0]["source"]["provenance_id"] = provenance_id
        payload["sheets"][4]["source"]["provenance_id"] = dict(provenance_id)
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(
                PublicDatasetError,
                "globally unique across splits",
            ):
                validate_public_dataset_plan(path)

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

    def test_materialized_prompt_requires_distinct_start_and_end(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        crop["source_crop_xywh"] = [0, 0, 10, 10]
        crop["prompt"] = {
            "schema_version": "archaeotrace-trace-prompt/2",
            "start_xy": [4, 5],
            "end_xy": [4.0, 5.0],
            "positive_xy": [],
            "negative_xy": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, payload)
            with self.assertRaisesRegex(
                PublicDatasetError,
                "start_xy and end_xy must differ",
            ):
                validate_public_dataset_plan(path)

    def test_ordered_reference_requires_positive_geometric_length(self):
        payload = _template_payload()
        crop = payload["sheets"][0]["crops"][0]
        image = _gray_png(10, 10)
        reference = (
            json.dumps(
                {
                    "schema_version": "archaeotrace-centerline/1",
                    "coordinate_space": "pixel_xy",
                    "image_size": {"width": 10, "height": 10},
                    "paths": [
                        {
                            "id": "degenerate",
                            "closed": False,
                            "points": [[4, 5], [4.0, 5.0], [4, 5]],
                        }
                    ],
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        crop.update(
            source_crop_xywh=[0, 0, 10, 10],
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
            with self.assertRaisesRegex(
                PublicDatasetError,
                "positive geometric length",
            ):
                validate_public_dataset_plan(path)


if __name__ == "__main__":
    unittest.main()
