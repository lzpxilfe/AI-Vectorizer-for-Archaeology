"""Rights-bound, sheet-isolated local datasets; importing never starts training."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

SCHEMA_VERSION = "archaeotrace-learning-dataset/1"
SPLITS = ("train", "validation", "calibration", "holdout")
PILOT_COUNTS = {"train": (6, 12), "validation": (2, 12), "calibration": (4, 6), "holdout": (4, 6)}


class DatasetError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetError("duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path, *, limit=8 * 1024 * 1024):
    try:
        payload = Path(path).read_bytes()
        if len(payload) > limit:
            raise DatasetError("JSON exceeds allocation limit")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(DatasetError("non-finite JSON")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError("could not read strict JSON: " + str(path)) from exc
    if not isinstance(value, dict):
        raise DatasetError("JSON root must be an object")
    return value


def safe_path(root, relative, *, must_exist=True):
    if not isinstance(relative, str) or not relative:
        raise DatasetError("artifact path is missing")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or part == Path("."):
        raise DatasetError("artifact path must stay within dataset")
    root = Path(root).resolve()
    current = root
    for component in part.parts:
        current = current / component
        if current.is_symlink():
            raise DatasetError("artifact path cannot traverse symlinks")
    if must_exist and not current.is_file():
        raise DatasetError("missing artifact: " + relative)
    return current


def _hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DatasetError(label + " must be SHA-256")
    return value


def checked_file(root, name, digest):
    path = safe_path(root, name)
    if sha256_file(path) != _hash(digest, name):
        raise DatasetError("artifact checksum mismatch: " + name)
    return path


def normalized_family(state, name):
    if not isinstance(state, str) or not isinstance(name, str) or not state.strip() or not name.strip():
        raise DatasetError("state and quadrangle_name are required")
    return state.lower().strip() + "-" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _http_url(value):
    if not isinstance(value, str):
        raise DatasetError("source URL missing")
    p = urlparse(value)
    if p.scheme != "https" or not p.hostname:
        raise DatasetError("source must have an HTTPS URL")
    return p.hostname.lower() + unquote(p.path)


def image_pixel_sha256(path):
    from PIL import Image
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (512, 512):
            raise DatasetError("learning crops must be lossless 512x512 native RGB PNGs")
        return hashlib.sha256(image.tobytes()).hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    payload: dict
    manifest_sha256: str

    @property
    def sha256(self):
        return self.manifest_sha256

    @property
    def root(self):
        return self.path.parent

    @property
    def sources(self):
        return {value["id"]: value for value in self.payload["sources"]}

    def samples_for(self, split):
        if split not in SPLITS:
            raise DatasetError("unsupported split")
        return tuple(sorted((item for item in self.payload["samples"] if item["split"] == split), key=lambda item: item["id"]))

    def sample_ids(self, split):
        return tuple(item["id"] for item in self.samples_for(split))


def load_dataset_manifest(path, *, require_approved=False, verify_files=True, verify_splits=None):
    """Validate all split identities, then optionally require train/val approval.

    Calibration and holdout labels are never loaded here. Their provenance and
    image identities are checked to catch aliases across the training boundary.
    """
    path = Path(path).resolve()
    data = read_json(path)
    selected_splits = set(SPLITS if verify_splits is None else verify_splits)
    if not selected_splits <= set(SPLITS):
        raise DatasetError("unsupported verification split")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError("unsupported dataset schema")
    if data.get("publication_ranking_eligible") is not False:
        raise DatasetError("a pilot dataset cannot claim publication eligibility")
    policy = data.get("split_policy", {})
    if policy.get("assignment_unit") != "sheet_family" or policy.get("frozen") is not True:
        raise DatasetError("sheet-family assignment must be frozen")
    sources = data.get("sources", [])
    samples = data.get("samples", [])
    if not isinstance(sources, list) or not isinstance(samples, list) or not sources or not samples or len(samples) > 4096:
        raise DatasetError("bounded source and sample lists required")
    if not isinstance(data.get("seed"), str) or not data["seed"]:
        raise DatasetError("dataset selection seed missing")
    if data.get("dataset_id") == "usgs-contour-pilot-16x144-v1" and "source_selection" not in data:
        raise DatasetError("pilot source selection must be frozen before cropping")
    if "source_selection" in data:
        frozen = data["source_selection"]
        selection = read_json(checked_file(path.parent, frozen["path"], frozen["sha256"]))
        before = {(s["id"], s["family_id"], s["split"], s["source_url"]) for s in selection["sources"]}
        after = {(s["id"], s["family_id"], s["split"], s["source_url"]) for s in sources}
        if before != after or selection.get("seed") != data["seed"]:
            raise DatasetError("dataset changed the frozen source assignment")
    seen_ids = set()
    identities = {}
    source_map = {}
    split_sources = {split: 0 for split in SPLITS}
    for source in sources:
        sid, split = source.get("id"), source.get("split")
        if not isinstance(sid, str) or not sid or sid in seen_ids or split not in SPLITS:
            raise DatasetError("source IDs and splits must be unique and valid")
        seen_ids.add(sid)
        source_map[sid] = source
        split_sources[split] += 1
        family = normalized_family(source.get("state"), source.get("quadrangle_name"))
        if source.get("family_id") != family:
            raise DatasetError("family ID must match quadrangle and state, not edition")
        if source.get("rights_status") not in ("public_domain", "open_license") or not source.get("license"):
            raise DatasetError("source redistribution rights remain unresolved")
        _http_url(source.get("rights_statement_url"))
        digest = _hash(source.get("source_sha256"), "source hash")
        provenance = source.get("provenance_id", {})
        if set(provenance) != {"authority", "namespace", "value"} or not all(isinstance(v, str) and v for v in provenance.values()):
            raise DatasetError("immutable source provenance missing")
        aliases = source.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias.strip() for alias in aliases):
            raise DatasetError("source aliases must be a string list")
        keys = [("family", family), ("hash", digest), ("url", _http_url(source.get("source_url"))),
                ("provenance", tuple(sorted(provenance.items())))]
        keys.extend(("family", alias.lower().strip()) for alias in aliases)
        for key in keys:
            if key in identities and identities[key] != split:
                raise DatasetError("source alias, edition, hash or URL leaks across splits")
            identities[key] = split
        if verify_files and split in selected_splits:
            checked_file(path.parent, source["rights_snapshot"], source["rights_snapshot_sha256"])
            if source.get("source_file") is not None:
                checked_file(path.parent, source["source_file"], digest)
    ids, pixel_splits = set(), {}
    split_samples = {split: 0 for split in SPLITS}
    for sample in samples:
        sid = sample.get("id")
        if not isinstance(sid, str) or not sid or sid in ids or sample.get("source_id") not in source_map:
            raise DatasetError("sample IDs must be unique and point to a source")
        ids.add(sid)
        split = sample.get("split")
        if split != source_map[sample["source_id"]]["split"]:
            raise DatasetError("sample cannot cross its source split")
        split_samples[split] += 1
        if sample.get("difficulty", "unassessed") not in ("unassessed", "standard", "hard"):
            raise DatasetError("case difficulty must be explicitly marked or unassessed")
        crop = sample.get("source_crop_xywh")
        if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop) or crop[0] < 0 or crop[1] < 0 or crop[2:] != [512, 512]:
            raise DatasetError("crop must be a bounded native 512px source rectangle")
        if sample.get("source_tile_origin_xy") != crop[:2]:
            raise DatasetError("source-grid origin must match crop coordinates")
        source = source_map[sample["source_id"]]
        if "source_size" in source:
            if len(source["source_size"]) != 2 or crop[0]+512>source["source_size"][0] or crop[1]+512>source["source_size"][1]:
                raise DatasetError("crop leaves the native source raster")
        transform = sample.get("geotransform")
        if not isinstance(transform, list) or len(transform) != 6 or not all(type(v) in (int, float) and math.isfinite(v) for v in transform):
            raise DatasetError("finite native affine required")
        if abs(transform[1] * transform[5] - transform[2] * transform[4]) <= 1e-24:
            raise DatasetError("singular native affine")
        if "source_geotransform" in source:
            g = source["source_geotransform"]
            if len(g) != 6 or not all(type(v) in (int,float) and math.isfinite(v) for v in g):
                raise DatasetError("source affine must be finite")
            expected = [g[0]+crop[0]*g[1]+crop[1]*g[2],g[1],g[2],g[3]+crop[0]*g[4]+crop[1]*g[5],g[4],g[5]]
            if not all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-8) for a,b in zip(transform,expected)):
                raise DatasetError("crop affine must preserve the native source grid")
        _hash(sample.get("image_sha256"), "image hash")
        pixel_hash = _hash(sample.get("image_pixel_sha256"), "pixel content hash")
        if pixel_hash in pixel_splits and pixel_splits[pixel_hash] != split:
            raise DatasetError("identical pixels leak across splits, despite different file names/encoding")
        pixel_splits[pixel_hash] = split
        if verify_files and split in selected_splits:
            image_path = checked_file(path.parent, sample["image"], sample["image_sha256"])
            if image_pixel_sha256(image_path) != pixel_hash:
                raise DatasetError("decoded pixel hash mismatch")
        if require_approved and split in ("train", "validation") and split in selected_splits:
            from .annotations import load_annotation
            annotation = sample.get("annotation", {})
            annotation_path = checked_file(path.parent, annotation.get("path"), annotation.get("sha256"))
            load_annotation(annotation_path, sample=sample, require_approved=True)
    expected_sources = policy.get("source_counts")
    expected_samples = policy.get("sample_counts")
    if expected_sources != split_sources or expected_samples != split_samples:
        raise DatasetError("frozen split counts disagree with materialized dataset")
    if data.get("dataset_id") == "usgs-contour-pilot-16x144-v1":
        if split_sources != {s: n for s, (n, _) in PILOT_COUNTS.items()} or split_samples != {s: n*c for s, (n, c) in PILOT_COUNTS.items()}:
            raise DatasetError("pilot requires 16 independent sheets and 144 crops")
    return DatasetManifest(path, data, sha256_file(path))


class ContourDataset:
    """Approved Train/Validation only; no implicit masks, download or approval."""
    def __init__(self, manifest_path, split, augmentation=None):
        if split not in ("train", "validation"):
            raise DatasetError("training access is restricted to train/validation")
        self.manifest = load_dataset_manifest(manifest_path, require_approved=False, verify_splits={split})
        self.manifest_sha256 = self.manifest.manifest_sha256
        self.samples = self.manifest.samples_for(split)
        self.sample_ids = tuple(s["id"] for s in self.samples)
        self.augmentation = augmentation
        from .annotations import load_annotation, rasterize_annotation
        for sample in self.samples:
            annotation = sample.get("annotation", {})
            try:
                p = checked_file(self.manifest.root, annotation.get("path"), annotation.get("sha256"))
                annotation_value = load_annotation(p, sample=sample, require_approved=True)
                if not rasterize_annotation(annotation_value)["valid_mask"].any():
                    raise DatasetError("sample has no supervised pixels")
            except (DatasetError, ValueError) as exc:
                raise DatasetError("sample needs explicit manual review: " + sample["id"]) from exc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        from .annotations import load_annotation, rasterize_annotation
        sample = self.samples[index]
        path = checked_file(self.manifest.root, sample["image"], sample["image_sha256"])
        with Image.open(path) as image:
            pixels = np.asarray(image, dtype=np.uint8).copy()
        info = sample["annotation"]
        annotation = load_annotation(checked_file(self.manifest.root, info["path"], info["sha256"]), sample=sample, require_approved=True)
        result = {"image": pixels, "sample_id": sample["id"], **rasterize_annotation(annotation)}
        return self.augmentation(result) if self.augmentation is not None else result
