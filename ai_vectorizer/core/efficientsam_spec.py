"""Pinned EfficientSAM model specifications.

This module deliberately contains data and standard-library validation only.
Importing it must not require NumPy, ONNX Runtime, QGIS, or network access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePath
import re
from typing import Tuple
from urllib.parse import urlsplit


_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class ArtifactSpec:
    """Immutable identity and download contract for one model artifact."""

    identifier: str
    filename: str
    url: str
    sha256: str
    size_bytes: int

    @property
    def id(self) -> str:
        """Public short alias used by CLI and worker evidence."""

        return self.identifier

    def __post_init__(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.identifier):
            raise ValueError(f"Invalid artifact identifier: {self.identifier!r}")
        if (
            not self.filename
            or PurePath(self.filename).name != self.filename
            or self.filename in {".", ".."}
            or "/" in self.filename
            or "\\" in self.filename
        ):
            raise ValueError(f"Invalid artifact filename: {self.filename!r}")
        parsed = urlsplit(self.url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Artifact URL must be a fixed HTTPS URL: {self.url!r}")
        if not _SHA256_PATTERN.fullmatch(self.sha256):
            raise ValueError(f"Invalid artifact SHA-256: {self.sha256!r}")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ValueError("Artifact size must be an integer.")
        if self.size_bytes <= 0:
            raise ValueError("Artifact size must be positive.")


@dataclass(frozen=True)
class ModelBundleSpec:
    """Immutable, versioned set of artifacts required by one model adapter."""

    identifier: str
    version: str
    source_repository: str
    source_commit: str
    license_spdx: str
    license_url: str
    artifacts: Tuple[ArtifactSpec, ...]

    @property
    def id(self) -> str:
        """Public short alias used by CLI and worker evidence."""

        return self.identifier

    def __post_init__(self) -> None:
        if not _IDENTIFIER_PATTERN.fullmatch(self.identifier):
            raise ValueError(f"Invalid bundle identifier: {self.identifier!r}")
        if not _IDENTIFIER_PATTERN.fullmatch(self.version):
            raise ValueError(f"Invalid bundle version: {self.version!r}")
        if not _COMMIT_PATTERN.fullmatch(self.source_commit):
            raise ValueError(f"Invalid source commit: {self.source_commit!r}")
        for label, url in (
            ("source repository", self.source_repository),
            ("license", self.license_url),
        ):
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.fragment
            ):
                raise ValueError(f"Invalid {label} URL: {url!r}")
        if not self.license_spdx or any(character.isspace() for character in self.license_spdx):
            raise ValueError("license_spdx must be a non-empty SPDX identifier.")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise ValueError("A model bundle must contain at least one artifact.")
        identifiers = [artifact.identifier for artifact in self.artifacts]
        filenames = [artifact.filename for artifact in self.artifacts]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Artifact identifiers must be unique within a bundle.")
        if len(set(filenames)) != len(filenames):
            raise ValueError("Artifact filenames must be unique within a bundle.")


def _fingerprint_document(spec: ModelBundleSpec) -> dict:
    return {
        "artifacts": [
            {
                "filename": artifact.filename,
                "id": artifact.identifier,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "url": artifact.url,
            }
            for artifact in spec.artifacts
        ],
        "id": spec.identifier,
        "license_spdx": spec.license_spdx,
        "license_url": spec.license_url,
        "source_commit": spec.source_commit,
        "source_repository": spec.source_repository,
        "version": spec.version,
    }


def bundle_fingerprint(spec: ModelBundleSpec) -> str:
    """Return the stable SHA-256 identity of a complete bundle contract."""

    if not isinstance(spec, ModelBundleSpec):
        raise TypeError("spec must be a ModelBundleSpec.")
    raw = json.dumps(
        _fingerprint_document(spec),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


EFFICIENTSAM_SOURCE_REPOSITORY = "https://github.com/yformer/EfficientSAM"
EFFICIENTSAM_SOURCE_COMMIT = "d525f622e6f640acf5a0fc37c7ca1f243da5bde0"
_EFFICIENTSAM_RAW_ROOT = (
    "https://raw.githubusercontent.com/yformer/EfficientSAM/"
    f"{EFFICIENTSAM_SOURCE_COMMIT}/weights"
)

EFFICIENTSAM_TI_ENCODER = ArtifactSpec(
    identifier="encoder",
    filename="efficient_sam_vitt_encoder.onnx",
    url=f"{_EFFICIENTSAM_RAW_ROOT}/efficient_sam_vitt_encoder.onnx",
    sha256="84ed466ffcc5c1f8d08409bc34a23bb364ab2c15e402cb12d4335a42be0e0951",
    size_bytes=24_799_761,
)

EFFICIENTSAM_TI_DECODER = ArtifactSpec(
    identifier="decoder",
    filename="efficient_sam_vitt_decoder.onnx",
    url=f"{_EFFICIENTSAM_RAW_ROOT}/efficient_sam_vitt_decoder.onnx",
    sha256="a62f8fa5ea080447c0689418d69e58f1e83e0b7adf9c142e2bd9bcc8045c0b11",
    size_bytes=16_565_728,
)

EFFICIENTSAM_TI_SPLIT = ModelBundleSpec(
    identifier="efficientsam-ti-split-onnx",
    version="v1",
    source_repository=EFFICIENTSAM_SOURCE_REPOSITORY,
    source_commit=EFFICIENTSAM_SOURCE_COMMIT,
    license_spdx="Apache-2.0",
    license_url=(
        f"{EFFICIENTSAM_SOURCE_REPOSITORY}/blob/"
        f"{EFFICIENTSAM_SOURCE_COMMIT}/LICENSE"
    ),
    artifacts=(EFFICIENTSAM_TI_ENCODER, EFFICIENTSAM_TI_DECODER),
)


__all__ = [
    "ArtifactSpec",
    "EFFICIENTSAM_SOURCE_COMMIT",
    "EFFICIENTSAM_SOURCE_REPOSITORY",
    "EFFICIENTSAM_TI_DECODER",
    "EFFICIENTSAM_TI_ENCODER",
    "EFFICIENTSAM_TI_SPLIT",
    "ModelBundleSpec",
    "bundle_fingerprint",
]
