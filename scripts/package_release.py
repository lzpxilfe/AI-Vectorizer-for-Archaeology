#!/usr/bin/env python3
"""Build and verify the packaged ArchaeoTrace plugin release."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import io
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR_NAME = "ai_vectorizer"
PLUGIN_DIR = ROOT / PLUGIN_DIR_NAME
DIST_DIR = ROOT / "dist"
TOP_LEVEL_ITEMS = (
    "__init__.py",
    "plugin.py",
    "config.py",
    "recovery.py",
    "metadata.txt",
    ".secrets.baseline",
    "README.md",
    "LICENSE",
    "icon.png",
    "core",
    "tools",
    "ui",
)
IGNORED_NAMES = {"__pycache__", ".DS_Store"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
IGNORED_WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx", ".ckpt", ".bin", ".caffemodel"}
IGNORED_MODEL_TEMP_SUFFIXES = {".download", ".migration", ".rollback", ".tmp"}
IGNORED_MODEL_NAME_SUFFIXES = {".meta.json"}
IGNORED_HED_TEMP_PREFIXES = {"hed_prototxt_", "hed_weights_"}
PROHIBITED_NATIVE_SUFFIXES = {".dll", ".dylib", ".exe", ".pyd", ".so"}
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9._-]+)?")
DEFAULT_SOURCE_DATE_EPOCH = 315532800  # 1980-01-01, the earliest ZIP timestamp.
MAX_SOURCE_DATE_EPOCH = 4354819198  # 2107-12-31 23:59:58 UTC.
# The public QGIS publishing requirements say packages should not exceed a
# decimal 20 MB. The repository backend currently has a looser 25,000,000-byte
# hard cap; enforcing the published 20 MB guideline satisfies both gates.
MAX_UPLOAD_BYTES = 20_000_000
ZIP_FILE_MODE = 0o100644
ZIP_COMPRESSION = zipfile.ZIP_STORED
# Repository-local release candidates are immutable inputs after their version
# is assigned. These hashes are not
# necessarily the identity of an artifact already published by an external
# repository.  A differing source build must never replace the local filename
# unless the operator explicitly confirms the metadata version.
FROZEN_RELEASE_SHA256 = {
    "0.1.5": "d2925198dc2192bbb7eebe579bb48207c860179a94d4216df77e746d0451789a",
    "0.1.6": "fffceee8607bdb19178b224e2c94493791d0175a1e35039368f0a41fa7447b7a",
}


def load_version() -> str:
    # Match the official QGIS plugin repository parser. In particular, this
    # forces malformed interpolation tokens (such as an unescaped ``%``) in
    # any metadata value to fail the release build instead of the upload.
    parser = configparser.ConfigParser()
    parser.optionxform = str
    with (PLUGIN_DIR / "metadata.txt").open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    metadata = dict(parser.items("general"))
    version = metadata["version"].strip()
    if VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"Invalid plugin metadata version: {version!r}")
    return version


def release_dir(version: str) -> Path:
    return ROOT / f"{PLUGIN_DIR_NAME} {version}"


def zip_path(version: str) -> Path:
    return DIST_DIR / f"{PLUGIN_DIR_NAME}-{version}.zip"


def should_skip(path: Path) -> bool:
    if path.name in IGNORED_NAMES or path.suffix in IGNORED_SUFFIXES:
        return True

    in_models_directory = any(parent.name == "models" for parent in path.parents)
    if in_models_directory:
        suffix = path.suffix.lower()
        if suffix in IGNORED_WEIGHT_SUFFIXES | IGNORED_MODEL_TEMP_SUFFIXES:
            return True
        if any(
            path.name.endswith(name_suffix)
            for name_suffix in IGNORED_MODEL_NAME_SUFFIXES
        ):
            return True
        if any(path.name.startswith(prefix) for prefix in IGNORED_HED_TEMP_PREFIXES):
            return True

    return False


def assert_publishable_file(path: Path, relative_path: Path) -> None:
    """Reject hidden residue and native binaries forbidden by QGIS policy."""
    if any(part.startswith(".") for part in relative_path.parts):
        raise ValueError(f"Hidden files are not allowed in the plugin package: {path}")
    lower_name = path.name.lower()
    if path.suffix.lower() in PROHIBITED_NATIVE_SUFFIXES or ".so." in lower_name:
        raise ValueError(f"Native binaries are not allowed in the plugin package: {path}")


def is_link_like(path: Path) -> bool:
    """Return whether *path* is a symlink or Windows reparse-point directory."""
    if path.is_symlink():
        return True

    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            if is_junction():
                return True
        except OSError:
            return True

    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def assert_safe_source_path(path: Path, source_root: Path) -> None:
    """Reject link-like paths and paths that resolve outside the source root."""
    if is_link_like(path):
        raise ValueError(
            f"Plugin source symlinks are not allowed (including junctions/reparse points): {path}"
        )
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(source_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Plugin source path escapes the source tree: {path}") from exc


def iter_source_files() -> list[tuple[Path, Path]]:
    if is_link_like(PLUGIN_DIR):
        raise ValueError(
            "Plugin source symlinks are not allowed "
            f"(including junctions/reparse points): {PLUGIN_DIR}"
        )
    source_root = PLUGIN_DIR.resolve(strict=True)
    files: list[tuple[Path, Path]] = []
    for item_name in TOP_LEVEL_ITEMS:
        src = PLUGIN_DIR / item_name
        if not src.exists():
            raise FileNotFoundError(f"Missing required plugin item: {src}")
        assert_safe_source_path(src, source_root)
        if src.is_file():
            files.append((src, Path(item_name)))
            continue
        for child in sorted(src.rglob("*")):
            assert_safe_source_path(child, source_root)
            if child.is_dir():
                continue
            relative_child = child.relative_to(PLUGIN_DIR)
            if any(should_skip(parent) for parent in relative_child.parents):
                continue
            if should_skip(child):
                continue
            assert_publishable_file(child, relative_child)
            files.append((child, relative_child))
    return sorted(files, key=lambda pair: pair[1].as_posix())


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_hash(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    return {
        rel_path.as_posix(): file_hash(src_path)
        for src_path, rel_path in iter_source_files()
    }


def build_release_tree(version: str) -> Path:
    target_dir = release_dir(version)
    if os.path.lexists(target_dir) and is_link_like(target_dir):
        raise ValueError(
            "Release directory symlinks are not allowed "
            f"(including junctions/reparse points): {target_dir}"
        )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for src_path, rel_path in iter_source_files():
        destination = target_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, destination)

    return target_dir


def source_date_epoch() -> int:
    raw_value = os.environ.get("SOURCE_DATE_EPOCH")
    if raw_value is None:
        return DEFAULT_SOURCE_DATE_EPOCH
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError("SOURCE_DATE_EPOCH must be an integer") from error
    if value < DEFAULT_SOURCE_DATE_EPOCH:
        raise ValueError("SOURCE_DATE_EPOCH predates the ZIP timestamp range")
    if value > MAX_SOURCE_DATE_EPOCH:
        raise ValueError("SOURCE_DATE_EPOCH exceeds the ZIP timestamp range")
    return value


def build_release_zip_bytes() -> bytes:
    timestamp = time.gmtime(source_date_epoch())[:6]
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        "w",
        compression=ZIP_COMPRESSION,
    ) as archive:
        for src_path, rel_path in iter_source_files():
            archive_name = (Path(PLUGIN_DIR_NAME) / rel_path).as_posix()
            info = zipfile.ZipInfo(archive_name, date_time=timestamp)
            info.create_system = 3
            info.compress_type = ZIP_COMPRESSION
            info.external_attr = ZIP_FILE_MODE << 16
            archive.writestr(
                info,
                src_path.read_bytes(),
                compress_type=ZIP_COMPRESSION,
            )
    return buffer.getvalue()


def _same_path(left: Path, right: Path) -> bool:
    """Compare output paths across aliases and case-insensitive filesystems."""

    resolved_left = left.resolve(strict=False)
    resolved_right = right.resolve(strict=False)
    if resolved_left == resolved_right:
        return True

    # ``Path.resolve()`` preserves the spelling supplied by the caller on
    # case-insensitive APFS/NTFS.  When the frozen artifact exists, samefile()
    # is the authoritative check and also handles other filesystem aliases.
    try:
        if os.path.samefile(left, right):
            return True
    except (FileNotFoundError, OSError, ValueError):
        pass

    # Retain fail-closed protection when the target itself is absent but both
    # spellings address the same existing directory.  This is deliberately
    # conservative on a case-sensitive filesystem: an explicit output whose
    # basename differs only by case is not worth risking the frozen local ZIP.
    try:
        same_parent = os.path.samefile(left.parent, right.parent)
    except (FileNotFoundError, OSError, ValueError):
        same_parent = False
    return bool(same_parent and left.name.casefold() == right.name.casefold())


def _assert_production_write_allowed(
    version: str,
    target_zip: Path,
    payload: bytes,
    *,
    approved_version: Optional[str],
) -> None:
    """Protect every frozen local candidate, including older version paths."""

    payload_hash = bytes_hash(payload)
    for frozen_version, frozen_hash in FROZEN_RELEASE_SHA256.items():
        frozen_zip = zip_path(frozen_version)
        if not _same_path(target_zip, frozen_zip):
            continue
        if payload_hash == frozen_hash:
            return
        if approved_version == frozen_version and version == frozen_version:
            return
        raise ValueError(
            f"Refusing to replace frozen local candidate {frozen_zip.name}: "
            f"metadata {version} source builds SHA-256 {payload_hash}, not frozen "
            f"{frozen_hash}. Build current source with --output PATH outside every "
            "frozen candidate, or use "
            f"--approve-release-overwrite {frozen_version} only while metadata is "
            f"exactly {frozen_version}."
        )


def build_release_zip(
    version: str,
    *,
    output_path: Optional[Path] = None,
    approved_version: Optional[str] = None,
) -> Path:
    target_zip = Path(output_path) if output_path is not None else zip_path(version)
    payload = build_release_zip_bytes()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"Release ZIP exceeds the {MAX_UPLOAD_BYTES // 1_000_000} MB "
            "QGIS repository package upload limit"
        )
    _assert_production_write_allowed(
        version,
        target_zip,
        payload,
        approved_version=approved_version,
    )

    target_zip.parent.mkdir(parents=True, exist_ok=True)

    descriptor = None
    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target_zip.name}.",
            suffix=".tmp",
            dir=target_zip.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o644)
        except (AttributeError, OSError):
            pass
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target_zip)
        temporary_path = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    return target_zip


def release_manifest(version: str) -> dict[str, str]:
    target_dir = release_dir(version)
    if is_link_like(target_dir):
        raise ValueError(
            "Release directory symlinks are not allowed "
            f"(including junctions/reparse points): {target_dir}"
        )
    if not target_dir.exists():
        raise FileNotFoundError(f"Release directory does not exist: {target_dir}")

    release_root = target_dir.resolve(strict=True)
    manifest: dict[str, str] = {}
    for path in sorted(target_dir.rglob("*")):
        if is_link_like(path):
            raise ValueError(
                "Release tree symlinks are not allowed "
                f"(including junctions/reparse points): {path}"
            )
        try:
            path.resolve(strict=True).relative_to(release_root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Release path escapes the release tree: {path}") from exc
        if path.is_dir():
            continue
        if should_skip(path):
            continue
        manifest[path.relative_to(target_dir).as_posix()] = file_hash(path)
    return manifest


def zip_manifest(
    version: str,
    *,
    archive_path: Optional[Path] = None,
) -> dict[str, str]:
    archive_path = Path(archive_path) if archive_path is not None else zip_path(version)
    if not archive_path.exists():
        raise FileNotFoundError(f"Release zip does not exist: {archive_path}")

    manifest: dict[str, str] = {}
    prefix = f"{PLUGIN_DIR_NAME}/"
    with zipfile.ZipFile(archive_path, "r") as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/"):
                continue
            if not name.startswith(prefix):
                raise ValueError(f"Unexpected zip entry outside plugin root: {name}")
            rel_name = name[len(prefix):]
            if not rel_name:
                continue
            manifest[rel_name] = bytes_hash(archive.read(name))
    return manifest


def compare_manifests(label: str, expected: dict[str, str], actual: dict[str, str]) -> list[str]:
    problems: list[str] = []
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])

    for name in missing:
        problems.append(f"{label}: missing {name}")
    for name in extra:
        problems.append(f"{label}: unexpected {name}")
    for name in changed:
        problems.append(f"{label}: changed {name}")
    return problems


def run_check(
    version: str,
    *,
    archive_path: Optional[Path] = None,
) -> int:
    expected = source_manifest()
    problems: list[str] = []

    # An explicit output is the current-source/Unreleased path and does not use
    # or validate the metadata-derived repository-local candidate directory.
    if archive_path is None:
        try:
            problems.extend(compare_manifests("release dir", expected, release_manifest(version)))
        except Exception as exc:
            problems.append(str(exc))

    try:
        checked_zip = Path(archive_path) if archive_path is not None else zip_path(version)
        problems.extend(
            compare_manifests(
                "release zip",
                expected,
                zip_manifest(version, archive_path=checked_zip),
            )
        )
        archive_size = checked_zip.stat().st_size
        if archive_size > MAX_UPLOAD_BYTES:
            problems.append(
                f"release zip: exceeds the {MAX_UPLOAD_BYTES // 1_000_000} MB "
                "QGIS repository package upload limit"
            )
        expected_zip_hash = bytes_hash(build_release_zip_bytes())
        actual_zip_hash = file_hash(checked_zip)
        if actual_zip_hash != expected_zip_hash:
            problems.append(
                "release zip: archive bytes are not the deterministic source build"
            )
    except Exception as exc:
        problems.append(str(exc))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    if archive_path is None:
        print(f"Release artifacts are in sync for {version}.")
    else:
        print(f"Current-source package is in sync: {Path(archive_path)}")
    return 0


def run_build(
    version: str,
    *,
    output_path: Optional[Path] = None,
    approved_version: Optional[str] = None,
) -> int:
    # Guard/write the ZIP before mutating the repository-local candidate tree.
    # Thus a normal Unreleased invocation cannot partially replace it.
    target_zip = build_release_zip(
        version,
        output_path=output_path,
        approved_version=approved_version,
    )
    if output_path is None:
        target_dir = build_release_tree(version)
        print(f"Built release directory: {target_dir}")
        print(f"Built release zip: {target_zip}")
        print(f"Release zip SHA-256: {file_hash(target_zip)}")
    else:
        print(f"Built current-source package: {target_zip}")
        print(f"Current-source package SHA-256: {file_hash(target_zip)}")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the packaged ArchaeoTrace release artifacts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the generated release directory and zip match the root source tree.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Build or check a current-source ZIP at an explicit path without "
            "touching metadata-derived repository-local candidate artifacts."
        ),
    )
    parser.add_argument(
        "--approve-release-overwrite",
        metavar="VERSION",
        help=(
            "Explicitly approve replacing a frozen repository-local candidate ZIP "
            "for exactly "
            "this metadata version. Never use this for normal Unreleased checks."
        ),
    )
    args = parser.parse_args(argv)
    if args.approve_release_overwrite is not None and args.check:
        parser.error("--approve-release-overwrite cannot be used with --check")
    if args.approve_release_overwrite is not None and args.output is not None:
        parser.error("--approve-release-overwrite cannot be used with --output")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    version = load_version()
    if (
        args.approve_release_overwrite is not None
        and args.approve_release_overwrite != version
    ):
        raise ValueError(
            "--approve-release-overwrite must exactly match metadata version "
            f"{version}"
        )
    if args.check:
        return run_check(version, archive_path=args.output)
    return run_build(
        version,
        output_path=args.output,
        approved_version=args.approve_release_overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
