#!/usr/bin/env python3
"""Build and verify the packaged ArchaeoTrace plugin release."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR_NAME = "ai_vectorizer"
PLUGIN_DIR = ROOT / PLUGIN_DIR_NAME
DIST_DIR = ROOT / "dist"
TOP_LEVEL_ITEMS = (
    "__init__.py",
    "plugin.py",
    "config.py",
    "metadata.txt",
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


def load_version() -> str:
    parser = configparser.ConfigParser(interpolation=None)
    with (PLUGIN_DIR / "metadata.txt").open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser["general"]["version"].strip()


def release_dir(version: str) -> Path:
    return ROOT / f"{PLUGIN_DIR_NAME} {version}"


def zip_path(version: str) -> Path:
    return DIST_DIR / f"{PLUGIN_DIR_NAME}-{version}.zip"


def should_skip(path: Path) -> bool:
    if path.name in IGNORED_NAMES or path.suffix in IGNORED_SUFFIXES:
        return True

    if path.suffix.lower() in IGNORED_WEIGHT_SUFFIXES and path.parent.name == "models":
        return True

    return False


def iter_source_files() -> list[tuple[Path, Path]]:
    files: list[tuple[Path, Path]] = []
    for item_name in TOP_LEVEL_ITEMS:
        src = PLUGIN_DIR / item_name
        if not src.exists():
            raise FileNotFoundError(f"Missing required plugin item: {src}")
        if src.is_file():
            files.append((src, Path(item_name)))
            continue
        for child in sorted(src.rglob("*")):
            if child.is_dir():
                continue
            if any(should_skip(parent) for parent in child.relative_to(PLUGIN_DIR).parents):
                continue
            if should_skip(child):
                continue
            files.append((child, child.relative_to(PLUGIN_DIR)))
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
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for src_path, rel_path in iter_source_files():
        destination = target_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, destination)

    return target_dir


def build_release_zip(version: str) -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    target_zip = zip_path(version)
    if target_zip.exists():
        target_zip.unlink()

    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for src_path, rel_path in iter_source_files():
            archive.write(src_path, arcname=(Path(PLUGIN_DIR_NAME) / rel_path).as_posix())

    return target_zip


def release_manifest(version: str) -> dict[str, str]:
    target_dir = release_dir(version)
    if not target_dir.exists():
        raise FileNotFoundError(f"Release directory does not exist: {target_dir}")

    manifest: dict[str, str] = {}
    for path in sorted(target_dir.rglob("*")):
        if path.is_dir():
            continue
        if should_skip(path):
            continue
        manifest[path.relative_to(target_dir).as_posix()] = file_hash(path)
    return manifest


def zip_manifest(version: str) -> dict[str, str]:
    archive_path = zip_path(version)
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


def run_check(version: str) -> int:
    expected = source_manifest()
    problems: list[str] = []

    try:
        problems.extend(compare_manifests("release dir", expected, release_manifest(version)))
    except Exception as exc:
        problems.append(str(exc))

    try:
        problems.extend(compare_manifests("release zip", expected, zip_manifest(version)))
    except Exception as exc:
        problems.append(str(exc))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1

    print(f"Release artifacts are in sync for {version}.")
    return 0


def run_build(version: str) -> int:
    target_dir = build_release_tree(version)
    target_zip = build_release_zip(version)
    print(f"Built release directory: {target_dir}")
    print(f"Built release zip: {target_zip}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the packaged ArchaeoTrace release artifacts.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the generated release directory and zip match the root source tree.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    version = load_version()
    if args.check:
        return run_check(version)
    return run_build(version)


if __name__ == "__main__":
    raise SystemExit(main())
