import os
import zipfile
from pathlib import Path

VERSION = "0.1.2"
PLUGIN_DIRNAME = "ai_vectorizer"
MAX_UPLOAD_MB = 25.0
OUTPUT_NAME = f"ArchaeoTrace-v{VERSION}-qgis.zip"

EXCLUDED_DIRS = {"__pycache__", ".git", ".idea", ".vscode"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_FILENAMES = {
    "hed_pretrained_bsds.caffemodel",
    "mobile_sam.pt",
    "mobile_sam.meta.json",
}
EXCLUDED_WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx", ".ckpt", ".bin", ".caffemodel"}


def should_skip(path: Path) -> bool:
    name = path.name
    suffix = path.suffix.lower()

    if name in EXCLUDED_FILENAMES:
        return True

    if suffix in EXCLUDED_SUFFIXES:
        return True

    # Avoid bundling large runtime model weights in plugin upload ZIP.
    if suffix in EXCLUDED_WEIGHT_SUFFIXES and path.parent.name in {"models"}:
        return True

    return False


def create_zip() -> int:
    repo_root = Path(__file__).resolve().parent
    plugin_dir = repo_root / PLUGIN_DIRNAME

    if not plugin_dir.exists() or not plugin_dir.is_dir():
        print(f"ERROR: Plugin directory not found: {plugin_dir}")
        return 1

    desktop = Path.home() / "Desktop"
    zip_path = desktop / OUTPUT_NAME

    if zip_path.exists():
        zip_path.unlink()

    print(f"Creating ZIP: {zip_path}")
    print(f"Source dir:  {plugin_dir}")

    file_count = 0
    skipped = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(plugin_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            root_path = Path(root)
            for filename in files:
                abs_path = root_path / filename

                if should_skip(abs_path):
                    skipped += 1
                    continue

                rel_path = abs_path.relative_to(repo_root)
                zipf.write(abs_path, rel_path.as_posix())
                file_count += 1

        readme_path = repo_root / "README.md"
        if readme_path.exists():
            zipf.write(readme_path, f"{PLUGIN_DIRNAME}/README.md")
            file_count += 1

    final_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Files added: {file_count}")
    print(f"Files skipped: {skipped}")
    print(f"ZIP size: {final_size_mb:.2f} MB")

    if final_size_mb > MAX_UPLOAD_MB:
        print(f"FAIL: ZIP exceeds {MAX_UPLOAD_MB:.0f} MB QGIS upload limit")
        return 2

    print(f"OK: ZIP is within {MAX_UPLOAD_MB:.0f} MB upload limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(create_zip())
