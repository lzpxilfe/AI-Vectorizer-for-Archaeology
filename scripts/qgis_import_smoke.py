#!/usr/bin/env python3
"""Initialize QGIS and import the complete plugin entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]

from qgis.core import Qgis, QgsApplication  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plugin-root",
        type=Path,
        default=ROOT,
        help="Directory containing the ai_vectorizer package (defaults to the checkout).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plugin_root = args.plugin_root.expanduser().resolve(strict=True)
    expected_package = plugin_root / "ai_vectorizer"
    if not (expected_package / "metadata.txt").is_file():
        raise FileNotFoundError(
            f"Plugin root does not contain ai_vectorizer/metadata.txt: {plugin_root}"
        )
    sys.path.insert(0, str(plugin_root))

    prefix = os.environ.get("QGIS_PREFIX_PATH")
    if prefix:
        QgsApplication.setPrefixPath(prefix, True)

    application = QgsApplication([], False)
    application.initQgis()
    try:
        import ai_vectorizer
        from ai_vectorizer.plugin import AIVectorizer

        loaded_package = Path(ai_vectorizer.__file__).resolve()
        try:
            loaded_package.relative_to(expected_package.resolve(strict=True))
        except ValueError as exc:
            raise RuntimeError(
                "Smoke test imported a different ai_vectorizer package: "
                f"{loaded_package}"
            ) from exc
        plugin = ai_vectorizer.classFactory(None)
        if not isinstance(plugin, AIVectorizer):
            raise TypeError("classFactory did not return AIVectorizer")
        print(f"Imported ArchaeoTrace with QGIS {Qgis.QGIS_VERSION}")
        return 0
    finally:
        application.exitQgis()


if __name__ == "__main__":
    raise SystemExit(main())
