#!/usr/bin/env python3
"""Compatibility entry point for the canonical release packager.

New automation should call ``scripts/package_release.py`` directly. This wrapper
intentionally produces the same metadata-derived, reproducible artifact.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from scripts.package_release import main as release_main


def create_zip(argv: Optional[Sequence[str]] = None) -> int:
    print(
        "package_plugin.py is deprecated; delegating to scripts/package_release.py",
        file=sys.stderr,
    )
    return release_main(argv)


if __name__ == "__main__":
    raise SystemExit(create_zip())
