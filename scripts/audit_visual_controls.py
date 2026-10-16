#!/usr/bin/env python3
"""Flag visible diagnostic controls which are not on dark source pixels.

This is source-data QA, not a model, skeleton or contour-identity validator.
Passing it does not prove that two arms belong to the same contour. It reads
only frozen native raster pixels and explicit visible controls, never paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def audit_case(image, case):
    """Return raw-pixel checks for prompts and manually marked visible ink."""
    rgb = np.asarray(image)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Control QA requires original Byte RGB pixels")
    gray = rgb.astype(np.float64).mean(axis=2)
    points = [(float(col), float(row)) for row, col in case["prompt_rc"]]
    controls = case.get("visible_control_segments_xy")
    if controls is None:
        controls = [[(col, row) for row, col in segment]
                    for segment in case.get("visible_controls_rc", [])]
    points.extend(tuple(point) for segment in controls for point in segment)
    points.extend(tuple(point) for segment in case.get("foreign_visible_control_segments_xy", [])
                  for point in segment)
    checks = []
    for x, y in sorted(set(points)):
        if not np.isfinite((x, y)).all() or not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]):
            raise ValueError("Visible control is outside the native image")
        cx, cy = int(round(x)), int(round(y))
        if cx >= gray.shape[1] or cy >= gray.shape[0]:
            raise ValueError("Visible control rounds outside the native image")
        local = gray[max(0, cy - 7):cy + 8, max(0, cx - 7):cx + 8]
        paper = float(np.percentile(local, 90))
        value = float(gray[cy, cx])
        contrast = paper - value
        checks.append({
            "xy": [x, y], "gray": round(value, 3),
            "local_paper_gray": round(paper, 3),
            "contrast": round(contrast, 3),
            "dark_source_pixel": contrast >= max(32.0, paper * 0.20),
        })
    return {"id": case["id"], "checks": checks,
            "all_controls_on_dark_pixels": bool(checks) and all(c["dark_source_pixel"] for c in checks),
            "contour_identity_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    samples = {sample["id"]: sample for sample in manifest["samples"]}
    result = {"scope": "Raw visible-control QA only; no path/model inputs or contour-identity proof",
              "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(), "cases": []}
    for case in fixtures["cases"]:
        sample = samples[case["sample_id"]]
        if sample["split"] != "validation":
            raise ValueError("Only development validation crops may be opened")
        path = args.dataset / sample["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError("Frozen source pixels changed")
        result["cases"].append(audit_case(Image.open(path).convert("RGB"), case))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(result["cases"]),
                      "requiring_raw_review": [c["id"] for c in result["cases"]
                                               if not c["all_controls_on_dark_pixels"]]}))


if __name__ == "__main__":
    main()
