"""Rebuild the draft ordered references for the Korean calibration sheets.

This is a drafting aid, kept for provenance. It is not ArchaeoTrace code and
it is not imported by the plugin, the benchmark runner or the tests.

How the drafts were made
------------------------
A person picked waypoints by eye on enlarged views of each immutable crop
(``waypoints.json``). This script then joins consecutive waypoints with a
minimum-cost path over a colour mask of the target ink, so the line sits on
the printed ink instead of on straight segments between clicks:

* ``brown``/``faint`` follow contour ink, ``blue`` stream ink, ``grey`` black
  line work and ``water`` the filled double-line river band (centre favoured).
  Black grid lines, black text and the red overprint are never in a brown or
  blue mask.
* A waypoint marked ``"gap"`` is joined to the previous one with a straight
  segment. It is used where the line is hidden under a label.
* ``row-centroid`` takes, for each image row, the centroid of red-minus-green
  ink within 2 px of the guide; it is used where a pale contour runs against
  a grid line and the colour mask breaks up.

The joined line is smoothed (Gaussian, sigma 2 px) and resampled every 3 px.

Because the joins are minimum-cost paths, these drafts share an assumption
with Live-Wire style tracers. They are drafts for independent human review
and adjudication, and must not be used to publish rankings before that.

Requires numpy, scipy, scikit-image and Pillow.

Usage (from the repository root)::

    python3 benchmarks/data/public-8x6-template/drafting/draft_references.py OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv
from skimage.graph import route_through_array

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent / "images"


def ink_masks(rgb):
    im = rgb.astype(float) / 255.0
    h, s, v = np.moveaxis(rgb2hsv(im), -1, 0)
    r, g, _ = np.moveaxis(im, -1, 0)
    red = (s > 0.55) & (r > 0.75) & (g < 0.45)
    brownish = (h < 0.11) | (h > 0.97)
    return {
        "brown": brownish & (s > 0.28) & (v > 0.25) & (v < 0.9) & (r > g + 0.08) & ~red,
        "faint": brownish & (s > 0.16) & (v < 0.9) & (r > g + 0.05) & ~red,
        "blue": (h > 0.5) & (h < 0.7) & (s > 0.25) & (v < 0.85),
        "grey": (v < 0.6) & (s < 0.45),
        "water": (h > 0.35) & (h < 0.68) & (s > 0.12) & (v < 0.82),
    }


def smooth_resample(xy, sigma=2.0, step=3.0):
    xy = ndimage.gaussian_filter1d(np.asarray(xy, float), sigma, axis=0, mode="nearest")
    t = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    n = max(2, int(t[-1] // step) + 1)
    tt = np.linspace(0.0, t[-1], n)
    return np.stack([np.interp(tt, t, xy[:, 0]), np.interp(tt, t, xy[:, 1])], 1)


def min_cost_path(rgb, colour, via, exclude=()):
    mask = ink_masks(rgb)[colour]
    for x0, y0, x1, y1 in exclude:
        mask[y0:y1, x0:x1] = False
    if colour == "water":
        mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((3, 3)), iterations=2))
        dist = ndimage.distance_transform_edt(mask)
        cost = np.where(mask, 1.0 + 40.0 / (1.0 + dist) ** 2, 60.0)
    else:
        mask = ndimage.binary_closing(mask, np.ones((2, 2)))
        dist = ndimage.distance_transform_edt(mask)
        cost = np.where(mask, 1.0 + 3.0 / (1.0 + dist), 30.0)
    ys, xs = np.nonzero(mask)

    def snap(point):
        d2 = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
        k = int(np.argmin(d2))
        if d2[k] <= 49:
            return int(ys[k]), int(xs[k])
        return int(point[1]), int(point[0])

    nodes = [snap(p) for p in via]
    path = []
    for i in range(1, len(nodes)):
        a, b = nodes[i - 1], nodes[i]
        if len(via[i]) > 2 and via[i][2] == "gap":
            n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1])) + 1)
            seg = [(a[0] + (b[0] - a[0]) * k / (n - 1), a[1] + (b[1] - a[1]) * k / (n - 1)) for k in range(n)]
        else:
            seg, _ = route_through_array(cost, a, b, fully_connected=True, geometric=True)
        path.extend(seg if not path else seg[1:])
    return smooth_resample([(c, r) for r, c in path])


def row_centroid(rgb, guide):
    rg = rgb[..., 0].astype(float) - rgb[..., 1].astype(float)
    gy = [p[1] for p in guide]
    gx = [p[0] for p in guide]
    points = []
    for y in range(int(gy[0]), int(gy[-1]) + 1):
        g = float(np.interp(y, gy, gx))
        lo, hi = int(round(g)) - 2, int(round(g)) + 3
        weight = np.clip(rg[y, lo:hi] - 8.0, 0.0, None)
        x = lo + float((weight * np.arange(hi - lo)).sum() / weight.sum()) if weight.sum() > 15 else g
        points.append((x, y))
    xy = np.array(points)
    xy[:, 0] = ndimage.gaussian_filter1d(xy[:, 0], 3, mode="nearest")
    return xy[::3]


def build():
    specs = json.loads((HERE / "waypoints.json").read_text(encoding="utf-8"))["crops"]
    out = {}
    for crop_id, spec in specs.items():
        rgb = np.asarray(Image.open(IMAGES / f"{crop_id}.png").convert("RGB"))
        if spec["method"] == "row-centroid":
            xy = row_centroid(rgb, spec["waypoints"])
        else:
            xy = min_cost_path(rgb, spec["ink"], spec["waypoints"], spec.get("exclude", []))
        out[crop_id] = [[round(float(x), 1), round(float(y), 1)] for x, y in xy]
    return out


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
