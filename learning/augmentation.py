"""Train-only geometric and photometric augmentation with explicit RNG."""

import hashlib

# Ink colours for contour recolouring: black/grey monochrome print, blue
# reproductions and the brown of colour editions.
INK_PALETTE = ((30, 30, 32), (85, 85, 88), (40, 70, 165), (150, 88, 50))
LABEL_TEXT = "0123456789"
WINDOW_RADIUS = 32


def sample_rng(seed, epoch, sample_id):
    import numpy as np

    key = "{}:{}:{}".format(int(seed), int(epoch), sample_id).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def _paper_rgb(image):
    import numpy as np

    luminance = image.mean(axis=2)
    bright = luminance >= np.percentile(luminance, 85)
    return image[bright].mean(axis=0)


def recolor_contour_ink(result, rng):
    """Repaint labelled contour ink so colour alone cannot identify contours.

    Only pixels near the contour target change; other ink keeps its colour,
    so a black contour must be told apart from black roads by shape.
    """
    import numpy as np

    image = result["image"].astype(np.float32)
    near = result["centerline"] >= 0.25
    if near.sum() < 16:
        return result
    luminance = image.mean(axis=2)
    paper = _paper_rgb(image)
    paper_luminance = float(paper.mean())
    darkest = float(np.percentile(luminance[near], 5))
    strength = np.clip((paper_luminance - luminance) / max(paper_luminance - darkest, 1.0), 0.0, 1.0)
    ink = np.asarray(INK_PALETTE[int(rng.integers(0, len(INK_PALETTE)))], np.float32)
    ink = np.clip(ink + rng.normal(0.0, 12.0, 3), 0.0, 255.0)
    weight = strength[near][:, None]
    image[near] = paper * (1.0 - weight) + ink * weight
    result["image"] = np.rint(image).astype(np.uint8)
    return result


def _label_glyph(text, size, angle, stroke=0):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=size)
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    canvas = Image.new("L", (right - left + 4, bottom - top + 4), 0)
    ImageDraw.Draw(canvas).text((2 - left, 2 - top), text, fill=255, font=font,
                                stroke_width=stroke, stroke_fill=255)
    width = canvas.width
    rotated = canvas.rotate(angle, resample=Image.BILINEAR, expand=True)
    # Printed numerals are solid; thin antialiased strokes would read as faint.
    return np.clip(np.asarray(rotated, np.float32) / 160.0, 0.0, 1.0), width


def _dilate(mask):
    import numpy as np

    padded = np.pad(mask, 1)
    height, width = mask.shape
    result = np.zeros_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            result |= padded[dy:dy + height, dx:dx + width]
    return result


def _components(mask):
    """8-connected components of a small window, without SciPy."""
    import numpy as np

    labels = np.zeros(mask.shape, np.int32)
    count = 0
    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        count += 1
        labels[start] = count
        stack = [start]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                            and mask[ny, nx] and not labels[ny, nx]):
                        labels[ny, nx] = count
                        stack.append((ny, nx))
    return labels, count


def synthetic_label_gaps(result, rng, *, max_labels=3):
    """Cut labelled contours with elevation-style numbers and keep the bridge.

    Printed maps interrupt a contour where its elevation is written. The
    contour target is left continuous through the cut, so the model learns to
    join it; glyph ink becomes non-contour except inside the bridge corridor.
    """
    import numpy as np

    centerline = result["centerline"]
    valid = result["valid_mask"]
    height, width = centerline.shape
    r = WINDOW_RADIUS
    candidates = np.argwhere((centerline >= 0.97) & valid)
    candidates = candidates[(candidates[:, 0] >= r) & (candidates[:, 0] < height - r)
                            & (candidates[:, 1] >= r) & (candidates[:, 1] < width - r)]
    count = int(rng.integers(0, max_labels + 1))
    if not len(candidates) or not count:
        return result
    image = result["image"].astype(np.float32)
    other = result["other_ink"].copy()
    paper = _paper_rgb(image)
    grid_y, grid_x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    distance = np.hypot(grid_y, grid_x)
    size = 2 * r + 1
    for _ in range(count):
        y, x = (int(v) for v in candidates[int(rng.integers(0, len(candidates)))])
        window = np.s_[y - r:y + r + 1, x - r:x + r + 1]
        line = centerline[window]
        text = "".join(LABEL_TEXT[int(rng.integers(0, 10))] for _ in range(int(rng.integers(2, 4))))
        # The 1.5px band of this one contour; dense neighbours stay separate.
        band, _ = _components(line >= 0.6)
        band = band == band[r, r]
        nearby = np.argwhere(band & (line >= 0.9) & (distance <= 8)).astype(np.float32)
        if len(nearby) < 6:
            continue
        direction = np.linalg.svd(nearby - nearby.mean(0), full_matrices=False)[2][0]
        angle = float(np.degrees(np.arctan2(-direction[0], direction[1])))
        if rng.integers(0, 2):
            angle += 180.0
        glyph, text_length = _label_glyph(text, int(rng.integers(11, 20)), angle, int(rng.integers(0, 2)))
        half = text_length / 2.0 + float(rng.uniform(1.5, 4.0))
        gh, gw = glyph.shape
        if gh > size or gw > size or half > r - 4:
            continue
        gap = band & (distance <= half)
        # Cut only where the labelled contour passes through the whole cut;
        # a contour ending inside it would teach an unobserved continuation.
        through = band & (line >= 0.9) & (distance > half) & (distance <= half + 3)
        sides_count = _components(through)[1]
        if sides_count < 2 or (gap & (line >= 0.9)).sum() < 1.5 * half:
            continue
        top, left = r - gh // 2, r - gw // 2
        alpha = np.zeros((size, size), np.float32)
        alpha[top:top + gh, left:left + gw] = glyph
        ink = image[window][line >= 0.6]
        colour = ink[np.argsort(ink.mean(axis=1))[:max(1, len(ink) // 10)]].mean(axis=0)
        patch = image[window]
        patch[_dilate(gap) & (line >= 0.3)] = paper
        patch[:] = patch * (1.0 - alpha[..., None]) + colour * alpha[..., None]
        image[window] = patch
        glyph_ink = (alpha >= 0.3) & (line < 0.3)
        other[window] = np.where(glyph_ink, 1.0, other[window])
    result["image"] = np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    result["other_ink"] = (other * valid).astype(result["other_ink"].dtype)
    return result


def augment_sample(sample, rng, *, ink_recolor=False, label_gaps=False):
    """Return a new sample; all labels/ignore pixels share geometry exactly."""
    import numpy as np

    result = dict(sample)
    turns = int(rng.integers(0, 4))
    flip_x, flip_y = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
    for key in ("image", "centerline", "other_ink", "valid_mask"):
        value = np.rot90(np.asarray(sample[key]), turns, axes=(0, 1))
        if flip_x:
            value = np.flip(value, axis=1)
        if flip_y:
            value = np.flip(value, axis=0)
        result[key] = np.ascontiguousarray(value)
    if label_gaps:
        result = synthetic_label_gaps(result, rng)
    if ink_recolor:
        if rng.uniform() < 0.5:
            result = recolor_contour_ink(result, rng)
        if rng.uniform() < 0.15:
            grey = result["image"].mean(axis=2, keepdims=True)
            result["image"] = np.rint(np.repeat(grey, 3, axis=2)).astype(np.uint8)
    image = result["image"].astype(np.float32) / 255.0
    contrast = float(rng.uniform(0.85, 1.15))
    brightness = float(rng.uniform(-0.08, 0.08))
    noise = rng.normal(0.0, float(rng.uniform(0.0, 0.015)), image.shape).astype(np.float32)
    image = np.clip((image - 0.5) * contrast + 0.5 + brightness + noise, 0.0, 1.0)
    result["image"] = np.rint(image * 255.0).astype(np.uint8)
    return result
