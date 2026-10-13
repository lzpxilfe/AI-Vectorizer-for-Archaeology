"""Train-only geometric and photometric augmentation with explicit RNG."""

import hashlib


def sample_rng(seed, epoch, sample_id):
    import numpy as np

    key = "{}:{}:{}".format(int(seed), int(epoch), sample_id).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def augment_sample(sample, rng):
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
    image = result["image"].astype(np.float32) / 255.0
    contrast = float(rng.uniform(0.85, 1.15))
    brightness = float(rng.uniform(-0.08, 0.08))
    noise = rng.normal(0.0, float(rng.uniform(0.0, 0.015)), image.shape).astype(np.float32)
    image = np.clip((image - 0.5) * contrast + 0.5 + brightness + noise, 0.0, 1.0)
    result["image"] = np.rint(image * 255.0).astype(np.uint8)
    return result

