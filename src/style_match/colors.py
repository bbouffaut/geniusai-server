"""HSV colour histogram (16x8x8 = 1024 bins) computed on NEUTRAL renders.

Histograms must be computed on the same render basis on both sides (plan
§2.4): the neutral/camera rendering, never the edited look. A photo whose
neutral render is unavailable simply has no colour signal (weight
renormalized) and is backfilled by the next bootstrap run.
"""

import io

import numpy as np
from PIL import Image

H_BINS = 16
S_BINS = 8
V_BINS = 8
HISTOGRAM_SIZE = H_BINS * S_BINS * V_BINS

_HISTOGRAM_MAX_DIMENSION = 256  # downscale before binning (speed, tiny effect)


def compute_color_histogram(image_bytes):
    """Return the L1-normalized flattened HSV histogram, or None on failure."""
    if not image_bytes:
        return None
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        width, height = image.size
        scale = max(width, height) / float(_HISTOGRAM_MAX_DIMENSION)
        if scale > 1:
            resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            image = image.resize(
                (max(1, int(width / scale)), max(1, int(height / scale))),
                resampling,
            )
        hsv = np.asarray(image, dtype=np.float32) / 255.0
        hsv = _rgb_to_hsv(hsv)
        hue = np.clip((hsv[..., 0] * H_BINS).astype(np.int64), 0, H_BINS - 1)
        sat = np.clip((hsv[..., 1] * S_BINS).astype(np.int64), 0, S_BINS - 1)
        val = np.clip((hsv[..., 2] * V_BINS).astype(np.int64), 0, V_BINS - 1)
        flat = (hue * S_BINS + sat) * V_BINS + val
        histogram = np.bincount(flat.ravel(), minlength=HISTOGRAM_SIZE).astype(np.float32)
        total = histogram.sum()
        if total <= 0:
            return None
        return (histogram / total).tolist()
    except Exception:
        return None


def histogram_cosine(a, b):
    """Cosine similarity between two histograms (either may be None/empty)."""
    if not a or not b:
        return None
    vec_a = np.asarray(a, dtype=np.float32)
    vec_b = np.asarray(b, dtype=np.float32)
    if vec_a.shape != vec_b.shape or vec_a.size == 0:
        return None
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= 0 or norm_b <= 0:
        return None
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _rgb_to_hsv(rgb):
    """Vectorized RGB [0,1] -> HSV, hue in [0,1] (matches colorsys semantics)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.max(rgb, axis=-1)
    min_c = np.min(rgb, axis=-1)
    delta = max_c - min_c
    value = max_c
    saturation = np.where(max_c > 0, delta / np.maximum(max_c, 1e-12), 0.0)

    hue = np.zeros_like(max_c)
    mask = delta > 0
    r_mask = mask & (max_c == r)
    g_mask = mask & (max_c == g) & ~r_mask
    b_mask = mask & ~r_mask & ~g_mask
    hue = np.where(
        r_mask,
        ((g - b) / np.maximum(delta, 1e-12)) % 6.0,
        hue,
    )
    hue = np.where(g_mask, (b - r) / np.maximum(delta, 1e-12) + 2.0, hue)
    hue = np.where(b_mask, (r - g) / np.maximum(delta, 1e-12) + 4.0, hue)
    hue = hue / 6.0
    return np.stack([hue % 1.0, saturation, value], axis=-1)
