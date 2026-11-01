import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from style_match import colors


def _jpeg(rgb_array):
    image = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _solid_rgb(r, g, b, size=(64, 64)):
    return np.tile(np.array([r, g, b], dtype=np.float32), (size[1], size[0], 1))


def test_histogram_is_normalized_and_sized():
    histogram = colors.compute_color_histogram(_jpeg(_solid_rgb(255, 0, 0)))
    assert histogram is not None
    assert len(histogram) == colors.HISTOGRAM_SIZE
    assert abs(sum(histogram) - 1.0) < 1e-5
    # A pure red image puts (almost) all mass in the first hue ring.
    top_bin = int(np.argmax(histogram))
    assert top_bin < colors.S_BINS * colors.V_BINS  # hue bin 0, any sat/val


def test_identical_images_have_unit_cosine():
    data = _jpeg(_solid_rgb(30, 120, 220))
    histogram_a = colors.compute_color_histogram(data)
    histogram_b = colors.compute_color_histogram(data)
    assert colors.histogram_cosine(histogram_a, histogram_b) > 0.999


def test_different_colours_score_lower():
    blue = colors.compute_color_histogram(_jpeg(_solid_rgb(20, 60, 200)))
    orange = colors.compute_color_histogram(_jpeg(_solid_rgb(220, 140, 30)))
    same = colors.compute_color_histogram(_jpeg(_solid_rgb(20, 60, 200)))
    assert colors.histogram_cosine(blue, orange) < colors.histogram_cosine(blue, same)
    assert colors.histogram_cosine(blue, orange) < 0.9


def test_missing_histograms_return_none_similarity():
    assert colors.histogram_cosine(None, [0.1]) is None
    assert colors.histogram_cosine([0.1], None) is None
    assert colors.histogram_cosine([], [0.1]) is None


def test_invalid_bytes_return_none():
    assert colors.compute_color_histogram(b"not an image") is None
    assert colors.compute_color_histogram(b"") is None
