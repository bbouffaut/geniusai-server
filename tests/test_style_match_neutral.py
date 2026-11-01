"""Neutral-render extraction tests (server-side rawpy path, plan §2.4)."""

import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from style_match import colors, neutral


def _jpeg_bytes(color=(30, 120, 220), size=(64, 64)):
    array = np.tile(np.array(color, dtype=np.uint8), (size[1], size[0], 1))
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class _FakeThumb:
    def __init__(self, data, fmt):
        self.data = data
        self.format = fmt


class _FakeRaw:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_thumb(self):
        return self._payload


def test_jpeg_original_is_used_as_neutral_render():
    data = _jpeg_bytes()
    preview, source = neutral.neutral_render_from_original(data, "photo.jpg")
    assert source == "original"
    assert preview == data


def test_raw_embedded_jpeg_is_extracted(monkeypatch):
    embedded = _jpeg_bytes((200, 80, 20))
    fake_format = types.SimpleNamespace(name="JPEG")
    fake_rawpy = types.ModuleType("rawpy")
    fake_rawpy.imread = lambda _stream: _FakeRaw(_FakeThumb(embedded, fake_format))
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    preview, source = neutral.neutral_render_from_original(b"fake raw bytes", "IMG_0001.CR2")
    assert source == "camera_embedded"
    assert preview == embedded
    assert colors.compute_color_histogram(preview) is not None


def test_missing_rawpy_module_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "rawpy", None)  # import raises ImportError
    preview, source = neutral.neutral_render_from_original(b"fake raw bytes", "IMG_0001.NEF")
    assert preview is None
    assert source is None


def test_unparseable_raw_returns_none(monkeypatch):
    def _boom(_stream):
        raise ValueError("not a raw file")

    fake_rawpy = types.ModuleType("rawpy")
    fake_rawpy.imread = _boom
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    assert neutral.neutral_render_from_original(b"junk", "x.ARW") == (None, None)


def test_empty_input_returns_none():
    assert neutral.neutral_render_from_original(b"", "x.CR2") == (None, None)
    assert neutral.neutral_render_from_original(None, "x.CR2") == (None, None)


@pytest.mark.skipif(
    not Path("/tmp/RAW_CANON_40D.CR2").is_file(),
    reason="real RAW fixture not present at /tmp/RAW_CANON_40D.CR2",
)
def test_real_raw_fixture_extracts_embedded_jpeg():
    data = Path("/tmp/RAW_CANON_40D.CR2").read_bytes()
    preview, source = neutral.neutral_render_from_original(data, "RAW_CANON_40D.CR2")
    assert source == "camera_embedded"
    assert preview[:2] == b"\xff\xd8"
    histogram = colors.compute_color_histogram(preview)
    assert histogram is not None
    assert abs(sum(histogram) - 1.0) < 1e-4
