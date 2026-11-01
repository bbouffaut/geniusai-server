"""Neutral-render extraction on the server (plan §2.4).

The plugin cannot read a RAW's embedded JPEG from Lua, so the only way to keep
a single ingestion path is to let the plugin upload the *original* file and
extract the neutral render here. Extraction is cheap: rawpy reads only the
embedded thumbnail of the RAW file (no demosaic, tens of ms, a few MB of RAM).

Precedence for the HSV histogram source:
  1. an explicit `neutral_image` upload (neutral export from the client);
  2. here: the original file — RAW embedded preview ('camera_embedded') or the
     JPEG original itself ('original');
  3. nothing -> colour signal NULL, weights renormalized, backfill later.
"""

import io

JPEG_SUFFIXES = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".tif", ".tiff"}


def _suffix(filename):
    name = str(filename or "").lower()
    index = name.rfind(".")
    return name[index:] if index >= 0 else ""


def _thumb_to_jpeg(thumb):
    """Normalize a rawpy Thumbnail (JPEG bytes or bitmap array) to JPEG bytes."""
    data = getattr(thumb, "data", None)
    if data is None:
        return None

    raw_format = getattr(thumb, "format", None)
    format_name = getattr(raw_format, "name", str(raw_format))
    if format_name == "JPEG":
        return bytes(data)

    # Bitmap thumbnails come back as a numpy array; encode for the histogram.
    try:
        from PIL import Image
        import numpy as np

        array = np.asarray(data)
        array = array[:, :, :3] if array.ndim == 3 and array.shape[2] >= 3 else array
        buffer = io.BytesIO()
        Image.fromarray(array.astype("uint8"), mode="RGB").save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()
    except Exception:
        return None


def neutral_render_from_original(original_bytes, filename=""):
    """Return (image_bytes, render_source) for the neutral render, or (None, None).

    render_source is 'camera_embedded' for a RAW embedded preview and
    'original' when the uploaded file already is a rendered image.
    """
    if not original_bytes:
        return None, None

    suffix = _suffix(filename)
    if suffix in JPEG_SUFFIXES:
        return original_bytes, "original"

    try:
        import rawpy
    except ImportError:
        return None, None

    try:
        with rawpy.imread(io.BytesIO(original_bytes)) as raw:
            thumb = raw.extract_thumb()
    except Exception:
        return None, None

    preview = _thumb_to_jpeg(thumb)
    if not preview:
        return None, None
    return preview, "camera_embedded"
