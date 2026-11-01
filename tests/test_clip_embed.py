"""Real CLIP embedding gate for server_lifecycle.embed_image (T1.3).

Downloads/loads openai/clip-vit-base-patch32 through transformers — the same
path the server uses. Skipped unless STYLE_MATCH_CLIP_TEST=1 is set so the
default suite stays fast and offline.
"""

import importlib
import io
import os
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


def _make_logger():
    return types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )


def _install_real_lifecycle(monkeypatch):
    if os.environ.get("STYLE_MATCH_CLIP_TEST") != "1":
        pytest.skip("set STYLE_MATCH_CLIP_TEST=1 to run the real CLIP gate")

    fake_config = types.ModuleType("config")
    fake_config.CLIP_IMAGE_SIZE = 224
    fake_config.CLIP_MODEL_CACHE_PATH = os.environ.get("GENIUSAI_MODEL_CACHE_PATH")
    fake_config.CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    fake_config.EMBEDDING_MODEL_KEY = "qwen3-0.6b"
    fake_config.FETCH_MODELS = True
    fake_config.MODEL_CACHE_PATH = None
    fake_config.OK_FILE_PATH = "/tmp/clip-test.OK"
    fake_config.PID_FILE_PATH = "/tmp/clip-test.pid"
    fake_config.TEXT_EMBEDDING_MAX_LENGTH = 512
    fake_config.TEXT_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
    fake_config.TEXT_EMBEDDING_POOLING = "last_token"
    fake_config.TEXT_EMBEDDING_QUERY_INSTRUCTION = None
    fake_config.TORCH_DEVICE = "cpu"
    fake_config.logger = _make_logger()

    monkeypatch.setitem(sys.modules, "config", fake_config)
    sys.modules.pop("server_lifecycle", None)
    return importlib.import_module("server_lifecycle")


def _jpeg(color, size=(256, 256)):
    array = np.tile(np.array(color, dtype=np.uint8), (size[1], size[0], 1))
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def test_embed_image_is_deterministic_and_normalized(monkeypatch):
    server_lifecycle = _install_real_lifecycle(monkeypatch)
    try:
        data = _jpeg((30, 120, 220))
        vector_a = server_lifecycle.embed_image(data)
        assert vector_a is not None
        vector_a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
        assert vector_a.shape[0] == 512

        vector_b = np.asarray(
            server_lifecycle.embed_image(data), dtype=np.float32
        ).reshape(-1)
        cosine = float(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
        assert cosine >= 0.999
        # L2-normalized output.
        assert abs(float(np.linalg.norm(vector_a)) - 1.0) < 1e-4

        different = np.asarray(
            server_lifecycle.embed_image(_jpeg((220, 140, 30))), dtype=np.float32
        ).reshape(-1)
        cosine_diff = float(np.dot(vector_a, different) / (np.linalg.norm(vector_a) * np.linalg.norm(different)))
        assert cosine_diff < cosine
    finally:
        server_lifecycle.unload_clip_model()
