import importlib
import io
import sys
import types
from pathlib import Path

import numpy as np
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


def _jpeg(color, size=(32, 32)):
    array = np.tile(np.array(color, dtype=np.uint8), (size[1], size[0], 1))
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _unit(index, dim=8):
    vector = np.zeros(dim, dtype=np.float32)
    vector[index % dim] = 1.0
    return vector


def _install_style_match_fakes(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.DEFAULT_METADATA_LANGUAGE = "English"
    fake_config.DEFAULT_METADATA_PROVIDER = "ollama"
    fake_config.STYLE_MATCH_CANDIDATE_K = 10
    fake_config.STYLE_MATCH_DEFAULT_WEIGHTS = {"sem": 0.5, "desc": 0.3, "color": 0.1, "loc": 0.1}
    fake_config.STYLE_MATCH_EMBEDDING_MODEL = "clip-vit-b32"
    fake_config.STYLE_MATCH_ENABLED = True
    fake_config.STYLE_MATCH_LOC_TAU_KM = 50.0
    fake_config.STYLE_MATCH_SCORING_MODE = "linear"
    fake_config.logger = _make_logger()

    style_store = {}
    photo_metadata = {}

    fake_postgre = types.ModuleType("service_postgre")

    def fake_upsert_style_ref(**kwargs):
        key = (kwargs["catalog_id"], kwargs["lrc_local_id"])
        style_store[key] = dict(kwargs)

    def fake_query_style_ref_candidates(column, query_embedding, k, filters=None, exclude=None):
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        results = []
        for (catalog_id, lrc_local_id), row in style_store.items():
            if exclude and (catalog_id, lrc_local_id) == tuple(exclude):
                continue
            if filters and filters.get("has_edits") is not None:
                if bool(row.get("has_edits")) != bool(filters["has_edits"]):
                    continue
            vector = row.get(column)
            if vector is None:
                continue
            vector = np.asarray(vector, dtype=np.float32)
            similarity = float(np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector)))
            item = {key: value for key, value in row.items()}
            item["id"] = hash((catalog_id, lrc_local_id))
            item["similarity"] = similarity
            item["sem" if column == "embedding" else "desc"] = similarity
            results.append(item)
        results.sort(key=lambda item: -item["similarity"])
        return results[:k]

    fake_postgre.upsert_style_ref = fake_upsert_style_ref
    fake_postgre.query_style_ref_candidates = fake_query_style_ref_candidates
    fake_postgre.get_image = lambda uuid: photo_metadata.get(uuid)

    fake_lifecycle = types.ModuleType("server_lifecycle")
    fake_lifecycle.embed_image = lambda image_bytes: None
    fake_lifecycle.embed_document = lambda text: None

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre)
    monkeypatch.setitem(sys.modules, "server_lifecycle", fake_lifecycle)
    for module_name in ("service_style_match", "style_match.colors", "style_match", "style_match.scoring", "style_match.whitelist"):
        sys.modules.pop(module_name, None)

    return importlib.import_module("service_style_match"), fake_lifecycle, fake_postgre, style_store, photo_metadata


def _ref_meta(lrc_id, uuid, has_edits=True, develop_settings=None, latitude=None):
    return {
        "uuid": uuid,
        "filename": f"IMG_{lrc_id}.jpg",
        "catalog_id": "test-catalog",
        "lrc_local_id": lrc_id,
        "capture_time": "2024-05-12T19:34:00+02:00",
        "latitude": latitude,
        "longitude": None,
        "has_edits": has_edits,
        "develop_settings": develop_settings or {"Contrast2012": 25, "Exposure2012": -0.65, "LocalAdjustments": [{"x": 1}]},
        "description": None,
    }


def test_index_photo_upserts_with_stripped_masks_and_signals(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(1)
    fake_lifecycle.embed_document = lambda text: _unit(2)

    meta = _ref_meta(1, "uuid-1", has_edits=True)
    result = service.index_photo(_jpeg((200, 100, 20)), meta, neutral_image_bytes=_jpeg((10, 10, 10)))

    assert result["status"] == "processed"
    row = style_store[("test-catalog", 1)]
    assert row["embedding"] is not None
    assert row["color_histogram"] is not None
    # Scene keys stay in the snapshot (opt-in transfer); only local keys are
    # stripped server-side (plan §2.9).
    assert row["develop_settings"] == {"Contrast2012": 25, "Exposure2012": -0.65}
    assert result["local_adjustments"] == 1
    assert result["has_color_histogram"] is True


def test_index_photo_without_neutral_leaves_histogram_null(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(1)

    result = service.index_photo(_jpeg((200, 100, 20)), _ref_meta(2, "uuid-2"))
    assert style_store[("test-catalog", 2)]["color_histogram"] is None
    assert result["has_color_histogram"] is False


def test_index_photo_derives_histogram_from_uploaded_original(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(1)

    # A fake rawpy module returns an embedded JPEG for the uploaded original.
    import types as _types

    embedded = _jpeg((10, 10, 10))
    fake_format = _types.SimpleNamespace(name="JPEG")

    class FakeRaw:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_thumb(self):
            return _types.SimpleNamespace(data=embedded, format=fake_format)

    fake_rawpy = _types.ModuleType("rawpy")
    fake_rawpy.imread = lambda stream: FakeRaw()
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)

    result = service.index_photo(
        _jpeg((200, 100, 20)),
        _ref_meta(3, "uuid-3"),
        original_bytes=b"fake raw",
        original_filename="IMG_3.CR2",
    )
    row = style_store[("test-catalog", 3)]
    assert result["has_color_histogram"] is True
    assert row["color_histogram"] is not None
    assert row["render_source"] == "camera_embedded"


def test_description_embedding_reused_from_photo_metadata(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(1)
    photo_metadata["uuid-1"] = {
        "ids": ["uuid-1"],
        "embeddings": [_unit(3)],
        "metadatas": [{}],
    }

    service.index_photo(_jpeg((200, 100, 20)), _ref_meta(1, "uuid-1"))
    # Reused prose embedding lands in description_embedding untouched.
    assert np.allclose(np.asarray(style_store[("test-catalog", 1)]["description_embedding"]), _unit(3))


def test_search_ranks_edited_only_by_default_and_excludes_self(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(1)
    fake_lifecycle.embed_document = lambda text: _unit(2)

    service.index_photo(_jpeg((200, 100, 20)), _ref_meta(1, "uuid-1", has_edits=True))
    service.index_photo(
        _jpeg((200, 100, 20)), _ref_meta(2, "uuid-2", has_edits=True), neutral_image_bytes=_jpeg((10, 10, 10))
    )
    service.index_photo(_jpeg((200, 100, 20)), _ref_meta(3, "uuid-3", has_edits=False))

    # Self-exclusion: the query IS reference 1.
    response = service.search_photos(
        _jpeg((200, 100, 20)),
        neutral_image_bytes=_jpeg((10, 10, 10)),
        exclude=("test-catalog", 1),
        top_k=12,
    )
    returned_ids = {match["lrc_local_id"] for match in response["matches"]}
    assert returned_ids == {2}  # 1 excluded (self), 3 hidden (unedited default)
    assert response["query"]["description_used"] is False
    assert "color" in response["signals_used"]

    # Toggle brings the unedited reference back.
    response_all = service.search_photos(
        _jpeg((200, 100, 20)),
        filters={"has_edits": False},
        exclude=("test-catalog", 1),
    )
    returned_ids_all = {match["lrc_local_id"] for match in response_all["matches"]}
    assert 3 in returned_ids_all


def test_search_renormalizes_when_query_has_no_neutral_render(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(1)

    service.index_photo(_jpeg((200, 100, 20)), _ref_meta(2, "uuid-2", has_edits=True))
    response = service.search_photos(_jpeg((200, 100, 20)), exclude=("test-catalog", 1))
    match = response["matches"][0]
    assert match["signals"]["color"] is None
    assert "color" not in response["signals_used"]
    assert response["signals_used"] == ["sem"]


def test_search_uses_description_signal_when_provided(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(1)
    fake_lifecycle.embed_document = lambda text: _unit(2)

    service.index_photo(
        _jpeg((200, 100, 20)),
        {**_ref_meta(2, "uuid-2", has_edits=True), "description": "misty forest at dawn"},
    )
    response = service.search_photos(
        _jpeg((200, 100, 20)),
        description="misty forest at dawn",
        exclude=("test-catalog", 1),
    )
    assert response["query"]["description_used"] is True
    assert "desc" in response["signals_used"]
    assert response["matches"][0]["signals"]["desc"] is not None


def test_rrf_scoring_mode_ranking(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(1)

    service.index_photo(_jpeg((200, 100, 20)), _ref_meta(2, "uuid-2", has_edits=True))
    response = service.search_photos(
        _jpeg((200, 100, 20)),
        exclude=("test-catalog", 1),
        scoring_mode="rrf",
    )
    assert response["scoring_mode"] == "rrf"
    assert response["matches"], "RRF mode must still return matches"


def test_incremental_style_index_mirrors_new_photo(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_lifecycle.embed_image = lambda image_bytes: _unit(5)

    result = service.incremental_style_index(
        _jpeg((1, 2, 3)),
        "uuid-inc",
        "IMG_INC.jpg",
        {"caption": "a beach at dusk"},
        options={"catalog_id": "cat", "lrc_local_id": 99},
        prose_embedding=_unit(6),
    )
    assert result["status"] == "processed"
    row = style_store[("cat", 99)]
    assert row["has_edits"] is False
    assert row["description"] == "a beach at dusk"
    assert np.allclose(np.asarray(row["description_embedding"]), _unit(6))

    # Without a provided lrc_local_id a deterministic synthetic id is derived.
    service.incremental_style_index(_jpeg((1, 2, 3)), "uuid-synth", "X.jpg", {}, options={"catalog_id": "cat"})
    assert any(key[0] == "cat" and key != ("cat", 99) for key in style_store)


def test_incremental_style_index_disabled(monkeypatch):
    service, fake_lifecycle, fake_postgre, style_store, photo_metadata = _install_style_match_fakes(monkeypatch)
    fake_config = sys.modules["config"]
    fake_config.STYLE_MATCH_ENABLED = False
    try:
        result = service.incremental_style_index(_jpeg((1, 2, 3)), "uuid-x", "X.jpg", {}, options={})
        assert result is None
        assert style_store == {}
    finally:
        fake_config.STYLE_MATCH_ENABLED = True
