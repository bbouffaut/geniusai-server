import importlib
import io
import json
import sys
import types
from pathlib import Path

from flask import Flask

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


def _png_bytes():
    return b"\x89PNG-rubber-stamp-bytes"


def _install_routes_fakes(monkeypatch, index_result=None, search_result=None):
    fake_config = types.ModuleType("config")
    fake_config.STYLE_MATCH_DEFAULT_WEIGHTS = {"sem": 0.5, "desc": 0.3, "color": 0.1, "loc": 0.1}
    fake_config.STYLE_MATCH_EMBEDDING_MODEL = "clip-vit-b32"
    fake_config.STYLE_MATCH_MAX_ORIGINAL_BYTES = 200 * 1024 * 1024
    fake_config.STYLE_MATCH_SCORING_MODE = "linear"
    fake_config.logger = _make_logger()

    captured = {}

    fake_service = types.ModuleType("service_style_match")

    def fake_index_photo(image_bytes, meta, neutral_image_bytes=None, original_bytes=None,
                         original_filename=None, generate_description=True):
        captured["index"] = {
            "meta": meta,
            "neutral": neutral_image_bytes,
            "original": original_bytes,
            "original_filename": original_filename,
            "generate": generate_description,
        }
        return index_result or {"status": "processed", "uuid": meta["uuid"]}

    def fake_search_photos(image_bytes, **kwargs):
        captured["search"] = kwargs
        return search_result or {
            "status": "ok",
            "query": {"uuid": kwargs.get("query_uuid"), "description_used": False},
            "signals_used": ["sem"],
            "scoring_mode": "linear",
            "candidate_count": 1,
            "matches": [
                {
                    "rank": 1,
                    "lrc_local_id": 48213,
                    "catalog_id": "my-catalog",
                    "file_name": "IMG_1701.jpg",
                    "score": 0.87,
                    "signals": {"sem": 0.91, "desc": None, "color": None, "loc": None},
                    "capture_time": "2024-05-12T19:34:00+02:00",
                    "distance_km": None,
                    "camera_model": "ILCE-7M4",
                    "has_edits": True,
                    "description": "sunset over the lake",
                    "develop_settings": {"Contrast2012": 25},
                    "local_adjustments": 2,
                }
            ],
        }

    fake_service.index_photo = fake_index_photo
    fake_service.search_photos = fake_search_photos

    fake_postgre = types.ModuleType("service_postgre")
    fake_postgre.style_ref_stats = lambda: {
        "total": 3,
        "edited": 2,
        "with_clip_embedding": 3,
        "with_description_embedding": 2,
        "with_color_histogram": 1,
        "render_source_mix": {"develop_export": 1, "camera_embedded": 0, "original": 2},
        "catalogs": 1,
    }

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre)
    monkeypatch.setitem(sys.modules, "service_style_match", fake_service)
    sys.modules.pop("routes_style_match", None)

    module = importlib.import_module("routes_style_match")
    app = Flask(__name__)
    app.register_blueprint(module.style_match_bp)
    app.config["TESTING"] = True
    return app.test_client(), captured, fake_postgre


def test_index_endpoint_parses_multipart(monkeypatch):
    client, captured, _ = _install_routes_fakes(monkeypatch)

    response = client.post(
        "/style-match/index",
        data={
            "image": (io.BytesIO(_png_bytes()), "current.jpg"),
            "neutral_image": (io.BytesIO(_png_bytes()), "neutral.jpg"),
            "original": (io.BytesIO(b"RAW-BYTES"), "IMG_1.CR2"),
            "uuid": "abc-123",
            "filename": "IMG_1.jpg",
            "capture_time": "2024-05-12T19:34:00+02:00",
            "catalog_id": "my-catalog",
            "lrc_local_id": "48213",
            "develop_settings": json.dumps({"Contrast2012": 25, "LocalAdjustments": [{}]}),
            "has_edits": "true",
            "description": "a beach",
            "generate_description": "false",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "processed"
    meta = captured["index"]["meta"]
    assert meta["catalog_id"] == "my-catalog"
    assert meta["lrc_local_id"] == 48213
    assert meta["has_edits"] is True
    assert meta["develop_settings"] == {"Contrast2012": 25, "LocalAdjustments": [{}]}
    assert captured["index"]["neutral"] == _png_bytes()
    assert captured["index"]["original"] == b"RAW-BYTES"
    assert captured["index"]["generate"] is False


def test_index_endpoint_requires_image_and_keys(monkeypatch):
    client, captured, _ = _install_routes_fakes(monkeypatch)

    missing_image = client.post(
        "/style-match/index",
        data={"uuid": "x", "catalog_id": "c", "lrc_local_id": "1"},
        content_type="multipart/form-data",
    )
    assert missing_image.status_code == 400

    missing_keys = client.post(
        "/style-match/index",
        data={"image": (io.BytesIO(_png_bytes()), "a.jpg")},
        content_type="multipart/form-data",
    )
    assert missing_keys.status_code == 400


def test_search_endpoint_returns_matches_and_passes_options(monkeypatch):
    client, captured, _ = _install_routes_fakes(monkeypatch)

    response = client.post(
        "/style-match/search",
        data={
            "image": (io.BytesIO(_png_bytes()), "query.jpg"),
            "uuid": "q-1",
            "catalog_id": "my-catalog",
            "lrc_local_id": "7",
            "top_k": "5",
            "scoring_mode": "rrf",
            "weights": json.dumps({"sem": 0.6, "desc": 0.2, "color": 0.1, "loc": 0.1}),
            "filters": json.dumps({"has_edits": True, "camera_model": "ILCE-7M4", "lat": 48.85, "lon": 2.35, "radius_km": 20}),
            "description": "sunset over the lake",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["matches"][0]["rank"] == 1
    assert payload["matches"][0]["lrc_local_id"] == 48213
    kwargs = captured["search"]
    assert kwargs["exclude"] == ("my-catalog", 7)
    assert kwargs["top_k"] == 5
    assert kwargs["scoring_mode"] == "rrf"
    assert kwargs["weights"]["sem"] == 0.6
    assert kwargs["filters"]["camera_model"] == "ILCE-7M4"
    assert kwargs["filters"]["lat"] == 48.85
    assert kwargs["description"] == "sunset over the lake"


def test_search_rejects_unknown_scoring_mode(monkeypatch):
    client, _, _ = _install_routes_fakes(monkeypatch)
    response = client.post(
        "/style-match/search",
        data={
            "image": (io.BytesIO(_png_bytes()), "query.jpg"),
            "scoring_mode": "bogus",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_status_and_health_report_stats(monkeypatch):
    client, _, _ = _install_routes_fakes(monkeypatch)
    status = client.get("/style-match/status")
    assert status.status_code == 200
    assert status.get_json()["style_refs"]["total"] == 3

    health = client.get("/style-match/health")
    assert health.status_code == 200
    assert health.get_json()["embedding_model"] == "clip-vit-b32"
