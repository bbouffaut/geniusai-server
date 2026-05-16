import importlib
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
    )


def _install_get_fakes(monkeypatch, photo_data):
    fake_config = types.ModuleType("config")
    fake_config.logger = _make_logger()

    fake_postgre_service = types.ModuleType("service_postgre")
    fake_postgre_service.get_image = lambda *args, **kwargs: None
    fake_postgre_service.get_image_metadatas = lambda ids=None: photo_data
    fake_postgre_service.get_all_image_ids = lambda *args, **kwargs: []

    fake_service_index = types.ModuleType("service_index")
    fake_service_index.keywords_with_generation_model = lambda keywords, provider, model: keywords

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    monkeypatch.setitem(sys.modules, "service_index", fake_service_index)
    sys.modules.pop("service_get", None)
    sys.modules.pop("routes_get", None)

    routes_get = importlib.import_module("routes_get")

    app = Flask(__name__)
    app.register_blueprint(routes_get.get_bp)

    return app


def test_get_returns_all_photos_when_body_empty(monkeypatch):
    app = _install_get_fakes(
        monkeypatch,
        photo_data={
            "ids": ["photo-a", "photo-b", "photo-c"],
            "metadatas": [
                {
                    "uuid": "photo-a",
                    "filename": "alpha.jpg",
                    "title": "alpha",
                    "caption": "Alpha caption",
                    "alt_text": "Alpha alt text",
                    "keywords": "[\"sunset\", \"sea\"]",
                    "provider": "ollama",
                    "model": "gpt-4o",
                    "ai_model": "gpt-4o",
                    "ai_rundate": "2026-05-01 10:00:00",
                    "run_date": "2026-05-01 10:00:00",
                    "photo_date": "2026-04-30 09:30:00",
                    "overall_score": 9.1,
                    "composition_score": 8.2,
                    "quality_critique": "Strong composition and exposure.",
                    "exif": {"camera": "Nikon D850"},
                },
                {
                    "uuid": "photo-b",
                    "filename": "beta.jpg",
                    "title": "beta",
                    "provider": "ollama",
                    "model": "gpt-4o",
                    "photo_date": "2026-04-29 08:15:00",
                },
                {
                    "uuid": "photo-c",
                    "filename": "gamma.jpg",
                    "title": "gamma",
                    "ai_model": "claude",
                    "run_date": "2026-05-02 11:00:00",
                    "capture_time": "2026-04-28 07:00:00",
                },
            ],
        },
    )

    with app.test_client() as client:
        response = client.post("/get", data="", content_type="application/json")

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["count"] == 3
    assert [item["uuid"] for item in payload["photos"]] == ["photo-a", "photo-b", "photo-c"]
    assert payload["photos"][0]["filename"] == "alpha.jpg"
    assert payload["photos"][0]["ai_model"] == "gpt-4o"
    assert payload["photos"][0]["quality"]["overall_score"] == 9.1
    assert payload["photos"][0]["quality"]["composition_score"] == 8.2
    assert payload["photos"][0]["quality"]["quality_critique"] == "Strong composition and exposure."
    assert payload["photos"][0]["metadata"]["exif"]["camera"] == "Nikon D850"


def test_get_filters_photos_by_aliases_and_nested_metadata(monkeypatch):
    app = _install_get_fakes(
        monkeypatch,
        photo_data={
            "ids": ["photo-a", "photo-b", "photo-c"],
            "metadatas": [
                {
                    "uuid": "photo-a",
                    "filename": "alpha.jpg",
                    "title": "alpha",
                    "caption": "Alpha caption",
                    "keywords": "[\"sunset\", \"sea\"]",
                    "provider": "ollama",
                    "model": "gpt-4o",
                    "ai_model": "gpt-4o",
                    "ai_rundate": "2026-05-01 10:00:00",
                    "run_date": "2026-05-01 10:00:00",
                    "photo_date": "2026-04-30 09:30:00",
                    "overall_score": 9.1,
                    "composition_score": 8.2,
                    "quality_critique": "Strong composition and exposure.",
                    "exif": {"camera": "Nikon D850", "iso": 200},
                },
                {
                    "uuid": "photo-b",
                    "filename": "beta.jpg",
                    "title": "beta",
                    "provider": "ollama",
                    "model": "gpt-4o",
                    "photo_date": "2026-04-29 08:15:00",
                    "overall_score": 7.0,
                    "quality_critique": "Average exposure.",
                    "exif": {"camera": "Canon R5"},
                },
                {
                    "uuid": "photo-c",
                    "filename": "gamma.jpg",
                    "title": "gamma",
                    "ai_model": "claude",
                    "run_date": "2026-05-02 11:00:00",
                    "capture_time": "2026-04-28 07:00:00",
                    "overall_score": 8.0,
                    "exif": {"camera": "Sony A7"},
                },
            ],
        },
    )

    with app.test_client() as client:
        response = client.post(
            "/get",
            json={
                "filters": {
                    "uuid": "photo-a",
                    "filename": "alpha.jpg",
                    "ai_run_date": "2026-05-01 10:00:00",
                    "photos_date": "2026-04-30",
                    "quality": {
                        "overall_score": 9.1,
                    },
                    "metadata": {
                        "exif": {
                            "camera": "Nikon D850",
                        },
                    },
                }
            },
        )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["count"] == 1
    assert [item["uuid"] for item in payload["photos"]] == ["photo-a"]
    photo = payload["photos"][0]
    assert photo["filename"] == "alpha.jpg"
    assert photo["ai_model"] == "gpt-4o"
    assert photo["ai_rundate"] == "2026-05-01 10:00:00"
    assert photo["photo_date"] == "2026-04-30 09:30:00"
    assert photo["quality"]["overall_score"] == 9.1
    assert photo["metadata"]["title"] == "alpha"
    assert photo["metadata"]["exif"]["camera"] == "Nikon D850"
