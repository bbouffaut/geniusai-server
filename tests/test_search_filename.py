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
    )


def _install_core_fakes(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.DEFAULT_MIN_PERTINENCE_SCORE = 0.35
    fake_config.logger = _make_logger()

    fake_server_lifecycle = types.ModuleType("server_lifecycle")
    fake_server_lifecycle.embed_query = lambda term: [1.0, 0.0]

    fake_postgre_service = types.ModuleType("service_postgre")
    fake_postgre_service.query_images = lambda *args, **kwargs: {
        "ids": [["photo-a", "photo-b"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[
            {
                "filename": "alpha.jpg",
                "title": "alpha",
                "exif": {"camera": "Nikon D850"},
                "overall_score": 9.1,
                "composition_score": 8.2,
                "quality_critique": "Strong composition and exposure.",
                "model": "gpt-4o",
                "run_date": "2026-05-01 10:00:00",
                "capture_time": "2026-04-30 09:30:00",
            },
            {
                "filename": "beta.jpg",
                "title": "beta",
            },
        ]],
        "embeddings": [[[1.0, 0.0], [1.0, 0.0]]],
    }
    fake_postgre_service.get_image_metadatas = lambda ids=None: {
        "ids": ["photo-a", "photo-b", "photo-c"],
        "metadatas": [
            {
                "filename": "alpha.jpg",
                "title": "alpha",
                "exif": {"camera": "Nikon D850"},
                "overall_score": 9.1,
                "composition_score": 8.2,
                "quality_critique": "Strong composition and exposure.",
                "model": "gpt-4o",
                "run_date": "2026-05-01 10:00:00",
                "capture_time": "2026-04-30 09:30:00",
            },
            {
                "filename": "beta.jpg",
                "title": "beta",
            },
            {
                "filename": "gamma.jpg",
                "title": "contains search term",
                "exif": {"camera": "Leica M10"},
                "capture_time": "2026-04-29 08:15:00",
            },
        ],
    }
    fake_postgre_service.group_and_sort_images = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "server_lifecycle", fake_server_lifecycle)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)


def test_search_images_exposes_internal_metadata(monkeypatch):
    _install_core_fakes(monkeypatch)
    sys.modules.pop("service_search", None)

    service_search = importlib.import_module("service_search")

    results = service_search.search_images("search term", None, None, 0.0)
    by_uuid = {item["uuid"]: item for item in results}

    assert by_uuid["photo-a"]["filename"] == "alpha.jpg"
    assert by_uuid["photo-a"]["metadata"]["exif"]["camera"] == "Nikon D850"
    assert by_uuid["photo-a"]["quality"]["overall_score"] == 9.1
    assert by_uuid["photo-a"]["quality"]["composition_score"] == 8.2
    assert by_uuid["photo-a"]["quality"]["quality_critique"] == "Strong composition and exposure."
    assert by_uuid["photo-a"]["ai_model"] == "gpt-4o"
    assert by_uuid["photo-a"]["ai_rundate"] == "2026-05-01 10:00:00"
    assert by_uuid["photo-a"]["photo_date"] == "2026-04-30 09:30:00"
    assert by_uuid["photo-b"]["filename"] == "beta.jpg"
    assert by_uuid["photo-c"]["filename"] == "gamma.jpg"
    assert by_uuid["photo-c"]["photo_date"] == "2026-04-29 08:15:00"
    assert by_uuid["photo-c"]["metadata"]["exif"]["camera"] == "Leica M10"
    assert by_uuid["photo-a"]["match_type"] == "semantic"
    assert by_uuid["photo-c"]["match_type"] == "metadata"
    assert by_uuid["photo-c"]["metadata_match"] is True


def test_search_route_returns_minimum_fields_by_default(monkeypatch):
    _install_core_fakes(monkeypatch)

    fake_service_search = types.ModuleType("service_search")
    fake_service_search.search_images = lambda term, quality_sort, uuids_to_search, min_pertinence_score: [
        {
            "uuid": "photo-a",
            "filename": "alpha.jpg",
            "distance": 0.1,
            "pertinence_score": 0.9,
            "match_type": "semantic",
            "photo_date": "2026-04-30 09:30:00",
            "ai_model": "gpt-4o",
            "ai_rundate": "2026-05-01 10:00:00",
            "metadata": {
                "uuid": "photo-a",
                "filename": "alpha.jpg",
                "title": "alpha",
                "exif": {"camera": "Nikon D850"},
                "capture_time": "2026-04-30 09:30:00",
                "run_date": "2026-05-01 10:00:00",
                "model": "gpt-4o",
            },
            "quality": {
                "overall_score": 9.1,
                "composition_score": 8.2,
                "quality_critique": "Strong composition and exposure.",
            },
            "metadata_match": True,
        }
    ]
    fake_service_search.group_similar_images = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "service_search", fake_service_search)
    sys.modules.pop("routes_search", None)

    routes_search = importlib.import_module("routes_search")

    app = Flask(__name__)
    app.register_blueprint(routes_search.search_bp)

    with app.test_client() as client:
        response = client.get("/search?term=alpha")

    assert response.status_code == 200
    assert response.get_json() == [
        {
            "ai_model": "gpt-4o",
            "ai_rundate": "2026-05-01 10:00:00",
            "distance": 0.1,
            "filename": "alpha.jpg",
            "match_type": "semantic",
            "photo_date": "2026-04-30 09:30:00",
        }
    ]


def test_search_route_includes_metadata_when_requested(monkeypatch):
    _install_core_fakes(monkeypatch)

    fake_service_search = types.ModuleType("service_search")
    fake_service_search.search_images = lambda term, quality_sort, uuids_to_search, min_pertinence_score: [
        {
            "uuid": "photo-a",
            "filename": "alpha.jpg",
            "distance": 0.1,
            "pertinence_score": 0.9,
            "match_type": "semantic",
            "photo_date": "2026-04-30 09:30:00",
            "ai_model": "gpt-4o",
            "ai_rundate": "2026-05-01 10:00:00",
            "metadata": {
                "uuid": "photo-a",
                "filename": "alpha.jpg",
                "title": "alpha",
                "exif": {"camera": "Nikon D850"},
                "capture_time": "2026-04-30 09:30:00",
                "run_date": "2026-05-01 10:00:00",
                "model": "gpt-4o",
            },
            "quality": {
                "overall_score": 9.1,
                "composition_score": 8.2,
                "quality_critique": "Strong composition and exposure.",
            },
            "metadata_match": True,
        }
    ]
    fake_service_search.group_similar_images = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "service_search", fake_service_search)
    sys.modules.pop("routes_search", None)

    routes_search = importlib.import_module("routes_search")

    app = Flask(__name__)
    app.register_blueprint(routes_search.search_bp)

    with app.test_client() as client:
        response = client.get("/search?term=alpha&return_metadata=true")

    assert response.status_code == 200
    assert response.get_json() == [
        {
            "ai_model": "gpt-4o",
            "ai_rundate": "2026-05-01 10:00:00",
            "distance": 0.1,
            "filename": "alpha.jpg",
            "match_type": "semantic",
            "metadata": {
                "uuid": "photo-a",
                "filename": "alpha.jpg",
                "title": "alpha",
                "exif": {"camera": "Nikon D850"},
                "capture_time": "2026-04-30 09:30:00",
                "run_date": "2026-05-01 10:00:00",
                "model": "gpt-4o",
            },
            "photo_date": "2026-04-30 09:30:00",
        }
    ]
