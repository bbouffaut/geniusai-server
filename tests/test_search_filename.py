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
    fake_postgre_service.get_image_metadatas = lambda ids=None, where_clause=None: {
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
    assert by_uuid["photo-a"]["capture_time"] == "2026-04-30 09:30:00"
    assert by_uuid["photo-b"]["filename"] == "beta.jpg"
    assert by_uuid["photo-c"]["filename"] == "gamma.jpg"
    assert by_uuid["photo-c"]["capture_time"] == "2026-04-29 08:15:00"
    assert by_uuid["photo-c"]["metadata"]["exif"]["camera"] == "Leica M10"
    assert by_uuid["photo-a"]["match_type"] == "semantic"
    assert by_uuid["photo-c"]["match_type"] == "metadata"
    assert by_uuid["photo-c"]["metadata_match"] is True


def test_parse_search_query_extracts_date_and_aperture_filters(monkeypatch):
    _install_core_fakes(monkeypatch)
    sys.modules.pop("service_search", None)

    service_search = importlib.import_module("service_search")

    annecy_query = service_search._parse_search_query("Annecy May 2026")
    assert annecy_query["semantic_term"] == "Annecy"
    assert annecy_query["metadata_filters"] == [
        {
            "field": "capture_time",
            "op": "gte",
            "value": "2026-05-01",
            "source": "query_month",
        },
        {
            "field": "capture_time",
            "op": "lt",
            "value": "2026-06-01",
            "source": "query_month",
        },
    ]

    lake_query = service_search._parse_search_query("Lake with F2.8")
    assert lake_query["semantic_term"] == "Lake"
    assert lake_query["metadata_filters"] == [
        {
            "field": "aperture",
            "op": "number_eq",
            "value": 2.8,
            "tolerance": 0.05,
            "source": "query_aperture",
        }
    ]


def test_search_images_pushes_query_filters_to_postgres(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.DEFAULT_MIN_PERTINENCE_SCORE = 0.35
    fake_config.logger = _make_logger()

    captured = {}

    fake_server_lifecycle = types.ModuleType("server_lifecycle")

    def fake_embed_query(term):
        captured["semantic_term"] = term
        return [1.0, 0.0]

    fake_server_lifecycle.embed_query = fake_embed_query

    photo_metadata = {
        "filename": "annecy.jpg",
        "title": "Annecy lake",
        "capture_time": "2026-05-20 10:00:00",
    }

    fake_postgre_service = types.ModuleType("service_postgre")

    def fake_query_images(query_embedding, n_results, where_clause=None, include_embeddings=False):
        captured["query_where_clause"] = where_clause
        return {
            "ids": [["photo-a"]],
            "distances": [[0.1]],
            "metadatas": [[photo_metadata]],
            "embeddings": [[[1.0, 0.0]]],
        }

    def fake_get_image_metadatas(ids=None, where_clause=None):
        captured["metadata_ids"] = ids
        captured["metadata_where_clause"] = where_clause
        return {
            "ids": ["photo-a"],
            "metadatas": [photo_metadata],
        }

    fake_postgre_service.query_images = fake_query_images
    fake_postgre_service.get_image_metadatas = fake_get_image_metadatas
    fake_postgre_service.group_and_sort_images = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "server_lifecycle", fake_server_lifecycle)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    sys.modules.pop("service_search", None)

    service_search = importlib.import_module("service_search")

    results = service_search.search_images("Annecy May 2026", None, None, 0.0)

    assert captured["semantic_term"] == "Annecy"
    assert captured["query_where_clause"] == {
        "metadata_filters": [
            {
                "field": "capture_time",
                "op": "gte",
                "value": "2026-05-01",
                "source": "query_month",
            },
            {
                "field": "capture_time",
                "op": "lt",
                "value": "2026-06-01",
                "source": "query_month",
            },
        ]
    }
    assert captured["metadata_where_clause"] == captured["query_where_clause"]
    assert results[0]["uuid"] == "photo-a"
    assert results[0]["match_type"] == "semantic+metadata"


def test_search_route_returns_minimum_fields_by_default(monkeypatch):
    _install_core_fakes(monkeypatch)

    fake_service_search = types.ModuleType("service_search")
    fake_service_search.search_images = lambda term, quality_sort, uuids_to_search, min_pertinence_score, search_filters=None: [
        {
            "uuid": "photo-a",
            "filename": "alpha.jpg",
            "distance": 0.1,
            "pertinence_score": 0.9,
            "match_type": "semantic",
            "capture_time": "2026-04-30 09:30:00",
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
            "pertinence_score": 0.9,
            "capture_time": "2026-04-30 09:30:00",
            "uuid": "photo-a",
        }
    ]


def test_search_route_includes_metadata_when_requested(monkeypatch):
    _install_core_fakes(monkeypatch)

    fake_service_search = types.ModuleType("service_search")
    fake_service_search.search_images = lambda term, quality_sort, uuids_to_search, min_pertinence_score, search_filters=None: [
        {
            "uuid": "photo-a",
            "filename": "alpha.jpg",
            "distance": 0.1,
            "pertinence_score": 0.9,
            "match_type": "semantic",
            "capture_time": "2026-04-30 09:30:00",
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
            "pertinence_score": 0.9,
            "metadata": {
                "uuid": "photo-a",
                "filename": "alpha.jpg",
                "title": "alpha",
                "exif": {"camera": "Nikon D850"},
                "capture_time": "2026-04-30 09:30:00",
                "run_date": "2026-05-01 10:00:00",
                "model": "gpt-4o",
            },
            "capture_time": "2026-04-30 09:30:00",
            "uuid": "photo-a",
        }
    ]


def test_search_route_forwards_structured_filters(monkeypatch):
    _install_core_fakes(monkeypatch)

    captured = {}
    fake_service_search = types.ModuleType("service_search")

    def fake_search_images(term, quality_sort, uuids_to_search, min_pertinence_score, search_filters=None):
        captured["term"] = term
        captured["filters"] = search_filters
        return []

    fake_service_search.search_images = fake_search_images
    fake_service_search.group_similar_images = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "service_search", fake_service_search)
    sys.modules.pop("routes_search", None)

    routes_search = importlib.import_module("routes_search")

    app = Flask(__name__)
    app.register_blueprint(routes_search.search_bp)

    with app.test_client() as client:
        response = client.post(
            "/search",
            json={
                "term": "lake",
                "filters": {
                    "capture_time": "2026-05",
                    "aperture_f_number": 2.8,
                },
            },
        )

    assert response.status_code == 200
    assert captured == {
        "term": "lake",
        "filters": {
            "capture_time": "2026-05",
            "aperture_f_number": 2.8,
        },
    }
