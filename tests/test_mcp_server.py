import asyncio
import sys
import types
from pathlib import Path

import pytest


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


def _load_mcp_server(monkeypatch, search_impl):
    """Import a fresh mcp_server with fake config + service_search installed."""
    fake_config = types.ModuleType("config")
    fake_config.DEFAULT_MIN_PERTINENCE_SCORE = 0.35
    fake_config.MCP_SERVER_HOST = "127.0.0.1"
    fake_config.MCP_SERVER_PORT = 8000
    fake_config.logger = _make_logger()

    fake_service_search = types.ModuleType("service_search")
    fake_service_search.search_images = search_impl
    # Reuse the real projection so the shape stays in sync with production.
    fake_service_search.project_search_result = _project_search_result

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_search", fake_service_search)
    sys.modules.pop("mcp_server", None)

    import mcp_server

    return mcp_server


def _project_search_result(result, return_metadata=False):
    metadata = result.get("metadata") or {}
    projected = {
        "ai_model": result.get("ai_model"),
        "ai_rundate": result.get("ai_rundate"),
        "distance": result.get("distance"),
        "filename": result.get("filename"),
        "match_type": result.get("match_type"),
        "pertinence_score": result.get("pertinence_score"),
        "capture_time": result.get("capture_time"),
        "uuid": result.get("uuid"),
    }
    if return_metadata:
        projected["metadata"] = metadata
    return projected


def _run(coro):
    return asyncio.run(coro)


def test_search_photos_tool_returns_projected_results(monkeypatch):
    captured = {}

    def fake_search_images(term, quality_sort, uuids, score, search_filters=None, limit=300):
        captured.update(
            term=term,
            uuids=uuids,
            score=score,
            search_filters=search_filters,
            limit=limit,
        )
        return [{"uuid": "abc", "filename": "a.jpg", "pertinence_score": 0.9}]

    mcp_server = _load_mcp_server(monkeypatch, fake_search_images)
    from fastmcp import Client

    async def scenario():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool("search_photos", {"query": "sunset over lake"})

    result = _run(scenario())

    assert captured["term"] == "sunset over lake"
    assert captured["score"] == 0.35
    assert captured["limit"] == 300
    assert result.data == [
        {
            "ai_model": None,
            "ai_rundate": None,
            "distance": None,
            "filename": "a.jpg",
            "match_type": None,
            "pertinence_score": 0.9,
            "capture_time": None,
            "uuid": "abc",
        }
    ]


def test_search_photos_tool_passes_filters_and_metadata(monkeypatch):
    captured = {}

    def fake_search_images(term, quality_sort, uuids, score, search_filters=None, limit=300):
        captured.update(term=term, search_filters=search_filters, score=score, limit=limit)
        return [{"uuid": "x", "filename": "x.jpg", "metadata": {"iso": 100}}]

    mcp_server = _load_mcp_server(monkeypatch, fake_search_images)
    from fastmcp import Client

    async def scenario():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool(
                "search_photos",
                {
                    "query": "lake",
                    "filters": {"camera_make": "Fujifilm"},
                    "min_pertinence_score": 0.5,
                    "limit": 10,
                    "return_metadata": True,
                },
            )

    result = _run(scenario())

    assert captured["search_filters"] == {"camera_make": "Fujifilm"}
    assert captured["score"] == 0.5
    assert captured["limit"] == 10
    assert result.data[0]["metadata"] == {"iso": 100}


def test_search_photos_tool_rejects_empty_query_and_filters(monkeypatch):
    def fake_search_images(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("search_images should not be called")

    mcp_server = _load_mcp_server(monkeypatch, fake_search_images)
    from fastmcp import Client

    async def scenario():
        async with Client(mcp_server.mcp) as client:
            with pytest.raises(Exception) as exc_info:
                await client.call_tool("search_photos", {"query": ""})
            return exc_info

    exc_info = _run(scenario())
    assert "Provide a search term" in str(exc_info.value)


def test_run_search_validates_score_bounds(monkeypatch):
    mcp_server = _load_mcp_server(monkeypatch, lambda *a, **k: [])

    with pytest.raises(ValueError):
        mcp_server._run_search(query="lake", min_pertinence_score=2)


def test_run_search_validates_limit_bounds(monkeypatch):
    mcp_server = _load_mcp_server(monkeypatch, lambda *a, **k: [])

    with pytest.raises(ValueError):
        mcp_server._run_search(query="lake", limit=0)
