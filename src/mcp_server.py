"""MCP (Model Context Protocol) interface for geniusai-server.

This exposes geniusai-server's photo search capability as an MCP tool, served
over HTTP (Streamable HTTP transport) on a dedicated port alongside the REST
API. It is the in-process descendant of the standalone geniusai-search-mcp
project: instead of calling geniusai-server's /search endpoint over HTTP, the
`search_photos` tool now invokes the search service directly in the same
process, so there is no network hop and no separate liveness check to perform.

MCP clients connect to http://<MCP_SERVER_HOST>:<MCP_SERVER_PORT>/mcp.
"""

from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from config import (
    DEFAULT_MIN_PERTINENCE_SCORE,
    MCP_SERVER_HOST,
    MCP_SERVER_PORT,
    logger,
)
import service_search

mcp = FastMCP("geniusai-search")


def _run_search(
    query: str = "",
    filters: dict[str, Any] | None = None,
    min_pertinence_score: float | None = None,
    limit: int | None = None,
    return_metadata: bool = False,
    uuids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate arguments and run an in-process search, returning projected hits.

    Mirrors the validation performed by the REST /search route so the MCP tool
    and the HTTP endpoint behave identically. Raises ValueError on bad input.
    """
    term = (query or "").strip()
    search_filters = filters if isinstance(filters, dict) and filters else None

    if not term and not search_filters:
        raise ValueError("Provide a search term, explicit filters, or both")

    score = DEFAULT_MIN_PERTINENCE_SCORE if min_pertinence_score is None else float(min_pertinence_score)
    if score < 0 or score > 1:
        raise ValueError("min_pertinence_score must be between 0 and 1")

    effective_limit = 300 if limit is None else int(limit)
    if effective_limit < 1 or effective_limit > 10000:
        raise ValueError("limit must be between 1 and 10000")

    results = service_search.search_images(
        term,
        None,
        uuids,
        score,
        search_filters=search_filters,
        limit=effective_limit,
    )
    return [
        service_search.project_search_result(result, bool(return_metadata))
        for result in results
    ]


@mcp.tool
def search_photos(
    query: str = "",
    filters: dict[str, Any] | None = None,
    min_pertinence_score: float | None = None,
    limit: int | None = None,
    return_metadata: bool = False,
    uuids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search indexed Lightroom photos via geniusai-server's semantic/metadata search.

    Provide a free-text `query`, structured `filters`, or both — at least one of
    them must be non-empty.

    `filters` supports keys such as capture_time, aperture_f_number, iso,
    focal_length_mm, focal_length_35mm, shutter_speed, exposure_bias, camera_make,
    camera_model, and lens. Each may be a scalar or an object with gte/gt/lte/lt/eq
    operators, e.g. {"aperture_f_number": {"lte": 2.8}, "camera_make": "Fujifilm"}.

    Args:
        query: Free-text semantic search term (e.g. "sunset over lake").
        filters: Structured filters dict, e.g. {"capture_time": "2026-05"}.
        min_pertinence_score: Minimum relevance score (0-1) to keep a match.
            Defaults to the server default (0.35) when omitted.
        limit: Maximum number of results to return (1-10000). Defaults to 300.
        return_metadata: If true, include each photo's full stored metadata payload.
        uuids: Optional list of photo UUIDs to restrict/re-score the search to.

    Returns:
        A list of photo matches, each with uuid, filename, capture_time, match_type
        ("semantic", "semantic+metadata", or "metadata"), pertinence_score, distance,
        ai_model, and ai_rundate — plus metadata when return_metadata is true.
    """
    try:
        return _run_search(
            query=query,
            filters=filters,
            min_pertinence_score=min_pertinence_score,
            limit=limit,
            return_metadata=return_metadata,
            uuids=uuids,
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the MCP client
        logger.error(f"Error during MCP search_photos: {exc}", exc_info=True)
        raise ToolError("An internal error occurred") from exc


def run_mcp_server(host: str = MCP_SERVER_HOST, port: int = MCP_SERVER_PORT) -> None:
    """Run the MCP HTTP server (blocking).

    Intended to be launched from a background thread by geniusai_server so it can
    coexist with the REST API's WSGI server. FastMCP/uvicorn skip installing OS
    signal handlers when not running in the main thread, so this is thread-safe.
    """
    logger.info(f"Starting MCP interface on http://{host}:{port}/mcp")
    try:
        mcp.run(transport="http", host=host, port=port)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"MCP interface stopped: {exc}", exc_info=True)


if __name__ == "__main__":
    run_mcp_server()
