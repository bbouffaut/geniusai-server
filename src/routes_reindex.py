"""
/re-index endpoint — re-compute embeddings for photos already in the DB.

Streams progress back to the caller as newline-delimited JSON (NDJSON).
One JSON object is written per photo processed; the final object is a summary.

Accepts GET (query-string) and POST (JSON body) with the same parameters:

  embedding    bool  (default true)  — recompute the prose embedding (caption)
  embedding_kw bool  (default true)  — recompute the keyword embedding
  uuids        str | list            — comma-separated or JSON array of UUIDs
                                       to process; omit to process all photos

NDJSON event shapes
-------------------
  {"event": "start",   "total": 4821, "prose": true, "kw": true}
  {"event": "indexed", "uuid": "…", "filename": "…", "prose": true, "kw": true, "index": 1, "total": 4821}
  {"event": "skipped", "uuid": "…", "filename": "…", "index": 2, "total": 4821}
  {"event": "error",   "uuid": "…", "filename": "…", "error": "…", "index": 3, "total": 4821}
  {"event": "done",    "total": 4821, "success_count": 4810, "skipped_count": 9, "failure_count": 2}

Examples
--------
GET  /re-index
GET  /re-index?embedding=true&embedding_kw=false
GET  /re-index?uuids=abc123,def456
POST /re-index
     {"embedding": true, "embedding_kw": true, "uuids": ["abc123", "def456"]}
"""

import json

from flask import Blueprint, Response, jsonify, request, stream_with_context

from config import logger
from service_reindex import reindex_embeddings_stream


reindex_bp = Blueprint("reindex", __name__)


def _coerce_bool(value, default=True):
    """Parse a bool from various representations (string, int, Python bool)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_uuids(raw):
    """Return a list of UUID strings, or None if not provided."""
    if raw is None:
        return None

    if isinstance(raw, list):
        parsed = [str(item).strip() for item in raw if str(item).strip()]
        return parsed or None

    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        if stripped.startswith("["):
            try:
                items = json.loads(stripped)
                if isinstance(items, list):
                    parsed = [str(item).strip() for item in items if str(item).strip()]
                    return parsed or None
            except json.JSONDecodeError:
                pass
        parsed = [item.strip() for item in stripped.split(",") if item.strip()]
        return parsed or None

    return None


def _parse_params(source):
    """Extract re-index parameters from a dict-like source."""
    recompute_prose = _coerce_bool(source.get("embedding"), default=True)
    recompute_kw = _coerce_bool(source.get("embedding_kw"), default=True)
    raw_uuids = source.get("uuids") or source.get("uuid")
    uuids = _parse_uuids(raw_uuids)
    return recompute_prose, recompute_kw, uuids


def _parse_request_body():
    """Return the POST body as a dict, or None on parse error."""
    if request.is_json:
        return request.get_json(silent=True) or {}
    if request.form:
        return {key: request.form.get(key) for key in request.form.keys()}
    raw = request.get_data(cache=True, as_text=True)
    if raw and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return {}


def _stream_reindex(recompute_prose, recompute_kw, uuids):
    """Return a streaming NDJSON Response that yields one line per photo."""
    uuid_desc = f"{len(uuids)} UUID(s)" if uuids else "all photos"
    logger.info(
        f"Re-index request: prose={recompute_prose}, "
        f"embedding_kw={recompute_kw}, scope={uuid_desc}"
    )

    @stream_with_context
    def _generate():
        for event in reindex_embeddings_stream(
            uuids=uuids,
            recompute_prose=recompute_prose,
            recompute_kw=recompute_kw,
        ):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return Response(
        _generate(),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )


@reindex_bp.route("/re-index", methods=["GET"])
def reindex_get():
    """Re-index via query-string parameters, streamed as NDJSON."""
    recompute_prose, recompute_kw, uuids = _parse_params(request.args)
    return _stream_reindex(recompute_prose, recompute_kw, uuids)


@reindex_bp.route("/re-index", methods=["POST"])
def reindex_post():
    """Re-index via JSON body, streamed as NDJSON."""
    body = _parse_request_body()
    if body is None:
        return jsonify({"status": "error", "error": "Request body must be valid JSON"}), 400
    if not isinstance(body, dict):
        return jsonify({"status": "error", "error": "Request body must be a JSON object"}), 400

    recompute_prose, recompute_kw, uuids = _parse_params(body)
    return _stream_reindex(recompute_prose, recompute_kw, uuids)
