"""
/re-index endpoint — re-compute embeddings for photos already in the DB.

Accepts GET (query-string) and POST (JSON body) with the same parameters:

  embedding    bool  (default true)  — recompute the prose embedding (caption)
  embedding_kw bool  (default true)  — recompute the keyword embedding
  uuids        str | list            — comma-separated or JSON array of UUIDs
                                       to process; omit to process all photos

Examples
--------
GET  /re-index
GET  /re-index?embedding=true&embedding_kw=false
GET  /re-index?uuids=abc123,def456
POST /re-index
     {"embedding": true, "embedding_kw": true, "uuids": ["abc123", "def456"]}
"""

import json

from flask import Blueprint, jsonify, request

from config import logger
from service_reindex import reindex_embeddings


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
        # Try JSON array first
        if stripped.startswith("["):
            try:
                items = json.loads(stripped)
                if isinstance(items, list):
                    parsed = [str(item).strip() for item in items if str(item).strip()]
                    return parsed or None
            except json.JSONDecodeError:
                pass
        # Fall back to comma-separated
        parsed = [item.strip() for item in stripped.split(",") if item.strip()]
        return parsed or None

    return None


def _parse_params(source):
    """
    Extract re-index parameters from a dict-like source
    (request.args or a parsed JSON body).
    """
    recompute_prose = _coerce_bool(source.get("embedding"), default=True)
    recompute_kw = _coerce_bool(source.get("embedding_kw"), default=True)

    # Accept both "uuid"/"uuids" for convenience
    raw_uuids = source.get("uuids") or source.get("uuid")
    uuids = _parse_uuids(raw_uuids)

    return recompute_prose, recompute_kw, uuids


def _run_reindex(recompute_prose, recompute_kw, uuids):
    uuid_desc = f"{len(uuids)} UUID(s)" if uuids else "all photos"
    logger.info(
        f"Re-index request: prose={recompute_prose}, "
        f"embedding_kw={recompute_kw}, scope={uuid_desc}"
    )

    if not recompute_prose and not recompute_kw:
        return jsonify({
            "status": "ok",
            "message": "Nothing to do: both embedding and embedding_kw are false.",
            "total": 0,
            "success_count": 0,
            "skipped_count": 0,
            "failure_count": 0,
        }), 200

    try:
        result = reindex_embeddings(
            uuids=uuids,
            recompute_prose=recompute_prose,
            recompute_kw=recompute_kw,
        )
        status_code = 200 if result["failure_count"] == 0 else 207
        return jsonify({"status": "ok", **result}), status_code

    except Exception as e:
        logger.error(f"Re-index failed: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


@reindex_bp.route("/re-index", methods=["GET"])
def reindex_get():
    """Re-index via query-string parameters."""
    recompute_prose, recompute_kw, uuids = _parse_params(request.args)
    return _run_reindex(recompute_prose, recompute_kw, uuids)


@reindex_bp.route("/re-index", methods=["POST"])
def reindex_post():
    """Re-index via JSON body (or form data)."""
    if request.is_json:
        body = request.get_json(silent=True) or {}
    elif request.form:
        body = {key: request.form.get(key) for key in request.form.keys()}
    else:
        raw = request.get_data(cache=True, as_text=True)
        if raw and raw.strip():
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return jsonify({"status": "error", "error": "Request body must be valid JSON"}), 400
        else:
            body = {}

    if not isinstance(body, dict):
        return jsonify({"status": "error", "error": "Request body must be a JSON object"}), 400

    recompute_prose, recompute_kw, uuids = _parse_params(body)
    return _run_reindex(recompute_prose, recompute_kw, uuids)
