"""Style Match REST endpoints (plan §3.3).

Same multipart conventions as the existing /index routes: the server never
touches the user's filesystem; the plugin/bootstrap uploads the renders.

  POST /style-match/index    – one reference photo (bootstrap + incremental hook)
  POST /style-match/search   – ranked matches for the incoming photo
  GET  /style-match/status   – feature status + style-ref counts
  GET  /style-match/health   – models loaded, DB reachable, counts, render mix
"""

import json

from flask import Blueprint, jsonify, request

from config import (
    STYLE_MATCH_DEFAULT_WEIGHTS,
    STYLE_MATCH_EMBEDDING_MODEL,
    STYLE_MATCH_MAX_ORIGINAL_BYTES,
    STYLE_MATCH_SCORING_MODE,
    logger,
)
import service_postgre as postgre_service

style_match_bp = Blueprint("style_match", __name__)

SIGNAL_KEYS = ("sem", "desc", "color", "loc")


def _parse_json_field(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else default
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default=None):
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_file_bytes(field_name):
    file_storage = request.files.get(field_name)
    if not file_storage:
        return None
    upload_stream = file_storage.stream
    if hasattr(upload_stream, "seek"):
        try:
            upload_stream.seek(0)
        except (OSError, ValueError):
            pass
    return upload_stream.read()


def _require_file(field_name):
    data = _read_file_bytes(field_name)
    if not data:
        return None
    return data


def _sanitize_filters(filters):
    clean = {}
    if not isinstance(filters, dict):
        return clean
    if "has_edits" in filters:
        clean["has_edits"] = _coerce_bool(filters.get("has_edits"), None)
    if filters.get("camera_model"):
        clean["camera_model"] = str(filters["camera_model"])
    for key in ("date_from", "date_to"):
        if filters.get(key):
            clean[key] = str(filters[key])
    for key in ("lat", "lon"):
        value = _coerce_float(filters.get(key))
        if value is not None:
            clean[key] = value
    radius_km = _coerce_float(filters.get("radius_km"))
    if radius_km is not None:
        clean["radius_km"] = radius_km
    return clean


@style_match_bp.route("/style-match/index", methods=["POST"])
def style_match_index():
    image_bytes = _require_file("image")
    if not image_bytes:
        return jsonify({"error": "Missing 'image' file (current-look render)."}), 400
    neutral_image_bytes = _read_file_bytes("neutral_image")

    # Optional original upload: the server extracts the RAW embedded preview
    # (or uses the JPEG itself) for the neutral HSV histogram, so the plugin
    # stays the single indexing path (plan §2.4).
    original_bytes = _read_file_bytes("original")
    if original_bytes and len(original_bytes) > STYLE_MATCH_MAX_ORIGINAL_BYTES:
        logger.warning(
            f"style-match/index: original upload for {request.form.get('uuid')} exceeds "
            f"{STYLE_MATCH_MAX_ORIGINAL_BYTES} bytes; colour signal skipped."
        )
        original_bytes = None

    form = request.form
    uuid = form.get("uuid")
    filename = form.get("filename") or ""
    catalog_id = form.get("catalog_id")
    lrc_local_id = form.get("lrc_local_id")
    if not uuid or not catalog_id or lrc_local_id in (None, ""):
        return jsonify({"error": "uuid, catalog_id and lrc_local_id are required."}), 400

    develop_settings = _parse_json_field(form.get("develop_settings"), None)
    meta = {
        "uuid": uuid,
        "filename": filename,
        "file_path": form.get("file_path") or filename,
        "capture_time": form.get("capture_time"),
        "catalog_id": catalog_id,
        "lrc_local_id": int(float(lrc_local_id)),
        "latitude": _coerce_float(form.get("latitude")),
        "longitude": _coerce_float(form.get("longitude")),
        "camera_make": form.get("camera_make"),
        "camera_model": form.get("camera_model"),
        "lens": form.get("lens"),
        "focal_length_mm": _coerce_float(form.get("focal_length_mm")),
        "aperture_f_number": _coerce_float(form.get("aperture_f_number")),
        "iso": int(form["iso"]) if form.get("iso") not in (None, "") else None,
        "description": form.get("description"),
        "develop_settings": develop_settings,
        "has_edits": _coerce_bool(form.get("has_edits"), False),
        "render_source": form.get("render_source"),
    }

    import service_style_match

    try:
        result = service_style_match.index_photo(
            image_bytes,
            meta,
            neutral_image_bytes=neutral_image_bytes,
            original_bytes=original_bytes,
            original_filename=form.get("original_filename") or filename,
            generate_description=_coerce_bool(form.get("generate_description"), True),
        )
    except Exception as e:
        logger.error(f"style-match/index failed for {uuid}: {e}", exc_info=True)
        return jsonify({"error": "Style-match indexing failed", "detail": str(e)}), 500

    status_code = 200 if result.get("status") == "processed" else 500
    if status_code == 200:
        settings_count = len(develop_settings) if isinstance(develop_settings, dict) else 0
        logger.info(
            f"style-match/index: indexed {filename or uuid} "
            f"(catalog={catalog_id} local_id={meta['lrc_local_id']} "
            f"render={result.get('render_source')} "
            f"colour={result.get('has_color_histogram')} "
            f"desc_emb={result.get('has_description_embedding')} "
            f"edits={meta['has_edits']} settings={settings_count})"
        )
    else:
        logger.warning(
            f"style-match/index: not processed for {filename or uuid}: "
            f"{result.get('error')}"
        )
    return jsonify(result), status_code


@style_match_bp.route("/style-match/search", methods=["POST"])
def style_match_search():
    image_bytes = _require_file("image")
    if not image_bytes:
        return jsonify({"error": "Missing 'image' file (current-look render)."}), 400
    neutral_image_bytes = _read_file_bytes("neutral_image")

    form = request.form
    weights = _parse_json_field(form.get("weights"), dict(STYLE_MATCH_DEFAULT_WEIGHTS))
    filters = _sanitize_filters(_parse_json_field(form.get("filters"), {}))
    top_k = int(float(form.get("top_k") or 12))
    scoring_mode = form.get("scoring_mode") or STYLE_MATCH_SCORING_MODE
    if scoring_mode not in ("linear", "rrf"):
        return jsonify({"error": "scoring_mode must be 'linear' or 'rrf'."}), 400

    exclude = None
    catalog_id = form.get("catalog_id")
    lrc_local_id = form.get("lrc_local_id")
    if catalog_id and lrc_local_id not in (None, ""):
        exclude = (catalog_id, int(float(lrc_local_id)))

    import service_style_match

    try:
        result = service_style_match.search_photos(
            image_bytes,
            neutral_image_bytes=neutral_image_bytes,
            description=form.get("description"),
            query_uuid=form.get("uuid"),
            weights=weights,
            filters=filters,
            exclude=exclude,
            top_k=top_k,
            scoring_mode=scoring_mode,
        )
    except Exception as e:
        logger.error(f"style-match/search failed: {e}", exc_info=True)
        return jsonify({"error": "Style-match search failed", "detail": str(e)}), 500

    status_code = 200 if result.get("status") == "ok" else 500
    return jsonify(result), status_code


@style_match_bp.route("/style-match/status", methods=["GET"])
def style_match_status():
    try:
        stats = postgre_service.style_ref_stats()
    except Exception as e:
        logger.error(f"style-match/status failed: {e}", exc_info=True)
        return jsonify({"error": "Style-match status unavailable", "detail": str(e)}), 500
    return jsonify(
        {
            "status": "ok",
            "embedding_model": STYLE_MATCH_EMBEDDING_MODEL,
            "style_refs": stats,
        }
    )


@style_match_bp.route("/style-match/health", methods=["GET"])
def style_match_health():
    # server_lifecycle is imported at server startup; reuse it instead of
    # re-importing (which would pull torch into lightweight contexts).
    import sys

    lifecycle = sys.modules.get("server_lifecycle")
    health = {
        "status": "ok",
        "embedding_model": STYLE_MATCH_EMBEDDING_MODEL,
        "clip_model_loaded": bool(getattr(lifecycle, "clip_model", None)) if lifecycle else False,
        "text_model_loaded": bool(getattr(lifecycle, "model", None)) if lifecycle else False,
    }
    try:
        health["style_refs"] = postgre_service.style_ref_stats()
    except Exception as e:
        health["status"] = "degraded"
        health["style_refs_error"] = str(e)
    return jsonify(health)
