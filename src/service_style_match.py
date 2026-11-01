"""Style Match service: index reference photos and search for style matches.

Signals (plan §2.1, §2.3):
  sem   – CLIP cosine similarity of the current-look renders (50%)
  desc  – description-embedding cosine similarity, reusing the existing Qwen3
          photo_metadata prose embeddings (30%)
  color – HSV-histogram cosine on NEUTRAL renders on both sides (10%)
  loc   – GPS haversine decay (10%)

capture_time is stored and displayed but NEVER scored.

Candidate generation is the union of the CLIP top-K and the description top-K
(§2.3): a sem-only truncation would drop exactly the matches the description
weight exists for. Missing signals are dropped and the remaining weights are
renormalized; the response flags which signals were used.
"""

import hashlib
import math

import numpy as np

import server_lifecycle
import service_postgre as postgre_service
from config import (
    DEFAULT_METADATA_LANGUAGE,
    DEFAULT_METADATA_PROVIDER,
    STYLE_MATCH_CANDIDATE_K,
    STYLE_MATCH_DEFAULT_WEIGHTS,
    STYLE_MATCH_EMBEDDING_MODEL,
    STYLE_MATCH_ENABLED,
    STYLE_MATCH_LOC_TAU_KM,
    STYLE_MATCH_SCORING_MODE,
    logger,
)
from style_match import colors, scoring, whitelist

SIGNALS = ("sem", "desc", "color", "loc")


def _haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def _location_similarity(distance_km):
    """exp(-(d / tau)^2) — 1.0 at the same spot, ~0 beyond ~2 tau."""
    if distance_km is None:
        return None
    return math.exp(-((float(distance_km) / STYLE_MATCH_LOC_TAU_KM) ** 2))


def _strip_develop_settings(develop_settings):
    """Strip local corrections; return (clean_snapshot, local_adjustments)."""
    clean, local_count = whitelist.strip_local_adjustments(develop_settings)
    if not clean:
        return None, local_count
    return clean, local_count


def _description_embedding_for(uuid, description_text, generate_description=True, image_bytes=None, filename=None):
    """Resolve the description embedding: reuse photo_metadata first.

    Order (plan §2.7 / T1.4):
      1. existing prose embedding in photo_metadata for the uuid (same model
         as the search pipeline — no regeneration cost);
      2. embed the provided description text (document-side, no instruction
         prefix, so it is comparable with reference embeddings);
      3. optionally generate a caption via the existing LLM pipeline
         (generate_description=True for bootstrap, False for incremental
         pushes) and embed it.
    """
    if uuid:
        try:
            existing = postgre_service.get_image(uuid)
        except Exception as e:
            logger.warning(f"Could not read photo_metadata for {uuid}: {e}")
            existing = None
        if existing and existing.get("embeddings"):
            vector = existing["embeddings"][0]
            if vector is not None:
                return np.asarray(vector, dtype=np.float32), True

    if description_text and str(description_text).strip():
        vector = server_lifecycle.embed_document(str(description_text))
        if vector is not None:
            return np.asarray(vector, dtype=np.float32), False

    if generate_description and image_bytes:
        generated = _generate_description(image_bytes, uuid, filename)
        if generated:
            vector = server_lifecycle.embed_document(generated)
            if vector is not None:
                return np.asarray(vector, dtype=np.float32), False

    return None, False


def _generate_description(image_bytes, uuid, filename):
    """Generate a caption via the existing annotation pipeline (best effort)."""
    try:
        from service_metadata import get_analysis_service

        options = {
            "provider": DEFAULT_METADATA_PROVIDER,
            "language": DEFAULT_METADATA_LANGUAGE,
            "compute_quality": False,
            "compute_embeddings": False,
        }
        _, _, metadata_results, _ = get_analysis_service().analyze_batch(
            [(image_bytes, uuid or "style-match", filename or "photo.jpg")],
            options,
            None,
            None,
            set(),
            {uuid or "style-match"},
            set(),
        )
        if metadata_results and metadata_results[0].success and metadata_results[0].caption:
            return metadata_results[0].caption
    except Exception as e:
        logger.warning(f"Description generation failed for {uuid}: {e}")
    return None


def index_photo(
    image_bytes,
    meta,
    neutral_image_bytes=None,
    generate_description=True,
):
    """Embed and upsert one reference photo.

    meta carries: uuid, filename, capture_time, catalog_id, lrc_local_id,
    file_path, latitude, longitude, camera_make, camera_model, lens,
    focal_length_mm, aperture_f_number, iso, develop_settings, has_edits,
    description, render_source.

    `image_bytes` MUST be the current-look render (CLIP); `neutral_image_bytes`
    the camera/neutral render (HSV histogram). When the neutral render is
    absent the colour signal is left NULL and backfilled by the next bootstrap
    run (plan §2.4).
    """
    catalog_id = meta.get("catalog_id")
    lrc_local_id = meta.get("lrc_local_id")
    uuid = meta.get("uuid")
    if not catalog_id or lrc_local_id is None or not uuid:
        return {"status": "error", "error": "catalog_id, lrc_local_id and uuid are required"}

    embedding = server_lifecycle.embed_image(image_bytes)
    if embedding is None:
        return {"status": "error", "error": "CLIP embedding failed"}

    histogram = colors.compute_color_histogram(neutral_image_bytes) if neutral_image_bytes else None
    render_source = meta.get("render_source") or (
        "develop_export" if image_bytes else None
    )

    description_embedding, description_reused = _description_embedding_for(
        uuid,
        meta.get("description"),
        generate_description=generate_description,
        image_bytes=image_bytes,
        filename=meta.get("filename"),
    )

    clean_settings, local_count = _strip_develop_settings(meta.get("develop_settings"))

    postgre_service.upsert_style_ref(
        catalog_id=catalog_id,
        lrc_local_id=int(lrc_local_id),
        uuid=uuid,
        file_path=meta.get("file_path") or meta.get("filename") or "",
        file_name=meta.get("filename") or "",
        embedding=embedding,
        description_embedding=description_embedding,
        capture_time=meta.get("capture_time"),
        latitude=meta.get("latitude"),
        longitude=meta.get("longitude"),
        camera_make=meta.get("camera_make"),
        camera_model=meta.get("camera_model"),
        lens=meta.get("lens"),
        focal_length_mm=meta.get("focal_length_mm"),
        aperture_f_number=meta.get("aperture_f_number"),
        iso=meta.get("iso"),
        description=meta.get("description"),
        color_histogram=histogram,
        render_source=render_source,
        develop_settings=clean_settings,
        has_edits=bool(meta.get("has_edits", False)),
        embedding_model=STYLE_MATCH_EMBEDDING_MODEL,
    )

    return {
        "status": "processed",
        "uuid": uuid,
        "lrc_local_id": int(lrc_local_id),
        "catalog_id": catalog_id,
        "has_clip": True,
        "has_color_histogram": histogram is not None,
        "has_description_embedding": description_embedding is not None,
        "description_reused": description_reused,
        "local_adjustments": local_count,
        "render_source": render_source,
    }


def incremental_style_index(image_bytes, uuid, filename, metadata, options=None, prose_embedding=None):
    """T1.6 hook: after a successful /index write, mirror the photo into
    style_ref_photos so newly annotated photos become comparable immediately.

    The /index contract has no Lightroom develop state, so:
      - catalog_id comes from options (default "default");
      - lrc_local_id comes from options; when absent a deterministic synthetic
        id is derived from the uuid (the plugin's /style-match/index push later
        supersedes it with the true local id);
      - has_edits comes from options and defaults to False — the plugin's
        dedicated push (T2.1/T3.3) must be preferred for accurate flags;
      - the prose embedding just computed for photo_metadata is reused as the
        description embedding (same Qwen3 model, no extra cost);
      - no develop snapshot and no neutral render exist here: colour signal
        stays NULL until backfilled by the next bootstrap run.

    Never raises: indexing must not break the annotation flow.
    """
    if not STYLE_MATCH_ENABLED or not image_bytes or not uuid:
        return None
    try:
        options = options or {}
        metadata = metadata or {}
        catalog_id = str(options.get("catalog_id") or "default")
        lrc_local_id = options.get("lrc_local_id")
        if lrc_local_id in (None, ""):
            lrc_local_id = abs(int(hashlib.sha256(str(uuid).encode("utf-8")).hexdigest()[:15], 16))
        else:
            lrc_local_id = int(float(lrc_local_id))

        embedding = server_lifecycle.embed_image(image_bytes)
        if embedding is None:
            logger.warning(f"Style-match incremental index skipped for {uuid}: CLIP embedding failed.")
            return None

        description_embedding = None
        if prose_embedding is not None:
            description_embedding = np.asarray(prose_embedding, dtype=np.float32)

        postgre_service.upsert_style_ref(
            catalog_id=catalog_id,
            lrc_local_id=lrc_local_id,
            uuid=uuid,
            file_path=metadata.get("file_path") or filename or "",
            file_name=filename or "",
            embedding=embedding,
            description_embedding=description_embedding,
            capture_time=metadata.get("capture_time"),
            latitude=metadata.get("gps_latitude"),
            longitude=metadata.get("gps_longitude"),
            camera_make=metadata.get("camera_make"),
            camera_model=metadata.get("camera_model"),
            lens=metadata.get("lens"),
            focal_length_mm=metadata.get("focal_length_mm"),
            aperture_f_number=metadata.get("aperture_f_number"),
            iso=metadata.get("iso"),
            description=metadata.get("caption"),
            color_histogram=None,
            render_source="original",
            develop_settings=None,
            has_edits=bool(options.get("has_edits", False)),
            embedding_model=STYLE_MATCH_EMBEDDING_MODEL,
        )
        logger.info(f"Style-match incremental index written for {uuid}.")
        return {"status": "processed", "uuid": uuid}
    except Exception as e:
        logger.warning(f"Style-match incremental index failed for {uuid}: {e}")
        return None


def search_photos(
    image_bytes,
    neutral_image_bytes=None,
    description=None,
    query_uuid=None,
    weights=None,
    filters=None,
    exclude=None,
    top_k=12,
    scoring_mode=None,
):
    """Rank reference photos against the incoming photo (plan §2.3, §3.3)."""
    if not image_bytes:
        return {"status": "error", "error": "image (current-look render) is required"}

    top_k = int(top_k or 12)
    weights = weights or dict(STYLE_MATCH_DEFAULT_WEIGHTS)
    scoring_mode = scoring_mode or STYLE_MATCH_SCORING_MODE
    filters = filters or {}
    query_filters = dict(filters)
    # Edited references rank by default (§2.6); the toggle re-enables unedited.
    query_filters.setdefault("has_edits", True)

    query_embedding = server_lifecycle.embed_image(image_bytes)
    if query_embedding is None:
        return {"status": "error", "error": "CLIP embedding failed"}

    query_histogram = colors.compute_color_histogram(neutral_image_bytes) if neutral_image_bytes else None

    description_used = False
    description_vector, _ = _description_embedding_for(
        query_uuid,
        description,
        generate_description=False,
    )
    if description_vector is not None:
        description_used = True

    candidate_k = STYLE_MATCH_CANDIDATE_K
    candidates = postgre_service.query_style_ref_candidates(
        "embedding", query_embedding, candidate_k, filters=query_filters, exclude=exclude
    ) or []

    if description_vector is not None:
        desc_candidates = postgre_service.query_style_ref_candidates(
            "description_embedding",
            description_vector,
            candidate_k,
            filters=query_filters,
            exclude=exclude,
        ) or []
        candidates.extend(desc_candidates)

    # Union by uuid (rows from /index and /style-match/index may carry
    # different synthetic vs real lrc_local_id for the same photo), keeping
    # each candidate's best per-column similarity.
    merged = {}
    for candidate in candidates:
        key = candidate.get("uuid") or candidate["id"]
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue
        for signal in ("sem", "desc"):
            value = candidate.get(signal)
            if value is not None and (existing.get(signal) is None or value > existing[signal]):
                existing[signal] = value
    candidate_list = list(merged.values())

    query_latitude = filters.get("lat")
    query_longitude = filters.get("lon")

    for candidate in candidate_list:
        candidate["color"] = colors.histogram_cosine(query_histogram, candidate.get("color_histogram"))
        distance_km = None
        if query_latitude is not None and query_longitude is not None and candidate.get("latitude") is not None:
            distance_km = _haversine_km(
                float(query_latitude),
                float(query_longitude),
                float(candidate["latitude"]),
                float(candidate["longitude"]),
            )
        candidate["distance_km"] = distance_km
        candidate["loc"] = _location_similarity(distance_km)
        clean_settings, local_count = _strip_develop_settings(candidate.get("develop_settings"))
        candidate["develop_settings"] = clean_settings
        candidate["local_adjustments"] = local_count

    ranks = scoring.rank_candidates(candidate_list)
    for position, candidate in enumerate(candidate_list):
        if scoring_mode == "rrf":
            candidate_ranks = {
                signal: ranks[signal][position]
                for signal in SIGNALS
                if position in ranks.get(signal, {})
            }
            total, weights_used = scoring.rrf_score(candidate_ranks, weights)
        else:
            candidate_signals = {signal: candidate.get(signal) for signal in SIGNALS}
            candidate_signals = {
                signal: value for signal, value in candidate_signals.items() if value is not None
            }
            total, weights_used = scoring.linear_score(candidate_signals, weights)
        candidate["score"] = total
        candidate["weights_used"] = weights_used

    candidate_list.sort(key=lambda candidate: -float(candidate.get("score") or 0.0))
    top = candidate_list[:top_k]

    signals_used = set()
    for candidate in top:
        for signal in SIGNALS:
            if candidate.get(signal) is not None:
                signals_used.add(signal)

    matches = []
    for rank, candidate in enumerate(top, start=1):
        matches.append(
            {
                "rank": rank,
                "lrc_local_id": candidate["lrc_local_id"],
                "catalog_id": candidate["catalog_id"],
                "uuid": candidate.get("uuid"),
                "file_name": candidate.get("file_name"),
                "file_path": candidate.get("file_path"),
                "score": round(float(candidate.get("score") or 0.0), 6),
                "signals": {
                    signal: (
                        round(float(candidate[signal]), 6)
                        if candidate.get(signal) is not None
                        else None
                    )
                    for signal in SIGNALS
                },
                "capture_time": candidate.get("capture_time"),
                "distance_km": (
                    round(float(candidate["distance_km"]), 3)
                    if candidate.get("distance_km") is not None
                    else None
                ),
                "camera_model": candidate.get("camera_model"),
                "has_edits": candidate.get("has_edits"),
                "description": candidate.get("description"),
                "develop_settings": candidate.get("develop_settings"),
                "local_adjustments": candidate.get("local_adjustments", 0),
                "render_source": candidate.get("render_source"),
            }
        )

    return {
        "status": "ok",
        "query": {
            "uuid": query_uuid,
            "description_used": description_used,
            "has_color": query_histogram is not None,
        },
        "signals_used": sorted(signals_used),
        "scoring_mode": scoring_mode,
        "candidate_count": len(candidate_list),
        "matches": matches,
    }
