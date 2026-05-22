"""
Re-index service: re-computes embeddings for photos already stored in the DB.

No LLM calls are made — this only re-runs the local embedding model on text
that is already present in the metadata JSONB column (caption and keywords).
This lets you refresh embeddings after:
  - Switching to a new embedding model (requires full re-index from scratch)
  - Activating the dual-embedding feature (new embedding_kw column)
  - Changing which fields are used for each embedding vector
"""

import json

import server_lifecycle
import service_postgre as postgre_service
from config import TEXT_EMBEDDING_MODEL_ID, logger
from service_index import (
    _build_keywords_document,
    _build_metadata_embedding_document,
    _build_prose_document,
    _flatten_keywords,
)


def reindex_embeddings(
    uuids=None,
    recompute_prose=True,
    recompute_kw=True,
):
    """
    Re-compute prose and/or keyword embeddings for photos already in the DB.

    Reads each photo's stored metadata (caption, flattened_keywords, etc.),
    runs the embedding model locally, and writes the updated vectors back.
    The metadata JSONB is also updated with housekeeping fields
    (embedding_source, embedding_model, has_embedding, …).

    Args:
        uuids (list[str] | None): UUIDs to process. None means all photos.
        recompute_prose (bool): Rebuild the ``embedding`` column from caption.
        recompute_kw (bool): Rebuild the ``embedding_kw`` column from keywords.

    Returns:
        dict with keys: total, success_count, skipped_count, failure_count.
             skipped means the photo had no usable source text for the
             requested embedding type(s) — nothing was written to the DB.
    """
    if not recompute_prose and not recompute_kw:
        logger.info("Re-index: nothing to do (both prose and kw are disabled).")
        return {"total": 0, "success_count": 0, "skipped_count": 0, "failure_count": 0}

    # Fetch records from DB — no embeddings needed, just metadata JSONB.
    raw = postgre_service.get_image_metadatas(ids=uuids if uuids else None)
    all_ids = raw.get("ids", [])
    all_metadatas = raw.get("metadatas", [])
    total = len(all_ids)

    logger.info(
        f"Re-index starting: {total} photo(s) — "
        f"prose={recompute_prose}, embedding_kw={recompute_kw}"
    )

    success_count = 0
    skipped_count = 0
    failure_count = 0

    for i, (uuid, raw_metadata) in enumerate(zip(all_ids, all_metadatas)):
        try:
            metadata = dict(raw_metadata or {})

            # Decode keywords JSON string if needed so helpers can flatten it.
            kw_raw = metadata.get("keywords")
            if isinstance(kw_raw, str) and kw_raw.strip():
                try:
                    metadata["keywords"] = json.loads(kw_raw)
                except json.JSONDecodeError:
                    pass  # keep as-is; _flatten_keywords handles strings too

            # Ensure flattened_keywords exists for _build_keywords_document.
            if not metadata.get("flattened_keywords") and metadata.get("keywords"):
                metadata["flattened_keywords"] = _flatten_keywords(metadata["keywords"])

            prose_embedding = None
            kw_embedding = None
            updated = False

            # ----------------------------------------------------------------
            # Prose embedding — built from caption only
            # ----------------------------------------------------------------
            if recompute_prose:
                prose_doc = _build_prose_document(metadata)
                if prose_doc:
                    prose_embedding = server_lifecycle.embed_document(prose_doc)
                    if prose_embedding is not None:
                        metadata["metadata_search_text"] = prose_doc
                        metadata["embedding_source"] = "prose"
                        metadata["embedding_model"] = TEXT_EMBEDDING_MODEL_ID
                        updated = True
                    else:
                        logger.warning(f"Re-index: prose embedding failed for {uuid}.")
                else:
                    logger.debug(f"Re-index: no caption text for {uuid} — prose skipped.")

            # ----------------------------------------------------------------
            # Keyword embedding — built from flattened_keywords
            # ----------------------------------------------------------------
            if recompute_kw:
                kw_doc = _build_keywords_document(metadata)
                if kw_doc:
                    kw_embedding = server_lifecycle.embed_document(kw_doc)
                    if kw_embedding is not None:
                        metadata["keywords_search_text"] = kw_doc
                        metadata["embedding_kw_source"] = "keywords"
                        if not metadata.get("embedding_model"):
                            metadata["embedding_model"] = TEXT_EMBEDDING_MODEL_ID
                        updated = True
                    else:
                        logger.warning(f"Re-index: keyword embedding failed for {uuid}.")
                else:
                    logger.debug(f"Re-index: no keyword text for {uuid} — kw skipped.")

            if not updated:
                skipped_count += 1
                continue

            # Update has_embedding: preserve True if the photo was already
            # embedded (existing value), and set True if we just produced one.
            existing_has_embedding = bool(metadata.get("has_embedding", False))
            metadata["has_embedding"] = (
                existing_has_embedding
                or (prose_embedding is not None)
                or (kw_embedding is not None)
            )

            # Rebuild the full-text document column (for SQL ILIKE search).
            document = _build_metadata_embedding_document(metadata) or None

            postgre_service.update_image(
                uuid,
                metadata,
                embedding=prose_embedding,      # None keeps the existing vector
                embedding_kw=kw_embedding,      # None keeps the existing vector
                document=document,
            )
            success_count += 1

            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(f"Re-index progress: {i + 1}/{total}")

        except Exception as e:
            logger.error(f"Re-index: unexpected error for {uuid}: {e}", exc_info=True)
            failure_count += 1

    logger.info(
        f"Re-index done — updated={success_count}, "
        f"skipped={skipped_count}, failed={failure_count}, total={total}"
    )
    return {
        "total": total,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
    }
