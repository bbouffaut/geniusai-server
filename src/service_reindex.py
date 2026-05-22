"""
Re-index service: re-computes embeddings for photos already stored in the DB.

No LLM calls are made — this only re-runs the local embedding model on text
that is already present in the metadata JSONB column (caption and keywords).
This lets you refresh embeddings after:
  - Switching to a new embedding model (requires full re-index from scratch)
  - Activating the dual-embedding feature (new embedding_kw column)
  - Changing which fields are used for each embedding vector

The public entry-point is ``reindex_embeddings_stream``, a generator that
yields one dict per processed photo plus a final summary line, allowing the
caller to stream results back to the HTTP client as NDJSON.
"""

import json
from typing import Generator

import server_lifecycle
import service_postgre as postgre_service
from config import TEXT_EMBEDDING_MODEL_ID, logger
from service_index import (
    _build_keywords_document,
    _build_metadata_embedding_document,
    _build_prose_document,
    _flatten_keywords,
)


def reindex_embeddings_stream(
    uuids=None,
    recompute_prose=True,
    recompute_kw=True,
) -> Generator[dict, None, None]:
    """
    Generator that re-computes prose and/or keyword embeddings for photos
    already in the DB, yielding one status dict per photo.

    Yielded event shapes
    --------------------
    Start (first):
        {"event": "start", "total": <int>, "prose": <bool>, "kw": <bool>}

    Per photo — success:
        {"event": "indexed", "uuid": "…", "filename": "…",
         "prose": <bool>, "kw": <bool>, "index": <int>, "total": <int>}
        ``prose`` / ``kw`` indicate which embeddings were actually updated.

    Per photo — skipped (no usable source text):
        {"event": "skipped", "uuid": "…", "filename": "…",
         "index": <int>, "total": <int>}

    Per photo — error:
        {"event": "error", "uuid": "…", "filename": "…",
         "error": "…", "index": <int>, "total": <int>}

    Done (last):
        {"event": "done", "total": <int>, "success_count": <int>,
         "skipped_count": <int>, "failure_count": <int>}

    Args:
        uuids (list[str] | None): UUIDs to process. None means all photos.
        recompute_prose (bool): Rebuild the ``embedding`` column from caption.
        recompute_kw (bool): Rebuild the ``embedding_kw`` column from keywords.
    """
    if not recompute_prose and not recompute_kw:
        logger.info("Re-index: nothing to do (both prose and kw disabled).")
        yield {"event": "done", "total": 0,
               "success_count": 0, "skipped_count": 0, "failure_count": 0}
        return

    raw = postgre_service.get_image_metadatas(ids=uuids if uuids else None)
    all_ids = raw.get("ids", [])
    all_metadatas = raw.get("metadatas", [])
    total = len(all_ids)

    logger.info(
        f"Re-index starting: {total} photo(s) — "
        f"prose={recompute_prose}, embedding_kw={recompute_kw}"
    )
    yield {"event": "start", "total": total,
           "prose": recompute_prose, "kw": recompute_kw}

    success_count = 0
    skipped_count = 0
    failure_count = 0

    for i, (uuid, raw_metadata) in enumerate(zip(all_ids, all_metadatas)):
        index = i + 1
        metadata = dict(raw_metadata or {})
        filename = metadata.get("filename") or ""

        try:
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
            prose_updated = False
            kw_updated = False

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
                        prose_updated = True
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
                        kw_updated = True
                    else:
                        logger.warning(f"Re-index: keyword embedding failed for {uuid}.")
                else:
                    logger.debug(f"Re-index: no keyword text for {uuid} — kw skipped.")

            if not prose_updated and not kw_updated:
                skipped_count += 1
                yield {"event": "skipped", "uuid": uuid, "filename": filename,
                       "index": index, "total": total}
                continue

            # Update has_embedding: preserve True if the photo was already
            # embedded, and set True if we just produced a new vector.
            existing_has_embedding = bool(metadata.get("has_embedding", False))
            metadata["has_embedding"] = (
                existing_has_embedding
                or prose_updated
                or kw_updated
            )

            # Rebuild the full-text document column (for SQL ILIKE search).
            document = _build_metadata_embedding_document(metadata) or None

            postgre_service.update_image(
                uuid,
                metadata,
                embedding=prose_embedding,   # None keeps the existing vector
                embedding_kw=kw_embedding,   # None keeps the existing vector
                document=document,
            )

            success_count += 1
            yield {
                "event": "indexed",
                "uuid": uuid,
                "filename": filename,
                "prose": prose_updated,
                "kw": kw_updated,
                "index": index,
                "total": total,
            }

        except Exception as e:
            logger.error(f"Re-index: unexpected error for {uuid}: {e}", exc_info=True)
            failure_count += 1
            yield {"event": "error", "uuid": uuid, "filename": filename,
                   "error": str(e), "index": index, "total": total}

    logger.info(
        f"Re-index done — updated={success_count}, "
        f"skipped={skipped_count}, failed={failure_count}, total={total}"
    )
    yield {
        "event": "done",
        "total": total,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
    }
