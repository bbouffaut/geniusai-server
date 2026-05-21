#!/usr/bin/env python3
"""
Re-index photo embeddings from one PostgreSQL database to another
using a (potentially different) embedding model.

Source database and connection details are read from the environment
(typically set by a dotenv file loaded by migrate.sh).

Usage (via migrate.sh):
    ./migrate.sh --dotenv <env_file> --target-db <db> --target-model <model>

Direct invocation (env vars must be set beforehand):
    GENIUSAI_DATABASE_NAME=src_db GENIUSAI_POSTGRES_URL=... \\
        python src/migrate.py --target-db <db> --target-model <model>
"""

import argparse
import sys
import os

# ──────────────────────────────────────────────────────────────────────────────
# Read source-side configuration from the environment.
# These are set by migrate.sh after loading the dotenv file, so they are
# available here before any project module is imported.
# ──────────────────────────────────────────────────────────────────────────────
_SOURCE_DB       = os.environ.get("GENIUSAI_DATABASE_NAME")
_POSTGRE_URL     = os.environ.get("GENIUSAI_POSTGRES_URL", "postgresql://localhost:5432/postgres")
_POSTGRE_USER    = os.environ.get("GENIUSAI_POSTGRES_USER")
_POSTGRE_PASSWORD = os.environ.get("GENIUSAI_POSTGRES_PASSWORD")
_MODEL_CACHE_PATH = os.environ.get("MODEL_CACHE_PATH")  # same var as run.sh

if not _SOURCE_DB:
    print(
        "error: GENIUSAI_DATABASE_NAME is not set.\n"
        "Use --dotenv <file> with migrate.sh, or export GENIUSAI_DATABASE_NAME before calling this script.",
        file=sys.stderr,
    )
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Parse migration-specific arguments BEFORE importing any project module.
# config.py runs argparse at module-level; sys.argv must be patched first.
# ──────────────────────────────────────────────────────────────────────────────
_mig_parser = argparse.ArgumentParser(
    description="Migrate photo embeddings to a new model and/or target database",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_mig_parser.add_argument(
    "--target-db", required=True, metavar="DB",
    help="Target PostgreSQL database name (created if absent)",
)
_mig_parser.add_argument(
    "--target-model", required=True, metavar="MODEL",
    help="Embedding model key for the target database (see config.py EMBEDDING_MODELS)",
)
_mig_parser.add_argument(
    "--batch-size", type=int, default=32, metavar="N",
    help="Number of photos to embed per GPU batch",
)
mig_args = _mig_parser.parse_args()

# Build a synthetic sys.argv that satisfies config.py's required flags.
# config.py is pointed at the TARGET database/model because service_postgre
# writes to the target.
# --fetch-models is always enabled: migration may use a model not yet cached,
# so downloading must be allowed unconditionally.
sys.argv = [
    sys.argv[0],
    "--embedding-model", mig_args.target_model,
    "--database-name",   mig_args.target_db,
    "--postgre-url",     _POSTGRE_URL,
    "--fetch-models",
]
if _POSTGRE_USER:
    sys.argv += ["--postgre-user", _POSTGRE_USER]
if _POSTGRE_PASSWORD:
    sys.argv += ["--postgre-password", _POSTGRE_PASSWORD]
if _MODEL_CACHE_PATH:
    sys.argv += ["--model-cache-path", _MODEL_CACHE_PATH]

# Ensure src/ is on the module search path when called from the repo root.
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Safe to import project modules now (config.py will parse the patched sys.argv).
import config           # noqa: E402
import service_postgre  # noqa: E402
import server_lifecycle # noqa: E402

import psycopg                              # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

logger = config.logger


def _source_conninfo() -> str:
    overrides: dict = {"dbname": _SOURCE_DB}
    if _POSTGRE_USER:
        overrides["user"] = _POSTGRE_USER
    if _POSTGRE_PASSWORD:
        overrides["password"] = _POSTGRE_PASSWORD
    return make_conninfo(_POSTGRE_URL, **overrides)


def _iter_source_pages(page_size: int):
    """Yield pages of (uuid, metadata_dict, document) from the source database."""
    with psycopg.connect(_source_conninfo()) as conn:
        total: int = conn.execute("SELECT COUNT(*) FROM photo_metadata").fetchone()[0]
        logger.info(f"Source '{_SOURCE_DB}': {total} photos to migrate.")
        offset = 0
        while True:
            rows = conn.execute(
                "SELECT uuid, metadata, document FROM photo_metadata "
                "ORDER BY uuid LIMIT %s OFFSET %s",
                (page_size, offset),
            ).fetchall()
            if not rows:
                break
            yield [(r[0], r[1] or {}, r[2]) for r in rows]
            offset += len(rows)


def _existing_uuids(uuids: list) -> set:
    """Return the subset of *uuids* that already exist in the target database."""
    if not uuids:
        return set()
    with service_postgre._connect_to_target() as conn:
        rows = conn.execute(
            "SELECT uuid FROM photo_metadata WHERE uuid = ANY(%s)",
            (uuids,),
        ).fetchall()
    return {row[0] for row in rows}


def _flush(batch: list) -> tuple:
    """Embed and upsert one batch into the target database.

    Photos whose UUID already exists in the target are skipped entirely.
    Returns (inserted, no_doc_count, failed, skipped) counts for the batch.
    """
    existing = _existing_uuids([uuid for uuid, _, _ in batch])
    skipped  = len(existing)
    if existing:
        batch = [(uuid, meta, doc) for uuid, meta, doc in batch if uuid not in existing]

    with_doc    = [(uuid, meta, doc) for uuid, meta, doc in batch if doc]
    without_doc = [(uuid, meta)      for uuid, meta, doc in batch if not doc]

    embeddings_by_uuid: dict = {}
    if with_doc:
        texts = [doc for _, _, doc in with_doc]
        results = server_lifecycle.embed_texts(texts, input_type="document")
        if results is not None:
            for idx, (uuid, _, _) in enumerate(with_doc):
                embeddings_by_uuid[uuid] = results[idx]
        else:
            logger.warning("embed_texts returned None for a batch; photos will be stored without embeddings.")

    inserted = 0
    failed   = 0

    for uuid, meta, doc in with_doc:
        embedding = embeddings_by_uuid.get(uuid)
        updated_meta = {
            **meta,
            "embedding_model":  config.TEXT_EMBEDDING_MODEL_ID,
            "embedding_source": mig_args.target_model,
            "has_embedding":    embedding is not None,
        }
        try:
            service_postgre.add_image(uuid, embedding, updated_meta, document=doc)
            inserted += 1
        except Exception as exc:
            logger.error(f"Failed to insert {uuid}: {exc}")
            failed += 1

    for uuid, meta in without_doc:
        updated_meta = {**meta, "has_embedding": False}
        try:
            service_postgre.add_image(uuid, None, updated_meta, document=None)
            inserted += 1
        except Exception as exc:
            logger.error(f"Failed to insert {uuid} (no document): {exc}")
            failed += 1

    return inserted, len(without_doc), failed, skipped


def migrate() -> None:
    logger.info(
        f"Starting migration: '{_SOURCE_DB}' → '{mig_args.target_db}' "
        f"| model: '{mig_args.target_model}' | batch size: {mig_args.batch_size}"
    )

    if _SOURCE_DB == mig_args.target_db:
        logger.warning(
            "Source and target databases are the same. "
            "Embeddings will be recomputed in place."
        )

    # Create target database and schema if they do not exist yet.
    service_postgre._ensure_initialized()

    # Load the embedding model upfront so the model-loading message appears before processing.
    logger.info("Loading embedding model…")
    server_lifecycle.load_model()
    logger.info("Embedding model ready.")

    total_inserted = 0
    total_no_doc   = 0
    total_failed   = 0
    total_skipped  = 0

    for page in _iter_source_pages(mig_args.batch_size):
        inserted, no_doc, failed, skipped = _flush(page)
        total_inserted += inserted
        total_no_doc   += no_doc
        total_failed   += failed
        total_skipped  += skipped
        logger.info(
            f"Progress — inserted: {total_inserted}, "
            f"without-embedding: {total_no_doc}, skipped: {total_skipped}, failed: {total_failed}"
        )

    logger.info(
        f"Migration complete: {total_inserted} photos copied "
        f"({total_no_doc} stored without embeddings), {total_skipped} already present skipped, "
        f"{total_failed} failed."
    )
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    migrate()
