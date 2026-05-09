import numpy as np
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.types.json import Jsonb

from config import (
    POSTGRE_DATABASE_NAME,
    POSTGRE_PASSWORD,
    POSTGRE_URL,
    POSTGRE_USER,
    TEXT_EMBEDDING_DIMENSION,
    logger,
)


_initialized = False


class PostgreStartupError(RuntimeError):
    pass


def _make_postgre_conninfo(dbname=None):
    overrides = {}
    if dbname:
        overrides["dbname"] = dbname
    if POSTGRE_USER:
        overrides["user"] = POSTGRE_USER
    if POSTGRE_PASSWORD:
        overrides["password"] = POSTGRE_PASSWORD

    return make_conninfo(POSTGRE_URL, **overrides)


def _redacted_postgre_url():
    try:
        conninfo = conninfo_to_dict(_make_postgre_conninfo())
    except Exception:
        return "<invalid PostgreSQL URL>"

    parts = []
    for key in ("host", "hostaddr", "port", "dbname", "user"):
        value = conninfo.get(key)
        if value:
            parts.append(f"{key}={value}")

    if conninfo.get("password"):
        parts.append("password=<redacted>")

    return " ".join(parts) if parts else "<default local PostgreSQL connection>"


def _redact_sensitive_text(value):
    text = str(value)
    if POSTGRE_URL:
        text = text.replace(POSTGRE_URL, _redacted_postgre_url())
    if POSTGRE_PASSWORD:
        text = text.replace(POSTGRE_PASSWORD, "<redacted>")
    return text


def _format_startup_error(error):
    cause = str(error).strip() or error.__class__.__name__
    cause = _redact_sensitive_text(cause)
    return (
        "PostgreSQL initialization failed, so LrGenius Server cannot start.\n"
        f"Connection: {_redacted_postgre_url()}\n"
        f"Database: {POSTGRE_DATABASE_NAME}\n"
        f"Cause: {cause}\n"
        "Please check that PostgreSQL is running, the host/port are reachable, "
        "the credentials are valid, and the pgvector extension is installed."
    )


def describe_connection_target():
    return _redacted_postgre_url()


def _target_conninfo():
    return _make_postgre_conninfo(dbname=POSTGRE_DATABASE_NAME)


def _maintenance_conninfo():
    conninfo = conninfo_to_dict(_make_postgre_conninfo())
    return _make_postgre_conninfo(dbname=conninfo.get("dbname") or "postgres")


def _connect_to_target():
    return psycopg.connect(_target_conninfo())


def _ensure_initialized():
    global _initialized
    if _initialized:
        return

    logger.info(f"Initializing PostgreSQL database '{POSTGRE_DATABASE_NAME}' with pgvector...")
    with psycopg.connect(_maintenance_conninfo(), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (POSTGRE_DATABASE_NAME,),
        ).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(POSTGRE_DATABASE_NAME)))
            logger.info(f"Created PostgreSQL database '{POSTGRE_DATABASE_NAME}'.")

    with _connect_to_target() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS photo_metadata (
                    uuid TEXT PRIMARY KEY,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    document TEXT,
                    embedding vector({dimension}),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            ).format(dimension=sql.Literal(TEXT_EMBEDDING_DIMENSION))
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS photo_metadata_metadata_gin_idx
            ON photo_metadata USING gin (metadata)
            """
        )

    try:
        with _connect_to_target() as conn:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS photo_metadata_embedding_hnsw_idx
                ON photo_metadata USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL
                """
            )
    except psycopg.Error as e:
        logger.warning(
            "Could not create pgvector HNSW index. Semantic search will still work, "
            f"but may be slower until the index is created manually: {e}"
        )
        with _connect_to_target() as conn:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS photo_metadata_embedding_ivfflat_idx
                ON photo_metadata USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                WHERE embedding IS NOT NULL
                """
            )

    _initialized = True


def initialize():
    try:
        _ensure_initialized()
    except (psycopg.Error, ValueError) as e:
        raise PostgreStartupError(_format_startup_error(e)) from e


def _embedding_literal(embedding):
    if embedding is None:
        return None

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape[0] != TEXT_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: expected {TEXT_EMBEDDING_DIMENSION}, got {vector.shape[0]}"
        )
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _embedding_from_text(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    stripped = str(value).strip()
    if not stripped:
        return None
    return [float(item) for item in stripped.strip("[]").split(",") if item]


def _result(ids, metadatas=None, embeddings=None, documents=None, distances=None, grouped=False):
    if grouped:
        payload = {"ids": [ids]}
        if metadatas is not None:
            payload["metadatas"] = [metadatas]
        if embeddings is not None:
            payload["embeddings"] = [embeddings]
        if documents is not None:
            payload["documents"] = [documents]
        if distances is not None:
            payload["distances"] = [distances]
        return payload

    payload = {"ids": ids}
    if metadatas is not None:
        payload["metadatas"] = metadatas
    if embeddings is not None:
        payload["embeddings"] = embeddings
    if documents is not None:
        payload["documents"] = documents
    if distances is not None:
        payload["distances"] = distances
    return payload


def _upsert_record(uuid, metadata, embedding=None, document=None, update_embedding=True):
    embedding_value = _embedding_literal(embedding)
    metadata = metadata or {}

    with _connect_to_target() as conn:
        if update_embedding:
            conn.execute(
                """
                INSERT INTO photo_metadata (uuid, metadata, document, embedding)
                VALUES (%s, %s, %s, %s::vector)
                ON CONFLICT (uuid) DO UPDATE SET
                    metadata = EXCLUDED.metadata,
                    document = COALESCE(EXCLUDED.document, photo_metadata.document),
                    embedding = EXCLUDED.embedding,
                    updated_at = now()
                """,
                (uuid, Jsonb(metadata), document, embedding_value),
            )
        else:
            conn.execute(
                """
                INSERT INTO photo_metadata (uuid, metadata, document)
                VALUES (%s, %s, %s)
                ON CONFLICT (uuid) DO UPDATE SET
                    metadata = EXCLUDED.metadata,
                    document = COALESCE(EXCLUDED.document, photo_metadata.document),
                    updated_at = now()
                """,
                (uuid, Jsonb(metadata), document),
            )


def add_image(uuid, embedding, metadata, document=None):
    _ensure_initialized()
    try:
        _upsert_record(uuid, metadata, embedding=embedding, document=document, update_embedding=True)
        logger.debug(f"image {uuid} is well UPSERTED in PostgreSQL. Metadata = {metadata}")
    except Exception as e:
        logger.error(
            f"Failed to add image {uuid} to PostgreSQL (embedding provided: {embedding is not None}): {e}",
            exc_info=True,
        )
        raise


def update_image(uuid, metadata, embedding=None, document=None):
    _ensure_initialized()
    _upsert_record(
        uuid,
        metadata,
        embedding=embedding,
        document=document,
        update_embedding=embedding is not None,
    )
    logger.debug(f"image {uuid} is well UPSERTED in PostgreSQL. Metadata = {metadata}")


def get_image(uuid):
    _ensure_initialized()
    with _connect_to_target() as conn:
        row = conn.execute(
            """
            SELECT uuid, metadata, document, embedding::text AS embedding
            FROM photo_metadata
            WHERE uuid = %s
            """,
            (uuid,),
        ).fetchone()

    if row is None:
        return None

    return _result(
        [row[0]],
        metadatas=[row[1] or {}],
        embeddings=[_embedding_from_text(row[3])],
        documents=[row[2]],
    )


def delete_image(uuid):
    _ensure_initialized()
    with _connect_to_target() as conn:
        conn.execute("DELETE FROM photo_metadata WHERE uuid = %s", (uuid,))


def _uuid_filter(where_clause):
    if not where_clause:
        return None

    uuid_clause = where_clause.get("uuid")
    if isinstance(uuid_clause, dict) and "$in" in uuid_clause:
        return list(uuid_clause["$in"])
    if isinstance(uuid_clause, str):
        return [uuid_clause]
    return None


def query_images(query_embedding, n_results, where_clause=None, include_embeddings=False):
    _ensure_initialized()
    query_vector = _embedding_literal(query_embedding)
    uuid_filter = _uuid_filter(where_clause)

    try:
        select_embedding = ", embedding::text AS embedding" if include_embeddings else ""
        params = [query_vector]
        filter_sql = sql.SQL("")
        if uuid_filter:
            filter_sql = sql.SQL("AND uuid = ANY(%s)")
            params.append(uuid_filter)
        params.append(n_results)

        query = sql.SQL(
            """
            SELECT uuid, metadata, embedding <=> %s::vector AS distance {select_embedding}
            FROM photo_metadata
            WHERE embedding IS NOT NULL
            AND COALESCE((metadata->>'has_embedding')::boolean, true)
            {filter_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(
            select_embedding=sql.SQL(select_embedding),
            filter_sql=filter_sql,
        )
        params.insert(-1, query_vector)

        with _connect_to_target() as conn:
            rows = conn.execute(query, params).fetchall()

        ids = []
        metadatas = []
        distances = []
        embeddings = [] if include_embeddings else None
        for row in rows:
            ids.append(row[0])
            metadatas.append(row[1] or {})
            distances.append(float(row[2]) if row[2] is not None else None)
            if include_embeddings:
                embeddings.append(_embedding_from_text(row[3]))

        return _result(ids, metadatas=metadatas, embeddings=embeddings, distances=distances, grouped=True)
    except Exception as e:
        logger.error(f"Error querying images: {e}", exc_info=True)
        fallback = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        if include_embeddings:
            fallback["embeddings"] = [[]]
        return fallback


def get_image_metadatas(ids=None):
    _ensure_initialized()
    with _connect_to_target() as conn:
        if ids:
            rows = conn.execute(
                """
                SELECT uuid, metadata
                FROM photo_metadata
                WHERE uuid = ANY(%s)
                ORDER BY uuid
                """,
                (list(ids),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT uuid, metadata
                FROM photo_metadata
                ORDER BY uuid
                """
            ).fetchall()

    return _result([row[0] for row in rows], metadatas=[row[1] or {} for row in rows])


def get_all_image_ids(has_embedding=None):
    _ensure_initialized()
    with _connect_to_target() as conn:
        if has_embedding is None:
            rows = conn.execute("SELECT uuid FROM photo_metadata ORDER BY uuid").fetchall()
        else:
            rows = conn.execute(
                """
                SELECT uuid
                FROM photo_metadata
                WHERE COALESCE((metadata->>'has_embedding')::boolean, true) = %s
                ORDER BY uuid
                """,
                (has_embedding,),
            ).fetchall()

    return [row[0] for row in rows]


def group_and_sort_images(uuids, phash_threshold, clip_threshold, time_delta):
    logger.warning("group_and_sort_images is not yet implemented.")
    return []


def get_db_stats():
    _ensure_initialized()
    with _connect_to_target() as conn:
        count = conn.execute("SELECT count(*) FROM photo_metadata").fetchone()[0]
        db_size = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0] / (1024 * 1024)
        stats = conn.execute(
            """
            SELECT
                count(*) FILTER (WHERE metadata ? 'phash') AS phash_count,
                count(*) FILTER (WHERE metadata ? 'capture_time') AS capture_time_count,
                count(*) FILTER (WHERE metadata ? 'overall_score') AS aesthetic_rated_count
            FROM photo_metadata
            """
        ).fetchone()

    return {
        "database": POSTGRE_DATABASE_NAME,
        "num_images": count,
        "db_size_mb": round(db_size, 2),
        "num_with_phash": stats[0],
        "num_rated_aesthetic": stats[2],
        "num_with_capture_time": stats[1],
    }
