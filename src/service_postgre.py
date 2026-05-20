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


CAPTURE_TIME_METADATA_PATHS = [
    ["capture_time"],
    ["exif", "capture_time"],
    ["exif", "DateTimeOriginal"],
    ["exif", "date_time_original"],
    ["exif", "CreateDate"],
    ["exif", "create_date"],
]

APERTURE_METADATA_PATHS = [
    ["aperture"],
    ["aperture_f_number"],
    ["f_number"],
    ["fnumber"],
    ["f_stop"],
    ["fstop"],
    ["exif", "aperture"],
    ["exif", "aperture_f_number"],
    ["exif", "f_number"],
    ["exif", "fnumber"],
    ["exif", "f_stop"],
    ["exif", "fstop"],
    ["exif", "FNumber"],
    ["exif", "Aperture"],
    ["exif", "ApertureValue"],
]


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
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
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


def _metadata_filter_entries(where_clause):
    if not where_clause:
        return []

    filters = where_clause.get("metadata_filters")
    if filters is None:
        filters = where_clause.get("filters")
    if filters is None:
        return []
    if isinstance(filters, dict):
        filters = [filters]
    if not isinstance(filters, list):
        return []

    return [item for item in filters if isinstance(item, dict)]


def _metadata_paths_for_filter(filter_item):
    raw_paths = filter_item.get("paths")
    if isinstance(raw_paths, list) and raw_paths:
        paths = []
        for path in raw_paths:
            if isinstance(path, str):
                paths.append([path])
            elif isinstance(path, list) and path:
                paths.append([str(part) for part in path])
        if paths:
            return paths

    field = str(filter_item.get("field", "")).strip().casefold()
    if field == "capture_time":
        return CAPTURE_TIME_METADATA_PATHS
    if field in {"aperture", "aperture_f_number", "f_number", "fnumber", "f_stop", "fstop"}:
        return APERTURE_METADATA_PATHS
    if field:
        return [[field]]
    return []


def _metadata_path_expr():
    return sql.SQL("metadata #>> %s::text[]")


def _metadata_number_expr():
    return sql.SQL(
        "NULLIF(substring({value_expr} from '[-+]?[0-9]+\\.?[0-9]*'), '')::double precision"
    ).format(value_expr=_metadata_path_expr())


def _metadata_text_filter_clause(filter_item):
    paths = _metadata_paths_for_filter(filter_item)
    if not paths:
        return None, []

    op = str(filter_item.get("op", "eq")).casefold()
    value = filter_item.get("value")
    if value is None:
        return None, []

    operator_map = {
        "eq": "=",
        "equals": "=",
        "gte": ">=",
        "gt": ">",
        "lte": "<=",
        "lt": "<",
    }
    operator = operator_map.get(op)
    if operator is None:
        return None, []

    clauses = []
    params = []
    for path in paths:
        clauses.append(
            sql.SQL("({value_expr} {operator} %s)").format(
                value_expr=_metadata_path_expr(),
                operator=sql.SQL(operator),
            )
        )
        params.extend([path, str(value)])

    return sql.SQL("(") + sql.SQL(" OR ").join(clauses) + sql.SQL(")"), params


def _metadata_numeric_filter_clause(filter_item):
    paths = _metadata_paths_for_filter(filter_item)
    if not paths:
        return None, []

    try:
        value = float(filter_item.get("value"))
    except (TypeError, ValueError):
        return None, []

    op = str(filter_item.get("op", "number_eq")).casefold()
    clauses = []
    params = []

    if op in {"number_eq", "eq", "equals"}:
        tolerance = filter_item.get("tolerance", 0.0)
        try:
            tolerance = abs(float(tolerance))
        except (TypeError, ValueError):
            tolerance = 0.0
        lower_bound = value - tolerance
        upper_bound = value + tolerance
        for path in paths:
            clauses.append(
                sql.SQL("({value_expr} BETWEEN %s AND %s)").format(
                    value_expr=_metadata_number_expr(),
                )
            )
            params.extend([path, lower_bound, upper_bound])
    else:
        operator_map = {
            "gte": ">=",
            "gt": ">",
            "lte": "<=",
            "lt": "<",
        }
        operator = operator_map.get(op)
        if operator is None:
            return None, []

        for path in paths:
            clauses.append(
                sql.SQL("({value_expr} {operator} %s)").format(
                    value_expr=_metadata_number_expr(),
                    operator=sql.SQL(operator),
                )
            )
            params.extend([path, value])

    return sql.SQL("(") + sql.SQL(" OR ").join(clauses) + sql.SQL(")"), params


def _metadata_filter_clause(filter_item):
    field = str(filter_item.get("field", "")).strip().casefold()
    op = str(filter_item.get("op", "")).casefold()
    if field in {"aperture", "aperture_f_number", "f_number", "fnumber", "f_stop", "fstop"} or op.startswith("number"):
        return _metadata_numeric_filter_clause(filter_item)
    return _metadata_text_filter_clause(filter_item)


def _filter_clauses(where_clause):
    clauses = []
    params = []

    uuid_filter = _uuid_filter(where_clause)
    if uuid_filter:
        clauses.append(sql.SQL("uuid = ANY(%s)"))
        params.append(uuid_filter)

    for filter_item in _metadata_filter_entries(where_clause):
        clause, clause_params = _metadata_filter_clause(filter_item)
        if clause is None:
            continue
        clauses.append(clause)
        params.extend(clause_params)

    return clauses, params


def _filter_sql(where_clause, prefix):
    clauses, params = _filter_clauses(where_clause)
    if not clauses:
        return sql.SQL(""), []

    return sql.SQL(f"{prefix} ") + sql.SQL(" AND ").join(clauses), params


def query_images(query_embedding, n_results, where_clause=None, include_embeddings=False):
    _ensure_initialized()
    query_vector = _embedding_literal(query_embedding)

    try:
        select_embedding = ", embedding::text AS embedding" if include_embeddings else ""
        filter_sql, filter_params = _filter_sql(where_clause, "AND")
        params = [query_vector, *filter_params, query_vector, n_results]

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


def get_image_metadatas(ids=None, where_clause=None):
    _ensure_initialized()
    effective_where_clause = dict(where_clause or {})
    if ids:
        effective_where_clause["uuid"] = {"$in": list(ids)}

    filter_sql, filter_params = _filter_sql(effective_where_clause, "WHERE")

    with _connect_to_target() as conn:
        rows = conn.execute(
            sql.SQL(
                """
                SELECT uuid, metadata
                FROM photo_metadata
                {filter_sql}
                ORDER BY uuid
                """
            ).format(filter_sql=filter_sql),
            filter_params,
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


def group_and_sort_images(uuids, phash_threshold, time_delta):
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
