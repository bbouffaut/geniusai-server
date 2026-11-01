import json
import re
from datetime import datetime, timezone

import numpy as np
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.types.json import Jsonb

from config import (
    CLIP_EMBEDDING_DIMENSION,
    DEBUG_LEVEL,
    POSTGRE_DATABASE_NAME,
    POSTGRE_PASSWORD,
    POSTGRE_URL,
    POSTGRE_USER,
    TEXT_EMBEDDING_DIMENSION,
    logger,
)


_initialized = False


NORMALIZED_METADATA_COLUMNS = [
    ("capture_time", "TIMESTAMP"),
    ("aperture_f_number", "DOUBLE PRECISION"),
    ("shutter_speed", "DOUBLE PRECISION"),
    ("exposure_bias", "DOUBLE PRECISION"),
    ("iso", "INTEGER"),
    ("focal_length_mm", "DOUBLE PRECISION"),
    ("focal_length_35mm", "DOUBLE PRECISION"),
    ("camera_make", "TEXT"),
    ("camera_model", "TEXT"),
    ("lens", "TEXT"),
    ("gps_latitude", "DOUBLE PRECISION"),
    ("gps_longitude", "DOUBLE PRECISION"),
]

NORMALIZED_METADATA_COLUMN_NAMES = [column for column, _ in NORMALIZED_METADATA_COLUMNS]

CAPTURE_TIME_METADATA_PATHS = [
    ["capture_time"],
    ["exif", "capture_time"],
    ["exif", "DateTimeOriginal"],
    ["exif", "date_time_original"],
    ["exif", "CreateDate"],
    ["exif", "create_date"],
    # Lightroom classic catalog format
    ["exif", "all_raw", "dateTimeOriginalISO8601"],
    ["exif", "raw", "dateTimeOriginalISO8601"],
    ["exif", "all_raw", "dateTimeISO8601"],
    ["exif", "raw", "dateTimeISO8601"],
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
    # Lightroom classic catalog format (numeric value in all_raw/raw)
    ["exif", "all_raw", "aperture"],
    ["exif", "raw", "aperture"],
]

APERTURE_VALUE_METADATA_PATHS = [
    ["aperture_value"],
    ["exif", "aperture_value"],
    ["exif", "ApertureValue"],
]

ISO_METADATA_PATHS = [
    ["iso"],
    ["iso_speed_rating"],
    ["iso_speed_ratings"],
    ["exif", "iso"],
    ["exif", "ISO"],
    ["exif", "ISOSpeedRatings"],
    ["exif", "iso_speed_rating"],
    ["exif", "iso_speed_ratings"],
    # Lightroom classic catalog format
    ["exif", "all_raw", "isoSpeedRating"],
    ["exif", "raw", "isoSpeedRating"],
]

FOCAL_LENGTH_METADATA_PATHS = [
    ["focal_length_mm"],
    ["focal_length"],
    ["exif", "focal_length_mm"],
    ["exif", "focal_length"],
    ["exif", "FocalLength"],
    # Lightroom classic catalog format
    ["exif", "all_raw", "focalLength"],
    ["exif", "raw", "focalLength"],
]

FOCAL_LENGTH_35MM_METADATA_PATHS = [
    ["focal_length_35mm"],
    ["focalLength35mm"],
    ["exif", "FocalLengthIn35mmFilm"],
    # Lightroom classic catalog format
    ["exif", "all_raw", "focalLength35mm"],
    ["exif", "raw", "focalLength35mm"],
]

SHUTTER_SPEED_METADATA_PATHS = [
    ["shutter_speed"],
    ["shutterSpeed"],
    ["exif", "ExposureTime"],
    ["exif", "ShutterSpeedValue"],
    # Lightroom classic catalog format (value in seconds as float)
    ["exif", "all_raw", "shutterSpeed"],
    ["exif", "raw", "shutterSpeed"],
]

EXPOSURE_BIAS_METADATA_PATHS = [
    ["exposure_bias"],
    ["exposureBias"],
    ["exif", "ExposureBiasValue"],
    # Lightroom classic catalog format
    ["exif", "all_raw", "exposureBias"],
    ["exif", "raw", "exposureBias"],
]

CAMERA_MAKE_METADATA_PATHS = [
    ["camera_make"],
    ["make"],
    ["exif", "camera_make"],
    ["exif", "make"],
    ["exif", "Make"],
    # Lightroom classic catalog format
    ["exif", "all_formatted", "cameraMake"],
    ["exif", "formatted", "cameraMake"],
]

CAMERA_MODEL_METADATA_PATHS = [
    ["camera_model"],
    ["camera"],
    ["exif", "camera_model"],
    ["exif", "camera"],
    ["exif", "model"],
    ["exif", "Model"],
    # Lightroom classic catalog format
    ["exif", "all_formatted", "cameraModel"],
    ["exif", "formatted", "cameraModel"],
]

LENS_METADATA_PATHS = [
    ["lens"],
    ["lens_model"],
    ["exif", "lens"],
    ["exif", "lens_model"],
    ["exif", "Lens"],
    ["exif", "LensModel"],
    # Lightroom classic catalog format
    ["exif", "all_raw", "lens"],
    ["exif", "raw", "lens"],
    ["exif", "all_formatted", "lens"],
    ["exif", "formatted", "lens"],
]

GPS_LATITUDE_METADATA_PATHS = [
    ["gps_latitude"],
    ["latitude"],
    ["gps", "latitude"],
    ["exif", "gps_latitude"],
    ["exif", "latitude"],
    ["exif", "GPSLatitude"],
]

GPS_LONGITUDE_METADATA_PATHS = [
    ["gps_longitude"],
    ["longitude"],
    ["gps", "longitude"],
    ["exif", "gps_longitude"],
    ["exif", "longitude"],
    ["exif", "GPSLongitude"],
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


def _maybe_parse_json_container(value):
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_metadata_key(key):
    return re.sub(r"[^a-z0-9]+", "", str(key).casefold())


def _metadata_get_path(metadata, path):
    value = _maybe_parse_json_container(metadata)
    for key in path:
        value = _maybe_parse_json_container(value)
        if not isinstance(value, dict):
            return None

        if key in value:
            value = value[key]
            continue

        normalized_key = _normalize_metadata_key(key)
        matched_key = next(
            (candidate for candidate in value.keys() if _normalize_metadata_key(candidate) == normalized_key),
            None,
        )
        if matched_key is None:
            return None

        value = value[matched_key]

    return value


def _first_metadata_value(metadata, paths):
    nested_metadata = None
    if isinstance(metadata, dict):
        nested_metadata = _maybe_parse_json_container(metadata.get("metadata"))

    for path in paths:
        value = _metadata_get_path(metadata, path)
        if value in (None, "", [], {}):
            if nested_metadata not in (None, "", [], {}) and isinstance(nested_metadata, dict):
                value = _metadata_get_path(nested_metadata, path)
            if value in (None, "", [], {}):
                continue
        return value
    return None


def _coerce_text(value):
    if value in (None, "", [], {}):
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value):
    if value in (None, "", [], {}):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    rational_match = re.search(r"([-+]?[0-9]+(?:\.[0-9]+)?)\s*/\s*([-+]?[0-9]+(?:\.[0-9]+)?)", text)
    if rational_match:
        numerator = float(rational_match.group(1))
        denominator = float(rational_match.group(2))
        if denominator != 0:
            return numerator / denominator

    number_match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", text)
    if number_match:
        return float(number_match.group(0))

    return None


def _coerce_int(value):
    number = _coerce_float(value)
    if number is None:
        return None
    return int(round(number))


def _coerce_aperture_f_number(metadata):
    f_number = _coerce_float(_first_metadata_value(metadata, APERTURE_METADATA_PATHS))
    if f_number is not None:
        return f_number

    aperture_value = _coerce_float(_first_metadata_value(metadata, APERTURE_VALUE_METADATA_PATHS))
    if aperture_value is None:
        return None

    return 2 ** (aperture_value / 2)


def _coerce_capture_time(value):
    if value in (None, "", [], {}):
        return None

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = str(value).strip()
        if not text:
            return None

        normalized_text = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized_text)
        except ValueError:
            parsed = None
            for date_format in (
                "%Y:%m:%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%d",
                "%Y/%m/%d",
            ):
                try:
                    parsed = datetime.strptime(text, date_format)
                    break
                except ValueError:
                    continue

            if parsed is None:
                return None

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)

    return parsed


def _extract_normalized_metadata(metadata):
    metadata = metadata or {}
    latitude = _coerce_float(_first_metadata_value(metadata, GPS_LATITUDE_METADATA_PATHS))
    longitude = _coerce_float(_first_metadata_value(metadata, GPS_LONGITUDE_METADATA_PATHS))

    return {
        "capture_time": _coerce_capture_time(_first_metadata_value(metadata, CAPTURE_TIME_METADATA_PATHS)),
        "aperture_f_number": _coerce_aperture_f_number(metadata),
        "shutter_speed": _coerce_float(_first_metadata_value(metadata, SHUTTER_SPEED_METADATA_PATHS)),
        "exposure_bias": _coerce_float(_first_metadata_value(metadata, EXPOSURE_BIAS_METADATA_PATHS)),
        "iso": _coerce_int(_first_metadata_value(metadata, ISO_METADATA_PATHS)),
        "focal_length_mm": _coerce_float(_first_metadata_value(metadata, FOCAL_LENGTH_METADATA_PATHS)),
        "focal_length_35mm": _coerce_float(_first_metadata_value(metadata, FOCAL_LENGTH_35MM_METADATA_PATHS)),
        "camera_make": _coerce_text(_first_metadata_value(metadata, CAMERA_MAKE_METADATA_PATHS)),
        "camera_model": _coerce_text(_first_metadata_value(metadata, CAMERA_MODEL_METADATA_PATHS)),
        "lens": _coerce_text(_first_metadata_value(metadata, LENS_METADATA_PATHS)),
        "gps_latitude": latitude,
        "gps_longitude": longitude,
    }


def _has_normalized_metadata_value(normalized_metadata):
    return any(value is not None for value in normalized_metadata.values())


def _backfill_normalized_metadata_columns(conn):
    null_checks = sql.SQL(" OR ").join(
        sql.SQL("{col} IS NULL").format(col=sql.Identifier(col))
        for col in NORMALIZED_METADATA_COLUMN_NAMES
    )
    rows = conn.execute(
        sql.SQL(
            """
            SELECT uuid, metadata
            FROM photo_metadata
            WHERE (
                metadata ? 'exif'
                OR metadata ? 'metadata'
                OR metadata ? 'capture_time'
                OR metadata ? 'aperture'
                OR metadata ? 'aperture_f_number'
                OR metadata ? 'iso'
                OR metadata ? 'focal_length'
                OR metadata ? 'focal_length_mm'
                OR metadata ? 'gps'
            )
            AND ({null_checks})
            """
        ).format(null_checks=null_checks)
    ).fetchall()

    set_clauses = sql.SQL(", ").join(
        sql.SQL("{col} = COALESCE(%s, {col})").format(col=sql.Identifier(col))
        for col in NORMALIZED_METADATA_COLUMN_NAMES
    )

    backfilled_count = 0
    for uuid, metadata in rows:
        normalized_metadata = _extract_normalized_metadata(metadata or {})
        if not _has_normalized_metadata_value(normalized_metadata):
            continue

        normalized_values = [normalized_metadata[col] for col in NORMALIZED_METADATA_COLUMN_NAMES]
        conn.execute(
            sql.SQL("UPDATE photo_metadata SET {set_clauses} WHERE uuid = %s").format(
                set_clauses=set_clauses,
            ),
            (*normalized_values, uuid),
        )
        backfilled_count += 1

    if backfilled_count:
        logger.info(f"Backfilled normalized metadata columns for {backfilled_count} photo(s).")


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
                    capture_time TIMESTAMP,
                    aperture_f_number DOUBLE PRECISION,
                    shutter_speed DOUBLE PRECISION,
                    exposure_bias DOUBLE PRECISION,
                    iso INTEGER,
                    focal_length_mm DOUBLE PRECISION,
                    focal_length_35mm DOUBLE PRECISION,
                    camera_make TEXT,
                    camera_model TEXT,
                    lens TEXT,
                    gps_latitude DOUBLE PRECISION,
                    gps_longitude DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            ).format(dimension=sql.Literal(TEXT_EMBEDDING_DIMENSION))
        )
        for column_name, column_type in NORMALIZED_METADATA_COLUMNS:
            conn.execute(
                sql.SQL("ALTER TABLE photo_metadata ADD COLUMN IF NOT EXISTS {} {}").format(
                    sql.Identifier(column_name),
                    sql.SQL(column_type),
                )
            )
        conn.execute(
            sql.SQL(
                "ALTER TABLE photo_metadata ADD COLUMN IF NOT EXISTS embedding_kw vector({dimension})"
            ).format(dimension=sql.Literal(TEXT_EMBEDDING_DIMENSION))
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS photo_metadata_metadata_gin_idx
            ON photo_metadata USING gin (metadata)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS photo_metadata_capture_time_idx
            ON photo_metadata (capture_time)
            WHERE capture_time IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS photo_metadata_aperture_f_number_idx
            ON photo_metadata (aperture_f_number)
            WHERE aperture_f_number IS NOT NULL
            """
        )
        _backfill_normalized_metadata_columns(conn)

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

    try:
        with _connect_to_target() as conn:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS photo_metadata_embedding_kw_hnsw_idx
                ON photo_metadata USING hnsw (embedding_kw vector_cosine_ops)
                WHERE embedding_kw IS NOT NULL
                """
            )
    except psycopg.Error as e:
        logger.warning(
            "Could not create pgvector HNSW index for embedding_kw. "
            f"Keyword-embedding search will still work but may be slower: {e}"
        )
        with _connect_to_target() as conn:
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS photo_metadata_embedding_kw_ivfflat_idx
                ON photo_metadata USING ivfflat (embedding_kw vector_cosine_ops)
                WITH (lists = 100)
                WHERE embedding_kw IS NOT NULL
                """
            )

    try:
        with _connect_to_target() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS photo_metadata_document_trgm_idx
                ON photo_metadata USING gin (document gin_trgm_ops)
                WHERE document IS NOT NULL
                """
            )
    except psycopg.Error as e:
        logger.warning(
            f"Could not create pg_trgm index on document column. "
            f"Text search will still work but may be slower: {e}"
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


def _result(ids, metadatas=None, embeddings=None, embeddings_kw=None, documents=None, distances=None, grouped=False):
    if grouped:
        payload = {"ids": [ids]}
        if metadatas is not None:
            payload["metadatas"] = [metadatas]
        if embeddings is not None:
            payload["embeddings"] = [embeddings]
        if embeddings_kw is not None:
            payload["embeddings_kw"] = [embeddings_kw]
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
    if embeddings_kw is not None:
        payload["embeddings_kw"] = embeddings_kw
    if documents is not None:
        payload["documents"] = documents
    if distances is not None:
        payload["distances"] = distances
    return payload


def _upsert_record(uuid, metadata, embedding=None, embedding_kw=None, document=None, update_embedding=True):
    embedding_value = _embedding_literal(embedding)
    embedding_kw_value = _embedding_literal(embedding_kw)
    metadata = metadata or {}
    normalized_metadata = _extract_normalized_metadata(metadata)
    normalized_values = [normalized_metadata[col] for col in NORMALIZED_METADATA_COLUMN_NAMES]

    norm_cols = sql.SQL(", ").join(sql.Identifier(col) for col in NORMALIZED_METADATA_COLUMN_NAMES)
    norm_placeholders = sql.SQL(", ").join(sql.SQL("%s") for _ in NORMALIZED_METADATA_COLUMN_NAMES)
    norm_set_clauses = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col))
        for col in NORMALIZED_METADATA_COLUMN_NAMES
    )

    with _connect_to_target() as conn:
        if update_embedding:
            conn.execute(
                sql.SQL(
                    """
                    INSERT INTO photo_metadata (uuid, metadata, document, embedding, embedding_kw, {norm_cols})
                    VALUES (%s, %s, %s, %s::vector, %s::vector, {norm_placeholders})
                    ON CONFLICT (uuid) DO UPDATE SET
                        metadata = EXCLUDED.metadata,
                        document = COALESCE(EXCLUDED.document, photo_metadata.document),
                        embedding = COALESCE(EXCLUDED.embedding, photo_metadata.embedding),
                        embedding_kw = COALESCE(EXCLUDED.embedding_kw, photo_metadata.embedding_kw),
                        {norm_set_clauses},
                        updated_at = now()
                    """
                ).format(
                    norm_cols=norm_cols,
                    norm_placeholders=norm_placeholders,
                    norm_set_clauses=norm_set_clauses,
                ),
                (uuid, Jsonb(metadata), document, embedding_value, embedding_kw_value, *normalized_values),
            )
        else:
            conn.execute(
                sql.SQL(
                    """
                    INSERT INTO photo_metadata (uuid, metadata, document, {norm_cols})
                    VALUES (%s, %s, %s, {norm_placeholders})
                    ON CONFLICT (uuid) DO UPDATE SET
                        metadata = EXCLUDED.metadata,
                        document = COALESCE(EXCLUDED.document, photo_metadata.document),
                        {norm_set_clauses},
                        updated_at = now()
                    """
                ).format(
                    norm_cols=norm_cols,
                    norm_placeholders=norm_placeholders,
                    norm_set_clauses=norm_set_clauses,
                ),
                (uuid, Jsonb(metadata), document, *normalized_values),
            )


def add_image(uuid, embedding, metadata, embedding_kw=None, document=None):
    _ensure_initialized()
    if DEBUG_LEVEL >= 1:
        logger.info(
            f"Saving image {uuid} to PostgreSQL ("
            f"embedding={embedding is not None}, "
            f"embedding_kw={embedding_kw is not None}, "
            f"document={'yes' if document else 'no'})"
        )
    try:
        _upsert_record(
            uuid, metadata,
            embedding=embedding,
            embedding_kw=embedding_kw,
            document=document,
            update_embedding=(embedding is not None) or (embedding_kw is not None),
        )
        if DEBUG_LEVEL >= 1:
            logger.info(f"Image {uuid} saved successfully to PostgreSQL.")
    except Exception as e:
        logger.error(
            f"FAILED to save image {uuid} to PostgreSQL (embedding provided: {embedding is not None}): {e}",
            exc_info=True,
        )
        raise


def update_image(uuid, metadata, embedding=None, embedding_kw=None, document=None):
    _ensure_initialized()
    _upsert_record(
        uuid,
        metadata,
        embedding=embedding,
        embedding_kw=embedding_kw,
        document=document,
        update_embedding=(embedding is not None) or (embedding_kw is not None),
    )
    if DEBUG_LEVEL >= 2:
        logger.debug(f"image {uuid} is well UPSERTED in PostgreSQL. Metadata = {metadata}")


def get_image(uuid):
    _ensure_initialized()
    with _connect_to_target() as conn:
        row = conn.execute(
            """
            SELECT uuid, metadata, document,
                   embedding::text AS embedding,
                   embedding_kw::text AS embedding_kw
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
        embeddings_kw=[_embedding_from_text(row[4])],
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


def _capture_time_column_filter_clause(filter_item):
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

    return (
        sql.SQL("(capture_time IS NOT NULL AND capture_time {operator} %s::timestamp)").format(
            operator=sql.SQL(operator),
        ),
        [str(value)],
    )


def _metadata_text_filter_clause_with_column(filter_item, column_clause, column_name):
    fallback_clause, fallback_params = _metadata_text_filter_clause(filter_item)
    if column_clause is None:
        return fallback_clause, fallback_params

    column_sql, column_params = column_clause
    if fallback_clause is None:
        return column_sql, column_params

    return (
        (
            sql.SQL("(")
            + column_sql
            + sql.SQL(" OR ({column} IS NULL AND ").format(column=sql.Identifier(column_name))
            + fallback_clause
            + sql.SQL("))")
        ),
        [*column_params, *fallback_params],
    )


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


def _numeric_column_filter_clause(filter_item, column_name):
    try:
        value = float(filter_item.get("value"))
    except (TypeError, ValueError):
        return None, []

    op = str(filter_item.get("op", "number_eq")).casefold()

    if op in {"number_eq", "eq", "equals"}:
        tolerance = filter_item.get("tolerance", 0.0)
        try:
            tolerance = abs(float(tolerance))
        except (TypeError, ValueError):
            tolerance = 0.0

        return (
            sql.SQL("({column} IS NOT NULL AND {column} BETWEEN %s AND %s)").format(
                column=sql.Identifier(column_name),
            ),
            [value - tolerance, value + tolerance],
        )

    operator_map = {
        "gte": ">=",
        "gt": ">",
        "lte": "<=",
        "lt": "<",
    }
    operator = operator_map.get(op)
    if operator is None:
        return None, []

    return (
        sql.SQL("({column} IS NOT NULL AND {column} {operator} %s)").format(
            column=sql.Identifier(column_name),
            operator=sql.SQL(operator),
        ),
        [value],
    )


def _metadata_numeric_filter_clause_with_column(filter_item, column_name):
    column_clause, column_params = _numeric_column_filter_clause(filter_item, column_name)
    fallback_clause, fallback_params = _metadata_numeric_filter_clause(filter_item)

    if column_clause is None:
        return fallback_clause, fallback_params
    if fallback_clause is None:
        return column_clause, column_params

    return (
        (
            sql.SQL("(")
            + column_clause
            + sql.SQL(" OR ({column} IS NULL AND ").format(column=sql.Identifier(column_name))
            + fallback_clause
            + sql.SQL("))")
        ),
        [*column_params, *fallback_params],
    )


_NUMERIC_COLUMN_FIELD_MAP = {
    "iso": "iso",
    "iso_speed_rating": "iso",
    "iso_speed_ratings": "iso",
    "focal_length_mm": "focal_length_mm",
    "focal_length": "focal_length_mm",
    "focal_length_35mm": "focal_length_35mm",
    "focal_length35mm": "focal_length_35mm",
    "shutter_speed": "shutter_speed",
    "shutter": "shutter_speed",
    "exposure_bias": "exposure_bias",
    "exposurebias": "exposure_bias",
}

_TEXT_ILIKE_COLUMN_FIELD_MAP = {
    "camera_make": "camera_make",
    "make": "camera_make",
    "camera_model": "camera_model",
    "camera": "camera_model",
    "lens": "lens",
    "lens_model": "lens",
}


def _text_column_ilike_filter_clause(filter_item, column_name):
    value = filter_item.get("value")
    if not value:
        return None, []
    return (
        sql.SQL("({column} IS NOT NULL AND {column} ILIKE %s)").format(
            column=sql.Identifier(column_name),
        ),
        [f"%{value}%"],
    )


def _metadata_text_ilike_filter_clause(filter_item):
    paths = _metadata_paths_for_filter(filter_item)
    if not paths:
        return None, []
    value = filter_item.get("value")
    if not value:
        return None, []
    pattern = f"%{value}%"
    clauses = []
    params = []
    for path in paths:
        clauses.append(
            sql.SQL("({value_expr} ILIKE %s)").format(value_expr=_metadata_path_expr())
        )
        params.extend([path, pattern])
    return sql.SQL("(") + sql.SQL(" OR ").join(clauses) + sql.SQL(")"), params


def _metadata_ilike_filter_clause_with_column(filter_item, column_name):
    column_clause, column_params = _text_column_ilike_filter_clause(filter_item, column_name)
    fallback_clause, fallback_params = _metadata_text_ilike_filter_clause(filter_item)
    if column_clause is None:
        return fallback_clause, fallback_params
    if fallback_clause is None:
        return column_clause, column_params
    return (
        (
            sql.SQL("(")
            + column_clause
            + sql.SQL(" OR ({column} IS NULL AND ").format(column=sql.Identifier(column_name))
            + fallback_clause
            + sql.SQL("))")
        ),
        [*column_params, *fallback_params],
    )


def _metadata_filter_clause(filter_item):
    field = str(filter_item.get("field", "")).strip().casefold()
    op = str(filter_item.get("op", "")).casefold()

    if field == "capture_time":
        return _metadata_text_filter_clause_with_column(
            filter_item,
            _capture_time_column_filter_clause(filter_item),
            "capture_time",
        )

    if field in {"aperture", "aperture_f_number", "f_number", "fnumber", "f_stop", "fstop"}:
        return _metadata_numeric_filter_clause_with_column(filter_item, "aperture_f_number")

    col = _NUMERIC_COLUMN_FIELD_MAP.get(field)
    if col:
        return _metadata_numeric_filter_clause_with_column(filter_item, col)

    col = _TEXT_ILIKE_COLUMN_FIELD_MAP.get(field)
    if col:
        return _metadata_ilike_filter_clause_with_column(filter_item, col)

    if op.startswith("number"):
        return _metadata_numeric_filter_clause_with_column(filter_item, "aperture_f_number")

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
    """Search photos by semantic similarity using the best score across both embeddings.

    For each candidate photo, the distance is the minimum cosine distance across
    the prose embedding (``embedding``) and the keyword embedding (``embedding_kw``).
    Photos that only have one embedding type still match correctly: the missing
    embedding contributes a worst-case distance of 1.0 which LEAST ignores.
    """
    _ensure_initialized()
    query_vector = _embedding_literal(query_embedding)

    try:
        select_embedding = (
            ", embedding::text AS embedding, embedding_kw::text AS embedding_kw"
            if include_embeddings else ""
        )
        filter_sql, filter_params = _filter_sql(where_clause, "AND")
        # query_vector appears twice in the SELECT (once per COALESCE distance expression).
        params = [query_vector, query_vector, *filter_params, n_results]

        query = sql.SQL(
            """
            SELECT uuid, metadata,
                LEAST(
                    COALESCE(embedding    <=> %s::vector, 1.0),
                    COALESCE(embedding_kw <=> %s::vector, 1.0)
                ) AS distance
                {select_embedding}
            FROM photo_metadata
            WHERE (embedding IS NOT NULL OR embedding_kw IS NOT NULL)
            AND COALESCE((metadata->>'has_embedding')::boolean, true)
            {filter_sql}
            ORDER BY distance
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
        embeddings_kw = [] if include_embeddings else None
        for row in rows:
            ids.append(row[0])
            metadatas.append(row[1] or {})
            distances.append(float(row[2]) if row[2] is not None else None)
            if include_embeddings:
                embeddings.append(_embedding_from_text(row[3]))
                embeddings_kw.append(_embedding_from_text(row[4]))

        return _result(
            ids,
            metadatas=metadatas,
            embeddings=embeddings,
            embeddings_kw=embeddings_kw,
            distances=distances,
            grouped=True,
        )
    except Exception as e:
        logger.error(f"Error querying images: {e}", exc_info=True)
        fallback = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        if include_embeddings:
            fallback["embeddings"] = [[]]
            fallback["embeddings_kw"] = [[]]
        return fallback


def search_document_text(term, where_clause=None, limit=300, use_word_boundary=False):
    """Return {uuid: metadata} for rows matching term, with optional column filters.

    If term is None or empty, only column filters from where_clause are applied.
    Returns an empty dict when both term and filters are absent to avoid a full-table fetch.

    When use_word_boundary=True the term is matched as a whole word using a
    PostgreSQL case-insensitive regex (``~*``) with word-boundary anchors (``\\m``/``\\M``).
    This prevents "lac" from matching "glacier", "place", etc.
    Fall back to plain ILIKE if the regex fails (e.g. term contains regex metacharacters).
    """
    _ensure_initialized()

    filter_clauses, filter_params = _filter_clauses(where_clause)

    if not term and not filter_clauses:
        return {}

    conditions = list(filter_clauses)
    all_params = list(filter_params)

    if term:
        if use_word_boundary:
            import re as _re
            escaped = _re.escape(term)
            conditions.insert(0, sql.SQL("document ~* %s"))
            all_params.insert(0, rf"\m{escaped}\M")
        else:
            conditions.insert(0, sql.SQL("document ILIKE %s"))
            all_params.insert(0, f"%{term}%")

    where_sql = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions)

    query = sql.SQL(
        """
        SELECT uuid, metadata
        FROM photo_metadata
        {where_sql}
        LIMIT %s
        """
    ).format(where_sql=where_sql)
    all_params.append(limit)

    try:
        with _connect_to_target() as conn:
            rows = conn.execute(query, all_params).fetchall()
        return {row[0]: row[1] or {} for row in rows}
    except Exception as e:
        logger.error(f"Error in search_document_text: {e}", exc_info=True)
        return {}


def search_keyword_field(term, where_clause=None, limit=300):
    """Return {uuid: metadata} for rows where flattened_keywords contains term as a whole word.

    This is more precise than searching the full document because it avoids matches
    in caption or quality_critique prose where the word may appear in a different context.
    Uses PostgreSQL word-boundary regex for exact keyword matching.
    """
    _ensure_initialized()

    filter_clauses, filter_params = _filter_clauses(where_clause)

    if not term:
        return {}

    import re as _re
    escaped = _re.escape(term)
    pattern = rf"\m{escaped}\M"

    conditions = list(filter_clauses)
    all_params = list(filter_params)
    conditions.insert(0, sql.SQL("metadata->>'flattened_keywords' ~* %s"))
    all_params.insert(0, pattern)

    where_sql = sql.SQL("WHERE ") + sql.SQL(" AND ").join(conditions)

    query = sql.SQL(
        """
        SELECT uuid, metadata
        FROM photo_metadata
        {where_sql}
        LIMIT %s
        """
    ).format(where_sql=where_sql)
    all_params.append(limit)

    try:
        with _connect_to_target() as conn:
            rows = conn.execute(query, all_params).fetchall()
        return {row[0]: row[1] or {} for row in rows}
    except Exception as e:
        logger.error(f"Error in search_keyword_field: {e}", exc_info=True)
        return {}


def get_image_metadatas(ids=None, where_clause=None, include_embedding=False):
    _ensure_initialized()
    effective_where_clause = dict(where_clause or {})
    if ids:
        effective_where_clause["uuid"] = {"$in": list(ids)}

    filter_sql, filter_params = _filter_sql(effective_where_clause, "WHERE")

    if include_embedding:
        select_cols = sql.SQL(
            "uuid, metadata, document, embedding::text AS embedding, embedding_kw::text AS embedding_kw"
        )
    else:
        select_cols = sql.SQL("uuid, metadata")

    with _connect_to_target() as conn:
        rows = conn.execute(
            sql.SQL(
                """
                SELECT {select_cols}
                FROM photo_metadata
                {filter_sql}
                ORDER BY uuid
                """
            ).format(select_cols=select_cols, filter_sql=filter_sql),
            filter_params,
        ).fetchall()

    if include_embedding:
        return _result(
            [row[0] for row in rows],
            metadatas=[row[1] or {} for row in rows],
            documents=[row[2] for row in rows],
            embeddings=[_embedding_from_text(row[3]) for row in rows],
            embeddings_kw=[_embedding_from_text(row[4]) for row in rows],
        )
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


# ---------------------------------------------------------------------------
# Style Match — reference photos (plan §3.2)
# ---------------------------------------------------------------------------
# Reference photos live in their own table, keyed by (catalog_id, lrc_local_id)
# because Lightroom local identifiers are only unique per catalog. The CLIP
# embedding (topic) and the description embedding (existing Qwen3 pipeline) are
# separate columns so candidate generation can query both.

_style_refs_initialized = False

STYLE_REF_EMBEDDING_COLUMNS = {
    "embedding": CLIP_EMBEDDING_DIMENSION,
    "description_embedding": TEXT_EMBEDDING_DIMENSION,
}

STYLE_REF_COLUMN_NAMES = [
    "id",
    "catalog_id",
    "lrc_local_id",
    "uuid",
    "file_path",
    "file_name",
    "capture_time",
    "latitude",
    "longitude",
    "camera_make",
    "camera_model",
    "lens",
    "focal_length_mm",
    "aperture_f_number",
    "iso",
    "description",
    "color_histogram",
    "render_source",
    "develop_settings",
    "has_edits",
    "embedding_model",
    "created_at",
    "updated_at",
]


def _style_ref_literal(embedding, dimension):
    if embedding is None:
        return None
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape[0] != dimension:
        raise ValueError(
            f"Style-ref embedding dimension mismatch: expected {dimension}, got {vector.shape[0]}"
        )
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def _ensure_style_refs():
    """Create the style_ref_photos table and indexes (idempotent, flag-gated).

    Kept separate from _ensure_initialized so a style-ref DDL problem can never
    break the existing photo_metadata features.
    """
    global _style_refs_initialized
    if _style_refs_initialized:
        return

    with _connect_to_target() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS style_ref_photos (
                    id                    BIGSERIAL PRIMARY KEY,
                    catalog_id            TEXT        NOT NULL,
                    lrc_local_id          BIGINT      NOT NULL,
                    uuid                  UUID        NOT NULL,
                    file_path             TEXT        NOT NULL,
                    file_name             TEXT        NOT NULL,
                    capture_time          TIMESTAMPTZ,
                    latitude              DOUBLE PRECISION,
                    longitude             DOUBLE PRECISION,
                    camera_make           TEXT,
                    camera_model          TEXT,
                    lens                  TEXT,
                    focal_length_mm       REAL,
                    aperture_f_number     REAL,
                    iso                   INTEGER,
                    embedding             vector({clip_dim}),
                    description_embedding vector({text_dim}),
                    description           TEXT,
                    color_histogram       REAL[],
                    render_source         TEXT,
                    develop_settings      JSONB,
                    has_edits             BOOLEAN     NOT NULL DEFAULT FALSE,
                    embedding_model       TEXT        NOT NULL,
                    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (catalog_id, lrc_local_id)
                )
                """
            ).format(
                clip_dim=sql.Literal(CLIP_EMBEDDING_DIMENSION),
                text_dim=sql.Literal(TEXT_EMBEDDING_DIMENSION),
            )
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS style_ref_gps_idx
            ON style_ref_photos (latitude, longitude)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS style_ref_has_edits_idx
            ON style_ref_photos (has_edits)
            WHERE has_edits
            """
        )

    for column in STYLE_REF_EMBEDDING_COLUMNS:
        # The index NAME must be a plain identifier: interpolating
        # sql.Identifier into the name string renders quoted text
        # (style_ref_"embedding"_hnsw_idx) and the DDL fails with a syntax error.
        hnsw_name = "style_ref_%s_hnsw_idx" % column
        try:
            with _connect_to_target() as conn:
                conn.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {idx}
                        ON style_ref_photos USING hnsw ({col} vector_cosine_ops)
                        WHERE {col} IS NOT NULL
                        """
                    ).format(idx=sql.Identifier(hnsw_name), col=sql.Identifier(column))
                )
        except psycopg.Error as e:
            logger.warning(
                f"Could not create pgvector HNSW index on style_ref_photos.{column}. "
                f"Style-match search will still work, but may be slower until the "
                f"index is created manually: {e}"
            )
            try:
                ivfflat_name = "style_ref_%s_ivfflat_idx" % column
                with _connect_to_target() as conn:
                    conn.execute(
                        sql.SQL(
                            """
                            CREATE INDEX IF NOT EXISTS {idx}
                            ON style_ref_photos USING ivfflat ({col} vector_cosine_ops)
                            WITH (lists = 100)
                            WHERE {col} IS NOT NULL
                            """
                        ).format(idx=sql.Identifier(ivfflat_name), col=sql.Identifier(column))
                    )
            except psycopg.Error as fallback_error:
                logger.warning(
                    f"Could not create IVFFlat fallback index on style_ref_photos.{column}: "
                    f"{fallback_error}"
                )

    _style_refs_initialized = True


def _style_ref_filter_clauses(filters):
    """Build WHERE clauses for style-ref candidate queries.

    filters.has_edits: True -> only edited references (query-time default,
    plan §2.6), False -> only unedited, None/absent -> no filter. Date / camera
    / geo filters narrow the candidate set but never enter the score (§2.1).
    """
    clauses = []
    params = []
    if not filters:
        return clauses, params

    has_edits = filters.get("has_edits")
    if has_edits is not None:
        clauses.append(sql.SQL("has_edits = %s"))
        params.append(bool(has_edits))

    if filters.get("camera_model"):
        clauses.append(sql.SQL("camera_model = %s"))
        params.append(str(filters["camera_model"]))

    date_from = filters.get("date_from")
    if date_from:
        clauses.append(sql.SQL("capture_time >= %s"))
        params.append(date_from)

    date_to = filters.get("date_to")
    if date_to:
        clauses.append(sql.SQL("capture_time <= %s"))
        params.append(date_to)

    latitude = filters.get("lat")
    longitude = filters.get("lon")
    radius_km = filters.get("radius_km")
    if latitude is not None and longitude is not None and radius_km:
        # Bounding-box prefilter in degrees; the exact haversine score is
        # applied per candidate in the re-rank.
        lat_delta = float(radius_km) / 111.32
        lon_scale = 111.32 * max(abs(float(latitude)) * 0.017453292519943295, 0.01)
        lon_delta = min(float(radius_km) / lon_scale, 180.0)
        clauses.append(sql.SQL("latitude BETWEEN %s AND %s AND longitude BETWEEN %s AND %s"))
        params.extend(
            [
                float(latitude) - lat_delta,
                float(latitude) + lat_delta,
                float(longitude) - lon_delta,
                float(longitude) + lon_delta,
            ]
        )

    return clauses, params


def upsert_style_ref(
    catalog_id,
    lrc_local_id,
    uuid,
    file_path,
    file_name,
    embedding=None,
    description_embedding=None,
    capture_time=None,
    latitude=None,
    longitude=None,
    camera_make=None,
    camera_model=None,
    lens=None,
    focal_length_mm=None,
    aperture_f_number=None,
    iso=None,
    description=None,
    color_histogram=None,
    render_source=None,
    develop_settings=None,
    has_edits=False,
    embedding_model="clip-vit-b32",
):
    """Insert or update one style reference photo; idempotent upsert.

    Vector/histogram/description values are COALESCEd so an incremental push
    without a neutral render cannot erase a previously computed histogram.
    """
    _ensure_style_refs()
    embedding_value = _style_ref_literal(embedding, CLIP_EMBEDDING_DIMENSION)
    description_value = _style_ref_literal(description_embedding, TEXT_EMBEDDING_DIMENSION)
    if color_histogram is not None:
        color_histogram = [float(value) for value in color_histogram]

    with _connect_to_target() as conn:
        conn.execute(
            """
            INSERT INTO style_ref_photos (
                catalog_id, lrc_local_id, uuid, file_path, file_name,
                capture_time, latitude, longitude, camera_make, camera_model,
                lens, focal_length_mm, aperture_f_number, iso,
                embedding, description_embedding, description,
                color_histogram, render_source, develop_settings,
                has_edits, embedding_model, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s::vector, %s::vector, %s,
                %s, %s, %s,
                %s, %s, now()
            )
            ON CONFLICT (catalog_id, lrc_local_id) DO UPDATE SET
                uuid = EXCLUDED.uuid,
                file_path = EXCLUDED.file_path,
                file_name = EXCLUDED.file_name,
                capture_time = EXCLUDED.capture_time,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                camera_make = EXCLUDED.camera_make,
                camera_model = EXCLUDED.camera_model,
                lens = EXCLUDED.lens,
                focal_length_mm = EXCLUDED.focal_length_mm,
                aperture_f_number = EXCLUDED.aperture_f_number,
                iso = EXCLUDED.iso,
                embedding = COALESCE(EXCLUDED.embedding, style_ref_photos.embedding),
                description_embedding = COALESCE(EXCLUDED.description_embedding, style_ref_photos.description_embedding),
                description = COALESCE(EXCLUDED.description, style_ref_photos.description),
                color_histogram = COALESCE(EXCLUDED.color_histogram, style_ref_photos.color_histogram),
                render_source = COALESCE(EXCLUDED.render_source, style_ref_photos.render_source),
                develop_settings = COALESCE(EXCLUDED.develop_settings, style_ref_photos.develop_settings),
                has_edits = EXCLUDED.has_edits,
                embedding_model = EXCLUDED.embedding_model,
                updated_at = now()
            """,
            (
                catalog_id,
                int(lrc_local_id),
                str(uuid),
                file_path,
                file_name,
                capture_time,
                latitude,
                longitude,
                camera_make,
                camera_model,
                lens,
                focal_length_mm,
                aperture_f_number,
                iso,
                embedding_value,
                description_value,
                description,
                color_histogram,
                render_source,
                Jsonb(develop_settings) if develop_settings is not None else None,
                bool(has_edits),
                embedding_model,
            ),
        )


def _style_ref_row_to_dict(row):
    return {
        "id": row[0],
        "catalog_id": row[1],
        "lrc_local_id": row[2],
        "uuid": str(row[3]) if row[3] is not None else None,
        "file_path": row[4],
        "file_name": row[5],
        "capture_time": row[6].isoformat() if row[6] is not None else None,
        "latitude": row[7],
        "longitude": row[8],
        "camera_make": row[9],
        "camera_model": row[10],
        "lens": row[11],
        "focal_length_mm": row[12],
        "aperture_f_number": row[13],
        "iso": row[14],
        "description": row[15],
        "color_histogram": row[16],
        "render_source": row[17],
        "develop_settings": row[18],
        "has_edits": row[19],
        "embedding_model": row[20],
        "created_at": row[21].isoformat() if row[21] is not None else None,
        "updated_at": row[22].isoformat() if row[22] is not None else None,
    }


def _style_ref_select_columns():
    return sql.SQL(", ").join(
        [sql.Identifier(col) for col in STYLE_REF_COLUMN_NAMES]
        + [
            sql.SQL("embedding::text"),
            sql.SQL("description_embedding::text"),
        ]
    )


def get_style_ref_by_lrc(catalog_id, lrc_local_id):
    """Return one style-ref row (dict) or None."""
    _ensure_style_refs()
    with _connect_to_target() as conn:
        row = conn.execute(
            sql.SQL(
                """
                SELECT {columns}
                FROM style_ref_photos
                WHERE catalog_id = %s AND lrc_local_id = %s
                """
            ).format(columns=_style_ref_select_columns()),
            (catalog_id, int(lrc_local_id)),
        ).fetchone()
    if not row:
        return None
    result = _style_ref_row_to_dict(row)
    result["embedding"] = _embedding_from_text(row[23])
    result["description_embedding"] = _embedding_from_text(row[24])
    return result


def get_style_ref_rows(ids):
    """Return {id: row_dict} for the given style_ref_photos primary keys."""
    _ensure_style_refs()
    ids = [int(id_) for id_ in (ids or []) if id_ is not None]
    if not ids:
        return {}
    with _connect_to_target() as conn:
        rows = conn.execute(
            sql.SQL(
                """
                SELECT {columns}
                FROM style_ref_photos
                WHERE id = ANY(%s)
                """
            ).format(columns=_style_ref_select_columns()),
            (ids,),
        ).fetchall()
    results = {}
    for row in rows:
        result = _style_ref_row_to_dict(row)
        result["embedding"] = _embedding_from_text(row[23])
        result["description_embedding"] = _embedding_from_text(row[24])
        results[row[0]] = result
    return results


def query_style_ref_candidates(column, query_embedding, k, filters=None, exclude=None):
    """Top-k style-ref candidates by cosine similarity on one embedding column.

    column: "embedding" (CLIP, 50% signal) or "description_embedding" (Qwen3,
    30% signal). Rows with a NULL in that column cannot be matched by it.
    exclude: optional (catalog_id, lrc_local_id) tuple for self-exclusion.

    Returns a list of row dicts, best first, each carrying "similarity"
    (1 - cosine distance) for the queried column under the key
    "<column>_similarity".
    """
    _ensure_style_refs()
    if column not in STYLE_REF_EMBEDDING_COLUMNS:
        raise ValueError(f"Unknown style-ref embedding column: {column}")
    dimension = STYLE_REF_EMBEDDING_COLUMNS[column]
    query_vector = _style_ref_literal(query_embedding, dimension)

    clauses, params = _style_ref_filter_clauses(filters)
    clauses.append(sql.SQL("{col} IS NOT NULL").format(col=sql.Identifier(column)))
    if exclude and exclude[0] is not None and exclude[1] is not None:
        clauses.append(sql.SQL("NOT (catalog_id = %s AND lrc_local_id = %s)"))
        params.extend([str(exclude[0]), int(exclude[1])])

    query = sql.SQL(
        """
        SELECT {columns}, 1 - ({col} <=> %s::vector) AS similarity
        FROM style_ref_photos
        WHERE {where}
        ORDER BY {col} <=> %s::vector
        LIMIT %s
        """
    ).format(
        columns=_style_ref_select_columns(),
        col=sql.Identifier(column),
        where=sql.SQL(" AND ").join(clauses),
    )
    params = [query_vector, *params, query_vector, int(k)]

    with _connect_to_target() as conn:
        rows = conn.execute(query, params).fetchall()

    candidates = []
    for row in rows:
        candidate = _style_ref_row_to_dict(row)
        similarity = row[25]
        candidate["similarity"] = float(similarity) if similarity is not None else None
        if column == "embedding":
            candidate["sem"] = candidate["similarity"]
        else:
            candidate["desc"] = candidate["similarity"]
        candidates.append(candidate)
    return candidates


def delete_style_ref(catalog_id, lrc_local_id):
    """Remove one style reference photo (used when the catalog row is gone)."""
    _ensure_style_refs()
    with _connect_to_target() as conn:
        conn.execute(
            "DELETE FROM style_ref_photos WHERE catalog_id = %s AND lrc_local_id = %s",
            (catalog_id, int(lrc_local_id)),
        )


def style_ref_stats():
    """Counts for /style-match/health (rows, edited share, render-source mix)."""
    _ensure_style_refs()
    with _connect_to_target() as conn:
        stats = conn.execute(
            """
            SELECT
                count(*) AS total,
                count(*) FILTER (WHERE has_edits) AS edited,
                count(*) FILTER (WHERE embedding IS NOT NULL) AS with_clip,
                count(*) FILTER (WHERE description_embedding IS NOT NULL) AS with_desc,
                count(*) FILTER (WHERE color_histogram IS NOT NULL) AS with_color,
                count(*) FILTER (WHERE render_source = 'develop_export') AS render_develop_export,
                count(*) FILTER (WHERE render_source = 'camera_embedded') AS render_camera_embedded,
                count(*) FILTER (WHERE render_source = 'original') AS render_original,
                count(DISTINCT catalog_id) AS catalogs
            FROM style_ref_photos
            """
        ).fetchone()
    return {
        "total": stats[0],
        "edited": stats[1],
        "with_clip_embedding": stats[2],
        "with_description_embedding": stats[3],
        "with_color_histogram": stats[4],
        "render_source_mix": {
            "develop_export": stats[5],
            "camera_embedded": stats[6],
            "original": stats[7],
        },
        "catalogs": stats[8],
    }
