import json
import re
from datetime import datetime, timezone

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
                    INSERT INTO photo_metadata (uuid, metadata, document, embedding, {norm_cols})
                    VALUES (%s, %s, %s, %s::vector, {norm_placeholders})
                    ON CONFLICT (uuid) DO UPDATE SET
                        metadata = EXCLUDED.metadata,
                        document = COALESCE(EXCLUDED.document, photo_metadata.document),
                        embedding = COALESCE(EXCLUDED.embedding, photo_metadata.embedding),
                        {norm_set_clauses},
                        updated_at = now()
                    """
                ).format(
                    norm_cols=norm_cols,
                    norm_placeholders=norm_placeholders,
                    norm_set_clauses=norm_set_clauses,
                ),
                (uuid, Jsonb(metadata), document, embedding_value, *normalized_values),
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


def add_image(uuid, embedding, metadata, document=None):
    _ensure_initialized()
    logger.info(f"Saving image {uuid} to PostgreSQL (embedding={embedding is not None}, document={'yes' if document else 'no'})")
    try:
        _upsert_record(uuid, metadata, embedding=embedding, document=document, update_embedding=embedding is not None)
        logger.info(f"Image {uuid} saved successfully to PostgreSQL.")
    except Exception as e:
        logger.error(
            f"FAILED to save image {uuid} to PostgreSQL (embedding provided: {embedding is not None}): {e}",
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


def _metadata_filter_clause(filter_item):
    field = str(filter_item.get("field", "")).strip().casefold()
    op = str(filter_item.get("op", "")).casefold()
    if field == "capture_time":
        return _metadata_text_filter_clause_with_column(
            filter_item,
            _capture_time_column_filter_clause(filter_item),
            "capture_time",
        )
    if field in {"aperture", "aperture_f_number", "f_number", "fnumber", "f_stop", "fstop"} or op.startswith("number"):
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


def get_image_metadatas(ids=None, where_clause=None, include_embedding=False):
    _ensure_initialized()
    effective_where_clause = dict(where_clause or {})
    if ids:
        effective_where_clause["uuid"] = {"$in": list(ids)}

    filter_sql, filter_params = _filter_sql(effective_where_clause, "WHERE")

    if include_embedding:
        select_cols = sql.SQL("uuid, metadata, document, embedding::text AS embedding")
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
