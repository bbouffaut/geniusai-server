import json
from datetime import datetime

import service_postgre as postgre_service
from config import logger
from service_index import keywords_with_generation_model


QUALITY_SCORE_FIELDS = {
    "overall_score",
    "composition_score",
    "lighting_score",
    "motiv_score",
    "colors_score",
    "emotion_score",
    "quality_critique",
}

GET_RESERVED_FIELDS = {
    "filters",
    "metadata",
    "quality",
    "count",
    "photos",
    "status",
    "include",
    "fields",
    "limit",
    "offset",
    "order",
    "sort",
    "direction",
    "return_metadata",
}

GET_MULTI_VALUE_FIELDS = {
    "id",
    "ids",
    "uuid",
    "uuids",
    "filename",
    "filenames",
}

GET_DATE_FIELDS = {
    "ai_run_date",
    "ai_rundate",
    "capture_time",
    "photo_date",
    "run_date",
}

GET_BOOL_FIELDS = {
    "has_embedding",
}

GET_NUMERIC_FIELDS = QUALITY_SCORE_FIELDS - {"quality_critique"}


def _first_non_blank_value(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _maybe_parse_json_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _normalize_get_filter_value(key, value):
    if isinstance(value, str):
        normalized_value = value.strip()
        if normalized_value and normalized_value[0] in "[{":
            try:
                return json.loads(normalized_value)
            except json.JSONDecodeError:
                pass

        if key.lower() in GET_MULTI_VALUE_FIELDS and "," in normalized_value:
            return [item.strip() for item in normalized_value.split(",") if item.strip()]

        return value

    if isinstance(value, list):
        return [_normalize_get_filter_value(key, item) for item in value]

    if isinstance(value, dict):
        return {
            nested_key: _normalize_get_filter_value(nested_key, nested_value)
            for nested_key, nested_value in value.items()
        }

    return value


def _extend_get_filters(filters, payload):
    if not isinstance(payload, dict):
        return

    for key, value in payload.items():
        if key in {"filters", "metadata", "quality"} and isinstance(value, dict):
            _extend_get_filters(filters, value)
            continue

        filters[key] = _normalize_get_filter_value(key, value)


def extract_get_filters(data):
    filters = {}

    for section_key in ("filters", "metadata", "quality"):
        _extend_get_filters(filters, data.get(section_key))

    for key, value in data.items():
        if key in GET_RESERVED_FIELDS:
            continue
        filters[key] = _normalize_get_filter_value(key, value)

    return filters


def extract_requested_uuids(filters):
    requested_uuids = []

    for key in ("uuid", "uuids", "id", "ids"):
        value = filters.get(key)
        if value is None or value == "":
            continue

        if isinstance(value, list):
            requested_uuids.extend(value)
        else:
            requested_uuids.append(value)

    normalized = []
    seen = set()
    for uuid in requested_uuids:
        uuid_text = str(uuid).strip()
        if not uuid_text or uuid_text in seen:
            continue
        seen.add(uuid_text)
        normalized.append(uuid_text)

    return normalized


def _coerce_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False

    return None


def _coerce_number(value):
    if isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify_date_like(value):
    parsed_value = _maybe_parse_json_value(value)
    if parsed_value is None:
        return None

    if isinstance(parsed_value, (int, float)) and not isinstance(parsed_value, bool):
        try:
            return datetime.fromtimestamp(float(parsed_value)).strftime("%Y-%m-%d %H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return str(parsed_value).strip()

    return str(parsed_value).strip()


def _value_matches_filter(actual_value, expected_value, field_name):
    expected_value = _maybe_parse_json_value(expected_value)
    actual_value = _maybe_parse_json_value(actual_value)

    if expected_value in (None, "", [], {}):
        return True

    if isinstance(expected_value, list):
        expected_items = [item for item in expected_value if item not in (None, "", [], {})]
        if not expected_items:
            return True
        return any(
            _value_matches_filter(actual_value, expected_item, field_name)
            for expected_item in expected_items
        )

    if isinstance(actual_value, list):
        return any(
            _value_matches_filter(candidate, expected_value, field_name)
            for candidate in actual_value
        )

    if isinstance(actual_value, dict):
        if isinstance(expected_value, dict):
            for key, nested_expected in expected_value.items():
                if key not in actual_value:
                    return False
                if not _value_matches_filter(actual_value[key], nested_expected, key):
                    return False
            return True

        return any(
            _value_matches_filter(candidate, expected_value, field_name)
            for candidate in actual_value.values()
        )

    if isinstance(expected_value, dict):
        return False

    actual_bool = _coerce_bool(actual_value)
    expected_bool = _coerce_bool(expected_value)
    if actual_bool is not None and expected_bool is not None and field_name in GET_BOOL_FIELDS:
        return actual_bool == expected_bool

    if field_name in GET_NUMERIC_FIELDS or isinstance(actual_value, (int, float)) or isinstance(expected_value, (int, float)):
        actual_number = _coerce_number(actual_value)
        expected_number = _coerce_number(expected_value)
        if actual_number is not None and expected_number is not None:
            return abs(actual_number - expected_number) < 1e-9

    if field_name in GET_DATE_FIELDS:
        actual_date = _stringify_date_like(actual_value)
        expected_date = _stringify_date_like(expected_value)
        if actual_date is None or expected_date is None:
            return False

        actual_date = actual_date.casefold()
        expected_date = expected_date.casefold()
        return actual_date.startswith(expected_date) or expected_date.startswith(actual_date)

    actual_text = str(actual_value).strip().casefold()
    expected_text = str(expected_value).strip().casefold()
    return actual_text == expected_text


def _photo_matches_filters(uuid, metadata, filters):
    if not filters:
        return True

    normalized_metadata = metadata or {}

    for key, expected_value in filters.items():
        normalized_key = key.strip().lower()
        if normalized_key in GET_RESERVED_FIELDS:
            continue

        if normalized_key in {"uuid", "id"}:
            actual_value = uuid
        elif normalized_key in {"uuids", "ids"}:
            actual_value = uuid
        elif normalized_key in {"filename", "filenames"}:
            actual_value = normalized_metadata.get("filename")
        elif normalized_key in {"ai_model", "model"}:
            actual_value = _first_non_blank_value(
                normalized_metadata.get("ai_model"),
                normalized_metadata.get("model"),
            )
        elif normalized_key in {"ai_rundate", "ai_run_date", "run_date"}:
            actual_value = _first_non_blank_value(
                normalized_metadata.get("ai_rundate"),
                normalized_metadata.get("run_date"),
            )
        elif normalized_key == "photo_date":
            actual_value = normalized_metadata.get("photo_date")
        elif normalized_key == "capture_time":
            actual_value = normalized_metadata.get("capture_time")
        else:
            actual_value = normalized_metadata.get(key)
            if actual_value is None and key != normalized_key:
                actual_value = normalized_metadata.get(normalized_key)

        if not _value_matches_filter(actual_value, expected_value, normalized_key):
            return False

    return True


def _build_photo_response(uuid, metadata_dict):
    metadata_dict = dict(metadata_dict or {})
    quality_fields = {}
    metadata_fields = {}

    ai_model = _first_non_blank_value(metadata_dict.get("ai_model"), metadata_dict.get("model"))
    ai_rundate = _first_non_blank_value(metadata_dict.get("ai_rundate"), metadata_dict.get("run_date"))
    photo_date = metadata_dict.get("photo_date")
    filename = metadata_dict.get("filename")
    provider = metadata_dict.get("provider")

    for key, value in metadata_dict.items():
        if key in QUALITY_SCORE_FIELDS:
            if value is not None:
                quality_fields[key] = value
            continue

        if key == "keywords":
            keywords_value = value
            if isinstance(value, str) and value:
                try:
                    keywords_value = json.loads(value)
                except json.JSONDecodeError:
                    keywords_value = value

            metadata_fields[key] = keywords_with_generation_model(
                keywords_value,
                provider,
                ai_model or metadata_dict.get("model"),
            )
            continue

        if key == "tokens_used" and isinstance(value, str) and value:
            try:
                metadata_fields[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                metadata_fields[key] = []
            continue

        metadata_fields[key] = value

    if "keywords" not in metadata_fields and (provider or ai_model):
        metadata_fields["keywords"] = keywords_with_generation_model(
            {},
            provider,
            ai_model or metadata_dict.get("model"),
        )

    return {
        "uuid": uuid,
        "filename": filename,
        "provider": provider,
        "ai_model": ai_model,
        "ai_rundate": ai_rundate,
        "photo_date": photo_date,
        "metadata": metadata_fields,
        "quality": quality_fields,
    }


def _fetch_photo_records(requested_uuids=None):
    if hasattr(postgre_service, "get_image_metadatas"):
        if requested_uuids:
            raw_data = postgre_service.get_image_metadatas(ids=requested_uuids)
            ids = raw_data.get("ids", []) if isinstance(raw_data, dict) else []
            metadatas = raw_data.get("metadatas", []) if isinstance(raw_data, dict) else []

            metadata_by_uuid = {}
            for index, photo_uuid in enumerate(ids):
                metadata_by_uuid[photo_uuid] = metadatas[index] if index < len(metadatas) and metadatas[index] else {}

            ordered_ids = [photo_uuid for photo_uuid in requested_uuids if photo_uuid in metadata_by_uuid]
            return ordered_ids, [metadata_by_uuid[photo_uuid] for photo_uuid in ordered_ids]

        raw_data = postgre_service.get_image_metadatas()
        ids = raw_data.get("ids", []) if isinstance(raw_data, dict) else []
        metadatas = raw_data.get("metadatas", []) if isinstance(raw_data, dict) else []
        return ids, metadatas

    candidate_ids = requested_uuids
    if not candidate_ids and hasattr(postgre_service, "get_all_image_ids"):
        candidate_ids = postgre_service.get_all_image_ids()

    ids = []
    metadatas = []
    for photo_uuid in candidate_ids:
        photo_data = postgre_service.get_image(photo_uuid)
        if not photo_data or not photo_data.get("ids"):
            continue

        ids.append(photo_uuid)
        metadatas.append(photo_data.get("metadatas", [{}])[0] if photo_data.get("metadatas") else {})

    return ids, metadatas


def get_photos(data=None):
    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")

    filters = extract_get_filters(data)
    requested_uuids = extract_requested_uuids(filters)
    ids, metadatas = _fetch_photo_records(requested_uuids or None)

    photos = []
    for index, photo_uuid in enumerate(ids):
        metadata_dict = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        if not _photo_matches_filters(photo_uuid, metadata_dict, filters):
            continue

        photos.append(_build_photo_response(photo_uuid, metadata_dict))

    logger.info(f"Returning {len(photos)} photo record(s) for applied filters")
    return {
        "status": "success",
        "count": len(photos),
        "photos": photos,
    }
