from flask import Blueprint, request, jsonify
from datetime import datetime
import time
from collections import deque
import os
import shutil
import tempfile

#import service_chroma as postgre_service
import service_postgre as postgre_service
from config import DEFAULT_METADATA_LANGUAGE, UPLOAD_TEMP_DIR, logger
from service_index import keywords_with_generation_model, process_image_task
import base64
import json

index_bp = Blueprint('index', __name__)

# Store timestamps of the last 100 requests to calculate processing speed
request_timestamps = deque(maxlen=100)

INDEX_UPLOAD_REQUIRED_FIELDS = ("image", "uuid", "filename", "capture_time")
INDEX_UNSUPPORTED_CAPTURE_TIME_FIELDS = ("photo_date", "date_time", "photos_date")
INDEX_UPLOAD_RESERVED_FIELDS = {
    "image",
    "images",
    "uuid",
    "uuids",
    "filename",
    "filenames",
    "provider",
    "model",
    "api_key",
    "language",
    "max_tokens",
    "generate_keywords",
    "generate_caption",
    "generate_title",
    "generate_alt_text",
    "submit_gps",
    "submit_keywords",
    "submit_folder_names",
    "existing_keywords",
    "gps_coordinates",
    "folder_names",
    "user_context",
    "keyword_categories",
    "replace_ss",
    "regenerate_metadata",
    "regenerateMetadata",
    "prompt",
    "capture_time",
    "tasks",
}

def _logged_form_values(form):
    values = {}
    for key in form.keys():
        items = form.getlist(key)
        if key == "api_key":
            items = ["<redacted>" if item else item for item in items]
        values[key] = items[0] if len(items) == 1 else items
    return values


def _logged_upload_files(images):
    return [
        {
            "field": "image",
            "filename": image.filename,
            "content_type": image.content_type,
            "content_length": image.content_length,
        }
        for image in images
    ]

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
    "run_date",
}

GET_BOOL_FIELDS = {
    "has_embedding",
}

GET_NUMERIC_FIELDS = QUALITY_SCORE_FIELDS - {"quality_critique"}

def _extract_options(data):
    """Extracts options from request data (form or json)."""
    options = {}
    try:
        from config import logger as _tmp_logger
        _tmp_logger.info(f"Raw indexing option keys received: {list(getattr(data, 'keys', lambda: [])())}")
    except Exception:
        pass
    options['provider'] = data.get('provider')
    options['model'] = data.get('model')
    options['api_key'] = data.get('api_key')
    options['language'] = data.get('language', DEFAULT_METADATA_LANGUAGE)
    options['max_tokens'] = data.get('max_tokens')
    options['generate_keywords'] = str(data.get('generate_keywords', 'true')).lower() == 'true'
    options['generate_caption'] = str(data.get('generate_caption', 'true')).lower() == 'true'
    options['generate_title'] = str(data.get('generate_title', 'true')).lower() == 'true'
    options['generate_alt_text'] = str(data.get('generate_alt_text', 'true')).lower() == 'true'
    options['submit_gps'] = str(data.get('submit_gps', 'false')).lower() == 'true'
    options['submit_keywords'] = str(data.get('submit_keywords', 'false')).lower() == 'true'
    options['submit_folder_names'] = str(data.get('submit_folder_names', 'false')).lower() == 'true'
    options['existing_keywords'] = data.get('existing_keywords')
    options['gps_coordinates'] = data.get('gps_coordinates')
    options['folder_names'] = data.get('folder_names')
    options['user_context'] = data.get('user_context')
    
    keyword_categories_raw = data.get('keyword_categories', '[]')
    if isinstance(keyword_categories_raw, str):
        try:
            keyword_categories = json.loads(keyword_categories_raw)
        except json.JSONDecodeError:
            keyword_categories = []
    else:
        keyword_categories = keyword_categories_raw

    if isinstance(keyword_categories, dict):
        keyword_categories = {
            name: children
            for name, children in keyword_categories.items()
            if isinstance(children, dict) and children
        }

    options['keyword_categories'] = keyword_categories

    options['replace_ss'] = str(data.get('replace_ss', 'false')).lower() == 'true'
    # Support both snake_case and camelCase keys from clients
    reg_val = data.get('regenerate_metadata')
    if reg_val is None:
        reg_val = data.get('regenerateMetadata', 'true')
    options['regenerate_metadata'] = str(reg_val).lower() == 'true'
    options['prompt'] = data.get('prompt')
    options['capture_time'] = data.get('capture_time')

    tasks_raw = data.get('tasks')
    if tasks_raw:
        if isinstance(tasks_raw, str):
            try:
                tasks = json.loads(tasks_raw) if tasks_raw.startswith('[') else [t.strip() for t in tasks_raw.split(',')]
            except (json.JSONDecodeError, AttributeError):
                tasks = [t.strip() for t in tasks_raw.split(',')]
        else:
            tasks = tasks_raw
    else:
        tasks = ['metadata', 'embeddings', 'quality'] # Default tasks

    options['compute_embeddings'] = 'embeddings' in tasks
    options['compute_metadata'] = 'metadata' in tasks
    options['compute_quality'] = 'quality' in tasks
    
    return options


def _normalize_capture_time_value(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    return text


def _unsupported_capture_time_fields(data):
    return [field for field in INDEX_UNSUPPORTED_CAPTURE_TIME_FIELDS if field in data]


def _extract_uploaded_images(files):
    for field_name in ('image', 'images'):
        uploaded_images = files.getlist(field_name)
        if uploaded_images:
            return uploaded_images
    return []


def _extract_uploaded_uuids(form):
    uuids = form.getlist('uuid')
    if uuids:
        return [uuid.strip() for uuid in uuids if str(uuid).strip()]

    uuids = form.getlist('uuids')
    if len(uuids) > 1:
        return [uuid.strip() for uuid in uuids if str(uuid).strip()]

    raw_uuids = form.get('uuids')
    if not raw_uuids:
        return []

    if isinstance(raw_uuids, str):
        try:
            parsed_uuids = json.loads(raw_uuids)
        except json.JSONDecodeError:
            parsed_uuids = [uuid.strip() for uuid in raw_uuids.split(',')]
    else:
        parsed_uuids = raw_uuids

    if isinstance(parsed_uuids, list):
        return [str(uuid).strip() for uuid in parsed_uuids if str(uuid).strip()]

    return []


def _extract_uploaded_filenames(form):
    filenames = form.getlist('filename')
    if filenames:
        return [filename.strip() for filename in filenames if str(filename).strip()]

    filenames = form.getlist('filenames')
    if len(filenames) > 1:
        return [filename.strip() for filename in filenames if str(filename).strip()]

    raw_filenames = form.get('filenames')
    if not raw_filenames:
        return []

    if isinstance(raw_filenames, str):
        try:
            parsed_filenames = json.loads(raw_filenames)
        except json.JSONDecodeError:
            parsed_filenames = [filename.strip() for filename in raw_filenames.split(',')]
    else:
        parsed_filenames = raw_filenames

    if isinstance(parsed_filenames, list):
        return [str(filename).strip() for filename in parsed_filenames if str(filename).strip()]

    return []


def _normalize_metadata_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
        return value

    if isinstance(value, list):
        return [_normalize_metadata_value(item) for item in value]

    if isinstance(value, dict):
        return {
            key: _normalize_metadata_value(item)
            for key, item in value.items()
        }

    return value


def _extract_uploaded_metadata(form, batch_size):
    if batch_size <= 0:
        return []

    metadata_by_index = [dict() for _ in range(batch_size)]

    for key in form.keys():
        if key in INDEX_UPLOAD_RESERVED_FIELDS:
            continue

        values = form.getlist(key)
        if not values:
            continue

        normalized_values = [_normalize_metadata_value(value) for value in values]

        if len(normalized_values) == batch_size:
            for index, value in enumerate(normalized_values):
                metadata_by_index[index][key] = value
            continue

        if len(normalized_values) == 1:
            for metadata in metadata_by_index:
                metadata[key] = normalized_values[0]
            continue

        for metadata in metadata_by_index:
            metadata[key] = normalized_values

    return metadata_by_index


def _extract_json_metadata(data):
    metadata = {}
    for key, value in data.items():
        if key in INDEX_UPLOAD_RESERVED_FIELDS:
            continue
        metadata[key] = _normalize_metadata_value(value)

    return metadata


def _process_image_task_with_metadata(image_triplets, options, additional_metadata_list=None):
    try:
        return process_image_task(
            image_triplets,
            options=options,
            additional_metadata_list=additional_metadata_list,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'additional_metadata_list'" not in str(exc):
            raise
        return process_image_task(image_triplets, options)


def _record_batch_timing(batch_size):
    current_time = time.time()
    for _ in range(batch_size):
        request_timestamps.append(current_time)

    if len(request_timestamps) <= 10:
        return

    time_span = request_timestamps[-1] - request_timestamps[0]
    if time_span <= 1:
        return

    images_per_second = len(request_timestamps) / time_span
    logger.info(f"Indexing at {images_per_second:.2f} images/sec")


def _is_upload_temp_path(path):
    if not path:
        return False

    try:
        absolute_path = os.path.abspath(path)
        return (
            os.path.isfile(absolute_path)
            and os.path.commonpath([absolute_path, UPLOAD_TEMP_DIR]) == UPLOAD_TEMP_DIR
        )
    except (TypeError, ValueError):
        return False


def _ensure_uploaded_file_path(file_storage):
    staged_path = getattr(getattr(file_storage, 'stream', None), 'name', None)
    if _is_upload_temp_path(staged_path):
        return os.path.abspath(staged_path), False

    suffix = os.path.splitext(file_storage.filename or '')[1]
    with tempfile.NamedTemporaryFile(
        mode='w+b',
        prefix='geniusai-upload-',
        suffix=suffix,
        dir=UPLOAD_TEMP_DIR,
        delete=False,
    ) as staged_file:
        upload_stream = file_storage.stream
        if hasattr(upload_stream, 'seek'):
            try:
                upload_stream.seek(0)
            except (OSError, ValueError):
                pass
        shutil.copyfileobj(upload_stream, staged_file)
        return staged_file.name, True


def _build_uploaded_image_triplets(images, uuids, filenames):
    image_triplets = []
    upload_failures = 0

    for index, file_storage in enumerate(images):
        uuid = uuids[index]
        filename = filenames[index]
        if not file_storage or not uuid:
            logger.warning("Skipping an entry in the batch due to missing file or uuid.")
            upload_failures += 1
            continue

        staged_path = None
        should_delete_staged_path = False
        try:
            staged_path, should_delete_staged_path = _ensure_uploaded_file_path(file_storage)
            with open(staged_path, 'rb') as staged_file:
                image_data = staged_file.read()

            filename = filename or file_storage.filename or os.path.basename(staged_path)
            logger.debug(f"Prepared uploaded image for UUID {uuid} from temp file {staged_path}")
            image_triplets.append((image_data, uuid, filename))
        except Exception as e:
            logger.error(f"Error preparing uploaded file for UUID {uuid}: {e}", exc_info=True)
            upload_failures += 1
        finally:
            if should_delete_staged_path and staged_path:
                try:
                    os.remove(staged_path)
                except OSError as cleanup_error:
                    logger.warning(f"Failed to remove staged upload temp file {staged_path}: {cleanup_error}")

    return image_triplets, upload_failures


def _index_uploaded_images(request_log_message):
    logger.info(request_log_message)

    unsupported_fields = _unsupported_capture_time_fields(request.form)
    if unsupported_fields:
        return jsonify({
            "error": (
                f"Unsupported capture time field(s): {', '.join(unsupported_fields)}. "
                "Use 'capture_time'."
            )
        }), 400

    images = _extract_uploaded_images(request.files)
    uuids = _extract_uploaded_uuids(request.form)
    filenames = _extract_uploaded_filenames(request.form)
    options = _extract_options(request.form)
    options['capture_time'] = _normalize_capture_time_value(options.get('capture_time'))
    batch_size = len(images)
    additional_metadata = _extract_uploaded_metadata(request.form, batch_size)
    logger.info(
        "Index multipart payload values received: %s",
        json.dumps(
            {
                "form": _logged_form_values(request.form),
                "files": _logged_upload_files(images),
                "additional_metadata": additional_metadata,
            },
            ensure_ascii=False,
            default=str,
        ),
    )

    required_values = {
        "image": images,
        "uuid": uuids,
        "filename": filenames,
        "capture_time": options.get('capture_time'),
    }
    missing_fields = [field for field, values in required_values.items() if not values]
    if missing_fields:
        return jsonify({
            "error": (
                f"Missing required fields: {', '.join(missing_fields)}. "
                "Upload files as multipart/form-data with repeated 'image', 'uuid', 'filename', and 'capture_time' fields."
            )
        }), 400

    if len(images) != len(uuids) or len(images) != len(filenames):
        return jsonify({
            "error": (
                "Mismatch between the number of images, UUIDs, and filenames. "
                "Upload files as multipart/form-data with matching repeated 'image', 'uuid', and 'filename' fields."
            )
        }), 400

    _record_batch_timing(batch_size)

    image_triplets, upload_failures = _build_uploaded_image_triplets(images, uuids, filenames)

    if not image_triplets:
        logger.info("No valid uploaded images to process in the batch.")
        return jsonify({
            "status": "processed",
            "success_count": 0,
            "failure_count": upload_failures or batch_size,
        }), 200

    success_count, processing_failures = _process_image_task_with_metadata(
        image_triplets,
        options=options,
        additional_metadata_list=additional_metadata,
    )
    total_failures = upload_failures + processing_failures

    logger.info(f"Batch processing complete. Success: {success_count}, Failures: {total_failures}.")

    if success_count == 0:
        logger.warning("No images were successfully processed in the batch.")
        return jsonify({"error": "No images were successfully processed"}), 500

    return jsonify({
        "status": "processed",
        "success_count": success_count,
        "failure_count": total_failures,
    }), 200


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


def _extract_get_filters(data):
    filters = {}

    for section_key in ("filters", "metadata", "quality"):
        _extend_get_filters(filters, data.get(section_key))

    for key, value in data.items():
        if key in GET_RESERVED_FIELDS:
            continue
        filters[key] = _normalize_get_filter_value(key, value)

    return filters


def _extract_requested_uuids(filters):
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
    capture_time = metadata_dict.get("capture_time")
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
        "capture_time": capture_time,
        "metadata": metadata_fields,
        "quality": quality_fields,
    }


def _request_body_dict():
    if request.is_json:
        data = request.get_json(silent=True)
        if data is not None:
            return data
        if request.content_length in (None, 0):
            return {}
        return None

    if request.form:
        payload = {}
        for key in request.form.keys():
            values = request.form.getlist(key)
            payload[key] = values if len(values) > 1 else request.form.get(key)
        return payload

    raw_body = request.get_data(cache=True, as_text=True)
    if raw_body and raw_body.strip():
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return None

    return {}

@index_bp.route('/index', methods=['POST'])
def index_images_batch():
    """
    Receives a batch of images, processes them synchronously, and indexes them.
    Returns a 200 OK status once all images are processed.
    """
    return _index_uploaded_images("Index request received")

@index_bp.route('/index_base64', methods=['POST'])
def index_images_batch_base64():
    """
    Receives a single image base64 encoded, processes it, and indexes it.
    Returns a 200 OK status once processed.
    """
    logger.info("Index base64 request received")
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    unsupported_fields = _unsupported_capture_time_fields(data)
    if unsupported_fields:
        return jsonify({
            "error": (
                f"Unsupported capture time field(s): {', '.join(unsupported_fields)}. "
                "Use 'capture_time'."
            )
        }), 400
    
    # Extract required fields
    image = data.get('image')
    uuid = data.get('uuid')
    filename = data.get('filename')

    missing_fields = [field for field in INDEX_UPLOAD_REQUIRED_FIELDS if not data.get(field)]
    if missing_fields:
        logger.info(f"{image}, {uuid}, {filename}")
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    options = _extract_options(data)
    options['capture_time'] = _normalize_capture_time_value(options.get('capture_time'))
    image_bytes = base64.b64decode(image.encode('ascii'))

    additional_metadata = [_extract_json_metadata(data)]

    success_count, failure_count = _process_image_task_with_metadata(
        [(image_bytes, uuid, filename)],
        options=options,
        additional_metadata_list=additional_metadata,
    )
    
    logger.info(f"Batch processing complete. Success: {success_count}, Failures: {failure_count}.")

    if success_count == 0:
        logger.warning("No images were successfully processed in the batch.")
        return jsonify({"error": "No images were successfully processed"}), 500
        
    return jsonify({"status": "processed", "success_count": success_count, "failure_count": failure_count}), 200


@index_bp.route('/index_by_reference', methods=['POST'])
def index_images_batch_by_reference():
    """
    Deprecated alias of /index. Path-based indexing is no longer supported.
    Clients must upload image files as multipart/form-data.
    """
    if request.is_json:
        logger.warning("Rejected path-based /index_by_reference payload. Multipart uploads are required.")
        return jsonify({
            "error": (
                "Path-based indexing is no longer supported. "
                "Upload files as multipart/form-data with repeated 'image', 'uuid', and 'filename' fields."
            )
        }), 400

    return _index_uploaded_images("Index by reference request received as multipart upload")


@index_bp.route('/remove', methods=['POST'])
def remove_image():
    logger.info("Remove request received")
    data = request.get_json(silent=True) or {}
    if 'uuid' not in data:
        return jsonify({"error": "No uuid provided"}), 400
    uuid = data.get('uuid')
    
    try:
        postgre_service.delete_image(uuid)
        logger.info(f"Image ID {uuid} removed from PostgreSQL.")
        return jsonify({"status": "removed", "uuid": uuid})
    except Exception as e:
        logger.error(f"Error removing image {uuid}: {e}")
        return jsonify({"error": "UUID not found or error during removal"}), 404
        

@index_bp.route('/get/ids', methods=['GET'])
def get_ids():
    """Get all indexed image IDs, optionally filtered by embedding status.
    
    Query parameters:
        has_embedding (string): 'true' to get only images with real metadata embeddings,
                               'false' to get only images with dummy embeddings,
                               omit to get all images.
    """
    logger.info("Get IDs request received")
    
    # Parse has_embedding parameter
    has_embedding_param = request.args.get('has_embedding')
    has_embedding = None
    if has_embedding_param is not None:
        has_embedding = has_embedding_param.lower() == 'true'
        logger.info(f"Filtering IDs by has_embedding={has_embedding}")
    
    ids_data = postgre_service.get_all_image_ids(has_embedding=has_embedding)
    logger.info(f"Returning {len(ids_data)} image IDs")
    return jsonify(ids_data)
