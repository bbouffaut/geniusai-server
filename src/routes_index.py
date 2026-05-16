from flask import Blueprint, request, jsonify
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

INDEX_UPLOAD_REQUIRED_FIELDS = ("image", "uuid", "filename")
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
    "date_time",
    "tasks",
}

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
    options['date_time'] = data.get('date_time')

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

    images = _extract_uploaded_images(request.files)
    uuids = _extract_uploaded_uuids(request.form)
    filenames = _extract_uploaded_filenames(request.form)
    options = _extract_options(request.form)

    required_values = dict(zip(INDEX_UPLOAD_REQUIRED_FIELDS, (images, uuids, filenames)))
    missing_fields = [field for field, values in required_values.items() if not values]
    if missing_fields:
        return jsonify({
            "error": (
                f"Missing required fields: {', '.join(missing_fields)}. "
                "Upload files as multipart/form-data with repeated 'image', 'uuid', and 'filename' fields."
            )
        }), 400

    if len(images) != len(uuids) or len(images) != len(filenames):
        return jsonify({
            "error": (
                "Mismatch between the number of images, UUIDs, and filenames. "
                "Upload files as multipart/form-data with matching repeated 'image', 'uuid', and 'filename' fields."
            )
        }), 400

    batch_size = len(images)
    _record_batch_timing(batch_size)
    additional_metadata = _extract_uploaded_metadata(request.form, batch_size)

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
    
    # Extract required fields
    image = data.get('image')
    uuid = data.get('uuid')
    filename = data.get('filename')

    missing_fields = [field for field in ("image", "uuid", "filename") if not data.get(field)]
    if missing_fields:
        logger.info(f"{image}, {uuid}, {filename}")
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    options = _extract_options(data)
    additional_metadata = [_extract_json_metadata(data)]

    success_count, failure_count = _process_image_task_with_metadata(
        [(base64.b64decode(image.encode('ascii')), uuid, filename)],
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
    if 'uuid' not in request.json:
        return jsonify({"error": "No uuid provided"}), 400
    uuid = request.json.get('uuid')
    
    try:
        postgre_service.delete_image(uuid)
        logger.info(f"Image ID {uuid} removed from PostgreSQL.")
        return jsonify({"status": "removed", "uuid": uuid})
    except Exception as e:
        logger.error(f"Error removing image {uuid}: {e}")
        return jsonify({"error": "UUID not found or error during removal"}), 404
        

@index_bp.route('/get', methods=['POST'])
def get_photo_data():
    """
    Retrieves metadata and quality scores for a photo by UUID.
    
    JSON body parameters:
    - uuid (string): The UUID of the photo to retrieve
    
    Returns:
    - status: "success" or "error"
    - uuid: The photo's UUID
    - metadata: Dictionary with all metadata fields (title, caption, keywords, etc.)
    - quality: Dictionary with quality scores (overall_score, composition_score, etc.)
    """
    logger.info("Get photo data request received")
    
    if 'uuid' not in request.json:
        return jsonify({"status": "error", "error": "No uuid provided"}), 400
    
    uuid = request.json.get('uuid')
    
    try:
        # Get photo data from PostgreSQL.
        photo_data = postgre_service.get_image(uuid)
        logger.debug(f"Retrieved photo data for UUID {uuid}: {photo_data}")
        
        if not photo_data or not photo_data['ids']:
            logger.warning(f"Photo with UUID {uuid} not found in database")
            return jsonify({"status": "error", "error": "Photo not found"}), 404
        
        # Extract metadata
        metadata_dict = photo_data['metadatas'][0] if photo_data['metadatas'] else {}
        
        # Separate metadata into user-facing metadata and quality scores
        metadata_fields = {}
        quality_fields = {}
        
        # Quality score field names
        quality_keys = {
            'overall_score', 'composition_score', 'lighting_score', 
            'motiv_score', 'colors_score', 'emotion_score', 'quality_critique'
        }
        
        # User metadata field names (from metadata generation)
        metadata_keys = {
            'title', 'caption', 'keywords', 'alt_text'
        }
        
        ai_model = metadata_dict.get('model')
        ai_rundate = metadata_dict.get('run_date')

        for key, value in metadata_dict.items():
            if key in quality_keys:
                quality_fields[key] = value
            elif key in metadata_keys:
                logger.info(f"Processing metadata field {key}: {value}")
                # Keywords must be returned as JSON string (not parsed) for plugin to handle
                if key == 'keywords' and isinstance(value, str) and value:
                    # Keep keywords as JSON string for plugin to parse
                    # The plugin expects either:
                    # - JSON array: ["kw1", "kw2"]
                    # - JSON object: {"Category": ["kw1"], ...}
                    metadata_fields[key] = keywords_with_generation_model(
                        json.loads(value),
                        metadata_dict.get('provider'),
                        metadata_dict.get('model'),
                    )
                elif key == 'keywords':
                    metadata_fields[key] = keywords_with_generation_model(
                        value,
                        metadata_dict.get('provider'),
                        metadata_dict.get('model'),
                    )
                elif key == 'tokens_used' and isinstance(value, str) and value:
                    try:
                        metadata_fields[key] = json.loads(value) if value else []
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(f"Error decoding JSON for {key}: {value}")
                        metadata_fields[key] = []
                else:
                    metadata_fields[key] = value

        if "keywords" not in metadata_fields and (metadata_dict.get('provider') or metadata_dict.get('model')):
            metadata_fields["keywords"] = keywords_with_generation_model(
                {},
                metadata_dict.get('provider'),
                metadata_dict.get('model'),
            )
        
        logger.info(f"Retrieved data for photo {uuid}: {len(metadata_fields)} metadata fields, {len(quality_fields)} quality fields")
        
        return jsonify({
            "status": "success",
            "uuid": uuid,
            "metadata": metadata_fields,
            "quality": quality_fields,
            "ai_model": ai_model,
            "ai_rundate": ai_rundate
        })
        
    except Exception as e:
        logger.error(f"Error retrieving photo data for {uuid}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


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
