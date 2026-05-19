from config import logger
#import service_chroma as chroma_service
import service_postgre as postgre_service
import json
from datetime import datetime as time
from service_index import _flatten_keywords


OPTIONAL_TEXT_FIELDS = (
    'filename',
    'provider',
    'model',
    'ai_model',
    'ai_rundate',
    'capture_time',
)


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ''
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _existing_metadata(existing_record: dict | None) -> dict:
    if not existing_record or not existing_record.get('ids'):
        return {}

    metadatas = existing_record.get('metadatas') or []
    if not metadatas or not isinstance(metadatas[0], dict):
        return {}

    return dict(metadatas[0])


def _metadata_update_from_item(item: dict) -> dict:
    metadata_to_update = {}

    if not _is_blank(item.get('keywords')):
        logger.debug(f"Importing keywords for UUID {item.get('uuid')}: {item['keywords']}")
        metadata_to_update['keywords'] = json.dumps(item['keywords'])
        metadata_to_update['flattened_keywords'] = _flatten_keywords(item['keywords'])

    for key in ('title', 'caption', 'alt_text'):
        if not _is_blank(item.get(key)):
            metadata_to_update[key] = item[key]

    for key in OPTIONAL_TEXT_FIELDS:
        if not _is_blank(item.get(key)):
            metadata_to_update[key] = item[key]

    if not _is_blank(item.get('exif')):
        metadata_to_update['exif'] = item['exif']

    return metadata_to_update


def import_metadata_task(metadata_items: list[dict]) -> tuple[int, int]:
    """
    Process a batch of metadata imports.
    
    Args:
        metadata_items: List of dictionaries, each with uuid and metadata.
        
    Returns:
        Tuple of (success_count, failure_count)
    """
    success_count = 0
    failure_count = 0
    total_items = len(metadata_items)

    logger.info(f"Starting metadata import task for {total_items} items...")

    for item in metadata_items:
        uuid = item.get('uuid')
        if not uuid:
            logger.warning("Skipping item due to missing uuid.")
            failure_count += 1
            continue

        try:
            existing_record = postgre_service.get_image(uuid)

            metadata_to_update = _metadata_update_from_item(item)

            if not metadata_to_update:
                if existing_record and existing_record.get('ids'):
                    logger.info(f"No nonblank metadata provided for existing UUID {uuid}. Preserving existing record.")
                    success_count += 1
                else:
                    logger.warning(f"No metadata provided to update for UUID {uuid}. Skipping.")
                    failure_count += 1
                continue

            # If the record doesn't exist in the DB we will add a metadata-only
            # entry (no embedding). This makes it possible to import metadata
            # independently of embeddings.
            if not existing_record or not existing_record['ids']:
                metadata_to_update['run_date'] = time.now().strftime("%Y-%m-%d %H:%M:%S")
                postgre_service.add_image(uuid, None, metadata_to_update)
                logger.info(f"Created metadata-only entry for UUID {uuid}.")
                success_count += 1
                continue

            metadata_to_update['run_date'] = time.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata_to_update = {
                **_existing_metadata(existing_record),
                **metadata_to_update,
            }

            postgre_service.update_image(uuid, metadata_to_update)
            logger.info(f"Successfully imported metadata for UUID {uuid}.")
            success_count += 1

        except Exception as e:
            logger.error(f"Error importing metadata for UUID {uuid}: {str(e)}", exc_info=True)
            failure_count += 1
            
    return success_count, failure_count
