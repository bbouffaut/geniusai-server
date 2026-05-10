import chromadb
from chromadb.config import Settings
import os
import numpy as np
from config import DB_PATH, TEXT_EMBEDDING_DIMENSION, logger


# --- ChromaDB Client and Collection Initialization (Lazy) ---
METADATA_COLLECTION_NAME = "metadata_embeddings"
LEGACY_IMAGE_COLLECTION_NAME = "image_embeddings"
chroma_client = None
collection = None
legacy_image_collection = None

def _ensure_initialized():
    """Initialize ChromaDB client and collection on first use (lazy loading)."""
    global chroma_client, collection, legacy_image_collection
    if chroma_client is not None:
        return
    
    logger.info("Initializing ChromaDB client (lazy)...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))

    # Store/search metadata embeddings separately from the previous image-vector
    # collection because Chroma collection dimensions are immutable.
    try:
        collection = chroma_client.get_collection(name=METADATA_COLLECTION_NAME)
        logger.info(f"Loaded existing ChromaDB {METADATA_COLLECTION_NAME} collection.")
    except Exception:
        collection = chroma_client.create_collection(name=METADATA_COLLECTION_NAME)
        logger.info(f"Created new ChromaDB {METADATA_COLLECTION_NAME} collection.")

    try:
        legacy_image_collection = chroma_client.get_collection(name=LEGACY_IMAGE_COLLECTION_NAME)
        logger.info(f"Loaded legacy ChromaDB {LEGACY_IMAGE_COLLECTION_NAME} collection for metadata fallback.")
    except Exception:
        legacy_image_collection = None


def _upsert_record(uuid, metadata, embedding=None, document=None):
    if embedding is None:
        embedding = np.zeros(TEXT_EMBEDDING_DIMENSION, dtype=np.float32).tolist()

    payload = {
        "ids": [uuid],
        "metadatas": [metadata],
        "embeddings": [embedding],
    }
    if document is not None:
        payload["documents"] = [document]

    collection.upsert(**payload)


def _active_record_exists(uuid):
    result = collection.get(ids=[uuid], include=[])
    return bool(result and result.get("ids"))


def add_image(uuid, embedding, metadata, document=None):
    """Add a new photo metadata record to the Chroma collection.

    embedding may be None for metadata-only records; in that case we add
    a dummy zero vector with the expected text embedding dimensionality to satisfy
    ChromaDB's requirements while still allowing metadata-only storage.
    """
    _ensure_initialized()
    try:
        _upsert_record(uuid, metadata, embedding=embedding, document=document)
        logger.debug(f"image {uuid} is well UPSERTED in metadata collection. Metadata = {metadata}")
    except Exception as e:
        # Surface a helpful log message and re-raise so callers can decide what to do.
        logger.error(f"Failed to add image {uuid} to ChromaDB (embedding provided: {embedding is not None}): {e}", exc_info=True)
        raise


def update_image(uuid, metadata, embedding=None, document=None):
    _ensure_initialized()
    if not _active_record_exists(uuid):
        _upsert_record(uuid, metadata, embedding=embedding, document=document)
        logger.debug(f"image {uuid} was missing from metadata collection and is now UPSERTED. Metadata = {metadata}")
        return

    payload = {
        "ids": [uuid],
        "metadatas": [metadata],
    }
    if embedding is not None:
        payload["embeddings"] = [embedding]
    if document is not None:
        payload["documents"] = [document]

    collection.update(**payload)
    logger.debug(f"image {uuid} is well UPSERTED in metadata collection. Metadata = {metadata}")


def get_image(uuid):
    _ensure_initialized()
    result = collection.get(ids=[uuid], include=['metadatas', 'embeddings', 'documents'])
    if not result or not result.get('ids'):
        if legacy_image_collection is None:
            return None
        result = legacy_image_collection.get(ids=[uuid], include=['metadatas', 'embeddings'])
    if not result or not result.get('ids'):
        return None
    return result


def delete_image(uuid):
    _ensure_initialized()
    collection.delete(ids=[uuid])
    if legacy_image_collection is not None:
        try:
            legacy_image_collection.delete(ids=[uuid])
        except Exception:
            pass


def query_images(query_embedding, n_results, where_clause=None, include_embeddings=False):
    _ensure_initialized()
    include = ['metadatas', 'distances']
    if include_embeddings:
        include.append('embeddings')

    try:
        return collection.query(
            where=where_clause,
            query_embeddings=query_embedding,
            n_results=n_results,
            include=include
        )
    except Exception as e:
        logger.error(f"Error querying images: {e}", exc_info=True)
        fallback = {'ids': [[]], 'distances': [[]], 'metadatas': [[]]}
        if include_embeddings:
            fallback['embeddings'] = [[]]
        return fallback


def _empty_metadata_result():
    return {"ids": [], "metadatas": []}


def _get_metadatas_from(collection_obj, ids=None):
    if collection_obj is None:
        return _empty_metadata_result()

    if ids:
        return collection_obj.get(ids=ids, include=["metadatas"])
    return collection_obj.get(include=["metadatas"])


def _merge_metadata_results(primary, secondary):
    merged_ids = []
    merged_metadatas = []
    seen = set()

    for result in (primary, secondary):
        for index, uuid in enumerate(result.get("ids", [])):
            if uuid in seen:
                continue
            seen.add(uuid)
            merged_ids.append(uuid)
            metadatas = result.get("metadatas", [])
            merged_metadatas.append(metadatas[index] if index < len(metadatas) else {})

    return {"ids": merged_ids, "metadatas": merged_metadatas}


def get_image_metadatas(ids=None):
    _ensure_initialized()
    primary = _get_metadatas_from(collection, ids=ids)
    secondary = _get_metadatas_from(legacy_image_collection, ids=ids)
    return _merge_metadata_results(primary, secondary)


def get_all_image_ids(has_embedding=None):
    """Get all image IDs, optionally filtered by embedding status.
    
    Args:
        has_embedding: If True, only return IDs with real embeddings.
                      If False, only return IDs with dummy embeddings.
                      If None, return all IDs.
    """
    _ensure_initialized()
    result = get_image_metadatas()

    if has_embedding is None:
        return result['ids']

    filtered_ids = []

    for i, metadata in enumerate(result['metadatas']):
        # Default to True for backwards compatibility with existing entries
        has_emb = metadata.get('has_embedding', True) if metadata else True
        if has_emb == has_embedding:
            filtered_ids.append(result['ids'][i])
    
    return filtered_ids


def group_and_sort_images(uuids, phash_threshold, time_delta):
    """
    [NOT IMPLEMENTED] Groups a list of images by similarity and sorts them by quality.
    """
    logger.warning("group_and_sort_images is not yet implemented.")
    return []


def get_db_stats():
    _ensure_initialized()
    metadata_result = get_image_metadatas()
    count = len(metadata_result['ids'])
    db_size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(DB_PATH) for filename in filenames) / (1024 * 1024)
    
    min_aesthetic_score, max_aesthetic_score, aesthetic_rated_count = None, None, 0
    min_technical_score, max_technical_score, technical_rated_count = None, None, 0
    phash_count = 0
    capture_time_count = 0
    
    all_metadatas = metadata_result['metadatas']
    for metadata in all_metadatas:
        if not metadata:
            continue
        
        if metadata.get("phash") is not None:
            phash_count += 1

        if metadata.get("capture_time") is not None:
            capture_time_count += 1

        aesthetic_score = metadata.get("aesthetic_score")
        if aesthetic_score is not None:
            aesthetic_rated_count += 1
            if min_aesthetic_score is None or aesthetic_score < min_aesthetic_score: min_aesthetic_score = aesthetic_score
            if max_aesthetic_score is None or aesthetic_score > max_aesthetic_score: max_aesthetic_score = aesthetic_score

    return { 
        "num_images": count, 
        "db_size_mb": round(db_size, 2), 
        "num_with_phash": phash_count,
        "num_rated_aesthetic": aesthetic_rated_count,
        "num_with_capture_time": capture_time_count,
    }
