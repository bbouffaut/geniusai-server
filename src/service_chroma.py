import chromadb
from chromadb.config import Settings
import os
import numpy as np
from config import DB_PATH, logger


# --- ChromaDB Client and Collection Initialization (Lazy) ---
chroma_client = None
collection = None

def _ensure_initialized():
    """Initialize ChromaDB client and collection on first use (lazy loading)."""
    global chroma_client, collection
    if chroma_client is not None:
        return
    
    logger.info("Initializing ChromaDB client (lazy)...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    
    # Initialize image_embeddings collection
    try:
        collection = chroma_client.get_collection(name="image_embeddings")
        logger.info("Loaded existing ChromaDB image_embeddings collection.")
    except Exception:
        collection = chroma_client.create_collection(name="image_embeddings")
        logger.info("Created new ChromaDB image_embeddings collection.")


def add_image(uuid, embedding, metadata):
    """Add a new image record to the Chroma collection.

    embedding may be None for metadata-only records; in that case we add
    a dummy zero vector with the expected dimensionality (1152) to satisfy
    ChromaDB's requirements while still allowing metadata-only storage.
    
    Note: Metadata-only entries are marked with has_embedding=False in their
    metadata and are filtered out of semantic search results in service_search.py.
    They can still be found via metadata keyword searches.
    """
    _ensure_initialized()
    try:
        if embedding is None:
            # Add metadata-only record with a dummy zero embedding
            # The collection expects 1152-dimensional embeddings (from vision model)
            dummy_embedding = np.zeros(1152, dtype=np.float32).tolist()
            collection.add(embeddings=[dummy_embedding], metadatas=[metadata], ids=[uuid])
            logger.debug(f"image {uuid} with NO Embeddings is well ADDED in collection. Metadata = {metadata}")
        else:
            collection.add(embeddings=[embedding], metadatas=[metadata], ids=[uuid])
            logger.debug(f"image {uuid} with Embeddings is well ADDED in collection. Metadata = {metadata}")

    except Exception as e:
        # Surface a helpful log message and re-raise so callers can decide what to do.
        logger.error(f"Failed to add image {uuid} to ChromaDB (embedding provided: {embedding is not None}): {e}", exc_info=True)
        raise


def update_image(uuid, metadata, embedding=None):
    _ensure_initialized()
    if embedding is not None:
        collection.update(ids=[uuid], metadatas=[metadata], embeddings=[embedding])
        logger.debug(f"image {uuid} with Embeddings is well UPDATED in collection. Metadata = {metadata}")
    else:
        collection.update(ids=[uuid], metadatas=[metadata])
        logger.debug(f"image {uuid} with NO Embeddings is well UPDATED in collection. Metadata = {metadata}")



def get_image(uuid):
    _ensure_initialized()
    result = collection.get(ids=[uuid], include=['metadatas', 'embeddings'])
    if not result or not result.get('ids'):
        return None
    return result


def delete_image(uuid):
    _ensure_initialized()
    collection.delete(ids=[uuid])


def query_images(query_embedding, n_results, where_clause=None):
    _ensure_initialized()
    try:
        return collection.query(
            where=where_clause,
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['metadatas', 'distances']
        )
    except Exception as e:
        logger.error(f"Error querying images: {e}", exc_info=True)
        return {'ids': [[]], 'distances': [[]], 'metadatas': [[]]}

def get_all_image_ids(has_embedding=None):
    """Get all image IDs, optionally filtered by embedding status.
    
    Args:
        has_embedding: If True, only return IDs with real embeddings.
                      If False, only return IDs with dummy embeddings.
                      If None, return all IDs.
    """
    _ensure_initialized()
    if has_embedding is None:
        return collection.get(include=[])['ids']
    
    # Need to get metadata to filter by has_embedding flag
    result = collection.get(include=['metadatas'])
    filtered_ids = []
    
    for i, metadata in enumerate(result['metadatas']):
        # Default to True for backwards compatibility with existing entries
        has_emb = metadata.get('has_embedding', True) if metadata else True
        if has_emb == has_embedding:
            filtered_ids.append(result['ids'][i])
    
    return filtered_ids


def group_and_sort_images(uuids, phash_threshold, clip_threshold, time_delta):
    """
    [NOT IMPLEMENTED] Groups a list of images by similarity and sorts them by quality.
    """
    logger.warning("group_and_sort_images is not yet implemented.")
    return []


def get_db_stats():
    _ensure_initialized()
    count = collection.count()
    db_size = sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(DB_PATH) for filename in filenames) / (1024 * 1024)
    
    min_aesthetic_score, max_aesthetic_score, aesthetic_rated_count = None, None, 0
    min_technical_score, max_technical_score, technical_rated_count = None, None, 0
    phash_count = 0
    capture_time_count = 0
    
    all_metadatas = collection.get(include=['metadatas'])['metadatas']
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
