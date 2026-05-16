import json
import numpy as np
import unicodedata

#import service_chroma as chroma_service
import service_postgre as postgre_service
from config import DEFAULT_MIN_PERTINENCE_SCORE, logger
import server_lifecycle as server_lifecycle

QUALITY_SCORE_FIELDS = [
    ("overall", "overall_score"),
    ("composition", "composition_score"),
    ("lighting", "lighting_score"),
    ("motiv", "motiv_score"),
    ("colors", "colors_score"),
    ("emotion", "emotion_score"),
]


def _clamp_score(score):
    if not np.isfinite(score):
        return 0.0
    return min(1.0, max(0.0, float(score)))


def _score_from_distance(distance):
    # pgvector cosine distance is 1 - cosine similarity.
    if distance is None:
        return 0.0
    return _clamp_score(1.0 - float(distance))


def _score_from_embedding(query_embedding, result_embedding, distance):
    """Returns a 0..1 pertinence score for normalized text metadata embeddings."""
    if result_embedding is None:
        return _score_from_distance(distance)

    try:
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        result_vector = np.asarray(result_embedding, dtype=np.float32)

        query_norm = np.linalg.norm(query_vector)
        result_norm = np.linalg.norm(result_vector)
        if query_norm == 0 or result_norm == 0:
            return _score_from_distance(distance)

        cosine_similarity = np.dot(query_vector, result_vector) / (query_norm * result_norm)
        return _clamp_score(cosine_similarity)
    except (TypeError, ValueError):
        return _score_from_distance(distance)


def _first_result_group(results, key):
    values = results.get(key)
    if values is None:
        return []

    try:
        if len(values) == 0:
            return []
        group = values[0]
        if group is None:
            return []
        return group
    except TypeError:
        return []


def _normalize_search_text(value):
    normalized = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _metadata_value_to_search_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return _metadata_value_to_search_text(json.loads(stripped))
            except json.JSONDecodeError:
                return stripped
        return stripped
    if isinstance(value, dict):
        return " ".join(_metadata_value_to_search_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_metadata_value_to_search_text(item) for item in value)
    return str(value)


def _metadata_by_uuid(metadata_results):
    metadata_by_id = {}
    if not metadata_results:
        return metadata_by_id

    ids = metadata_results.get('ids', [])
    metadatas = metadata_results.get('metadatas', [])
    for index, uuid in enumerate(ids):
        metadata_by_id[uuid] = metadatas[index] if index < len(metadatas) and metadatas[index] else {}

    return metadata_by_id


def _display_text(value, fallback="-", max_length=80):
    if value is None:
        return fallback

    text = " ".join(str(value).split())
    if not text:
        return fallback
    if len(text) > max_length:
        return f"{text[:max_length - 3]}..."
    return text


def _display_score(value, digits=4):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _format_quality_scores(metadata):
    score_parts = []
    for label, key in QUALITY_SCORE_FIELDS:
        score = _display_score(metadata.get(key), digits=2)
        if score != "-":
            score_parts.append(f"{label}={score}")

    return ", ".join(score_parts) if score_parts else "quality=n/a"


def _log_retrieved_photos(term, final_results, metadata_by_id):
    if not final_results:
        logger.info(f"Search results for '{term}': no photos retrieved")
        return

    lines = [f"Search results for '{term}': {len(final_results)} photo(s) retrieved"]
    for rank, result in enumerate(final_results, start=1):
        metadata = metadata_by_id.get(result.get('uuid'), {})
        title = _display_text(metadata.get('title'), fallback="Untitled", max_length=70)
        filename = _display_text(metadata.get('filename'), fallback="unknown filename", max_length=60)
        pertinence_score = _display_score(result.get('pertinence_score'), digits=4)
        distance = _display_score(result.get('distance'), digits=4)
        match_type = result.get('match_type', "-")
        quality_scores = _format_quality_scores(metadata)

        lines.append(
            f"  {rank:>3}. title=\"{title}\" | filename=\"{filename}\" | "
            f"pertinence={pertinence_score} | distance={distance} | match={match_type} | {quality_scores}"
        )

    logger.info("\n".join(lines))


def _transform_and_sort_results(results, quality_sort, query_embedding, min_pertinence_score):
    """Transforms vector DB results and sorts them based on quality or distance."""
    if not results:
        return []

    ids = _first_result_group(results, 'ids')
    if len(ids) == 0:
        return []

    distances = _first_result_group(results, 'distances')
    metadatas = _first_result_group(results, 'metadatas')
    embeddings = _first_result_group(results, 'embeddings')

    transformed_results = []
    for i in range(len(ids)):
        # Skip metadata-only entries (with dummy embeddings) from semantic search
        metadata = metadatas[i] if i < len(metadatas) else {}
        if metadata and not metadata.get('has_embedding', True):
            continue

        distance = distances[i] if i < len(distances) else None
        embedding = embeddings[i] if i < len(embeddings) else None
        pertinence_score = _score_from_embedding(query_embedding, embedding, distance)
        if pertinence_score < min_pertinence_score:
            continue

        transformed_results.append({
            "uuid": ids[i],
            "filename": metadata.get("filename"),
            "distance": float(round(distance, 4)) if distance is not None else None,
            "pertinence_score": float(round(pertinence_score, 4)),
            "match_type": "semantic",
        })

    transformed_results.sort(
        key=lambda x: (
            -x['pertinence_score'],
            x['distance'] if x['distance'] is not None else float('inf'),
        )
    )
    return transformed_results


def search_images(term, quality_sort, uuids_to_search, min_pertinence_score=DEFAULT_MIN_PERTINENCE_SCORE):
    logger.info(
        f"Searching for '{term}' (quality_sort: {quality_sort}, scoped: {uuids_to_search is not None}, "
        f"min_pertinence_score: {min_pertinence_score})"
    )

    # 1. Semantic search over metadata text embeddings
    query_embedding = server_lifecycle.embed_query(term)
    if query_embedding is not None:
        db_results = postgre_service.query_images(
            query_embedding=query_embedding,
            n_results=300,
            where_clause={"uuid": {"$in": uuids_to_search}} if uuids_to_search else None,
            include_embeddings=True,
        )

        sorted_semantic_results = _transform_and_sort_results(
            db_results,
            quality_sort,
            query_embedding,
            min_pertinence_score,
        )
        semantic_uuids = {res['uuid'] for res in sorted_semantic_results}
    else:
        logger.info("Text embedding model not loaded, skipping semantic metadata search.")
        sorted_semantic_results = []
        semantic_uuids = set()

    # 2. Metadata Search (in-memory)
    logger.info("Performing exact metadata search in-memory. This may be slow for large databases without a UUID filter.")

    if uuids_to_search:
        target_uuids = list(uuids_to_search)
        all_metadata_raw = postgre_service.get_image_metadatas(ids=target_uuids)
    else:
        all_metadata_raw = postgre_service.get_image_metadatas()

    metadata_by_id = _metadata_by_uuid(all_metadata_raw)
    metadata_uuids = set()
    normalized_term = _normalize_search_text(term)

    for i, uuid in enumerate(all_metadata_raw['ids']):
        metadata = all_metadata_raw['metadatas'][i]
        if not metadata:
            continue

        metadata_text = _metadata_value_to_search_text(metadata)
        if normalized_term in _normalize_search_text(metadata_text):
            metadata_uuids.add(uuid)

    # 3. Combine results
    for result in sorted_semantic_results:
        if result['uuid'] in metadata_uuids:
            result['metadata_match'] = True
            result['match_type'] = "semantic+metadata"

    metadata_only_uuids = metadata_uuids - semantic_uuids
    metadata_only_results = [
        {
            "uuid": uuid,
            "filename": metadata_by_id.get(uuid, {}).get("filename"),
            "distance": None,
            "pertinence_score": 1.0,
            "match_type": "metadata",
            "metadata_match": True,
        }
        for uuid in metadata_only_uuids
    ]

    final_results = sorted_semantic_results + metadata_only_results
    final_results.sort(
        key=lambda x: (
            -x['pertinence_score'],
            x['distance'] if x['distance'] is not None else float('inf'),
        )
    )

    logger.info(f"Total results: {len(final_results)} ({len(sorted_semantic_results)} semantic, {len(metadata_only_results)} metadata-only)")
    _log_retrieved_photos(term, final_results, metadata_by_id)

    return final_results


def group_similar_images(uuids, phash_threshold, time_delta):
    """Groups a list of images by similarity and sorts them by quality."""
    logger.info(f"Grouping {len(uuids)} UUIDs with phash_threshold='{phash_threshold}' and time_delta='{time_delta}s'.")

    try:
        grouped_results = postgre_service.group_and_sort_images(uuids, phash_threshold, time_delta)
        return grouped_results
    except Exception as e:
        logger.error(f"Error during similarity grouping: {str(e)}")
        raise e
