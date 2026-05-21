from config import TEXT_EMBEDDING_MODEL_ID, logger
#import service_chroma as chroma_service
import service_postgre as postgre_service
from service_metadata import get_analysis_service
import server_lifecycle as server_lifecycle
import json
from datetime import datetime as time


MODEL_KEYWORD_CATEGORY = "AI Model"
UNCATEGORIZED_KEYWORD_CATEGORY = "Keywords"

def _flatten_keywords(keywords):
    """
    Flatten keywords from various formats to a comma-separated string.
    
    Handles:
    - Flat list: ["Keyword1", "Keyword2"] -> "Keyword1, Keyword2"
    - Nested dict: {"Category": ["Kw1", "Kw2"], ...} -> "Kw1, Kw2, ..."
    - Already a string: "Keyword1, Keyword2" -> "Keyword1, Keyword2"
    
    Args:
        keywords: List, dict, or string of keywords
        
    Returns:
        Comma-separated string of all keywords
    """
    if not keywords:
        return ""
    
    if isinstance(keywords, str):
        # Already a string, return as-is
        return keywords
    
    if isinstance(keywords, list):
        # Flat list of strings
        return ', '.join(str(kw) for kw in keywords if kw)
    
    if isinstance(keywords, dict):
        # Nested dict - recursively collect all keywords
        all_keywords = []
        
        def collect_keywords(d):
            for key, value in d.items():
                if isinstance(value, list):
                    # Leaf node with keywords
                    all_keywords.extend(str(kw) for kw in value if kw)
                elif isinstance(value, dict) and value:
                    # Nested dict, recurse
                    collect_keywords(value)
                else:
                    # Single keyword value
                    if value:
                        all_keywords.append(str(key))
        
        collect_keywords(keywords)
        return ', '.join(all_keywords)
    
    return ""


def _generation_model_label(provider, model_name):
    if provider and model_name:
        return f"{provider}/{model_name}"
    if model_name:
        return str(model_name)
    if provider:
        return str(provider)
    return None


def _append_unique_keyword(values, keyword):
    if isinstance(values, list):
        updated = [value for value in values if value]
    elif values:
        updated = [values]
    else:
        updated = []

    keyword_text = str(keyword)
    if keyword_text not in [str(value) for value in updated]:
        updated.append(keyword_text)

    return updated


def keywords_with_generation_model(keywords, provider, model_name):
    model_label = _generation_model_label(provider, model_name)
    if not model_label:
        return keywords

    if isinstance(keywords, str):
        stripped = keywords.strip()
        if stripped:
            try:
                keywords = json.loads(stripped)
            except json.JSONDecodeError:
                keywords = [stripped]
        else:
            keywords = {}

    if isinstance(keywords, dict):
        updated_keywords = keywords.copy()
        updated_keywords[MODEL_KEYWORD_CATEGORY] = _append_unique_keyword(
            updated_keywords.get(MODEL_KEYWORD_CATEGORY),
            model_label,
        )
        return updated_keywords

    if isinstance(keywords, list):
        return {
            UNCATEGORIZED_KEYWORD_CATEGORY: keywords,
            MODEL_KEYWORD_CATEGORY: [model_label],
        }

    return {MODEL_KEYWORD_CATEGORY: [model_label]}


def _store_generation_model_keyword(metadata):
    keywords_with_model = keywords_with_generation_model(
        metadata.get("keywords"),
        metadata.get("provider"),
        metadata.get("model"),
    )
    if not keywords_with_model:
        return

    metadata["keywords"] = json.dumps(keywords_with_model)
    metadata["flattened_keywords"] = _flatten_keywords(keywords_with_model)


_RESERVED_METADATA_KEYS = {
    "uuid",
    "filename",
    "provider",
    "model",
    "run_date",
    "has_embedding",
    "embedding_model",
    "embedding_source",
    "metadata_search_text",
}


def _merge_additional_metadata(base_metadata, additional_metadata):
    if not additional_metadata:
        return

    for key, value in additional_metadata.items():
        if key in _RESERVED_METADATA_KEYS:
            continue
        if isinstance(base_metadata.get(key), dict) and isinstance(value, dict):
            merged_value = base_metadata[key].copy()
            merged_value.update(value)
            base_metadata[key] = merged_value
            continue
        base_metadata[key] = value


def _build_base_metadata(uuid, filename, options, extra_metadata=None):
    main_metadata = {
        "provider": options.get("provider") if options else None,
        "model": options.get("model") if options else None,
    }

    _merge_additional_metadata(main_metadata, extra_metadata)

    main_metadata["filename"] = filename
    main_metadata["uuid"] = uuid
    main_metadata["run_date"] = time.now().strftime("%Y-%m-%d %H:%M:%S")
    _ensure_search_fields(main_metadata, options)
    return main_metadata



# ---------------------------------------------------------------------------
# Fields included in the embedding document (allowlist).
# Only human-readable, semantically rich AI-generated fields are embedded.
# Numeric EXIF fields (ISO, focal length, shutter speed, scores…) and
# bookkeeping fields (uuid, filename, model, dates…) are intentionally
# excluded — they dilute the semantic vector and are handled by SQL column
# filters instead.
# The order matters: richer descriptions first so the model sees them early.
# ---------------------------------------------------------------------------
_SEARCHABLE_METADATA_KEYS = [
    "title",
    "caption",
    "alt_text",
    "keywords",
    "flattened_keywords",
    "quality_critique",
]


def _metadata_value_to_text(value):
    if value is None:
        return ""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""

        if stripped[0] in "[{":
            try:
                return _metadata_value_to_text(json.loads(stripped))
            except json.JSONDecodeError:
                return stripped

        return stripped

    if isinstance(value, list):
        return ", ".join(filter(None, (_metadata_value_to_text(item) for item in value)))

    if isinstance(value, dict):
        entries = []
        for key, nested_value in value.items():
            nested_text = _metadata_value_to_text(nested_value)
            if nested_text:
                entries.append(f"{key}: {nested_text}")
        return "; ".join(entries)

    return str(value)


def _build_metadata_embedding_document(metadata):
    """Build a searchable text document from the metadata stored for a photo.

    Only the fields listed in _SEARCHABLE_METADATA_KEYS are included.
    The resulting string is both stored in the ``document`` DB column and
    passed to the embedding model, so they are always identical.
    """
    parts = []
    for key in _SEARCHABLE_METADATA_KEYS:
        value_text = _metadata_value_to_text(metadata.get(key))
        if value_text:
            parts.append(f"{key.replace('_', ' ')}: {value_text}")

    return "\n".join(parts)


def _first_non_blank(*values):
    for value in values:
        if value is not None:
            if isinstance(value, str) and not value.strip():
                continue
            return value
    return None


def _ensure_search_fields(main_metadata, options):
    ai_model = _first_non_blank(main_metadata.get("model"), main_metadata.get("ai_model"))
    if ai_model is not None:
        main_metadata["ai_model"] = ai_model

    ai_rundate = _first_non_blank(main_metadata.get("run_date"), main_metadata.get("ai_rundate"))
    if ai_rundate is not None:
        main_metadata["ai_rundate"] = ai_rundate

    capture_time = _first_non_blank(
        options.get("capture_time") if options else None,
        main_metadata.get("capture_time"),
    )
    if capture_time is not None:
        main_metadata["capture_time"] = capture_time

def process_image_task(
    image_triplets: list[tuple[bytes, str, str]],
    options: dict,
    additional_metadata_list=None,
) -> tuple[int, int]:
    """
    Process a batch of images for indexing.

    Builds metadata, runs LLM analysis and embedding generation, then stores
    each record exactly once (UPSERT). No pre-storing, no multi-stage updates.

    Args:
        image_triplets: List of (image_bytes, uuid, filename) tuples
        options: Dictionary with processing options

    Returns:
        Tuple of (success_count, failure_count)
    """
    success_count = 0
    failure_count = 0
    total_images = len(image_triplets)

    if additional_metadata_list is not None and len(additional_metadata_list) != total_images:
        logger.warning(
            "Ignoring additional metadata because its batch size does not match the image batch size."
        )
        additional_metadata_list = None

    try:
        provider = options.get('provider')
        model_name = options.get('model')
        replace_ss = options.get('replace_ss', False)
        compute_embeddings = options.get('compute_embeddings', True)
        compute_metadata = options.get('compute_metadata', False)
        compute_quality = options.get('compute_quality', True)
        if compute_embeddings:
            compute_metadata = True

        logger.info(f"Starting batch processing of {total_images} images...")
        logger.info(
            f"compute_embeddings={compute_embeddings}, "
            f"compute_metadata={compute_metadata}, compute_quality={compute_quality}"
        )

        all_uuids = {uuid for _, uuid, _ in image_triplets}

        metadata_results = None
        ratings = None
        if compute_metadata or compute_quality:
            try:
                analysis_service = get_analysis_service()
                _, _datetimes, metadata_results, ratings = analysis_service.analyze_batch(
                    image_triplets, options, None, None,
                    set(),
                    all_uuids if compute_metadata else set(),
                    all_uuids if compute_quality else set(),
                )
            except Exception as e:
                logger.error(
                    f"Analysis batch failed — storing base metadata without LLM results: {e}",
                    exc_info=True,
                )

        for i, (_image_bytes, uuid, filename) in enumerate(image_triplets):
            try:
                extra_metadata = (
                    additional_metadata_list[i]
                    if additional_metadata_list is not None
                    else None
                )
                rating_data = ratings[i] if ratings else None
                metadata_data = metadata_results[i] if metadata_results else None

                main_metadata = _build_base_metadata(uuid, filename, options, extra_metadata=extra_metadata)

                if rating_data and rating_data.success:
                    main_metadata["overall_score"] = rating_data.overall_score
                    main_metadata["composition_score"] = rating_data.composition_score
                    main_metadata["lighting_score"] = rating_data.lighting_score
                    main_metadata["motiv_score"] = rating_data.motiv_score
                    main_metadata["colors_score"] = rating_data.colors_score
                    main_metadata["emotion_score"] = rating_data.emotion_score
                    main_metadata["quality_critique"] = rating_data.critique
                    main_metadata["provider"] = provider
                    main_metadata["model"] = model_name
                elif compute_quality:
                    logger.error(f"Quality rating generation failed for {uuid}.")

                if metadata_data and metadata_data.success:
                    if metadata_data.title:
                        main_metadata['title'] = metadata_data.title
                    if metadata_data.caption:
                        main_metadata['caption'] = metadata_data.caption
                    if metadata_data.alt_text:
                        main_metadata['alt_text'] = metadata_data.alt_text
                    if metadata_data.keywords:
                        main_metadata['keywords'] = json.dumps(metadata_data.keywords)
                        main_metadata['flattened_keywords'] = _flatten_keywords(metadata_data.keywords)
                    if not main_metadata.get("provider"):
                        main_metadata["provider"] = provider
                    if not main_metadata.get("model"):
                        main_metadata["model"] = model_name
                elif compute_metadata:
                    logger.error(f"Metadata generation failed for {uuid}.")

                _store_generation_model_keyword(main_metadata)

                if replace_ss:
                    for key, value in main_metadata.items():
                        if isinstance(value, str):
                            main_metadata[key] = value.replace("ß", "ss")

                document = None
                embedding = None
                if compute_embeddings:
                    document = _build_metadata_embedding_document(main_metadata)
                    if not document:
                        logger.error(f"No metadata text available for embedding {uuid}. Storing without embedding.")
                    else:
                        embedding = server_lifecycle.embed_document(document)
                        if embedding is None:
                            logger.error(
                                f"Metadata embedding generation failed for {uuid}. "
                                "Storing with document but without embedding."
                            )
                        else:
                            main_metadata["metadata_search_text"] = document
                            main_metadata["embedding_source"] = "metadata"
                            main_metadata["embedding_model"] = TEXT_EMBEDDING_MODEL_ID

                main_metadata['has_embedding'] = embedding is not None

                logger.info(
                    f"Processing done for {uuid}: "
                    f"metadata_ok={metadata_data is not None and metadata_data.success if metadata_data else False}, "
                    f"quality_ok={rating_data is not None and rating_data.success if rating_data else False}, "
                    f"embedding={'yes' if embedding is not None else 'no'}, "
                    f"document={'yes' if document else 'no'}"
                )
                postgre_service.add_image(uuid, embedding, main_metadata, document=document)
                success_count += 1

            except Exception as e:
                logger.error(f"FAILED to process image {uuid}: {str(e)}", exc_info=True)
                failure_count += 1

        return success_count, failure_count

    except Exception as e:
        logger.error(f"Error during batch processing task: {str(e)}", exc_info=True)
        return 0, total_images
