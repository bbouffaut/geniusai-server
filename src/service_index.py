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
        base_metadata[key] = value


_NON_SEARCHABLE_METADATA_KEYS = {
    "uuid",
    "provider",
    "model",
    "run_date",
    "ai_model",
    "ai_rundate",
    "photo_date",
    "has_embedding",
    "embedding_model",
    "embedding_source",
    "metadata_search_text",
}


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
    """Build a searchable text document from the metadata stored for a photo."""
    parts = []
    for key in sorted(metadata.keys()):
        if key in _NON_SEARCHABLE_METADATA_KEYS or key.startswith("_"):
            continue

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

    photo_date = _first_non_blank(
        options.get("photo_date") if options else None,
        main_metadata.get("photo_date"),
    )
    if photo_date is not None:
        main_metadata["photo_date"] = photo_date

def process_image_task(
    image_triplets: list[tuple[bytes, str, str]],
    options: dict,
    additional_metadata_list=None,
) -> tuple[int, int]:
    """
    Process a batch of images for indexing.
    
    Args:
        image_triplets: List of (image_bytes, uuid, filename) tuples
        options: Dictionary with all processing options
        
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
        regenerate_metadata = options.get('regenerate_metadata', True)
        compute_embeddings = options.get('compute_embeddings', True)
        compute_metadata = options.get('compute_metadata', False)
        compute_quality = options.get('compute_quality', True)
        if compute_embeddings:
            compute_metadata = True

        logger.info(f"Starting batch processing of {total_images} images...")
        logger.info(f"regenerate_metadata={regenerate_metadata}, compute_embeddings={compute_embeddings}, "
                   f"compute_metadata={compute_metadata}, compute_quality={compute_quality}")
        
        # Check existing records if regenerate_metadata is False
        existing_records = {}
        if not regenerate_metadata:
            logger.info("Checking existing records to determine what needs generation...")
            for _, uuid, _ in image_triplets:
                existing_record = postgre_service.get_image(uuid)
                if existing_record and existing_record['ids']:
                    existing_records[uuid] = existing_record['metadatas'][0] if existing_record['metadatas'] else {}
        
        # Determine what actually needs to be computed for each image
        images_needing_embeddings = []
        images_needing_metadata = []
        images_needing_quality = []
        
        for _, uuid, _ in image_triplets:
            existing = existing_records.get(uuid, {})
            
            # Check if embedding is needed
            needs_embedding = compute_embeddings and (
                regenerate_metadata
                or not existing.get('has_embedding', False)
                or existing.get('embedding_source') != 'metadata'
                or existing.get('embedding_model') != TEXT_EMBEDDING_MODEL_ID
            )
            if needs_embedding:
                images_needing_embeddings.append(uuid)
            
            # Check if metadata is needed
            has_any_metadata = existing.get('title') or existing.get('caption') or existing.get('alt_text') or existing.get('keywords')
            needs_metadata = compute_metadata and (regenerate_metadata or not has_any_metadata)
            if existing and compute_metadata:
                logger.info(f"UUID {uuid}: has_metadata={has_any_metadata}, regenerate={regenerate_metadata}, needs_metadata={needs_metadata}")
                logger.info(f"  Existing fields: title={bool(existing.get('title'))}, caption={bool(existing.get('caption'))}, "
                          f"alt_text={bool(existing.get('alt_text'))}, keywords={bool(existing.get('keywords'))}")
            if needs_metadata:
                images_needing_metadata.append(uuid)
            
            # Check if quality scores are needed
            needs_quality = compute_quality and (regenerate_metadata or not existing.get('overall_score'))
            if needs_quality:
                images_needing_quality.append(uuid)
        
        logger.info(f"Generation needed: {len(images_needing_embeddings)} embeddings, "
                   f"{len(images_needing_metadata)} metadata, {len(images_needing_quality)} quality scores")

        # If absolutely nothing needs to be generated, treat this as a successful no-op
        if len(images_needing_embeddings) == 0 and len(images_needing_metadata) == 0 and len(images_needing_quality) == 0:
            logger.info("No generation required (regenerate_metadata=False and all fields present). Returning success without changes.")
            return len(image_triplets), 0
        
        analysis_service = get_analysis_service()

        # Convert lists to sets for faster lookup in analyze_batch
        _, datetimes, metadata_results, ratings = analysis_service.analyze_batch(
            image_triplets, options, None, None,
            set(), set(images_needing_metadata), set(images_needing_quality)
        )

        for i, (_image_bytes, uuid, filename) in enumerate(image_triplets):
            try:
                embedding = None
                rating_data = ratings[i] if ratings else None
                metadata_data = metadata_results[i] if metadata_results else None
                extra_metadata = (
                    additional_metadata_list[i]
                    if additional_metadata_list is not None
                    else None
                )
                
                existing = existing_records.get(uuid, {})
                
                need_embedding = uuid in images_needing_embeddings
                need_metadata = uuid in images_needing_metadata
                need_quality = uuid in images_needing_quality

                # Validate that required new data was generated if needed
                if need_quality and (not rating_data or not rating_data.success):
                    logger.error(f"Quality rating generation failed for {uuid}. Skipping.")
                    failure_count += 1
                    continue

                if need_metadata and (not metadata_data or not metadata_data.success):
                    logger.error(f"Metadata generation failed for {uuid}. Skipping.")
                    failure_count += 1
                    continue

                # If nothing needed for this UUID (already complete) just count success and move on
                if not need_embedding and not need_metadata and not need_quality and not regenerate_metadata:
                    logger.info(f"UUID {uuid}: already fully indexed; skipping update.")
                    success_count += 1
                    continue

                # Start with existing metadata if not regenerating
                if not regenerate_metadata and existing:
                    main_metadata = existing.copy()
                else:
                    main_metadata = {
                        "provider": provider,
                        "model": model_name,
                    }

                _merge_additional_metadata(main_metadata, extra_metadata)

                # Update only basic fields that should always be current
                main_metadata["filename"] = filename
                main_metadata["uuid"] = uuid

                # Update/add datetime if present
                if datetimes:
                    capture_time = datetimes.get(uuid)
                    if capture_time:
                        main_metadata["capture_time"] = capture_time

                # Update quality scores if newly generated
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

                # Update metadata fields if newly generated
                if metadata_data and metadata_data.success:
                    if metadata_data.title:
                        main_metadata['title'] = metadata_data.title
                    if metadata_data.caption:
                        main_metadata['caption'] = metadata_data.caption
                    if metadata_data.alt_text:
                        main_metadata['alt_text'] = metadata_data.alt_text
                    if metadata_data.keywords:
                        main_metadata['keywords'] = json.dumps(metadata_data.keywords)
                        #logger.debug(f"UUID {uuid}: keywords JSON data: {main_metadata['keywords']}")
                        main_metadata['flattened_keywords'] = _flatten_keywords(metadata_data.keywords)
                    if not main_metadata.get("provider"):
                        main_metadata["provider"] = provider
                    if not main_metadata.get("model"):
                        main_metadata["model"] = model_name

                _store_generation_model_keyword(main_metadata)
                main_metadata['run_date'] = time.now().strftime("%Y-%m-%d %H:%M:%S")
                _ensure_search_fields(main_metadata, options)

                if replace_ss:
                    for key, value in main_metadata.items():
                        if isinstance(value, str):
                            main_metadata[key] = value.replace("ß", "ss")

                document = None
                if need_embedding:
                    document = _build_metadata_embedding_document(main_metadata)
                    if not document:
                        logger.error(f"No metadata text available for embedding {uuid}. Skipping.")
                        failure_count += 1
                        continue

                    embedding = server_lifecycle.embed_document(document)
                    if embedding is None:
                        logger.error(f"Metadata embedding generation failed for {uuid}. Skipping.")
                        failure_count += 1
                        continue

                    main_metadata["metadata_search_text"] = document
                    main_metadata["embedding_source"] = "metadata"
                    main_metadata["embedding_model"] = TEXT_EMBEDDING_MODEL_ID

                # Update embedding status
                if embedding is not None:
                    main_metadata['has_embedding'] = True
                elif not regenerate_metadata and existing:
                    # Keep existing embedding status
                    main_metadata['has_embedding'] = existing.get('has_embedding', False)
                else:
                    main_metadata['has_embedding'] = False

                # Determine if we need to update the embedding
                # Only update embedding if we generated a new one
                update_embedding = embedding if embedding is not None else None
                
                if existing and not regenerate_metadata:
                    logger.info(f"UUID {uuid} already exists. Updating (embedding: {update_embedding is not None}).")
                    postgre_service.update_image(uuid, main_metadata, embedding=update_embedding, document=document)
                elif regenerate_metadata:
                    logger.info(f"UUID {uuid} set to regenerate. Updating (embedding: {update_embedding is not None}).")
                    if postgre_service.get_image(uuid) is not None:
                        postgre_service.update_image(uuid, main_metadata, embedding=update_embedding, document=document)
                    else:
                        postgre_service.add_image(uuid, embedding, main_metadata, document=document)
                else:
                    # New record
                    if embedding is not None:
                        logger.info(f"UUID {uuid} is new. Indexing with metadata embeddings.")
                    else:
                        logger.info(f"UUID {uuid} is new. Indexing metadata-only entry (no embedding).")
                    postgre_service.add_image(uuid, embedding, main_metadata, document=document)
                
                success_count += 1

            except Exception as e:
                logger.error(f"Error processing image {uuid}: {str(e)}", exc_info=True)
                failure_count += 1

        return success_count, failure_count
    except Exception as e:
        logger.error(f"Error during batch processing task: {str(e)}", exc_info=True)
        return 0, total_images
