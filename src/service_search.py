import json
import numpy as np
import re
import unicodedata
from datetime import date, timedelta

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

MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

MONTH_PATTERN = "|".join(sorted(MONTH_NAMES.keys(), key=len, reverse=True))
MONTH_YEAR_RE = re.compile(rf"\b({MONTH_PATTERN})\.?\s+([12][0-9]{{3}})\b", re.IGNORECASE)
YEAR_MONTH_RE = re.compile(rf"\b([12][0-9]{{3}})\s+({MONTH_PATTERN})\.?\b", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"\b([12][0-9]{3})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])\b")
ISO_MONTH_RE = re.compile(r"\b([12][0-9]{3})[-/](0?[1-9]|1[0-2])\b")
APERTURE_RE = re.compile(
    r"\b(?:f\s*/?\s*|f-?number\s*|f-?stop\s*|aperture\s*)([0-9]+(?:\.[0-9]+)?)\b",
    re.IGNORECASE,
)
QUERY_EDGE_FILLER_WORDS = {"with", "at", "on", "in", "during", "from", "for", "of"}


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


def _span_overlaps(candidate, spans):
    start, end = candidate
    return any(start < existing_end and end > existing_start for existing_start, existing_end in spans)


def _replace_spans_with_spaces(text, spans):
    if not spans:
        return text

    characters = list(text)
    for start, end in spans:
        for index in range(start, end):
            characters[index] = " "

    return "".join(characters)


def _cleanup_semantic_query(text):
    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    while tokens and tokens[0].casefold() in QUERY_EDGE_FILLER_WORDS:
        tokens.pop(0)
    while tokens and tokens[-1].casefold() in QUERY_EDGE_FILLER_WORDS:
        tokens.pop()
    return " ".join(tokens)


def _next_month(year, month):
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)


def _capture_time_range_filters(start_date, end_date, source):
    return [
        {
            "field": "capture_time",
            "op": "gte",
            "value": start_date.isoformat(),
            "source": source,
        },
        {
            "field": "capture_time",
            "op": "lt",
            "value": end_date.isoformat(),
            "source": source,
        },
    ]


def _parse_iso_date_match(match):
    try:
        start_date = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None
    return start_date, start_date + timedelta(days=1)


def _parse_iso_month_match(match):
    try:
        year = int(match.group(1))
        month = int(match.group(2))
        start_date = date(year, month, 1)
    except ValueError:
        return None
    return start_date, _next_month(year, month)


def _parse_month_year_match(month_value, year_value):
    month = MONTH_NAMES.get(str(month_value).casefold().rstrip("."))
    if month is None:
        return None

    try:
        year = int(year_value)
        start_date = date(year, month, 1)
    except ValueError:
        return None

    return start_date, _next_month(year, month)


def _add_date_filter_from_match(filters, spans, match, range_value, source):
    if range_value is None or _span_overlaps(match.span(), spans):
        return

    start_date, end_date = range_value
    filters.extend(_capture_time_range_filters(start_date, end_date, source))
    spans.append(match.span())


def _parse_aperture_value(value):
    try:
        aperture = float(value)
    except (TypeError, ValueError):
        return None

    if aperture <= 0:
        return None
    return aperture


def _aperture_filter(value, source):
    aperture = _parse_aperture_value(value)
    if aperture is None:
        return None

    return {
        "field": "aperture",
        "op": "number_eq",
        "value": aperture,
        "tolerance": 0.05,
        "source": source,
    }


def _parse_search_query(term):
    filters = []
    spans = []
    text = str(term or "")

    for match in ISO_DATE_RE.finditer(text):
        _add_date_filter_from_match(
            filters,
            spans,
            match,
            _parse_iso_date_match(match),
            "query_date",
        )

    for match in MONTH_YEAR_RE.finditer(text):
        _add_date_filter_from_match(
            filters,
            spans,
            match,
            _parse_month_year_match(match.group(1), match.group(2)),
            "query_month",
        )

    for match in YEAR_MONTH_RE.finditer(text):
        _add_date_filter_from_match(
            filters,
            spans,
            match,
            _parse_month_year_match(match.group(2), match.group(1)),
            "query_month",
        )

    for match in ISO_MONTH_RE.finditer(text):
        _add_date_filter_from_match(
            filters,
            spans,
            match,
            _parse_iso_month_match(match),
            "query_month",
        )

    for match in APERTURE_RE.finditer(text):
        if _span_overlaps(match.span(), spans):
            continue
        aperture_filter = _aperture_filter(match.group(1), "query_aperture")
        if aperture_filter is None:
            continue
        filters.append(aperture_filter)
        spans.append(match.span())

    semantic_term = _cleanup_semantic_query(_replace_spans_with_spaces(text, spans))
    return {
        "semantic_term": semantic_term,
        "metadata_filters": filters,
    }


def _case_insensitive_get(mapping, *keys):
    if not isinstance(mapping, dict):
        return None

    lowered = {str(key).casefold(): value for key, value in mapping.items()}
    for key in keys:
        value = lowered.get(str(key).casefold())
        if value is not None:
            return value
    return None


def _date_filter_range_from_text(value):
    text = str(value).strip()
    if not text:
        return None

    match = ISO_DATE_RE.fullmatch(text)
    if match:
        return _parse_iso_date_match(match)

    match = ISO_MONTH_RE.fullmatch(text)
    if match:
        return _parse_iso_month_match(match)

    match = MONTH_YEAR_RE.fullmatch(text)
    if match:
        return _parse_month_year_match(match.group(1), match.group(2))

    match = YEAR_MONTH_RE.fullmatch(text)
    if match:
        return _parse_month_year_match(match.group(2), match.group(1))

    return None


def _explicit_capture_time_filters(value):
    if value in (None, "", [], {}):
        return []

    if isinstance(value, dict):
        filters = []
        for op in ("gte", "gt", "lte", "lt"):
            op_value = _case_insensitive_get(value, op)
            if op_value is not None:
                filters.append({
                    "field": "capture_time",
                    "op": op,
                    "value": str(op_value).strip(),
                    "source": "explicit_filter",
                })

        eq_value = _case_insensitive_get(value, "eq", "equals")
        if eq_value is not None:
            range_value = _date_filter_range_from_text(eq_value)
            if range_value:
                return filters + _capture_time_range_filters(*range_value, source="explicit_filter")
            filters.append({
                "field": "capture_time",
                "op": "eq",
                "value": str(eq_value).strip(),
                "source": "explicit_filter",
            })

        return filters

    range_value = _date_filter_range_from_text(value)
    if range_value:
        return _capture_time_range_filters(*range_value, source="explicit_filter")

    return [{
        "field": "capture_time",
        "op": "eq",
        "value": str(value).strip(),
        "source": "explicit_filter",
    }]


def _explicit_numeric_filter(field, value):
    if value in (None, "", [], {}):
        return []

    if isinstance(value, dict):
        filters = []
        for op in ("gte", "gt", "lte", "lt"):
            op_value = _case_insensitive_get(value, op)
            if op_value is not None:
                numeric_value = _parse_aperture_value(op_value)
                if numeric_value is not None:
                    filters.append({
                        "field": field,
                        "op": op,
                        "value": numeric_value,
                        "source": "explicit_filter",
                    })

        eq_value = _case_insensitive_get(value, "eq", "equals")
        if eq_value is not None:
            numeric_value = _parse_aperture_value(eq_value)
            if numeric_value is not None:
                filters.append({
                    "field": field,
                    "op": "number_eq",
                    "value": numeric_value,
                    "tolerance": 0.05,
                    "source": "explicit_filter",
                })
        return filters

    numeric_value = _parse_aperture_value(value)
    if numeric_value is None:
        return []

    return [{
        "field": field,
        "op": "number_eq",
        "value": numeric_value,
        "tolerance": 0.05,
        "source": "explicit_filter",
    }]


def _normalize_explicit_search_filters(search_filters):
    if not isinstance(search_filters, dict):
        return []

    filters = []
    payloads = [search_filters]
    nested_metadata = search_filters.get("metadata")
    if isinstance(nested_metadata, dict):
        payloads.append(nested_metadata)

    for payload in payloads:
        capture_time_value = _case_insensitive_get(payload, "capture_time", "date", "datetime")
        filters.extend(_explicit_capture_time_filters(capture_time_value))

        aperture_value = _case_insensitive_get(
            payload,
            "aperture",
            "aperture_f_number",
            "f_number",
            "fnumber",
            "f_stop",
            "fstop",
        )
        filters.extend(_explicit_numeric_filter("aperture", aperture_value))

        exif_value = _case_insensitive_get(payload, "exif")
        if isinstance(exif_value, dict):
            exif_aperture = _case_insensitive_get(
                exif_value,
                "aperture",
                "aperture_f_number",
                "f_number",
                "fnumber",
                "f_stop",
                "fstop",
            )
            filters.extend(_explicit_numeric_filter("aperture", exif_aperture))

    return filters


def _build_search_where_clause(uuids_to_search=None, metadata_filters=None):
    where_clause = {}
    if uuids_to_search:
        where_clause["uuid"] = {"$in": list(uuids_to_search)}
    if metadata_filters:
        where_clause["metadata_filters"] = metadata_filters
    return where_clause or None


def _get_filtered_metadatas(ids=None, where_clause=None):
    return postgre_service.get_image_metadatas(ids=ids, where_clause=where_clause)


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


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _format_quality_scores(metadata):
    score_parts = []
    for label, key in QUALITY_SCORE_FIELDS:
        score = _display_score(metadata.get(key), digits=2)
        if score != "-":
            score_parts.append(f"{label}={score}")

    return ", ".join(score_parts) if score_parts else "quality=n/a"


def _extract_quality_fields(metadata):
    quality_fields = {}
    for _, key in QUALITY_SCORE_FIELDS:
        if key in metadata and metadata.get(key) is not None:
            quality_fields[key] = metadata.get(key)
    if metadata.get("quality_critique") is not None:
        quality_fields["quality_critique"] = metadata.get("quality_critique")
    return quality_fields


def _build_search_result(uuid, metadata, distance, pertinence_score, match_type, metadata_match=False):
    metadata = dict(metadata or {})
    result = {
        "uuid": uuid,
        "filename": metadata.get("filename"),
        "distance": float(round(distance, 4)) if distance is not None else None,
        "pertinence_score": float(round(pertinence_score, 4)),
        "match_type": match_type,
        "metadata": metadata,
        "quality": _extract_quality_fields(metadata),
        "ai_model": _first_non_none(metadata.get("ai_model"), metadata.get("model")),
        "ai_rundate": _first_non_none(metadata.get("ai_rundate"), metadata.get("run_date")),
        "capture_time": metadata.get("capture_time"),
    }

    if metadata_match:
        result["metadata_match"] = True

    return result


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

        transformed_results.append(
            _build_search_result(
                ids[i],
                metadata,
                distance,
                pertinence_score,
                "semantic",
            )
        )

    transformed_results.sort(
        key=lambda x: (
            -x['pertinence_score'],
            x['distance'] if x['distance'] is not None else float('inf'),
        )
    )
    return transformed_results


def search_images(
    term,
    quality_sort,
    uuids_to_search,
    min_pertinence_score=DEFAULT_MIN_PERTINENCE_SCORE,
    search_filters=None,
):
    parsed_query = _parse_search_query(term)
    metadata_filters = [
        *parsed_query["metadata_filters"],
        *_normalize_explicit_search_filters(search_filters),
    ]
    semantic_term = parsed_query["semantic_term"]
    if not semantic_term and not metadata_filters:
        semantic_term = term

    where_clause = _build_search_where_clause(uuids_to_search, metadata_filters)

    logger.info(
        f"Searching for '{term}' (quality_sort: {quality_sort}, scoped: {uuids_to_search is not None}, "
        f"min_pertinence_score: {min_pertinence_score}, semantic_term: '{semantic_term}', "
        f"metadata_filters: {metadata_filters})"
    )

    # 1. Semantic search over metadata text embeddings
    query_embedding = server_lifecycle.embed_query(semantic_term) if semantic_term else None
    if query_embedding is not None:
        db_results = postgre_service.query_images(
            query_embedding=query_embedding,
            n_results=300,
            where_clause=where_clause,
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
        logger.info("Text embedding model not loaded or semantic term is empty, skipping semantic metadata search.")
        sorted_semantic_results = []
        semantic_uuids = set()

    # 2. Metadata Search (in-memory)
    logger.info("Performing exact metadata search in-memory. This may be slow for large databases without a UUID filter.")

    if uuids_to_search:
        target_uuids = list(uuids_to_search)
        all_metadata_raw = _get_filtered_metadatas(ids=target_uuids, where_clause=where_clause)
    else:
        all_metadata_raw = _get_filtered_metadatas(where_clause=where_clause)

    metadata_by_id = _metadata_by_uuid(all_metadata_raw)
    metadata_uuids = set()
    metadata_search_term = semantic_term.strip()
    normalized_term = _normalize_search_text(metadata_search_term) if metadata_search_term else ""

    for i, uuid in enumerate(all_metadata_raw['ids']):
        metadata = all_metadata_raw['metadatas'][i]
        if not metadata:
            continue

        if not normalized_term and metadata_filters:
            metadata_uuids.add(uuid)
            continue

        metadata_text = _metadata_value_to_search_text(metadata)
        if normalized_term and normalized_term in _normalize_search_text(metadata_text):
            metadata_uuids.add(uuid)

    # 3. Combine results
    for result in sorted_semantic_results:
        if result['uuid'] in metadata_uuids:
            result['metadata_match'] = True
            result['match_type'] = "semantic+metadata"

    metadata_only_uuids = metadata_uuids - semantic_uuids
    metadata_only_results = [
        _build_search_result(
            uuid,
            metadata_by_id.get(uuid, {}),
            None,
            1.0,
            "metadata",
            metadata_match=True,
        )
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
