import json
import numpy as np
import re
import unicodedata
from datetime import date, timedelta

#import service_chroma as chroma_service
import service_postgre as postgre_service
from config import DEBUG_LEVEL, DEFAULT_MIN_PERTINENCE_SCORE, logger
import server_lifecycle as server_lifecycle

def _accent_fold(text):
    """Strip combining diacritics (accents) without changing character positions.

    For European Latin text each accented character decomposes to one base character
    plus one or more combining marks; removing the marks leaves a string of the same
    length, so span offsets from the folded text map 1-to-1 back to the original.
    This lets us match month names regardless of how the user typed them
    (e.g. "Février", "fevrier" and "FEVRIER" all reduce to "fevrier").
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


QUALITY_SCORE_FIELDS = [
    ("overall", "overall_score"),
    ("composition", "composition_score"),
    ("lighting", "lighting_score"),
    ("motiv", "motiv_score"),
    ("colors", "colors_score"),
    ("emotion", "emotion_score"),
]

MONTH_NAMES = {
    # English
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
    # French (accent-folded; accents are stripped before lookup — see _accent_fold)
    "janv": 1, "janvier": 1,
    "fevr": 2, "fevrier": 2,
    "mars": 3,
    "avr": 4, "avril": 4,
    "mai": 5,
    "juin": 6,
    "juil": 7, "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    # German (accent-folded)
    "januar": 1, "februar": 2, "marz": 3, "juni": 6,
    "juli": 7, "oktober": 10, "dezember": 12,
    # Spanish / Portuguese
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
    "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
    "octubre": 10, "noviembre": 11, "diciembre": 12,
    # Italian
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5,
    "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9,
    "ottobre": 10, "novembre": 11, "dicembre": 12,
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
ISO_RE = re.compile(r"\biso\s*[-:]?\s*([0-9]+)\b", re.IGNORECASE)
FOCAL_LENGTH_RE = re.compile(
    r"(?<![-\d])(?<![/])([0-9]+(?:\.[0-9]+)?)\s*mm\b",
    re.IGNORECASE,
)
SHUTTER_FRACTION_RE = re.compile(
    r"\b1\s*/\s*([0-9]{2,5})\b",
    re.IGNORECASE,
)
SHUTTER_SECONDS_RE = re.compile(
    r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:seconds?|sec)\b",
    re.IGNORECASE,
)
EXPOSURE_BIAS_RE = re.compile(
    r"([+-][0-9]+(?:\.[0-9]+)?)\s*(?:ev|stops?)?\b|\b([0-9]+(?:\.[0-9]+)?)\s*(?:ev|stops?)\b",
    re.IGNORECASE,
)

_CAMERA_MAKE_NAMES = sorted(
    [
        "canon", "nikon", "sony", "fujifilm", "fuji",
        "olympus", "panasonic", "leica", "hasselblad",
        "ricoh", "pentax", "sigma", "apple", "samsung", "dji", "gopro",
    ],
    key=len,
    reverse=True,
)
CAMERA_MAKE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in _CAMERA_MAKE_NAMES) + r")\b",
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


def _iso_filter(value, source):
    try:
        iso = int(float(value))
    except (TypeError, ValueError):
        return None
    if iso <= 0:
        return None
    return {"field": "iso", "op": "number_eq", "value": float(iso), "tolerance": 0.0, "source": source}


def _focal_length_filter(value, source):
    try:
        mm = float(value)
    except (TypeError, ValueError):
        return None
    if mm <= 0:
        return None
    return {"field": "focal_length_mm", "op": "number_eq", "value": mm, "tolerance": 1.0, "source": source}


def _shutter_speed_filter(seconds, source):
    try:
        speed = float(seconds)
    except (TypeError, ValueError):
        return None
    if speed <= 0:
        return None
    tolerance = max(speed * 0.15, 1e-6)
    return {"field": "shutter_speed", "op": "number_eq", "value": speed, "tolerance": tolerance, "source": source}


def _exposure_bias_filter(value, source):
    try:
        ev = float(value)
    except (TypeError, ValueError):
        return None
    return {"field": "exposure_bias", "op": "number_eq", "value": ev, "tolerance": 0.1, "source": source}


def _camera_make_filter(value, source):
    text = str(value).strip()
    if not text:
        return None
    return {"field": "camera_make", "op": "ilike", "value": text, "source": source}


def _add_filter_from_match(filters, spans, match, filter_item):
    if filter_item is None or _span_overlaps(match.span(), spans):
        return
    filters.append(filter_item)
    spans.append(match.span())


def _parse_search_query(term):
    filters = []
    spans = []
    text = str(term or "")

    # Fold accents so that e.g. "Février", "fevrier" and "FEVRIER" all match the
    # same dictionary entries.  For European Latin text, NFKD folding is a 1-to-1
    # character mapping, so span offsets from `folded` apply unchanged to `text`.
    folded = _accent_fold(text)

    for match in ISO_DATE_RE.finditer(folded):
        _add_date_filter_from_match(filters, spans, match, _parse_iso_date_match(match), "query_date")

    for match in MONTH_YEAR_RE.finditer(folded):
        _add_date_filter_from_match(
            filters, spans, match,
            _parse_month_year_match(match.group(1), match.group(2)), "query_month",
        )

    for match in YEAR_MONTH_RE.finditer(folded):
        _add_date_filter_from_match(
            filters, spans, match,
            _parse_month_year_match(match.group(2), match.group(1)), "query_month",
        )

    for match in ISO_MONTH_RE.finditer(folded):
        _add_date_filter_from_match(filters, spans, match, _parse_iso_month_match(match), "query_month")

    for match in APERTURE_RE.finditer(folded):
        _add_filter_from_match(filters, spans, match, _aperture_filter(match.group(1), "query_aperture"))

    for match in ISO_RE.finditer(folded):
        _add_filter_from_match(filters, spans, match, _iso_filter(match.group(1), "query_iso"))

    for match in FOCAL_LENGTH_RE.finditer(folded):
        _add_filter_from_match(filters, spans, match, _focal_length_filter(match.group(1), "query_focal_length"))

    for match in SHUTTER_FRACTION_RE.finditer(folded):
        denominator = match.group(1)
        try:
            speed = 1.0 / float(denominator)
        except (ValueError, ZeroDivisionError):
            continue
        _add_filter_from_match(filters, spans, match, _shutter_speed_filter(speed, "query_shutter"))

    for match in SHUTTER_SECONDS_RE.finditer(folded):
        _add_filter_from_match(filters, spans, match, _shutter_speed_filter(match.group(1), "query_shutter"))

    for match in EXPOSURE_BIAS_RE.finditer(folded):
        value = match.group(1) or match.group(2)
        _add_filter_from_match(filters, spans, match, _exposure_bias_filter(value, "query_exposure_bias"))

    for match in CAMERA_MAKE_RE.finditer(folded):
        _add_filter_from_match(filters, spans, match, _camera_make_filter(match.group(0), "query_camera_make"))

    # Build the semantic term from the ORIGINAL text so accents are preserved
    # for the embedding model, then strip the filter spans that were consumed.
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


def _explicit_text_ilike_filter(field, value):
    if value in (None, "", [], {}):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [{"field": field, "op": "ilike", "value": text, "source": "explicit_filter"}]


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
            payload, "aperture", "aperture_f_number", "f_number", "fnumber", "f_stop", "fstop",
        )
        filters.extend(_explicit_numeric_filter("aperture", aperture_value))

        iso_value = _case_insensitive_get(payload, "iso", "iso_speed_rating", "iso_speed_ratings")
        filters.extend(_explicit_numeric_filter("iso", iso_value))

        focal_value = _case_insensitive_get(payload, "focal_length_mm", "focal_length", "focallength")
        filters.extend(_explicit_numeric_filter("focal_length_mm", focal_value))

        focal35_value = _case_insensitive_get(payload, "focal_length_35mm", "focal_length35mm")
        filters.extend(_explicit_numeric_filter("focal_length_35mm", focal35_value))

        shutter_value = _case_insensitive_get(payload, "shutter_speed", "shutter", "shutterspeed")
        filters.extend(_explicit_numeric_filter("shutter_speed", shutter_value))

        bias_value = _case_insensitive_get(payload, "exposure_bias", "exposurebias")
        filters.extend(_explicit_numeric_filter("exposure_bias", bias_value))

        make_value = _case_insensitive_get(payload, "camera_make", "make")
        filters.extend(_explicit_text_ilike_filter("camera_make", make_value))

        model_value = _case_insensitive_get(payload, "camera_model", "camera")
        filters.extend(_explicit_text_ilike_filter("camera_model", model_value))

        lens_value = _case_insensitive_get(payload, "lens", "lens_model", "lensmodel")
        filters.extend(_explicit_text_ilike_filter("lens", lens_value))

        exif_value = _case_insensitive_get(payload, "exif")
        if isinstance(exif_value, dict):
            exif_aperture = _case_insensitive_get(
                exif_value, "aperture", "aperture_f_number", "f_number", "fnumber", "f_stop", "fstop",
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


def _log_query_plan(term, semantic_term, query_filters, explicit_filters,
                    uuids_to_search, min_pertinence_score, limit, mode="query-parsing"):
    if DEBUG_LEVEL < 1:
        return

    lines = [f"[search] Query plan for '{term}'  (mode: {mode})"]

    if semantic_term and semantic_term != term:
        lines.append(f"  semantic term : '{semantic_term}' (extracted from query)")
    elif semantic_term:
        lines.append(f"  semantic term : '{semantic_term}'")
    else:
        lines.append("  semantic term : (none — skipped or fully consumed by filters)")

    if mode == "explicit-filters":
        if explicit_filters:
            for f in explicit_filters:
                field = f.get("field", "?")
                op = f.get("op", "?")
                value = f.get("value")
                tol = f.get("tolerance")
                tol_str = f" ±{tol}" if tol else ""
                lines.append(f"  filter        : {field} {op} {value}{tol_str}")
        else:
            lines.append("  filter        : (none)")
        lines.append("  query parsing : disabled (explicit filters take full control)")
    else:
        if query_filters:
            for f in query_filters:
                field = f.get("field", "?")
                op = f.get("op", "?")
                value = f.get("value")
                tol = f.get("tolerance")
                tol_str = f" ±{tol}" if tol else ""
                lines.append(f"  filter (query): {field} {op} {value}{tol_str}  [{f.get('source', '')}]")
        else:
            lines.append("  filter (query): (none)")

    if uuids_to_search is not None:
        lines.append(f"  uuid scope   : {len(uuids_to_search)} UUIDs")
    else:
        lines.append("  uuid scope   : all photos")

    lines.append(f"  min score    : {min_pertinence_score}")
    lines.append(f"  limit        : {limit}")
    logger.info("\n".join(lines))


def _log_retrieved_photos(term, final_results, metadata_by_id):
    if not final_results:
        if DEBUG_LEVEL >= 1:
            logger.info(f"Search results for '{term}': no photos retrieved")
        return

    if DEBUG_LEVEL < 2:
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
    """Transforms vector DB results and sorts them based on quality or distance.

    When both prose and keyword embeddings are present for a photo, the
    pertinence score is the *maximum* cosine similarity across the two
    (late fusion).  Photos that only carry one embedding type degrade
    gracefully to the single-embedding score.
    """
    if not results:
        return []

    ids = _first_result_group(results, 'ids')
    if len(ids) == 0:
        return []

    distances = _first_result_group(results, 'distances')
    metadatas = _first_result_group(results, 'metadatas')
    embeddings = _first_result_group(results, 'embeddings')
    embeddings_kw = _first_result_group(results, 'embeddings_kw')

    transformed_results = []
    for i in range(len(ids)):
        # Skip metadata-only entries (with dummy embeddings) from semantic search
        metadata = metadatas[i] if i < len(metadatas) else {}
        if metadata and not metadata.get('has_embedding', True):
            continue

        distance = distances[i] if i < len(distances) else None
        embedding = embeddings[i] if i < len(embeddings) else None
        embedding_kw = embeddings_kw[i] if i < len(embeddings_kw) else None

        # Score from prose embedding (falls back to SQL distance when embedding is None).
        prose_score = _score_from_embedding(query_embedding, embedding, distance)

        # Score from keyword embedding; take the best of the two (late fusion).
        if embedding_kw is not None:
            kw_score = _score_from_embedding(query_embedding, embedding_kw, None)
            pertinence_score = max(prose_score, kw_score)
        else:
            pertinence_score = prose_score

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
    limit=300,
):
    # --- Stage 1: Query parsing ---
    # When explicit filters are provided in the request body, the term is used
    # purely for semantic vector search — no filter extraction is attempted.
    # When there are no explicit filters, the term is parsed to extract structured
    # column filters (date, aperture, ISO, …) and the remainder becomes the
    # semantic term.
    if search_filters:
        query_filters = []
        explicit_filters = _normalize_explicit_search_filters(search_filters)
        metadata_filters = explicit_filters
        semantic_term = (term or "").strip()
    else:
        parsed_query = _parse_search_query(term)
        query_filters = parsed_query["metadata_filters"]
        explicit_filters = []
        metadata_filters = query_filters
        semantic_term = parsed_query["semantic_term"]
        if not semantic_term and not metadata_filters:
            semantic_term = term

    _log_query_plan(
        term, semantic_term, query_filters, explicit_filters,
        uuids_to_search, min_pertinence_score, limit,
        mode="explicit-filters" if search_filters else "query-parsing",
    )

    where_clause = _build_search_where_clause(uuids_to_search, metadata_filters)

    # Short queries (1-2 words) have much less discriminating power in embedding
    # space — raise the threshold to avoid flooding results with loosely related photos.
    word_count = len(semantic_term.split()) if semantic_term else 0
    is_short_query = word_count <= 2
    effective_threshold = max(min_pertinence_score, 0.60) if is_short_query else min_pertinence_score
    use_word_boundary = is_short_query

    # --- Stage 2: Semantic vector search ---
    query_embedding = server_lifecycle.embed_query(semantic_term) if semantic_term else None
    if query_embedding is not None:
        if DEBUG_LEVEL >= 1:
            logger.info(f"[search] Stage 2: semantic vector search for '{semantic_term}' (top {limit}, threshold={effective_threshold}{' [short-query]' if is_short_query else ''})")
        db_results = postgre_service.query_images(
            query_embedding=query_embedding,
            n_results=limit,
            where_clause=where_clause,
            include_embeddings=True,
        )
        sorted_semantic_results = _transform_and_sort_results(
            db_results, quality_sort, query_embedding, effective_threshold,
        )
        semantic_uuids = {res['uuid'] for res in sorted_semantic_results}
        if DEBUG_LEVEL >= 1:
            logger.info(f"[search] Stage 2 done: {len(sorted_semantic_results)} results above threshold")
    else:
        if DEBUG_LEVEL >= 1:
            reason = "embedding model not loaded" if not server_lifecycle.embed_query else "no semantic term after filter extraction"
            logger.info(f"[search] Stage 2: skipped ({reason})")
        sorted_semantic_results = []
        semantic_uuids = set()

    # --- Stage 3: SQL document text search ---
    doc_term = semantic_term or None
    if DEBUG_LEVEL >= 1:
        filter_desc = f"{len(metadata_filters)} column filter(s)" if metadata_filters else "no column filters"
        mode_desc = "word-boundary regex" if use_word_boundary else "ILIKE substring"
        if doc_term:
            logger.info(f"[search] Stage 3: document {mode_desc} '{doc_term}' + {filter_desc}")
        else:
            logger.info(f"[search] Stage 3: column-filters-only query ({filter_desc})")

    text_match_rows = postgre_service.search_document_text(
        term=doc_term,
        where_clause=where_clause,
        limit=limit,
        use_word_boundary=use_word_boundary,
    )
    text_match_uuids = set(text_match_rows.keys())

    # For short queries also search the dedicated keyword field — more precise than
    # searching the full document (avoids prose matches in captions/critiques).
    if doc_term and is_short_query:
        keyword_match_rows = postgre_service.search_keyword_field(
            term=doc_term,
            where_clause=where_clause,
            limit=limit,
        )
        keyword_match_uuids = set(keyword_match_rows.keys())
        # Merge keyword matches into text matches (keyword hits are authoritative)
        text_match_rows.update(keyword_match_rows)
        text_match_uuids |= keyword_match_uuids
        if DEBUG_LEVEL >= 1:
            logger.info(f"[search] Stage 3b: keyword-field search found {len(keyword_match_uuids)} additional matches")
    else:
        keyword_match_uuids = set()

    if DEBUG_LEVEL >= 1:
        logger.info(f"[search] Stage 3 done: {len(text_match_uuids)} document matches")

    # --- Stage 4: Merge & rank ---
    overlap = len(semantic_uuids & text_match_uuids)
    if DEBUG_LEVEL >= 1:
        logger.info(
            f"[search] Stage 4: merging — {len(semantic_uuids)} semantic, "
            f"{len(text_match_uuids)} text, {overlap} overlap"
        )

    for result in sorted_semantic_results:
        if result['uuid'] in text_match_uuids:
            result['metadata_match'] = True
            result['match_type'] = "semantic+metadata"
            # Boost photos that also matched by keyword so they always rank above
            # semantic-only results regardless of their raw similarity score.
            result['pertinence_score'] = max(result['pertinence_score'], 0.95)

    text_only_uuids = text_match_uuids - semantic_uuids
    text_only_results = [
        _build_search_result(
            uuid,
            text_match_rows[uuid],
            None,
            1.0,
            "metadata",
            metadata_match=True,
        )
        for uuid in text_only_uuids
    ]

    final_results = sorted_semantic_results + text_only_results
    final_results.sort(
        key=lambda x: (
            # Keyword-matched results first (pertinence_score >= 0.95), then semantic-only
            0 if x.get('metadata_match') else 1,
            -x['pertinence_score'],
            x['distance'] if x['distance'] is not None else float('inf'),
        )
    )

    metadata_by_id = dict(text_match_rows)
    for result in sorted_semantic_results:
        if result['uuid'] not in metadata_by_id:
            metadata_by_id[result['uuid']] = result.get('metadata', {})

    if DEBUG_LEVEL >= 1:
        logger.info(
            f"[search] Done: {len(final_results)} result(s) for '{term}' "
            f"({len(sorted_semantic_results)} semantic, {len(text_only_results)} text-only, "
            f"{overlap} both)"
        )
    _log_retrieved_photos(term, final_results, metadata_by_id)

    return final_results


def group_similar_images(uuids, phash_threshold, time_delta):
    """Groups a list of images by similarity and sorts them by quality."""
    if DEBUG_LEVEL >= 1:
        logger.info(f"Grouping {len(uuids)} UUIDs with phash_threshold='{phash_threshold}' and time_delta='{time_delta}s'.")

    try:
        grouped_results = postgre_service.group_and_sort_images(uuids, phash_threshold, time_delta)
        return grouped_results
    except Exception as e:
        logger.error(f"Error during similarity grouping: {str(e)}")
        raise e


def project_search_result(result, return_metadata=False):
    """Project a raw search_images() result to the canonical search-facing shape.

    Shared by the REST /search route and the MCP search_photos tool so both
    interfaces return identical fields. When return_metadata is True the stored
    metadata payload is included under the "metadata" key.
    """
    metadata = result.get('metadata')
    if metadata is None:
        metadata = {}

    return {
        "ai_model": result.get("ai_model"),
        "ai_rundate": result.get("ai_rundate"),
        "distance": result.get("distance"),
        "filename": result.get("filename"),
        "match_type": result.get("match_type"),
        "pertinence_score": result.get("pertinence_score"),
        "capture_time": result.get("capture_time") or (
            metadata.get("capture_time") if isinstance(metadata, dict) else None
        ),
        "uuid": result.get("uuid") or (
            metadata.get("uuid") if isinstance(metadata, dict) else None
        ),
        **({"metadata": metadata} if return_metadata else {}),
    }
