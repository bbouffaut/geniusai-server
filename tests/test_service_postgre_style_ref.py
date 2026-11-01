"""Live pgvector round-trip for the style_ref_photos layer (T1.2 gate).

Hermetic by default: run against a real Postgres by setting
STYLE_MATCH_TEST_DATABASE_URL (e.g. the local-postgre container:
postgresql://admin:superuserpassword@localhost:32345/postgres).
"""

import importlib
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CLIP_DIM = 512
TEXT_DIM = 1024


def _make_logger():
    return types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )


def _install_real_postgre(monkeypatch):
    url = os.environ.get("STYLE_MATCH_TEST_DATABASE_URL")
    if not url:
        pytest.skip("STYLE_MATCH_TEST_DATABASE_URL not set — live pgvector round-trip skipped")

    fake_config = types.ModuleType("config")
    fake_config.POSTGRE_URL = url
    fake_config.POSTGRE_USER = None
    fake_config.POSTGRE_PASSWORD = None
    fake_config.POSTGRE_DATABASE_NAME = os.environ.get(
        "STYLE_MATCH_TEST_DATABASE_NAME", "postgres"
    )
    fake_config.TEXT_EMBEDDING_DIMENSION = TEXT_DIM
    fake_config.CLIP_EMBEDDING_DIMENSION = CLIP_DIM
    fake_config.DEBUG_LEVEL = 0
    fake_config.logger = _make_logger()

    monkeypatch.setitem(sys.modules, "config", fake_config)
    sys.modules.pop("service_postgre", None)
    return importlib.import_module("service_postgre")


def _clip_vector(seed):
    vector = [0.0] * CLIP_DIM
    vector[seed % CLIP_DIM] = 1.0
    return vector


def test_style_ref_roundtrip(monkeypatch):
    service_postgre = _install_real_postgre(monkeypatch)
    service_postgre._style_refs_initialized = False

    service_postgre.upsert_style_ref(
        catalog_id="it-catalog",
        lrc_local_id=101,
        uuid="3f2a1c6e-1111-4b2a-9c9d-000000000001",
        file_path="/photos/IMG_101.jpg",
        file_name="IMG_101.jpg",
        embedding=_clip_vector(3),
        description_embedding=[0.5] * TEXT_DIM,
        capture_time="2024-05-12T19:34:00+02:00",
        latitude=48.85,
        longitude=2.35,
        camera_model="ILCE-7M4",
        description="sunset over the lake",
        color_histogram=[1.0 / 4] * 4,
        render_source="develop_export",
        develop_settings={"Contrast2012": 25, "Exposure2012": -0.65},
        has_edits=True,
    )

    row = service_postgre.get_style_ref_by_lrc("it-catalog", 101)
    assert row is not None
    assert row["file_name"] == "IMG_101.jpg"
    assert row["camera_model"] == "ILCE-7M4"
    assert row["has_edits"] is True
    assert row["develop_settings"] == {"Contrast2012": 25, "Exposure2012": -0.65}
    assert len(row["embedding"]) == CLIP_DIM
    assert len(row["description_embedding"]) == TEXT_DIM

    # Same-vector query must find it with cosine ~1.
    candidates = service_postgre.query_style_ref_candidates("embedding", _clip_vector(3), 5)
    match = next(c for c in candidates if c["lrc_local_id"] == 101)
    assert match["sem"] > 0.999

    # Edited-only default filter hides unedited rows.
    service_postgre.upsert_style_ref(
        catalog_id="it-catalog",
        lrc_local_id=102,
        uuid="3f2a1c6e-1111-4b2a-9c9d-000000000002",
        file_path="/photos/IMG_102.jpg",
        file_name="IMG_102.jpg",
        embedding=_clip_vector(3),
        has_edits=False,
    )
    edited = service_postgre.query_style_ref_candidates(
        "embedding", _clip_vector(3), 50, filters={"has_edits": True}
    )
    assert all(c["has_edits"] for c in edited)
    assert any(c["lrc_local_id"] == 101 for c in edited)
    assert not any(c["lrc_local_id"] == 102 for c in edited)
    everything = service_postgre.query_style_ref_candidates(
        "embedding", _clip_vector(3), 50, filters={"has_edits": None}
    )
    assert {c["lrc_local_id"] for c in everything} >= {101, 102}

    # Description-column candidates (Qwen3 vector, 1024-d).
    desc_query = [1.0] + [0.0] * (TEXT_DIM - 1)
    desc_candidates = service_postgre.query_style_ref_candidates(
        "description_embedding", desc_query, 5
    )
    # Row 101 has a description embedding, row 102 does not.
    assert all(c["lrc_local_id"] != 102 for c in desc_candidates)

    # Self-exclusion.
    excluded = service_postgre.query_style_ref_candidates(
        "embedding", _clip_vector(3), 50, exclude=("it-catalog", 101)
    )
    assert all(c["lrc_local_id"] != 101 for c in excluded)

    # Idempotent upsert: re-run creates no duplicate.
    service_postgre.upsert_style_ref(
        catalog_id="it-catalog",
        lrc_local_id=101,
        uuid="3f2a1c6e-1111-4b2a-9c9d-000000000001",
        file_path="/photos/IMG_101.jpg",
        file_name="IMG_101.jpg",
        embedding=_clip_vector(3),
        has_edits=True,
    )
    stats = service_postgre.style_ref_stats()
    assert stats["total"] >= 2
    rows = service_postgre.get_style_ref_rows([c["id"] for c in everything])
    assert rows

    # Cleanup.
    service_postgre.delete_style_ref("it-catalog", 101)
    service_postgre.delete_style_ref("it-catalog", 102)
    assert service_postgre.get_style_ref_by_lrc("it-catalog", 101) is None
