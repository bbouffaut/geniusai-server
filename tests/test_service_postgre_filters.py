import importlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _make_logger():
    return types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )


def _install_postgre_config(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.POSTGRE_DATABASE_NAME = "test-db"
    fake_config.POSTGRE_PASSWORD = None
    fake_config.POSTGRE_URL = "postgresql://localhost/postgres"
    fake_config.POSTGRE_USER = None
    fake_config.TEXT_EMBEDDING_DIMENSION = 2
    fake_config.logger = _make_logger()

    monkeypatch.setitem(sys.modules, "config", fake_config)
    sys.modules.pop("service_postgre", None)

    return importlib.import_module("service_postgre")


def test_postgre_filter_builder_includes_uuid_date_and_aperture_params(monkeypatch):
    service_postgre = _install_postgre_config(monkeypatch)

    clauses, params = service_postgre._filter_clauses(
        {
            "uuid": {"$in": ["photo-a", "photo-b"]},
            "metadata_filters": [
                {
                    "field": "capture_time",
                    "op": "gte",
                    "value": "2026-05-01",
                },
                {
                    "field": "aperture",
                    "op": "number_eq",
                    "value": 2.8,
                    "tolerance": 0.05,
                },
            ],
        }
    )

    assert len(clauses) == 3
    assert params[0] == ["photo-a", "photo-b"]
    assert ["capture_time"] in params
    assert "2026-05-01" in params
    assert ["aperture"] in params
    assert any(isinstance(value, float) and abs(value - 2.75) < 1e-9 for value in params)
    assert any(isinstance(value, float) and abs(value - 2.85) < 1e-9 for value in params)
