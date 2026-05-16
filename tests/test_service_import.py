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


def test_import_metadata_task_normalizes_photos_date(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.logger = _make_logger()

    captured_updates = []

    fake_postgre_service = types.ModuleType("service_postgre")
    fake_postgre_service.get_image = lambda uuid: {"ids": [uuid], "metadatas": [{}]}
    fake_postgre_service.update_image = lambda uuid, metadata: captured_updates.append(
        {
            "uuid": uuid,
            "metadata": metadata,
        }
    )
    fake_postgre_service.add_image = lambda *args, **kwargs: None

    fake_service_index = types.ModuleType("service_index")
    fake_service_index._flatten_keywords = lambda keywords: ", ".join(keywords)

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    monkeypatch.setitem(sys.modules, "service_index", fake_service_index)

    sys.modules.pop("service_import", None)
    service_import = importlib.import_module("service_import")

    success_count, failure_count = service_import.import_metadata_task(
        [
            {
                "uuid": "photo-1",
                "caption": "Sunset over the sea",
                "photos_date": "2026-05-02 12:34:56",
            }
        ]
    )

    assert success_count == 1
    assert failure_count == 0
    assert len(captured_updates) == 1
    assert captured_updates[0]["metadata"]["caption"] == "Sunset over the sea"
    assert captured_updates[0]["metadata"]["photo_date"] == "2026-05-02 12:34:56"
    assert "photos_date" not in captured_updates[0]["metadata"]
