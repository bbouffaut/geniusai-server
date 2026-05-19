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


def test_import_metadata_task_stores_capture_time_contract_field(monkeypatch):
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
                "capture_time": "2026-05-02 12:34:56",
                "exif": {"camera_make": "Nikon", "camera_model": "D850"},
            }
        ]
    )

    assert success_count == 1
    assert failure_count == 0
    assert len(captured_updates) == 1
    assert captured_updates[0]["metadata"]["caption"] == "Sunset over the sea"
    assert captured_updates[0]["metadata"]["capture_time"] == "2026-05-02 12:34:56"
    assert captured_updates[0]["metadata"]["exif"] == {
        "camera_make": "Nikon",
        "camera_model": "D850",
    }


def test_import_metadata_task_preserves_existing_fields_when_request_fields_are_blank(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.logger = _make_logger()

    existing_metadata = {
        "filename": "original.jpg",
        "provider": "chatgpt",
        "model": "gpt-5",
        "ai_model": "chatgpt/gpt-5",
        "capture_time": "2026-05-02 12:34:56",
        "caption": "Old caption",
    }
    captured_updates = []

    fake_postgre_service = types.ModuleType("service_postgre")
    fake_postgre_service.get_image = lambda uuid: {
        "ids": [uuid],
        "metadatas": [existing_metadata],
    }
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
                "filename": "",
                "provider": None,
                "model": "   ",
                "ai_model": "",
                "capture_time": "",
                "caption": "Reviewed caption",
            }
        ]
    )

    assert success_count == 1
    assert failure_count == 0
    assert len(captured_updates) == 1
    assert captured_updates[0]["metadata"]["filename"] == "original.jpg"
    assert captured_updates[0]["metadata"]["provider"] == "chatgpt"
    assert captured_updates[0]["metadata"]["model"] == "gpt-5"
    assert captured_updates[0]["metadata"]["ai_model"] == "chatgpt/gpt-5"
    assert captured_updates[0]["metadata"]["capture_time"] == "2026-05-02 12:34:56"
    assert captured_updates[0]["metadata"]["caption"] == "Reviewed caption"


def test_import_metadata_task_treats_blank_existing_update_as_noop_success(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.logger = _make_logger()

    captured_updates = []

    fake_postgre_service = types.ModuleType("service_postgre")
    fake_postgre_service.get_image = lambda uuid: {
        "ids": [uuid],
        "metadatas": [{"filename": "original.jpg"}],
    }
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
                "filename": "",
                "provider": "",
                "ai_model": "",
                "caption": "",
            }
        ]
    )

    assert success_count == 1
    assert failure_count == 0
    assert captured_updates == []
