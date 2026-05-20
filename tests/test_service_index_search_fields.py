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


def test_process_image_task_stores_search_fields(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.TEXT_EMBEDDING_MODEL_ID = "text-model"
    fake_config.logger = _make_logger()

    captured_add_calls = []

    fake_postgre_service = types.ModuleType("service_postgre")
    fake_postgre_service.get_image = lambda uuid: None
    fake_postgre_service.add_image = lambda uuid, embedding, metadata, document=None: captured_add_calls.append(
        {
            "uuid": uuid,
            "embedding": embedding,
            "metadata": metadata,
            "document": document,
        }
    )
    fake_postgre_service.update_image = lambda *args, **kwargs: None

    fake_server_lifecycle = types.ModuleType("server_lifecycle")
    fake_server_lifecycle.embed_document = lambda document: None

    class FakeAnalysisService:
        def analyze_batch(self, image_triplets, options, _image_model=None, _image_processor=None,
                          uuids_needing_embeddings=None, uuids_needing_metadata=None, uuids_needing_quality=None):
            metadata_result = types.SimpleNamespace(
                success=True,
                title="Sunset",
                caption="Sunset over the sea",
                alt_text=None,
                keywords={"Keywords": ["sea", "sunset"]},
            )
            return (
                None,
                {image_triplets[0][1]: 1717000000.0},
                [metadata_result],
                None,
            )

    fake_service_metadata = types.ModuleType("service_metadata")
    fake_service_metadata.get_analysis_service = lambda: FakeAnalysisService()

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    monkeypatch.setitem(sys.modules, "server_lifecycle", fake_server_lifecycle)
    monkeypatch.setitem(sys.modules, "service_metadata", fake_service_metadata)

    sys.modules.pop("service_index", None)
    service_index = importlib.import_module("service_index")

    success_count, failure_count = service_index.process_image_task(
        [(b"fake-image", "photo-1", "alpha.jpg")],
        options={
            "provider": "ollama",
            "model": "gpt-4o",
            "capture_time": "2026-05-02 12:34:56",
            "compute_embeddings": False,
            "compute_metadata": True,
            "compute_quality": False,
            "regenerate_metadata": True,
        },
        additional_metadata_list=[
            {
                "exif": {
                    "camera_make": "Nikon",
                    "camera_model": "D850",
                    "iso_speed_rating": 200,
                }
            }
        ],
    )

    assert success_count == 1
    assert failure_count == 0
    assert len(captured_add_calls) == 1

    stored_metadata = captured_add_calls[0]["metadata"]
    assert stored_metadata["filename"] == "alpha.jpg"
    assert stored_metadata["uuid"] == "photo-1"
    assert stored_metadata["model"] == "gpt-4o"
    assert stored_metadata["ai_model"] == "gpt-4o"
    assert stored_metadata["ai_rundate"] == stored_metadata["run_date"]
    assert stored_metadata["capture_time"] == "2026-05-02 12:34:56"
    assert stored_metadata["exif"] == {
        "camera_make": "Nikon",
        "camera_model": "D850",
        "iso_speed_rating": 200,
    }


def test_process_image_task_stores_capture_time_contract_field(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.TEXT_EMBEDDING_MODEL_ID = "text-model"
    fake_config.logger = _make_logger()

    captured_add_calls = []

    fake_postgre_service = types.ModuleType("service_postgre")
    fake_postgre_service.get_image = lambda uuid: None
    fake_postgre_service.add_image = lambda uuid, embedding, metadata, document=None: captured_add_calls.append(
        {
            "uuid": uuid,
            "embedding": embedding,
            "metadata": metadata,
            "document": document,
        }
    )
    fake_postgre_service.update_image = lambda *args, **kwargs: None

    fake_server_lifecycle = types.ModuleType("server_lifecycle")
    fake_server_lifecycle.embed_document = lambda document: None

    class FakeAnalysisService:
        def analyze_batch(self, image_triplets, options, _image_model=None, _image_processor=None,
                          uuids_needing_embeddings=None, uuids_needing_metadata=None, uuids_needing_quality=None):
            metadata_result = types.SimpleNamespace(
                success=True,
                title="Sunset",
                caption="Sunset over the sea",
                alt_text=None,
                keywords={"Keywords": ["sea", "sunset"]},
            )
            return (
                None,
                {},
                [metadata_result],
                None,
            )

    fake_service_metadata = types.ModuleType("service_metadata")
    fake_service_metadata.get_analysis_service = lambda: FakeAnalysisService()

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    monkeypatch.setitem(sys.modules, "server_lifecycle", fake_server_lifecycle)
    monkeypatch.setitem(sys.modules, "service_metadata", fake_service_metadata)

    sys.modules.pop("service_index", None)
    service_index = importlib.import_module("service_index")

    success_count, failure_count = service_index.process_image_task(
        [(b"fake-image", "photo-1", "alpha.jpg")],
        options={
            "provider": "ollama",
            "model": "gpt-4o",
            "capture_time": "2026-05-02 12:34:56",
            "compute_embeddings": False,
            "compute_metadata": True,
            "compute_quality": False,
            "regenerate_metadata": True,
        },
    )

    assert success_count == 1
    assert failure_count == 0
    assert len(captured_add_calls) == 1
    assert captured_add_calls[0]["metadata"]["capture_time"] == "2026-05-02 12:34:56"


def test_process_image_task_prestores_client_metadata_before_generation(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.TEXT_EMBEDDING_MODEL_ID = "text-model"
    fake_config.logger = _make_logger()

    records = {}
    captured_updates = []

    fake_postgre_service = types.ModuleType("service_postgre")

    def fake_get_image(uuid):
        if uuid not in records:
            return None
        return {"ids": [uuid], "metadatas": [records[uuid]]}

    def fake_update_image(uuid, metadata, embedding=None, document=None):
        captured_updates.append(
            {
                "uuid": uuid,
                "metadata": metadata.copy(),
                "embedding": embedding,
                "document": document,
            }
        )
        records[uuid] = metadata.copy()

    fake_postgre_service.get_image = fake_get_image
    fake_postgre_service.update_image = fake_update_image
    fake_postgre_service.add_image = lambda uuid, embedding, metadata, document=None: fake_update_image(
        uuid,
        metadata,
        embedding=embedding,
        document=document,
    )

    fake_server_lifecycle = types.ModuleType("server_lifecycle")
    fake_server_lifecycle.embed_document = lambda document: None

    class FakeAnalysisService:
        def analyze_batch(self, image_triplets, options, _image_model=None, _image_processor=None,
                          uuids_needing_embeddings=None, uuids_needing_metadata=None, uuids_needing_quality=None):
            assert "photo-1" in records
            assert records["photo-1"]["exif"]["Model"] == "Client Camera"
            metadata_result = types.SimpleNamespace(
                success=True,
                title="Mountain lake",
                caption="A mountain lake at sunset",
                alt_text=None,
                keywords={"Keywords": ["lake"]},
            )
            return None, {}, [metadata_result], None

    fake_service_metadata = types.ModuleType("service_metadata")
    fake_service_metadata.get_analysis_service = lambda: FakeAnalysisService()

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    monkeypatch.setitem(sys.modules, "server_lifecycle", fake_server_lifecycle)
    monkeypatch.setitem(sys.modules, "service_metadata", fake_service_metadata)

    sys.modules.pop("service_index", None)
    service_index = importlib.import_module("service_index")

    success_count, failure_count = service_index.process_image_task(
        [(b"fake-image", "photo-1", "alpha.jpg")],
        options={
            "provider": "anthropic",
            "model": "claude",
            "capture_time": "2026-05-02 12:34:56",
            "compute_embeddings": False,
            "compute_metadata": True,
            "compute_quality": False,
            "regenerate_metadata": True,
        },
        additional_metadata_list=[
            {
                "exif": {
                    "Model": "Client Camera",
                    "LensModel": "Client Lens",
                }
            }
        ],
    )

    assert success_count == 1
    assert failure_count == 0
    assert len(captured_updates) >= 2
    assert captured_updates[0]["metadata"]["has_embedding"] is False
    assert captured_updates[0]["metadata"]["exif"]["Model"] == "Client Camera"
    assert records["photo-1"]["title"] == "Mountain lake"
    assert records["photo-1"]["exif"]["LensModel"] == "Client Lens"


def test_process_image_task_does_not_extract_exif_from_image_bytes(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.TEXT_EMBEDDING_MODEL_ID = "text-model"
    fake_config.logger = _make_logger()

    records = {}

    fake_postgre_service = types.ModuleType("service_postgre")

    def fake_get_image(uuid):
        if uuid not in records:
            return None
        return {"ids": [uuid], "metadatas": [records[uuid]]}

    def fake_update_image(uuid, metadata, embedding=None, document=None):
        records[uuid] = metadata.copy()

    fake_postgre_service.get_image = fake_get_image
    fake_postgre_service.update_image = fake_update_image
    fake_postgre_service.add_image = lambda uuid, embedding, metadata, document=None: fake_update_image(
        uuid,
        metadata,
        embedding=embedding,
        document=document,
    )

    fake_server_lifecycle = types.ModuleType("server_lifecycle")
    fake_server_lifecycle.embed_document = lambda document: None

    class FakeAnalysisService:
        def analyze_batch(self, image_triplets, options, _image_model=None, _image_processor=None,
                          uuids_needing_embeddings=None, uuids_needing_metadata=None, uuids_needing_quality=None):
            return None, {}, [None], None

    fake_service_metadata = types.ModuleType("service_metadata")
    fake_service_metadata.get_analysis_service = lambda: FakeAnalysisService()

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    monkeypatch.setitem(sys.modules, "server_lifecycle", fake_server_lifecycle)
    monkeypatch.setitem(sys.modules, "service_metadata", fake_service_metadata)

    sys.modules.pop("service_index", None)
    service_index = importlib.import_module("service_index")

    success_count, failure_count = service_index.process_image_task(
        [(b"image-with-unread-exif", "photo-1", "alpha.jpg")],
        options={
            "provider": "anthropic",
            "model": "claude",
            "capture_time": "2026-05-02 12:34:56",
            "compute_embeddings": False,
            "compute_metadata": False,
            "compute_quality": False,
            "regenerate_metadata": True,
        },
    )

    assert success_count == 1
    assert failure_count == 0
    assert "exif" not in records["photo-1"]


def test_process_image_task_stores_metadata_when_embedding_generation_fails(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.TEXT_EMBEDDING_MODEL_ID = "text-model"
    fake_config.logger = _make_logger()

    records = {}

    fake_postgre_service = types.ModuleType("service_postgre")

    def fake_get_image(uuid):
        if uuid not in records:
            return None
        return {"ids": [uuid], "metadatas": [records[uuid]]}

    def fake_update_image(uuid, metadata, embedding=None, document=None):
        records[uuid] = metadata.copy()

    fake_postgre_service.get_image = fake_get_image
    fake_postgre_service.update_image = fake_update_image
    fake_postgre_service.add_image = lambda uuid, embedding, metadata, document=None: fake_update_image(
        uuid,
        metadata,
        embedding=embedding,
        document=document,
    )

    fake_server_lifecycle = types.ModuleType("server_lifecycle")
    fake_server_lifecycle.embed_document = lambda document: None

    class FakeAnalysisService:
        def analyze_batch(self, image_triplets, options, _image_model=None, _image_processor=None,
                          uuids_needing_embeddings=None, uuids_needing_metadata=None, uuids_needing_quality=None):
            metadata_result = types.SimpleNamespace(
                success=True,
                title="Sunset",
                caption="Sunset over the sea",
                alt_text=None,
                keywords={"Keywords": ["sea", "sunset"]},
            )
            return None, {}, [metadata_result], None

    fake_service_metadata = types.ModuleType("service_metadata")
    fake_service_metadata.get_analysis_service = lambda: FakeAnalysisService()

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    monkeypatch.setitem(sys.modules, "server_lifecycle", fake_server_lifecycle)
    monkeypatch.setitem(sys.modules, "service_metadata", fake_service_metadata)

    sys.modules.pop("service_index", None)
    service_index = importlib.import_module("service_index")

    success_count, failure_count = service_index.process_image_task(
        [(b"fake-image", "photo-1", "alpha.jpg")],
        options={
            "provider": "ollama",
            "model": "gpt-4o",
            "capture_time": "2026-05-02 12:34:56",
            "compute_embeddings": True,
            "compute_metadata": True,
            "compute_quality": False,
            "regenerate_metadata": True,
        },
        additional_metadata_list=[
            {
                "exif": {
                    "FNumber": "2.8",
                    "ISOSpeedRatings": "400",
                }
            }
        ],
    )

    assert success_count == 1
    assert failure_count == 0
    assert records["photo-1"]["title"] == "Sunset"
    assert records["photo-1"]["has_embedding"] is False
    assert records["photo-1"]["exif"]["FNumber"] == "2.8"
    assert "metadata_search_text" not in records["photo-1"]
