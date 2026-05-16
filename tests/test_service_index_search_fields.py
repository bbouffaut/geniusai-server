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
            "date_time": "2026-05-02 12:34:56",
            "compute_embeddings": False,
            "compute_metadata": True,
            "compute_quality": False,
            "regenerate_metadata": True,
        },
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
    assert stored_metadata["photo_date"] == "2026-05-02 12:34:56"
