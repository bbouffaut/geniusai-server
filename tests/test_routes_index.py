import base64
import importlib
import io
import sys
import types
from pathlib import Path

from flask import Flask


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


def _install_index_fakes(monkeypatch):
    fake_config = types.ModuleType("config")
    fake_config.DEFAULT_METADATA_LANGUAGE = "English"
    fake_config.UPLOAD_TEMP_DIR = "/tmp"
    fake_config.logger = _make_logger()

    fake_postgre_service = types.ModuleType("service_postgre")
    fake_postgre_service.delete_image = lambda *args, **kwargs: None
    fake_postgre_service.get_image = lambda *args, **kwargs: None
    fake_postgre_service.get_all_image_ids = lambda *args, **kwargs: []

    captured_calls = []

    def fake_process_image_task(image_triplets, options, additional_metadata_list=None):
        captured_calls.append(
            {
                "image_triplets": image_triplets,
                "options": options,
                "additional_metadata_list": additional_metadata_list,
            }
        )
        return len(image_triplets), 0

    fake_service_index = types.ModuleType("service_index")
    fake_service_index.keywords_with_generation_model = lambda keywords, provider, model: keywords
    fake_service_index.process_image_task = fake_process_image_task

    monkeypatch.setitem(sys.modules, "config", fake_config)
    monkeypatch.setitem(sys.modules, "service_postgre", fake_postgre_service)
    monkeypatch.setitem(sys.modules, "service_index", fake_service_index)
    sys.modules.pop("routes_index", None)

    routes_index = importlib.import_module("routes_index")

    app = Flask(__name__)
    app.register_blueprint(routes_index.index_bp)

    return app, captured_calls


def test_index_requires_filename(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)

    with app.test_client() as client:
        response = client.post(
            "/index",
            data={
                "image": (io.BytesIO(b"fake-image"), "upload.jpg"),
                "uuid": "photo-1",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
    assert "filename" in response.get_json()["error"]
    assert captured_calls == []


def test_index_requires_photo_date(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)

    with app.test_client() as client:
        response = client.post(
            "/index",
            data={
                "image": (io.BytesIO(b"fake-image"), "upload.jpg"),
                "uuid": "photo-1",
                "filename": "photo-1.jpg",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
    assert "photo_date" in response.get_json()["error"]
    assert captured_calls == []


def test_index_accepts_extra_metadata(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)

    with app.test_client() as client:
        response = client.post(
            "/index",
            data={
                "image": (io.BytesIO(b"fake-image"), "upload.jpg"),
                "uuid": "photo-1",
                "filename": "photo-1.jpg",
                "photo_date": "2026-05-02 12:34:56",
                "exif": "{\"camera\": \"Nikon\", \"iso\": 200}",
                "album": "Summer",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert response.get_json()["success_count"] == 1
    assert captured_calls[0]["image_triplets"] == [(b"fake-image", "photo-1", "photo-1.jpg")]
    assert captured_calls[0]["options"]["photo_date"] == "2026-05-02 12:34:56"
    assert captured_calls[0]["additional_metadata_list"] == [
        {
            "exif": {"camera": "Nikon", "iso": 200},
            "album": "Summer",
        }
    ]


def test_index_keeps_photo_date_out_of_extra_metadata(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)

    with app.test_client() as client:
        response = client.post(
            "/index",
            data={
                "image": (io.BytesIO(b"fake-image"), "upload.jpg"),
                "uuid": "photo-1",
                "filename": "photo-1.jpg",
                "photo_date": "2026-05-02 12:34:56",
                "album": "Summer",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert response.get_json()["success_count"] == 1
    assert captured_calls[0]["options"]["photo_date"] == "2026-05-02 12:34:56"
    assert captured_calls[0]["additional_metadata_list"] == [
        {
            "album": "Summer",
        }
    ]


def test_index_rejects_legacy_photo_date_fields(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)

    with app.test_client() as client:
        response = client.post(
            "/index",
            data={
                "image": (io.BytesIO(b"fake-image"), "upload.jpg"),
                "uuid": "photo-1",
                "filename": "photo-1.jpg",
                "photos_date": "2026-05-02 12:34:56",
            },
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
    assert "Use 'photo_date'" in response.get_json()["error"]
    assert captured_calls == []


def test_index_base64_accepts_extra_metadata(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)
    encoded_image = base64.b64encode(b"fake-image").decode("ascii")

    with app.test_client() as client:
        response = client.post(
            "/index_base64",
            json={
                "image": encoded_image,
                "uuid": "photo-1",
                "filename": "photo-1.jpg",
                "photo_date": "2026-05-02 12:34:56",
                "exif": {"camera": "Nikon"},
                "album": "Summer",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["success_count"] == 1
    assert captured_calls[0]["image_triplets"] == [(b"fake-image", "photo-1", "photo-1.jpg")]
    assert captured_calls[0]["options"]["photo_date"] == "2026-05-02 12:34:56"
    assert captured_calls[0]["additional_metadata_list"] == [
        {
            "exif": {"camera": "Nikon"},
            "album": "Summer",
        }
    ]


def test_index_base64_keeps_photo_date_out_of_extra_metadata(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)
    encoded_image = base64.b64encode(b"fake-image").decode("ascii")

    with app.test_client() as client:
        response = client.post(
            "/index_base64",
            json={
                "image": encoded_image,
                "uuid": "photo-1",
                "filename": "photo-1.jpg",
                "photo_date": "2026-05-02 12:34:56",
                "album": "Summer",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["success_count"] == 1
    assert captured_calls[0]["options"]["photo_date"] == "2026-05-02 12:34:56"
    assert captured_calls[0]["additional_metadata_list"] == [
        {
            "album": "Summer",
        }
    ]


def test_index_base64_rejects_legacy_photo_date_fields(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)
    encoded_image = base64.b64encode(b"fake-image").decode("ascii")

    with app.test_client() as client:
        response = client.post(
            "/index_base64",
            json={
                "image": encoded_image,
                "uuid": "photo-1",
                "filename": "photo-1.jpg",
                "date_time": "2026-05-02 12:34:56",
            },
        )

    assert response.status_code == 400
    assert "Use 'photo_date'" in response.get_json()["error"]
    assert captured_calls == []


def test_index_base64_requires_photo_date(monkeypatch):
    app, captured_calls = _install_index_fakes(monkeypatch)
    encoded_image = base64.b64encode(b"fake-image").decode("ascii")

    with app.test_client() as client:
        response = client.post(
            "/index_base64",
            json={
                "image": encoded_image,
                "uuid": "photo-1",
                "filename": "photo-1.jpg",
            },
        )

    assert response.status_code == 400
    assert "photo_date" in response.get_json()["error"]
    assert captured_calls == []
