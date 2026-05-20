import importlib
import sys
import types
from datetime import datetime
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


def test_extract_normalized_metadata_for_storage(monkeypatch):
    service_postgre = _install_postgre_config(monkeypatch)

    normalized = service_postgre._extract_normalized_metadata(
        {
            "capture_time": "2026-05-20 10:30:00",
            "exif": {
                "FNumber": "f/2.8",
                "ISO": "200",
                "FocalLength": "50 mm",
                "Make": "Nikon",
                "Model": "D850",
                "LensModel": "NIKKOR 50mm",
                "GPSLatitude": "45.899",
                "GPSLongitude": "6.129",
            },
        }
    )

    assert normalized == {
        "capture_time": datetime(2026, 5, 20, 10, 30, 0),
        "aperture_f_number": 2.8,
        "shutter_speed": None,
        "exposure_bias": None,
        "iso": 200,
        "focal_length_mm": 50.0,
        "focal_length_35mm": None,
        "camera_make": "Nikon",
        "camera_model": "D850",
        "lens": "NIKKOR 50mm",
        "gps_latitude": 45.899,
        "gps_longitude": 6.129,
    }


def test_extract_normalized_metadata_accepts_lightroom_exif_format(monkeypatch):
    service_postgre = _install_postgre_config(monkeypatch)

    normalized = service_postgre._extract_normalized_metadata(
        {
            "capture_time": "2026-04-25T19:15:03+01:00",
            "exif": {
                "capture_time": "2026-04-25T19:15:03+01:00",
                "all_raw": {
                    "aperture": 1.8,
                    "isoSpeedRating": 100,
                    "focalLength": 50,
                    "focalLength35mm": 50,
                    "shutterSpeed": 0.0008,
                    "exposureBias": -0.7,
                    "lens": "FE 50mm F1.8",
                },
                "all_formatted": {
                    "cameraMake": "Sony",
                    "cameraModel": "ILCE-7M3",
                },
            },
        }
    )

    assert normalized["capture_time"] == datetime(2026, 4, 25, 18, 15, 3)  # UTC
    assert normalized["aperture_f_number"] == 1.8
    assert normalized["shutter_speed"] == 0.0008
    assert normalized["exposure_bias"] == -0.7
    assert normalized["iso"] == 100
    assert normalized["focal_length_mm"] == 50.0
    assert normalized["focal_length_35mm"] == 50.0
    assert normalized["camera_make"] == "Sony"
    assert normalized["camera_model"] == "ILCE-7M3"
    assert normalized["lens"] == "FE 50mm F1.8"
    assert normalized["gps_latitude"] is None
    assert normalized["gps_longitude"] is None


def test_extract_normalized_metadata_accepts_loose_exif_keys(monkeypatch):
    service_postgre = _install_postgre_config(monkeypatch)

    normalized = service_postgre._extract_normalized_metadata(
        {
            "exif": {
                "Date Time Original": "2026:05:20 10:30:00",
                "F Number": "2.8",
                "ISO Speed Ratings": "400",
                "Focal Length": "35 mm",
                "Lens Model": "XF 35mm",
                "GPS Latitude": "45.899",
                "GPS Longitude": "6.129",
            },
        }
    )

    assert normalized["capture_time"] == datetime(2026, 5, 20, 10, 30, 0)
    assert normalized["aperture_f_number"] == 2.8
    assert normalized["iso"] == 400
    assert normalized["focal_length_mm"] == 35.0
    assert normalized["lens"] == "XF 35mm"
    assert normalized["gps_latitude"] == 45.899
    assert normalized["gps_longitude"] == 6.129


def test_extract_normalized_metadata_accepts_exif_json_string(monkeypatch):
    service_postgre = _install_postgre_config(monkeypatch)

    normalized = service_postgre._extract_normalized_metadata(
        {
            "exif": '{"F Number":"2.8","ISO Speed Ratings":"400","Focal Length":"35 mm"}',
        }
    )

    assert normalized["aperture_f_number"] == 2.8
    assert normalized["iso"] == 400
    assert normalized["focal_length_mm"] == 35.0


def test_extract_normalized_metadata_converts_exif_aperture_value(monkeypatch):
    service_postgre = _install_postgre_config(monkeypatch)

    normalized = service_postgre._extract_normalized_metadata(
        {
            "exif": {
                "ApertureValue": 2.970854,
            },
        }
    )

    assert round(normalized["aperture_f_number"], 1) == 2.8


def test_extract_normalized_metadata_accepts_historical_nested_metadata(monkeypatch):
    service_postgre = _install_postgre_config(monkeypatch)

    normalized = service_postgre._extract_normalized_metadata(
        {
            "metadata": {
                "exif": {
                    "F Number": "2.8",
                    "ISO Speed Ratings": "400",
                    "Focal Length": "35 mm",
                }
            }
        }
    )

    assert normalized["aperture_f_number"] == 2.8
    assert normalized["iso"] == 400
    assert normalized["focal_length_mm"] == 35.0


def test_upsert_record_sends_normalized_columns(monkeypatch):
    service_postgre = _install_postgre_config(monkeypatch)
    captured = {}

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, query, params):
            captured["params"] = params

    monkeypatch.setattr(service_postgre, "_connect_to_target", lambda: FakeConnection())

    service_postgre._upsert_record(
        "photo-a",
        {
            "capture_time": "2026-05-20 10:30:00",
            "exif": {
                "capture_time": "2026-05-20T10:30:00",
                "all_raw": {
                    "aperture": 2.8,
                    "isoSpeedRating": 200,
                    "focalLength": 50,
                    "focalLength35mm": 75,
                    "shutterSpeed": 0.001,
                    "exposureBias": -0.7,
                    "lens": "NIKKOR 50mm",
                    "GPSLatitude": "45.899",
                    "GPSLongitude": "6.129",
                },
                "all_formatted": {
                    "cameraMake": "Nikon",
                    "cameraModel": "D850",
                },
            },
        },
        embedding=[1.0, 0.0],
        document="metadata text",
    )

    n = len(service_postgre.NORMALIZED_METADATA_COLUMN_NAMES)
    norm_params = captured["params"][-n:]
    param_dict = dict(zip(service_postgre.NORMALIZED_METADATA_COLUMN_NAMES, norm_params))

    assert param_dict["capture_time"] == datetime(2026, 5, 20, 10, 30, 0)
    assert param_dict["aperture_f_number"] == 2.8
    assert param_dict["shutter_speed"] == 0.001
    assert param_dict["exposure_bias"] == -0.7
    assert param_dict["iso"] == 200
    assert param_dict["focal_length_mm"] == 50.0
    assert param_dict["focal_length_35mm"] == 75.0
    assert param_dict["camera_make"] == "Nikon"
    assert param_dict["camera_model"] == "D850"
    assert param_dict["lens"] == "NIKKOR 50mm"
