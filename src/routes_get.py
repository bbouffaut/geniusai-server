import json

from flask import Blueprint, jsonify, request

from config import logger
import service_get


get_bp = Blueprint('get', __name__)


def _request_body_dict():
    if request.is_json:
        data = request.get_json(silent=True)
        if data is not None:
            return data
        if request.content_length in (None, 0):
            return {}
        return None

    if request.form:
        payload = {}
        for key in request.form.keys():
            values = request.form.getlist(key)
            payload[key] = values if len(values) > 1 else request.form.get(key)
        return payload

    raw_body = request.get_data(cache=True, as_text=True)
    if raw_body and raw_body.strip():
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            return None

    return {}


@get_bp.route('/get', methods=['POST'])
def get_photo_data():
    logger.info("Get photo data request received")

    data = _request_body_dict()
    if data is None:
        return jsonify({"status": "error", "error": "Request body must be valid JSON"}), 400

    if not isinstance(data, dict):
        return jsonify({"status": "error", "error": "Request body must be a JSON object"}), 400

    try:
        return jsonify(service_get.get_photos(data))
    except ValueError as e:
        logger.error(f"Invalid get photo data request: {e}")
        return jsonify({"status": "error", "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error retrieving photo data: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500
