
import json
import time
from flask import Blueprint, Response, jsonify, request, stream_with_context
#import service_chroma as chroma_service
import service_postgre as postgre_service

import server_lifecycle
import service_migrate
from config import EMBEDDING_MODELS, logger
from service_metadata import get_analysis_service
server_bp = Blueprint('server', __name__)


def _request_data():
    if request.is_json:
        return request.get_json(silent=True) or {}
    if request.form:
        return request.form
    return {}


def _request_value(data, key):
    return request.args.get(key) or data.get(key)

@server_bp.route('/ping', methods=['GET'])
def ping():
    #logger.info("Ping request received")
    return "pong"


@server_bp.route('/shutdown', methods=['POST'])
def shutdown():
    server_lifecycle.request_shutdown()
    return jsonify({"status": "Server is shutting down..."})

@server_bp.route('/stats', methods=['GET'])
def stats():
    logger.info("Statistics request received")
    results = postgre_service.get_db_stats()
    return jsonify(results)


@server_bp.route('/models', methods=['GET', 'POST'])
def list_models():
    """
    Returns all available multimodal models from all providers.
    
    Dynamically checks availability of Ollama and LM Studio on each request.
    Uses provider APIs when cloud API keys are supplied. Providers return an
    empty list when no models are returned. Always filters for multimodal
    (vision-capable) models only.
    
    POST JSON or form data: { 
        openai_apikey?: str,  # Optional OpenAI API key for ChatGPT models
        gemini_apikey?: str,  # Optional Gemini API key for Gemini models
        mistral_apikey?: str,  # Optional Mistral API key for Mistral models
        anthropic_apikey?: str  # Optional Anthropic API key for Claude models
    }
    
    Returns: {
        "models": {
            "qwen": ["model1", "model2"],
            "ollama": [...],
            "lmstudio": [...],
            "chatgpt": [...],
            "gemini": [...],
            "mistral": [...],
            "anthropic": [...]
        }
    }
    """
    data = _request_data()
    openai_apikey = _request_value(data, 'openai_apikey')
    gemini_apikey = _request_value(data, 'gemini_apikey')
    mistral_apikey = _request_value(data, 'mistral_apikey')
    anthropic_apikey = _request_value(data, 'anthropic_apikey')

    logger.info("Models request received - checking all providers")
    
    try:
        # Get all available multimodal models
        # This will dynamically re-check Ollama and LM Studio availability
        models = get_analysis_service().get_available_models(
            openai_apikey=openai_apikey,
            gemini_apikey=gemini_apikey,
            mistral_apikey=mistral_apikey,
            anthropic_apikey=anthropic_apikey,
        )
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@server_bp.route('/migrate', methods=['POST'])
def start_migrate():
    """
    Start a migration from the server's current database to a new one and stream
    progress back to the caller as newline-delimited JSON (NDJSON).

    The connection stays open for the duration of the migration.  One JSON object
    is written per second while the migration runs; the final object carries the
    completed/failed status.  HTTP status is always 200 once streaming begins.

    POST JSON body:
      {
        "target_db":    "<new database name>",   // required
        "target_model": "<embedding model key>", // required — e.g. "bge-m3", "qwen3-0.6b"
        "batch_size":   32                       // optional, default 32
      }

    Returns 409 Conflict (JSON, non-streaming) if a migration is already running.

    Postman: the response body accumulates NDJSON lines live; the final line
    shows the outcome.  Each line is valid JSON on its own.
    """
    data = _request_data()
    target_db    = _request_value(data, 'target_db')
    target_model = _request_value(data, 'target_model')
    batch_size   = _request_value(data, 'batch_size')

    if not target_db:
        return jsonify({"error": "target_db is required"}), 400
    if not target_model:
        return jsonify({"error": "target_model is required"}), 400
    if target_model not in EMBEDDING_MODELS:
        return jsonify({
            "error": f"Unknown model '{target_model}'. Supported: {list(EMBEDDING_MODELS.keys())}"
        }), 400

    try:
        batch_size = int(batch_size) if batch_size is not None else 32
    except (ValueError, TypeError):
        return jsonify({"error": "batch_size must be an integer"}), 400

    try:
        service_migrate.start_migration(target_db, target_model, batch_size)
    except service_migrate.MigrationAlreadyRunningError:
        return jsonify({
            "error": "A migration is already running.",
            "migration": service_migrate.get_status(),
        }), 409
    except Exception as e:
        logger.error(f"Failed to start migration: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    @stream_with_context
    def _stream():
        while True:
            state = service_migrate.get_status()
            yield json.dumps({"migration": state}, ensure_ascii=False) + "\n"
            if state["status"] != "running":
                break
            time.sleep(1)

    return Response(
        _stream(),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )


@server_bp.route('/migrate', methods=['GET'])
def migrate_status():
    """
    Return the status of the current or most-recent migration.

    Returns 404 if no migration has ever been started.
    """
    state = service_migrate.get_status()
    if state is None:
        return jsonify({"error": "No migration has been started."}), 404
    return jsonify({"migration": state})
