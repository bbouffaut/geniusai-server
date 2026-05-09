
from flask import Blueprint, jsonify, request
#import service_chroma as chroma_service
import service_postgre as postgre_service

import server_lifecycle
from config import logger
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
