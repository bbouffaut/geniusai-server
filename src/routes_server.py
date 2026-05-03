
from flask import Blueprint, jsonify, request
import service_chroma as chroma_service

import server_lifecycle
from config import logger
from service_metadata import get_analysis_service
from server_lifecycle import get_model, start_download_embedding_model, get_download_status

server_bp = Blueprint('server', __name__)

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
    results = chroma_service.get_db_stats()
    return jsonify(results)


@server_bp.route('/models', methods=['GET', 'POST'])
def list_models():
    """
    Returns all available multimodal models from all providers.
    
    Dynamically checks availability of Ollama and LM Studio on each request.
    Uses provider APIs when cloud API keys are supplied, with static fallbacks
    when keys are missing or listing fails. Always filters for multimodal
    (vision-capable) models only.
    
    POST JSON: { 
        openai_apikey?: str,  # Optional OpenAI API key for ChatGPT models
        gemini_apikey?: str,  # Optional Gemini API key for Gemini models
        mistral_apikey?: str  # Optional Mistral API key for Mistral models
    }
    
    Returns: {
        "models": {
            "qwen": ["model1", "model2"],
            "ollama": [...],
            "lmstudio": [...],
            "chatgpt": [...],
            "gemini": [...],
            "mistral": [...]
        }
    }
    """
    # Parse API keys from request
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        openai_apikey = data.get('openai_apikey')
        gemini_apikey = data.get('gemini_apikey')
        mistral_apikey = data.get('mistral_apikey')
    else:
        # Support GET for backward compatibility
        openai_apikey = request.args.get('openai_apikey')
        gemini_apikey = request.args.get('gemini_apikey')
        mistral_apikey = request.args.get('mistral_apikey')

    logger.info("Models request received - checking all providers")
    
    try:
        # Get all available multimodal models
        # This will dynamically re-check Ollama and LM Studio availability
        models = get_analysis_service().get_available_models(
            openai_apikey=openai_apikey,
            gemini_apikey=gemini_apikey,
            mistral_apikey=mistral_apikey,
        )
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@server_bp.route('/clip/status', methods=['GET'])
def clip_cached():
    try:
        model = get_model()
        if model:
            return jsonify({"embedding": "ready", "clip": "ready", "message": "Text embedding model is loaded and ready."})
        else:
            return jsonify({"embedding": "not_ready", "clip": "not_ready", "message": "Text embedding model is not loaded."})
        
    except Exception as e:
        logger.error(f"Error while loading text embedding model: {e}", exc_info=True)
        return jsonify({"embedding": "not_ready", "clip": "not_ready", "message": str(e)})
    
@server_bp.route('/clip/download/start', methods=['POST'])
def download_clip_model_start():
    logger.info("Download text embedding model request received")

    try:
        start_download_embedding_model()
        return jsonify({"download": "started"})
    except Exception as e:
        logger.error(f"Error while starting to download text embedding model: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@server_bp.route('/clip/download/status', methods=['GET'])
def download_clip_model_status():
    logger.info("Download text embedding model status request received")

    try:
        status = get_download_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error while getting download status for text embedding model: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
        
