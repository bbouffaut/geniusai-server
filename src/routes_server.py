
from flask import Blueprint, jsonify, request
import service_chroma as chroma_service

import server_lifecycle
from config import logger
from service_metadata import get_analysis_service
from server_lifecycle import get_model, start_download_clip_model, get_download_status

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
    Always filters for multimodal (vision-capable) models only.
    
    POST JSON: { 
        openai_apikey?: str,  # Optional OpenAI API key for ChatGPT models
        gemini_apikey?: str   # Optional Gemini API key for Gemini models
    }
    
    Returns: {
        "models": {
            "qwen": ["model1", "model2"],
            "ollama": [...],
            "lmstudio": [...],
            "chatgpt": [...],
            "gemini": [...]
        }
    }
    """
    # Parse API keys from request
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        openai_apikey = data.get('openai_apikey')
        gemini_apikey = data.get('gemini_apikey')
    else:
        # Support GET for backward compatibility
        openai_apikey = request.args.get('openai_apikey')
        gemini_apikey = request.args.get('gemini_apikey')

    logger.info("Models request received - checking all providers")
    
    try:
        # Get all available multimodal models
        # This will dynamically re-check Ollama and LM Studio availability
        models = get_analysis_service().get_available_models(
            openai_apikey=openai_apikey,
            gemini_apikey=gemini_apikey
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
            return jsonify({"clip": "ready", "message": "CLIP model is loaded and ready."})
        else:
            return jsonify({"clip": "not_ready", "message": "CLIP model is not loaded."})           
        
    except Exception as e:
        logger.error(f"Error while loading CLIP model: {e}", exc_info=True)
        return jsonify({"clip": "not_ready", "message": str(e)})
    
@server_bp.route('/clip/download/start', methods=['POST'])
def download_clip_model_start():
    logger.info("Download CLIP model request received")

    try:
        start_download_clip_model()
        return jsonify({"download": "started"})
    except Exception as e:
        logger.error(f"Error while starting to download CLIP model: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@server_bp.route('/clip/download/status', methods=['GET'])
def download_clip_model_status():
    logger.info("Download CLIP model status request received")

    try:
        status = get_download_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error while getting download status for CLIP model: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
        
        