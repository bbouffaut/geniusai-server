import os
import sys
import json
import tempfile
from flask import Flask, Request, jsonify, request
from waitress import serve
import datetime

# Import modularized components
from config import (
    DEBUG_IN_FILE,
    DEBUG_IN_FILE_PATH,
    POSTGRE_DATABASE_NAME,
    PRELOAD_MODELS,
    SERVER_HOST,
    SERVER_PORT,
    UPLOAD_TEMP_DIR,
    args,
    logger,
    raw_debug_logger,
)
logger.info("Imported config")

# Lazy import server_lifecycle to speed up startup
import server_lifecycle
logger.info("Imported server_lifecycle")

# Import blueprints only (services are imported by routes when needed)
from routes_get import get_bp
from routes_index import index_bp
from routes_reindex import reindex_bp
from routes_search import search_bp
from routes_server import server_bp
from routes_import import import_bp
import service_postgre as postgre_service


class GeniusAIRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return tempfile.NamedTemporaryFile(mode="w+b", prefix="geniusai-upload-", dir=UPLOAD_TEMP_DIR, delete=True)


app = Flask(__name__)
app.request_class = GeniusAIRequest
logger.info("Flask app created")


SENSITIVE_REQUEST_KEYS = {
    "api_key",
    "apikey",
    "anthropic_apikey",
    "authorization",
    "cookie",
    "gemini_apikey",
    "mistral_apikey",
    "openai_apikey",
    "password",
    "secret",
    "token",
}
LARGE_REQUEST_KEYS = {"data", "image", "image_data", "images"}
MAX_DEBUG_REQUEST_STRING_LENGTH = 4000
MAX_DEBUG_RAW_BODY_LENGTH = 20000


def _is_sensitive_request_key(key):
    normalized_key = str(key).lower().replace("-", "_")
    return normalized_key in SENSITIVE_REQUEST_KEYS or normalized_key.endswith("_token")


def _summarize_debug_string(value, key=None):
    if _is_sensitive_request_key(key):
        return f"<redacted; length={len(value)}>"

    normalized_key = str(key).lower() if key is not None else ""
    if normalized_key in LARGE_REQUEST_KEYS or value.startswith("data:image/"):
        return f"<omitted large request string; {len(value)} chars>"

    if len(value) > MAX_DEBUG_REQUEST_STRING_LENGTH:
        omitted_chars = len(value) - MAX_DEBUG_REQUEST_STRING_LENGTH
        return f"{value[:MAX_DEBUG_REQUEST_STRING_LENGTH]}... [truncated {omitted_chars} chars]"

    return value


def _sanitize_debug_request_value(value, key=None):
    if isinstance(value, dict):
        return {
            item_key: _sanitize_debug_request_value(item_value, item_key)
            for item_key, item_value in value.items()
        }

    if isinstance(value, list):
        return [_sanitize_debug_request_value(item, key) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_debug_request_value(item, key) for item in value]

    if isinstance(value, bytes):
        return f"<omitted request bytes; {len(value)} bytes>"

    if isinstance(value, str):
        return _summarize_debug_string(value, key)

    return value


def _request_multidict_to_dict(values):
    return {
        key: values.getlist(key) if len(values.getlist(key)) > 1 else values.get(key)
        for key in values.keys()
    }


def _debug_file_request_payload():
    payload = {
        "method": request.method,
        "path": request.path,
        "full_path": request.full_path,
        "remote_addr": request.remote_addr,
        "content_type": request.content_type,
        "content_length": request.content_length,
        "headers": _sanitize_debug_request_value(dict(request.headers)),
    }

    if request.args:
        payload["query"] = _sanitize_debug_request_value(_request_multidict_to_dict(request.args))

    if request.is_json:
        payload["json"] = _sanitize_debug_request_value(request.get_json(silent=True))
    elif request.form or request.files:
        if request.form:
            payload["form"] = _sanitize_debug_request_value(_request_multidict_to_dict(request.form))
        if request.files:
            payload["files"] = [
                {
                    "field": field_name,
                    "filename": file_storage.filename,
                    "content_type": file_storage.content_type,
                    "content_length": file_storage.content_length,
                }
                for field_name in request.files.keys()
                for file_storage in request.files.getlist(field_name)
            ]
    else:
        raw_body = request.get_data(cache=True)
        if raw_body:
            if len(raw_body) > MAX_DEBUG_RAW_BODY_LENGTH:
                payload["body"] = f"<omitted raw request body; {len(raw_body)} bytes>"
            else:
                payload["body"] = raw_body.decode("utf-8", errors="replace")

    return payload


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, apikey"
    return response


@app.before_request
def handle_options_preflight():
    if request.method == "OPTIONS":
        from flask import make_response
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, apikey"
        return response, 204


@app.before_request
def log_incoming_request_for_debug_file():
    if not DEBUG_IN_FILE:
        return

    try:
        raw_debug_logger.debug(
            "Incoming HTTP request:\n%s",
            json.dumps(
                _debug_file_request_payload(),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
        )
    except Exception as e:
        logger.warning(f"Failed to write incoming request to debug file: {e}")


# Register blueprints
app.register_blueprint(get_bp)
app.register_blueprint(index_bp)
app.register_blueprint(reindex_bp)
app.register_blueprint(search_bp)
app.register_blueprint(server_bp)
app.register_blueprint(import_bp)

@app.errorhandler(500)
def handle_internal_server_error(e):
    logger.error(f"Internal Server Error: {e}")
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("LrGenius Server starting...")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PostgreSQL: {postgre_service.describe_connection_target()}")
    logger.info(f"Database: {POSTGRE_DATABASE_NAME}")
    logger.info(f"Upload temp dir: {UPLOAD_TEMP_DIR}")
    if DEBUG_IN_FILE_PATH:
        logger.warning(
            "Raw debug logging enabled; unredacted LLM request payloads "
            f"will be written to {DEBUG_IN_FILE_PATH}"
        )
    logger.info("=" * 60)

    logger.info("Initializing PostgreSQL before accepting requests...")
    try:
        postgre_service.initialize()
    except postgre_service.PostgreStartupError as e:
        logger.critical("%s", e)
        if DEBUG_IN_FILE:
            raw_debug_logger.debug("PostgreSQL startup traceback", exc_info=True)
        sys.exit(1)
    logger.info("PostgreSQL initialized")

    should_preload_models = PRELOAD_MODELS and (
        not args.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    )

    if should_preload_models:
        logger.info("Preloading embedding model before accepting requests...")
        model = server_lifecycle.get_model()
        if model is None:
            logger.warning("Embedding model preload did not complete; semantic features will retry on first use.")
        else:
            logger.info("Embedding model preloaded")
    
    # Mark server as ready for startup scripts
    server_lifecycle.write_ok_file()
    logger.info("Server initialized and ready to accept connections")
    
    # Write PID for lifecycle management
    server_lifecycle.write_pid_file()
    
    try:
        if args.debug:
            logger.info(
                f"Starting Flask development server in debug mode on http://{SERVER_HOST}:{SERVER_PORT}"
            )
            app.run(debug=True, host=SERVER_HOST, port=SERVER_PORT)
        else:
            logger.info(f"Starting production server on http://{SERVER_HOST}:{SERVER_PORT}")
            if PRELOAD_MODELS:
                logger.info("Embedding model preloaded; PostgreSQL initialized")
            else:
                logger.info("PostgreSQL initialized; AI models will load on first request")
            serve(app, host=SERVER_HOST, port=SERVER_PORT, threads=4)
    finally:
        logger.info("Shutting down server...")
        server_lifecycle.remove_pid_file()
        server_lifecycle.remove_ok_file()
        logger.info("Bye.")
