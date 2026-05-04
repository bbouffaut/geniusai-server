import os
import sys
from flask import Flask, jsonify
from waitress import serve
import datetime

# Import modularized components
from config import logger, args, DEBUG_IN_FILE_PATH, PRELOAD_MODELS
logger.info("Imported config")

# Lazy import server_lifecycle to speed up startup
import server_lifecycle
logger.info("Imported server_lifecycle")

# Import blueprints only (services are imported by routes when needed)
from routes_index import index_bp
from routes_search import search_bp
from routes_server import server_bp
from routes_import import import_bp

app = Flask(__name__)
logger.info("Flask app created")

# Register blueprints
app.register_blueprint(index_bp)
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
    logger.info(f"Database: {args.db_path}")
    if DEBUG_IN_FILE_PATH:
        logger.warning(
            "Raw debug logging enabled; unredacted LLM request payloads "
            f"will be written to {DEBUG_IN_FILE_PATH}"
        )
    logger.info("=" * 60)

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
    logger.info("✓ Server initialized and ready to accept connections")
    
    # Write PID for lifecycle management
    server_lifecycle.write_pid_file()
    
    try:
        if args.debug:
            logger.info("Starting Flask development server in debug mode on http://127.0.0.1:19819")
            app.run(debug=True, host="127.0.0.1", port=19819)
        else:
            logger.info("Starting production server on http://127.0.0.1:19819")
            if PRELOAD_MODELS:
                logger.info("Embedding model preloaded; ChromaDB will load on first request")
            else:
                logger.info("Heavy modules (ChromaDB, AI models) will load on first request")
            serve(app, host="127.0.0.1", port=19819, threads=4)
    finally:
        logger.info("Shutting down server...")
        server_lifecycle.remove_pid_file()
        server_lifecycle.remove_ok_file()
        logger.info("Bye.")
