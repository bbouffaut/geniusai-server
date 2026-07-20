import argparse
import logging
import sys
import os
import torch

# ---------------------------------------------------------------------------
# Supported embedding models
# Each entry drives every aspect of how texts are embedded: which HuggingFace
# model to load, the vector dimension stored in pgvector, the pooling strategy,
# the token budget, and whether queries should carry an instruction prefix.
#
# pooling values:
#   "last_token" – last non-padding hidden state (decoder-style, e.g. Qwen3)
#   "cls"        – first token / CLS hidden state (encoder-style, e.g. BGE-M3)
#   "mean"       – attention-masked mean of all token hidden states (e.g. E5)
#
# query_instruction: prepended as "Instruct: …\nQuery: …" for instruction-
#   aware models; set to None for models that use plain text for both sides.
# ---------------------------------------------------------------------------
EMBEDDING_MODELS = {
    "qwen3-0.6b": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 1024,
        "pooling": "last_token",
        "max_length": 512,
        "query_instruction": (
            "Given a photo metadata search query, retrieve relevant photo metadata records"
        ),
    },
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dimension": 1024,
        "pooling": "cls",
        "max_length": 8192,      # BGE-M3 supports long contexts – no more truncation
        "query_instruction": None,  # plain text for both queries and documents
    },
}



# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='LrGenius Server')
parser.add_argument(
    '--postgre-url',
    '--postgres-url',
    dest='postgre_url',
    type=str,
    default=os.environ.get("GENIUSAI_POSTGRES_URL", "postgresql://localhost:5432/postgres"),
    help='PostgreSQL connection URL used to create/connect databases',
)
parser.add_argument(
    '--postgre-user',
    '--postgres-user',
    dest='postgre_user',
    type=str,
    default=os.environ.get("GENIUSAI_POSTGRES_USER"),
    help='PostgreSQL username. Overrides any user embedded in --postgre-url.',
)
parser.add_argument(
    '--postgre-password',
    '--postgres-password',
    dest='postgre_password',
    type=str,
    default=os.environ.get("GENIUSAI_POSTGRES_PASSWORD"),
    help='PostgreSQL password. Overrides any password embedded in --postgre-url.',
)
parser.add_argument(
    '--database-name',
    type=str,
    default=os.environ.get("GENIUSAI_DATABASE_NAME"),
    help='PostgreSQL database name (required — set via GENIUSAI_DATABASE_NAME or this flag).',
)
parser.add_argument(
    '--host',
    type=str,
    default='127.0.0.1',
    help='Host interface to bind the HTTP server to',
)
parser.add_argument(
    '--port',
    type=int,
    default=19819,
    help='Port to listen on',
)

parser.add_argument('--debug', action='store_true', help='Enable debug mode with auto-reloading and debug log level')
parser.add_argument(
    '--debug-level',
    type=int,
    default=int(os.environ.get("GENIUSAI_DEBUG_LEVEL") or 0),
    choices=[0, 1, 2],
    dest='debug_level',
    help='Debug verbosity: 0=off, 1=brief messages, 2=messages with data dumps',
)
parser.add_argument(
    '--debug-in-file',
    type=str,
    help='Write DEBUG logs and full LLM request payloads to this file',
)
parser.add_argument('--fetch-models', action='store_true', help='Fetch models from HF-Hub')
parser.add_argument('--model-cache-path', type=str, help='Path to store/load the embedding model cache')
parser.add_argument('--preload-models', action='store_true', help='Load embedding models during server startup')
parser.add_argument(
    '--embedding-model',
    dest='embedding_model',
    type=str,
    default=os.environ.get("GENIUSAI_EMBEDDING_MODEL"),
    choices=list(EMBEDDING_MODELS.keys()),
    help=(
        'Embedding model to use (required — set via GENIUSAI_EMBEDDING_MODEL or this flag). '
        'Each model gets its own database so switching models requires re-indexing. '
        + " | ".join(f"{k}: {v['model_id']}" for k, v in EMBEDDING_MODELS.items())
    ),
)
args = parser.parse_args()

if not args.embedding_model:
    parser.error(
        "Embedding model is required. "
        "Set GENIUSAI_EMBEDDING_MODEL in your .env file or pass --embedding-model. "
        "Supported models: " + ", ".join(EMBEDDING_MODELS.keys())
    )
if args.embedding_model not in EMBEDDING_MODELS:
    parser.error(
        f"Unknown embedding model: '{args.embedding_model}'. "
        "Use the short key, not the full model ID. "
        "Supported models: " + ", ".join(f"{k} ({v['model_id']})" for k, v in EMBEDDING_MODELS.items())
    )

# Resolve the selected embedding model config.
# All downstream constants are derived from this single selection.
_selected_embedding = EMBEDDING_MODELS[args.embedding_model]
TEXT_EMBEDDING_MODEL_ID       = _selected_embedding["model_id"]
TEXT_EMBEDDING_DIMENSION      = _selected_embedding["dimension"]
TEXT_EMBEDDING_POOLING        = _selected_embedding["pooling"]
TEXT_EMBEDDING_MAX_LENGTH     = _selected_embedding["max_length"]
TEXT_EMBEDDING_QUERY_INSTRUCTION = _selected_embedding["query_instruction"]


def _resolve_data_dir():
    configured_data_dir = os.environ.get("GENIUSAI_DATA_DIR")
    return os.path.abspath(os.path.expanduser(configured_data_dir or os.getcwd()))


def _resolve_upload_temp_dir(data_dir):
    configured_upload_temp_dir = os.environ.get("GENIUSAI_UPLOAD_TEMP_DIR")
    base_path = configured_upload_temp_dir or os.path.join(data_dir, "uploads-temp")
    return os.path.abspath(os.path.expanduser(base_path))


# --- Constants ---
POSTGRE_URL = args.postgre_url
POSTGRE_USER = args.postgre_user
POSTGRE_PASSWORD = args.postgre_password
if not args.database_name:
    parser.error(
        "Database name is required. "
        "Set GENIUSAI_DATABASE_NAME in your .env file or pass --database-name."
    )
POSTGRE_DATABASE_NAME = args.database_name
FETCH_MODELS = args.fetch_models
MODEL_CACHE_PATH = os.path.abspath(os.path.expanduser(args.model_cache_path)) if args.model_cache_path else None
PRELOAD_MODELS = args.preload_models
DEBUG_MODE = args.debug
DEBUG_LEVEL = args.debug_level
DEBUG_IN_FILE_PATH = (
    os.path.abspath(os.path.expanduser(args.debug_in_file))
    if args.debug_in_file
    else None
)
DEBUG_IN_FILE = DEBUG_IN_FILE_PATH is not None
SERVER_HOST = args.host
SERVER_PORT = args.port
DATA_DIR = _resolve_data_dir()
UPLOAD_TEMP_DIR = _resolve_upload_temp_dir(DATA_DIR)
LOG_PATH = os.path.join(DATA_DIR, "lrgenius-server.log")
PID_FILE_PATH = os.path.join(DATA_DIR, "lrgenius-server.pid")
OK_FILE_PATH = os.path.join(DATA_DIR, "lrgenius-server.OK")

# --- Code Style Preferences ---
USE_EMOJIS = False  # Set to False to avoid emojis in logs and output

def format_log_message(message: str, emoji: str = "") -> str:
    """
    Format log message with optional emoji based on USE_EMOJIS setting.
    Use this function to ensure consistent logging style.
    """
    if USE_EMOJIS and emoji:
        return f"{emoji} {message}"
    return message

# --- Model & Path Definitions ---
# Platform-specific device selection:
# - macOS: Use Metal GPU (MPS) if available
# - Windows: CPU-only for optimized binary size and compatibility
# - Linux: CUDA if available, otherwise CPU
if sys.platform == "darwin":  # macOS
    TORCH_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
elif sys.platform == "win32":  # Windows
    TORCH_DEVICE = "cpu"
else:  # Linux and other Unix-like platforms
    TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Legacy alias kept for compatibility with older scripts/imports.
IMAGE_MODEL_ID = TEXT_EMBEDDING_MODEL_ID

# Human-readable label shown in logs and status endpoints.
EMBEDDING_MODEL_KEY = args.embedding_model

LLM_BATCH_SIZE = 3  # Optimized batch size for better performance

# --- Search Settings ---
# Cosine similarity threshold for normalized query/metadata text embeddings.
# Clients can override this per request with min_pertinence_score.
DEFAULT_MIN_PERTINENCE_SCORE = 0.35

# --- Prompts for Quality Scoring ---
# Optimized prompts for faster processing and better JSON compliance
QUALITY_SCORING_USER_PROMPT = """Rate this photo critically. Respond exclusively with JSON in this format:
{"overall_score": <1.0-10.0>, "composition_score": <1.0-10.0>, "lighting_score": <1.0-10.0>, "motiv_score": <1.0-10.0>, "colors_score": <1.0-10.0>, "emotion_score": <1.0-10.0>, "critique": "<brief specific critique>"}

Use the full 1-10 scale. Be critical and specific about weaknesses."""

QUALITY_SCORING_SYSTEM_PROMPT = """You are a professional photography critic. Evaluate technical and artistic image quality with concise, specific, and critical judgment."""

# Legacy aliases for backward compatibility with Qwen provider
USER_PROMPT = QUALITY_SCORING_USER_PROMPT
SYSTEM_PROMPT = QUALITY_SCORING_SYSTEM_PROMPT

# --- Prompts for Metadata Generation ---
METADATA_GENERATION_SYSTEM_PROMPT = """You are a professional photography analyst with expertise in object recognition and computer-generated image description. 
You also try to identify famous buildings and landmarks as well as the location where the photo was taken. 
Furthermore, you aim to specify animal and plant species as accurately as possible. 
You also describe objects—such as vehicle types and manufacturers—as specifically as you can."""

# ---------------------------------------------------------------------------
# Caption (= IPTC Caption/Abstract in Lightroom) is the primary semantic field.
# It must be detailed enough to serve as the main prose-embedding anchor so
# that semantic search can match fine-grained queries about subjects, scene,
# lighting, mood, and location.
#
# Alt text is kept brief and accessibility-focused (screen-reader context);
# it is NOT used for embedding.
#
# Title is a short display heading; it is NOT used for embedding.
# ---------------------------------------------------------------------------
METADATA_GENERATION_USER_PROMPT_TEMPLATE = """Analyze the uploaded photo and generate the following data:
* Caption: Write a detailed description (2-4 sentences) of the scene covering subjects (people, animals, objects), setting and background, colors and lighting, mood and atmosphere, and any identifiable locations or landmarks.
* Alt text: A concise visual description for screen readers (1-2 sentences).
* Title: A short, descriptive title for the photo.
* Keywords

All results should be generated in {language}."""

BASE_PROMPT = "Analyze the uploaded photo and generate the following data:\n"
ALT_TEXT_PROMPT_ADDON = "* Alt text: A concise visual description for screen readers (1-2 sentences).\n"
CAPTION_TEXT_PROMPT_ADDON = (
    "* Caption: Write a detailed description (2-4 sentences) of the scene covering: "
    "subjects (people, animals, objects), setting and background, colors and lighting, "
    "mood and atmosphere, and any identifiable locations or landmarks.\n"
)
TITLE_TEXT_PROMPT_ADDON = "* Title: A short, descriptive title for the photo.\n"
KEYWORDS_TEXT_PROMPT_ADDON = "* Keywords: the most relevant keywords.\n"
LANGUAGE_TEXT_INSTRUCTION_ADDON = "\n\nAll results should be generated in %s."


# --- LLM Provider Configuration ---
# Environment variables or default values for external LLM providers

# Default provider selection (can be overridden per request)
DEFAULT_METADATA_PROVIDER = "ollama"

# Metadata Generation Settings
DEFAULT_METADATA_LANGUAGE = "English"
DEFAULT_KEYWORD_CATEGORIES = [
    "People", "Activities", "Objects", "Locations", "Events", 
    "Colors", "Mood", "Technical", "Composition"
]

LMSTUDIO_HOST = "localhost:1234"
OLLAMA_BASE_URL = "http://localhost:11434"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_API_VERSION = "2023-06-01"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
root_log_level = logging.DEBUG if DEBUG_MODE or DEBUG_IN_FILE else logging.INFO

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_TEMP_DIR, exist_ok=True)
main_file_handler = logging.FileHandler(LOG_PATH, encoding='utf-8')
main_file_handler.setLevel(log_level)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(log_level)

handlers = [
    main_file_handler,
    stream_handler
]

debug_file_handler = None
if DEBUG_IN_FILE_PATH:
    debug_log_dir = os.path.dirname(DEBUG_IN_FILE_PATH)
    if debug_log_dir:
        os.makedirs(debug_log_dir, exist_ok=True)
    debug_file_handler = logging.FileHandler(DEBUG_IN_FILE_PATH, encoding='utf-8')
    debug_file_handler.setLevel(logging.DEBUG)
    handlers.append(debug_file_handler)

# Configure logging with UTF-8 encoding to handle Unicode characters
logging.basicConfig(
    level=root_log_level,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=handlers
)
logger = logging.getLogger("geniusai-server")

raw_debug_logger = logging.getLogger("geniusai-server.raw-debug")
raw_debug_logger.setLevel(logging.DEBUG)
raw_debug_logger.propagate = False
if debug_file_handler is not None:
    raw_debug_logger.addHandler(debug_file_handler)
