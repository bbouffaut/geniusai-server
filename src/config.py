import argparse
import logging
import sys
import os
import re
import torch

# --- Model & Path Definitions ---
TEXT_EMBEDDING_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
TEXT_EMBEDDING_DIMENSION = 1024
TEXT_EMBEDDING_QUERY_INSTRUCTION = "Given a photo metadata search query, retrieve relevant photo metadata records"
TEXT_EMBEDDING_MAX_LENGTH = 8192


def _normalize_database_part(value):
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_.-")
    return normalized or "default"


def build_database_name(llm_id, embedding_id):
    base_name = f"{_normalize_database_part(llm_id)}-{_normalize_database_part(embedding_id)}"
    return base_name[:63].rstrip("_.-") or "geniusai"


# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='LrGenius Server')
parser.add_argument(
    '--db-path',
    type=str,
    help='Runtime data directory for logs and pid files (kept for backwards compatibility)',
)
parser.add_argument('--data-dir', type=str, help='Runtime data directory for logs and pid files')
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
    help='PostgreSQL database name to use. Defaults to <llm-id>-<embedding-id>.',
)

parser.add_argument('--debug', action='store_true', help='Enable debug mode with auto-reloading and debug log level')
parser.add_argument(
    '--debug-in-file',
    type=str,
    help='Write DEBUG logs and full LLM request payloads to this file',
)
parser.add_argument('--fetch-models', action='store_true', help='Fetch models from HF-Hub')
parser.add_argument('--model-cache-path', type=str, help='Path to store/load the embedding model cache')
parser.add_argument('--preload-models', action='store_true', help='Load embedding models during server startup')
args = parser.parse_args()

# --- Constants ---
DATA_DIR = os.path.abspath(os.path.expanduser(args.data_dir or args.db_path or os.getcwd()))
DB_PATH = DATA_DIR
POSTGRE_URL = args.postgre_url
POSTGRE_USER = args.postgre_user
POSTGRE_PASSWORD = args.postgre_password
POSTGRE_DATABASE_NAME = args.database_name or build_database_name(args.llm_id, args.embedding_id)
FETCH_MODELS = args.fetch_models
MODEL_CACHE_PATH = os.path.abspath(os.path.expanduser(args.model_cache_path)) if args.model_cache_path else None
PRELOAD_MODELS = args.preload_models
DEBUG_MODE = args.debug
DEBUG_IN_FILE_PATH = (
    os.path.abspath(os.path.expanduser(args.debug_in_file))
    if args.debug_in_file
    else None
)
DEBUG_IN_FILE = DEBUG_IN_FILE_PATH is not None

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

# Legacy names kept for compatibility with older scripts/imports.
CLIP_MODEL_NAME = TEXT_EMBEDDING_MODEL_ID
IMAGE_MODEL_ID = TEXT_EMBEDDING_MODEL_ID

LLM_BATCH_SIZE = 3  # Optimized batch size for better performance
LLM_TEMPERATURE = 0.2  # Reduced for faster, more deterministic responses

# --- Search Settings ---
# Cosine similarity threshold for normalized query/metadata text embeddings.
# Clients can override this per request with min_pertinence_score.
DEFAULT_MIN_PERTINENCE_SCORE = 0.35

# --- Prompts for Quality Scoring ---
# Optimized prompts for faster processing and better JSON compliance
QUALITY_SCORING_USER_PROMPT = """Rate this photo critically. Respond exclusively with JSON in this format:
{"overall_score": <1.0-10.0>, "composition_score": <1.0-10.0>, "lighting_score": <1.0-10.0>, "motiv_score": <1.0-10.0>, "colors_score": <1.0-10.0>, "emotion_score": <1.0-10.0>, "critique": "<brief specific critique>"}

Use the full 1-10 scale. Be critical and specific about weaknesses."""

QUALITY_SCORING_SYSTEM_PROMPT = """
"""

# Legacy aliases for backward compatibility with Qwen provider
USER_PROMPT = QUALITY_SCORING_USER_PROMPT
SYSTEM_PROMPT = QUALITY_SCORING_SYSTEM_PROMPT

# --- Prompts for Metadata Generation ---
METADATA_GENERATION_SYSTEM_PROMPT = """You are a professional photography analyst with expertise in object recognition and computer-generated image description. 
You also try to identify famous buildings and landmarks as well as the location where the photo was taken. 
Furthermore, you aim to specify animal and plant species as accurately as possible. 
You also describe objects—such as vehicle types and manufacturers—as specifically as you can."""

METADATA_GENERATION_USER_PROMPT_TEMPLATE = """Analyze the uploaded photo and generate the following data:
* Alt text (with context for screen readers)
* Image caption
* Image title
* Keywords

All results should be generated in {language}."""

BASE_PROMPT = "Analyze the uploaded photo and generate the following data:\n"
ALT_TEXT_PROMPT_ADDON = "* Alt text (with context for screen readers)\n"
CAPTION_TEXT_PROMPT_ADDON = "* Image caption\n"
TITLE_TEXT_PROMPT_ADDON = "* Image title\n"
KEYWORDS_TEXT_PROMPT_ADDON = "* the 5 most pertinent Keywords\n"
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

# --- Logger Setup ---
os.makedirs(DATA_DIR, exist_ok=True)
LOG_PATH = os.path.join(DATA_DIR, "lrgenius-server.log")

log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
root_log_level = logging.DEBUG if DEBUG_MODE or DEBUG_IN_FILE else logging.INFO

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
