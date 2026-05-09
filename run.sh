#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE

DOTENV_FILE=""
args=("$@")

while [[ ${#args[@]} -gt 0 ]]; do
  case "${args[0]}" in
    --dotenv|--env-file)
      if [[ ${#args[@]} -lt 2 ]]; then
        echo "error: ${args[0]} requires a value" >&2
        exit 1
      fi
      DOTENV_FILE="${args[1]}"
      args=("${args[@]:2}")
      ;;
    *)
      args=("${args[@]:1}")
      ;;
  esac
done

load_dotenv() {
  local path="$1"

  if [[ ! -f "$path" ]]; then
    echo "error: dotenv file not found: $path" >&2
    exit 1
  fi

  set -a
  # shellcheck source=/dev/null
  . "$path"
  set +a
}

if [[ -n "$DOTENV_FILE" ]]; then
  load_dotenv "$DOTENV_FILE"
fi

POSTGRE_URL="${POSTGRE_URL:-${POSTGRES_URL:-${GENIUSAI_POSTGRES_URL:-${GENIUSAI_POSTGRE_URL:-postgresql://localhost:5432/postgres}}}}"
POSTGRE_USER="${POSTGRE_USER:-${POSTGRES_USER:-${GENIUSAI_POSTGRES_USER:-}}}"
POSTGRE_PASSWORD="${POSTGRE_PASSWORD:-${POSTGRES_PASSWORD:-${GENIUSAI_POSTGRES_PASSWORD:-}}}"
DATABASE_NAME="${GENIUSAI_DATABASE_NAME:-}"
DATA_DIR="${GENIUSAI_DATA_DIR:-}"
LLM_ID="${GENIUSAI_LLM_ID:-ollama}"
EMBEDDING_ID="${GENIUSAI_EMBEDDING_ID:-Qwen/Qwen3-Embedding-0.6B}"
MODEL_CACHE_PATH="${MODEL_CACHE_PATH:-./cache}"
FETCH_MODELS_FLAG=""
DEBUG_FLAG=""
DEBUG_IN_FILE_PATH=""
PRELOAD_MODELS_FLAG="--preload-models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fetch-models)
      FETCH_MODELS_FLAG="--fetch-models"
      shift
      ;;
    --debug)
      DEBUG_FLAG="--debug"
      shift
      ;;
    --debug-in-file)
      if [[ $# -lt 2 ]]; then
        echo "error: --debug-in-file requires a value" >&2
        exit 1
      fi
      DEBUG_IN_FILE_PATH="$2"
      shift 2
      ;;
    --preload-models)
      PRELOAD_MODELS_FLAG="--preload-models"
      shift
      ;;
    --lazy-load-models)
      PRELOAD_MODELS_FLAG=""
      shift
      ;;
    --dotenv|--env-file)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value" >&2
        exit 1
      fi
      DOTENV_FILE="$2"
      shift 2
      ;;
    --data-dir|--db-path)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value" >&2
        exit 1
      fi
      DATA_DIR="$2"
      shift 2
      ;;
    --postgre-url|--postgres-url)
      if [[ $# -lt 2 ]]; then
        echo "error: --postgre-url requires a value" >&2
        exit 1
      fi
      POSTGRE_URL="$2"
      shift 2
      ;;
    --postgre-user|--postgres-user)
      if [[ $# -lt 2 ]]; then
        echo "error: --postgre-user requires a value" >&2
        exit 1
      fi
      POSTGRE_USER="$2"
      shift 2
      ;;
    --postgre-password|--postgres-password)
      if [[ $# -lt 2 ]]; then
        echo "error: --postgre-password requires a value" >&2
        exit 1
      fi
      POSTGRE_PASSWORD="$2"
      shift 2
      ;;
    --database-name)
      if [[ $# -lt 2 ]]; then
        echo "error: --database-name requires a value" >&2
        exit 1
      fi
      DATABASE_NAME="$2"
      shift 2
      ;;
    --model-cache-path)
      if [[ $# -lt 2 ]]; then
        echo "error: --model-cache-path requires a value" >&2
        exit 1
      fi
      MODEL_CACHE_PATH="$2"
      shift 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ensure_data_dir_writable() {
  local path="$1"

  if [[ -e "$path" && ! -d "$path" ]]; then
    echo "error: --data-dir must be a directory, got file: $path" >&2
    return 1
  fi

  if [[ ! -d "$path" ]]; then
    mkdir -p "$path" 2>/dev/null || {
      echo "error: failed to create data directory: $path" >&2
      return 1
    }
  fi

  if [[ ! -w "$path" ]]; then
    chmod u+w "$path" 2>/dev/null || true
  fi

  if [[ ! -w "$path" ]]; then
    echo "error: data directory is not writable: $path" >&2
    return 1
  fi
}

ensure_data_dir_writable "$MODEL_CACHE_PATH"

if [[ -n "$MODEL_CACHE_PATH" ]]; then
  if [[ -e "$MODEL_CACHE_PATH" && ! -d "$MODEL_CACHE_PATH" ]]; then
    echo "error: --model-cache-path must be a directory, got file: $MODEL_CACHE_PATH" >&2
    exit 1
  fi

  if [[ ! -d "$MODEL_CACHE_PATH" ]]; then
    mkdir -p "$MODEL_CACHE_PATH" 2>/dev/null || {
      echo "error: failed to create model cache directory: $MODEL_CACHE_PATH" >&2
      exit 1
    }
  fi

  if [[ ! -w "$MODEL_CACHE_PATH" ]]; then
    chmod u+w "$MODEL_CACHE_PATH" 2>/dev/null || true
  fi

  if [[ ! -w "$MODEL_CACHE_PATH" ]]; then
    echo "error: model cache directory is not writable: $MODEL_CACHE_PATH" >&2
    exit 1
  fi
fi

cmd=(
  uv run python src/geniusai_server.py
  --postgre-url "$POSTGRE_URL"
  --database-name "$DATABASE_NAME"
)

if [[ -n "$DATA_DIR" ]]; then
  cmd+=(--data-dir "$DATA_DIR")
fi

if [[ -n "$FETCH_MODELS_FLAG" ]]; then
  cmd+=("$FETCH_MODELS_FLAG")
fi

if [[ -n "$POSTGRE_USER" ]]; then
  cmd+=(--postgre-user "$POSTGRE_USER")
fi

if [[ -n "$POSTGRE_PASSWORD" ]]; then
  cmd+=(--postgre-password "$POSTGRE_PASSWORD")
fi

if [[ -n "$DEBUG_FLAG" ]]; then
  cmd+=("$DEBUG_FLAG")
fi

if [[ -n "$DEBUG_IN_FILE_PATH" ]]; then
  cmd+=(--debug-in-file "$DEBUG_IN_FILE_PATH")
fi

if [[ -n "$PRELOAD_MODELS_FLAG" ]]; then
  cmd+=("$PRELOAD_MODELS_FLAG")
fi

if [[ -n "$MODEL_CACHE_PATH" ]]; then
  cmd+=(--model-cache-path "$MODEL_CACHE_PATH")
fi

"${cmd[@]}"
