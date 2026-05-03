#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE

DB_PATH="/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db"
MODEL_CACHE_PATH=""
FETCH_MODELS_FLAG=""
DEBUG_FLAG=""

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
    --db-path)
      if [[ $# -lt 2 ]]; then
        echo "error: --db-path requires a value" >&2
        exit 1
      fi
      DB_PATH="$2"
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
      DB_PATH="$1"
      shift
      ;;
  esac
done

ensure_db_path_writable() {
  local path="$1"

  if [[ -e "$path" && ! -d "$path" ]]; then
    echo "error: --db-path must be a directory, got file: $path" >&2
    return 1
  fi

  if [[ ! -d "$path" ]]; then
    mkdir -p "$path" 2>/dev/null || {
      echo "error: failed to create db directory: $path" >&2
      return 1
    }
  fi

  if [[ ! -w "$path" ]]; then
    chmod u+w "$path" 2>/dev/null || true
  fi

  if [[ ! -w "$path" ]]; then
    echo "error: db directory is not writable: $path" >&2
    return 1
  fi
}

ensure_db_path_writable "$DB_PATH"

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

cmd=(uv run python src/geniusai_server.py --db-path "$DB_PATH")

if [[ -n "$FETCH_MODELS_FLAG" ]]; then
  cmd+=("$FETCH_MODELS_FLAG")
fi

if [[ -n "$DEBUG_FLAG" ]]; then
  cmd+=("$DEBUG_FLAG")
fi

if [[ -n "$MODEL_CACHE_PATH" ]]; then
  cmd+=(--model-cache-path "$MODEL_CACHE_PATH")
fi

"${cmd[@]}"
