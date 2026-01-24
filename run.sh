#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE

DB_PATH="/Volumes/Extreme SSD/Lightroom Plugins/lrgeniusAI-data/lrgenius.db"
FETCH_MODELS_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fetch-models)
      FETCH_MODELS_FLAG="--fetch-models"
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

uv run python src/geniusai_server.py --db-path "$DB_PATH" $FETCH_MODELS_FLAG
