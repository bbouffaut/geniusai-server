#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE

DB_PATH="$HOME/Pictures/Lightroom/lrgenius.db"
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

uv run python src/geniusai_server.py --db-path "$DB_PATH" $FETCH_MODELS_FLAG
