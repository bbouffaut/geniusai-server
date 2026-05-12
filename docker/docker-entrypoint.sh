#!/usr/bin/env bash
set -euo pipefail

DOTENV_FILENAME="${GENIUSAI_DOTENV_FILENAME:-}"

if [[ -z "$DOTENV_FILENAME" ]]; then
  echo "error: GENIUSAI_DOTENV_FILENAME is required" >&2
  exit 1
fi

if [[ "$DOTENV_FILENAME" = /* ]]; then
  DOTENV_FILE="$DOTENV_FILENAME"
else
  DOTENV_FILE="/config/$DOTENV_FILENAME"
fi

if [[ ! -f "$DOTENV_FILE" ]]; then
  echo "error: dotenv file not found: $DOTENV_FILE" >&2
  exit 1
fi

exec /app/run.sh --dotenv "$DOTENV_FILE" "$@"
