#!/usr/bin/env bash
# migrate.sh — Re-index photo embeddings from one database to another.
#
# Source database and connection details are read from a dotenv file
# (via GENIUSAI_DATABASE_NAME, GENIUSAI_POSTGRES_URL, etc.).
#
# Usage:
#   ./migrate.sh --dotenv <file> --target-db <db> --target-model <model> [options]
#
# Required:
#   --dotenv <FILE>          Load source database config from this dotenv file
#   --target-db <DB>         Target PostgreSQL database name (created if absent)
#   --target-model <MODEL>   Embedding model key for the target DB (e.g. bge-m3, qwen3-0.6b)
#
# Optional:
#   --batch-size <N>         Photos per GPU batch (default: 32)
#   --fetch-models           Download model from HuggingFace Hub if not cached
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATE_SCRIPT="${ROOT_DIR}/src/migrate.py"

export KMP_DUPLICATE_LIB_OK=TRUE

DOTENV_FILE=""
TARGET_DB=""
TARGET_MODEL=""
FETCH_MODELS_FLAG=""
BATCH_SIZE="32"

# First pass: extract --dotenv so env vars are set before the main parse.
_pre_args=("$@")
while [[ ${#_pre_args[@]} -gt 0 ]]; do
  case "${_pre_args[0]}" in
    --dotenv|--env-file)
      if [[ ${#_pre_args[@]} -lt 2 ]]; then
        echo "error: ${_pre_args[0]} requires a value" >&2; exit 1
      fi
      DOTENV_FILE="${_pre_args[1]}"
      _pre_args=("${_pre_args[@]:2}")
      ;;
    *) _pre_args=("${_pre_args[@]:1}") ;;
  esac
done

load_dotenv() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "error: dotenv file not found: $path" >&2; exit 1
  fi
  set -a
  # shellcheck source=/dev/null
  . "$path"
  set +a
}

if [[ -n "$DOTENV_FILE" ]]; then
  load_dotenv "$DOTENV_FILE"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-db)
      [[ $# -lt 2 ]] && { echo "error: --target-db requires a value" >&2; exit 1; }
      TARGET_DB="$2"; shift 2 ;;
    --target-model)
      [[ $# -lt 2 ]] && { echo "error: --target-model requires a value" >&2; exit 1; }
      TARGET_MODEL="$2"; shift 2 ;;
    --batch-size)
      [[ $# -lt 2 ]] && { echo "error: --batch-size requires a value" >&2; exit 1; }
      BATCH_SIZE="$2"; shift 2 ;;
    --fetch-models)
      FETCH_MODELS_FLAG="--fetch-models"; shift ;;
    --dotenv|--env-file)
      [[ $# -lt 2 ]] && { echo "error: $1 requires a value" >&2; exit 1; }
      shift 2 ;;  # already handled in the first pass
    *)
      echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${GENIUSAI_DATABASE_NAME:-}" ]]; then
  echo "error: GENIUSAI_DATABASE_NAME is not set." >&2
  echo "       Provide it via --dotenv <file> or export it before calling this script." >&2
  exit 1
fi
if [[ -z "$TARGET_DB" ]]; then
  echo "error: --target-db is required" >&2; exit 1
fi
if [[ -z "$TARGET_MODEL" ]]; then
  echo "error: --target-model is required" >&2; exit 1
fi

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_CMD=("${ROOT_DIR}/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv --project "$ROOT_DIR" run python)
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=(python3)
else
  echo "error: no python interpreter found" >&2; exit 1
fi

cmd=(
  "${PYTHON_CMD[@]}" "$MIGRATE_SCRIPT"
  --target-db    "$TARGET_DB"
  --target-model "$TARGET_MODEL"
  --batch-size   "$BATCH_SIZE"
)

[[ -n "$FETCH_MODELS_FLAG" ]] && cmd+=("$FETCH_MODELS_FLAG")

exec "${cmd[@]}"
