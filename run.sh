#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_SCRIPT="${ROOT_DIR}/src/geniusai_server.py"

export KMP_DUPLICATE_LIB_OK=TRUE

# On macOS/exFAT (and network) volumes, macOS creates AppleDouble "._*" sidecar
# files on copy/write. Binary "._*.py" files inside the venv break libraries that
# scan and read .py files by directory (e.g. transformers -> UnicodeDecodeError).
# Prune them from the venv and source tree before starting the server. Scoped to
# .venv and src to stay fast (skips .git). Skipped when GENIUSAI_SKIP_APPLEDOUBLE_CLEAN is truthy.
if [[ "$(uname -s)" == "Darwin" && "${GENIUSAI_SKIP_APPLEDOUBLE_CLEAN:-}" != "1" ]]; then
  find "${ROOT_DIR}/.venv" "${ROOT_DIR}/src" -name '._*' -type f -delete 2>/dev/null || true
fi

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

is_truthy() {
  case "${1:-}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
      return 0
      ;;
    0|[Ff][Aa][Ll][Ss][Ee]|[Nn][Oo]|[Oo][Ff][Ff]|"")
      return 1
      ;;
    *)
      echo "error: invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}

ensure_dir_writable() {
  local path="$1"
  local flag_name="$2"

  if [[ -e "$path" && ! -d "$path" ]]; then
    echo "error: ${flag_name} must be a directory, got file: $path" >&2
    return 1
  fi

  if [[ ! -d "$path" ]]; then
    mkdir -p "$path" 2>/dev/null || {
      echo "error: failed to create directory for ${flag_name}: $path" >&2
      return 1
    }
  fi

  if [[ ! -w "$path" ]]; then
    chmod u+w "$path" 2>/dev/null || true
  fi

  if [[ ! -w "$path" ]]; then
    echo "error: directory for ${flag_name} is not writable: $path" >&2
    return 1
  fi
}

if [[ -n "$DOTENV_FILE" ]]; then
  load_dotenv "$DOTENV_FILE"
fi

POSTGRE_URL="${POSTGRE_URL:-${POSTGRES_URL:-${GENIUSAI_POSTGRES_URL:-${GENIUSAI_POSTGRE_URL:-postgresql://localhost:5432/postgres}}}}"
POSTGRE_USER="${POSTGRE_USER:-${POSTGRES_USER:-${GENIUSAI_POSTGRES_USER:-}}}"
POSTGRE_PASSWORD="${POSTGRE_PASSWORD:-${POSTGRES_PASSWORD:-${GENIUSAI_POSTGRES_PASSWORD:-}}}"
DATABASE_NAME="${GENIUSAI_DATABASE_NAME:-}"
DATA_DIR="${GENIUSAI_DATA_DIR:-}"
UPLOAD_TEMP_DIR="${GENIUSAI_UPLOAD_TEMP_DIR:-}"
SERVER_HOST="${GENIUSAI_SERVER_HOST:-}"
SERVER_PORT="${GENIUSAI_SERVER_PORT:-}"
MODEL_CACHE_PATH="${MODEL_CACHE_PATH:-./cache}"
FETCH_MODELS_FLAG=""
DEBUG_LEVEL_VALUE="${GENIUSAI_DEBUG_LEVEL:-}"
DEBUG_IN_FILE_PATH="${GENIUSAI_DEBUG_IN_FILE:-}"
PRELOAD_MODELS_FLAG="--preload-models"
MCP_HOST="${GENIUSAI_MCP_SERVER_HOST:-}"
MCP_PORT="${GENIUSAI_MCP_SERVER_PORT:-}"
MCP_FLAG=""

if [[ -n "${GENIUSAI_MCP_ENABLED:-}" ]]; then
  if is_truthy "${GENIUSAI_MCP_ENABLED}"; then
    MCP_FLAG="--mcp"
  else
    MCP_FLAG="--no-mcp"
  fi
fi

if [[ -n "${GENIUSAI_FETCH_MODELS:-}" ]] && is_truthy "${GENIUSAI_FETCH_MODELS}"; then
  FETCH_MODELS_FLAG="--fetch-models"
fi

if [[ -n "${GENIUSAI_PRELOAD_MODELS:-}" ]]; then
  if is_truthy "${GENIUSAI_PRELOAD_MODELS}"; then
    PRELOAD_MODELS_FLAG="--preload-models"
  else
    PRELOAD_MODELS_FLAG=""
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fetch-models)
      FETCH_MODELS_FLAG="--fetch-models"
      shift
      ;;
    --debug-level)
      if [[ $# -lt 2 ]]; then
        echo "error: --debug-level requires a value (0, 1, or 2)" >&2
        exit 1
      fi
      DEBUG_LEVEL_VALUE="$2"
      shift 2
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
    --host)
      if [[ $# -lt 2 ]]; then
        echo "error: --host requires a value" >&2
        exit 1
      fi
      SERVER_HOST="$2"
      shift 2
      ;;
    --port)
      if [[ $# -lt 2 ]]; then
        echo "error: --port requires a value" >&2
        exit 1
      fi
      SERVER_PORT="$2"
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
    --mcp)
      MCP_FLAG="--mcp"
      shift
      ;;
    --no-mcp)
      MCP_FLAG="--no-mcp"
      shift
      ;;
    --mcp-host)
      if [[ $# -lt 2 ]]; then
        echo "error: --mcp-host requires a value" >&2
        exit 1
      fi
      MCP_HOST="$2"
      shift 2
      ;;
    --mcp-port)
      if [[ $# -lt 2 ]]; then
        echo "error: --mcp-port requires a value" >&2
        exit 1
      fi
      MCP_PORT="$2"
      shift 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$DATA_DIR" ]]; then
  ensure_dir_writable "$DATA_DIR" "GENIUSAI_DATA_DIR"
fi

if [[ -n "$UPLOAD_TEMP_DIR" ]]; then
  ensure_dir_writable "$UPLOAD_TEMP_DIR" "GENIUSAI_UPLOAD_TEMP_DIR"
fi

if [[ -n "$MODEL_CACHE_PATH" ]]; then
  ensure_dir_writable "$MODEL_CACHE_PATH" "--model-cache-path"
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
  echo "error: no python interpreter found" >&2
  exit 1
fi

cmd=("${PYTHON_CMD[@]}" "$SERVER_SCRIPT" --postgre-url "$POSTGRE_URL")

if [[ -n "$DATABASE_NAME" ]]; then
  cmd+=(--database-name "$DATABASE_NAME")
fi

if [[ -n "$SERVER_HOST" ]]; then
  cmd+=(--host "$SERVER_HOST")
fi

if [[ -n "$SERVER_PORT" ]]; then
  cmd+=(--port "$SERVER_PORT")
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

if [[ -n "$DEBUG_LEVEL_VALUE" ]]; then
  cmd+=(--debug-level "$DEBUG_LEVEL_VALUE")
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

if [[ -n "$MCP_FLAG" ]]; then
  cmd+=("$MCP_FLAG")
fi

if [[ -n "$MCP_HOST" ]]; then
  cmd+=(--mcp-host "$MCP_HOST")
fi

if [[ -n "$MCP_PORT" ]]; then
  cmd+=(--mcp-port "$MCP_PORT")
fi

exec "${cmd[@]}"
