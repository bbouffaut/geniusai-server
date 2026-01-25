#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="${ROOT_DIR}"
BUILD_DIR="${ROOT_DIR}/build"


if ! command -v uv >/dev/null 2>&1; then
  echo "Missing server build at ${SERVER_DIR}/dist and 'uv' is not installed."
  echo "Install uv, then re-run this script:"
  echo "  https://docs.astral.sh/uv/"
  exit 1
fi

echo "Building server with uv..."
pushd "${SERVER_DIR}" >/dev/null

if [[ -f pyproject.toml ]]; then
  uv sync
else
  echo "No pyproject.toml found in ${SERVER_DIR}."
  echo "This script expects an uv project with pyproject.toml."
  popd >/dev/null
  exit 1
fi

# Ensure pyinstaller is available for the build.
uv pip install pyinstaller
KMP_DUPLICATE_LIB_OK=TRUE uv run pyinstaller geniusai_server.spec --noconfirm

popd >/dev/null
