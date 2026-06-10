#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8010}"
HOST="${HOST:-127.0.0.1}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

if [ -x ".venv/bin/python" ]; then
  VENV_PYTHON=".venv/bin/python"
else
  VENV_PYTHON=".venv/Scripts/python.exe"
fi

if [ "${AIGC_SKIP_INSTALL:-0}" = "1" ]; then
  echo "Skipping dependency installation because AIGC_SKIP_INSTALL=1."
elif [ ! -f ".venv/.aigc_deps_installed" ]; then
  echo "Installing dependencies. This may take a few minutes on the first run..."
  "$VENV_PYTHON" -m pip install --upgrade pip
  "$VENV_PYTHON" -m pip install -e ".[dev]"
  echo "ok" > ".venv/.aigc_deps_installed"
else
  echo "Dependencies already installed. Delete .venv/.aigc_deps_installed to reinstall."
fi

if [ -z "${ARK_API_KEY:-}" ]; then
  export AIGC_DISABLE_LLM="${AIGC_DISABLE_LLM:-1}"
  export AIGC_DISABLE_VIDEO_MODEL="${AIGC_DISABLE_VIDEO_MODEL:-1}"
fi

export HOST
export PORT

echo "Starting AIGC Video System at http://${HOST}:${PORT}"
"$VENV_PYTHON" task_creation_demo_app.py
