#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Dolos CLI (code-check tab) lives in /opt/npm-global/bin when installed
# via the nixpacks install phase; keep it on PATH for the app.
export PATH="/opt/npm-global/bin:$PATH"
# Single worker to stay tiny on the free plan; $PORT is provided by Railway.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
