#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Dolos CLI (code-check tab) lives in /opt/npm-global/bin and Node in
# /opt/node/bin (installed by the nixpacks install phase, not nixpkgs);
# keep both on PATH for the app (Dolos's shebang needs `node`).
export PATH="/opt/npm-global/bin:/opt/node/bin:$PATH"
# Single worker to stay tiny on the free plan; $PORT is provided by Railway.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
