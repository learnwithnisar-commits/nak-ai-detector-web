#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Python deps live in /opt/venv (PEP 668 forbids system-wide pip install
# on Debian's apt Python); Dolos CLI in /opt/npm-global/bin; Node in
# /opt/node/bin (Dolos's shebang needs `node`).
export PATH="/opt/venv/bin:/opt/npm-global/bin:/opt/node/bin:$PATH"
# Single worker to stay tiny on the free plan; $PORT is provided by Railway.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
