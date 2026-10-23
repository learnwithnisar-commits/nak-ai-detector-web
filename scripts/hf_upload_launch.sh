#!/bin/bash
# One-command (re)launcher for the round-3 HF upload daemon.
# Safe to run any time: pgrep guard + in-script flock + idempotent sharding
# + skip-files-already-on-HF. The HF write token is read from .hf_token
# (mode 600) and never printed.
set -u
cd /home/hatch/workspace/ai-detector || exit 1

if pgrep -f "hf_upload_index.py" >/dev/null 2>&1; then
  echo "upload daemon already running - nothing to do"
  exit 0
fi

if [ ! -s .hf_token ]; then
  echo "ERROR: .hf_token missing" >&2
  exit 1
fi

export HF_TOKEN="$(cat .hf_token)"
# Correct-case namespace: the HF account username is `Nisarahmedkatyar` (capital
# N); the lowercase default repo gets 403 from the Xet upload backend.
export HF_REPO_ID="Nisarahmedkatyar/nak-plag-index-r3"
# Strip bracketed IPv6 NO_PROXY entries: huggingface_hub's vendored httpx2
# cannot parse them ("Invalid port: ':1]'"). HF is external -> egress proxy.
_clean="$(echo "${NO_PROXY:-}" | tr ',' '\n' | grep -v '\[' | grep -v '\]' | paste -sd, -)"
export NO_PROXY="$_clean" no_proxy="$_clean"

nohup .venv/bin/python scripts/hf_upload_index.py >> data/mega_index/hf_upload.log 2>&1 &
echo "launched hf_upload_index.py pid $!"
