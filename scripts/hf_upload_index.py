#!/usr/bin/env python3
"""Upload the round-3 mega plagiarism index to Hugging Face Datasets (free, 1TB).

Waits for the final build (/tmp/r3_FINAL_DONE), shards the SQLite DB if needed,
and uploads to a public HF dataset repo. Run with HF_TOKEN env var set.

Usage: HF_TOKEN=<set-via-env> .venv/bin/python scripts/hf_upload_index.py
"""
import os
import sys
import time
import hashlib
from pathlib import Path

# --- egress-proxy compat -----------------------------------------------------
# This VM's NO_PROXY contains bracketed IPv6 literals (e.g. [::1]) which the
# httpx2 build vendored by huggingface_hub cannot parse
# ("Invalid port: ':1]'"). Strip bracketed entries BEFORE huggingface_hub
# builds its HTTP client. HF is an external host, so it always uses the
# egress proxy anyway; nothing local is affected.
for _pk in ("NO_PROXY", "no_proxy"):
    _pv = os.environ.get(_pk)
    if _pv:
        os.environ[_pk] = ",".join(
            p for p in _pv.split(",") if "[" not in p and "]" not in p
        )

# --- single instance ----------------------------------------------------------
# Never run two upload daemons at once (cron + manual launches). Non-blocking;
# a second launcher exits quietly.
import fcntl
_HF_LOCK = open("/tmp/hf_upload_index.lock", "w")
try:
    fcntl.flock(_HF_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    print("[hf_upload] another upload daemon is already running - exiting", flush=True)
    sys.exit(0)

PROJ = Path("/home/hatch/workspace/ai-detector")
DB_PATH = PROJ / "data" / "mega_index" / "plag_index.db"
DONE_FLAG = Path("/tmp/r3_FINAL_DONE")
SHARD_SIZE = 2 * 1024 * 1024 * 1024  # 2GB shards
REPO_ID = os.environ.get("HF_REPO_ID", "nisarahmedkatyar/nak-plag-index-r3")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def wait_for_build(timeout_h=12):
    log("Waiting for final build (/tmp/r3_FINAL_DONE)...")
    waited = 0
    while not DONE_FLAG.exists():
        if not DB_PATH.exists():
            log("DB not found yet, waiting...")
        time.sleep(300)
        waited += 5 / 60
        if waited > timeout_h:
            log("TIMEOUT waiting for build")
            sys.exit(1)
    log("Build done flag found!")

def shard_db():
    size = DB_PATH.stat().st_size
    log(f"DB size: {size / 1e9:.2f} GB")
    out_dir = PROJ / "data" / "mega_index" / "hf_shards"
    out_dir.mkdir(exist_ok=True)
    manifest = out_dir / "manifest.txt"
    shards = []
    if size <= SHARD_SIZE:
        log("Single file, no sharding needed")
        return [DB_PATH]
    # Idempotent: if a complete shard set already exists, reuse it (crash-safe).
    n_expected = (size + SHARD_SIZE - 1) // SHARD_SIZE
    if manifest.exists():
        reuse, parts = True, []
        for i in range(n_expected):
            p = out_dir / f"plag_index.db.part{i:03d}"
            exp = SHARD_SIZE if i < n_expected - 1 else size - SHARD_SIZE * (n_expected - 1)
            if not p.exists() or p.stat().st_size != exp:
                reuse = False
                break
            parts.append(p)
        if reuse:
            log(f"Shard set already complete ({n_expected} parts) - skipping re-shard")
            parts.append(manifest)
            return parts
        log("Existing shards incomplete - re-sharding")
    log(f"Sharding into {SHARD_SIZE / 1e9:.0f}GB pieces...")
    with open(DB_PATH, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(SHARD_SIZE)
            if not chunk:
                break
            sp = out_dir / f"plag_index.db.part{idx:03d}"
            sp.write_bytes(chunk)
            shards.append(sp)
            log(f"  shard {idx}: {len(chunk) / 1e9:.2f}GB")
            idx += 1
    # manifest
    manifest = out_dir / "manifest.txt"
    with open(manifest, "w") as mf:
        for sp in shards:
            h = hashlib.sha256(sp.read_bytes()).hexdigest()[:16]
            mf.write(f"{sp.name} {sp.stat().st_size} {h}\n")
    shards.append(manifest)
    return shards

def upload(shards):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        log("ERROR: HF_TOKEN env var not set")
        sys.exit(1)
    api = HfApi(token=token)
    log(f"Creating repo {REPO_ID}...")
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    try:
        remote = {f.path: f.size for f in api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")}
    except Exception as e:
        log(f"Could not list remote files (fresh repo?): {e}")
        remote = {}
    for sp in shards:
        dest = f"index/{sp.name}"
        if remote.get(dest) == sp.stat().st_size:
            log(f"  already on HF, skipping: {sp.name}")
            continue
        log(f"Uploading {sp.name} ({sp.stat().st_size / 1e9:.2f}GB)...")
        api.upload_file(
            path_or_fileobj=str(sp),
            path_in_repo=dest,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        log(f"  done: {sp.name}")
    # dataset card
    card = f"""---
license: other
---
# NAK Plagiarism Index Round 3

Offline fingerprint reference library for the NAK AI Detector plagiarism module.
- Documents: ~11M (Common Crawl, Wikipedia, PubMed, Crossref, theses, DOAJ)
- Format: SQLite (winnowing fingerprints, k=5/w=4), sharded
- Built: 2026-10-14
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    log(f"DONE: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    wait_for_build()
    shards = shard_db()
    upload(shards)
