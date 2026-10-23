#!/usr/bin/env python3
"""Upload the web academic plagiarism index (plag_web.db, ~603MB) to Hugging Face.

One file -> new dataset repo Nisarahmedkatyar/nak-plag-web-index.
The Railway app boot-downloads it via WEB_INDEX_URL:
  https://huggingface.co/datasets/Nisarahmedkatyar/nak-plag-web-index/resolve/main/plag_web.db

Idempotent: skips upload if the file already exists on the repo.
Token: read from .hf_token (mode 600), never printed.
"""
import os
import sys
import time

PROJ = os.path.expanduser("~/workspace/ai-detector")
os.chdir(PROJ)

# Strip bracketed IPv6 NO_PROXY entries: huggingface_hub's vendored httpx
# cannot parse them ("Invalid port: ':1]'"). HF is external -> egress proxy.
_clean = ",".join(
    e for e in os.environ.get("NO_PROXY", "").split(",")
    if e and "[" not in e and "]" not in e
)
os.environ["NO_PROXY"] = _clean
os.environ["no_proxy"] = _clean

tok = open(os.path.join(PROJ, ".hf_token")).read().strip()
assert tok, ".hf_token empty"
os.environ["HF_TOKEN"] = tok

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "Nisarahmedkatyar/nak-plag-index-r3"  # capital N = account namespace
# NOTE: token cannot create new repos (403); the web index goes into the
# existing dataset repo under web_index/.
LOCAL = os.path.join(PROJ, "data", "web_academic_index", "plag_web.db")

def _retry(fn, what, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"{what}: attempt {i+1}/{tries} failed: {str(e)[:100]}", flush=True)
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


api = HfApi(token=tok)
print(f"repo: {REPO} (exists; skipping create_repo - token is repo-scoped)", flush=True)

# idempotent: check if already there with matching size
def _check():
    info = api.repo_info(REPO, repo_type="dataset")
    return {f.rfilename: f.size for f in info.siblings}

try:
    have = _retry(_check, "repo_info")
    local_size = os.path.getsize(LOCAL)
    if have.get("web_index/plag_web.db") == local_size:
        print(f"already on HF with matching size ({local_size} bytes) - skipping upload")
        print(f"URL: https://huggingface.co/datasets/{REPO}/resolve/main/web_index/plag_web.db")
        sys.exit(0)
    print(f"remote has: {have}", flush=True)
except Exception as e:
    print(f"repo_info check failed after retries: {e}", flush=True)

print(f"uploading {LOCAL} ({os.path.getsize(LOCAL)} bytes)...", flush=True)


def _up():
    return api.upload_file(
        path_or_fileobj=LOCAL,
        path_in_repo="web_index/plag_web.db",
        repo_id=REPO,
        repo_type="dataset",
    )


_retry(_up, "upload_file", tries=4)
print("UPLOAD DONE", flush=True)
print(f"URL: https://huggingface.co/datasets/{REPO}/resolve/main/web_index/plag_web.db")
