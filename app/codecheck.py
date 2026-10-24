"""Dolos-backed source-code similarity check (NAK code-check tab).

Runs the Dolos CLI (when installed on the host) over uploaded code files
and returns pairwise similarity. Dolos catches copies even when variable
names, spacing, or comments were changed, because it fingerprints
normalized token streams, not raw text.

Dolos analyses one language per run, so files are grouped by language
and each group is analysed separately; pairs are merged. A group with a
single file cannot be compared and is reported as skipped.

This module never raises for a missing engine: it returns an error dict
so the endpoint can answer 503 gracefully (e.g. on a host where the
Node/Dolos install step has not run yet). No file ever leaves the host.
"""

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

MAX_FILES = 50
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 10_000_000
TIMEOUT_S = 180

# Extension -> (language family, Dolos `-l` id or None for auto-detect).
# Families group related extensions (.c with .h, .js with .jsx) so copies
# across them are still compared. Every entry was validated locally
# against the installed Dolos build; Kotlin and Swift are NOT supported
# by this build and are deliberately absent.
EXT_FAMILY = {
    "py": ("python", "python"),
    "js": ("javascript", "javascript"), "jsx": ("javascript", "javascript"),
    "ts": ("typescript", "typescript"), "tsx": ("typescript", "typescript"),
    "java": ("java", "java"),
    "c": ("c", "c"), "h": ("c", "c"),
    "cpp": ("cpp", "cpp"), "cc": ("cpp", "cpp"), "cxx": ("cpp", "cpp"),
    "cs": ("cs", None), "go": ("go", "go"), "rs": ("rust", "rust"),
    "php": ("php", "php"), "scala": ("scala", None),
}
ALLOWED_EXTS = set(EXT_FAMILY)

# Explicit-language allowlist for the UI dropdown (Dolos language ids).
LANGUAGES = {
    "python", "javascript", "typescript", "java", "c", "cpp",
    "go", "rust", "php",
}
# Which extensions belong to each dropdown language.
LANG_EXTS = {
    "python": {"py"}, "javascript": {"js", "jsx"},
    "typescript": {"ts", "tsx"}, "java": {"java"},
    "c": {"c", "h"}, "cpp": {"cpp", "cc", "cxx"},
    "go": {"go"}, "rust": {"rs"}, "php": {"php"},
}


def _safe_name(name: str, used: set[str]) -> str:
    base = Path(name or "file").name or "file"
    if base not in used:
        used.add(base)
        return base
    stem, suffix = Path(base).stem, Path(base).suffix
    i = 2
    while f"{stem}_{i}{suffix}" in used:
        i += 1
    out = f"{stem}_{i}{suffix}"
    used.add(out)
    return out


def _run_dolos_group(tmpdir: Path, group: list[tuple[str, str]],
                     lang_id: str | None) -> list[dict]:
    """Run Dolos over one single-language group; returns pair dicts."""
    paths = []
    for name, text in group:
        p = tmpdir / name
        p.write_text(text, encoding="utf-8", errors="replace")
        paths.append(str(p))
    outdir = tmpdir / ("report_" + Path(group[0][0]).suffix.lstrip("."))
    cmd = ["dolos", "run", "-f", "csv", "-o", str(outdir)]
    if lang_id:
        cmd += ["-l", lang_id]
    cmd += paths
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=TIMEOUT_S)
    pairs_csv = outdir / "pairs.csv"
    if proc.returncode != 0 or not pairs_csv.exists():
        return []
    pairs = []
    with open(pairs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                sim = float(row.get("similarity") or 0)
            except ValueError:
                sim = 0.0
            pairs.append({
                "left": Path(row.get("leftFilePath", "")).name,
                "right": Path(row.get("rightFilePath", "")).name,
                "similarity": round(sim, 4),
                "similarity_pct": round(sim * 100, 1),
                "overlap": int(float(row.get("totalOverlap") or 0)),
                "longest_fragment": int(float(row.get("longestFragment") or 0)),
            })
    return pairs


def run_codecheck(files: list[tuple[str, str]], language: str = "auto") -> dict:
    """files: [(filename, text)]. Returns a result dict or an error dict
    with an 'error' key and a 'status' key (503 engine missing, 422 input)."""
    if shutil.which("dolos") is None:
        return {"error": "code check engine not installed on this host",
                "status": 503}
    if len(files) < 2:
        return {"error": "upload at least 2 code files", "status": 422}
    if len(files) > MAX_FILES:
        return {"error": f"at most {MAX_FILES} files per check", "status": 422}

    total = 0
    clean: list[tuple[str, str]] = []
    used: set[str] = set()
    for name, text in files:
        ext = Path(name).suffix.lstrip(".").lower()
        if ext not in ALLOWED_EXTS:
            return {"error": f"unsupported file type: .{ext}", "status": 422}
        data = text.encode("utf-8", errors="replace")
        total += len(data)
        if len(data) > MAX_FILE_BYTES:
            return {"error": f"{name} exceeds 2MB", "status": 422}
        clean.append((_safe_name(name, used), text))
    if total > MAX_TOTAL_BYTES:
        return {"error": "files exceed 10MB total", "status": 422}
    if total == 0:
        return {"error": "uploaded files are empty", "status": 422}

    lang = (language or "auto").strip().lower()
    if lang != "auto" and lang in LANGUAGES:
        wanted = LANG_EXTS[lang]
        bad = [n for n, _ in clean
               if Path(n).suffix.lstrip(".").lower() not in wanted]
        if bad:
            return {"error":
                    f"language mismatch: {', '.join(bad)} "
                    f"is not {lang}", "status": 422}
        groups = {lang: clean}
        lang_ids = {lang: lang}
    else:
        groups: dict[str, list[tuple[str, str]]] = {}
        lang_ids: dict[str, str | None] = {}
        for name, text in clean:
            ext = Path(name).suffix.lstrip(".").lower()
            key, lid = EXT_FAMILY[ext]
            groups.setdefault(key, []).append((name, text))
            lang_ids[key] = lid
        lang = "auto"

    pairs: list[dict] = []
    skipped: list[dict] = []
    try:
        with tempfile.TemporaryDirectory(prefix="nak_codecheck_") as tmp:
            tmpdir = Path(tmp)
            for key, group in groups.items():
                if len(group) < 2:
                    skipped.append({
                        "files": [n for n, _ in group],
                        "reason": "only 1 file for this language; "
                                  "needs at least 2 to compare",
                    })
                    continue
                pairs.extend(_run_dolos_group(tmpdir, group, lang_ids[key]))
    except subprocess.TimeoutExpired:
        return {"error": "code check timed out", "status": 504}
    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return {
        "engine": "dolos",
        "language": lang,
        "files": [n for n, _ in clean],
        "pairs": pairs,
        "skipped": skipped,
    }
