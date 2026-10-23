"""Plagiarism (similarity) detection engine for NAK AI Detector.

Method: winnowing fingerprints over normalized word 5-grams (window 4,
md5 -> int64), matched against:
  (a) the offline OA reference library (plag_index/plag_index.db, SQLite),
  (b) fingerprints of previously uploaded documents (plag_doc_fps table).

Fingerprint parameters MUST stay in sync with scripts/build_plag_index.py
(same K, W, normalization, hash, signed conversion).

Privacy: only irreversible fingerprints of uploaded documents are retained
for cross-document comparison; the documents themselves are auto-deleted
30 minutes after each scan (see the sweeper in app/main.py). This is
disclosed on the similarity report page.
"""

from __future__ import annotations

import hashlib
import html as _html
import re
import sqlite3
from collections import deque
from pathlib import Path

K = 5  # shingle size in words
W = 4  # winnowing window -> detects shared runs >= K+W-1 = 8 words
MIN_MATCH_WORDS = 8  # ignore sources with fewer matched words (noise floor)
MERGE_GAP = 12  # merge matched spans separated by <= this many words
MIN_PLAG_WORDS = 30  # skip plagiarism for shorter documents
DOC_FP_LOOKUP_CAP = 200_000  # cap fingerprints used for the doc-vs-doc lookup
MAX_SOURCES = 25  # cap ranked sources shown

# Own color palette (soft pastels) - deliberately distinct from the
# reference reports' red/magenta highlight colors.
PLAG_COLORS = [
    "#FFE3A3",  # amber
    "#BFE3FF",  # blue
    "#C9F0C0",  # green
    "#DCCBFF",  # purple
    "#FFC9DE",  # pink
    "#B8F1E4",  # teal
    "#FFD9B3",  # orange
    "#FFF3A3",  # yellow
    "#D7E3F4",  # slate blue
    "#EAD9B8",  # tan
]

_PUNCT = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def norm_word(w: str) -> str:
    return _PUNCT.sub("", w.lower())


# ---------------------------------------------------------------- quote exclusion
# "Exclude quotes" filter (plagiarism only): quoted spans are not counted in
# the similarity % and not highlighted. Detects "..." / "..." / '...' spans
# plus whole-block indented block quotes.

_QUOTE_PATTERNS = (
    re.compile(r'"([^"]{2,})"'),
    re.compile(r'"([^"]{2,})"'),
    re.compile(r"(?<!\w)'([^']{2,})'(?!\w)"),
)
_BLOCKQUOTE_RE = re.compile(r"(?: {4,}|\t)")


def quote_exclusion_spans(text: str) -> list[tuple[int, int]]:
    """Merged (start, end) word-index spans (split_keep norm coordinates)
    covering quoted text inside one block."""
    regions: list[tuple[int, int]] = []
    for pat in _QUOTE_PATTERNS:
        for m in pat.finditer(text):
            regions.append((m.start(1), m.end(1)))
    if _BLOCKQUOTE_RE.match(text):
        # whole block is an indented block quote
        regions.append((0, len(text)))
    if not regions:
        return []
    regions.sort()
    # map to kept-word indices (same filtering as split_keep)
    excl: list[int] = []
    idx = 0
    wi = 0
    for w in text.split():
        s = text.find(w, idx)
        if s < 0:
            break
        e = s + len(w)
        idx = e
        if not norm_word(w):
            continue
        if any(s < re_ and e > rs for rs, re_ in regions):
            excl.append(wi)
        wi += 1
    if not excl:
        return []
    # merge into spans
    spans: list[tuple[int, int]] = []
    ms = ps = excl[0]
    for x in excl[1:]:
        if x == ps + 1:
            ps = x
        else:
            spans.append((ms, ps + 1))
            ms = ps = x
    spans.append((ms, ps + 1))
    return spans


def _clip_spans(spans: list[tuple[int, int]],
                excl: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove excluded ranges from (start, end) spans."""
    if not excl:
        return spans
    out: list[tuple[int, int]] = []
    for ms, me in spans:
        cur = [(ms, me)]
        for es, ee in excl:
            nxt: list[tuple[int, int]] = []
            for cs, ce in cur:
                if ce <= es or cs >= ee:
                    nxt.append((cs, ce))
                else:
                    if cs < es:
                        nxt.append((cs, es))
                    if ce > ee:
                        nxt.append((ee, ce))
            cur = nxt
            if not cur:
                break
        out.extend(cur)
    return [(s, e) for s, e in out if e > s]


def split_keep(text: str) -> tuple[list[str], list[str]]:
    """Split text into (original, normalized) token lists, 1:1 aligned,
    dropping tokens that normalize to empty."""
    orig, norm = [], []
    for w in text.split():
        n = norm_word(w)
        if n:
            orig.append(w)
            norm.append(n)
    return orig, norm


def _h64(s: str) -> int:
    h = int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")
    h &= (1 << 64) - 1
    return h - (1 << 64) if h >= (1 << 63) else h  # signed, matches index build


class StreamWinnower:
    """Streaming winnowing over a token stream.

    Feed (global_pos, token); finalized (hash, kgram_start_pos) pairs are
    appended to self.finalized in order. Identical output to the batch
    winnow() in scripts/build_plag_index.py for the same token stream.
    """

    def __init__(self, k: int = K, w: int = W):
        self.k = k
        self.w = w
        self._toks: deque[str] = deque()
        self._poss: deque[int] = deque()
        self._hashes: deque[tuple[int, int]] = deque()
        self._last: tuple[int, int] | None = None
        self.finalized: list[tuple[int, int]] = []

    def feed(self, pos: int, tok: str) -> None:
        self._toks.append(tok)
        self._poss.append(pos)
        if len(self._toks) > self.k:
            self._toks.popleft()
            self._poss.popleft()
        if len(self._toks) == self.k:
            h = _h64(" ".join(self._toks))
            self._hashes.append((h, self._poss[0]))
            if len(self._hashes) > self.w:
                self._hashes.popleft()
            if len(self._hashes) == self.w:
                m = min(self._hashes, key=lambda t: (t[0], -t[1]))
                if m != self._last:
                    self.finalized.append(m)
                    self._last = m

    def drain(self) -> list[tuple[int, int]]:
        out, self.finalized = self.finalized, []
        return out


class OAIndex:
    """Read-only handle on plag_index/plag_index.db."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        # NOTE (2026-10-12): with the 905k-doc mega-index, preloading all
        # titles (~300MB) risks OOM on small hosts. Meta is now lazy-loaded
        # per matched doc (PK lookup, <1ms) with a small bounded cache.
        self._meta_cache: dict[int, tuple[str, str]] = {}

    def available(self) -> bool:
        return self.db_path.exists()

    def open(self) -> None:
        if self._conn is not None:
            return
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True,
                               check_same_thread=False)
        self._conn = conn

    def lookup(self, fps: list[int]) -> dict[int, list[int]]:
        """Batch lookup: {fp -> [doc_rowid, ...]}. fps may contain dupes."""
        self.open()
        assert self._conn is not None
        out: dict[int, list[int]] = {}
        uniq = list(dict.fromkeys(fps))
        for i in range(0, len(uniq), 900):
            batch = uniq[i:i + 900]
            q = f"SELECT fp, doc FROM fps WHERE fp IN ({','.join('?' * len(batch))})"
            for fp, doc in self._conn.execute(q, batch):
                out.setdefault(int(fp), []).append(int(doc))
        return out

    def meta(self, doc_rowid: int) -> tuple[str, str]:
        self.open()
        key = int(doc_rowid)
        hit = self._meta_cache.get(key)
        if hit is not None:
            return hit
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT title, source FROM docs WHERE id=?", (key,)).fetchone()
        val = ((row[0] or "", row[1] or "") if row else ("", ""))
        if len(self._meta_cache) < 20000:
            self._meta_cache[key] = val
        return val

    def doc_count(self) -> int:
        self.open()
        assert self._conn is not None
        return int(self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])


class MultiOAIndex:
    """Wraps multiple OAIndex instances, presenting a unified interface.

    Doc rowids are namespaced as (index_idx, rowid) tuples encoded as strings
    to avoid collisions between indexes.
    """

    def __init__(self, indexes: list[OAIndex | None]):
        self.indexes = [idx for idx in indexes if idx is not None and idx.available()]

    def available(self) -> bool:
        return len(self.indexes) > 0

    def lookup(self, fps: list[int]) -> dict[int, list[str]]:
        """Batch lookup across all indexes. Returns {fp -> [namespaced_id, ...]}."""
        out: dict[int, list[str]] = {}
        for idx_num, idx in enumerate(self.indexes):
            hits = idx.lookup(fps)
            for fp, doc_rowids in hits.items():
                for rowid in doc_rowids:
                    # Namespace: "idxnum:rowid"
                    ns_id = f"{idx_num}:{rowid}"
                    out.setdefault(int(fp), []).append(ns_id)
        return out

    def meta(self, namespaced_id: str) -> tuple[str, str]:
        """Get (title, source) for a namespaced doc id."""
        try:
            idx_num_str, rowid_str = namespaced_id.split(":", 1)
            idx = self.indexes[int(idx_num_str)]
            return idx.meta(int(rowid_str))
        except (ValueError, IndexError):
            return ("", "")


# ---------------------------------------------------------------- matching

def _merge_spans(spans: list[tuple[int, int]], gap: int = MERGE_GAP) -> list[tuple[int, int]]:
    """Merge (start, end) word spans separated by <= gap words."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def match_blocks(
    blocks: list[dict],
    oa_index: OAIndex | None,
    doc_lookup,  # fn(list[int]) -> {fp: [(doc_ref, filename)]}
) -> dict:
    """Run plagiarism matching over parsed blocks.

    blocks: [{"tag":..., "text":...}, ...] (plain text per block).
    Returns a result dict with sources, passages, percentages.
    """
    # 1. tokenize + stream winnowing (single continuous stream across blocks)
    winnower = StreamWinnower()
    block_words: list[list[str]] = []   # original tokens per block
    block_starts: list[int] = []         # global word offset per block
    gpos = 0
    for b in blocks:
        orig, norm = split_keep(b.get("text", ""))
        block_words.append(orig)
        block_starts.append(gpos)
        for t in norm:
            winnower.feed(gpos, t)
            gpos += 1
    total_words = gpos
    fps = winnower.drain()  # [(hash, kgram_start_global_pos)]
    if total_words < MIN_PLAG_WORDS or not fps:
        return _empty_result(total_words)

    # 2. lookups
    uniq_fps = list(dict.fromkeys(h for h, _ in fps))
    oa_hits: dict[int, list[int]] = {}
    if oa_index is not None and oa_index.available():
        oa_hits = oa_index.lookup(uniq_fps)
    doc_hits: dict[int, list[tuple[str, str]]] = {}
    capped = uniq_fps[:DOC_FP_LOOKUP_CAP]
    if doc_lookup is not None and capped:
        doc_hits = doc_lookup(capped)

    # map global pos -> (block_idx, local_idx) via block_starts
    import bisect
    starts = block_starts

    def locate(pos: int) -> tuple[int, int]:
        bi = bisect.bisect_right(starts, pos) - 1
        return bi, pos - starts[bi]

    # 3. collect per-source spans: key -> {"kind","label","spans":[(bi,ls,le)]}
    sources: dict[tuple[str, str], dict] = {}

    def add_hit(key, kind, label, gstart):
        bi, ls = locate(gstart)
        le = ls + K
        s = sources.get(key)
        if s is None:
            s = sources[key] = {"kind": kind, "label": label, "spans": []}
        s["spans"].append((bi, ls, le))

    for h, gs in fps:
        for doc_rowid in oa_hits.get(h, ()):
            title, source = oa_index.meta(doc_rowid) if oa_index else ("", "")
            label = title or f"Reference document {doc_rowid}"
            add_hit(("oa", str(doc_rowid)), "library", label, gs)
        for doc_ref, filename in doc_hits.get(h, ()):
            add_hit(("doc", doc_ref), "upload", filename or doc_ref, gs)

    # 4. per source: merge spans per block -> passages, count words
    ranked = []
    for key, s in sources.items():
        by_block: dict[int, list[tuple[int, int]]] = {}
        for bi, ls, le in s["spans"]:
            by_block.setdefault(bi, []).append((ls, le))
        passages: list[tuple[int, int, int]] = []  # (bi, s, e)
        words = 0
        for bi, spans in by_block.items():
            for ms, me in _merge_spans(spans):
                me = min(me, len(block_words[bi]))
                if me > ms:
                    passages.append((bi, ms, me))
                    words += me - ms
        if words >= MIN_MATCH_WORDS:
            ranked.append({
                "key": key, "kind": s["kind"], "label": s["label"],
                "words": words, "passages": passages,
            })
    ranked.sort(key=lambda r: -r["words"])
    ranked = ranked[:MAX_SOURCES]

    # 5. overall %: unique matched words across all sources
    matched_global: set[int] = set()
    lib_words = 0
    doc_words = 0
    per_block_spans: dict[int, list[tuple[int, int, int]]] = {}  # bi -> [(s,e,color_idx)]
    for ci, r in enumerate(ranked):
        r["color"] = PLAG_COLORS[ci % len(PLAG_COLORS)]
        r["color_idx"] = ci
        r["category"] = _source_category(r["kind"], r["label"])
        for bi, ms, me in r["passages"]:
            for p in range(block_starts[bi] + ms, block_starts[bi] + me):
                matched_global.add(p)
            per_block_spans.setdefault(bi, []).append((ms, me, ci))
        if r["kind"] == "library":
            lib_words += r["words"]
        else:
            doc_words += r["words"]
        r["pct"] = round(r["words"] / total_words * 100, 1) if total_words else 0.0

    sim_pct = round(len(matched_global) / total_words * 100, 1) if total_words else 0.0
    return {
        "total_words": total_words,
        "sim_pct": sim_pct,
        "lib_pct": round(lib_words / total_words * 100, 1) if total_words else 0.0,
        "doc_pct": round(doc_words / total_words * 100, 1) if total_words else 0.0,
        "sources": ranked,
        "block_spans": per_block_spans,   # bi -> [(local_s, local_e, color_idx)]
        "block_words": block_words,       # original tokens per block
        "query_fps": fps,                 # for doc-vs-doc retention
    }


def _empty_result(total_words: int) -> dict:
    return {
        "total_words": total_words, "excluded_words": 0,
        "sim_pct": 0.0, "lib_pct": 0.0, "doc_pct": 0.0,
        "sources": [], "block_spans": {}, "block_words": [], "query_fps": [],
        "span_details": [],
    }


# ---------------------------------------------------------------- rendering

def _esc(t: str) -> str:
    return _html.escape(t)


def render_sim_block(orig_tokens: list[str],
                     spans: list[tuple[int, int, int]],
                     colors: list[str]) -> str:
    """Render one block's tokens with color-coded source spans.

    spans: [(local_start, local_end, color_idx)]. Overlaps are split at
    boundaries; each atomic segment takes the color of the highest-ranked
    (lowest color_idx) covering source.
    """
    n = len(orig_tokens)
    if not spans or n == 0:
        return "<p>" + _esc(" ".join(orig_tokens)) + "</p>"
    pts = {0, n}
    for s, e, _ci in spans:
        pts.add(max(0, s))
        pts.add(min(n, e))
    pts = sorted(pts)
    # color_idx -> color lookup by rank order passed in
    out: list[str] = []
    prev_ci: int | None = None  # to badge only the start of each source run
    for a, b in zip(pts, pts[1:]):
        if a >= b:
            continue
        covering = [ci for (s, e, ci) in spans if s <= a and e >= b]
        seg = _esc(" ".join(orig_tokens[a:b]))
        if covering:
            ci = min(covering)
            badge = ""
            if ci != prev_ci:
                # source number badge at the start of each contiguous run
                badge = (f'<sup class="psrc-num" title="Source {ci + 1}">'
                         f'{ci + 1}</sup>')
            out.append(
                f'<span class="psrc" data-ci="{ci}" '
                f'style="background:{colors[ci % len(colors)]}">{badge}{seg}</span>'
            )
            prev_ci = ci
        else:
            out.append(seg)
            prev_ci = None
    return "<p>" + " ".join(out) + "</p>"


def _fmt_pct(x: float) -> str:
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


_URL_LIKE = re.compile(r"^(https?://|www\.)|^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?:[/\s]|$)", re.I)


def _source_category(kind: str, label: str) -> str:
    """Top-Sources category pill label (own wording, NAK taxonomy).

    - previously scanned documents -> "Submitted documents"
    - reference-library hits that look like a web address -> "Internet"
    - everything else in the library (scholarly publications) -> "Publication"
    """
    if kind == "upload":
        return "Submitted documents"
    lab = (label or "").strip()
    if _URL_LIKE.search(lab) or "wikipedia.org" in lab.lower():
        return "Internet"
    return "Publication"


def _cat_slug(cat: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", cat.lower()).strip("-")


def sim_cover_html(*, sim_pct: float, lib_pct: float, doc_pct: float,
                   sources: list[dict], total_words: int,
                   notes: list[str] | None = None) -> str:
    """Similarity cover page HTML (own layout + own wording, NAK branding)."""
    if sources:
        rows = []
        for i, r in enumerate(sources, 1):
            cat = r.get("category") or _source_category(r["kind"], r["label"])
            pct_txt = _fmt_pct(r["pct"]) if r["pct"] >= 1 else (
                "less than 1" if r["pct"] > 0 else "0")
            rows.append(
                f'<div class="topsrc" data-ci="{r["color_idx"]}" '
                f'onclick="plagFilter({r["color_idx"]})">'
                f'<span class="topsrc-num" style="background:{r["color"]}">{i}</span>'
                f'<span class="topsrc-main">'
                f'<span class="topsrc-topline">'
                f'<span class="topsrc-cat topsrc-cat-{_cat_slug(cat)}">{_esc(cat)}</span>'
                f'<span class="topsrc-pct">{pct_txt}%</span>'
                f'</span>'
                f'<span class="topsrc-title">{_esc(r["label"][:140])}</span>'
                f'<span class="topsrc-sub">{r["words"]:,} matched words</span>'
                f'</span>'
                f"</div>"
            )
        src_html = (
            '<div class="nak-section">Top Sources</div>\n'
            '<p class="nak-section-desc">Sources with the most text matching this '
            'document &mdash; strongest match first. Select a source to isolate '
            'its highlighted passages.</p>\n'
            '<div class="psrcall" onclick="plagFilter(-1)">Show all matches</div>\n'
            + "\n".join(rows)
        )
    else:
        src_html = ('<div class="nosrc">No matching passages found in the '
                    "reference library or previously scanned documents.</div>")
    notes = notes or []
    if notes:
        notes_html = ('<div class="simnotes">' + "".join(
            f'<div class="simnote">{_esc(n)}</div>' for n in notes) + "</div>")
    else:
        notes_html = ""
    # ---- premium category cards ----
    _cat_words: dict[str, int] = {}
    for _r in sources:
        _c = _r.get("category") or _source_category(_r["kind"], _r["label"])
        _cat_words[_c] = _cat_words.get(_c, 0) + _r["words"]
    _cat_order = ["Submitted documents", "Internet", "Publication"]
    _cat_colors = {"Submitted documents": "#f59e0b",
                   "Internet": "#0ea5e9", "Publication": "#8b5cf6"}
    _cat_cards = []
    for _c in _cat_order:
        _w = _cat_words.get(_c, 0)
        _p = round(_w / total_words * 100, 1) if total_words else 0.0
        _cat_cards.append(
            f'<div class="nak-cat" style="--cc:{_cat_colors[_c]}">'
            f'<div class="nak-cat-lbl">{_esc(_c)}</div>'
            f'<div class="nak-cat-pct">{_fmt_pct(_p)}%</div>'
            f'</div>')
    _cat_html = '<div class="nak-cats">\n' + "\n".join(_cat_cards) + "\n</div>"
    # note about overlapping sources (same text on multiple sites)
    _overlap_note = ('<p class="simoverlap">Note: the same passage can match '
        'more than one source (for example, when a website republishes text '
        'from another site). Category percentages may therefore add up to more '
        'than the total similarity, which counts each matched word once.</p>')
    # gauge color based on score
    _gc = "#22c55e" if sim_pct < 15 else ("#f59e0b" if sim_pct < 40 else "#ef4444")
    return f"""<div class="nak-hero">
<div class="nak-hero-brand">NAK AI Detector</div>
<div class="nak-hero-title">Similarity Report</div>
<p class="nak-hero-sub">Plagiarism analysis &mdash; text overlap with reference sources</p>
</div>
<div class="nak-score-wrap">
<div class="nak-gauge" style="--gp:{sim_pct:.1f};--gc:{_gc}">
<div class="nak-gauge-inner">
<div class="nak-gauge-pct">{_fmt_pct(sim_pct)}%</div>
<div class="nak-gauge-lbl">Similarity</div>
</div>
</div>
<div class="nak-score-info">
<div class="nak-score-head">Text Overlap Analysis</div>
<p class="nak-score-desc">Similarity estimates how much of this document matches text
found in our reference library or in documents scanned before. It measures
text overlap &mdash; not intent. Properly quoted or cited passages can still
match, so always review the highlighted passages in context before drawing
conclusions.</p>
</div>
</div>
<div class="nak-section">Match Breakdown</div>
{_cat_html}
{_overlap_note}
{src_html}
{notes_html}
<div class="legend"><span class="swatch" style="background:#FFE3A3"></span>
Each source gets its own highlight color in the document pages below.</div>
<hr class="rule">
<div class="disclaimer">
  <div class="disclaimer-title">About this similarity check</div>
  <p>Matches are found by comparing irreversible word fingerprints against an
  offline reference library and fingerprints of previously scanned documents.
  Small incidental overlaps (under {MIN_MATCH_WORDS} words per source) are
  ignored. Quoted material, references and common phrases may legitimately
  match. For cross-document comparison we keep only one-way fingerprints of
  past scans &mdash; the documents themselves are auto-deleted 30 minutes
  after each scan and can never be reconstructed from fingerprints.</p>
</div>
<script>
function plagFilter(ci){{
  document.querySelectorAll('#simdoc .psrc').forEach(function(el){{
    el.classList.toggle('pdim', ci >= 0 && el.getAttribute('data-ci') != String(ci));
  }});
  document.querySelectorAll('.topsrc').forEach(function(el){{
    el.classList.toggle('pactive', el.getAttribute('data-ci') == String(ci));
  }});
}}
</script>
"""


def sim_css() -> str:
    return """
/* ============ NAK Premium Similarity Report ============ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

/* ---- Hero header with gradient ---- */
.nak-hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0ea5e9 100%);
  border-radius: 16px; padding: 28px 32px; margin: 0 0 24px;
  color: #fff; position: relative; overflow: hidden;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.nak-hero::before {
  content: ''; position: absolute; top: -50%; right: -10%;
  width: 300px; height: 300px; border-radius: 50%;
  background: rgba(255,255,255,0.06);
}
.nak-hero-brand { font-size: 13px; font-weight: 700; letter-spacing: 3px;
  text-transform: uppercase; opacity: 0.85; margin-bottom: 6px; }
.nak-hero-title { font-size: 26px; font-weight: 900; margin: 0 0 4px; }
.nak-hero-sub { font-size: 13px; opacity: 0.75; margin: 0; }

/* ---- Score gauge ---- */
.nak-score-wrap { display: flex; align-items: center; gap: 24px; margin: 0 0 24px; }
.nak-gauge {
  flex: none; width: 130px; height: 130px; border-radius: 50%;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: conic-gradient(var(--gc, #0ea5e9) calc(var(--gp, 0) * 1%), #e2e8f0 0);
  position: relative;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.nak-gauge-inner {
  width: 102px; height: 102px; border-radius: 50%; background: #fff;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.nak-gauge-pct { font-size: 30px; font-weight: 900; color: #0f172a; line-height: 1; }
.nak-gauge-lbl { font-size: 10px; font-weight: 700; color: #64748b;
  text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; }
.nak-score-info { flex: 1; }
.nak-score-head { font-size: 17px; font-weight: 800; color: #0f172a; margin: 0 0 6px; }
.nak-score-desc { font-size: 13px; color: #475569; line-height: 1.6; margin: 0; }

/* ---- Category cards ---- */
.nak-cats { display: flex; gap: 12px; margin: 0 0 8px; }
.nak-cat {
  flex: 1; border-radius: 12px; padding: 14px 16px;
  border: 1px solid #e2e8f0; background: #fff;
  border-top: 4px solid var(--cc, #0ea5e9);
}
.nak-cat-lbl { font-size: 11px; font-weight: 700; color: #64748b;
  text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
.nak-cat-pct { font-size: 24px; font-weight: 900; color: #0f172a; }
.simoverlap { font-size: 11.5px; color: #94a3b8; margin: 10px 0 0; line-height: 1.5;
  font-style: italic; }

/* ---- Section headers ---- */
.nak-section { font-size: 16px; font-weight: 800; color: #0f172a;
  margin: 28px 0 4px; padding-bottom: 8px;
  border-bottom: 2px solid #f1f5f9; }
.nak-section-desc { font-size: 12.5px; color: #64748b; margin: 0 0 12px; line-height: 1.5; }

/* ---- Top source cards ---- */
.topsrc { display: flex; align-items: flex-start; gap: 12px;
  border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px;
  margin-bottom: 10px; cursor: pointer; background: #fff;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
  transition: box-shadow 0.15s, border-color 0.15s; }
.topsrc:hover { border-color: #94a3b8; box-shadow: 0 4px 12px rgba(15,23,42,0.08); }
.topsrc.pactive { border-color: #0284c7; box-shadow: 0 0 0 2px rgba(2,132,199,.25); }
.topsrc-num { flex: none; width: 32px; height: 32px; border-radius: 10px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 15px; font-weight: 800; color: #0f172a; margin-top: 2px;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.topsrc-main { flex: 1; min-width: 0; }
.topsrc-topline { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.topsrc-cat { font-size: 10.5px; font-weight: 800; padding: 3px 12px; border-radius: 999px;
  white-space: nowrap; text-transform: uppercase; letter-spacing: 0.5px;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.topsrc-cat-publication { background: #ede9fe; color: #6d28d9; }
.topsrc-cat-internet { background: #e0f2fe; color: #0369a1; }
.topsrc-cat-submitted-documents { background: #fef3c7; color: #b45309; }
.topsrc-pct { font-size: 20px; font-weight: 900; color: #0f172a; flex: none; }
.topsrc-title { display: block; font-size: 14px; font-weight: 700; color: #0f172a;
  margin-top: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.topsrc-sub { display: block; font-size: 12px; color: #94a3b8; margin-top: 2px; }

/* ---- Document highlights ---- */
.simhead { font-size: 20px; font-weight: 800; color: #0f172a; margin: 4px 0 8px; }
#simdoc .psrc { border-radius: 4px; padding: 2px 3px;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
#simdoc .psrc.pdim { background: transparent !important; }
#simdoc .psrc.pdim .psrc-num { display: none; }
#simdoc .psrc-num { display: inline-flex; align-items: center; justify-content: center;
  min-width: 16px; height: 16px; border-radius: 999px; padding: 0 4px;
  background: #0f172a; color: #fff; font-size: 10px; font-weight: 800;
  line-height: 1; vertical-align: 2px; margin-right: 4px;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.sblock { padding: 10px 14px; border-radius: 8px; margin: 6px 0;
  line-height: 1.7; background: #f8fafc; border: 1px solid #f1f5f9; }
.sblock p { margin: 0 0 10px; } .sblock p:last-child { margin-bottom: 0; }
.nosrc { font-size: 14px; color: #475569; background: #f0fdf4;
  border: 1px solid #bbf7d0; border-radius: 12px; padding: 18px;
  text-align: center; font-weight: 600; }

/* ---- Legacy compat ---- */
.simbrk { margin: 18px 0; }
.simbrkrow { font-size: 14px; color: #475569; margin-bottom: 8px;
  display: flex; align-items: center; gap: 8px; }
.pchip { display: inline-block; width: 16px; height: 16px; border-radius: 4px;
  border: 1px solid #94a3b8; flex: none;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.psrchead { font-size: 14px; font-weight: 800; color: #0f172a; margin: 20px 0 10px; }
.psrcall { display: inline-block; font-size: 13px; color: #0284c7; cursor: pointer;
  margin-bottom: 10px; text-decoration: underline; }

@media (max-width: 640px) {
  .nak-hero { padding: 20px; }
  .nak-hero-title { font-size: 20px; }
  .nak-cats { flex-direction: column; }
  .nak-score-wrap { flex-direction: column; text-align: center; }
  .topsrc-pct { font-size: 17px; }
}
"""


# ---------------------------------------------------------------- full scan pass

def _pb_path(tmp_dir, scan_id):
    return Path(tmp_dir) / f"{scan_id}.pb.jsonl"


def _pf_path(tmp_dir, scan_id):
    return Path(tmp_dir) / f"{scan_id}.pf.jsonl"


def _sim_parts_path(tmp_dir, scan_id):
    return Path(tmp_dir) / f"{scan_id}.simparts.html"


def _sim_manifest_path(tmp_dir, scan_id):
    return Path(tmp_dir) / f"{scan_id}.simmanifest.jsonl"


def write_plag_block(text: str, tmp_dir, scan_id, exclude_quotes: bool = False) -> None:
    """Append one raw-text block during the parse loop (temp file, streaming).

    With exclude_quotes, quoted spans are recorded alongside the block so the
    similarity pass can skip them (not counted, not highlighted).
    """
    import json

    if not text or not text.strip():
        return
    rec: dict = {"text": text}
    if exclude_quotes:
        excl = quote_exclusion_spans(text)
        if excl:
            rec["excl"] = excl
    p = _pb_path(tmp_dir, scan_id)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def cleanup_plag_tmp(tmp_dir, scan_id) -> None:
    """Delete plagiarism working temp files (block/fingerprint streams).

    The sim parts + manifest are NOT deleted here: assemble_report() reads
    them after the scan pass. Use cleanup_sim_parts() once assembled.
    """
    for p in (_pb_path(tmp_dir, scan_id), _pf_path(tmp_dir, scan_id)):
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def cleanup_sim_parts(tmp_dir, scan_id) -> None:
    """Delete rendered similarity parts + manifest (after report assembly)."""
    for p in (_sim_parts_path(tmp_dir, scan_id), _sim_manifest_path(tmp_dir, scan_id)):
        try:
            Path(p).unlink(missing_ok=True)
        except OSError:
            pass


def run_plagiarism(scan_id: str, filename: str, tmp_dir,
                   oa_index: OAIndex | None,
                   doc_lookup_fn,
                   record_doc_fn,
                   exclude_quotes: bool = False) -> dict:
    """Full plagiarism pass over the temp block file.

    1. Stream blocks -> winnowing fingerprints (temp fps file).
    2. Batch lookups against the OA index + uploaded-doc fingerprints.
    3. Aggregate per-source passages, rank, assign colors.
    4. Render similarity blocks into the sim parts file.
    5. Record this document's fingerprints for future doc-vs-doc (after
       matching, so it never matches itself) -- unless record_doc_fn is
       None ("no repository" mode).
    Quoted spans (when exclude_quotes) are clipped from every match span:
    not counted in the similarity %, not highlighted.
    Returns the result dict (stats + ranked sources).
    """
    import bisect
    import json

    tmp_dir = Path(tmp_dir)
    pb = _pb_path(tmp_dir, scan_id)
    if not pb.exists():
        return _empty_result(0)

    # ---- pass 1: winnowing over the block stream ----
    winnower = StreamWinnower()
    block_starts: list[int] = []
    block_nwords: list[int] = []
    block_excl: dict[int, list[tuple[int, int]]] = {}
    excluded_words = 0
    gpos = 0
    pf = _pf_path(tmp_dir, scan_id)
    with open(pb, encoding="utf-8") as bf, open(pf, "w", encoding="utf-8") as ff:
        for line in bf:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                text = rec.get("text", "")
            except Exception:
                continue
            _orig, norm = split_keep(text)
            bi = len(block_starts)
            if exclude_quotes:
                excl = [tuple(x) for x in rec.get("excl", [])]
                if excl:
                    block_excl[bi] = excl
                    excluded_words += sum(e - s for s, e in excl)
            block_starts.append(gpos)
            block_nwords.append(len(norm))
            for t in norm:
                winnower.feed(gpos, t)
                gpos += 1
            for h, gs in winnower.drain():
                ff.write(json.dumps([h, gs]) + "\n")
    total_words = gpos
    n_blocks = len(block_starts)
    if total_words < MIN_PLAG_WORDS:
        cleanup_plag_tmp(tmp_dir, scan_id)
        return _empty_result(total_words)

    def locate(pos: int) -> tuple[int, int]:
        bi = bisect.bisect_right(block_starts, pos) - 1
        bi = max(0, min(bi, n_blocks - 1))
        return bi, pos - block_starts[bi]

    # ---- pass 2: batched lookups ----
    uniq: list[int] = []
    seen_u: set[int] = set()
    with open(pf, encoding="utf-8") as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
            try:
                h = int(json.loads(line)[0])
            except Exception:
                continue
            if h not in seen_u:
                seen_u.add(h)
                uniq.append(h)

    oa_hits: dict[int, list[int]] = {}
    if oa_index is not None and oa_index.available():
        oa_hits = oa_index.lookup(uniq)
    doc_hits: dict[int, list[tuple[str, str]]] = {}
    if doc_lookup_fn is not None:
        doc_hits = doc_lookup_fn(uniq[:DOC_FP_LOOKUP_CAP])

    sources: dict[tuple[str, str], dict] = {}
    # per-block per-source spans, for rendering after ranking
    block_src: dict[int, dict[tuple[str, str], list[tuple[int, int]]]] = {}

    def add_hit(key, kind, label, gstart):
        bi, ls = locate(gstart)
        le = min(ls + K, block_nwords[bi])
        if le <= ls:
            return
        s = sources.get(key)
        if s is None:
            s = sources[key] = {"kind": kind, "label": label, "spans": []}
        excl = block_excl.get(bi)
        clipped = _clip_spans([(ls, le)], excl) if excl else [(ls, le)]
        for cs, ce in clipped:
            s["spans"].append((bi, cs, ce))
            block_src.setdefault(bi, {}).setdefault(key, []).append((cs, ce))

    with open(pf, encoding="utf-8") as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
            try:
                h, gs = json.loads(line)
                h, gs = int(h), int(gs)
            except Exception:
                continue
            if oa_index is not None:
                for doc_rowid in oa_hits.get(h, ()):
                    title, _src = oa_index.meta(doc_rowid)
                    add_hit(("oa", str(doc_rowid)), "library",
                            title or f"Reference document {doc_rowid}", gs)
            for doc_ref, fname in doc_hits.get(h, ()):
                add_hit(("doc", doc_ref), "upload", fname or doc_ref, gs)

    # ---- aggregate, rank, color ----
    ranked = []
    for key, s in sources.items():
        by_block: dict[int, list[tuple[int, int]]] = {}
        for bi, ls, le in s["spans"]:
            by_block.setdefault(bi, []).append((ls, le))
        words = 0
        for spans in by_block.values():
            for ms, me in _merge_spans(spans):
                words += me - ms
        if words >= MIN_MATCH_WORDS:
            ranked.append({"key": key, "kind": s["kind"], "label": s["label"],
                           "words": words})
    ranked.sort(key=lambda r: -r["words"])
    ranked = ranked[:MAX_SOURCES]
    keep_keys = {r["key"] for r in ranked}
    # Excluded (e.g. quoted) words are removed from the similarity
    # denominator entirely: not counted, not highlighted.
    eff_words = max(1, total_words - excluded_words)
    for ci, r in enumerate(ranked):
        r["color"] = PLAG_COLORS[ci % len(PLAG_COLORS)]
        r["color_idx"] = ci
        r["category"] = _source_category(r["kind"], r["label"])
        r["pct"] = round(r["words"] / eff_words * 100, 1)
    key_to_ci = {r["key"]: r["color_idx"] for r in ranked}

    matched_global: set[int] = set()
    lib_words = 0
    doc_words = 0
    for r in ranked:
        if r["kind"] == "library":
            lib_words += r["words"]
        else:
            doc_words += r["words"]
    # unique matched words (union across sources, per block merged)
    for bi, per_src in block_src.items():
        all_spans: list[tuple[int, int]] = []
        for key, spans in per_src.items():
            if key in keep_keys:
                all_spans.extend(spans)
        for ms, me in _merge_spans(all_spans):
            base = block_starts[bi]
            for p in range(base + ms, base + me):
                matched_global.add(p)

    # ---- pass 3: render similarity blocks ----
    sim_parts = _sim_parts_path(tmp_dir, scan_id)
    sim_manifest = _sim_manifest_path(tmp_dir, scan_id)
    colors = [r["color"] for r in ranked]
    with open(pb, encoding="utf-8") as bf, \
         open(sim_parts, "w", encoding="utf-8") as pf_out, \
         open(sim_manifest, "w", encoding="utf-8") as mf:
        offset = 0
        for bi, line in enumerate(bf):
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            orig, _norm = split_keep(text)
            spans: list[tuple[int, int, int]] = []
            per_src = block_src.get(bi, {})
            for key, sps in per_src.items():
                ci = key_to_ci.get(key)
                if ci is None:
                    continue
                for ms, me in _merge_spans(sps):
                    me = min(me, len(orig))
                    if me > ms:
                        spans.append((ms, me, ci))
            html = render_sim_block(orig, spans, colors)
            data = f'<div class="sblock">\n{html}\n</div>\n'.encode("utf-8")
            pf_out.write(data.decode("utf-8"))
            mf.write(json.dumps({"o": offset, "l": len(data),
                                 "w": len(orig)}) + "\n")
            offset += len(data)

    # ---- record this doc's fingerprints for future doc-vs-doc ----
    try:
        if record_doc_fn is not None and total_words >= MIN_PLAG_WORDS:
            fps: list[tuple[int, int]] = []
            with open(pf, encoding="utf-8") as ff:
                for line in ff:
                    line = line.strip()
                    if line:
                        try:
                            h, gs = json.loads(line)
                            fps.append((int(h), int(gs)))
                        except Exception:
                            pass
            if len(fps) > 50000:  # uniform sample for giant docs
                step = len(fps) // 50000
                fps = fps[::step]
            record_doc_fn(scan_id, filename, total_words, fps)
    except Exception:
        pass

    # temp block/fp files no longer needed (sim parts stay for assembly)
    for p in (pb, pf):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    sim_pct = round(len(matched_global) / eff_words * 100, 1)
    # Per-source span details for API consumers: (block_idx, word_start,
    # word_end) in split_keep word coordinates, per ranked source. The caller
    # converts these to document char offsets using the block texts.
    span_details = []
    for r in ranked:
        key = r["key"]
        s = sources.get(key, {})
        by_block: dict[int, list[tuple[int, int]]] = {}
        for bi, ls, le in s.get("spans", []):
            by_block.setdefault(bi, []).append((ls, le))
        blocks = []
        for bi in sorted(by_block):
            for ms, me in _merge_spans(by_block[bi]):
                blocks.append([bi, ms, me])
        span_details.append({
            "source_key": list(key),
            "source_label": r["label"],
            "kind": r["kind"],
            "blocks": blocks,
        })
    return {
        "total_words": total_words,
        "excluded_words": excluded_words,
        "sim_pct": sim_pct,
        "lib_pct": round(lib_words / eff_words * 100, 1),
        "doc_pct": round(doc_words / eff_words * 100, 1),
        "sources": [
            {"label": r["label"], "kind": r["kind"], "words": r["words"],
             "pct": r["pct"], "color": r["color"], "color_idx": r["color_idx"],
             "category": r["category"]}
            for r in ranked
        ],
        "span_details": span_details,
    }


def read_sim_manifest(tmp_dir, scan_id) -> list[dict]:
    import json

    out = []
    mp = _sim_manifest_path(tmp_dir, scan_id)
    if mp.exists():
        with open(mp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def read_sim_ranges(tmp_dir, scan_id, entries: list[dict]) -> str:
    parts = _sim_parts_path(tmp_dir, scan_id)
    out: list[str] = []
    with open(parts, "rb") as pf:
        for e in entries:
            pf.seek(e["o"])
            out.append(pf.read(e["l"]).decode("utf-8"))
    return "".join(out)
