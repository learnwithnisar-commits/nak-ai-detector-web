"""Chunk-level detection and streaming HTML report builder.

- Units from app.parse are grouped into ~150-250 word chunks.
- Each chunk is scored with the model (via a score callback).
- AI TEXT PERCENTAGE = words in chunks with p>=0.7 / total words * 100.
- The report is written to disk incrementally (chunk divs stream into a
  parts file, then the cover is written and the parts copied over), so a
  very large document never requires the whole HTML in RAM.

AI highlight color (locked, sampled from Nisar's reference): #B9E8F0.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

AI_HIGHLIGHT = "#B9E8F0"  # locked - do not change
AI_THRESHOLD = 0.5
SEGMENT_WINDOW = 8  # sentences per scoring segment
SEGMENT_STRIDE = 4  # sliding-window stride (each sentence covered by ~2 segments)
MIN_PROSE_WORDS = 300  # below this, detection is flagged as unreliable
WORDS_PER_PAGE_ESTIMATE = 500  # for .docx/.txt page-count estimates
PAGE_TARGET_WORDS = 450  # target words per document page (pages 3+) in the report

_SENT_SPLIT = re.compile(r"(?<=[.!?\u2026\u06d4\u061f])\s+")
_PKT = ZoneInfo("Asia/Karachi")


def make_submission_id() -> str:
    """Our own submission ID format: NAK-YYYYMMDD-XXXXXX (random alphanumerics)."""
    import random
    import string

    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"NAK-{datetime.now(_PKT).strftime('%Y%m%d')}-{rand}"


def fmt_dt(ts: float) -> str:
    """Format a timestamp like 'Oct 11, 2026, 11:32 PM GMT+5' (Asia/Karachi)."""
    dt = datetime.fromtimestamp(ts, _PKT)
    hour12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.strftime('%b %d, %Y')}, {hour12}:{dt.minute:02d} {ampm} GMT+5"


def human_size(n: int) -> str:
    """Human-readable file size, e.g. '255.5 KB'."""
    n = max(0, int(n))
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def count_words(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Split prose text into sentences."""
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


class SentenceScorer:
    """Streaming sliding-window sentence scorer.

    Segments are built as a sliding window of ``SEGMENT_WINDOW`` sentences
    with ``SEGMENT_STRIDE`` (defaults 8 / 4), so every sentence is scored in
    context by ~2 segments. Each segment is scored with ``score_fn`` (0-1);
    a sentence's score is the mean of the segment scores covering it.
    Finalized ``(gidx, score)`` pairs are appended to ``self.finalized`` in
    ascending sentence order. Bounded memory: only ~window+stride sentences
    are held at a time, so multi-MB documents never blow up RAM.
    """

    def __init__(self, score_fn, window: int = SEGMENT_WINDOW, stride: int = SEGMENT_STRIDE):
        self.score_fn = score_fn
        self.window = window
        self.stride = stride
        self.texts: dict[int, str] = {}
        self.acc: dict[int, list[float]] = {}
        self.next_gidx = 0
        self.next_seg = 0
        self.finalized: deque[tuple[int, float]] = deque()

    def feed(self, text: str) -> int:
        """Add one sentence; returns its global index."""
        g = self.next_gidx
        self.next_gidx += 1
        self.texts[g] = text
        self.acc[g] = []
        while self.next_seg <= g - self.window + 1:
            self._emit(self.next_seg, self.next_seg + self.window)
            self.next_seg += self.stride
        return g

    def _emit(self, start: int, end: int) -> None:
        seg_text = " ".join(self.texts[j] for j in range(start, end))
        try:
            p = float(self.score_fn(seg_text))
        except Exception:
            p = 0.0
        p = min(max(p, 0.0), 1.0)
        for j in range(start, end):
            self.acc[j].append(p)
            if (j // self.stride) * self.stride == start:
                # `start` is j's last covering segment -> sentence is done
                scores = self.acc.pop(j)
                del self.texts[j]
                self.finalized.append((j, sum(scores) / len(scores)))

    def finish(self) -> None:
        """Emit tail segments and finalize every remaining sentence."""
        n = self.next_gidx
        s = self.next_seg
        while s < n:
            self._emit(s, min(s + self.window, n))
            s += self.stride


def render_prose_paragraph(sentences: list[tuple[str, float]]) -> str:
    """Render a prose paragraph; sentences scoring >= 0.5 get highlighted."""
    parts = []
    for text, score in sentences:
        e = _escape(text)
        if score >= AI_THRESHOLD:
            parts.append(f'<span class="sai">{e}</span>')
        else:
            parts.append(e)
    return "<p>" + " ".join(parts) + "</p>"


def render_unit_html(unit: dict) -> str:
    """Render a non-prose unit (heading / table / plain para) unhighlighted."""
    tag = unit["tag"]
    if tag in ("h1", "h2", "h3"):
        return f"<{tag}>{unit['html']}</{tag}>"
    if tag == "table":
        return unit["html"]
    return f"<p>{unit['html']}</p>"


def render_list_html(units: list[dict]) -> str:
    """Render a run of same-type list items as <ul> or <ol>."""
    list_tag = "ul" if units[0]["tag"] == "li-bullet" else "ol"
    items = "".join(f"<li>{u['html']}</li>" for u in units)
    return f"<{list_tag}>{items}</{list_tag}>"


def _escape(text: str) -> str:
    import html as _h

    return _h.escape(text)


def _parts_path(tmp_dir: Path, scan_id: str) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / f"{scan_id}.parts.html"


def _manifest_path(tmp_dir: Path, scan_id: str) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / f"{scan_id}.manifest.jsonl"


def reset_parts(tmp_dir: Path, scan_id: str) -> None:
    _parts_path(tmp_dir, scan_id).unlink(missing_ok=True)
    _manifest_path(tmp_dir, scan_id).unlink(missing_ok=True)


def append_block_html(html: str, words: int, tmp_dir: Path, scan_id: str) -> None:
    """Append one rendered block to the streaming parts file (+ manifest)."""
    import json

    parts = _parts_path(tmp_dir, scan_id)
    data = f'<div class="ablock">\n{html}\n</div>\n'.encode("utf-8")
    offset = parts.stat().st_size if parts.exists() else 0
    with open(parts, "ab") as pf:
        pf.write(data)
    with open(_manifest_path(tmp_dir, scan_id), "a", encoding="utf-8") as mf:
        mf.write(json.dumps({"o": offset, "l": len(data), "w": words}) + "\n")


def read_manifest(tmp_dir: Path, scan_id: str) -> list[dict]:
    import json

    manifest: list[dict] = []
    mp = _manifest_path(tmp_dir, scan_id)
    if mp.exists():
        with open(mp, encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if line:
                    manifest.append(json.loads(line))
    return manifest


def paginate_manifest(manifest: list[dict], target_words: int = PAGE_TARGET_WORDS) -> list[list[dict]]:
    """Group chunk manifest entries into document pages of ~target_words."""
    pages: list[list[dict]] = []
    current: list[dict] = []
    current_words = 0
    for entry in manifest:
        if current and current_words + entry["w"] > target_words:
            pages.append(current)
            current, current_words = [], 0
        current.append(entry)
        current_words += entry["w"]
    if current:
        pages.append(current)
    return pages


def read_parts_ranges(tmp_dir: Path, scan_id: str, entries: list[dict]) -> str:
    """Read the byte ranges for a page's chunks from the parts file."""
    parts = _parts_path(tmp_dir, scan_id)
    out: list[str] = []
    with open(parts, "rb") as pf:
        for e in entries:
            pf.seek(e["o"])
            out.append(pf.read(e["l"]).decode("utf-8"))
    return "".join(out)


_DOC_ICON_SVG = """<svg width="16" height="20" viewBox="0 0 16 20" fill="none" aria-hidden="true">
<path d="M2.5 1.5h6.5l4 4v12a1 1 0 0 1-1 1h-9.5a1 1 0 0 1-1-1v-15a1 1 0 0 1 1-1z" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1.2"/>
<path d="M9 1.5v4h4" fill="#ffffff" stroke="#94a3b8" stroke-width="1.2"/>
<line x1="4.5" y1="10" x2="11.5" y2="10" stroke="#94a3b8" stroke-width="1.2"/>
<line x1="4.5" y1="13" x2="11.5" y2="13" stroke="#94a3b8" stroke-width="1.2"/>
<line x1="4.5" y1="16" x2="9.5" y2="16" stroke="#94a3b8" stroke-width="1.2"/>
</svg>"""


def _details_page_html(
    *,
    title_no_ext: str,
    filename: str,
    submission_id: str,
    submission_date: str,
    download_date: str,
    file_size_str: str,
    pages: int,
    words: int,
    chars: int,
) -> str:
    """PAGE 1 of the report: Document Details (own layout, NAK branding only)."""
    kv = [
        ("Submission ID", submission_id),
        ("Submission Date", submission_date),
        ("Download Date", download_date),
        ("File Name", filename),
        ("File Size", file_size_str),
    ]
    kv_html = "\n".join(
        f'<div class="kv"><div class="k">{_escape(k)}</div>'
        f'<div class="v">{_escape(v)}</div></div>'
        for k, v in kv
    )
    return f"""<div class="doc-title">{_escape(title_no_ext)}</div>
<div class="doc-sub">{_DOC_ICON_SVG}<span>{_escape(filename)}</span></div>
<hr class="rule">
<div class="details-heading">Document Details</div>
<div class="details-cols">
  <div class="details-left">
{kv_html}
  </div>
  <div class="statbox">
    <div class="statline">{pages:,} Pages</div>
    <div class="statline">{words:,} Words</div>
    <div class="statline">{chars:,} Characters</div>
  </div>
</div>
"""


def assemble_report(
    *,
    scan_id: str,
    filename: str,
    words: int,
    chars: int,
    ai_pct: float,
    pages: int,
    file_size_bytes: int,
    submission_id: str,
    submission_ts: float,
    short_notice: bool,
    tmp_dir: Path,
    reports_dir: Path,
    deletes_at_ms: int,
    sim_pct: float = 0.0,
    sim_lib_pct: float = 0.0,
    sim_doc_pct: float = 0.0,
    sim_sources: list | None = None,
    store_repo: bool = False,
    bib_excluded: int = 0,
    quote_excluded: int = 0,
    cited_excluded: int = 0,
    small_filtered: int = 0,
    ai_status: str = "scored",
    ai_min_words: int = 300,
) -> Path:
    """Write the final paginated report HTML (plus AI-only and
    similarity-only variants for the separate PDF downloads).

    Order: 1) Document Details cover, 2) AI summary, 3+) full document with
    AI chunks highlighted, then the Similarity section: cover page with the
    overall similarity % and ranked matched sources, followed by the
    document with color-coded per-source highlights. Every page carries the
    NAK AI Detector header (official logo) and a "Page i of N" footer with
    the submission ID.
    """
    from app import plag as plagmod

    reports_dir.mkdir(parents=True, exist_ok=True)
    download_date = fmt_dt(datetime.now(_PKT).timestamp())
    title_no_ext = Path(filename).stem or filename
    sim_sources = sim_sources or []

    manifest = read_manifest(tmp_dir, scan_id)
    doc_pages = paginate_manifest(manifest)
    sim_manifest = plagmod.read_sim_manifest(tmp_dir, scan_id)
    sim_pages = paginate_manifest(sim_manifest)

    # similarity-section notes: repository choice + exclusion filters
    sim_notes: list[str] = []
    if store_repo:
        sim_notes.append(
            "This document was saved to the comparison repository and may "
            "match future scans.")
    else:
        sim_notes.append(
            "This document was NOT saved to the repository (no-repository "
            "mode), so re-scanning it will not match itself.")
    if bib_excluded > 0:
        sim_notes.append(
            f"Bibliography excluded ({bib_excluded:,} words) — not counted "
            "in the similarity %, not highlighted.")
    if quote_excluded > 0:
        sim_notes.append(
            f"Quoted text excluded ({quote_excluded:,} words) — not counted "
            "in the similarity %, not highlighted.")
    if cited_excluded > 0:
        sim_notes.append(
            f"Cited text excluded ({cited_excluded:,} words) — not counted "
            "in the similarity %, not highlighted.")
    if small_filtered > 0:
        sim_notes.append(
            f"Small matches excluded ({small_filtered:,} words) — not counted "
            "in the similarity %, not highlighted.")

    # Collect pages as (kind, kind_label, body); kinds: details / ai / sim
    details_body = _details_page_html(
        title_no_ext=title_no_ext,
        filename=filename,
        submission_id=submission_id,
        submission_date=fmt_dt(submission_ts),
        download_date=download_date,
        file_size_str=human_size(file_size_bytes),
        pages=pages,
        words=words,
        chars=chars,
    )
    ai_summary_body = _ai_summary_html(
        ai_pct=ai_pct, deletes_at_ms=deletes_at_ms, short_notice=short_notice,
        ai_status=ai_status, ai_min_words=ai_min_words,
    )
    sim_cover_body = plagmod.sim_cover_html(
        sim_pct=sim_pct, lib_pct=sim_lib_pct, doc_pct=sim_doc_pct,
        sources=sim_sources, total_words=words, notes=sim_notes,
    )
    page_specs: list[tuple[str, str, str]] = [
        ("details", "Cover Page", details_body),
        ("ai", "AI Detection Report", ai_summary_body),
    ]
    for entries in doc_pages:
        page_specs.append(
            ("ai", "AI Detection Report",
             read_parts_ranges(tmp_dir, scan_id, entries)))
    page_specs.append(("sim", "Similarity Report", sim_cover_body))
    for entries in sim_pages:
        page_specs.append(
            ("sim", "Similarity Report",
             '<div id="simdoc">'
             + plagmod.read_sim_ranges(tmp_dir, scan_id, entries)
             + "</div>"))

    def _write_variant(path: Path, kinds: set[str]) -> None:
        sel = [p for p in page_specs if p[0] in kinds]
        total = len(sel)
        with open(path, "w", encoding="utf-8") as rf:
            rf.write(_REPORT_HEAD)
            rf.write(f'<div class="psubholder">{_escape(submission_id)}</div>\n')
            rf.write(
                f'<div class="toolbar">'
                f'<a class="dlbtn" href="/report/{_escape(scan_id)}/pdf/ai">'
                f"&#8681; Download AI Report (PDF)</a>\n"
                f'<a class="dlbtn" href="/report/{_escape(scan_id)}/pdf/similarity">'
                f"&#8681; Download Similarity Report (PDF)</a></div>\n"
            )
            for i, (kind, kind_label, body) in enumerate(sel, start=1):
                rf.write(_report_page(
                    num=i, total=total, kind=kind, kind_label=kind_label,
                    submission_id=submission_id, body=body))

    report_path = reports_dir / f"{scan_id}.html"
    _write_variant(report_path, {"details", "ai", "sim"})
    _write_variant(reports_dir / f"{scan_id}.ai.html", {"details", "ai"})
    _write_variant(reports_dir / f"{scan_id}.sim.html", {"details", "sim"})
    rf_tail = _REPORT_TAIL
    for p in (report_path,
              reports_dir / f"{scan_id}.ai.html",
              reports_dir / f"{scan_id}.sim.html"):
        with open(p, "a", encoding="utf-8") as rf:
            rf.write(rf_tail)

    reset_parts(tmp_dir, scan_id)  # parts + manifest no longer needed
    plagmod.cleanup_sim_parts(tmp_dir, scan_id)  # sim parts no longer needed
    plagmod.cleanup_plag_tmp(tmp_dir, scan_id)  # any leftover temp files
    return report_path



def display_pct(ai_pct: float) -> str:
    """Headline percentage display logic (exact spec):
    - exactly 0        -> "0%"
    - above 0, below 20 -> "*%" (suppressed: higher false-positive risk)
    - 20 or above      -> actual rounded percentage, e.g. "47%"
    """
    if ai_pct == 0:
        return "0%"
    if ai_pct < 20:
        return "*%"
    return f"{int(round(ai_pct))}%"


def _ai_summary_html(*, ai_pct: float | None, deletes_at_ms: int,
                     short_notice: bool = False, ai_status: str = "scored",
                     ai_min_words: int = 300) -> str:
    """PAGE 2: Ultra-premium 'Clarity' design. Light, airy, Apple-grade
    minimalism with dramatic score typography. All wording original."""
    if ai_status == "not_scored_too_short":
        return f"""<div class="clx-wrap">
  <div class="clx-eyebrow">NAK AI Detector &mdash; Forensic Report</div>
  <h1 class="clx-h1">AI Content Analysis</h1>
  <div class="clx-score-zone">
    <div class="clx-notscored">Not scored</div>
    <div class="clx-notscored-sub">Document too short for reliable AI detection</div>
  </div>
  <p class="clx-lead">AI scoring needs at least {ai_min_words} words of prose.
  The similarity (plagiarism) check ran in full.</p>
</div>"""
    _ap = ai_pct if ai_pct is not None else 0.0
    headline = display_pct(_ap)
    # Sophisticated tier colors
    if _ap >= 80:
        c1, c2, rl, rc = "#dc2626", "#991b1b", "High AI likelihood", "xh"
    elif _ap >= 50:
        c1, c2, rl, rc = "#d97706", "#b45309", "Moderate AI likelihood", "xm"
    elif _ap >= 20:
        c1, c2, rl, rc = "#0284c7", "#0369a1", "Low AI likelihood", "xl"
    elif _ap > 0:
        c1, c2, rl, rc = "#64748b", "#475569", "Minimal AI signals", "xn"
    else:
        c1, c2, rl, rc = "#16a34a", "#15803d", "No AI detected", "xz"
    import math
    circ = 2 * math.pi * 92
    filled = circ * min(100.0, max(0.0, _ap)) / 100.0
    suppressed = ""
    if _ap != 0 and _ap < 20:
        suppressed = ('<div class="clx-suppressed">Exact score withheld &mdash; '
                      'scores below 20% carry higher false-positive risk.</div>')
    notice = ""
    if short_notice:
        notice = ('<div class="shortnotice">Too short for reliable detection '
                  "(minimum 300 words of prose)</div>")
    pct_show = headline if not (_ap != 0 and _ap < 20) else "*%"
    # Verdict sentence
    if _ap >= 80:
        verdict = ("Strong AI-generated signals were found throughout this document. "
                   "Careful human review of the highlighted passages is strongly advised.")
    elif _ap >= 50:
        verdict = ("A mix of human and AI-like writing patterns was detected. "
                   "Review the highlighted sections in context.")
    elif _ap >= 20:
        verdict = ("Light AI-like patterns were found. These may reflect ordinary "
                   "writing style as much as AI assistance.")
    elif _ap > 0:
        verdict = ("Only minimal AI-like signals were detected &mdash; well within "
                   "the range of ordinary human writing variation.")
    else:
        verdict = ("No AI-generated patterns were detected in this document. "
                   "The writing appears fully human.")
    return f"""{notice}<div class="clx-wrap">
  <div class="clx-eyebrow">NAK AI Detector &mdash; Forensic Report</div>
  <h1 class="clx-h1">AI Content Analysis</h1>
  <div class="clx-score-zone">
    <svg class="clx-ring" viewBox="0 0 220 220">
      <defs>
        <linearGradient id="clxg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{c1}"/><stop offset="100%" stop-color="{c2}"/>
        </linearGradient>
      </defs>
      <circle cx="110" cy="110" r="92" fill="none" stroke="#f1f5f9" stroke-width="10"/>
      <circle cx="110" cy="110" r="92" fill="none" stroke="url(#clxg)" stroke-width="10"
              stroke-linecap="round" stroke-dasharray="{filled:.1f} {circ:.1f}"
              transform="rotate(-90 110 110)"/>
    </svg>
    <div class="clx-score-text">
      <div class="clx-pct">{pct_show}</div>
      <div class="clx-pct-lbl">AI-generated</div>
    </div>
  </div>
  <div class="clx-verdict {rc}">{rl}</div>
  {suppressed}
  <p class="clx-verdict-text">{verdict}</p>
  <div class="clx-divider"><span></span></div>
  <div class="clx-cols">
    <div class="clx-col">
      <div class="clx-col-h">Understanding the score</div>
      <p>Our model scores every sentence and averages the result. AI detection is
      an <em>estimate</em> &mdash; not proof. Low scores are less certain, because
      ordinary writing can resemble AI text. No detector is free of error.</p>
    </div>
    <div class="clx-col">
      <div class="clx-col-h">What to do next</div>
      <p>Read the <span class="swatch-inline"></span> highlighted passages yourself.
      Consider the context and the writer&rsquo;s history. Never treat a score alone
      as evidence &mdash; human judgment is always required.</p>
    </div>
  </div>
</div>
<div class="clx-scale-card">
  <div class="clx-col-h" style="margin-bottom:12px">Score interpretation</div>
  <div class="clx-bar">
    <div class="clx-seg" style="background:#dcfce7;flex:20"><span>0%</span></div>
    <div class="clx-seg" style="background:#e0f2fe;flex:30"><span>20%</span></div>
    <div class="clx-seg" style="background:#fef3c7;flex:30"><span>50%</span></div>
    <div class="clx-seg" style="background:#fee2e2;flex:20"><span>80&ndash;100%</span></div>
    <div class="clx-pin" style="left:{min(100,max(0,_ap)):.1f}%"></div>
  </div>
  <div class="clx-bar-labels"><span>Human</span><span>Uncertain</span><span>Mixed</span><span>AI-like</span></div>
</div>
<div class="legend"><span class="swatch"></span>
  Sky-blue highlighting marks chunks the model flagged as AI-generated
  (probability &ge; 70%).</div>
<hr class="rule">
<div class="disclaimer">
  <div class="disclaimer-title">Disclaimer</div>
  <p>Our AI assessment may not always be accurate. Both false positives
  (human writing flagged as AI) and false negatives (AI writing missed) are
  possible. This report must not be the sole basis for any adverse decision
  &mdash; a grade, a hiring choice, or disciplinary action. Human judgment
  and careful review are always required.</p>
</div>
<div id="countdown" data-deletes-at="{deletes_at_ms}">
  This report auto-deletes in <b id="cd-time">--:--</b>
</div>
"""


def _logo_data_uri() -> str:
    """Official NAK AI Detector logo as a base64 data URI (self-contained report)."""
    import base64

    logo_path = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "media-generation-nak-ai-detector-logo-0-f792afd1-7bc0-47de-821f-2c86a5388273.webp"
    )
    try:
        return "data:image/webp;base64," + base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except OSError:
        return ""


_LOGO_DATA_URI = _logo_data_uri()


def _report_page(*, num: int, total: int, kind: str, kind_label: str, submission_id: str, body: str) -> str:
    """One paginated report page: shared header (official logo), body, footer."""
    return f"""<div class="rpage" data-kind="{kind}">
  <span class="plabelholder">{_escape(kind_label)}</span>
  <div class="rhead">
    <img class="rlogo" src="{_LOGO_DATA_URI}" alt="NAK AI Detector">
    <span class="rhead-right">Submission ID {_escape(submission_id)}</span>
  </div>
  <div class="rbody">
{body}
  </div>
  <div class="rfoot"><span>Page {num} of {total} - {kind_label}</span><span>NAK AI Detector</span></div>
</div>
"""


_REPORT_HEAD = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NAK AI Detector - Report</title>
<style>
/* ---- Print/PDF page geometry: US Letter, 0.5in content margins ---- */
@page {
  size: Letter;
  margin: 36pt 36pt 48pt 36pt;
  @top-left {
    content: url("__LOGO_HDR_URI__");
  }
  @top-right {
    content: "Page " counter(page) " of " counter(pages) " - " string(plabel);
    font-size: 6pt; color: #64748b;
  }
  @bottom-left {
    content: "Page " counter(page) " of " counter(pages) " - " string(plabel);
    font-size: 6pt; color: #64748b;
  }
  @bottom-right {
    content: "Submission ID " string(psub);
    font-size: 6pt; color: #64748b;
  }
}
/* per-page label for the running header/footer + document submission id */
.rpage .plabelholder { string-set: plabel content(); font-size: 0; line-height: 0; height: 0; margin: 0; display: block; }
.psubholder { string-set: psub content(); font-size: 0; line-height: 0; height: 0; margin: 0; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Roboto, Arial, sans-serif;
       background: #f1f5f9; color: #0f172a; margin: 0; padding: 24px; }
.page { max-width: 760px; margin: 0 auto; background: #fff; border-radius: 12px;
        padding: 40px; box-shadow: 0 4px 24px rgba(0,0,0,.08); }
/* ---- Paginated report chrome: header/footer on every page ---- */
.rpage { margin-bottom: 8px; }
.rhead { display: flex; justify-content: space-between; align-items: center;
         border-bottom: 0.5pt solid #cbd5e1; padding-bottom: 6pt; margin-bottom: 14pt; }
.rlogo { height: 20pt; width: auto; }
.rhead-right { color: #64748b; font-size: 6pt; }
.rbody { min-height: 120px; }
.rfoot { display: flex; justify-content: space-between; gap: 12pt;
         border-top: 0.5pt solid #cbd5e1; margin-top: 18pt; padding-top: 6pt;
         color: #64748b; font-size: 6pt; }
/* Cover page (Document Details): breathing room above the title block */
.rpage[data-kind="details"] .rbody { padding-top: 250pt; }
.rule { border: none; border-top: 1px solid #e2e8f0; margin: 26px 0; }
.twocol { display: flex; gap: 24px; text-align: left; align-items: flex-start; flex-wrap: wrap; }
.scorecol { flex: 1 1 280px; min-width: 0; }
.bigpct { font-size: 17pt; font-weight: 700; color: #0f172a; line-height: 1.15; }
/* ---- CLARITY: Elon-grade minimal AI report (clx) ---- */
.clx-wrap { text-align: center; padding: 20px 10px 10px; }
.clx-eyebrow { font-size: 10px; font-weight: 800; letter-spacing: 4px;
  color: #0284c7; margin-bottom: 10px; }
.clx-h1 { font-size: 30px; font-weight: 800; color: #0f172a; margin: 0 0 28px;
  letter-spacing: -1px; }
.clx-score-zone { position: relative; display: inline-block; margin-bottom: 18px; }
.clx-ring { width: 210px; height: 210px; }
.clx-score-text { position: absolute; inset: 0; display: flex;
  flex-direction: column; justify-content: center; align-items: center; }
.clx-pct { font-size: 52px; font-weight: 800; color: #0f172a; letter-spacing: -2px;
  line-height: 1; }
.clx-pct-lbl { font-size: 12px; color: #64748b; font-weight: 600; margin-top: 6px;
  letter-spacing: 1px; }
.clx-notscored { font-size: 28px; font-weight: 800; color: #0f172a; }
.clx-notscored-sub { font-size: 13px; color: #64748b; margin-top: 6px; }
.clx-verdict { display: inline-block; font-size: 17px; font-weight: 800;
  padding: 10px 28px; border-radius: 30px; margin-bottom: 14px; letter-spacing: .3px;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.clx-verdict.xh { background: #fee2e2; color: #991b1b; }
.clx-verdict.xm { background: #fef3c7; color: #92400e; }
.clx-verdict.xl { background: #e0f2fe; color: #0c4a6e; }
.clx-verdict.xn { background: #f1f5f9; color: #475569; }
.clx-verdict.xz { background: #dcfce7; color: #166534; }
.clx-suppressed { font-size: 12px; color: #94a3b8; margin-bottom: 10px; }
.clx-verdict-text { max-width: 560px; margin: 0 auto 8px; font-size: 15px;
  color: #334155; line-height: 1.7; }
.clx-lead { max-width: 520px; margin: 16px auto 0; font-size: 14px; color: #64748b;
  line-height: 1.7; }
.clx-divider { max-width: 560px; margin: 26px auto; }
.clx-divider span { display: block; height: 1px;
  background: linear-gradient(90deg, transparent, #cbd5e1, transparent); }
.clx-cols { display: flex; gap: 28px; text-align: left; max-width: 640px;
  margin: 0 auto; flex-wrap: wrap; }
.clx-col { flex: 1 1 260px; min-width: 0; }
.clx-col-h { font-size: 13px; font-weight: 800; color: #0f172a; margin-bottom: 8px;
  letter-spacing: .5px; }
.clx-col p { font-size: 13.5px; color: #475569; line-height: 1.7; margin: 0; }
.swatch-inline { display: inline-block; width: 12px; height: 12px; border-radius: 3px;
  background: __AIHL__; vertical-align: -1px;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
/* interpretation bar */
.clx-scale-card { max-width: 640px; margin: 26px auto 0; background: #f8fafc;
  border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px 24px; text-align: left; }
.clx-bar { position: relative; display: flex; height: 34px; border-radius: 10px;
  overflow: visible;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.clx-seg { position: relative; display: flex; align-items: center; justify-content: center;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.clx-seg:first-child { border-radius: 10px 0 0 10px; }
.clx-seg:last-child { border-radius: 0 10px 10px 0; }
.clx-seg span { font-size: 10px; font-weight: 700; color: rgba(15,23,42,.55); }
.clx-pin { position: absolute; top: -6px; width: 5px; height: 46px; background: #0f172a;
  border-radius: 3px; transform: translateX(-2.5px);
  box-shadow: 0 2px 8px rgba(15,23,42,.4); }
.clx-bar-labels { display: flex; justify-content: space-between; margin-top: 10px;
  font-size: 11px; color: #64748b; font-weight: 600; }
.suppressed-note { font-size: 12px; color: #94a3b8; margin: 6px 0 0; }
.explain { color: #475569; font-size: 14px; line-height: 1.65; margin: 12px 0 0; }
.callout { flex: 1 1 240px; min-width: 0; background: __AIHL__; border-radius: 12px; padding: 18px 20px;
           -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.callout-title { font-weight: 800; font-size: 16px; margin-bottom: 8px; color: #0f172a; }
.callout p { font-size: 14px; line-height: 1.6; margin: 0; color: #0f172a; }
.disclaimer { text-align: left; }
.disclaimer-title { font-weight: 800; font-size: 16px; margin-bottom: 8px; }
.disclaimer p { font-size: 13px; color: #475569; line-height: 1.65; margin: 0; }
/* ---- Page 1: Document Details ---- */
.doc-title { font-size: 15pt; font-weight: 700; color: #0f172a; line-height: 1.25;
             word-break: break-word; }
.doc-sub { display: flex; align-items: center; gap: 8px; color: #64748b;
           font-size: 9pt; margin-top: 6pt; }
.rule { border: none; border-top: 0.5pt solid #cbd5e1; margin: 18pt 0; }
.details-heading { font-size: 10pt; font-weight: 700; color: #0f172a; margin: 4pt 0 16pt; }
.details-cols { display: flex; gap: 32px; align-items: flex-start; flex-wrap: wrap; }
.details-left { flex: 1 1 240px; min-width: 0; }
.kv { margin-bottom: 13pt; }
.kv .k { font-size: 7pt; font-weight: 600; color: #64748b; margin-bottom: 2pt; }
.kv .v { font-size: 10pt; color: #0f172a; font-weight: 400; word-break: break-word; }
.statbox { min-width: 200px; flex: 1 1 200px; padding-top: 1pt; }
.statline { font-size: 10pt; font-weight: 400; color: #0f172a; margin-bottom: 11pt; }
.statline:last-child { margin-bottom: 0; }
.legend { margin: 22px auto 0; max-width: 520px; font-size: 13px; color: #475569;
          background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; }
.simnotes { margin: 18px auto 0; max-width: 520px; }
.simnote { font-size: 13px; color: #0c4a6e; background: #e0f2fe;
           border: 1px solid #bae6fd; border-radius: 8px; padding: 8px 12px;
           margin-bottom: 8px; }
.swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px;
          background: __AIHL__; vertical-align: -2px; margin-right: 6px;
          border: 1px solid #94a3b8; }
#countdown { margin: 26px auto 0; max-width: 520px; text-align: center; font-size: 14px;
             color: #b45309; background: #fef3c7; border: 1px solid #fcd34d;
             border-radius: 8px; padding: 10px 14px; }
.shortnotice { background: #fef3c7; border: 2px solid #f59e0b; border-radius: 10px;
               padding: 14px 18px; font-size: 16px; font-weight: 800; color: #92400e;
               text-align: center; margin-bottom: 22px;
               -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.ablock { padding: 8px 12px; border-radius: 6px; margin: 4px 0; line-height: 1.65; }
.ablock p { margin: 0 0 10px; } .ablock p:last-child { margin-bottom: 0; }
.ablock h1 { font-size: 24px; margin: 14px 0 8px; } .ablock h1:first-child { margin-top: 0; }
.ablock h2 { font-size: 20px; margin: 12px 0 6px; }
.ablock h3 { font-size: 17px; margin: 10px 0 6px; }
.ablock ul, .ablock ol { margin: 0 0 10px 20px; padding: 0; }
.sai { background: __AIHL__; border-radius: 3px; padding: 1px 2px;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
table.doctable { border-collapse: collapse; margin: 8px 0; width: 100%; }
table.doctable td { border: 1px solid #cbd5e1; padding: 6px 10px; font-size: 14px; }
.report-foot { margin-top: 30px; text-align: center; color: #94a3b8; font-size: 12px;
               border-top: 1px solid #e2e8f0; padding-top: 14px; }
@media print {
  body { background: #fff; padding: 0; }
  .page { box-shadow: none; border-radius: 0; max-width: none; padding: 0; }
  .rpage { page-break-after: always; }
  .rpage:last-child { page-break-after: auto; }
  /* running header/footer come from @page margin boxes in print */
  .rhead, .rfoot { display: none; }
  #countdown { display: none; }
  .toolbar { display: none; }
  .report-foot { display: none; }
  .callout { margin-top: 0; }
  .ablock { font-size: 10pt; line-height: 1.5; }
}
/* ---- Download toolbar (screen only, hidden in print/PDF) ---- */
@media screen {
.toolbar { display: flex; justify-content: flex-end; margin-bottom: 18px; }
.dlbtn { display: inline-block; background: #0284c7; color: #fff; font-weight: 800;
         padding: 13px 26px; border-radius: 10px; text-decoration: none; font-size: 16px;
         box-shadow: 0 2px 10px rgba(2,132,199,.35); }
.dlbtn:hover { background: #0369a1; }
}
@media (max-width: 640px) {
  .twocol { flex-direction: column; }
  .details-cols { flex-direction: column; }
  .doc-title { font-size: 26px; }
  .clx-h1 { font-size: 24px; }
  .clx-pct { font-size: 42px; }
  .clx-cols { flex-direction: column; }
}
/* ---- Sticky section mini-nav (screen only, built by JS in _REPORT_TAIL) ---- */
.repnav { position: sticky; top: 0; z-index: 50; display: flex; gap: 8px;
  flex-wrap: wrap; align-items: center; background: rgba(255,255,255,.97);
  border-bottom: 1px solid #e2e8f0; padding: 10px 14px; margin: -24px -24px 18px; }
.repnav .rn-label { font-size: 11.5px; font-weight: 800; color: #64748b;
  text-transform: uppercase; letter-spacing: .6px; margin-right: 2px; }
.repnav a { font-size: 13.5px; font-weight: 700; color: #0369a1; text-decoration: none;
  padding: 8px 14px; border-radius: 999px; background: #f0f9ff;
  border: 1px solid #bae6fd; white-space: nowrap; }
.repnav a:hover { background: #e0f2fe; }
.rpage { scroll-margin-top: 78px; }
@media print { .repnav { display: none; } }
/* ---- Report responsive: mobile-first polish (highlight colors untouched) ---- */
@media (max-width: 640px) {
  body { padding: 10px; }
  .repnav { margin: -10px -10px 14px; padding: 8px 10px; }
  .bigpct { font-size: 30px; }
  .doc-title { font-size: 24px; }
  .statbox { min-width: 0; width: 100%; }
  .rhead { flex-wrap: wrap; gap: 8px; }
  .rlogo { height: 36px; }
  .kv .v { font-size: 14px; }
  .toolbar { flex-direction: column; align-items: stretch; }
  .dlbtn { width: 100%; text-align: center; }
  .rfoot { font-size: 11px; }
  .details-heading { font-size: 18px; }
  .ablock h1 { font-size: 20px; }
  .ablock h2 { font-size: 17px; }
  .ablock h3 { font-size: 15px; }
  .callout { padding: 14px 16px; }
  #countdown { font-size: 13px; }
}
</style></head>
<body><div class="page">
""".replace("__AIHL__", AI_HIGHLIGHT)

# Similarity-report styles (own palette); appended so the single <style>
# block keeps everything WeasyPrint needs for the PDF export.
from app import plag as _plagmod

_REPORT_HEAD = _REPORT_HEAD.replace("</style>", _plagmod.sim_css() + "</style>", 1)

# File URI of the official logo for the @page margin-box header (print/PDF).
_LOGO_FILE_URI = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "media-generation-nak-ai-detector-logo-0-f792afd1-7bc0-47de-821f-2c86a5388273.webp"
).as_uri()
_LOGO_HDR_URI = (
    Path(__file__).resolve().parent.parent / "assets" / "logo_header.png"
).as_uri()
_REPORT_HEAD = _REPORT_HEAD.replace("__LOGO_URI__", _LOGO_FILE_URI)
_REPORT_HEAD = _REPORT_HEAD.replace("__LOGO_HDR_URI__", _LOGO_HDR_URI)

_REPORT_TAIL = """
<div class="report-foot">NAK AI Detector &mdash; own model, honest results.</div>
</div>
<script>
(function () {
  var el = document.getElementById('countdown');
  var t = document.getElementById('cd-time');
  var end = parseInt(el.getAttribute('data-deletes-at'), 10);
  function tick() {
    var ms = end - Date.now();
    if (ms <= 0) { t.textContent = 'expired'; return; }
    var s = Math.floor(ms / 1000);
    var m = Math.floor(s / 60); s = s % 60;
    t.textContent = String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
    setTimeout(tick, 1000);
  }
  tick();
})();
/* Sticky section mini-nav: one link per report section kind (Details / AI / Similarity). */
(function () {
  var pages = document.querySelectorAll('.rpage');
  if (!pages.length) return;
  var nav = document.createElement('nav');
  nav.className = 'repnav';
  nav.setAttribute('aria-label', 'Report sections');
  var lab = document.createElement('span');
  lab.className = 'rn-label';
  lab.textContent = 'Sections';
  nav.appendChild(lab);
  var seen = {};
  pages.forEach(function (p, i) {
    p.id = 'rpage-' + (i + 1);
    var foot = p.querySelector('.rfoot span');
    var kind = foot ? (foot.textContent.split(' - ').pop() || '').trim() : '';
    if (!kind || seen[kind]) return;
    seen[kind] = true;
    var a = document.createElement('a');
    a.href = '#rpage-' + (i + 1);
    a.textContent = kind;
    nav.appendChild(a);
  });
  var page = document.querySelector('.page');
  if (page && page.firstChild) page.insertBefore(nav, page.firstChild);
})();
</script>
</body></html>
"""
