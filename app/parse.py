"""Streaming document parsers for NAK AI Detector.

Each parser is a generator yielding *units*:
    {"tag": "p"|"h1"|"h2"|"h3"|"li-bullet"|"li-number"|"table",
     "html": inner HTML (already escaped), "text": plain text}

Design notes:
- .txt is read line-by-line (buffered), so a 100MB file never sits in RAM.
- .pdf is read page-by-page with pypdf (best-effort paragraphs).
- .docx is parsed with python-docx preserving headings, bold/italic/
  underline, lists and simple tables. (python-docx necessarily holds the
  document in memory; uploads are still streamed to disk first.)
"""

from __future__ import annotations

import html
import re
import subprocess
import tempfile
from pathlib import Path


def parse_doc(path: Path):
    """Parse old binary .doc via LibreOffice conversion to .docx."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert .doc -> .docx using LibreOffice headless
        result = subprocess.run(
            ["soffice", "--headless", "--convert-to", "docx",
             "--outdir", tmpdir, str(path)],
            capture_output=True, timeout=120,
        )
        # Find the converted file
        converted = list(Path(tmpdir).glob("*.docx"))
        if not converted:
            raise ValueError(
                "could not convert .doc file - it may be corrupted; "
                "try saving as .docx in Word")
        yield from parse_docx(converted[0])

ALLOWED_EXTS = {".txt", ".pdf", ".doc", ".docx"}


def _flush_para(lines: list[str]):
    text = " ".join(l.strip() for l in lines).strip()
    if not text:
        return None
    m = re.match(r"^(#{1,3})\s+(.*)$", text)
    if m:
        level = len(m.group(1))
        body = m.group(2).strip()
        return {"tag": f"h{level}", "html": html.escape(body), "text": body}
    return {"tag": "p", "html": html.escape(text), "text": text}


def parse_txt(path: Path):
    """Stream a .txt file line-by-line; paragraphs split on blank lines."""
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:  # buffered iterator: never loads the whole file
            if line.strip():
                lines.append(line.rstrip("\n"))
            elif lines:
                unit = _flush_para(lines)
                lines = []
                if unit:
                    yield unit
    if lines:
        unit = _flush_para(lines)
        if unit:
            yield unit


def parse_pdf(path: Path):
    """Page-by-page PDF text extraction; best-effort paragraph recovery."""
    from pypdf import PdfReader

    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text = page.extract_text() or ""
            lines: list[str] = []
            for line in text.splitlines():
                if line.strip():
                    lines.append(line.strip())
                elif lines:
                    unit = _flush_para(lines)
                    lines = []
                    if unit:
                        yield unit
            if lines:
                unit = _flush_para(lines)
                if unit:
                    yield unit


def _runs_html(paragraph) -> tuple[str, str]:
    parts: list[str] = []
    texts: list[str] = []
    for run in paragraph.runs:
        t = run.text or ""
        if not t.strip():
            continue
        texts.append(t)
        e = html.escape(t)
        if run.bold:
            e = f"<b>{e}</b>"
        if run.italic:
            e = f"<i>{e}</i>"
        if run.underline:
            e = f"<u>{e}</u>"
        parts.append(e)
    return "".join(parts), "".join(texts)


def _para_tag(paragraph) -> str:
    try:
        name = (paragraph.style.name or "").lower()
    except Exception:
        name = ""
    if name.startswith("heading 1") or name == "title":
        return "h1"
    if name.startswith("heading 2"):
        return "h2"
    if name.startswith("heading 3"):
        return "h3"
    if "bullet" in name:
        return "li-bullet"
    if "number" in name:
        return "li-number"
    return "p"


def _table_unit(table) -> dict | None:
    rows_html: list[str] = []
    texts: list[str] = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            ct = " ".join(p.text for p in cell.paragraphs).strip()
            texts.append(ct)
            cells.append(f"<td>{html.escape(ct)}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    if not any(texts):
        return None
    return {
        "tag": "table",
        "html": '<table class="doctable">' + "".join(rows_html) + "</table>",
        "text": " ".join(t for t in texts if t),
    }


def parse_docx(path: Path):
    """python-docx parse preserving structure, in document order."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    body = doc.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            inner_html, text = _runs_html(p)
            if not text.strip():
                continue
            yield {"tag": _para_tag(p), "html": inner_html, "text": text.strip()}
        elif tag == "tbl":
            unit = _table_unit(Table(child, doc))
            if unit:
                yield unit


def parse_document(path: Path, ext: str):
    """Dispatch to the right parser. Raises ValueError for unsupported types."""
    ext = ext.lower()
    if ext == ".txt":
        yield from parse_txt(path)
    elif ext == ".pdf":
        yield from parse_pdf(path)
    elif ext == ".docx":
        yield from parse_docx(path)
    elif ext == ".doc":
        yield from parse_doc(path)
    else:
        raise ValueError(f"unsupported file type: {ext}")
