"""Load notes (PDF, Markdown, text) and split them into overlapping chunks.

Each chunk keeps where it came from (file, page, nearest heading) so answers and
flashcards can cite it.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    section: str

    def citation(self) -> str:
        parts = [self.source]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(f"§ {self.section}")
        return ", ".join(parts)

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("id")
        d["page"] = -1 if self.page is None else self.page  # Chroma metadata can't be None
        return d


def load_pages(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    """Return [(page_number or None, text)]."""
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if name.endswith((".md", ".markdown", ".txt")):
        return [(None, data.decode("utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type: {filename} (use .pdf, .md or .txt)")


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def _paragraphs(text: str):
    """Yield (heading, paragraph) pairs, tracking the most recent Markdown heading."""
    heading = ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        first = block.splitlines()[0]
        m = _HEADING.match(first)
        if m:
            heading = m.group(1).strip()
            rest = "\n".join(block.splitlines()[1:]).strip()
            if rest:
                yield heading, rest
            continue
        yield heading, re.sub(r"[ \t]+", " ", block)


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # prefer to break at a sentence end
        cut = text.rfind(". ", start + size // 2, end)
        if cut != -1 and end < len(text):
            end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_document(filename: str, data: bytes, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    doc_hash = hashlib.sha1(data).hexdigest()[:10]
    chunks: list[Chunk] = []
    for page, text in load_pages(filename, data):
        buf, buf_heading = "", ""
        pieces: list[tuple[str, str]] = []
        for heading, para in _paragraphs(text):
            if heading != buf_heading and buf:  # new section -> flush
                pieces.append((buf_heading, buf))
                buf = ""
            buf_heading = heading
            if len(buf) + len(para) + 2 <= size:
                buf = f"{buf}\n\n{para}" if buf else para
            else:
                if buf:
                    pieces.append((buf_heading, buf))
                    tail = buf[-overlap:] if overlap else ""
                    tail = tail[tail.find(" ") + 1:] if " " in tail else tail
                    buf = f"{tail}\n\n{para}" if tail else para
                else:
                    buf = para
                if len(buf) > size:
                    parts = _split_long(buf, size, overlap)
                    pieces.extend((buf_heading, p) for p in parts[:-1])
                    buf = parts[-1]
        if buf:
            pieces.append((buf_heading, buf))
        for heading, piece in pieces:
            if len(piece.strip()) < 40:  # skip page numbers, stray headers
                continue
            chunks.append(Chunk(id=f"{doc_hash}-{len(chunks)}", text=piece.strip(),
                                source=filename, page=page, section=heading))
    return chunks
