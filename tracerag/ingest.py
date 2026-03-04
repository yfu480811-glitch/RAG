from __future__ import annotations

import hashlib
import re
from pathlib import Path

from tracerag.chunking import chunk_text
from tracerag.models import Chunk, Document, SourceType, TextSegment
from tracerag.storage import SQLiteStore

SUPPORTED_EXTS = {".pdf", ".md", ".markdown", ".txt"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_pdf(path: Path) -> list[TextSegment]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    segments: list[TextSegment] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append(TextSegment(text=text, page_number=i))
    return segments


def read_markdown(path: Path) -> list[TextSegment]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    segments: list[TextSegment] = []
    current_heading = "root"
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            segments.append(TextSegment(text="\n".join(buffer).strip(), heading=current_heading))
            buffer.clear()

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if m:
            flush()
            current_heading = m.group(2).strip()
            continue
        buffer.append(line)

    flush()
    return [seg for seg in segments if seg.text]


def read_txt(path: Path) -> list[TextSegment]:
    text = path.read_text(encoding="utf-8")
    return [TextSegment(text=text, heading="root")]


def read_html_from_url(url: str, timeout_s: float = 15.0) -> tuple[str | None, list[TextSegment]]:
    import httpx
    from bs4 import BeautifulSoup

    resp = httpx.get(url, timeout=timeout_s)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = "\n".join(line.strip() for line in main.get_text("\n").splitlines() if line.strip())
    return title, [TextSegment(text=text, heading="web")]


def parse_source(source: str) -> tuple[SourceType, str | None, list[TextSegment]]:
    if source.startswith("http://") or source.startswith("https://"):
        title, segments = read_html_from_url(source)
        return "html", title, segments

    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf", path.name, read_pdf(path)
    if suffix in {".md", ".markdown"}:
        return "markdown", path.name, read_markdown(path)
    if suffix == ".txt":
        return "txt", path.name, read_txt(path)
    raise ValueError(f"unsupported source: {source}")


def build_chunks(
    *,
    doc_id: int,
    source: str,
    source_type: SourceType,
    title: str | None,
    segments: list[TextSegment],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    for seg in segments:
        for piece in chunk_text(seg.text, chunk_size=chunk_size, overlap=overlap):
            chunk_id = f"doc{doc_id}_c{chunk_index}"
            metadata = {
                "source": source,
                "source_type": source_type,
                "title": title,
                "page_number": seg.page_number,
                "heading": seg.heading,
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=piece.text,
                    metadata=metadata,
                )
            )
            chunk_index += 1
    return chunks


def ingest_source(
    source: str,
    store: SQLiteStore,
    chunk_size: int = 1000,
    overlap: int = 180,
) -> dict:
    source_type, title, segments = parse_source(source)
    merged_text = "\n".join(seg.text for seg in segments)
    content_hash = sha256_text(merged_text)

    existing = store.get_document_by_source(source)
    if existing and existing.content_hash == content_hash:
        store.log_ingest_run(source, "skipped", "unchanged hash")
        return {"status": "skipped", "reason": "unchanged", "doc_id": existing.doc_id, "chunks": 0}

    if existing:
        doc_id = int(existing.doc_id)
        store.update_document(doc_id=doc_id, content_hash=content_hash, title=title)
        chunks = build_chunks(
            doc_id=doc_id,
            source=source,
            source_type=source_type,
            title=title,
            segments=segments,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        store.replace_chunks(doc_id, chunks)
        store.log_ingest_run(source, "updated", f"chunks={len(chunks)}")
        return {"status": "updated", "doc_id": doc_id, "chunks": len(chunks)}

    doc_id = store.insert_document(
        Document(doc_id=None, source=source, source_type=source_type, content_hash=content_hash, title=title)
    )
    chunks = build_chunks(
        doc_id=doc_id,
        source=source,
        source_type=source_type,
        title=title,
        segments=segments,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    store.replace_chunks(doc_id, chunks)
    store.log_ingest_run(source, "ingested", f"chunks={len(chunks)}")
    return {"status": "ingested", "doc_id": doc_id, "chunks": len(chunks)}


def collect_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files: list[Path] = []
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return sorted(files)
