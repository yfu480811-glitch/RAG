from __future__ import annotations


def _build_location(meta: dict) -> str:
    page = meta.get("page_number")
    heading = meta.get("heading")
    if page is not None:
        return f"page {page}"
    if heading:
        return f"heading {heading}"
    return "n/a"


def assign_citations(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    cited_chunks: list[dict] = []
    sources: list[dict] = []

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        cited = {**chunk, "citation_index": i, "citation_tag": f"[{i}]"}
        cited_chunks.append(cited)
        sources.append(
            {
                "index": i,
                "title": meta.get("title"),
                "source": meta.get("source") or chunk.get("source", "unknown"),
                "location": _build_location(meta),
                "chunk_id": chunk.get("chunk_id", "unknown"),
            }
        )

    return cited_chunks, sources


def format_sources_text(sources: list[dict]) -> str:
    lines = []
    for s in sources:
        title = s.get("title") or "untitled"
        lines.append(f"[{s['index']}] {title} - {s['source']} - {s['location']} - {s['chunk_id']}")
    return "\n".join(lines)
