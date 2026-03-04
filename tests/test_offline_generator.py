from tracerag.generator import OfflineGenerator


def test_offline_generator_contains_citation_marker() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "text": "TraceRAG uses hybrid retrieval over chunks.",
            "metadata": {"source": "doc.txt", "title": "Doc", "heading": "Intro", "page_number": None},
            "citation_tag": "[1]",
        }
    ]
    sources = [{"index": 1, "title": "Doc", "source": "doc.txt", "location": "heading Intro", "chunk_id": "c1"}]

    out = OfflineGenerator().generate("What is TraceRAG?", chunks, sources)
    assert "[1]" in out
