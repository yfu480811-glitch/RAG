from tracerag.citations import assign_citations, format_sources_text


def test_citation_numbering_and_sources_format() -> None:
    chunks = [
        {
            "chunk_id": "c1",
            "text": "alpha",
            "metadata": {"title": "DocA", "source": "a.md", "heading": "Intro", "page_number": None},
        },
        {
            "chunk_id": "c2",
            "text": "beta",
            "metadata": {"title": "DocB", "source": "b.pdf", "page_number": 2, "heading": None},
        },
    ]

    cited, sources = assign_citations(chunks)
    assert cited[0]["citation_tag"] == "[1]"
    assert cited[1]["citation_tag"] == "[2]"

    text = format_sources_text(sources)
    assert "[1] DocA - a.md - heading Intro - c1" in text
    assert "[2] DocB - b.pdf - page 2 - c2" in text
