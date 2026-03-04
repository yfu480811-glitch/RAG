from tracerag.chunking import chunk_text


def test_chunk_text_overlap_and_count() -> None:
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 4
    assert chunks[0].start == 0
    assert chunks[1].start == 800


def test_chunk_text_invalid_window() -> None:
    try:
        chunk_text("abc", chunk_size=100, overlap=100)
        assert False, "expected ValueError"
    except ValueError:
        assert True
