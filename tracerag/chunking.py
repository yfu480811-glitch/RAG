from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChunkPiece:
    chunk_index: int
    text: str
    start: int
    end: int


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 180) -> list[ChunkPiece]:
    clean = " ".join(text.split())
    if not clean:
        return []

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[ChunkPiece] = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        piece = clean[start:end].strip()
        if piece:
            chunks.append(ChunkPiece(chunk_index=idx, text=piece, start=start, end=end))
            idx += 1
        start += step

    return chunks
