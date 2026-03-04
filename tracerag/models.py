from dataclasses import dataclass, field
from typing import Literal

SourceType = Literal["pdf", "markdown", "txt", "html"]


@dataclass(slots=True)
class Document:
    doc_id: int | None
    source: str
    source_type: SourceType
    content_hash: str
    title: str | None = None


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: int
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class TextSegment:
    text: str
    page_number: int | None = None
    heading: str | None = None
