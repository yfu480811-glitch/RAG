from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(default=6, ge=1, le=20)


class SourceItem(BaseModel):
    index: int
    title: str | None = None
    source: str
    location: str
    chunk_id: str


class ChatSyncResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    sources: list[SourceItem]
