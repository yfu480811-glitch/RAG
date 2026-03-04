from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TRACERAG_", extra="ignore")

    app_name: str = "TraceRAG"
    host: str = "127.0.0.1"
    port: int = 8000

    data_dir: Path = Field(default=Path("./data"))
    db_path: Path = Field(default=Path("./data/tracerag.db"))

    chunk_size: int = 1000
    chunk_overlap: int = 180

    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    semantic_top_k: int = 8
    bm25_top_k: int = 8
    rrf_k: int = 60

    vector_index_path: Path = Field(default=Path("./data/index.faiss"))
    vector_map_path: Path = Field(default=Path("./data/index_map.json"))


settings = Settings()
