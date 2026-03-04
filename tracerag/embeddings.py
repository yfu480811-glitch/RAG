from __future__ import annotations

from abc import ABC, abstractmethod

from tracerag.config import settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Install dependencies to use local embeddings."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in vectors]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OpenAI embedding provider requires TRACERAG_OPENAI_API_KEY in environment.")
        raise NotImplementedError("OpenAIEmbeddingProvider is a placeholder for Milestone 3.")


def get_embedding_provider(provider: str | None = None) -> EmbeddingProvider:
    target = (provider or settings.embedding_provider).lower()
    if target == "openai":
        return OpenAIEmbeddingProvider()
    return SentenceTransformerEmbeddingProvider()
