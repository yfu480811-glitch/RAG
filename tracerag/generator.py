from __future__ import annotations

from abc import ABC, abstractmethod

from tracerag.citations import format_sources_text

PROMPT_TEMPLATE = """你是企业级 RAG 助手。请严格遵守：
1) 只使用给定 context 回答，禁止编造。
2) 若 context 不足，明确回答“我不知道”。
3) 每个关键结论后必须附上引用编号，如 [1][2]。
4) 引用编号只能来自给定 chunk 编号。

问题：{query}

Context:
{context}

请输出：
- 简明答案（含引用）
- Sources 列表（每条: [i] title - source - page/heading - chunk_id）
"""


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, cited_chunks: list[dict], sources: list[dict]) -> str:
        raise NotImplementedError


class OfflineGenerator(BaseGenerator):
    """Rule-based generator for offline/demo/testing."""

    def generate(self, query: str, cited_chunks: list[dict], sources: list[dict]) -> str:
        if not cited_chunks:
            return "我不知道。当前没有检索到可用证据。"

        lines = [f"针对问题“{query}”，基于检索证据可得："]
        for chunk in cited_chunks[:4]:
            snippet = " ".join(chunk.get("text", "").split())[:140]
            lines.append(f"- {snippet}。{chunk['citation_tag']}")

        lines.append("\nSources:")
        lines.append(format_sources_text(sources))
        return "\n".join(lines)


class LLMGenerator(BaseGenerator):
    """Optional external-LLM generator placeholder."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def generate(self, query: str, cited_chunks: list[dict], sources: list[dict]) -> str:
        if not self.enabled:
            raise RuntimeError("LLMGenerator not enabled. Use OfflineGenerator or enable external provider.")
        raise NotImplementedError("LLMGenerator provider wiring is reserved for next step.")


def build_context(cited_chunks: list[dict]) -> str:
    parts = []
    for c in cited_chunks:
        meta = c.get("metadata", {})
        parts.append(
            f"{c['citation_tag']} source={meta.get('source')} heading={meta.get('heading')} "
            f"page={meta.get('page_number')}\n{c.get('text', '')}"
        )
    return "\n\n".join(parts)
