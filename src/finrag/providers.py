"""Provider abstraction: embeddings + LLM, swappable across
offline / OpenAI / Azure OpenAI / Anthropic behind one interface.

The graph, retriever and eval never touch a vendor SDK directly — they call
`get_embeddings()` / `get_llm()`. That indirection is what makes the same code
run on a laptop with `provider=offline` and in production on Azure AI Foundry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings
from . import offline_nlp as onlp


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class HashingEmbeddings(Embeddings):
    """Deterministic offline embeddings (LangChain-compatible)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [onlp.hash_embed(t, self.dim) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return onlp.hash_embed(text, self.dim)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.provider == "offline":
        return HashingEmbeddings(s.embedding_dim)
    if s.provider in ("openai", "azure"):
        from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

        if s.provider == "azure":
            return AzureOpenAIEmbeddings(
                azure_endpoint=s.azure_openai_endpoint,
                api_key=s.azure_openai_api_key,
                api_version=s.azure_openai_api_version,
                model=s.embedding_model,
            )
        return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)
    if s.provider == "anthropic":
        # Anthropic has no first-party embeddings; use OpenAI embeddings alongside.
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)
    raise ValueError(f"unknown provider {s.provider}")


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: dict = field(default_factory=dict)


class LLMClient:
    """Thin task-oriented wrapper. Each method is a single logical LLM call.

    Node code reads the same whether backed by a real model or the offline
    heuristics, which keeps the *orchestration* honest and vendor-neutral.
    """

    # ---- generation ----
    def answer(self, question: str, contexts: list[str]) -> LLMResult: ...
    # ---- judges (LLM-as-judge online; heuristics offline) ----
    def grade_document(self, question: str, doc: str) -> bool: ...
    def score_groundedness(self, answer: str, contexts: list[str]) -> float: ...
    def score_answer_relevance(self, question: str, answer: str) -> float: ...
    def route(self, question: str) -> str: ...


class OfflineLLM(LLMClient):
    def answer(self, question: str, contexts: list[str]) -> LLMResult:
        text = onlp.extractive_answer(question, contexts)
        return LLMResult(text=text, prompt_tokens=sum(len(c) for c in contexts) // 4,
                         completion_tokens=len(text) // 4)

    def grade_document(self, question: str, doc: str) -> bool:
        return onlp.overlap(question, doc) >= 0.15

    def score_groundedness(self, answer: str, contexts: list[str]) -> float:
        joined = " ".join(contexts)
        return round(onlp.overlap(answer, joined), 3)

    def score_answer_relevance(self, question: str, answer: str) -> float:
        return round(onlp.overlap(question, answer), 3)

    def route(self, question: str) -> str:
        ql = question.lower()
        if any(w in ql for w in ("hi", "hello", "who are you", "thanks")):
            return "chitchat"
        return "financial_qa"


_ANSWER_SYS = (
    "You are a meticulous financial analyst. Answer ONLY using the provided "
    "context from SEC filings. Every figure must appear in the context. If the "
    "context is insufficient, say so explicitly. Cite the source id in [brackets]."
)


class RealLLM(LLMClient):
    """Backed by a LangChain chat model (OpenAI / Azure / Anthropic)."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.model = self._build_model(settings)

    @staticmethod
    def _build_model(s: Settings):
        if s.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=s.llm_model, api_key=s.anthropic_api_key,
                                 temperature=0)
        from langchain_openai import AzureChatOpenAI, ChatOpenAI

        if s.provider == "azure":
            return AzureChatOpenAI(
                azure_endpoint=s.azure_openai_endpoint,
                api_key=s.azure_openai_api_key,
                api_version=s.azure_openai_api_version,
                azure_deployment=s.llm_model,
                temperature=0,
            )
        return ChatOpenAI(model=s.llm_model, api_key=s.openai_api_key, temperature=0)

    def _call(self, system: str, user: str) -> LLMResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = self.model.invoke([SystemMessage(system), HumanMessage(user)])
        usage = getattr(resp, "usage_metadata", None) or {}
        return LLMResult(
            text=resp.content if isinstance(resp.content, str) else str(resp.content),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )

    def answer(self, question: str, contexts: list[str]) -> LLMResult:
        ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
        return self._call(_ANSWER_SYS, f"Context:\n{ctx}\n\nQuestion: {question}")

    def grade_document(self, question: str, doc: str) -> bool:
        out = self._call(
            "You grade whether a document is relevant to a question. "
            "Reply with only 'yes' or 'no'.",
            f"Question: {question}\n\nDocument: {doc}\n\nRelevant?",
        )
        return out.text.strip().lower().startswith("y")

    def score_groundedness(self, answer: str, contexts: list[str]) -> float:
        ctx = "\n\n".join(contexts)
        out = self._call(
            "Score 0.0-1.0 how fully the answer is supported by the context. "
            "Reply with only the number.",
            f"Context:\n{ctx}\n\nAnswer: {answer}",
        )
        return _parse_float(out.text)

    def score_answer_relevance(self, question: str, answer: str) -> float:
        out = self._call(
            "Score 0.0-1.0 how well the answer addresses the question. "
            "Reply with only the number.",
            f"Question: {question}\n\nAnswer: {answer}",
        )
        return _parse_float(out.text)

    def route(self, question: str) -> str:
        out = self._call(
            "Classify the user message as 'financial_qa' or 'chitchat'. "
            "Reply with only the label.",
            question,
        )
        return "chitchat" if "chitchat" in out.text.lower() else "financial_qa"


def _parse_float(text: str) -> float:
    import re

    m = re.search(r"[01](?:\.\d+)?", text)
    return float(m.group()) if m else 0.0


def get_llm(settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    return OfflineLLM() if s.provider == "offline" else RealLLM(s)
