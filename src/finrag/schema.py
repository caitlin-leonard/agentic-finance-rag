"""Shared data structures passed between ingestion, retrieval and the graph."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    """One retrievable unit of a filing.

    The structured metadata (company/year/section/kind) is first-class, not an
    afterthought — it powers SQL pre-filtering so a question about "Apple 2022
    risk factors" never retrieves Microsoft 2019 boilerplate.
    """

    text: str
    company: str
    ticker: str
    year: int
    section: str            # e.g. "Item 7 - MD&A", "Item 1A - Risk Factors"
    kind: str = "prose"     # "prose" | "table"
    doc_id: str = ""
    chunk_id: str = ""
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id:
            h = hashlib.sha1(
                f"{self.ticker}|{self.year}|{self.section}|{self.text[:64]}".encode()
            ).hexdigest()[:16]
            self.chunk_id = h
        if not self.doc_id:
            self.doc_id = f"{self.ticker}-{self.year}"

    @property
    def citation(self) -> str:
        return f"{self.ticker} {self.year} · {self.section}"

    def to_metadata(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    source: str  # "dense" | "lexical" | "fused"


@dataclass
class Citation:
    chunk_id: str
    citation: str
    snippet: str


@dataclass
class QueryTrace:
    """Everything the UI/observability layer needs to explain one answer."""

    question: str
    answer: str
    route: str = "financial_qa"
    citations: list[Citation] = field(default_factory=list)
    groundedness: float = 0.0
    answer_relevance: float = 0.0
    context_grade: float = 0.0
    self_corrections: int = 0
    refused: bool = False
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieved: list[RetrievedChunk] = field(default_factory=list)
