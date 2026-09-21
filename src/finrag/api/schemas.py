"""API request/response models (Pydantic v2)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)


class CitationOut(BaseModel):
    chunk_id: str
    citation: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str
    refused: bool
    groundedness: float
    answer_relevance: float
    context_grade: float
    self_corrections: int
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    est_cost_usd: float
    citations: list[CitationOut]


class IngestRequest(BaseModel):
    # Optional inline records; if omitted the bundled sample corpus is loaded.
    records: list[dict] | None = None


class IngestResponse(BaseModel):
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    provider: str
    chunks_indexed: int
