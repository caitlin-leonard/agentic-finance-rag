"""Typed state threaded through the LangGraph nodes."""
from __future__ import annotations

from typing import TypedDict

from ..schema import RetrievedChunk


class GraphState(TypedDict, total=False):
    # inputs
    question: str
    # routing
    route: str
    scope_ok: bool
    # retrieval
    retrieved: list[RetrievedChunk]
    graded: list[RetrievedChunk]
    context_grade: float
    # generation
    answer: str
    prompt_tokens: int
    completion_tokens: int
    # guardrails
    groundedness: float
    answer_relevance: float
    refused: bool
    self_corrections: int
