"""Assemble the LangGraph StateGraph and wrap it in a callable pipeline.

Topology:

    route ─┬─(chitchat)→ chitchat ─────────────────────────────→ END
           └─(financial_qa)→ retrieve → grade_documents ─┐
                                                         │ enough?
              ┌───────────── insufficient ←──────────────┤
              ▼                                          ▼ generate
        self_correct → grade_documents                generate → guardrail
                                                         │ grounded?
                              accept ──────────────→ END │
                              retry  ──→ self_correct ───┘
                              refuse ──→ refuse ─────→ END
"""
from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..providers import LLMClient, get_llm
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.store import VectorStore, build_store
from ..ingest import load_sample_corpus, records_to_chunks
from ..ingest.loader import load_corpus_file
from ..schema import Citation, QueryTrace
from .nodes import Nodes
from .state import GraphState


def build_graph(retriever: HybridRetriever, llm: LLMClient, settings: Settings):
    n = Nodes(retriever, llm, settings)
    g = StateGraph(GraphState)

    g.add_node("router", n.route)
    g.add_node("chitchat", n.chitchat)
    g.add_node("scope_check", n.scope_check)
    g.add_node("retrieve", n.retrieve)
    g.add_node("grade_documents", n.grade_documents)
    g.add_node("generate", n.generate)
    g.add_node("guardrail", n.guardrail)
    g.add_node("self_correct", n.self_correct)
    g.add_node("refuse", n.refuse)

    g.set_entry_point("router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"chitchat": "chitchat",
                             "financial_qa": "scope_check"})
    g.add_edge("chitchat", END)
    g.add_conditional_edges("scope_check", n.in_scope,
                            {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", n.enough_context,
                            {"generate": "generate",
                             "insufficient": "self_correct"})
    g.add_edge("self_correct", "grade_documents")
    g.add_edge("generate", "guardrail")
    g.add_conditional_edges("guardrail", n.is_grounded,
                            {"accept": END,
                             "retry": "self_correct",
                             "refuse": "refuse"})
    g.add_edge("refuse", END)
    return g.compile()


class RagPipeline:
    """High-level entry point used by the API, UI and eval harness."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.store: VectorStore = build_store(self.s)
        self.llm = get_llm(self.s)
        self._ingested = False

    def ingest(self, records: list[dict] | None = None) -> int:
        if records is None:
            records = (load_corpus_file(self.s.corpus_path)
                       if self.s.corpus_path else load_sample_corpus())
        chunks = records_to_chunks(records)
        self.store.add(chunks)
        self.retriever = HybridRetriever(self.store, self.s)
        self.app = build_graph(self.retriever, self.llm, self.s)
        self._ingested = True
        return len(chunks)

    def _ensure(self) -> None:
        if not self._ingested:
            self.ingest()

    def query(self, question: str) -> QueryTrace:
        self._ensure()
        t0 = time.perf_counter()
        final: GraphState = self.app.invoke(
            {"question": question, "self_corrections": 0}
        )
        latency = (time.perf_counter() - t0) * 1000

        graded = final.get("graded", [])
        citations = [
            Citation(chunk_id=r.chunk.chunk_id, citation=r.chunk.citation,
                     snippet=r.chunk.text[:220])
            for r in graded
        ]
        return QueryTrace(
            question=question,
            answer=final.get("answer", ""),
            route=final.get("route", "financial_qa"),
            citations=citations,
            groundedness=final.get("groundedness", 0.0),
            answer_relevance=final.get("answer_relevance", 0.0),
            context_grade=final.get("context_grade", 0.0),
            self_corrections=final.get("self_corrections", 0),
            refused=final.get("refused", False),
            latency_ms=round(latency, 1),
            prompt_tokens=final.get("prompt_tokens", 0),
            completion_tokens=final.get("completion_tokens", 0),
            retrieved=graded,
        )
