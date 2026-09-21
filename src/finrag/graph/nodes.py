"""Graph nodes. Each node is a pure-ish function State -> partial State.

The agentic behaviour lives in the *edges* (build.py): grade documents, and if
too few are relevant, reformulate and re-retrieve; generate, and if the answer
isn't grounded, self-correct; if it still isn't, refuse rather than hallucinate.
That refuse-instead-of-guess path is the point — in finance a confident wrong
number is worse than "I don't know."
"""
from __future__ import annotations

from ..config import Settings
from ..providers import LLMClient
from ..retrieval.hybrid import HybridRetriever, detect_unknown_company
from .state import GraphState


class Nodes:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient,
                 settings: Settings) -> None:
        self.retriever = retriever
        self.llm = llm
        self.s = settings

    # --------------------------------------------------------------------- #
    def route(self, state: GraphState) -> GraphState:
        return {"route": self.llm.route(state["question"]),
                "self_corrections": state.get("self_corrections", 0)}

    def chitchat(self, state: GraphState) -> GraphState:
        return {
            "answer": ("I'm a financial-filings assistant. Ask me about revenue, "
                       "margins, risk factors or other figures from the filings "
                       "in the knowledge base."),
            "groundedness": 1.0, "answer_relevance": 1.0, "refused": False,
            "graded": [], "retrieved": [],
        }

    def scope_check(self, state: GraphState) -> GraphState:
        unknown = detect_unknown_company(
            state["question"], self.retriever._company_map,
            self.retriever._known_tickers)
        return {"scope_ok": not unknown}

    def retrieve(self, state: GraphState) -> GraphState:
        return {"retrieved": self.retriever.retrieve(state["question"])}

    def grade_documents(self, state: GraphState) -> GraphState:
        q = state["question"]
        retrieved = state.get("retrieved", [])
        graded = [r for r in retrieved if self.llm.grade_document(q, r.chunk.text)]
        grade = (len(graded) / len(retrieved)) if retrieved else 0.0
        # keep at least the top fused hit so generation always has something
        if not graded and retrieved:
            graded = retrieved[:1]
        return {"graded": graded, "context_grade": round(grade, 3)}

    def generate(self, state: GraphState) -> GraphState:
        ctx = [r.chunk.text for r in state.get("graded", [])]
        res = self.llm.answer(state["question"], ctx)
        return {
            "answer": res.text,
            "prompt_tokens": state.get("prompt_tokens", 0) + res.prompt_tokens,
            "completion_tokens": state.get("completion_tokens", 0)
            + res.completion_tokens,
        }

    def guardrail(self, state: GraphState) -> GraphState:
        ctx = [r.chunk.text for r in state.get("graded", [])]
        g = self.llm.score_groundedness(state["answer"], ctx)
        a = self.llm.score_answer_relevance(state["question"], state["answer"])
        return {"groundedness": g, "answer_relevance": a}

    def self_correct(self, state: GraphState) -> GraphState:
        # widen retrieval on retry by dropping to the raw fused list again
        return {
            "self_corrections": state.get("self_corrections", 0) + 1,
            "retrieved": self.retriever.retrieve(state["question"]),
        }

    def refuse(self, state: GraphState) -> GraphState:
        if not state.get("scope_ok", True):
            msg = ("That company isn't in the filings I have indexed, so I can't "
                   "answer about it without inventing figures. Load its filing "
                   "(e.g. via scripts/fetch_edgar.py) and ask again.")
        else:
            msg = ("I couldn't find sufficiently grounded information in the "
                   "filings to answer that confidently, so I'm declining to "
                   "avoid reporting an unverified figure.")
        return {"answer": msg, "refused": True}

    # edge predicate
    def in_scope(self, state: GraphState) -> str:
        return "retrieve" if state.get("scope_ok", True) else "refuse"

    # --------------------------------------------------------------------- #
    # edge predicates
    def enough_context(self, state: GraphState) -> str:
        if state.get("context_grade", 0.0) >= self.s.min_context_grade:
            return "generate"
        # exhausted the reformulation budget -> generate anyway; the
        # groundedness guardrail downstream will refuse if it can't be supported.
        if state.get("self_corrections", 0) >= self.s.max_self_corrections:
            return "generate"
        return "insufficient"

    def is_grounded(self, state: GraphState) -> str:
        if state.get("groundedness", 0.0) >= self.s.min_groundedness:
            return "accept"
        if state.get("self_corrections", 0) < self.s.max_self_corrections:
            return "retry"
        return "refuse"
