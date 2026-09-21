"""RAG evaluation metrics.

Two implementations behind one interface:

* offline / CI  -> deterministic, judge-free proxies of the standard RAG metrics
                   (token-overlap based). Hermetic and reproducible, so the CI
                   quality-gate is stable.
* provider != offline -> real Ragas + DeepEval LLM-judged metrics (see
                   `ragas_adapter.py`), which is what you report on the resume.

The four metrics mirror the RAG evaluation canon:
  - faithfulness      : is the answer supported by the retrieved context?
  - answer_relevancy  : does the answer address the question?
  - context_precision : are the retrieved chunks on-topic?
  - context_recall    : did we retrieve what the ground truth needs?
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from ..offline_nlp import overlap, tokenize


@dataclass
class EvalCase:
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    refused: bool = False


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    per_case: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "n_cases": len(self.per_case),
        }


def score_case(case: EvalCase) -> dict:
    """Deterministic proxy metrics for a single case."""
    joined_ctx = " ".join(case.contexts)
    faithfulness = overlap(case.answer, joined_ctx)
    answer_relevancy = overlap(case.ground_truth, case.answer)
    # context precision: fraction of contexts sharing tokens with ground truth
    gt_tokens = set(tokenize(case.ground_truth))
    hits = sum(1 for c in case.contexts if gt_tokens & set(tokenize(c)))
    context_precision = hits / len(case.contexts) if case.contexts else 0.0
    # context recall: fraction of ground-truth tokens present in the contexts
    ctx_tokens = set(tokenize(joined_ctx))
    context_recall = (len(gt_tokens & ctx_tokens) / len(gt_tokens)
                      if gt_tokens else 0.0)
    return {
        "question": case.question,
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(answer_relevancy, 4),
        "context_precision": round(context_precision, 4),
        "context_recall": round(context_recall, 4),
        "refused": case.refused,
    }


def aggregate(cases: list[EvalCase]) -> EvalResult:
    scored = [score_case(c) for c in cases]

    def m(key: str) -> float:
        vals = [s[key] for s in scored]
        return mean(vals) if vals else 0.0

    return EvalResult(
        faithfulness=m("faithfulness"),
        answer_relevancy=m("answer_relevancy"),
        context_precision=m("context_precision"),
        context_recall=m("context_recall"),
        per_case=scored,
    )
