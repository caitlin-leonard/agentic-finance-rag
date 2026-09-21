"""Real Ragas + DeepEval scoring (used when provider != offline).

Kept isolated so the heavy, network-bound eval libraries are only imported when
an LLM judge is actually available. This is the code path that produces the
head-line numbers you cite ("faithfulness 0.9x on FinanceBench").
"""
from __future__ import annotations

from .metrics import EvalCase, EvalResult


def ragas_scores(cases: list[EvalCase]) -> EvalResult:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy, context_precision, context_recall, faithfulness,
    )

    ds = Dataset.from_dict({
        "question": [c.question for c in cases],
        "answer": [c.answer for c in cases],
        "contexts": [c.contexts for c in cases],
        "ground_truth": [c.ground_truth for c in cases],
    })
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = result.to_pandas()
    return EvalResult(
        faithfulness=float(df["faithfulness"].mean()),
        answer_relevancy=float(df["answer_relevancy"].mean()),
        context_precision=float(df["context_precision"].mean()),
        context_recall=float(df["context_recall"].mean()),
        per_case=df.to_dict(orient="records"),
    )


def deepeval_cases(cases: list[EvalCase]):
    """Build DeepEval test cases + metrics for use inside pytest.

    DeepEval's `assert_test` integrates cleanly with pytest so RAG quality
    checks become first-class CI tests, not a separate script.
    """
    from deepeval.metrics import (
        AnswerRelevancyMetric, FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    tc = [
        LLMTestCase(
            input=c.question, actual_output=c.answer,
            expected_output=c.ground_truth, retrieval_context=c.contexts,
        )
        for c in cases
    ]
    metrics = [FaithfulnessMetric(threshold=0.8),
               AnswerRelevancyMetric(threshold=0.75)]
    return tc, metrics
