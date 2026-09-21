"""Run the eval set through the pipeline, score it, and enforce a quality gate.

This is what CI runs. It exits non-zero if any metric falls below its
configured threshold, so a regression in retrieval or generation *fails the
build* — "AI validation baked into CI/CD", not a manual notebook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..config import Settings, get_settings
from ..graph import RagPipeline
from ..ingest.loader import load_eval_set
from .metrics import EvalCase, EvalResult, aggregate


def _build_cases(pipe: RagPipeline, eval_rows: list[dict]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for row in eval_rows:
        trace = pipe.query(row["question"])
        cases.append(EvalCase(
            question=row["question"],
            ground_truth=row["ground_truth"],
            answer=trace.answer,
            contexts=[c.snippet for c in trace.citations] or
                     [r.chunk.text for r in trace.retrieved],
            refused=trace.refused,
        ))
    return cases


def _score(cases: list[EvalCase], settings: Settings) -> EvalResult:
    if settings.provider == "offline":
        return aggregate(cases)
    from .ragas_adapter import ragas_scores

    return ragas_scores(cases)


def _gate(result: EvalResult, s: Settings) -> list[str]:
    failures = []
    if result.faithfulness < s.eval_min_faithfulness:
        failures.append(
            f"faithfulness {result.faithfulness:.3f} < {s.eval_min_faithfulness}")
    if result.answer_relevancy < s.eval_min_answer_relevancy:
        failures.append(
            f"answer_relevancy {result.answer_relevancy:.3f} < "
            f"{s.eval_min_answer_relevancy}")
    if result.context_recall < s.eval_min_context_recall:
        failures.append(
            f"context_recall {result.context_recall:.3f} < "
            f"{s.eval_min_context_recall}")
    return failures


def run_eval(out_path: str = "eval_report.json") -> tuple[EvalResult, list[str]]:
    s = get_settings()
    pipe = RagPipeline(s)
    pipe.ingest()
    rows = load_eval_set()
    cases = _build_cases(pipe, rows)
    result = _score(cases, s)
    failures = _gate(result, s)

    report = {
        "provider": s.provider,
        "summary": result.as_dict(),
        "thresholds": {
            "faithfulness": s.eval_min_faithfulness,
            "answer_relevancy": s.eval_min_answer_relevancy,
            "context_recall": s.eval_min_context_recall,
        },
        "passed": not failures,
        "failures": failures,
        "per_case": result.per_case,
    }
    Path(out_path).write_text(json.dumps(report, indent=2))
    return result, failures


def main() -> None:
    result, failures = run_eval()
    print(json.dumps(result.as_dict(), indent=2))
    if failures:
        print("\nQUALITY GATE FAILED:", file=sys.stderr)
        for f in failures:
            print("  -", f, file=sys.stderr)
        sys.exit(1)
    print("\n✅ quality gate passed")


if __name__ == "__main__":
    main()
