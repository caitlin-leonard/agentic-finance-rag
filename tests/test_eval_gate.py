"""The RAG quality gate as a pytest test (offline proxy metrics)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("FINRAG_PROVIDER", "offline")

from finrag.config import get_settings  # noqa: E402
from finrag.eval.run_eval import run_eval  # noqa: E402


def test_quality_gate_passes():
    result, failures = run_eval(out_path="eval_report.json")
    s = get_settings()
    assert result.faithfulness >= s.eval_min_faithfulness, result.as_dict()
    assert result.answer_relevancy >= s.eval_min_answer_relevancy, result.as_dict()
    assert result.context_recall >= s.eval_min_context_recall, result.as_dict()
    assert not failures
