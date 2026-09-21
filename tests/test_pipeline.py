"""Unit + behavioural tests for the pipeline (offline provider)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("FINRAG_PROVIDER", "offline")

from finrag.graph import RagPipeline  # noqa: E402
from finrag.retrieval.hybrid import parse_filters  # noqa: E402


@pytest.fixture(scope="module")
def pipe():
    p = RagPipeline()
    p.ingest()
    return p


def test_ingest_indexes_chunks(pipe):
    assert pipe.store.all_chunks(), "corpus should be indexed"


def test_grounded_answer_has_citations(pipe):
    t = pipe.query("What was Acme Robotics total net revenue in fiscal 2023?")
    assert not t.refused
    assert "12,480" in t.answer
    assert t.groundedness >= 0.7
    assert t.citations


def test_metadata_filter_extraction():
    flt = parse_filters("What was GLOB revenue in 2023 risk factors?",
                        {"GLOB", "ACME"})
    assert flt.ticker == "GLOB"
    assert flt.year == 2023
    assert flt.section == "Item 1A - Risk Factors"


def test_out_of_scope_is_refused(pipe):
    t = pipe.query("What was Acme free cash flow in 1901?")
    assert t.refused


def test_chitchat_route(pipe):
    t = pipe.query("hello there")
    assert t.route == "chitchat"
    assert not t.refused
