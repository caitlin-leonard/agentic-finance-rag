"""Corpus loaders.

Offline / CI: read the bundled illustrative JSONL corpus (hermetic, no network).
Online: the same `records_to_chunks` path accepts records pulled from real SEC
EDGAR filings (see scripts/fetch_edgar.py) — the schema is identical.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import Chunk
from .chunker import chunk_records

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_sample_corpus() -> list[dict]:
    return _read_jsonl(DATA_DIR / "sample_filings.jsonl")


def load_corpus_file(path: str) -> list[dict]:
    """Load an arbitrary JSONL corpus (e.g. real filings from fetch_edgar.py)."""
    return _read_jsonl(Path(path))


def load_eval_set() -> list[dict]:
    return _read_jsonl(DATA_DIR / "financebench_sample.jsonl")


def records_to_chunks(records: list[dict]) -> list[Chunk]:
    return chunk_records(records)
