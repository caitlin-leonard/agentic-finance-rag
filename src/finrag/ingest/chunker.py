"""Structure-aware chunking for filings.

Financial filings mix long narrative (MD&A, risk factors) with dense numeric
tables. Splitting a table with a generic character splitter destroys the
row/column context a downstream answer depends on. So we branch on `kind`:

* ``table``  -> kept as a single atomic chunk (never split).
* ``prose``  -> recursive character splitting with overlap so figures that
                straddle a boundary still appear whole in at least one chunk.
"""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..schema import Chunk

_PROSE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_records(records: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rec in records:
        kind = rec.get("kind", "prose")
        base = {k: rec[k] for k in ("company", "ticker", "year", "section")
                if k in rec}
        base.setdefault("ticker", rec.get("ticker", "UNK"))
        if kind == "table":
            chunks.append(Chunk(text=rec["text"].strip(), kind="table", **base))
            continue
        for piece in _PROSE_SPLITTER.split_text(rec["text"]):
            piece = piece.strip()
            if len(piece) < 40:  # drop fragments
                continue
            chunks.append(Chunk(text=piece, kind="prose", **base))
    return chunks
