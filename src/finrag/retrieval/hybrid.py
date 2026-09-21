"""Hybrid retrieval: dense + lexical, fused with Reciprocal Rank Fusion.

Why hybrid: dense embeddings capture paraphrase ("top line" ~ "revenue") but
miss exact tokens; BM25 nails exact tokens (tickers, "2.47", "Item 1A") but
misses paraphrase. Financial questions need both. RRF fuses the two ranked
lists without needing calibrated scores, which is robust and standard.

A light query-understanding step first extracts hard metadata filters (ticker,
year) so retrieval is pre-scoped with SQL before any similarity is computed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ..config import Settings, get_settings
from ..offline_nlp import tokenize
from ..schema import Chunk, RetrievedChunk
from .store import VectorStore


@dataclass
class MetadataFilter:
    ticker: str | None = None
    year: int | None = None
    section: str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items() if v is not None}


_SECTION_HINTS = {
    "risk": "Item 1A - Risk Factors",
    "md&a": "Item 7 - MD&A",
    "management discussion": "Item 7 - MD&A",
    "balance sheet": "Item 8 - Financial Statements",
    "cash flow": "Item 8 - Financial Statements",
    "income statement": "Item 8 - Financial Statements",
}


def parse_filters(question: str, known_tickers: set[str],
                  company_map: dict[str, str] | None = None) -> MetadataFilter:
    """Extract ticker / year / section constraints from the question.

    Matches both the ticker symbol (``GLOB``) and the company name the user is
    far more likely to type (``Globex``), so the metadata pre-filter actually
    fires on natural questions. Online this can be an LLM extraction call; the
    deterministic version here keeps retrieval cheap and predictable.
    """
    flt = MetadataFilter()
    up = question.upper()
    for tk in known_tickers:
        if re.search(rf"\b{re.escape(tk)}\b", up):
            flt.ticker = tk
            break
    if flt.ticker is None and company_map:
        low = question.lower()
        for name_word, tk in company_map.items():
            if re.search(rf"\b{re.escape(name_word)}\b", low):
                flt.ticker = tk
                break
    m = re.search(r"\b(19|20)\d{2}\b", question)
    if m:
        flt.year = int(m.group())
    low = question.lower()
    for hint, section in _SECTION_HINTS.items():
        if hint in low:
            flt.section = section
            break
    return flt


# Capitalized words that are NOT company names — so we don't mistake them for an
# out-of-corpus entity. Question words, finance vocab, section words.
_COMMON_CAPS = {
    "what", "how", "much", "many", "was", "were", "did", "does", "is", "are",
    "the", "a", "an", "in", "of", "for", "and", "or", "which", "when", "who",
    "give", "tell", "show", "list", "compare", "why", "whats", "its",
    "revenue", "margin", "income", "fiscal", "year", "net", "gross", "cash",
    "flow", "free", "risk", "factors", "sales", "profit", "earnings", "share",
    "assets", "debt", "equity", "operating", "total", "percentage", "company",
    "quarter", "growth", "data", "center", "cost", "expense", "dividend",
}


def detect_unknown_company(question: str, company_map: dict[str, str],
                           known_tickers: set[str]) -> bool:
    """True if the question names a company that isn't in the loaded corpus.

    This is the entity-scope guardrail: without it, asking about "Apple" when
    only Globex/Acme are indexed would answer from the wrong company's filing
    (the text is grounded, just about the wrong entity). If a known company IS
    referenced we're fine; if an *unrecognised* proper noun appears, we treat
    the question as out of scope and let the graph refuse.
    """
    low = question.lower()
    for name_word in company_map:
        if re.search(rf"\b{re.escape(name_word)}\b", low):
            return False
    up = question.upper()
    for tk in known_tickers:
        if re.search(rf"\b{re.escape(tk)}\b", up):
            return False
    # No known entity matched. Is there a capitalised proper noun that looks
    # like a company name (and isn't just a question/finance word)?
    for tok in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", question):
        base = tok.replace("'s", "").replace("’s", "").lower()
        if base not in _COMMON_CAPS:
            return True
    return False


def _rrf(rank: int, k: int) -> float:
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    def __init__(self, store: VectorStore, settings: Settings | None = None) -> None:
        self.store = store
        self.s = settings or get_settings()
        chunks = store.all_chunks()
        self._known_tickers = {c.ticker for c in chunks}
        # map distinctive company-name words -> ticker (e.g. "globex" -> GLOB,
        # "apple" -> AAPL) so questions phrased with the company name still filter.
        self._company_map: dict[str, str] = {}
        for c in chunks:
            for word in c.company.lower().replace(",", "").split():
                if word in {"inc", "inc.", "corp", "corp.", "group", "ab",
                            "co", "co.", "ltd", "plc", "the", "&"}:
                    continue
                self._company_map.setdefault(word, c.ticker)

    # -- lexical index is built lazily over the (optionally filtered) corpus -- #
    def _bm25(self, flt: dict | None):
        corpus = self.store.all_chunks(flt)
        tokenized = [tokenize(c.text) for c in corpus]
        # BM25Okapi needs a non-empty corpus; guard degenerate cases.
        return (BM25Okapi(tokenized) if tokenized else None), corpus

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        flt = parse_filters(question, self._known_tickers,
                            self._company_map).as_dict()

        # 1) dense
        dense = self.store.search(question, self.s.top_k_dense, flt)
        # 2) lexical
        bm25, corpus = self._bm25(flt)
        lexical: list[tuple[Chunk, float]] = []
        if bm25 is not None:
            scores = bm25.get_scores(tokenize(question))
            ranked = sorted(zip(corpus, scores), key=lambda x: -x[1])
            lexical = ranked[: self.s.top_k_lexical]

        # 3) Reciprocal Rank Fusion
        fused: dict[str, tuple[Chunk, float]] = {}
        for rank, (c, _) in enumerate(dense):
            fused[c.chunk_id] = (c, _rrf(rank, self.s.rrf_k))
        for rank, (c, _) in enumerate(lexical):
            prev = fused.get(c.chunk_id)
            add = _rrf(rank, self.s.rrf_k)
            fused[c.chunk_id] = (c, (prev[1] if prev else 0.0) + add)

        ordered = sorted(fused.values(), key=lambda x: -x[1])
        results = [RetrievedChunk(chunk=c, score=round(s, 5), source="fused")
                   for c, s in ordered]

        if self.s.use_reranker:
            results = self._rerank(question, results)
        return results[: self.s.top_k_final]

    def _rerank(self, question: str, results: list[RetrievedChunk]
                ) -> list[RetrievedChunk]:
        """Cross-encoder rerank hook (online). No-op ordering offline."""
        try:
            from sentence_transformers import CrossEncoder  # optional dep

            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(question, r.chunk.text) for r in results]
            for r, sc in zip(results, ce.predict(pairs)):
                r.score = float(sc)
            results.sort(key=lambda r: -r.score)
        except Exception:
            pass
        return results
