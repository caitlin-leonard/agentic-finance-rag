"""Deterministic, dependency-free NLP used by the *offline* provider.

The point of this module is that the entire agentic pipeline — retrieval,
grading, generation, guardrails, and even the LLM-as-judge eval — produces
*meaningful, reproducible* behaviour with no API key and no network. Online,
these heuristics are replaced by real model calls; the graph code is identical.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|\$[0-9,.]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "was",
    "were", "on", "at", "by", "with", "as", "that", "this", "it", "be", "from",
    "what", "which", "how", "much", "many", "did", "does", "do", "s",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def hash_embed(text: str, dim: int) -> list[float]:
    """Feature-hashing bag-of-words embedding.

    Unlike a random vector, this makes cosine similarity track lexical overlap,
    so offline retrieval genuinely ranks relevant chunks higher — the pipeline
    behaves plausibly end-to-end instead of returning noise.
    """
    vec = [0.0] * dim
    for tok, cnt in Counter(tokenize(text)).items():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(cnt))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def overlap(a: str, b: str) -> float:
    """Jaccard-ish token overlap of `a` against `b` (recall-weighted to `a`)."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def extractive_answer(question: str, contexts: list[str]) -> str:
    """Pick the sentence(s) from context most relevant to the question.

    A stand-in for an LLM's grounded generation: it can only ever return text
    that exists in the retrieved context, so the offline system is faithful by
    construction — useful for demoing the guardrail path without a model.
    """
    q_toks = set(tokenize(question))
    best: list[tuple[float, str]] = []
    for ctx in contexts:
        for sent in re.split(r"(?<=[.!?])\s+", ctx):
            score = len(q_toks & set(tokenize(sent)))
            if score:
                best.append((score, sent.strip()))
    best.sort(key=lambda x: -x[0])
    if not best:
        return "I don't have enough grounded information to answer that."
    top = [s for _, s in best[:2]]
    return " ".join(top)
