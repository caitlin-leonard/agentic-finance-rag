"""Vector store abstraction with two backends.

* ``InMemoryVectorStore`` — numpy cosine search; the default, hermetic, used in
  tests and for a laptop demo.
* ``PgVectorStore`` — Postgres + pgvector; the production backend. Vectors and
  the *structured* filing metadata live in the same relational row, so a single
  SQL statement does metadata pre-filtering AND ANN search (``ORDER BY
  embedding <=> query``). That co-location is the whole reason to use pgvector
  over a bolt-on vector DB: one system, one transaction, real SQL joins.

Both honour a metadata filter (company / year / section), which is what stops a
"Globex 2023" question from ever surfacing an "Acme 2022" chunk.
"""
from __future__ import annotations

import math
from typing import Iterable, Protocol

from ..config import Settings, get_settings
from ..providers import get_embeddings
from ..schema import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, k: int, flt: dict | None = None
               ) -> list[tuple[Chunk, float]]: ...
    def all_chunks(self, flt: dict | None = None) -> list[Chunk]: ...


def _passes(chunk: Chunk, flt: dict | None) -> bool:
    if not flt:
        return True
    for key, val in flt.items():
        if val is None:
            continue
        if getattr(chunk, key, None) != val:
            return False
    return True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.embeddings = get_embeddings(self.s)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        vecs = self.embeddings.embed_documents(texts)
        for c, v in zip(chunks, vecs):
            c.embedding = v
            self._chunks.append(c)

    def search(self, query: str, k: int, flt: dict | None = None
               ) -> list[tuple[Chunk, float]]:
        qv = self.embeddings.embed_query(query)
        scored = [
            (c, _cosine(qv, c.embedding))
            for c in self._chunks
            if c.embedding is not None and _passes(c, flt)
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def all_chunks(self, flt: dict | None = None) -> list[Chunk]:
        return [c for c in self._chunks if _passes(c, flt)]


class PgVectorStore:
    """Production backend. Kept import-light so offline runs never touch psycopg."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.embeddings = get_embeddings(self.s)
        from sqlalchemy import create_engine

        self.engine = create_engine(self.s.database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        from sqlalchemy import text

        dim = self.s.embedding_dim
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id   TEXT PRIMARY KEY,
                    doc_id     TEXT NOT NULL,
                    company    TEXT, ticker TEXT, year INT,
                    section    TEXT, kind TEXT, text TEXT,
                    embedding  vector({dim})
                )"""))
            # HNSW index for fast ANN over cosine distance.
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks "
                "USING hnsw (embedding vector_cosine_ops)"))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS chunks_meta_idx ON chunks "
                "(ticker, year, section)"))

    def add(self, chunks: list[Chunk]) -> None:
        from sqlalchemy import text

        vecs = self.embeddings.embed_documents([c.text for c in chunks])
        with self.engine.begin() as conn:
            for c, v in zip(chunks, vecs):
                conn.execute(text("""
                    INSERT INTO chunks
                      (chunk_id, doc_id, company, ticker, year, section, kind, text, embedding)
                    VALUES (:cid,:did,:co,:tk,:yr,:sec,:kind,:txt,:emb)
                    ON CONFLICT (chunk_id) DO NOTHING
                """), dict(cid=c.chunk_id, did=c.doc_id, co=c.company, tk=c.ticker,
                           yr=c.year, sec=c.section, kind=c.kind, txt=c.text,
                           emb=str(v)))

    def search(self, query: str, k: int, flt: dict | None = None
               ) -> list[tuple[Chunk, float]]:
        from sqlalchemy import text

        qv = str(self.embeddings.embed_query(query))
        where, params = self._where(flt)
        sql = text(f"""
            SELECT chunk_id, doc_id, company, ticker, year, section, kind, text,
                   1 - (embedding <=> :qv) AS score
            FROM chunks {where}
            ORDER BY embedding <=> :qv LIMIT :k
        """)
        params.update(qv=qv, k=k)
        with self.engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
        return [(self._row_to_chunk(r), float(r["score"])) for r in rows]

    def all_chunks(self, flt: dict | None = None) -> list[Chunk]:
        from sqlalchemy import text

        where, params = self._where(flt)
        with self.engine.connect() as conn:
            rows = conn.execute(text(
                f"SELECT chunk_id, doc_id, company, ticker, year, section, kind, text "
                f"FROM chunks {where}"), params).mappings().all()
        return [self._row_to_chunk(r) for r in rows]

    @staticmethod
    def _where(flt: dict | None):
        if not flt:
            return "", {}
        clauses, params = [], {}
        for key, val in flt.items():
            if val is None:
                continue
            clauses.append(f"{key} = :{key}")
            params[key] = val
        return ("WHERE " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _row_to_chunk(r) -> Chunk:
        return Chunk(text=r["text"], company=r["company"], ticker=r["ticker"],
                     year=r["year"], section=r["section"], kind=r["kind"],
                     doc_id=r["doc_id"], chunk_id=r["chunk_id"])


def build_store(settings: Settings | None = None) -> VectorStore:
    s = settings or get_settings()
    return PgVectorStore(s) if s.use_pgvector else InMemoryVectorStore(s)
