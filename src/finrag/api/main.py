"""FastAPI service exposing the agentic RAG pipeline.

Endpoints:
  GET  /health   liveness + what's loaded
  POST /ingest   (re)build the index from bundled or inline records
  POST /query    ask a question, get a grounded, cited, cost-tagged answer
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ..config import get_settings
from ..graph import RagPipeline
from ..obs import estimate_cost_usd, get_logger, setup_telemetry
from .schemas import (
    CitationOut, HealthResponse, IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
)

log = get_logger("finrag.api")
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_telemetry()
    s = get_settings()
    pipe = RagPipeline(s)
    n = pipe.ingest()  # warm the index at boot
    _state["pipeline"] = pipe
    _state["chunks"] = n
    log.info("startup", provider=s.provider, chunks=n)
    yield
    _state.clear()


app = FastAPI(title="Agentic Finance RAG", version="0.1.0", lifespan=lifespan)


def _pipe() -> RagPipeline:
    pipe = _state.get("pipeline")
    if pipe is None:
        raise HTTPException(503, "pipeline not ready")
    return pipe


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(status="ok", provider=s.provider,
                          chunks_indexed=_state.get("chunks", 0))


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    pipe = _pipe()
    n = pipe.ingest(req.records)
    _state["chunks"] = n
    return IngestResponse(chunks_indexed=n)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    s = get_settings()
    pipe = _pipe()
    t = pipe.query(req.question)
    cost = estimate_cost_usd(s.llm_model, t.prompt_tokens, t.completion_tokens)
    log.info("query", q=req.question, grounded=t.groundedness,
             refused=t.refused, latency_ms=t.latency_ms, cost=cost)
    return QueryResponse(
        question=t.question, answer=t.answer, route=t.route, refused=t.refused,
        groundedness=t.groundedness, answer_relevance=t.answer_relevance,
        context_grade=t.context_grade, self_corrections=t.self_corrections,
        latency_ms=t.latency_ms, prompt_tokens=t.prompt_tokens,
        completion_tokens=t.completion_tokens, est_cost_usd=cost,
        citations=[CitationOut(**vars(c)) for c in t.citations],
    )


def main() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run("finrag.api.main:app", host=s.api_host, port=s.api_port)


if __name__ == "__main__":
    main()
