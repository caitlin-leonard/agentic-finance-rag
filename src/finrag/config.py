"""Central configuration.

Everything is env-driven (12-factor). The single most important switch is
`provider`: set it to ``offline`` and the whole system runs deterministically
with zero API keys and zero network — that is what CI uses and what lets a
reviewer clone the repo and get a green build in one command.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["offline", "openai", "azure", "anthropic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FINRAG_", env_file=".env", extra="ignore"
    )

    # --- provider selection ---
    provider: Provider = "offline"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- secrets (only needed when provider != offline) ---
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-08-01-preview"

    # --- corpus ---
    # Point this at a JSONL of filing records (e.g. from scripts/fetch_edgar.py)
    # to index real filings instead of the bundled sample. Env: FINRAG_CORPUS_PATH
    corpus_path: str | None = None

    # --- storage ---
    # Default is a local file DB so `docker compose up` isn't required to try it.
    database_url: str = "postgresql+psycopg://finrag:finrag@localhost:5432/finrag"
    use_pgvector: bool = False  # flip on when a real Postgres+pgvector is available

    # --- retrieval knobs ---
    top_k_dense: int = 8
    top_k_lexical: int = 8
    top_k_final: int = 5
    rrf_k: int = 60
    use_reranker: bool = False

    # --- graph / guardrail knobs ---
    max_self_corrections: int = 1
    min_groundedness: float = 0.70   # answers below this are refused/retried
    min_context_grade: float = 0.30  # fraction of docs that must be relevant

    # --- serving ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    request_timeout_s: int = 60

    # --- eval gate (used by CI) ---
    eval_min_faithfulness: float = 0.80
    eval_min_answer_relevancy: float = 0.75
    eval_min_context_recall: float = 0.70


@lru_cache
def get_settings() -> Settings:
    return Settings()
