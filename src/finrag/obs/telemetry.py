"""Observability: structured logging, OpenTelemetry spans, and cost estimation.

Production LLM systems are judged on more than accuracy — you have to answer
"how slow, how expensive, and why did it say that?". This module gives every
query a trace id, per-stage spans, and a token->USD estimate, all of which the
UI surfaces so the behaviour is inspectable rather than a black box.
"""
from __future__ import annotations

import functools
import logging
from contextlib import contextmanager

import structlog

# Rough public list prices (USD per 1K tokens). Override via config as needed.
_PRICES = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "text-embedding-3-small": (0.00002, 0.0),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int
                      ) -> float:
    p_in, p_out = _PRICES.get(model, (0.0, 0.0))
    return round((prompt_tokens / 1000) * p_in
                 + (completion_tokens / 1000) * p_out, 6)


def setup_telemetry(service_name: str = "finrag") -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    try:  # OTel is optional at runtime; degrade gracefully if not installed.
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}))
        trace.set_tracer_provider(provider)
    except Exception:
        pass


def get_logger(name: str = "finrag"):
    return structlog.get_logger(name)


@contextmanager
def _span(name: str):
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("finrag")
        with tracer.start_as_current_span(name) as sp:
            yield sp
    except Exception:
        yield None


def traced(name: str):
    """Decorator wrapping a function in an OTel span (no-op if OTel absent)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _span(name):
                return fn(*args, **kwargs)

        return wrapper

    return deco
