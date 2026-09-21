"""Streamlit demo UI — neon-cyberpunk theme.

Talks to the FastAPI service if FINRAG_API_URL is set, otherwise runs the
pipeline in-process (handy for a one-command local demo). Surfaces the parts
recruiters/engineers care about: the grounded answer, its citations, the
guardrail scores, and cost/latency — i.e. the answer *and* why to trust it.
"""
from __future__ import annotations

import os

import requests
import streamlit as st

st.set_page_config(page_title="Agentic Finance RAG", page_icon="⚡", layout="wide")

API_URL = os.getenv("FINRAG_API_URL")  # e.g. http://localhost:8000

# --------------------------------------------------------------------------- #
# Neon / gothic theme
# --------------------------------------------------------------------------- #
NEON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --neon: #00e5ff;
  --neon-2: #3b82f6;
  --magenta: #b026ff;
  --bg: #06080f;
  --panel: #0b1020;
  --edge: rgba(0,229,255,.28);
}

/* page background: deep space + subtle grid */
.stApp {
  background:
    radial-gradient(1200px 600px at 15% -10%, rgba(0,229,255,.08), transparent 60%),
    radial-gradient(1000px 500px at 100% 0%, rgba(176,38,255,.08), transparent 55%),
    linear-gradient(180deg, #06080f 0%, #04050a 100%);
  background-attachment: fixed;
}
.stApp::before {
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(0,229,255,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,.045) 1px, transparent 1px);
  background-size: 44px 44px;
  mask-image: linear-gradient(180deg, black, transparent 85%);
}
.block-container { position:relative; z-index:1; }

/* headings */
h1 {
  font-family:'Orbitron',sans-serif !important; font-weight:900 !important;
  letter-spacing:2px; color:#eaf7ff !important;
  text-shadow:0 0 8px rgba(0,229,255,.65), 0 0 24px rgba(0,229,255,.35);
}
h2, h3 {
  font-family:'Rajdhani',sans-serif !important; font-weight:700 !important;
  text-transform:uppercase; letter-spacing:2px;
  color:var(--neon) !important; text-shadow:0 0 10px rgba(0,229,255,.4);
}
.stCaption, [data-testid="stCaptionContainer"] { color:#7fb6d9 !important; }
body, p, span, label, .stMarkdown { font-family:'Rajdhani',sans-serif; }

/* answer text a touch bigger + readable */
.answer-box {
  font-family:'Rajdhani',sans-serif; font-size:1.15rem; line-height:1.6;
  border-left:2px solid var(--neon); padding:.4rem 0 .4rem 1rem;
  text-shadow:0 0 6px rgba(0,229,255,.12);
}

/* text input: neon field */
.stTextInput input {
  background:#080c17 !important; color:#d7e3f4 !important;
  border:1px solid var(--edge) !important; border-radius:10px !important;
  font-family:'JetBrains Mono',monospace !important;
  box-shadow:inset 0 0 12px rgba(0,229,255,.08);
}
.stTextInput input:focus {
  border-color:var(--neon) !important;
  box-shadow:0 0 0 1px var(--neon), 0 0 18px rgba(0,229,255,.35) !important;
}

/* primary button: glowing */
.stButton > button {
  font-family:'Orbitron',sans-serif !important; font-weight:700 !important;
  letter-spacing:2px; text-transform:uppercase;
  color:#04121a !important;
  background:linear-gradient(135deg,var(--neon),var(--neon-2)) !important;
  border:none !important; border-radius:10px !important; padding:.5rem 1.6rem !important;
  box-shadow:0 0 18px rgba(0,229,255,.5), 0 0 40px rgba(59,130,246,.25);
  transition:transform .08s ease, box-shadow .2s ease;
}
.stButton > button:hover {
  transform:translateY(-1px);
  box-shadow:0 0 26px rgba(0,229,255,.8), 0 0 60px rgba(176,38,255,.35);
}

/* metric tiles */
[data-testid="stMetric"] {
  background:linear-gradient(180deg, rgba(12,17,32,.9), rgba(6,8,15,.9));
  border:1px solid var(--edge); border-radius:14px; padding:14px 16px;
  box-shadow:0 0 18px rgba(0,229,255,.10), inset 0 0 24px rgba(0,229,255,.05);
}
[data-testid="stMetricValue"] {
  font-family:'Orbitron',sans-serif !important;
  color:var(--neon) !important; text-shadow:0 0 12px rgba(0,229,255,.55);
}
[data-testid="stMetricLabel"] { color:#8fb8d6 !important; text-transform:uppercase; letter-spacing:1px; }

/* sidebar */
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#080b16,#05060c) !important;
  border-right:1px solid var(--edge);
  box-shadow:2px 0 24px rgba(0,229,255,.06);
}

/* expanders = citation cards */
[data-testid="stExpander"] {
  border:1px solid var(--edge) !important; border-radius:12px !important;
  background:rgba(11,16,32,.6) !important;
  box-shadow:0 0 14px rgba(0,229,255,.08);
}
[data-testid="stExpander"] summary { color:var(--neon) !important; font-family:'Rajdhani'; font-weight:600; }

/* alerts */
.stAlert { border-radius:12px !important; border:1px solid var(--magenta) !important;
  box-shadow:0 0 20px rgba(176,38,255,.25); }
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)


@st.cache_resource
def _local_pipeline():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from finrag.config import get_settings
    from finrag.graph import RagPipeline
    from finrag.obs import estimate_cost_usd

    pipe = RagPipeline(get_settings())
    pipe.ingest()
    return pipe, get_settings(), estimate_cost_usd


def ask(question: str) -> dict:
    if API_URL:
        r = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)
        r.raise_for_status()
        return r.json()
    pipe, s, cost_fn = _local_pipeline()
    t = pipe.query(question)
    return {
        "answer": t.answer, "route": t.route, "refused": t.refused,
        "groundedness": t.groundedness, "answer_relevance": t.answer_relevance,
        "context_grade": t.context_grade, "self_corrections": t.self_corrections,
        "latency_ms": t.latency_ms, "prompt_tokens": t.prompt_tokens,
        "completion_tokens": t.completion_tokens,
        "est_cost_usd": cost_fn(s.llm_model, t.prompt_tokens, t.completion_tokens),
        "citations": [vars(c) for c in t.citations],
    }


st.markdown("# ⚡ AGENTIC RAG · SEC FILINGS")
st.caption("LangGraph · hybrid retrieval · groundedness guardrails · Ragas / DeepEval")

with st.sidebar:
    st.markdown("### ◢ TRY THESE")
    samples = [
        "What was Acme Robotics total net revenue in fiscal 2023?",
        "What was Globex Semiconductor gross margin in fiscal 2023?",
        "What percentage of Acme revenue came from its three largest customers in 2023?",
        "What was Nordic Retail free cash flow in fiscal 2023?",
        "Who won the World Cup in 1998?",
    ]
    picked = st.radio("Sample questions", samples, index=0)
    mode = "remote API" if API_URL else "in-process"
    st.info(f"▶ backend: {mode}")

question = st.text_input("Ask a question about the filings", value=picked)

if st.button("Ask ▸", type="primary") and question:
    with st.spinner("running agentic RAG…"):
        res = ask(question)

    if res["refused"]:
        st.warning("🛑 GUARDRAIL REFUSED — insufficient grounded evidence.")
    st.markdown("## ◢ Answer")
    st.markdown(f"<div class='answer-box'>{res['answer']}</div>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Groundedness", f"{res['groundedness']:.2f}")
    c2.metric("Answer relevance", f"{res['answer_relevance']:.2f}")
    c3.metric("Latency (ms)", f"{res['latency_ms']:.0f}")
    c4.metric("Est. cost (USD)", f"${res['est_cost_usd']:.5f}")

    st.caption(
        f"route: {res['route']}  ·  context grade: {res['context_grade']:.2f}"
        f"  ·  self-corrections: {res['self_corrections']}"
        f"  ·  tokens: {res['prompt_tokens']}+{res['completion_tokens']}"
    )

    st.markdown("## ◢ Citations")
    if not res["citations"]:
        st.markdown("_None — nothing was grounded._")
    for c in res["citations"]:
        with st.expander(f"▨ {c['citation']}"):
            st.markdown(c["snippet"] + " …")
