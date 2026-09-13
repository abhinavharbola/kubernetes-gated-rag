import logging
import random
import sys
from pathlib import Path

# app.py lives in ui/, one level below the repo root, so the repo root
# (where the src/ package lives) has to be added explicitly. Without this,
# `from src...` only works by accident of whatever directory the process
# happened to be launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# httpx/httpcore log at INFO by default, which means every guardrail and
# provider HTTP call gets a verbose per-request line in a normal terminal.
# Quiet by default; flip back to INFO locally if you need to see what a
# provider call is actually doing.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import streamlit as st

from src.providers.clients import qdrant_client
from src.config import settings
from src.graph import run_turn
from src.retrieval.rerank import preload as preload_rerank

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Kubernetes RAG", page_icon="◆", layout="centered")


@st.cache_resource(show_spinner="Loading the local reranker (first run only)…")
def _warm_up_models() -> bool:
    # unguarded, this crashed the whole app at startup (before any UI
    # renders) on a model-download hiccup — no network on first run, disk
    # full, a stale/corrupted local cache. rerank_and_gate() already
    # handles a ranker failure gracefully per-request (fails closed), so a
    # failed warmup just means the first real rerank call pays the
    # (already-handled) load cost lazily instead of failing the whole app
    # before the user ever sees a chat box.
    try:
        preload_rerank()
    except Exception:
        logger.exception("reranker warmup failed, will retry lazily on first use")
    return True


_warm_up_models()

PIPELINE_ERROR_MESSAGE = (
    "Something went wrong completing that request. This is usually a transient "
    "provider issue — try again in a moment."
)

# ---------------------------------------------------------------------------
# Design system
#
# Modeled on how production chat products are actually structured: no
# boxed "app shell" card sitting on a different-colored canvas, just a flat
# page — a plain white/near-white conversation pane, a faintly tinted
# sidebar for separation, and a floating rounded input bar. User turns get
# a soft tinted bubble; assistant turns are plain text, the way most
# assistants render their own replies. One quiet accent color, one
# typeface for UI text, a monospace face reserved strictly for values that
# are actually data (trace segments, scores, stats).
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #FCFCFB;
    --bg-sidebar: #F4F4F2;
    --surface: #FFFFFF;
    --border: #E6E5E1;
    --border-strong: #D6D5D0;
    --text: #1C1C1A;
    --text-muted: #68676F;
    --text-faint: #9B9A95;
    --accent: #3B4FA0;
    --accent-hover: #303F84;
    --accent-soft: #EEF0FA;
    --user-bubble: #F0EEE7;
    --ready: #1E7A4C;
    --pending: #A16A08;
    --failed: #C1342A;
    --font-ui: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace;
}

.stApp { font-family: var(--font-ui); background: var(--bg); color: var(--text); font-size: 0.96rem; }
.stMarkdown, .stApp p, .stApp li { color: var(--text); line-height: 1.65; }
code, .mono { font-family: var(--font-data); }

[data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background: var(--bg) !important;
}
[data-testid="stBottomBlockContainer"] { display: flex !important; justify-content: center !important; }
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 16px !important;
    max-width: 780px !important;
    box-shadow: 0 2px 10px rgba(28, 28, 26, 0.06) !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; font-size: 0.95rem !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-faint) !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--accent) !important; border-radius: 10px !important; }
[data-testid="stChatInputSubmitButton"]:hover { background: var(--accent-hover) !important; }
[data-testid="stChatInputSubmitButton"] svg { color: #FFFFFF !important; fill: #FFFFFF !important; }

.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 780px !important;
    padding-top: 1.4rem !important;
    padding-bottom: 7rem !important;
}

.stButton button {
    border-radius: 8px !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text) !important;
    background: var(--surface) !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    transition: border-color 0.12s ease, background 0.12s ease !important;
}
.stButton button:hover { border-color: var(--accent) !important; color: var(--accent) !important; background: var(--accent-soft) !important; }

/* --- top bar --- */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 0.9rem; margin-bottom: 0.6rem; border-bottom: 1px solid var(--border);
}
.topbar-brand { display: flex; align-items: center; gap: 0.55rem; }
.topbar-avatar {
    width: 26px; height: 26px; border-radius: 7px; background: var(--accent); color: #FFFFFF;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-ui); font-size: 0.78rem; font-weight: 700;
}
.topbar-title { font-size: 0.95rem; font-weight: 600; color: var(--text); }
.topbar-status { font-family: var(--font-data); font-size: 0.72rem; color: var(--text-faint); }

/* --- architecture --- */
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 12px !important; background: var(--surface) !important; box-shadow: none !important; }
[data-testid="stExpander"] summary { font-size: 0.87rem !important; font-weight: 500 !important; }
.arch-intro { color: var(--text-muted); font-size: 0.85rem; line-height: 1.6; margin: 0.2rem 0 1rem 0; }
.arch-strip-wrap { overflow-x: auto; padding-bottom: 0.3rem; }
.arch-strip { display: flex; align-items: stretch; gap: 0; width: max-content; }
.arch-node {
    width: 146px; flex-shrink: 0; background: var(--bg-sidebar); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.65rem 0.75rem; display: flex; flex-direction: column; gap: 0.3rem;
}
.arch-node .arch-num {
    width: 18px; height: 18px; border-radius: 50%; background: var(--accent-soft); color: var(--accent);
    font-family: var(--font-data); font-size: 0.64rem; font-weight: 600;
    display: flex; align-items: center; justify-content: center;
}
.arch-node .arch-title { font-size: 0.8rem; font-weight: 600; color: var(--text); }
.arch-node .arch-desc { font-family: var(--font-data); font-size: 0.66rem; line-height: 1.45; color: var(--text-muted); }
.arch-connector { display: flex; align-items: center; justify-content: center; width: 28px; flex-shrink: 0; color: var(--text-faint); font-size: 1.05rem; }
.arch-note { font-family: var(--font-data); font-size: 0.72rem; color: var(--text-faint); margin-top: 0.8rem; line-height: 1.6; }

/* --- empty state: a quiet greeting, not a feature grid --- */
.empty-state { padding: 2.2rem 0 1rem 0; text-align: center; }
.empty-state h2 { font-size: 1.3rem; font-weight: 600; color: var(--text); margin: 0 0 0.4rem 0; }
.empty-state p { color: var(--text-muted); font-size: 0.88rem; max-width: 46ch; margin: 0 auto; }

/* --- chat turns --- */
[data-testid="stChatMessage"] { padding: 0.15rem 0; margin-bottom: 0.15rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--user-bubble); border-radius: 16px; padding: 0.8rem 1.05rem; margin: 0.35rem 0;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent; padding: 0.5rem 0.1rem 0.9rem 0.1rem;
}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
    width: 26px !important; height: 26px !important; font-size: 0.85rem !important;
}
[data-testid="stChatMessageAvatarAssistant"] { background: var(--accent) !important; }

/* --- pipeline trace: a row of small status pills, not a boxed panel --- */
.trace-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.55rem 0 0.1rem 0.1rem; }
.trace-pill {
    display: inline-flex; align-items: center; gap: 0.35rem; font-family: var(--font-data); font-size: 0.7rem;
    padding: 0.24rem 0.6rem; border-radius: 999px; background: var(--surface); border: 1px solid var(--border);
    color: var(--text-muted);
}
.trace-pill .dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.trace-pill.pass .dot { background: var(--ready); }
.trace-pill.fail { border-color: var(--failed); color: var(--failed); }
.trace-pill.fail .dot { background: var(--failed); }
.trace-pill.skip .dot { background: var(--pending); }
.trace-pill.neutral .dot { background: var(--accent); }
.trace-latency { font-family: var(--font-data); font-size: 0.7rem; color: var(--text-faint); align-self: center; margin-left: 0.1rem; }

.error-note {
    font-family: var(--font-data); font-size: 0.78rem; color: var(--failed);
    padding: 0.5rem 0.7rem; margin: 0.5rem 0 0 0.1rem; border-radius: 10px;
    background: #FBEBE9; border: 1px solid #F1CFCB;
}

.source-row {
    display: grid; grid-template-columns: 1fr 60px; align-items: center; gap: 0.6rem;
    font-family: var(--font-data); font-size: 0.78rem; padding: 0.4rem 0; border-bottom: 1px solid var(--border);
}
.source-row:last-child { border-bottom: none; }
.source-path { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
.source-score { color: var(--text-muted); text-align: right; }
.source-score.na { color: var(--text-faint); }

/* --- sidebar --- */
[data-testid="stSidebar"] { background: var(--bg-sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p { color: var(--text); }
.side-label {
    font-family: var(--font-ui); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
    text-transform: uppercase; color: var(--text-faint); margin: 1.3rem 0 0.55rem 0;
}
.side-label:first-child { margin-top: 0.2rem; }
.stat-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 0.24rem 0; font-size: 0.83rem; color: var(--text-muted);
}
.stat-row .stat-value { font-family: var(--font-data); color: var(--text); font-size: 0.83rem; }

.provider-row { display: flex; justify-content: space-between; align-items: center; padding: 0.32rem 0; }
.provider-row .provider-name { font-size: 0.82rem; color: var(--text); }
.provider-row .provider-role { display: block; font-family: var(--font-data); font-size: 0.63rem; color: var(--text-faint); }
.provider-status { display: flex; align-items: center; gap: 0.4rem; font-family: var(--font-data); font-size: 0.7rem; flex-shrink: 0; }
.provider-status .dot { width: 6px; height: 6px; border-radius: 50%; }
.provider-status.ok { color: var(--ready); }
.provider-status.ok .dot { background: var(--ready); }
.provider-status.missing { color: var(--failed); }
.provider-status.missing .dot { background: var(--failed); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# A single example rendered into the input's placeholder, chosen once per
# session rather than a grid of clickable cards — keeps the empty state to
# a plain greeting, the way a chat product's landing turn actually looks,
# while still surfacing what kind of question this tool is for.
PLACEHOLDER_EXAMPLES = [
    "How do I pin a container to a specific image version?",
    "What's the difference between a Deployment and a StatefulSet?",
    "How do I safely delete a single Pod without affecting the rest of a Deployment?",
    "How do I give each StatefulSet replica its own persistent storage?",
]
if "placeholder_example" not in st.session_state:
    st.session_state.placeholder_example = random.choice(PLACEHOLDER_EXAMPLES)

# Each "check" reports whether a provider is *configured* (an API key is
# present), not whether it's currently reachable — labeled "configured" /
# "missing key" below rather than an up/down status, so it doesn't claim a
# live health check it never actually performs.
PROVIDERS = [
    {"label": "Groq (account A)", "role": "generation, chain link 1", "check": lambda: bool(settings.groq_api_key)},
    {
        "label": "Groq (account B)",
        "role": "generation, chain link 2",
        "check": lambda: bool(settings.groq_api_key_secondary),
    },
    {
        "label": "NIM",
        "role": "generation fallback, planner, NeMoGuard",
        "check": lambda: bool(settings.nvidia_nim_api_key),
    },
    {"label": "Gemini", "role": "embeddings, eval judge", "check": lambda: bool(settings.gemini_api_key)},
    {
        "label": "Qdrant",
        "role": "vector store",
        "check": lambda: bool(settings.qdrant_url and settings.qdrant_api_key),
    },
]

# The pipeline a request actually takes, in order — mirrors the flow in
# src/graph.py's build_graph(). Kept short and declarative on purpose: this
# is a diagram for a person to orient by, not a substitute for reading the
# graph itself.
ARCHITECTURE_STAGES = [
    {"title": "Exact cache", "desc": "diskcache lookup on the normalized question"},
    {"title": "Safety gate", "desc": "deterministic jailbreak check + NeMoGuard"},
    {"title": "Topic gate", "desc": "planner classifier, Kubernetes-only"},
    {"title": "Semantic cache", "desc": "Qdrant cosine match above threshold"},
    {"title": "Retrieve", "desc": "Qdrant dense vector search, top K"},
    {"title": "Rerank", "desc": "FlashRank cross-encoder, hard threshold"},
    {"title": "Generate", "desc": "Groq \u2192 Groq (2nd acct) \u2192 NIM"},
    {"title": "Response safety", "desc": "NeMoGuard checks the generated answer"},
]


@st.cache_data(ttl=30)
def get_corpus_stats():
    try:
        docs = qdrant_client.count(collection_name=settings.qdrant_docs_collection, exact=False).count
        cached = qdrant_client.count(collection_name=settings.qdrant_cache_collection, exact=False).count
        return docs, cached
    except Exception:
        return None, None


def get_session_stats():
    assistant_turns = [t for t in st.session_state.history if t["role"] == "assistant"]
    total = len(assistant_turns)
    if total == 0:
        return 0, "—", "—"
    cache_hits = sum(1 for t in assistant_turns if t.get("details", {}).get("cache_layer"))
    latencies = [
        t["details"]["latency_seconds"]
        for t in assistant_turns
        if t.get("details", {}).get("latency_seconds") is not None
    ]
    avg_latency = f"{sum(latencies) / len(latencies):.2f}s" if latencies else "—"
    return total, f"{round(cache_hits / total * 100)}%", avg_latency


# chat_input is called early, even though it visually renders pinned to the
# bottom of the page regardless of call order (Streamlit's own behavior),
# doing this before rendering history lets the history block below know
# whether a new turn is about to be generated.
prompt = st.chat_input(f'Ask a Kubernetes question — e.g. "{st.session_state.placeholder_example}"')

# ---------- sidebar ----------

with st.sidebar:
    if st.button("+  New conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.placeholder_example = random.choice(PLACEHOLDER_EXAMPLES)
        get_corpus_stats.clear()
        st.rerun()

    st.markdown('<div class="side-label">Corpus</div>', unsafe_allow_html=True)
    doc_count, cache_count = get_corpus_stats()
    st.markdown(
        f'<div class="stat-row"><span>Chunks indexed</span>'
        f'<span class="stat-value">{doc_count if doc_count is not None else "—"}</span></div>'
        f'<div class="stat-row"><span>Cached answers</span>'
        f'<span class="stat-value">{cache_count if cache_count is not None else "—"}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">This session</div>', unsafe_allow_html=True)
    questions_asked, hit_rate, avg_latency = get_session_stats()
    st.markdown(
        f'<div class="stat-row"><span>Questions asked</span><span class="stat-value">{questions_asked}</span></div>'
        f'<div class="stat-row"><span>Cache hit rate</span><span class="stat-value">{hit_rate}</span></div>'
        f'<div class="stat-row"><span>Avg. latency</span><span class="stat-value">{avg_latency}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Providers</div>', unsafe_allow_html=True)
    provider_rows = ""
    for provider in PROVIDERS:
        ok = provider["check"]()
        status_class = "ok" if ok else "missing"
        status_text = "configured" if ok else "missing key"
        provider_rows += (
            f'<div class="provider-row">'
            f'<div><span class="provider-name">{provider["label"]}</span>'
            f'<span class="provider-role">{provider["role"]}</span></div>'
            f'<div class="provider-status {status_class}"><span class="dot"></span>{status_text}</div>'
            f"</div>"
        )
    st.markdown(provider_rows, unsafe_allow_html=True)

    st.markdown('<div class="side-label">Display</div>', unsafe_allow_html=True)
    show_trace = st.toggle("Show pipeline trace", value=True)


# ---------- pipeline trace ----------


def build_trace_segments(details: dict) -> list[dict]:
    if details.get("error"):
        return [{"text": "pipeline error", "status": "fail"}]

    blocked_stage = details.get("blocked_stage")
    cache_layer = details.get("cache_layer")
    cache_checked = details.get("exact_cache_checked", False)

    if cache_layer == "exact":
        return [{"text": "cache hit · exact", "status": "pass"}]

    segments = []
    if blocked_stage == "safety":
        segments.append({"text": "safety blocked", "status": "fail"})
        return segments
    segments.append({"text": "safety pass", "status": "pass"})

    if blocked_stage == "topic":
        segments.append({"text": "off-topic", "status": "fail"})
        return segments
    segments.append({"text": "on-topic", "status": "pass"})

    if cache_layer == "semantic":
        segments.append({"text": "cache hit · semantic", "status": "pass"})
        return segments
    segments.append({"text": "cache miss" if cache_checked else "cache lookup", "status": "skip"})

    if details.get("service_unavailable"):
        segments.append({"text": "retrieval unavailable", "status": "fail"})
        return segments

    candidates_count = details.get("candidates_count", 0)
    segments.append({"text": f"retrieved {candidates_count}", "status": "pass"})

    reranked_count = details.get("reranked_count", 0)
    if reranked_count == 0:
        segments.append({"text": "0 survived rerank", "status": "fail"})
        return segments
    segments.append({"text": f"reranked {reranked_count}/{candidates_count}", "status": "pass"})

    provider = details.get("provider")
    model = details.get("model")
    if provider:
        label = f"{provider} · {model}" if model else provider
        segments.append({"text": label, "status": "neutral"})

    if blocked_stage == "response_safety":
        segments.append({"text": "response blocked", "status": "fail"})

    return segments


def render_trace(details: dict) -> None:
    segments = build_trace_segments(details)
    if segments:
        pills = "".join(
            f'<span class="trace-pill {seg["status"]}"><span class="dot"></span>{seg["text"]}</span>'
            for seg in segments
        )
        latency = details.get("latency_seconds")
        latency_html = f'<span class="trace-latency">{latency:.2f}s</span>' if latency is not None else ""
        st.markdown(f'<div class="trace-row">{pills}{latency_html}</div>', unsafe_allow_html=True)

    if details.get("error"):
        st.markdown(f'<div class="error-note">{PIPELINE_ERROR_MESSAGE}</div>', unsafe_allow_html=True)
        return

    sources = details.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            rows = ""
            for source in sources:
                path = source["metadata"].get("source_path", "unknown")
                # rerank_score is absent when rerank_and_gate degraded to
                # unfiltered retrieval order (FlashRank failure, see
                # src/retrieval/rerank.py) — there's no score to show in
                # that case.
                score = source.get("rerank_score")
                score_text = f"{score:.3f}" if score is not None else "n/a"
                score_class = "" if score is not None else " na"
                rows += (
                    f'<div class="source-row"><span class="source-path">{path}</span>'
                    f'<span class="source-score{score_class}">{score_text}</span></div>'
                )
            st.markdown(rows, unsafe_allow_html=True)


def render_architecture() -> None:
    nodes_html = ""
    for i, stage in enumerate(ARCHITECTURE_STAGES):
        nodes_html += (
            f'<div class="arch-node"><span class="arch-num">{i + 1}</span>'
            f'<span class="arch-title">{stage["title"]}</span>'
            f'<span class="arch-desc">{stage["desc"]}</span></div>'
        )
        if i < len(ARCHITECTURE_STAGES) - 1:
            nodes_html += '<div class="arch-connector">&rsaquo;</div>'
    st.markdown(
        '<p class="arch-intro">Every question moves through this fixed sequence. A gate that blocks, '
        "a cache hit, or an empty rerank ends the turn early — later stages simply don't run.</p>"
        f'<div class="arch-strip-wrap"><div class="arch-strip">{nodes_html}</div></div>'
        '<p class="arch-note">cache hit → answer returned immediately, no remote call &nbsp;·&nbsp; '
        "gate blocked → refused before retrieval ever runs &nbsp;·&nbsp; "
        "rerank finds nothing → answer says so and is cached with a TTL, not treated as an outage</p>",
        unsafe_allow_html=True,
    )


# ---------- page ----------

configured_count = sum(1 for p in PROVIDERS if p["check"]())
st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-brand">
            <div class="topbar-avatar">K</div>
            <span class="topbar-title">Kubernetes RAG</span>
        </div>
        <span class="topbar-status">{configured_count}/{len(PROVIDERS)} providers configured</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Architecture — how a question becomes an answer", expanded=False):
    render_architecture()

if not st.session_state.history and not prompt:
    st.markdown(
        '<div class="empty-state"><h2>Ask about your Kubernetes docs</h2>'
        "<p>Every question runs through a safety gate, a topic gate, caching, retrieval, "
        "a relevance-gated rerank, and a response safety check before it's shown or cached.</p></div>",
        unsafe_allow_html=True,
    )

history_container = st.container(key="history_block")
with history_container:
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and show_trace and turn.get("details"):
                render_trace(turn["details"])

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running the pipeline…"):
            plain_history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.history[:-1]]
            try:
                result = run_turn(prompt, plain_history)
                error = None
            except Exception as exc:
                # narrowing RETRYABLE in src/providers/llm.py means
                # non-transient errors (bad request, auth, etc.) propagate
                # here instead of being swallowed into a generic
                # RuntimeError after wasted retries. Log the real exception
                # for debugging, show a clean message to the user rather
                # than a stack trace or raw provider error text.
                logger.exception("run_turn failed")
                result = None
                error = exc

        if error is not None:
            answer = PIPELINE_ERROR_MESSAGE
            st.markdown(answer)
            details = {"error": str(error)}
        else:
            answer = result["answer"]
            st.markdown(answer)
            details = {
                "blocked_stage": result.get("blocked_stage"),
                "cache_layer": result.get("cache_layer"),
                "exact_cache_checked": result.get("exact_cache_checked"),
                "provider": result.get("provider"),
                "model": result.get("model"),
                "candidates_count": len(result.get("candidates") or []),
                "reranked_count": len(result.get("reranked") or []),
                "service_unavailable": result.get("service_unavailable", False),
                "sources": result.get("reranked") if not result.get("cache_layer") else None,
                "latency_seconds": result.get("latency_seconds"),
            }
        if show_trace:
            render_trace(details)

    st.session_state.history.append({"role": "assistant", "content": answer, "details": details})
    get_corpus_stats.clear()