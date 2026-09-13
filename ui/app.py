import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import streamlit as st

from src.providers.clients import qdrant_client
from src.config import settings
from src.graph import run_turn
from src.retrieval.rerank import preload as preload_rerank

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Kubernetes Assistant", page_icon="⬡", layout="wide")


@st.cache_resource(show_spinner=False)
def _warm_up_models() -> bool:
    try:
        preload_rerank()
    except Exception:
        logger.exception("reranker warmup failed, will retry lazily on first use")
    return True


_warm_up_models()

PIPELINE_ERROR_MESSAGE = (
    "Something went wrong completing that request. This is usually a transient "
    "provider issue, try again in a moment."
)

PLACEHOLDER_EXAMPLES = [
    "How do I pin a container to a specific image version?",
    "What's the difference between a Deployment and a StatefulSet?",
    "How do I safely delete a single Pod without affecting the rest of a Deployment?",
    "How do I give each StatefulSet replica its own persistent storage?",
]

PROVIDERS = [
    {"label": "Groq A", "check": lambda: bool(settings.groq_api_key)},
    {"label": "Groq B", "check": lambda: bool(settings.groq_api_key_secondary)},
    {"label": "NIM", "check": lambda: bool(settings.nvidia_nim_api_key)},
    {"label": "Gemini", "check": lambda: bool(settings.gemini_api_key)},
    {"label": "Qdrant", "check": lambda: bool(settings.qdrant_url and settings.qdrant_api_key)},
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #EEF0F3;
    --bg-sidebar: #E4E7EC;
    --surface: #FFFFFF;
    --surface-sunken: #F3F4F6;
    --border: #D9DCE1;
    --border-strong: #C6CAD1;
    --text: #16181D;
    --text-muted: #5B5F68;
    --text-faint: #868B94;
    --accent: #2563EB;
    --accent-hover: #1D4ED8;
    --accent-soft: #EFF4FF;
    --user-bubble: #E4E7EC;
    --ok: #15803D;
    --warn: #B45309;
    --bad: #DC2626;
    --shadow-card: 0 1px 2px rgba(16, 20, 27, 0.06), 0 1px 1px rgba(16, 20, 27, 0.04);
    --font-ui: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace;
}

.stApp { font-family: var(--font-ui); background: var(--bg); color: var(--text); }
.stMarkdown, .stApp p, .stApp li { color: var(--text); line-height: 1.6; }
code { font-family: var(--font-mono); }

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border-strong);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--text-muted); }

.brand {
    display: flex; align-items: center; gap: 0.55rem;
    padding: 0.2rem 0 1.1rem 0;
}
.brand-mark {
    width: 28px; height: 28px; border-radius: 8px; background: var(--accent);
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.95rem; flex-shrink: 0;
}
.brand-name { font-size: 0.95rem; font-weight: 600; color: var(--text); }

.sidebar-section-title {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    color: var(--text-faint); margin: 1.4rem 0 0.5rem 0;
}
.sidebar-section-title:first-of-type { margin-top: 0.2rem; }

.stat-line {
    display: flex; justify-content: space-between; font-size: 0.82rem;
    color: var(--text-muted); padding: 0.22rem 0;
}
.stat-line .value { font-family: var(--font-mono); color: var(--text); }

.provider-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.75rem; padding: 0.24rem 0.55rem; border-radius: 999px;
    background: var(--surface); border: 1px solid var(--border); margin: 0 0.3rem 0.3rem 0;
    color: var(--text-muted);
}
.provider-pill .dot { width: 6px; height: 6px; border-radius: 50%; }
.provider-pill.ok .dot { background: var(--ok); }
.provider-pill.missing { color: var(--bad); border-color: #F3C9C4; }
.provider-pill.missing .dot { background: var(--bad); }

[data-testid="stSidebar"] .stButton button {
    width: 100%; text-align: left; border-radius: 10px; border: 1px solid var(--border-strong) !important;
    background: var(--surface) !important; color: var(--text) !important; font-weight: 500 !important;
    box-shadow: none !important; padding: 0.55rem 0.8rem !important;
}
[data-testid="stSidebar"] .stButton button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

.st-key-chat_scroll { max-width: 46rem; margin: 0 auto; padding: 0 1rem 9rem 1rem; }

.empty-hero { padding: 12vh 0 0 0; text-align: center; }
.empty-hero h1 { font-size: 1.7rem; font-weight: 600; color: var(--text); margin-bottom: 0.4rem; }
.empty-hero p { color: var(--text-muted); font-size: 0.95rem; margin: 0 auto 1.6rem auto; }

.st-key-suggestions { max-width: 40rem; margin: 0 auto; }
.st-key-suggestions .stButton button {
    text-align: left; white-space: normal; height: auto; border-radius: 12px;
    border: 1px solid var(--border) !important; background: var(--surface) !important;
    color: var(--text) !important; font-size: 0.85rem !important; padding: 0.75rem 0.9rem !important;
    box-shadow: var(--shadow-card) !important; font-weight: 400 !important;
}
.st-key-suggestions .stButton button:hover { border-color: var(--accent) !important; background: var(--accent-soft) !important; }

[data-testid="stChatMessage"] { padding: 0.1rem 0; margin-bottom: 0.1rem; gap: 0.7rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: transparent;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--user-bubble); border-radius: 18px; padding: 0.65rem 1rem; display: inline-block;
    box-shadow: var(--shadow-card);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent; padding-bottom: 0.6rem;
}
[data-testid="stChatMessageAvatarUser"] { background: var(--text-faint) !important; width: 26px !important; height: 26px !important; }
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--accent) !important; width: 26px !important; height: 26px !important; font-size: 0.8rem !important;
}

[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background: var(--bg) !important; }

[data-testid="stChatInput"] {
    background: var(--surface) !important; border: 1px solid var(--border-strong) !important;
    border-radius: 18px !important; box-shadow: 0 4px 16px rgba(16, 20, 27, 0.1) !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--accent) !important; border-radius: 10px !important; }
[data-testid="stChatInputSubmitButton"]:hover { background: var(--accent-hover) !important; }

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    background: var(--surface) !important; box-shadow: var(--shadow-card) !important; margin-top: 0.3rem;
}
[data-testid="stExpander"] summary { font-size: 0.78rem !important; color: var(--text-muted) !important; }

.pill-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0.3rem 0; }
.pill {
    display: inline-flex; align-items: center; gap: 0.3rem; font-family: var(--font-mono);
    font-size: 0.68rem; padding: 0.2rem 0.55rem; border-radius: 999px;
    background: var(--surface); border: 1px solid var(--border); color: var(--text-muted);
}
.pill .dot { width: 5px; height: 5px; border-radius: 50%; }
.pill.pass .dot { background: var(--ok); }
.pill.fail { border-color: #F3C9C4; color: var(--bad); }
.pill.fail .dot { background: var(--bad); }
.pill.skip .dot { background: var(--warn); }
.pill.info .dot { background: var(--accent); }
.pill-latency { font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-faint); align-self: center; }

.source-line {
    display: flex; justify-content: space-between; gap: 0.6rem; font-family: var(--font-mono);
    font-size: 0.75rem; padding: 0.3rem 0; border-bottom: 1px solid var(--border); color: var(--text);
}
.source-line:last-child { border-bottom: none; }
.source-line span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.source-line span:last-child { color: var(--text-faint); flex-shrink: 0; }

.error-note {
    font-size: 0.82rem; color: var(--bad); padding: 0.5rem 0.7rem; border-radius: 8px;
    background: #FDF1F0; border: 1px solid #F3C9C4; margin-top: 0.3rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "placeholder_example" not in st.session_state:
    st.session_state.placeholder_example = random.choice(PLACEHOLDER_EXAMPLES)
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


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
        return 0, "-", "-"
    cache_hits = sum(1 for t in assistant_turns if t.get("details", {}).get("cache_layer"))
    latencies = [
        t["details"]["latency_seconds"]
        for t in assistant_turns
        if t.get("details", {}).get("latency_seconds") is not None
    ]
    avg_latency = f"{sum(latencies) / len(latencies):.2f}s" if latencies else "-"
    return total, f"{round(cache_hits / total * 100)}%", avg_latency


def build_trace_segments(details: dict) -> list[dict]:
    if details.get("error"):
        return [{"text": "pipeline error", "status": "fail"}]

    blocked_stage = details.get("blocked_stage")
    cache_layer = details.get("cache_layer")
    cache_checked = details.get("exact_cache_checked", False)

    if cache_layer == "exact":
        return [{"text": "cache hit, exact", "status": "pass"}]

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
        segments.append({"text": "cache hit, semantic", "status": "pass"})
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
        label = f"{provider}, {model}" if model else provider
        segments.append({"text": label, "status": "info"})

    if blocked_stage == "response_safety":
        segments.append({"text": "response blocked", "status": "fail"})

    return segments


def render_details(details: dict) -> None:
    segments = build_trace_segments(details)
    latency = details.get("latency_seconds")
    latency_html = f'<span class="pill-latency">{latency:.2f}s</span>' if latency is not None else ""
    pills = "".join(
        f'<span class="pill {seg["status"]}"><span class="dot"></span>{seg["text"]}</span>' for seg in segments
    )

    with st.expander("Details"):
        st.markdown(f'<div class="pill-row">{pills}{latency_html}</div>', unsafe_allow_html=True)

        if details.get("error"):
            st.markdown(f'<div class="error-note">{details["error"]}</div>', unsafe_allow_html=True)
            return

        sources = details.get("sources") or []
        if sources:
            rows = ""
            for source in sources:
                path = source["metadata"].get("source_path", "unknown")
                score = source.get("rerank_score")
                score_text = f"{score:.3f}" if score is not None else "n/a"
                rows += f'<div class="source-line"><span>{path}</span><span>{score_text}</span></div>'
            st.markdown(rows, unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">K</div><div class="brand-name">Kubernetes Assistant</div></div>',
        unsafe_allow_html=True,
    )

    if st.button("New chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.placeholder_example = random.choice(PLACEHOLDER_EXAMPLES)
        get_corpus_stats.clear()
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Corpus</div>', unsafe_allow_html=True)
    doc_count, cache_count = get_corpus_stats()
    st.markdown(
        f'<div class="stat-line"><span>Chunks indexed</span><span class="value">{doc_count if doc_count is not None else "-"}</span></div>'
        f'<div class="stat-line"><span>Cached answers</span><span class="value">{cache_count if cache_count is not None else "-"}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">This session</div>', unsafe_allow_html=True)
    questions_asked, hit_rate, avg_latency = get_session_stats()
    st.markdown(
        f'<div class="stat-line"><span>Questions</span><span class="value">{questions_asked}</span></div>'
        f'<div class="stat-line"><span>Cache hit rate</span><span class="value">{hit_rate}</span></div>'
        f'<div class="stat-line"><span>Avg. latency</span><span class="value">{avg_latency}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">Providers</div>', unsafe_allow_html=True)
    provider_html = ""
    for provider in PROVIDERS:
        ok = provider["check"]()
        provider_html += (
            f'<span class="provider-pill {"ok" if ok else "missing"}">'
            f'<span class="dot"></span>{provider["label"]}</span>'
        )
    st.markdown(provider_html, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Display</div>', unsafe_allow_html=True)
    show_details = st.toggle("Show pipeline details", value=False)

with st.container(key="chat_scroll"):
    if not st.session_state.history:
        st.markdown(
            '<div class="empty-hero"><h1>Ask about your Kubernetes docs</h1>'
            "<p>Every question runs through safety and topic gates, caching, retrieval, "
            "a relevance-gated rerank, and a response safety check.</p></div>",
            unsafe_allow_html=True,
        )
        with st.container(key="suggestions"):
            cols = st.columns(2)
            for i, example in enumerate(PLACEHOLDER_EXAMPLES):
                if cols[i % 2].button(example, key=f"suggestion_{i}"):
                    st.session_state.pending_prompt = example
    else:
        for turn in st.session_state.history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn["role"] == "assistant" and show_details and turn.get("details"):
                    render_details(turn["details"])

typed_prompt = st.chat_input(f'Ask a Kubernetes question, e.g. "{st.session_state.placeholder_example}"')
prompt = typed_prompt or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})

    plain_history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.history[:-1]]
    try:
        result = run_turn(prompt, plain_history)
        error = None
    except Exception as exc:
        logger.exception("run_turn failed")
        result = None
        error = exc

    if error is not None:
        answer = PIPELINE_ERROR_MESSAGE
        details = {"error": str(error)}
    else:
        answer = result["answer"]
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

    st.session_state.history.append({"role": "assistant", "content": answer, "details": details})
    get_corpus_stats.clear()
    st.rerun()