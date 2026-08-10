import logging
import sys
from pathlib import Path

# app.py lives in ui/, one level below the repo root, so the repo root
# (where the src/ package lives) has to be added explicitly. Without this,
# `from src...` only works by accident of whatever directory the process
# happened to be launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NeMo Guardrails and httpx both log at INFO by default, including the full
# Colang phase-by-phase prompt/response trace for every gate check. That's
# not a separate or leaked conversation — it's this app's own safety_gate()
# doing its job — but it's not meant for a normal terminal, only useful when
# actually debugging the gate itself. Quiet by default; flip back to INFO
# locally if you need to see what a gate call is actually doing.
logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import streamlit as st

from src.clients import qdrant_client
from src.config import settings
from src.graph import run_turn

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Kubernetes RAG", page_icon="◧", layout="centered")

PIPELINE_ERROR_MESSAGE = (
    "Something went wrong completing that request. This is usually a transient "
    "provider issue, try again in a moment."
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #FBF7F2;
    --surface: #FFFFFF;
    --border: rgba(58, 46, 39, 0.10);
    --border-accent: rgba(199, 106, 63, 0.28);
    --text-primary: #3A2E27;
    --text-muted: #8A7A6D;
    --text-faint: #C2B3A5;
    --accent: #C76A3F;
    --accent-strong: #A8532F;
    --accent-soft: rgba(199, 106, 63, 0.10);
    --warn-strong: #A8532F;
    --warn-soft: rgba(199, 106, 63, 0.10);
    --danger: #B33A2E;
    --danger-soft: rgba(179, 58, 46, 0.08);
    --neutral: #C2B3A5;
    --font-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-mono: "JetBrains Mono", "Fira Code", ui-monospace, monospace;
}

.stApp { font-family: var(--font-sans); background: var(--bg); }
.stApp [data-testid="stChatMessage"] { gap: 0.6rem; }
code, .mono { font-family: var(--font-mono); }

/* --- header --- */
.app-header {
    display: flex; justify-content: space-between; align-items: flex-end;
    padding-bottom: 0.7rem; margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-accent);
}
.app-header .title-block h1 {
    margin: 0; font-family: var(--font-sans); font-size: 1.4rem;
    font-weight: 700; letter-spacing: -0.015em; color: var(--text-primary);
}
.app-header .title-block .tagline { color: var(--text-muted); font-size: 0.83rem; margin-top: 0.22rem; }
.status-pill {
    font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.02em;
    padding: 0.28rem 0.7rem; border-radius: 999px;
    display: inline-flex; align-items: center; gap: 0.4rem; white-space: nowrap;
}
.status-pill.operational { color: var(--accent-strong); border: 1px solid var(--border-accent); background: var(--accent-soft); }
.status-pill.degraded { color: var(--danger); border: 1px solid rgba(179, 58, 46, 0.3); background: var(--danger-soft); }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; display: inline-block; }

/* --- history fade while a new turn is generating --- */
.st-key-history_normal { opacity: 1; transition: opacity 0.3s ease; }
.st-key-history_dim { opacity: 0.38; filter: saturate(0.7); transition: opacity 0.3s ease; pointer-events: none; }

/* --- pipeline trace --- */
.trace-row { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.4rem; margin: 0.6rem 0 0.15rem 0; }
.trace-step {
    display: flex; flex-direction: column; justify-content: center; gap: 0.08rem;
    padding: 0.32rem 0.65rem; border-radius: 6px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    min-width: 82px;
}
.trace-step.fail { border-left-color: var(--danger); }
.trace-step.hit { border-left-color: var(--accent-strong); background: var(--accent-soft); }
.trace-step.skip { border-left-color: var(--neutral); opacity: 0.65; }
.trace-step.pending { border-left-color: var(--text-faint); border-left-style: dashed; opacity: 0.6; background: transparent; }
.trace-step .trace-label {
    font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--text-muted);
}
.trace-step .trace-value {
    font-family: var(--font-mono); font-size: 0.76rem; color: var(--text-primary);
    display: flex; align-items: center; gap: 0.3rem;
}
.trace-glyph { font-size: 0.7rem; line-height: 1; }
.trace-glyph.ok { color: var(--accent-strong); }
.trace-glyph.fail { color: var(--danger); }
.trace-glyph.skip { color: var(--text-faint); }
.trace-arrow { display: flex; align-items: center; color: var(--neutral); font-size: 0.85rem; padding: 0 0.05rem; }
.trace-latency {
    margin-left: auto; align-self: center; font-family: var(--font-mono);
    font-size: 0.7rem; color: var(--text-faint); white-space: nowrap; padding-left: 0.5rem;
}

/* --- sources --- */
.source-row {
    display: grid; grid-template-columns: 1fr 90px 54px; align-items: center; gap: 0.6rem;
    font-family: var(--font-mono); font-size: 0.78rem;
    padding: 0.42rem 0; border-bottom: 1px solid var(--border);
}
.source-row:last-child { border-bottom: none; }
.source-meta { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-primary); }
.source-score-wrap { height: 5px; border-radius: 3px; background: rgba(58, 46, 39, 0.08); overflow: hidden; }
.source-score-bar { height: 100%; background: var(--accent); }
.source-row .score { color: var(--accent-strong); text-align: right; }

/* --- welcome / onboarding --- */
.example-btn button { text-align: left !important; }
.welcome-eyebrow {
    font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-faint); margin: 0.2rem 0 0.4rem 0;
}
.welcome-note { color: var(--text-muted); font-size: 0.86rem; line-height: 1.55; margin: 0 0 0.7rem 0; }
.welcome-caption { color: var(--text-faint); font-size: 0.74rem; margin: 0.3rem 0 1.1rem 0; }

/* --- sidebar --- */
[data-testid="stSidebar"] { border-right: 1px solid var(--border-accent); background: var(--surface); }
.eyebrow {
    font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-faint); margin: 0.2rem 0 0.5rem 0;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono); font-size: 1.2rem; color: var(--accent-strong);
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.66rem; color: var(--text-muted); }

.provider-list { display: flex; flex-direction: column; gap: 0.42rem; }
.provider-row {
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-primary);
    display: flex; align-items: center; gap: 0.55rem;
}
.status-dot-inline { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.status-dot-inline.up { background: var(--accent-strong); box-shadow: 0 0 4px rgba(199, 106, 63, 0.5); }
.status-dot-inline.down { background: var(--danger); }

.error-note {
    font-family: var(--font-mono); font-size: 0.76rem; color: var(--danger);
    background: var(--danger-soft); border: 1px solid rgba(179, 58, 46, 0.25);
    border-radius: 6px; padding: 0.5rem 0.7rem; margin-top: 0.5rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

EXAMPLE_QUESTIONS = [
    "How do I pin a container to a specific image version?",
    "What's the difference between a Deployment and a StatefulSet?",
    "How do I safely delete a single Pod without affecting the rest of my Deployment?",
    "How do I give each StatefulSet replica its own persistent storage?",
]

PROVIDER_KEYS = {
    "NIM": lambda: bool(settings.nvidia_nim_api_key),
    "Groq": lambda: bool(settings.groq_api_key),
    "Gemini": lambda: bool(settings.gemini_api_key),
    "Qdrant": lambda: bool(settings.qdrant_url and settings.qdrant_api_key),
}

PIPELINE_STAGES = ["Safety", "Topic", "Cache", "Retrieve", "Rerank", "Generate"]

STATUS_GLYPH = {"ok": "&#10003;", "hit": "&#10003;", "fail": "&#10007;", "skip": "&#8722;"}


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
# bottom of the page regardless of call order (Streamlit's own behavior) —
# doing this before rendering history lets the history block below know
# whether a new turn is about to be generated, so it can fade itself.
prompt = st.chat_input("Ask a Kubernetes question")
if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

is_generating = bool(prompt)

# ---------- header ----------

provider_status = {label: check() for label, check in PROVIDER_KEYS.items()}
down_providers = [label for label, ok in provider_status.items() if not ok]
system_ok = not down_providers
status_class = "operational" if system_ok else "degraded"
status_label = "Operational" if system_ok else f"Degraded &middot; {', '.join(down_providers)}"

st.markdown(
    f"""
    <div class="app-header">
        <div class="title-block">
            <h1>Kubernetes Q&amp;A</h1>
            <div class="tagline">Retrieval-augmented, grounded in your ingested docs.</div>
        </div>
        <span class="status-pill {status_class}"><span class="status-dot"></span>{status_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- sidebar ----------

with st.sidebar:
    st.markdown('<div class="eyebrow">Corpus</div>', unsafe_allow_html=True)
    doc_count, cache_count = get_corpus_stats()
    corpus_col1, corpus_col2 = st.columns(2)
    corpus_col1.metric("Chunks indexed", doc_count if doc_count is not None else "—")
    corpus_col2.metric("Cached answers", cache_count if cache_count is not None else "—")

    st.markdown('<div class="eyebrow" style="margin-top: 1rem;">This session</div>', unsafe_allow_html=True)
    questions_asked, hit_rate, avg_latency = get_session_stats()
    session_col1, session_col2, session_col3 = st.columns(3)
    session_col1.metric("Asked", questions_asked)
    session_col2.metric("Cache hit", hit_rate)
    session_col3.metric("Avg time", avg_latency)

    st.markdown('<div class="eyebrow" style="margin-top: 1rem;">Providers</div>', unsafe_allow_html=True)
    provider_rows = "".join(
        f'<div class="provider-row"><span class="status-dot-inline {"up" if ok else "down"}"></span>{label}</div>'
        for label, ok in provider_status.items()
    )
    st.markdown(f'<div class="provider-list">{provider_rows}</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top: 1rem;">Display</div>', unsafe_allow_html=True)
    show_trace = st.toggle("Show pipeline trace", value=False)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        get_corpus_stats.clear()
        st.rerun()


# ---------- pipeline trace ----------

def build_trace_steps(details: dict) -> list[dict]:
    """Mirrors the actual LangGraph routing in src/graph.py, so what's shown
    here is the real path this specific turn took, not a generic summary."""
    if details.get("error"):
        return [{"label": "Pipeline", "value": "error", "status": "fail"}]

    steps = []
    blocked_stage = details.get("blocked_stage")

    if blocked_stage == "safety":
        steps.append({"label": "Safety", "value": "blocked", "status": "fail"})
        return steps
    steps.append({"label": "Safety", "value": "passed", "status": "ok"})

    if blocked_stage == "topic":
        steps.append({"label": "Topic", "value": "off-topic", "status": "fail"})
        return steps
    steps.append({"label": "Topic", "value": "on-topic", "status": "ok"})

    cache_layer = details.get("cache_layer")
    if cache_layer:
        steps.append({"label": "Cache", "value": f"{cache_layer} hit", "status": "hit"})
        return steps
    steps.append({"label": "Cache", "value": "miss", "status": "skip"})

    candidates_count = details.get("candidates_count", 0)
    steps.append({"label": "Retrieve", "value": f"{candidates_count} found", "status": "ok"})

    reranked_count = details.get("reranked_count", 0)
    if reranked_count == 0:
        steps.append({"label": "Rerank", "value": "0 survived", "status": "fail"})
        return steps
    steps.append({"label": "Rerank", "value": f"{reranked_count}/{candidates_count} passed", "status": "ok"})

    provider = details.get("provider")
    model = details.get("model")
    if provider:
        label = f"{provider} &middot; {model}" if model else provider
        steps.append({"label": "Generate", "value": label, "status": "ok"})

    return steps


def render_trace(details: dict) -> None:
    steps = build_trace_steps(details)
    step_html = ""
    for i, step in enumerate(steps):
        status = step.get("status", "fail")
        glyph = STATUS_GLYPH.get(status, "")
        glyph_html = f'<span class="trace-glyph {status}">{glyph}</span>' if glyph else ""
        step_html += (
            f'<div class="trace-step {status}">'
            f'<span class="trace-label">{step["label"]}</span>'
            f'<span class="trace-value">{glyph_html}{step["value"]}</span>'
            f"</div>"
        )
        if i < len(steps) - 1:
            step_html += '<div class="trace-arrow">&#8594;</div>'

    latency = details.get("latency_seconds")
    latency_html = f'<span class="trace-latency">{latency:.2f}s</span>' if latency is not None else ""

    st.markdown(f'<div class="trace-row">{step_html}{latency_html}</div>', unsafe_allow_html=True)

    if details.get("error"):
        st.markdown(f'<div class="error-note">{PIPELINE_ERROR_MESSAGE}</div>', unsafe_allow_html=True)
        return

    sources = details.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                path = source["metadata"].get("source_path", "unknown")
                score = source["rerank_score"]
                bar_width = max(0.0, min(1.0, score)) * 100
                st.markdown(
                    f'<div class="source-row">'
                    f'<div class="source-meta">{path}</div>'
                    f'<div class="source-score-wrap"><div class="source-score-bar" style="width:{bar_width:.0f}%"></div></div>'
                    f'<span class="score">{score:.3f}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ---------- conversation ----------

if not st.session_state.history:
    st.markdown('<div class="welcome-eyebrow">How this works</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="welcome-note">Every answer passes through safety and topic '
        "gating, a two-layer cache, and a hard-gated reranker before it reaches the "
        "model, or gets stopped along the way. Turn on \"Show pipeline trace\" in the "
        "sidebar to see exactly which stages ran and what each one decided.</p>",
        unsafe_allow_html=True,
    )
    if show_trace:
        preview_html = ""
        for i, stage in enumerate(PIPELINE_STAGES):
            preview_html += (
                f'<div class="trace-step pending">'
                f'<span class="trace-label">{stage}</span>'
                f'<span class="trace-value">pending</span>'
                f"</div>"
            )
            if i < len(PIPELINE_STAGES) - 1:
                preview_html += '<div class="trace-arrow">&#8594;</div>'
        st.markdown(f'<div class="trace-row">{preview_html}</div>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-caption">Try one of these, or ask your own below.</p>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_prompt = question
            st.markdown("</div>", unsafe_allow_html=True)

# past turns fade out while a new one is being generated, so attention goes
# to the active exchange below rather than the settled conversation above it
history_container = st.container(key="history_dim" if is_generating else "history_normal")
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
        with st.spinner("Running the pipeline..."):
            plain_history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.history[:-1]]
            try:
                result = run_turn(prompt, plain_history)
                error = None
            except Exception as exc:
                # narrowing RETRYABLE in src/llm.py means non-transient errors
                # (bad request, auth, etc.) now propagate here instead of being
                # swallowed into a RuntimeError after two wasted retries. Log
                # the real exception for debugging, show a clean message to the
                # user rather than a stack trace or raw provider error text.
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
                "provider": result.get("provider"),
                "model": result.get("model"),
                "candidates_count": len(result.get("candidates") or []),
                "reranked_count": len(result.get("reranked") or []),
                "sources": result.get("reranked") if not result.get("cache_layer") else None,
                "latency_seconds": result.get("latency_seconds"),
            }
        if show_trace:
            render_trace(details)

    st.session_state.history.append({"role": "assistant", "content": answer, "details": details})
    get_corpus_stats.clear()
