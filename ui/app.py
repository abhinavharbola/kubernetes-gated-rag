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
# not a separate or leaked conversation, it's this app's own safety_gate()
# doing its job, but it's not meant for a normal terminal, only useful when
# actually debugging the gate itself. Quiet by default; flip back to INFO
# locally if you need to see what a gate call is actually doing.
logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import streamlit as st

from src.providers.clients import qdrant_client
from src.config import settings
from src.graph import run_turn
from src.retrieval.rerank import preload as preload_rerank

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Kubernetes RAG", page_icon="◧", layout="centered")


@st.cache_resource(show_spinner="Warming up the local reranker (one-time, first load only)…")
def _warm_up_models() -> bool:
    preload_rerank()
    return True

_warm_up_models()

PIPELINE_ERROR_MESSAGE = (
    "Something went wrong completing that request. This is usually a transient "
    "provider issue, try again in a moment."
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* --- Design system ---
   This is a Kubernetes Q&A tool built around admission-style gating: every
   question passes through checks before it's allowed to reach generation,
   the same way the API server runs admission webhooks on a request before
   it's persisted. The live trace panel mirrors the same node-and-arrow
   pipeline diagram used in "How this works", just colored by what
   actually happened on this specific turn instead of a static overview,
   so a person only has to learn one visual language for the pipeline,
   not two.

   Light palette, but a clearly gray-blue canvas rather than a near-white
   one, panels are a soft off-white raised on top of it, not pure #FFF
   either, so the page reads as a considered light workspace rather than
   "everything is white". Monospace is reserved for real data (paths,
   scores, provider names, latency), not for decorative labels. */
:root {
    --bg: #D9DEE7;
    --panel: #F9FAFC;
    --panel-raised: #EFF2F6;
    --border: rgba(17, 24, 39, 0.11);
    --border-strong: rgba(17, 24, 39, 0.20);
    --text: #12151C;
    --text-muted: #565F70;
    --text-faint: #838C9D;
    --blue: #2E56D9;
    --blue-soft: rgba(46, 86, 217, 0.09);
    --green: #157F45;
    --green-soft: rgba(21, 127, 69, 0.11);
    --amber: #A8650A;
    --amber-soft: rgba(168, 101, 10, 0.11);
    --red: #C22E2E;
    --red-soft: rgba(194, 46, 46, 0.10);
    --shadow-sm: 0 1px 2px rgba(17, 24, 39, 0.06), 0 1px 1px rgba(17, 24, 39, 0.04);
    --shadow-md: 0 4px 14px rgba(17, 24, 39, 0.10), 0 1px 2px rgba(17, 24, 39, 0.06);
    --content-width: 820px;
    --font-display: "Sora", "Segoe UI", sans-serif;
    --font-sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-mono: "IBM Plex Mono", "Fira Code", ui-monospace, monospace;
}

.stApp { font-family: var(--font-sans); background: var(--bg); color: var(--text); }
.stApp [data-testid="stChatMessage"] { gap: 0.6rem; }
code, .mono { font-family: var(--font-mono); }

/* Streamlit paints its header bar and the bottom chat-input bar from
   separate containers that don't inherit .stApp's background, theme.toml
   now matches this palette too (belt and suspenders: the toml drives
   Streamlit's own native widget colors, this covers anything it doesn't). */
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: var(--bg) !important;
}
/* the chat-input bar spans the full viewport width by default; constrain
   and center the actual input box to match the content column so it reads
   as one deliberate composition instead of an edge-to-edge form field. */
[data-testid="stBottomBlockContainer"] { display: flex !important; justify-content: center !important; }
[data-testid="stChatInput"] {
    background: var(--panel) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 10px !important;
    box-shadow: var(--shadow-md) !important;
    max-width: var(--content-width) !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-faint) !important; }
.stButton button {
    border-radius: 8px !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text) !important;
    background: var(--panel) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
}
.stButton button:hover {
    border-color: var(--blue) !important;
    color: var(--blue) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}
.stMarkdown, .stApp p, .stApp li { color: var(--text); }

/* the "usable width" of centered layout is capped by Streamlit's own
   block-container max-width (~730px); this is the one place the content
   column's width is set — everything else (header, welcome block, chat
   input) matches var(--content-width) instead of picking its own number,
   so nothing in the main column reads as narrower or wider than anything
   else. padding-top must clear stHeader's own height (a fixed bar painted
   over the top of the scrollable content, same bg color as the page
   above). */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: var(--content-width) !important;
    padding-top: 3.5rem !important;
}

/* --- masthead --- */
.app-header {
    display: flex; flex-direction: column; align-items: center; text-align: center;
    margin: 0 auto 1.6rem auto;
    padding: 0 0 1.6rem 0; border-bottom: 1px solid var(--border);
}
.app-header .title-block h1 {
    margin: 0; font-family: var(--font-display); font-size: 2.3rem;
    font-weight: 700; letter-spacing: -0.01em; color: var(--text); line-height: 1.2;
}
.app-header .title-block .tagline {
    color: var(--text-muted); font-size: 1rem; margin: 0.7rem 0 0 0; line-height: 1.6;
}

/* --- welcome / onboarding --- */
.st-key-welcome_block { margin: 0 auto 0.4rem auto; }
.st-key-welcome_block [data-testid="stExpander"] {
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    box-shadow: var(--shadow-sm) !important; background: var(--panel) !important;
    overflow: hidden;
}

/* --- pipeline trace (live, per-turn) ---
   Same node-and-arrow shape as the static "How this works" diagram below,
   just colored by what actually happened on this turn instead of showing
   the same neutral blue for every stage: green passed, red blocked (the
   request stopped there, later nodes never ran), amber a non-blocking
   miss the request continued past anyway (currently only a cache miss),
   gray a stage the request never reached because an earlier one already
   blocked or resolved the turn. */
.trace-diagram-wrap { overflow-x: auto; padding: 0.2rem 0.1rem 0.5rem 0.1rem; margin-top: 0.6rem; }
.trace-diagram {
    display: flex; flex-wrap: nowrap; align-items: stretch;
    gap: 0.5rem; width: max-content; min-width: 100%;
}
.trace-node {
    position: relative; display: flex; flex-direction: column; gap: 0.22rem;
    padding: 0.65rem 0.75rem 0.6rem 0.75rem; border-radius: 8px; width: 128px; flex-shrink: 0;
    background: var(--panel); border: 1px solid var(--border); box-shadow: var(--shadow-sm);
    border-top: 3px solid var(--border-strong);
}
.trace-node.status-pass { border-top-color: var(--green); }
.trace-node.status-fail { border-top-color: var(--red); }
.trace-node.status-skip { border-top-color: var(--amber); }
.trace-node.status-unreached { border-top-color: var(--border-strong); opacity: 0.45; box-shadow: none; }
.trace-node .trace-node-badge {
    position: absolute; top: -10px; right: -10px; width: 19px; height: 19px; border-radius: 50%;
    color: #FFFFFF; font-size: 0.66rem; font-weight: 700; line-height: 1;
    display: flex; align-items: center; justify-content: center;
}
.trace-node.status-pass .trace-node-badge { background: var(--green); }
.trace-node.status-fail .trace-node-badge { background: var(--red); }
.trace-node.status-skip .trace-node-badge { background: var(--amber); }
.trace-node.status-unreached .trace-node-badge { background: var(--text-faint); }
.trace-node .trace-node-title {
    font-family: var(--font-sans); font-size: 0.78rem; font-weight: 600; color: var(--text);
}
.trace-node .trace-node-message {
    font-family: var(--font-mono); font-size: 0.68rem; line-height: 1.4; color: var(--text-muted);
    overflow-wrap: break-word;
}
.trace-arrow-live {
    display: flex; align-items: center; color: var(--text-faint); font-size: 1.1rem; flex-shrink: 0;
}
.trace-arrow-live.unreached { opacity: 0.35; }
.trace-footer {
    display: flex; justify-content: flex-end; margin: 0.35rem 0.1rem 0 0.1rem;
    font-family: var(--font-mono); font-size: 0.72rem; color: var(--text-faint);
}

/* --- pipeline diagram (static, explanatory, inside "How this works") ---
   This one genuinely is a numbered sequence, a request moves through these
   stages in order, so numbering here documents structure rather than
   decorating it. */
.pipeline-diagram-wrap { overflow-x: auto; padding: 0.2rem 0.1rem 0.5rem 0.1rem; }
.pipeline-diagram {
    display: flex; flex-wrap: nowrap; align-items: stretch;
    gap: 0.5rem; margin: 0.6rem 0 0.2rem 0; width: max-content; min-width: 100%;
}
.pipeline-node {
    position: relative; display: flex; flex-direction: column; gap: 0.22rem;
    padding: 0.65rem 0.75rem 0.6rem 0.75rem; border-radius: 8px; width: 118px; flex-shrink: 0;
    background: var(--panel); border: 1px solid var(--border); box-shadow: var(--shadow-sm);
    border-top: 2px solid var(--blue);
}
.pipeline-node .pipeline-node-badge {
    position: absolute; top: -9px; right: -9px; width: 18px; height: 18px; border-radius: 50%;
    background: var(--blue); color: #FFFFFF; font-family: var(--font-mono); font-size: 0.62rem;
    font-weight: 600; display: flex; align-items: center; justify-content: center;
}
.pipeline-node .pipeline-node-title {
    font-family: var(--font-sans); font-size: 0.78rem; font-weight: 600; color: var(--text);
}
.pipeline-node .pipeline-node-desc {
    font-size: 0.71rem; line-height: 1.4; color: var(--text-muted);
}
.pipeline-arrow {
    display: flex; align-items: center; color: var(--text-faint); font-size: 1.1rem; flex-shrink: 0;
}

/* --- sources: a manifest of what actually got retrieved and passed --- */
.source-row {
    display: grid; grid-template-columns: 1fr 90px 54px; align-items: center; gap: 0.6rem;
    font-family: var(--font-mono); font-size: 0.78rem;
    padding: 0.42rem 0; border-bottom: 1px solid var(--border);
}
.source-row:last-child { border-bottom: none; }
.source-meta { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
.source-score-wrap { height: 5px; border-radius: 3px; background: var(--border); overflow: hidden; }
.source-score-bar { height: 100%; background: var(--blue); }
.source-row .score { color: var(--text-muted); text-align: right; }

/* --- welcome / onboarding: example question buttons --- */
[class*="st-key-example_"] button {
    text-align: left !important;
    height: 96px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    overflow: hidden !important;
    line-height: 1.35;
    padding: 0.9rem 1.1rem !important;
}
.welcome-caption {
    color: var(--text-muted); font-size: 0.88rem; margin: 0.2rem 0 1.1rem 0; text-align: center;
}

/* --- sidebar: cluster status panel --- */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--border); background: var(--panel);
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p { color: var(--text); }
.section-label {
    font-family: var(--font-sans); font-size: 0.76rem; font-weight: 600;
    color: var(--text-muted); margin: 0.2rem 0 0.5rem 0;
    border-bottom: 1px solid var(--border); padding-bottom: 0.3rem;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono); font-size: 1.15rem; color: var(--text);
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.66rem; color: var(--text-muted); }

.provider-list { display: flex; flex-direction: column; gap: 0.6rem; }
.provider-row { padding: 0.1rem 0; }
.provider-row .provider-name-line {
    font-family: var(--font-sans); font-size: 0.83rem; font-weight: 500; color: var(--text);
    display: flex; align-items: center; gap: 0.55rem;
}
.provider-row .provider-role {
    font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-faint);
    line-height: 1.4; margin-top: 0.15rem; padding-left: 1.15rem;
}
.status-dot-inline { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.status-dot-inline.up { background: var(--green); box-shadow: 0 0 4px var(--green-soft); }
.status-dot-inline.down { background: var(--red); }

.error-note {
    font-family: var(--font-mono); font-size: 0.76rem; color: var(--red);
    background: var(--red-soft); border: 1px solid rgba(194, 46, 46, 0.22);
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

PROVIDERS = [
    {"label": "Groq", "role": "generation (account A)", "check": lambda: bool(settings.groq_api_key)},
    {
        "label": "Groq",
        "role": "generation (account B)",
        "check": lambda: bool(settings.groq_api_key_secondary),
    },
    {"label": "NIM", "role": "generation fallback, planner, NeMoGuard", "check": lambda: bool(settings.nvidia_nim_api_key)},
    {"label": "Gemini", "role": "embeddings, eval judge", "check": lambda: bool(settings.gemini_api_key)},
    {
        "label": "Qdrant",
        "role": "vector store",
        "check": lambda: bool(settings.qdrant_url and settings.qdrant_api_key),
    },
]

PIPELINE_STAGES = [
    {"label": "Cache", "desc": "fast exact + semantic lookup"},
    {"label": "Safety", "desc": "blocks unsafe content"},
    {"label": "Topic", "desc": "blocks off-topic questions"},
    {"label": "Retrieve", "desc": "dense vector search"},
    {"label": "Rerank", "desc": "cross-encoder relevance gate"},
    {"label": "Generate", "desc": "grounded answer"},
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
# whether a new turn is about to be generated, so it can fade itself.
prompt = st.chat_input("Ask a Kubernetes question")
if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# ---------- header ----------

st.markdown(
    """
    <div class="app-header">
        <div class="title-block">
            <h1>Kubernetes Q&amp;A</h1>
            <div class="tagline">Grounded Kubernetes answers from your own docs. Safety-checked,
            relevance-gated, and cached for speed.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- sidebar ----------

with st.sidebar:
    st.markdown('<div class="section-label">Corpus</div>', unsafe_allow_html=True)
    doc_count, cache_count = get_corpus_stats()
    corpus_col1, corpus_col2 = st.columns(2)
    corpus_col1.metric("Chunks indexed", doc_count if doc_count is not None else "—")
    corpus_col2.metric("Cached answers", cache_count if cache_count is not None else "—")

    st.markdown('<div class="section-label" style="margin-top: 1rem;">This session</div>', unsafe_allow_html=True)
    questions_asked, hit_rate, avg_latency = get_session_stats()
    session_col1, session_col2, session_col3 = st.columns(3)
    session_col1.metric("Asked", questions_asked)
    session_col2.metric("Cache hit", hit_rate)
    session_col3.metric("Avg time", avg_latency)

    st.markdown('<div class="section-label" style="margin-top: 1rem;">Providers</div>', unsafe_allow_html=True)
    provider_rows = "".join(
        f'<div class="provider-row">'
        f'<div class="provider-name-line">'
        f'<span class="status-dot-inline {"up" if provider["check"]() else "down"}"></span>'
        f'<span class="provider-name">{provider["label"]}</span>'
        f"</div>"
        f'<div class="provider-role">{provider["role"]}</div>'
        f"</div>"
        for provider in PROVIDERS
    )
    st.markdown(f'<div class="provider-list">{provider_rows}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top: 1rem;">Display</div>', unsafe_allow_html=True)
    show_trace = st.toggle("Show pipeline trace", value=False)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        get_corpus_stats.clear()
        st.rerun()


# ---------- pipeline trace ----------

def build_trace_conditions(details: dict) -> list[dict]:
    if details.get("error"):
        return [{"type": "Pipeline", "status": False, "message": "error"}]

    blocked_stage = details.get("blocked_stage")
    cache_layer = details.get("cache_layer")
    cache_checked = details.get("exact_cache_checked", False)

    if cache_layer == "exact":
        return [{"type": "Cache", "status": True, "message": "exact hit"}]

    conditions = []
    if blocked_stage == "safety":
        conditions.append({"type": "Safety", "status": False, "message": "blocked"})
        return conditions
    conditions.append({"type": "Safety", "status": True, "message": "passed"})

    if blocked_stage == "topic":
        conditions.append({"type": "Topic", "status": False, "message": "off-topic"})
        return conditions
    conditions.append({"type": "Topic", "status": True, "message": "on-topic"})

    if cache_layer == "semantic":
        conditions.append({"type": "Cache", "status": True, "message": "semantic hit"})
        return conditions
    conditions.append({"type": "Cache", "status": None, "message": "miss" if cache_checked else "lookup"})

    candidates_count = details.get("candidates_count", 0)
    conditions.append({"type": "Retrieve", "status": True, "message": f"{candidates_count} found"})
    reranked_count = details.get("reranked_count", 0)
    if reranked_count == 0:
        conditions.append({"type": "Rerank", "status": False, "message": "0 survived"})
        return conditions
    conditions.append({"type": "Rerank", "status": True, "message": f"{reranked_count}/{candidates_count} passed"})

    provider = details.get("provider")
    model = details.get("model")
    if provider:
        message = f"{provider} ({model})" if model else provider
        conditions.append({"type": "Generate", "status": True, "message": message})
    return conditions


def build_trace_nodes(details: dict) -> list[dict]:
    conditions = build_trace_conditions(details)
    nodes = []
    for condition in conditions:
        status = condition["status"]
        nodes.append({
            "label": condition["type"],
            "status": "fail" if status is False else "skip" if status is None else "pass",
            "message": condition["message"],
        })
    return nodes


def render_trace(details: dict) -> None:
    nodes = build_trace_nodes(details)
    if nodes:
        badge_icon = {"pass": "&#10003;", "fail": "&#10005;", "skip": "~"}
        node_html = ""
        for i, node in enumerate(nodes):
            node_html += (
                f'<div class="trace-node status-{node["status"]}">'
                f'<span class="trace-node-badge">{badge_icon[node["status"]]}</span>'
                f'<span class="trace-node-title">{node["label"]}</span>'
                f'<span class="trace-node-message">{node["message"]}</span>'
                f"</div>"
            )
            if i < len(nodes) - 1:
                node_html += '<div class="trace-arrow-live">&#8594;</div>'

        latency = details.get("latency_seconds")
        footer_html = f'<div class="trace-footer">{latency:.2f}s total</div>' if latency is not None else ""

        st.markdown(
            f'<div class="trace-diagram-wrap"><div class="trace-diagram">{node_html}</div></div>{footer_html}',
            unsafe_allow_html=True,
        )

    if details.get("error"):
        st.markdown(f'<div class="error-note">{PIPELINE_ERROR_MESSAGE}</div>', unsafe_allow_html=True)
        return

    sources = details.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                path = source["metadata"].get("source_path", "unknown")
                # rerank_score is absent when rerank_and_gate degraded to
                # unfiltered retrieval order (FlashRank failure, see
                # src/retrieval/rerank.py) — there's no score to gate on in
                # that case, so show "n/a" instead of a bar rather than
                # raising a KeyError on a candidate that was never scored.
                score = source.get("rerank_score")
                if score is None:
                    st.markdown(
                        f'<div class="source-row">'
                        f'<div class="source-meta">{path}</div>'
                        f'<div class="source-score-wrap"></div>'
                        f'<span class="score">n/a</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    continue
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

# checked against `prompt` too, not just history: history is only empty
# BEFORE this run's user message gets appended further down, so on the
# very first-ever submission this condition would otherwise still be true
# during the same run that's processing that submission, showing the
# welcome block and the "running the pipeline" spinner at once.
if not st.session_state.history and not prompt:
    with st.container(key="welcome_block"):
        with st.expander("How this works", expanded=False):
            diagram_html = ""
            for i, stage in enumerate(PIPELINE_STAGES):
                diagram_html += (
                    f'<div class="pipeline-node">'
                    f'<span class="pipeline-node-badge">{i + 1}</span>'
                    f'<span class="pipeline-node-title">{stage["label"]}</span>'
                    f'<span class="pipeline-node-desc">{stage["desc"]}</span>'
                    f"</div>"
                )
                if i < len(PIPELINE_STAGES) - 1:
                    diagram_html += '<div class="pipeline-arrow">&#8594;</div>'
            st.markdown(
                f'<div class="pipeline-diagram-wrap"><div class="pipeline-diagram">{diagram_html}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<p class="welcome-caption">Try one of these, or ask your own below.</p>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_prompt = question

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
                "exact_cache_checked": result.get("exact_cache_checked"),
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
