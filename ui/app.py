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

from src.providers.clients import qdrant_client
from src.config import settings
from src.graph import run_turn
from src.guardrails import preload as preload_guardrails
from src.retrieval.rerank import preload as preload_rerank

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Kubernetes RAG", page_icon="◧", layout="centered")


@st.cache_resource(show_spinner="Warming up safety and rerank models (one-time, first load only)…")
def _warm_up_models() -> bool:
    # NeMo Guardrails' Colang flow matcher and FlashRank's cross-encoder are
    # both lazily built on first use by default (see their preload()
    # docstrings in src/guardrails/gates.py and src/retrieval/rerank.py) —
    # deliberately so, since a process that never runs a real turn (tests,
    # ingest.py) shouldn't pay for either. This app always needs both, so
    # force them to build now, during page load with its own spinner,
    # rather than silently inside the first user question's latency.
    # st.cache_resource makes this run exactly once per server process,
    # shared across every session, not once per rerun.
    preload_guardrails()
    preload_rerank()
    return True


_warm_up_models()

PIPELINE_ERROR_MESSAGE = (
    "Something went wrong completing that request. This is usually a transient "
    "provider issue, try again in a moment."
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* --- Design system ---
   This app is a series of clearance gates (safety, topic, ingestion), so
   the signature motif is a customs/ledger "clearance stamp" rather than a
   generic pill badge — see .trace-stamp below. Palette is a rich warm
   ledger cream with a deep navy-indigo accent (a nod to Kubernetes' own
   brand blue, pushed darker/more saturated than a mid-tone slate for a
   more premium, higher-contrast feel, and deliberately not the
   terracotta-on-cream combo that reads as an obvious AI default) and a
   muted brass reserved for the stamp itself, the one place this page
   spends its visual boldness. */
:root {
    --bg: #F1E9D8;
    --surface: #FAF4E6;
    --surface-raised: #FFFDF7;
    --border: rgba(40, 34, 24, 0.14);
    --border-accent: rgba(31, 58, 92, 0.32);
    --ink: #29231A;
    --ink-muted: #6B5E4C;
    --ink-faint: #A69577;
    --accent: #1F3A5C;
    --accent-strong: #16283F;
    --accent-soft: rgba(31, 58, 92, 0.10);
    --brass: #93712F;
    --brass-soft: rgba(147, 113, 47, 0.14);
    --danger: #9C4A36;
    --danger-soft: rgba(156, 74, 54, 0.11);
    --neutral: #8C8370;
    --font-sans: "Manrope", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-mono: "IBM Plex Mono", "Fira Code", ui-monospace, monospace;
}

.stApp { font-family: var(--font-sans); background: var(--bg); color: var(--ink); }
.stApp [data-testid="stChatMessage"] { gap: 0.6rem; }
code, .mono { font-family: var(--font-mono); }

/* Streamlit paints its header bar and the bottom chat-input bar from
   separate containers that don't inherit .stApp's background — theme.toml
   now matches this palette too (belt and suspenders: the toml drives
   Streamlit's own native widget colors, this covers anything it doesn't). */
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background: var(--bg) !important;
}
[data-testid="stChatInput"] {
    background: var(--surface-raised) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 4px !important;
}
[data-testid="stChatInput"] textarea { color: var(--ink) !important; }
.stButton button {
    border-radius: 3px !important;
    border: 1px solid var(--border-accent) !important;
    color: var(--ink) !important;
    background: var(--surface-raised) !important;
}
.stButton button:hover {
    border-color: var(--accent) !important;
    color: var(--accent-strong) !important;
}
.stMarkdown, .stApp p, .stApp li { color: var(--ink); }

/* the "usable width" of centered layout is capped by Streamlit's own
   block-container max-width (~730px); widen it so the header, diagram, and
   example grid all have room to breathe instead of wrapping constantly.
   padding-top must clear stHeader's own height (a fixed bar painted over
   the top of the scrollable content, same bg color as the page above) —
   2rem wasn't enough, so the masthead eyebrow was visually clipped by the
   header bar sitting on top of it as content scrolled underneath. */
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 980px !important;
    padding-top: 4.5rem !important;
}

/* --- masthead ---
   Ledger/manifest framing: a small eyebrow label above the title, thin
   double rule below it (top rule solid, bottom rule dashed, like a form's
   cut line). Title is a clean sans-serif, not italic — a professional
   documentation-tool register rather than an editorial one. */
.app-header {
    display: flex; flex-direction: column; align-items: center; text-align: center;
    padding: 0.2rem 0 0.3rem 0; margin-bottom: 0;
}
.app-header .masthead-eyebrow {
    font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--ink-faint); margin-bottom: 0.55rem;
}
.app-header .title-block h1 {
    margin: 0; font-family: var(--font-sans); font-style: normal; font-size: 2.35rem;
    font-weight: 700; letter-spacing: -0.02em; color: var(--ink); line-height: 1.15;
}
.app-header .title-block .tagline {
    color: var(--ink-muted); font-size: 0.96rem; margin: 0.65rem auto 0 auto;
    max-width: 900px; line-height: 1.5; white-space: nowrap;
}
.app-header .masthead-rule {
    width: 100%; max-width: 820px; margin-top: 1.1rem;
    border: none; border-top: 1px solid var(--ink); opacity: 0.55;
}
.app-header .masthead-rule.dashed {
    margin-top: 0.28rem; border-top: 1px dashed var(--border-accent);
}
@media (max-width: 900px) {
    /* below this width the sidebar+content area can't fit the tagline on
       one line without horizontal overflow — fall back to normal wrapping
       rather than forcing a scrollbar. */
    .app-header .title-block .tagline { white-space: normal; }
}

/* --- welcome / onboarding --- */
.st-key-welcome_block { max-width: 900px; margin: 0 auto; text-align: center; }

/* --- pipeline trace (live, per-turn) ---
   Each step is a ledger entry; passed steps get a rotated brass clearance
   stamp instead of a checkmark icon — the signature element, used only
   here so it stays meaningful rather than decorative. */
.trace-row { display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.5rem; margin: 0.7rem 0 0.2rem 0; }
.trace-step {
    position: relative;
    display: flex; flex-direction: column; justify-content: center; gap: 0.1rem;
    padding: 0.36rem 0.7rem; border-radius: 3px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    min-width: 88px;
}
.trace-step.fail { border-left-color: var(--danger); }
.trace-step.hit { border-left-color: var(--brass); background: var(--brass-soft); }
.trace-step.skip { border-left-color: var(--neutral); opacity: 0.6; }
.trace-step.pending { border-left-color: var(--ink-faint); border-left-style: dashed; opacity: 0.55; background: transparent; }
.trace-step .trace-label {
    font-family: var(--font-mono); font-size: 0.6rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-muted);
}
.trace-step .trace-value {
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink);
    display: flex; align-items: center; gap: 0.32rem;
}
.trace-glyph { font-size: 0.72rem; line-height: 1; }
.trace-glyph.ok { color: var(--accent-strong); }
.trace-glyph.fail { color: var(--danger); }
.trace-glyph.skip { color: var(--ink-faint); }
.trace-arrow { display: flex; align-items: center; color: var(--neutral); font-size: 0.85rem; padding: 0 0.1rem; }
.trace-latency {
    margin-left: auto; align-self: center; font-family: var(--font-mono);
    font-size: 0.7rem; color: var(--ink-faint); white-space: nowrap; padding-left: 0.5rem;
}
/* the clearance stamp itself: a rotated brass ring in the corner of any
   "ok" step, evoking a customs clearance stamp on a manifest page */
.trace-step.ok::after {
    content: "CLEAR";
    position: absolute; top: -8px; right: -10px;
    width: 30px; height: 30px; border-radius: 50%;
    border: 1.5px solid var(--brass); color: var(--brass);
    font-family: var(--font-mono); font-size: 0.36rem; font-weight: 600;
    letter-spacing: 0.03em; text-align: center; line-height: 30px;
    transform: rotate(-14deg); background: var(--surface-raised);
    box-shadow: 0 1px 2px rgba(43, 38, 28, 0.12);
}

/* --- pipeline diagram (static, explanatory, inside "How this works") --- */
.pipeline-diagram-wrap { overflow-x: auto; padding: 0.2rem 0.1rem 0.5rem 0.1rem; }
.pipeline-diagram {
    display: flex; flex-wrap: nowrap; align-items: stretch;
    gap: 0.45rem; margin: 0.6rem 0 0.2rem 0; width: max-content; min-width: 100%;
    justify-content: center;
}
.pipeline-node {
    position: relative; display: flex; flex-direction: column; gap: 0.22rem;
    padding: 0.65rem 0.75rem 0.6rem 0.75rem; border-radius: 3px; width: 118px; flex-shrink: 0;
    background: var(--surface); border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
}
.pipeline-node .pipeline-node-badge {
    position: absolute; top: -9px; right: -9px; width: 18px; height: 18px; border-radius: 50%;
    background: var(--accent); color: var(--surface-raised); font-family: var(--font-mono); font-size: 0.62rem;
    font-weight: 600; display: flex; align-items: center; justify-content: center;
}
.pipeline-node .pipeline-node-title {
    font-family: var(--font-mono); font-size: 0.74rem; font-weight: 600;
    letter-spacing: 0.03em; color: var(--ink);
}
.pipeline-node .pipeline-node-desc {
    font-size: 0.71rem; line-height: 1.4; color: var(--ink-muted);
}
.pipeline-arrow {
    display: flex; align-items: center; color: var(--accent); font-size: 1.15rem; flex-shrink: 0;
    opacity: 0.6;
}

/* --- sources: styled as a cargo/document manifest table, not a card list --- */
.source-row {
    display: grid; grid-template-columns: 1fr 90px 54px; align-items: center; gap: 0.6rem;
    font-family: var(--font-mono); font-size: 0.78rem;
    padding: 0.42rem 0; border-bottom: 1px dashed var(--border);
}
.source-row:last-child { border-bottom: none; }
.source-meta { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink); }
.source-score-wrap { height: 5px; border-radius: 2px; background: var(--border); overflow: hidden; }
.source-score-bar { height: 100%; background: var(--accent); }
.source-row .score { color: var(--accent-strong); text-align: right; }

/* --- welcome / onboarding: example question buttons --- */
[class*="st-key-example_"] button {
    text-align: left !important;
    height: 104px !important;
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
.welcome-caption { color: var(--ink-faint); font-size: 0.74rem; margin: 0.3rem 0 1.1rem 0; }

/* --- sidebar: instrument-panel / ship's-log framing --- */
[data-testid="stSidebar"] { border-right: 1px solid var(--border-accent); background: var(--surface); }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p { color: var(--ink); }
.eyebrow {
    font-family: var(--font-mono); font-size: 0.68rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--ink-faint); margin: 0.2rem 0 0.5rem 0;
    border-bottom: 1px dashed var(--border); padding-bottom: 0.3rem;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono); font-size: 1.2rem; color: var(--accent-strong);
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.66rem; color: var(--ink-muted); }

.provider-list { display: flex; flex-direction: column; gap: 0.55rem; }
.provider-row { padding: 0.1rem 0; }
.provider-row .provider-name-line {
    font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink);
    display: flex; align-items: center; gap: 0.55rem;
}
.provider-row .provider-role {
    font-family: var(--font-mono); font-size: 0.64rem; color: var(--ink-faint);
    text-transform: uppercase; letter-spacing: 0.03em; line-height: 1.4;
    margin-top: 0.15rem; padding-left: 1.15rem;
}
.status-dot-inline { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.status-dot-inline.up { background: var(--accent-strong); box-shadow: 0 0 4px var(--accent-soft); }
.status-dot-inline.down { background: var(--danger); }

.error-note {
    font-family: var(--font-mono); font-size: 0.76rem; color: var(--danger);
    background: var(--danger-soft); border: 1px solid rgba(156, 74, 54, 0.22);
    border-radius: 3px; padding: 0.5rem 0.7rem; margin-top: 0.5rem;
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
    {"label": "Groq", "role": "generation \u00b7 acct A", "check": lambda: bool(settings.groq_api_key)},
    {
        "label": "Groq",
        "role": "generation \u00b7 acct B",
        "check": lambda: bool(settings.groq_api_key_secondary),
    },
    {"label": "NIM", "role": "generation fallback \u00b7 planner \u00b7 NeMoGuard", "check": lambda: bool(settings.nvidia_nim_api_key)},
    {"label": "Gemini", "role": "embeddings \u00b7 eval judge", "check": lambda: bool(settings.gemini_api_key)},
    {
        "label": "Qdrant",
        "role": "vector store",
        "check": lambda: bool(settings.qdrant_url and settings.qdrant_api_key),
    },
]

PIPELINE_STAGES = [
    {"label": "Safety", "desc": "blocks unsafe content"},
    {"label": "Topic", "desc": "blocks off-topic questions"},
    {"label": "Cache", "desc": "exact + semantic lookup"},
    {"label": "Retrieve", "desc": "dense vector search"},
    {"label": "Rerank", "desc": "cross-encoder relevance gate"},
    {"label": "Generate", "desc": "grounded answer"},
]

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

# ---------- header ----------

st.markdown(
    """
    <div class="app-header">
        <div class="masthead-eyebrow">Cluster documentation &middot; advisory service</div>
        <div class="title-block">
            <h1>Kubernetes Q&amp;A</h1>
            <div class="tagline">Grounded Kubernetes answers from your own docs, safety-checked,
            relevance-gated, and cached for speed.</div>
        </div>
        <hr class="masthead-rule" />
        <hr class="masthead-rule dashed" />
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

# checked against `prompt` too, not just history: history is only empty
# BEFORE this run's user message gets appended further down, so on the
# very first-ever submission this condition would otherwise still be true
# during the same run that's processing that submission — showing the
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