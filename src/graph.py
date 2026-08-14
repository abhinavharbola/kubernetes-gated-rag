import logging
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.retrieval.cache import (
    embed_canonical_question,
    exact_cache_get,
    exact_cache_set,
    semantic_cache_get,
    semantic_cache_set,
)
from src.config import settings
from src.guardrails import safety_gate, topic_gate
from src.providers.llm import generate_main, generate_planner
from src.retrieval.rerank import rerank_and_gate
from src.retrieval.search import retrieve
from src.tracing import node_span, log_cache_decision, turn_span

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = (
    "I don't have grounded documentation for that. Try rephrasing, or ask about "
    "a specific resource, provider, or module."
)


class GraphState(TypedDict, total=False):
    raw_message: str
    chat_history: list[dict]
    standalone_question: str
    canonical_question: str
    canonical_question_vector: list[float] | None
    allowed: bool
    refusal_reason: str | None
    blocked_stage: str | None
    answer: str
    provider: str | None
    model: str | None
    cache_layer: str | None
    candidates: list[dict]
    reranked: list[dict]
    latency_seconds: float | None


def safety_gate_node(state: GraphState) -> GraphState:
    # runs on the raw message, before the rewrite step, so a jailbreak attempt
    # is rejected before it costs a planner call.
    with node_span("safety_gate"):
        allowed, reason = safety_gate(state["raw_message"])
        return {"allowed": allowed, "refusal_reason": reason, "blocked_stage": None if allowed else "safety"}


def rewrite_with_history_node(state: GraphState) -> GraphState:
    with node_span("rewrite_with_history"):
        if not state.get("chat_history"):
            return {"standalone_question": state["raw_message"]}
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in state["chat_history"])
        result = generate_planner(
            [
                {
                    "role": "system",
                    "content": "Rewrite the user's latest message as a standalone question. "
                    "Only use chat history to resolve pronouns, ellipsis, or implicit references "
                    "the latest message depends on. If the latest message is already self-contained, "
                    "return it unchanged or with only minor grammatical cleanup: do not broaden it with "
                    "topics, details, or scope from earlier turns that this message didn't ask about. "
                    "If the latest message introduces a subject unrelated to the conversation so far, "
                    "treat it as self-contained and do not merge it with earlier context to make it read "
                    "as more related than it is, the downstream topic check needs to see the question as "
                    "actually asked, not a version reframed to look more on-topic. Preserve intent "
                    "exactly. Return only the rewritten question.",
                },
                {"role": "user", "content": f"History:\n{history_text}\n\nLatest: {state['raw_message']}"},
            ]
        )
        standalone = result.content.strip()
        if not standalone:
            logger.warning(
                "rewrite_with_history got an empty response from %s (%s), falling back to raw_message",
                result.provider,
                result.model,
            )
            standalone = state["raw_message"]
        return {"standalone_question": standalone}


def topic_gate_node(state: GraphState) -> GraphState:
    # runs on the rewritten standalone question, so context-dependent follow-ups
    # aren't misjudged as off-topic.
    with node_span("topic_gate"):
        allowed, reason = topic_gate(state["standalone_question"])
        return {"allowed": allowed, "refusal_reason": reason, "blocked_stage": None if allowed else "topic"}


def exact_cache_node(state: GraphState) -> GraphState:
    with node_span("exact_cache_check"):
        cached = exact_cache_get(state["standalone_question"])
        log_cache_decision("exact", hit=cached is not None)
        if cached:
            return {"answer": cached, "cache_layer": "exact"}
        return {}


def canonicalize_node(state: GraphState) -> GraphState:
    with node_span("canonicalize_question"):
        result = generate_planner(
            [
                {
                    "role": "system",
                    "content": "Normalize the phrasing of this Kubernetes question into a consistent "
                    "canonical form, so that different phrasings asking the same thing (e.g. 'what is X', "
                    "'tell me about X', 'explain X') produce the same canonical question. Preserve the "
                    "specific topic and scope exactly as asked, do not add or remove details, and do not "
                    "collapse distinct actions (create/destroy/update/read) into each other. Return only "
                    "the normalized question.",
                },
                {"role": "user", "content": state["standalone_question"]},
            ]
        )
        canonical = result.content.strip()
        if not canonical:
            # small/fast planner models occasionally return an empty
            # completion for an otherwise well-formed request. There's
            # nothing to normalize in that case — fall back to the
            # standalone question rather than letting an empty string
            # reach embed_canonical_question, which Gemini's embed_content
            # rejects outright with an opaque 400 (empty Part) error deep
            # in a retry stack, not this call site.
            logger.warning(
                "canonicalize_question got an empty response from %s (%s), falling back to standalone_question",
                result.provider,
                result.model,
            )
            canonical = state["standalone_question"]
        return {"canonical_question": canonical}


def semantic_cache_node(state: GraphState) -> GraphState:
    with node_span("semantic_cache_check"):
        # computed once here and carried in state; write_caches_node reuses
        # this same vector on a miss instead of re-embedding the identical
        # canonical_question text a second time.
        vector = embed_canonical_question(state["canonical_question"])
        cached = semantic_cache_get(vector)
        log_cache_decision("semantic", hit=cached is not None)
        if cached:
            return {"answer": cached, "cache_layer": "semantic", "canonical_question_vector": vector}
        return {"canonical_question_vector": vector}


def retrieve_node(state: GraphState) -> GraphState:
    with node_span("retrieve"):
        candidates = retrieve(state["canonical_question"])
        return {"candidates": candidates}


def rerank_node(state: GraphState) -> GraphState:
    with node_span("rerank_and_gate"):
        survivors = rerank_and_gate(state["canonical_question"], state.get("candidates", []))
        return {"reranked": survivors}


def generate_node(state: GraphState) -> GraphState:
    with node_span("generate"):
        context = "\n\n---\n\n".join(c["text"] for c in state["reranked"])
        result = generate_main(
            [
                {
                    "role": "system",
                    "content": "Answer the Kubernetes question using ONLY the provided context. "
                    "If the context doesn't fully cover the question, say what's missing "
                    "rather than guessing.",
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['standalone_question']}"},
            ]
        )
        return {"answer": result.content, "provider": result.provider, "model": result.model}


def write_caches_node(state: GraphState) -> GraphState:
    with node_span("write_caches"):
        exact_cache_set(state["standalone_question"], state["answer"])
        # reuses the vector computed in semantic_cache_node rather than
        # calling embed_canonical_question again for the same text.
        semantic_cache_set(state["canonical_question"], state["canonical_question_vector"], state["answer"])
        return {}


def cache_no_context_node(state: GraphState) -> GraphState:
    # zero reranked survivors means there's no grounded documentation for
    # this question right now. That's cached too (exact-match only, with a
    # TTL rather than forever) so a repeated identical question doesn't
    # re-pay embedding + retrieval + rerank for an answer that won't change
    # until the corpus does, while still expiring if the corpus is updated.
    with node_span("cache_no_context"):
        exact_cache_set(
            state["standalone_question"],
            NO_CONTEXT_MESSAGE,
            expire=settings.no_context_cache_ttl_seconds,
        )
        return {"answer": NO_CONTEXT_MESSAGE}


def route_after_safety_gate(state: GraphState) -> str:
    return "rewrite_with_history" if state["allowed"] else END


def route_after_topic_gate(state: GraphState) -> str:
    return "exact_cache_check" if state["allowed"] else END


def route_after_exact_cache(state: GraphState) -> str:
    return END if state.get("cache_layer") == "exact" else "canonicalize_question"


def route_after_semantic_cache(state: GraphState) -> str:
    return END if state.get("cache_layer") == "semantic" else "retrieve"


def route_after_rerank(state: GraphState) -> str:
    return "generate" if state["reranked"] else "cache_no_context"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("safety_gate", safety_gate_node)
    graph.add_node("rewrite_with_history", rewrite_with_history_node)
    graph.add_node("topic_gate", topic_gate_node)
    graph.add_node("exact_cache_check", exact_cache_node)
    graph.add_node("canonicalize_question", canonicalize_node)
    graph.add_node("semantic_cache_check", semantic_cache_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank_and_gate", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("write_caches", write_caches_node)
    graph.add_node("cache_no_context", cache_no_context_node)

    graph.add_edge(START, "safety_gate")
    graph.add_conditional_edges("safety_gate", route_after_safety_gate)
    graph.add_edge("rewrite_with_history", "topic_gate")
    graph.add_conditional_edges("topic_gate", route_after_topic_gate)
    graph.add_conditional_edges("exact_cache_check", route_after_exact_cache)
    graph.add_edge("canonicalize_question", "semantic_cache_check")
    graph.add_conditional_edges("semantic_cache_check", route_after_semantic_cache)
    graph.add_conditional_edges("rerank_and_gate", route_after_rerank)
    graph.add_edge("retrieve", "rerank_and_gate")
    graph.add_edge("generate", "write_caches")
    graph.add_edge("write_caches", END)
    graph.add_edge("cache_no_context", END)

    return graph.compile()


# compiled once at import time, safe to reuse across every Streamlit rerun.
compiled_graph = build_graph()


def run_turn(raw_message: str, chat_history: list[dict]) -> GraphState:
    start = time.perf_counter()
    with turn_span(raw_message):
        initial_state: GraphState = {"raw_message": raw_message, "chat_history": chat_history}
        final_state = compiled_graph.invoke(initial_state)
        if not final_state.get("allowed", True):
            final_state["answer"] = final_state["refusal_reason"]
    final_state["latency_seconds"] = time.perf_counter() - start
    return final_state