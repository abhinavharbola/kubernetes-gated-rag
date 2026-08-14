import logging

from flashrank import Ranker, RerankRequest

from src.config import settings

logger = logging.getLogger(__name__)

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    # lazily built on first use, not at import time: the ONNX model is a
    # real ~30-50MB load, paying that cost during `import src.retrieval.rerank` (e.g.
    # via `import src.graph` at app startup) delays every cold start by
    # however long that load takes, for a model that a given process might
    # not even need yet (cache hits never reach this code path at all).
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=settings.rerank_model)
    return _ranker


def preload() -> None:
    """Force the FlashRank ONNX model to load now, rather than lazily on
    the first real rerank_and_gate() call. See guardrails/gates.py's
    preload() for why this matters: paid once at UI startup instead of
    silently inflating the first user question's latency."""
    _get_ranker()


def rerank_and_gate(question: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    passages = [
        {"id": i, "text": candidate["text"], "meta": candidate["metadata"]}
        for i, candidate in enumerate(candidates)
    ]
    request = RerankRequest(query=question, passages=passages)

    try:
        reranked = _get_ranker().rerank(request)
    except Exception as error:
        # zero-downtime fallback: if the local ONNX model fails to load or
        # errors mid-inference, degrade to the Qdrant retrieval order rather
        # than failing the whole turn. This intentionally skips the hard
        # relevance threshold below, since there's no rerank score left to
        # gate on, so noise-filtering is temporarily weaker during a
        # FlashRank outage, not absent generation entirely.
        logger.error("FlashRank failed, falling back to retrieval order: %s", error)
        return sorted(candidates, key=lambda c: c.get("retrieval_score", 0.0), reverse=True)

    scored = []
    for result in reranked:
        candidate = candidates[result["id"]]
        scored.append({**candidate, "rerank_score": result["score"]})

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    survivors = [c for c in scored if c["rerank_score"] >= settings.rerank_score_threshold]
    return survivors
