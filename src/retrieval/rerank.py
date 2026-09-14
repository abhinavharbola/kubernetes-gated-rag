import logging

from flashrank import Ranker, RerankRequest

from src.config import settings

logger = logging.getLogger(__name__)
_ranker: Ranker | None = None


class RerankUnavailableError(Exception):
    """Raised when FlashRank itself failed to run (model load, ONNX runtime
    error, etc.) and settings.rerank_fail_closed is True. This is distinct
    from "the ranker ran fine and every candidate scored below the
    threshold" — that's a real, cacheable "no grounded documentation"
    outcome. A ranker crash is an infrastructure problem, and graph.py must
    not cache it as if it were a genuine relevance verdict."""


def _get_ranker() -> Ranker:
    global _ranker
    if _ranker is None:
        _ranker = Ranker(model_name=settings.rerank_model)
    return _ranker


def preload() -> None:
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
        logger.error("FlashRank failed: %s", error)
        if settings.rerank_fail_closed:
            raise RerankUnavailableError(str(error)) from error
        # Bug fix: this previously returned every candidate sorted by raw
        # retrieval_score with no threshold applied at all, which silently
        # disabled the hard relevance gate entirely whenever FlashRank
        # crashed and rerank_fail_closed was set to False - defeating the
        # one thing this function exists to enforce. rerank_score_threshold
        # itself isn't meaningful here since no rerank score was produced,
        # so a separate, retrieval-score-based threshold is applied instead
        # to keep this degrade mode a real (if cruder) gate rather than no
        # gate at all.
        return [
            c
            for c in sorted(candidates, key=lambda c: c.get("retrieval_score", 0.0), reverse=True)
            if c.get("retrieval_score", 0.0) >= settings.rerank_fallback_score_threshold
        ]

    scored = []
    for result in reranked:
        candidate = candidates[result["id"]]
        scored.append({**candidate, "rerank_score": result["score"]})
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return [c for c in scored if c["rerank_score"] >= settings.rerank_score_threshold]
