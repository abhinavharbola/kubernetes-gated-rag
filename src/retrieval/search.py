import logging

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.providers.clients import qdrant_client
from src.config import settings
from src.retrieval.embeddings import embed_query

logger = logging.getLogger(__name__)


class RetrievalUnavailableError(Exception):
    """Raised when retrieval could not be attempted or completed because of
    an infrastructure problem (embedding call, Qdrant call), as opposed to a
    retrieval that ran fine and legitimately found nothing. Callers must not
    treat this the same as a real empty result: an empty result is safe to
    cache as "no grounded documentation"; an unavailable retrieval service
    is not, since caching it would turn a transient outage into an
    extended, wrong answer for that question."""


def retrieve(
    question: str,
    manifest_kind: str | None = None,
    top_k: int | None = None,
) -> list[dict]:
    # Always embeds with task_type=RETRIEVAL_QUERY here, never accepts a
    # precomputed vector from the caller. The semantic-cache vector
    # (task_type=SEMANTIC_SIMILARITY, see src/retrieval/cache.py) looks like
    # the same computation since it's over the same question text, but
    # Gemini produces different vector spaces per task_type: RETRIEVAL_QUERY
    # vectors are trained to align with this collection's RETRIEVAL_DOCUMENT
    # vectors, SEMANTIC_SIMILARITY vectors are trained for symmetric
    # question-to-question comparison against the cache collection. Passing
    # the cache's vector in here would silently degrade retrieval relevance,
    # not save a real duplicate call. The persistent embedding cache in
    # embeddings.py (keyed by task_type) is what actually avoids repeat
    # Gemini calls for repeated questions, per task_type.
    #
    # Both the embedding call and the Qdrant query below are network calls
    # on the synchronous request path, unlike cache writes (which run in
    # graph.py's background executor and can't crash a turn). A timeout or
    # transient error here raises RetrievalUnavailableError rather than
    # returning an empty list: an empty list here would be indistinguishable
    # from "the search ran and genuinely found nothing", and graph.py caches
    # that outcome as an honest "no grounded documentation" answer with a
    # TTL. Silently conflating the two would let a transient Qdrant/Gemini
    # blip get baked into the cache as a wrong "no context" answer for up to
    # that TTL, well past when the outage actually ended. graph.py's
    # retrieve_node catches this and routes to a distinct, uncached
    # "temporarily unavailable" response instead.
    try:
        vector = embed_query(question)
        query_filter = None
        if manifest_kind:
            query_filter = Filter(
                must=[FieldCondition(key="metadata.manifest_kind", match=MatchValue(value=manifest_kind))]
            )
        results = qdrant_client.query_points(
            collection_name=settings.qdrant_docs_collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k or settings.rerank_top_k,
        ).points
    except Exception as error:
        logger.error("retrieve failed: %s", error)
        raise RetrievalUnavailableError(str(error)) from error

    return [
        {
            "text": point.payload["text"],
            "metadata": point.payload.get("metadata", {}),
            "retrieval_score": point.score,
        }
        for point in results
    ]



