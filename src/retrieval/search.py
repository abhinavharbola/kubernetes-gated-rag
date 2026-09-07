import logging

from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.providers.clients import qdrant_client
from src.config import settings
from src.retrieval.embeddings import embed_query

logger = logging.getLogger(__name__)


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
    # transient error here previously propagated all the way up through
    # compiled_graph.invoke() and killed the whole turn with a raw
    # traceback. Treating a failure here the same as "found nothing" keeps
    # it consistent with rerank_and_gate's existing fail-closed behavior:
    # it flows into the same cache_no_context path (a normal, honest
    # "no grounded documentation" answer, TTL-cached, not a crash) instead
    # of a 500 the user can't do anything about.
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
        logger.error("retrieve failed, returning no candidates: %s", error)
        return []

    return [
        {
            "text": point.payload["text"],
            "metadata": point.payload.get("metadata", {}),
            "retrieval_score": point.score,
        }
        for point in results
    ]
