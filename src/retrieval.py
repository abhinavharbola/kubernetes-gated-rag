from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.clients import qdrant_client
from src.config import settings
from src.embeddings import embed_query


def retrieve(question: str, manifest_kind: str | None = None, top_k: int | None = None) -> list[dict]:
    vector = embed_query(question)
    query_filter = None
    if manifest_kind:
        query_filter = Filter(
            must=[FieldCondition(key="manifest_kind", match=MatchValue(value=manifest_kind))]
        )

    results = qdrant_client.query_points(
        collection_name=settings.qdrant_docs_collection,
        query=vector,
        query_filter=query_filter,
        limit=top_k or settings.rerank_top_k,
    ).points

    return [
        {
            "text": point.payload["text"],
            "metadata": point.payload.get("metadata", {}),
            "retrieval_score": point.score,
        }
        for point in results
    ]