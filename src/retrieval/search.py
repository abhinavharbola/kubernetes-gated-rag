import logging

from src.config import settings
from src.providers.clients import qdrant_client
from src.retrieval.embeddings import embed_query

logger = logging.getLogger(__name__)


class RetrievalUnavailableError(Exception):
    pass


def retrieve(question: str) -> list[dict]:
    try:
        vector = embed_query(question)
        results = qdrant_client.query_points(
            collection_name=settings.qdrant_docs_collection,
            query=vector,
            limit=settings.rerank_top_k,
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
