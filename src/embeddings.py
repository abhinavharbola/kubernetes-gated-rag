import numpy as np
from google.genai import types

from src.clients import gemini_client
from src.config import settings

# Gemini's embed_content endpoint accepts a batch of texts in a single
# request, not just one. The previous implementation called it once per
# text, so ingesting N chunks cost N network round-trips and N slices of
# a free-tier RPM budget instead of ceil(N / EMBED_BATCH_SIZE). 100 is
# comfortably under Gemini's per-request batch limit; lower it if that
# limit changes.
EMBED_BATCH_SIZE = 100


def _normalize(vector: list[float]) -> list[float]:
    array = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return vector
    return (array / norm).tolist()


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """task_type: RETRIEVAL_DOCUMENT for ingestion, RETRIEVAL_QUERY for queries,
    SEMANTIC_SIMILARITY for cache lookups."""
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = gemini_client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dim,
            ),
        )
        vectors.extend(_normalize(embedding.values) for embedding in response.embeddings)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def embed_document(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]


def embed_for_cache(text: str) -> list[float]:
    return embed_texts([text], task_type="SEMANTIC_SIMILARITY")[0]
