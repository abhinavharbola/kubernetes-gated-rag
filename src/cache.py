import re
import string
import uuid

import diskcache
from qdrant_client.models import PointStruct

from src.clients import qdrant_client
from src.config import settings
from src.embeddings import embed_for_cache

_exact_cache = diskcache.Cache(".cache/exact")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_exact(question: str) -> str:
    # whitespace/case/punctuation only, nothing more aggressive. must not
    # conflate two genuinely different questions.
    collapsed = re.sub(r"\s+", " ", question.strip().lower())
    return collapsed.translate(_PUNCT_TABLE)


def exact_cache_get(question: str) -> str | None:
    return _exact_cache.get(normalize_exact(question))


def exact_cache_set(question: str, answer: str, expire: float | None = None) -> None:
    # expire=None (default) never expires, used for real generated answers.
    # A finite expire is used for cached no-context outcomes, so a stale
    # "no docs for this" verdict doesn't outlive a later re-ingest.
    _exact_cache.set(normalize_exact(question), answer, expire=expire)


def semantic_cache_get(canonical_question: str) -> str | None:
    vector = embed_for_cache(canonical_question)
    results = qdrant_client.query_points(
        collection_name=settings.qdrant_cache_collection,
        query=vector,
        limit=1,
        score_threshold=settings.semantic_cache_similarity_threshold,
    ).points
    if not results:
        return None
    return results[0].payload.get("answer")


def semantic_cache_set(canonical_question: str, answer: str) -> None:
    vector = embed_for_cache(canonical_question)
    qdrant_client.upsert(
        collection_name=settings.qdrant_cache_collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"question": canonical_question, "answer": answer},
            )
        ],
    )