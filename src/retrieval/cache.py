import hashlib
import logging
import re
import string
import threading
import uuid

import diskcache
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct

from src.providers.clients import qdrant_client
from src.config import settings
from src.retrieval.embeddings import embed_for_cache

logger = logging.getLogger(__name__)

_exact_cache = diskcache.Cache(".cache/exact")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("-", ""))

# Qdrant requires an explicit payload index before a field can be used in a
# query filter ("Index required but not found") — filtering on these three
# version fields in _semantic_filter() below 400s without it. Created
# lazily and idempotently here (rather than only in ingest.py's
# ensure_collection) so an already-existing cache collection self-heals the
# first time this process touches it, no --wipe or re-ingest required.
_VERSION_FIELDS = ("cache_schema_version", "policy_version", "corpus_version")
_indexes_ensured = False
_index_lock = threading.Lock()


def ensure_semantic_cache_indexes() -> None:
    global _indexes_ensured
    if _indexes_ensured:
        return
    with _index_lock:
        if _indexes_ensured:
            return
        for field in _VERSION_FIELDS:
            try:
                qdrant_client.create_payload_index(
                    collection_name=settings.qdrant_cache_collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as error:
                # Already exists, or the collection doesn't exist yet: both
                # fine to swallow here. A genuine connectivity problem
                # surfaces again on the very next real Qdrant call anyway.
                logger.debug("payload index ensure for %s: %s", field, error)
        _indexes_ensured = True


def normalize_exact(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip().lower())
    return collapsed.translate(_PUNCT_TABLE)


def normalize_semantic(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip())


def _exact_key(question: str) -> str:
    digest = hashlib.sha256(normalize_exact(question).encode("utf-8")).hexdigest()
    return (
        f"{settings.cache_schema_version}:exact:{settings.cache_policy_version}:"
        f"{settings.corpus_version}:{digest}"
    )


def exact_cache_get(question: str) -> str | None:
    value = _exact_cache.get(_exact_key(question))
    return value if isinstance(value, str) else None


def exact_cache_set(question: str, answer: str, expire: float | None = None) -> None:
    _exact_cache.set(_exact_key(question), answer, expire=expire)


def embed_canonical_question(canonical_question: str) -> list[float]:
    # task_type=SEMANTIC_SIMILARITY: a different vector space from the
    # RETRIEVAL_QUERY embedding src/retrieval/search.py computes for the
    # same question text on a cache miss. Not interchangeable with it, see
    # the comment in search.py's retrieve() for why.
    return embed_for_cache(canonical_question)


def _semantic_filter() -> Filter:
    return Filter(
        must=[
            FieldCondition(key="cache_schema_version", match=MatchValue(value=settings.cache_schema_version)),
            FieldCondition(key="policy_version", match=MatchValue(value=settings.cache_policy_version)),
            FieldCondition(key="corpus_version", match=MatchValue(value=settings.corpus_version)),
        ]
    )


def semantic_cache_get(canonical_question_vector: list[float]) -> str | None:
    ensure_semantic_cache_indexes()
    # A lookup failure (timeout, transient Qdrant error) degrades to a
    # cache miss rather than crashing the turn — the graph falls through
    # to retrieval either way, same as a genuine miss. See the matching
    # comment in src/retrieval/search.py's retrieve().
    try:
        results = qdrant_client.query_points(
            collection_name=settings.qdrant_cache_collection,
            query=canonical_question_vector,
            limit=1,
            score_threshold=settings.semantic_cache_similarity_threshold,
            query_filter=_semantic_filter(),
        ).points
    except Exception as error:
        logger.error("semantic cache lookup failed, treating as a miss: %s", error)
        return None
    if not results:
        return None
    answer = results[0].payload.get("answer")
    return answer if isinstance(answer, str) else None


def semantic_cache_set(canonical_question: str, canonical_question_vector: list[float], answer: str) -> None:
    ensure_semantic_cache_indexes()
    qdrant_client.upsert(
        collection_name=settings.qdrant_cache_collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=canonical_question_vector,
                payload={
                    "question": canonical_question,
                    "answer": answer,
                    "cache_schema_version": settings.cache_schema_version,
                    "policy_version": settings.cache_policy_version,
                    "corpus_version": settings.corpus_version,
                },
            )
        ],
    )
