import hashlib
import logging
import re
import threading
import time
import uuid

import diskcache
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct

from src.config import cache_root, settings
from src.providers.clients import qdrant_client
from src.retrieval.embeddings import embed_for_cache

logger = logging.getLogger(__name__)

_CACHE_DIR = cache_root()

_exact_cache = diskcache.Cache(str(_CACHE_DIR / "exact"))

_VERSION_FIELDS = ("cache_schema_version", "policy_version", "corpus_version")
_INDEX_RETRY_COOLDOWN_SECONDS = 30.0
_indexes_ensured = False
_next_index_attempt = 0.0
_index_lock = threading.Lock()

_CORPUS_VERSION_MARKER = _CACHE_DIR / "corpus_version"


def _current_corpus_version() -> str:
    try:
        value = _CORPUS_VERSION_MARKER.read_text().strip()
    except OSError:
        return settings.corpus_version
    return value or settings.corpus_version


def _is_already_exists_error(error: Exception) -> bool:
    return "already exist" in str(error).lower()


def ensure_semantic_cache_indexes() -> None:
    global _indexes_ensured, _next_index_attempt
    if _indexes_ensured or time.monotonic() < _next_index_attempt:
        return
    with _index_lock:
        if _indexes_ensured or time.monotonic() < _next_index_attempt:
            return
        all_succeeded = True
        for field in _VERSION_FIELDS:
            try:
                qdrant_client.create_payload_index(
                    collection_name=settings.qdrant_cache_collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as error:
                if _is_already_exists_error(error):
                    logger.debug("payload index for %s already exists", field)
                    continue
                all_succeeded = False
                logger.warning("payload index ensure for %s failed, will retry later: %s", field, error)
                break
        if all_succeeded:
            _indexes_ensured = True
        else:
            _next_index_attempt = time.monotonic() + _INDEX_RETRY_COOLDOWN_SECONDS


def normalize_exact(question: str) -> str:
    collapsed = re.sub(r"\s+", " ", question.strip())
    return collapsed.rstrip("?!. ")


def normalize_semantic(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip())


def _exact_key(question: str) -> str:
    digest = hashlib.sha256(normalize_exact(question).encode("utf-8")).hexdigest()
    return (
        f"{settings.cache_schema_version}:exact:{settings.cache_policy_version}:"
        f"{_current_corpus_version()}:{digest}"
    )


def exact_cache_get(question: str) -> str | None:
    try:
        value = _exact_cache.get(_exact_key(question))
    except Exception as error:
        logger.error("exact cache lookup failed, treating as a miss: %s", error)
        return None
    return value if isinstance(value, str) else None


def exact_cache_set(question: str, answer: str, expire: float | None = None) -> None:
    _exact_cache.set(_exact_key(question), answer, expire=expire)


def exact_cache_count() -> int:
    return len(_exact_cache)


def exact_cache_clear() -> int:
    count = len(_exact_cache)
    _exact_cache.clear()
    return count


def embed_canonical_question(canonical_question: str) -> list[float]:
    return embed_for_cache(canonical_question)


def _semantic_filter() -> Filter:
    return Filter(
        must=[
            FieldCondition(key="cache_schema_version", match=MatchValue(value=settings.cache_schema_version)),
            FieldCondition(key="policy_version", match=MatchValue(value=settings.cache_policy_version)),
            FieldCondition(key="corpus_version", match=MatchValue(value=_current_corpus_version())),
        ]
    )


def semantic_cache_get(canonical_question_vector: list[float]) -> str | None:
    ensure_semantic_cache_indexes()
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
                    "corpus_version": _current_corpus_version(),
                },
            )
        ],
    )
