import hashlib
import logging
import pathlib
import re
import threading
import uuid

import diskcache
from qdrant_client.models import FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct

from src.providers.clients import qdrant_client
from src.config import settings
from src.retrieval.embeddings import embed_for_cache

logger = logging.getLogger(__name__)

# Anchored to the repo root (three levels up from this file:
# src/retrieval/cache.py -> src/retrieval -> src -> repo root), not to the
# process's current working directory. Previously ".cache/exact" and
# ".cache/corpus_version" were relative paths, so ingest.py and
# `streamlit run ui/app.py` only agreed on where the cache lived if both
# happened to be launched from the same directory. Launch either from a
# different cwd (e.g. `cd ui && streamlit run app.py`) and they'd silently
# read/write two different .cache trees — no error, just a semantic cache
# that never sees the real corpus fingerprint and falls back to the static
# CORPUS_VERSION forever. src/retrieval/embeddings.py and ingest.py anchor
# the same way for the same reason.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CACHE_DIR = _PROJECT_ROOT / ".cache"

_exact_cache = diskcache.Cache(str(_CACHE_DIR / "exact"))

# Punctuation that's purely cosmetic in a typed question (trailing "?",
# stray quotes, parentheses) is safe to strip for exact-match normalization.
# Characters that carry real meaning in Kubernetes syntax -- "-" in
# resource/flag names, "/" in API groups like apps/v1, "." in dotted paths
# like pod.spec.containers, ":" in label/selector syntax -- are deliberately
# NOT stripped. Stripping them previously collapsed distinct questions
# ("apps/v1" vs "apps v1", "pod.spec.containers" vs "pod spec containers")
# onto the same cache key, returning a cached answer for a different
# question than the one actually asked.
_STRIP_CHARS = "?!,;'\"()[]{}"
_PUNCT_TABLE = str.maketrans("", "", _STRIP_CHARS)

# Qdrant requires an explicit payload index before a field can be used in a
# query filter ("Index required but not found") — filtering on these three
# version fields in _semantic_filter() below 400s without it. Created
# lazily and idempotently here (rather than only in ingest.py's
# ensure_collection) so an already-existing cache collection self-heals the
# first time this process touches it, no --wipe or re-ingest required.
_VERSION_FIELDS = ("cache_schema_version", "policy_version", "corpus_version")
_indexes_ensured = False
_index_lock = threading.Lock()

# Written by ingest.py after a successful ingestion run, containing a hash
# of the actual ingested content. Cache keys read this (falling back to the
# static settings.corpus_version when it doesn't exist yet, e.g. before the
# first ingest) instead of settings.corpus_version directly, so an
# incremental re-ingest that changes the corpus -- without a --wipe or a
# manually-edited .env -- still changes the effective corpus_version and
# invalidates stale cache entries automatically. See ingest.py's
# _write_corpus_version.
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
    global _indexes_ensured
    if _indexes_ensured:
        return
    with _index_lock:
        if _indexes_ensured:
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
                # A genuine failure (collection doesn't exist yet, a
                # connectivity blip during startup, etc.) must NOT latch
                # _indexes_ensured to True: that would permanently skip
                # index creation for the rest of this process's life, even
                # after the collection is created later (e.g. by a
                # subsequent ingest.py run), silently degrading the
                # semantic cache to an always-miss for the remainder of the
                # process. Leaving the flag False means the next call
                # retries.
                all_succeeded = False
                logger.warning(
                    "payload index ensure for %s failed, will retry on next call: %s", field, error
                )
        if all_succeeded:
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
        f"{_current_corpus_version()}:{digest}"
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
            FieldCondition(key="corpus_version", match=MatchValue(value=_current_corpus_version())),
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
                    "corpus_version": _current_corpus_version(),
                },
            )
        ],
    )
