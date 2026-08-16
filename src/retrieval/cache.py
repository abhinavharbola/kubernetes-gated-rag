import re
import string
import uuid

import diskcache
from qdrant_client.models import PointStruct

from src.providers.clients import qdrant_client
from src.config import settings
from src.retrieval.embeddings import embed_for_cache

_exact_cache = diskcache.Cache(".cache/exact")

# hyphen deliberately excluded: Kubernetes identifiers are full of hyphens
# ("kube-system", "front-end", "-n" flags) and stripping it merges genuinely
# different terms into the same normalized string ("kube-system" and
# "kubesystem" would otherwise collapse to one cache key). Every other
# punctuation character is still stripped.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("-", ""))


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


def embed_canonical_question(canonical_question: str) -> list[float]:
    # split out as its own function, callable once per turn: the canonical
    # question's embedding is needed by both the semantic-cache lookup and,
    # on a miss, the later semantic-cache write. Computing it once and
    # threading the vector through graph state (see src/graph.py) instead of
    # calling this twice avoids paying for a second identical Gemini
    # embedding round-trip on every non-cache-hit turn.
    return embed_for_cache(canonical_question)


def semantic_cache_get(canonical_question_vector: list[float]) -> str | None:
    results = qdrant_client.query_points(
        collection_name=settings.qdrant_cache_collection,
        query=canonical_question_vector,
        limit=1,
        score_threshold=settings.semantic_cache_similarity_threshold,
    ).points
    if not results:
        return None
    return results[0].payload.get("answer")


def semantic_cache_set(canonical_question: str, canonical_question_vector: list[float], answer: str) -> None:
    qdrant_client.upsert(
        collection_name=settings.qdrant_cache_collection,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=canonical_question_vector,
                payload={"question": canonical_question, "answer": answer},
            )
        ],
    )