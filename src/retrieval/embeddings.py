import hashlib
import logging
import re

import diskcache
import numpy as np
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt

from src.providers.clients import gemini_client
from src.config import settings

logger = logging.getLogger(__name__)
EMBED_BATCH_SIZE = 100
RETRY_DELAY_RE = re.compile(r"([\d.]+)\s*s")
_embedding_cache = diskcache.Cache(".cache/embeddings")


def _is_rate_limit_error(error: BaseException) -> bool:
    return isinstance(error, ClientError) and getattr(error, "code", None) == 429


def _extract_retry_delay_seconds(error: ClientError, default: float = 30.0) -> float:
    try:
        details = error.details.get("error", {}).get("details", [])
        for detail in details:
            if str(detail.get("@type", "")).endswith("RetryInfo"):
                match = RETRY_DELAY_RE.match(str(detail.get("retryDelay", "")))
                if match:
                    return float(match.group(1))
    except Exception:
        pass
    return default


def _rate_limit_wait(retry_state) -> float:
    error = retry_state.outcome.exception()
    delay = _extract_retry_delay_seconds(error)
    logger.warning("Gemini embed_content rate limited, waiting %.0fs before retry", delay)
    return delay + 1.0


@retry(
    stop=stop_after_attempt(4),
    wait=_rate_limit_wait,
    retry=retry_if_exception(_is_rate_limit_error),
    reraise=True,
)
def _embed_batch(batch: list[str], task_type: str) -> list[list[float]]:
    response = gemini_client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=batch,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.embedding_dim,
        ),
    )
    return [_normalize(embedding.values) for embedding in response.embeddings]


def _normalize(vector: list[float]) -> list[float]:
    array = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return vector
    return (array / norm).tolist()


def _cache_key(text: str, task_type: str) -> str:
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    return (
        f"{settings.cache_schema_version}:embedding:{settings.gemini_embedding_model}:"
        f"{settings.embedding_dim}:{task_type}:{digest}"
    )


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    if not texts:
        return []
    for i, text in enumerate(texts):
        if not text or not text.strip():
            raise ValueError(f"embed_texts received an empty string at index {i} (task_type={task_type})")

    vectors: list[list[float] | None] = [None] * len(texts)
    misses: list[tuple[int, str]] = []
    for i, text in enumerate(texts):
        cached = _embedding_cache.get(_cache_key(text, task_type))
        if cached is None:
            misses.append((i, text))
        else:
            vectors[i] = cached

    for start in range(0, len(misses), EMBED_BATCH_SIZE):
        batch_pairs = misses[start : start + EMBED_BATCH_SIZE]
        batch_vectors = _embed_batch([text for _, text in batch_pairs], task_type)
        if len(batch_vectors) != len(batch_pairs):
            raise RuntimeError(f"Gemini returned {len(batch_vectors)} embeddings for {len(batch_pairs)} inputs")
        for (index, text), vector in zip(batch_pairs, batch_vectors):
            vectors[index] = vector
            _embedding_cache.set(
                _cache_key(text, task_type),
                vector,
                expire=settings.embedding_cache_ttl_seconds,
            )

    return [vector for vector in vectors if vector is not None]


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def embed_document(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]


def embed_for_cache(text: str) -> list[float]:
    return embed_texts([text], task_type="SEMANTIC_SIMILARITY")[0]
