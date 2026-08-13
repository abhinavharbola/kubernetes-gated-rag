import logging
import re

import numpy as np
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt

from src.providers.clients import gemini_client
from src.config import settings

logger = logging.getLogger(__name__)

# Gemini's embed_content endpoint accepts a batch of texts in a single
# request, not just one. Looping one-call-per-text would cost N network
# round-trips and N slices of a free-tier RPM budget instead of
# ceil(N / EMBED_BATCH_SIZE). 100 is comfortably under Gemini's per-request
# batch limit; lower it if that limit changes.
EMBED_BATCH_SIZE = 100

RETRY_DELAY_RE = re.compile(r"([\d.]+)\s*s")


def _is_rate_limit_error(error: BaseException) -> bool:
    return isinstance(error, ClientError) and getattr(error, "code", None) == 429


def _extract_retry_delay_seconds(error: ClientError, default: float = 30.0) -> float:
    # the API tells us exactly how long to wait (google.rpc.RetryInfo.retryDelay,
    # e.g. "31s") — respecting that beats guessing with a fixed backoff schedule,
    # too short and we just hit the same 429 again, too long and we wait
    # longer than necessary.
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
    return delay + 1.0  # small buffer past the server's own estimate


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


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """task_type: RETRIEVAL_DOCUMENT for ingestion, RETRIEVAL_QUERY for queries,
    SEMANTIC_SIMILARITY for cache lookups."""
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(_embed_batch(batch, task_type))
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def embed_document(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]


def embed_for_cache(text: str) -> list[float]:
    return embed_texts([text], task_type="SEMANTIC_SIMILARITY")[0]
