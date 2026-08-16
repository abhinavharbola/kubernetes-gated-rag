import argparse
import logging
import time
import uuid
from pathlib import Path

from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, PointStruct, VectorParams
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.ingestion.chunking import chunk_document
from src.providers.clients import qdrant_client
from src.config import settings
from src.retrieval.embeddings import embed_texts
from src.ingestion.filters import is_relevant
from src.ingestion.parsers import PARSERS, parse_file
from src.tracing import node_span

logger = logging.getLogger(__name__)

# matches the on-disk convention from the source data: a DATA/ directory
# containing a true_data/ subfolder (the real corpus) and a noisy_data/
# subfolder (off-topic/adversarial content, expected to be rejected by
# is_relevant() before it's ever chunked or embedded, not a lower-trust
# second tier of valid content).
TRUE_DIR_NAME = "true_data"
NOISY_DIR_NAME = "noisy_data"

# Gemini's free-tier embed_content quota is 100 requests/minute. embed_texts()
# already retries on a 429 respecting the server's suggested backoff, but
# that's a reactive fix — spacing ingestion's own calls out proactively means
# fewer 429s to react to in the first place. 1s between files keeps a
# many-small-files corpus comfortably under the ceiling; raise this if you're
# still hitting 429s with a lot of files.
INGEST_EMBED_DELAY_SECONDS = 1.0

# a single large-file upsert (many points, each with a full vector + text
# payload) can be a big enough request body to hit a write timeout on a
# free-tier Qdrant Cloud connection. Splitting into smaller batches keeps
# each request quick and means a mid-file failure doesn't lose the points
# that already made it in.
UPSERT_BATCH_SIZE = 64


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(ResponseHandlingException),
    reraise=True,
)
def _upsert_batch(points: list[PointStruct]) -> None:
    # ResponseHandlingException wraps transport-level failures (timeouts,
    # connection resets), not a real server-side rejection of the request
    # (that's UnexpectedResponse, a 4xx/5xx with a body) — safe to retry.
    qdrant_client.upsert(collection_name=settings.qdrant_docs_collection, points=points)


def ensure_collection(wipe: bool = False) -> None:
    # wipe clears BOTH the docs collection and the semantic cache, not just
    # docs. A cached answer (semantic or the permanent, expire=None exact
    # cache) points at content that may no longer exist or may have changed
    # once the corpus is re-ingested; leaving the cache alone means repeat
    # or paraphrased questions keep silently serving pre-wipe answers
    # indefinitely, since only the "no grounded documentation" outcome has
    # a TTL, not a real generated answer.
    existing = {c.name for c in qdrant_client.get_collections().collections}

    if wipe:
        for collection_name in (settings.qdrant_docs_collection, settings.qdrant_cache_collection):
            if collection_name in existing:
                qdrant_client.delete_collection(collection_name)
                existing.discard(collection_name)
                logger.info("wiped existing collection: %s", collection_name)
        _wipe_exact_cache()

    if settings.qdrant_docs_collection not in existing:
        qdrant_client.create_collection(
            collection_name=settings.qdrant_docs_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
    if settings.qdrant_cache_collection not in existing:
        qdrant_client.create_collection(
            collection_name=settings.qdrant_cache_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )


def _wipe_exact_cache() -> None:
    # local import: ingest.py otherwise has no reason to touch
    # src.retrieval.cache, and importing it unconditionally at module load
    # would pull in diskcache's on-disk init for a CLI path that might
    # never need it (e.g. --help).
    from src.retrieval.cache import _exact_cache

    count = len(_exact_cache)
    _exact_cache.clear()
    logger.info("wiped %d entr%s from the local exact cache", count, "y" if count == 1 else "ies")


def ingest_directory(directory: Path) -> dict:
    ingested = 0
    skipped_irrelevant = 0

    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in PARSERS:
            continue

        with node_span("ingest_file", path=str(path)):
            text = parse_file(path)
            if not text.strip():
                continue

            if not is_relevant(text):
                logger.info("skipping (classified irrelevant to corpus): %s", path)
                skipped_irrelevant += 1
                continue

            chunks = chunk_document(text, base_metadata={"source_path": str(path)})
            if not chunks:
                continue

            # one batched call for every chunk in this file instead of
            # one Gemini round-trip per chunk.
            vectors = embed_texts([chunk["text"] for chunk in chunks], task_type="RETRIEVAL_DOCUMENT")
            time.sleep(INGEST_EMBED_DELAY_SECONDS)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": chunk["text"], "metadata": chunk["metadata"]},
                )
                for chunk, vector in zip(chunks, vectors)
            ]

            for batch_start in range(0, len(points), UPSERT_BATCH_SIZE):
                _upsert_batch(points[batch_start : batch_start + UPSERT_BATCH_SIZE])
            ingested += len(points)

    return {"ingested": ingested, "skipped_irrelevant": skipped_irrelevant}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest Kubernetes docs into Qdrant. Expects a data directory containing "
            f"'{TRUE_DIR_NAME}/' and/or '{NOISY_DIR_NAME}/' subfolders; both are run "
            "through the same relevance gate, noisy_data exists to prove the gate "
            "actually rejects off-topic content, not as a second valid content tier."
        )
    )
    parser.add_argument("data_dir", type=Path, help="e.g. DATA, containing true_data/ and/or noisy_data/")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="delete and recreate the docs collection AND the semantic/exact caches first "
        "(a corpus refresh without --wipe leaves stale cached answers pointing at old content)",
    )
    args = parser.parse_args()

    ensure_collection(wipe=args.wipe)

    true_dir = args.data_dir / TRUE_DIR_NAME
    noisy_dir = args.data_dir / NOISY_DIR_NAME

    if true_dir.is_dir():
        result = ingest_directory(true_dir)
        print(
            f"ingested {result['ingested']} chunks from {true_dir}, "
            f"skipped {result['skipped_irrelevant']} document(s) as irrelevant"
        )
    else:
        print(f"no {TRUE_DIR_NAME}/ found under {args.data_dir}, skipping")

    if noisy_dir.is_dir():
        result = ingest_directory(noisy_dir)
        print(
            f"ingested {result['ingested']} chunks from {noisy_dir}, "
            f"skipped {result['skipped_irrelevant']} document(s) as irrelevant "
            "(expect this to be at or near 100% of the files in noisy_data/)"
        )
    else:
        print(f"no {NOISY_DIR_NAME}/ found under {args.data_dir}, skipping")


if __name__ == "__main__":
    main()
