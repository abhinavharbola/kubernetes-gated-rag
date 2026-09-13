import argparse
import hashlib
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
from src.retrieval.cache import ensure_semantic_cache_indexes
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

# Anchored to this file's own directory (the repo root), not to the process's
# current working directory. ingest.py and ui/app.py previously both wrote
# ".cache/..." relative to cwd — if either was launched from a different
# directory than the other (e.g. `cd ui && streamlit run app.py`), they'd
# silently read and write two different .cache trees, breaking the corpus
# fingerprint handshake between them with no error, just a fallback to the
# static CORPUS_VERSION. src/retrieval/cache.py and src/retrieval/embeddings.py
# anchor the same way for the same reason.
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"

# src/retrieval/cache.py reads this file to get the "effective" corpus
# version for cache keys, falling back to settings.corpus_version when it
# doesn't exist. That marker is written below, from a hash of every file
# that actually made it into the corpus this run (path + content). Deriving
# it from content rather than relying on a human to bump CORPUS_VERSION
# means an incremental re-ingest (no --wipe, just adding/changing files)
# still invalidates stale cached answers automatically, instead of them
# silently surviving forever because nobody remembered to edit .env.
CORPUS_VERSION_MARKER = CACHE_DIR / "corpus_version"


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
    # payload index for cache_schema_version/policy_version/corpus_version,
    # required for the version filter in src/retrieval/cache.py's
    # semantic_cache_get/set. Idempotent, safe whether the collection was
    # just created above or already existed from a previous run.
    ensure_semantic_cache_indexes()


def _wipe_exact_cache() -> None:
    # local import: ingest.py otherwise has no reason to touch
    # src.retrieval.cache, and importing it unconditionally at module load
    # would pull in diskcache's on-disk init for a CLI path that might
    # never need it (e.g. --help).
    from src.retrieval.cache import _exact_cache

    count = len(_exact_cache)
    _exact_cache.clear()
    logger.info("wiped %d entr%s from the local exact cache", count, "y" if count == 1 else "ies")


def ingest_directory(directory: Path, corpus_hasher: "hashlib._Hash | None" = None) -> dict:
    ingested = 0
    skipped_irrelevant = 0
    failed = 0

    # Sorted so the corpus fingerprint is deterministic across runs on the
    # same content — rglob's own order depends on filesystem/OS details and
    # would otherwise make corpus_hasher's digest vary between two ingests
    # of identical files, defeating the point of fingerprinting.
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PARSERS:
            continue

        with node_span("ingest_file", path=str(path)):
            # Isolated per file: a single bad document (corrupt PDF, a
            # DOCX python-docx can't open, a transient embedding/Qdrant
            # error on just this file) used to raise straight out of
            # ingest_directory and abort the whole run, including files
            # that already succeeded and were upserted into Qdrant earlier
            # in the same loop. Those earlier files stayed in Qdrant, but
            # _write_corpus_version() at the end of main() never ran
            # because main() itself crashed — so the corpus fingerprint on
            # disk silently fell behind what Qdrant actually held, and
            # stale cache entries kept serving as if nothing had changed.
            # Catching here means one bad file is logged and skipped, and
            # the corpus fingerprint still reflects everything that
            # actually made it in.
            try:
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

                if corpus_hasher is not None:
                    corpus_hasher.update(str(path.relative_to(directory)).encode("utf-8"))
                    corpus_hasher.update(text.encode("utf-8"))
            except Exception as error:
                logger.error("failed to ingest %s, skipping this file: %s", path, error)
                failed += 1
                continue

    return {"ingested": ingested, "skipped_irrelevant": skipped_irrelevant, "failed": failed}


def _write_corpus_version(fingerprint: str) -> None:
    CORPUS_VERSION_MARKER.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_VERSION_MARKER.write_text(fingerprint)
    logger.info("corpus fingerprint written: %s", fingerprint)


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
        help="delete and recreate the docs collection AND the semantic/exact caches first. "
        "Not required for cache correctness any more (the corpus fingerprint written to "
        ".cache/corpus_version after every run already invalidates stale cached answers "
        "automatically), it's for reclaiming space / starting from a clean collection.",
    )
    args = parser.parse_args()

    true_dir = args.data_dir / TRUE_DIR_NAME
    noisy_dir = args.data_dir / NOISY_DIR_NAME

    if not true_dir.is_dir() and not noisy_dir.is_dir():
        parser.error(
            f"neither {TRUE_DIR_NAME}/ nor {NOISY_DIR_NAME}/ found under {args.data_dir}; "
            "refusing to run, since writing a corpus fingerprint from zero ingested files "
            "would invalidate every cached answer for the real corpus"
        )

    ensure_collection(wipe=args.wipe)

    corpus_hasher = hashlib.sha256()
    total_failed = 0

    if true_dir.is_dir():
        result = ingest_directory(true_dir, corpus_hasher)
        total_failed += result["failed"]
        print(
            f"ingested {result['ingested']} chunks from {true_dir}, "
            f"skipped {result['skipped_irrelevant']} document(s) as irrelevant, "
            f"failed on {result['failed']} document(s)"
        )
    else:
        print(f"no {TRUE_DIR_NAME}/ found under {args.data_dir}, skipping")

    if noisy_dir.is_dir():
        result = ingest_directory(noisy_dir, corpus_hasher)
        total_failed += result["failed"]
        print(
            f"ingested {result['ingested']} chunks from {noisy_dir}, "
            f"skipped {result['skipped_irrelevant']} document(s) as irrelevant "
            "(expect this to be at or near 100% of the files in noisy_data/), "
            f"failed on {result['failed']} document(s)"
        )
    else:
        print(f"no {NOISY_DIR_NAME}/ found under {args.data_dir}, skipping")

    _write_corpus_version(corpus_hasher.hexdigest()[:16])
    if total_failed:
        print(f"note: {total_failed} document(s) failed to ingest — see logs above for which ones and why.")


if __name__ == "__main__":
    main()



