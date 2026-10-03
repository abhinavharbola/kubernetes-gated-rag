import argparse
import hashlib
import logging
import time
import uuid
from pathlib import Path

from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import cache_root, settings
from src.ingestion.chunking import chunk_document
from src.ingestion.filters import is_relevant
from src.ingestion.parsers import PARSERS, parse_file
from src.providers.clients import qdrant_client
from src.retrieval.cache import ensure_semantic_cache_indexes, exact_cache_clear
from src.retrieval.embeddings import embed_texts
from src.tracing import node_span

logger = logging.getLogger(__name__)

TRUE_DIR_NAME = "true_data"
NOISY_DIR_NAME = "noisy_data"

INGEST_EMBED_DELAY_SECONDS = 1.0
UPSERT_BATCH_SIZE = 64
SOURCE_PATH_FIELD = "metadata.source_path"
YAML_SUFFIXES = {".yaml", ".yml"}

CACHE_DIR = cache_root()
CORPUS_VERSION_MARKER = CACHE_DIR / "corpus_version"

_transport_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(ResponseHandlingException),
    reraise=True,
)


@_transport_retry
def _upsert_batch(points: list[PointStruct]) -> None:
    qdrant_client.upsert(collection_name=settings.qdrant_docs_collection, points=points)


@_transport_retry
def _delete_source_points(source: str) -> None:
    qdrant_client.delete(
        collection_name=settings.qdrant_docs_collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key=SOURCE_PATH_FIELD, match=MatchValue(value=source))])
        ),
    )


def _point_id(source: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}"))


def _ensure_source_index() -> None:
    try:
        qdrant_client.create_payload_index(
            collection_name=settings.qdrant_docs_collection,
            field_name=SOURCE_PATH_FIELD,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception as error:
        if "already exist" not in str(error).lower():
            raise


def ensure_collection(wipe: bool = False) -> None:
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
    _ensure_source_index()
    ensure_semantic_cache_indexes()


def _wipe_exact_cache() -> None:
    count = exact_cache_clear()
    logger.info("wiped %d entr%s from the local exact cache", count, "y" if count == 1 else "ies")


def ingest_directory(directory: Path, root: Path | None = None, relevance_fail_open: bool = True) -> dict:
    ingested = 0
    skipped_irrelevant = 0
    failed = 0
    root = root or directory

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PARSERS:
            continue

        source = path.relative_to(root).as_posix()
        with node_span("ingest_file", path=source):
            try:
                text = parse_file(path)
                if not text.strip():
                    _delete_source_points(source)
                    continue

                if not is_relevant(text, fail_open=relevance_fail_open):
                    logger.info("skipping (classified irrelevant to corpus): %s", source)
                    _delete_source_points(source)
                    skipped_irrelevant += 1
                    continue

                chunks = chunk_document(
                    text,
                    base_metadata={"source_path": source},
                    markdown=path.suffix.lower() not in YAML_SUFFIXES,
                )
                if not chunks:
                    _delete_source_points(source)
                    continue

                vectors = embed_texts([chunk["text"] for chunk in chunks], task_type="RETRIEVAL_DOCUMENT")
                time.sleep(INGEST_EMBED_DELAY_SECONDS)

                points = [
                    PointStruct(
                        id=_point_id(source, index),
                        vector=vector,
                        payload={"text": chunk["text"], "metadata": chunk["metadata"]},
                    )
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors))
                ]

                _delete_source_points(source)
                for batch_start in range(0, len(points), UPSERT_BATCH_SIZE):
                    _upsert_batch(points[batch_start : batch_start + UPSERT_BATCH_SIZE])
                ingested += len(points)
            except Exception as error:
                logger.error("failed to ingest %s, skipping this file: %s", source, error)
                failed += 1
                continue

    return {"ingested": ingested, "skipped_irrelevant": skipped_irrelevant, "failed": failed}


def compute_corpus_fingerprint() -> str:
    digests = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=settings.qdrant_docs_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            source = (payload.get("metadata") or {}).get("source_path", "")
            digests.append(hashlib.sha256(f"{source}\0{payload.get('text', '')}".encode("utf-8")).hexdigest())
        if offset is None:
            break
    combined = hashlib.sha256()
    for digest in sorted(digests):
        combined.update(digest.encode("ascii"))
    return combined.hexdigest()[:16]


def _write_corpus_version(fingerprint: str) -> None:
    CORPUS_VERSION_MARKER.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_VERSION_MARKER.write_text(fingerprint)
    logger.info("corpus fingerprint written: %s", fingerprint)


def _current_fingerprint() -> str:
    try:
        return compute_corpus_fingerprint()
    except Exception as error:
        logger.error("could not fingerprint the corpus, using a unique marker to invalidate caches: %s", error)
        return uuid.uuid4().hex[:16]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

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
        "Required to drop documents whose source files were deleted from disk.",
    )
    args = parser.parse_args()

    true_dir = args.data_dir / TRUE_DIR_NAME
    noisy_dir = args.data_dir / NOISY_DIR_NAME

    if not true_dir.is_dir() and not noisy_dir.is_dir():
        parser.error(f"neither {TRUE_DIR_NAME}/ nor {NOISY_DIR_NAME}/ found under {args.data_dir}")

    ensure_collection(wipe=args.wipe)

    total_failed = 0

    if true_dir.is_dir():
        result = ingest_directory(true_dir, root=args.data_dir)
        total_failed += result["failed"]
        print(
            f"ingested {result['ingested']} chunks from {true_dir}, "
            f"skipped {result['skipped_irrelevant']} document(s) as irrelevant, "
            f"failed on {result['failed']} document(s)"
        )
    else:
        print(f"no {TRUE_DIR_NAME}/ found under {args.data_dir}, skipping")

    if noisy_dir.is_dir():
        result = ingest_directory(noisy_dir, root=args.data_dir, relevance_fail_open=False)
        total_failed += result["failed"]
        print(
            f"ingested {result['ingested']} chunks from {noisy_dir}, "
            f"skipped {result['skipped_irrelevant']} document(s) as irrelevant "
            "(expect this to be at or near 100% of the files in noisy_data/), "
            f"failed on {result['failed']} document(s)"
        )
    else:
        print(f"no {NOISY_DIR_NAME}/ found under {args.data_dir}, skipping")

    total_points = qdrant_client.count(collection_name=settings.qdrant_docs_collection, exact=True).count
    if total_points == 0:
        parser.exit(1, "the docs collection is empty after ingestion, corpus fingerprint not updated\n")

    _write_corpus_version(_current_fingerprint())
    if total_failed:
        print(f"note: {total_failed} document(s) failed to ingest, see logs above for which ones and why.")


if __name__ == "__main__":
    main()
