import argparse
import logging
import uuid
from pathlib import Path

from qdrant_client.models import Distance, PointStruct, VectorParams

from src.chunking import chunk_document
from src.clients import qdrant_client
from src.config import settings
from src.embeddings import embed_texts
from src.ingest_filter import is_relevant
from src.parsers import PARSERS, parse_file
from src.tracing import node_span

logger = logging.getLogger(__name__)

# matches the on-disk convention from the source data: a DATA/ directory
# containing a true_data/ subfolder (the real corpus) and a noisy_data/
# subfolder (off-topic/adversarial content, expected to be rejected by
# is_relevant() before it's ever chunked or embedded, not a lower-trust
# second tier of valid content).
TRUE_DIR_NAME = "true_data"
NOISY_DIR_NAME = "noisy_data"


def ensure_collection(wipe: bool = False) -> None:
    existing = {c.name for c in qdrant_client.get_collections().collections}

    if wipe and settings.qdrant_docs_collection in existing:
        qdrant_client.delete_collection(settings.qdrant_docs_collection)
        existing.discard(settings.qdrant_docs_collection)
        logger.info("wiped existing collection: %s", settings.qdrant_docs_collection)

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

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": chunk["text"], "metadata": chunk["metadata"]},
                )
                for chunk, vector in zip(chunks, vectors)
            ]

            qdrant_client.upsert(collection_name=settings.qdrant_docs_collection, points=points)
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
    parser.add_argument("--wipe", action="store_true", help="delete and recreate the docs collection first")
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
