import hashlib
from unittest.mock import MagicMock, patch

from ingest import _write_corpus_version, ensure_collection, ingest_directory
from src.config import settings


def _mock_collections(*names: str) -> MagicMock:
    response = MagicMock()
    response.collections = [MagicMock(name=n) for n in names]
    # MagicMock(name=...) sets the mock's repr, not a `.name` attribute
    # (name is a reserved constructor kwarg on Mock) — set it explicitly.
    for mock_collection, n in zip(response.collections, names):
        mock_collection.name = n
    return response


@patch("ingest.qdrant_client")
def test_ensure_collection_creates_both_collections_when_missing(mock_qdrant):
    mock_qdrant.get_collections.return_value = _mock_collections()
    ensure_collection(wipe=False)
    created = {call.kwargs["collection_name"] for call in mock_qdrant.create_collection.call_args_list}
    assert created == {settings.qdrant_docs_collection, settings.qdrant_cache_collection}
    assert mock_qdrant.delete_collection.called is False


@patch("ingest.qdrant_client")
def test_ensure_collection_does_not_touch_existing_collections_without_wipe(mock_qdrant):
    mock_qdrant.get_collections.return_value = _mock_collections(
        settings.qdrant_docs_collection, settings.qdrant_cache_collection
    )
    ensure_collection(wipe=False)
    assert mock_qdrant.delete_collection.called is False
    assert mock_qdrant.create_collection.called is False


@patch("ingest._wipe_exact_cache")
@patch("ingest.qdrant_client")
def test_wipe_deletes_both_docs_and_semantic_cache_collections(mock_qdrant, mock_wipe_exact_cache):
    # regression test: --wipe used to only clear the docs collection,
    # leaving the semantic cache (permanent, no TTL) serving stale answers
    # against a corpus that no longer matches them.
    mock_qdrant.get_collections.return_value = _mock_collections(
        settings.qdrant_docs_collection, settings.qdrant_cache_collection
    )
    ensure_collection(wipe=True)
    deleted = {call.args[0] for call in mock_qdrant.delete_collection.call_args_list}
    assert deleted == {settings.qdrant_docs_collection, settings.qdrant_cache_collection}
    # both get recreated after being wiped
    created = {call.kwargs["collection_name"] for call in mock_qdrant.create_collection.call_args_list}
    assert created == {settings.qdrant_docs_collection, settings.qdrant_cache_collection}


@patch("ingest._wipe_exact_cache")
@patch("ingest.qdrant_client")
def test_wipe_also_clears_the_local_exact_cache(mock_qdrant, mock_wipe_exact_cache):
    # regression test: --wipe left the permanent (expire=None) local exact
    # cache untouched, so a repeated question could still be served a
    # pre-wipe answer straight out of SQLite, bypassing Qdrant entirely.
    mock_qdrant.get_collections.return_value = _mock_collections()
    ensure_collection(wipe=True)
    assert mock_wipe_exact_cache.called is True


@patch("ingest.qdrant_client")
def test_wipe_is_a_noop_for_the_exact_cache_when_wipe_is_false(mock_qdrant):
    mock_qdrant.get_collections.return_value = _mock_collections(
        settings.qdrant_docs_collection, settings.qdrant_cache_collection
    )
    with patch("ingest._wipe_exact_cache") as mock_wipe_exact_cache:
        ensure_collection(wipe=False)
        assert mock_wipe_exact_cache.called is False


def test_write_corpus_version_writes_the_marker_file(tmp_path):
    marker = tmp_path / "nested" / "corpus_version"
    with patch("ingest.CORPUS_VERSION_MARKER", marker):
        _write_corpus_version("abc123")
    assert marker.read_text() == "abc123"


@patch("ingest._upsert_batch")
@patch("ingest.embed_texts", return_value=[[0.1] * 768])
@patch("ingest.is_relevant", return_value=True)
@patch("ingest.chunk_document")
@patch("ingest.parse_file")
def test_ingest_directory_updates_hasher_for_ingested_files_only(
    mock_parse_file, mock_chunk_document, mock_is_relevant, mock_embed, mock_upsert, tmp_path
):
    (tmp_path / "a.md").write_text("relevant content")
    mock_parse_file.return_value = "relevant content"
    mock_chunk_document.return_value = [{"text": "relevant content", "metadata": {}}]

    hasher = hashlib.sha256()
    ingest_directory(tmp_path, corpus_hasher=hasher)

    assert hasher.hexdigest() != hashlib.sha256().hexdigest()


@patch("ingest._upsert_batch")
@patch("ingest.embed_texts")
@patch("ingest.is_relevant", return_value=False)
@patch("ingest.chunk_document")
@patch("ingest.parse_file", return_value="off topic content")
def test_ingest_directory_does_not_hash_rejected_files(
    mock_parse_file, mock_chunk_document, mock_is_relevant, mock_embed, mock_upsert, tmp_path
):
    (tmp_path / "a.md").write_text("off topic content")

    hasher = hashlib.sha256()
    ingest_directory(tmp_path, corpus_hasher=hasher)

    # a rejected file contributes nothing to the corpus, so it must not
    # change the fingerprint either -- only content that actually became
    # part of the corpus should be able to invalidate old cache entries.
    assert hasher.hexdigest() == hashlib.sha256().hexdigest()
    mock_embed.assert_not_called()


@patch("ingest._upsert_batch")
@patch("ingest.embed_texts", return_value=[[0.1] * 768])
@patch("ingest.is_relevant", return_value=True)
@patch("ingest.chunk_document")
@patch("ingest.parse_file", return_value="same content")
def test_ingest_directory_fingerprint_is_deterministic_across_runs(
    mock_parse_file, mock_chunk_document, mock_is_relevant, mock_embed, mock_upsert, tmp_path
):
    (tmp_path / "a.md").write_text("same content")
    mock_chunk_document.return_value = [{"text": "same content", "metadata": {}}]

    first = hashlib.sha256()
    ingest_directory(tmp_path, corpus_hasher=first)
    second = hashlib.sha256()
    ingest_directory(tmp_path, corpus_hasher=second)

    assert first.hexdigest() == second.hexdigest()
