from unittest.mock import MagicMock, patch

from ingest import ensure_collection
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
