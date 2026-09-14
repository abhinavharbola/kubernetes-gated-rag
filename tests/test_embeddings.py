from unittest.mock import patch

import pytest

from src.retrieval.embeddings import embed_texts


@pytest.fixture(autouse=True)
def clear_embedding_cache():
    # _embedding_cache is a real diskcache.Cache persisted at .cache/embeddings
    # across process runs (not just across tests in one run) — without
    # clearing it, a question cached by a previous test run makes this
    # test's "first call populates the cache" assumption false on the next
    # run in the same checkout, and _embed_batch is never actually called.
    from src.retrieval.embeddings import _embedding_cache
    _embedding_cache.clear()
    yield
    _embedding_cache.clear()


def test_embed_texts_empty_list_returns_empty_list():
    assert embed_texts([], task_type="RETRIEVAL_QUERY") == []


def test_embed_texts_rejects_empty_string():
    with pytest.raises(ValueError, match="empty string at index 0"):
        embed_texts([""], task_type="SEMANTIC_SIMILARITY")


def test_embed_texts_rejects_whitespace_only_string():
    with pytest.raises(ValueError, match="empty string at index 0"):
        embed_texts(["   \n  "], task_type="SEMANTIC_SIMILARITY")


def test_embed_texts_rejects_empty_string_anywhere_in_batch():
    with pytest.raises(ValueError, match="empty string at index 1"):
        embed_texts(["a real question", "", "another real question"], task_type="RETRIEVAL_DOCUMENT")


def test_embed_texts_uses_persistent_cache_before_calling_gemini():
    with patch("src.retrieval.embeddings._embed_batch") as embed_batch:
        embed_batch.return_value = [[0.1] * 768]
        first = embed_texts(["cached question"], task_type="RETRIEVAL_QUERY")
        second = embed_texts(["cached question"], task_type="RETRIEVAL_QUERY")
    assert first == second
    embed_batch.assert_called_once()
