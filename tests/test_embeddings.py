from unittest.mock import patch

import pytest

from src.retrieval.embeddings import embed_texts


@pytest.fixture(autouse=True)
def clear_embedding_cache():
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


def test_interactive_calls_skip_the_retry_wrapper():
    from unittest.mock import patch

    from src.retrieval import embeddings

    with (
        patch.object(embeddings, "_embed_batch_once", return_value=[[1.0]]) as once,
        patch.object(embeddings, "_embed_batch_with_retry", return_value=[[2.0]]) as with_retry,
    ):
        assert embeddings._embed_batch(["x"], "RETRIEVAL_QUERY", interactive=True) == [[1.0]]
        assert embeddings._embed_batch(["x"], "RETRIEVAL_DOCUMENT") == [[2.0]]
    once.assert_called_once()
    with_retry.assert_called_once()


def test_query_and_cache_embeddings_are_interactive_and_documents_are_not():
    from unittest.mock import patch

    from src.retrieval import embeddings

    with patch.object(embeddings, "embed_texts", return_value=[[0.1]]) as embed_texts:
        embeddings.embed_query("q")
        embeddings.embed_for_cache("q")
    assert all(call.kwargs["interactive"] is True for call in embed_texts.call_args_list)


def test_retry_wait_is_capped():
    from unittest.mock import MagicMock, patch

    from src.retrieval import embeddings

    state = MagicMock()
    with patch.object(embeddings, "_extract_retry_delay_seconds", return_value=900.0):
        assert embeddings._rate_limit_wait(state) == embeddings.settings.embedding_max_retry_wait_seconds + 1.0
