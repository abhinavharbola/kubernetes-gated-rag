from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.cache import (
    embed_canonical_question,
    exact_cache_get,
    exact_cache_set,
    normalize_exact,
    semantic_cache_get,
    semantic_cache_set,
)


@pytest.fixture(autouse=True)
def clear_exact_cache():
    from src.retrieval.cache import _exact_cache
    _exact_cache.clear()
    yield
    _exact_cache.clear()


def test_normalize_collapses_whitespace_case_and_punctuation():
    assert normalize_exact("  How Do I Create a Resource?  ") == "how do i create a resource"


def test_normalize_does_not_collapse_different_questions():
    a = normalize_exact("how do I create a resource")
    b = normalize_exact("how do I destroy a resource")
    assert a != b


def test_exact_cache_roundtrip():
    exact_cache_set("How do I create a resource?", "answer text")
    assert exact_cache_get("how do i create a resource") == "answer text"


def test_exact_cache_miss_returns_none():
    assert exact_cache_get("nothing stored for this question") is None


@patch("src.retrieval.cache.embed_for_cache", return_value=[0.1] * 768)
def test_embed_canonical_question_delegates_to_embed_for_cache(mock_embed):
    vector = embed_canonical_question("how a Deployment differs from a StatefulSet")
    assert vector == [0.1] * 768
    mock_embed.assert_called_once_with("how a Deployment differs from a StatefulSet")


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_hit_above_threshold(mock_qdrant):
    mock_point = MagicMock()
    mock_point.payload = {"answer": "cached semantic answer"}
    mock_qdrant.query_points.return_value.points = [mock_point]

    result = semantic_cache_get([0.1] * 768)
    assert result == "cached semantic answer"


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_miss_below_threshold(mock_qdrant):
    mock_qdrant.query_points.return_value.points = []
    assert semantic_cache_get([0.1] * 768) is None


@patch("src.retrieval.cache.qdrant_client")
def test_semantic_cache_set_upserts_the_given_vector_without_re_embedding(mock_qdrant):
    # regression test: semantic_cache_set must NOT call embed_for_cache
    # itself, it should upsert whatever vector it's handed. Re-embedding
    # here would repeat the exact Gemini call semantic_cache_get already
    # made for the same canonical_question earlier in the same turn.
    with patch("src.retrieval.cache.embed_for_cache") as mock_embed:
        semantic_cache_set("canonical question", [0.2] * 768, "an answer")
        assert mock_embed.called is False

    assert mock_qdrant.upsert.called
    upserted_point = mock_qdrant.upsert.call_args.kwargs["points"][0]
    assert upserted_point.vector == [0.2] * 768
    assert upserted_point.payload == {"question": "canonical question", "answer": "an answer"}