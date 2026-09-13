from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.search import RetrievalUnavailableError, retrieve


@patch("src.retrieval.search.qdrant_client")
@patch("src.retrieval.search.embed_query", return_value=[0.1] * 768)
def test_retrieve_returns_candidates_on_success(mock_embed, mock_qdrant):
    point = MagicMock()
    point.payload = {"text": "Pods are the smallest deployable unit.", "metadata": {}}
    point.score = 0.9
    mock_qdrant.query_points.return_value.points = [point]

    results = retrieve("what is a Pod?")

    assert results == [
        {"text": "Pods are the smallest deployable unit.", "metadata": {}, "retrieval_score": 0.9}
    ]


@patch("src.retrieval.search.qdrant_client")
@patch("src.retrieval.search.embed_query", return_value=[0.1] * 768)
def test_retrieve_raises_unavailable_on_qdrant_failure(mock_embed, mock_qdrant):
    # A Qdrant failure must NOT come back as an empty list: graph.py treats
    # an empty list as "genuinely searched and found nothing" and caches it
    # as a permanent-ish "no grounded documentation" answer. Conflating an
    # outage with a real empty result would bake a transient failure into
    # the cache well past when the outage ended.
    mock_qdrant.query_points.side_effect = Exception("read timeout")
    with pytest.raises(RetrievalUnavailableError):
        retrieve("what is a Pod?")


@patch("src.retrieval.search.embed_query", side_effect=Exception("gemini timeout"))
def test_retrieve_raises_unavailable_on_embedding_failure(mock_embed):
    with pytest.raises(RetrievalUnavailableError):
        retrieve("what is a Pod?")



