from unittest.mock import MagicMock, patch

from src.retrieval.search import retrieve


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
def test_retrieve_returns_empty_list_on_qdrant_failure(mock_embed, mock_qdrant):
    mock_qdrant.query_points.side_effect = Exception("read timeout")
    assert retrieve("what is a Pod?") == []


@patch("src.retrieval.search.embed_query", side_effect=Exception("gemini timeout"))
def test_retrieve_returns_empty_list_on_embedding_failure(mock_embed):
    assert retrieve("what is a Pod?") == []
