from unittest.mock import patch

from src.retrieval.rerank import rerank_and_gate


def _candidate(text, manifest_kind=None, manifest_name=None):
    return {"text": text, "metadata": {"manifest_kind": manifest_kind, "manifest_name": manifest_name}}


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_empty_candidates_returns_empty(mock_ranker, mock_settings):
    assert rerank_and_gate("how do I create a resource?", []) == []


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_below_threshold_candidates_are_dropped_not_reordered_to_bottom(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_ranker.rerank.return_value = [
        {"id": 0, "score": 0.9},
        {"id": 1, "score": 0.2},
    ]
    candidates = [_candidate("relevant chunk"), _candidate("noisy chunk")]

    survivors = rerank_and_gate("question", candidates)

    assert len(survivors) == 1
    assert survivors[0]["text"] == "relevant chunk"


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_zero_survivors_when_all_below_threshold(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_ranker.rerank.return_value = [{"id": 0, "score": 0.1}]
    survivors = rerank_and_gate("question", [_candidate("weak chunk")])
    assert survivors == []


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_survivors_sorted_by_rerank_score_descending(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_ranker.rerank.return_value = [
        {"id": 0, "score": 0.55},
        {"id": 1, "score": 0.90},
    ]
    candidates = [_candidate("lower scored chunk"), _candidate("higher scored chunk")]

    survivors = rerank_and_gate("question", candidates)

    assert [c["text"] for c in survivors] == ["higher scored chunk", "lower scored chunk"]


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_ranker_failure_falls_back_to_retrieval_order_without_gating(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_ranker.rerank.side_effect = RuntimeError("ONNX load failed")
    candidates = [
        {**_candidate("lower retrieval score"), "retrieval_score": 0.3},
        {**_candidate("higher retrieval score"), "retrieval_score": 0.7},
    ]

    survivors = rerank_and_gate("question", candidates)

    # degrades to retrieval order, does not drop anything even though no
    # rerank score exists to gate on
    assert [c["text"] for c in survivors] == ["higher retrieval score", "lower retrieval score"]