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
def test_below_threshold_candidates_are_dropped(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_settings.rerank_fail_closed = True
    mock_ranker.rerank.return_value = [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.2}]
    survivors = rerank_and_gate("question", [_candidate("relevant chunk"), _candidate("noisy chunk")])
    assert [c["text"] for c in survivors] == ["relevant chunk"]


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_ranker_failure_fails_closed_by_default(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_settings.rerank_fail_closed = True
    mock_ranker.rerank.side_effect = RuntimeError("ONNX load failed")
    assert rerank_and_gate("question", [_candidate("chunk")]) == []


@patch("src.retrieval.rerank.settings")
@patch("src.retrieval.rerank._ranker")
def test_ranker_failure_can_use_availability_fallback_when_explicitly_enabled(mock_ranker, mock_settings):
    mock_settings.rerank_fail_closed = False
    mock_ranker.rerank.side_effect = RuntimeError("ONNX load failed")
    candidates = [{**_candidate("low"), "retrieval_score": 0.3}, {**_candidate("high"), "retrieval_score": 0.7}]
    assert [c["text"] for c in rerank_and_gate("question", candidates)] == ["high", "low"]
