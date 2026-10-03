from unittest.mock import patch

import pytest

import ingest as ingest_module
from ingest import ingest_directory


@pytest.fixture(autouse=True)
def fast_ingest(monkeypatch):
    monkeypatch.setattr(ingest_module, "INGEST_EMBED_DELAY_SECONDS", 0.0)


@patch("ingest._delete_source_points")
@patch("ingest._upsert_batch")
@patch("ingest.embed_texts", return_value=[[0.1] * 768])
@patch("ingest.is_relevant", return_value=True)
@patch("ingest.chunk_document")
def test_ingest_directory_skips_a_bad_file_and_continues(
    mock_chunk_document, mock_is_relevant, mock_embed, mock_upsert, mock_delete, tmp_path
):
    (tmp_path / "a_broken.md").write_text("will fail to parse")
    (tmp_path / "b_good.md").write_text("fine content")

    def _parse_file(path):
        if path.name == "a_broken.md":
            raise ValueError("simulated parser crash")
        return "fine content"

    mock_chunk_document.return_value = [{"text": "fine content", "metadata": {}}]

    with patch("ingest.parse_file", side_effect=_parse_file):
        result = ingest_directory(tmp_path)

    assert result["failed"] == 1
    assert result["ingested"] > 0


@patch("ingest._delete_source_points")
@patch("ingest._upsert_batch")
@patch("ingest.embed_texts", side_effect=RuntimeError("gemini down for this file only"))
@patch("ingest.is_relevant", return_value=True)
@patch("ingest.chunk_document")
@patch("ingest.parse_file", return_value="some content")
def test_ingest_directory_counts_failures_without_raising(
    mock_parse_file, mock_chunk_document, mock_is_relevant, mock_embed, mock_upsert, mock_delete, tmp_path
):
    (tmp_path / "a.md").write_text("some content")
    mock_chunk_document.return_value = [{"text": "some content", "metadata": {}}]

    result = ingest_directory(tmp_path)

    assert result["ingested"] == 0
    assert result["failed"] == 1
    mock_delete.assert_not_called()
