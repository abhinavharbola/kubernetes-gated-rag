import hashlib
from unittest.mock import patch

from ingest import ingest_directory


@patch("ingest._upsert_batch")
@patch("ingest.embed_texts", return_value=[[0.1] * 768])
@patch("ingest.is_relevant", return_value=True)
@patch("ingest.chunk_document")
def test_ingest_directory_skips_a_bad_file_and_continues(
    mock_chunk_document, mock_is_relevant, mock_embed, mock_upsert, tmp_path
):
    # regression test: a single file that raises (corrupt document, a
    # transient error) used to abort ingest_directory entirely, dropping
    # every file after it in sorted order and, because main() never
    # reached _write_corpus_version(), silently leaving the on-disk
    # fingerprint out of sync with whatever *did* make it into Qdrant
    # before the crash.
    (tmp_path / "a_broken.md").write_text("will fail to parse")
    (tmp_path / "b_good.md").write_text("fine content")

    def _parse_file(path):
        if path.name == "a_broken.md":
            raise ValueError("simulated parser crash")
        return "fine content"

    mock_chunk_document.return_value = [{"text": "fine content", "metadata": {}}]

    with patch("ingest.parse_file", side_effect=_parse_file):
        hasher = hashlib.sha256()
        result = ingest_directory(tmp_path, corpus_hasher=hasher)

    assert result["failed"] == 1
    assert result["ingested"] > 0
    # only the file that actually succeeded should be reflected in the
    # fingerprint
    assert hasher.hexdigest() != hashlib.sha256().hexdigest()


@patch("ingest._upsert_batch")
@patch("ingest.embed_texts", side_effect=RuntimeError("gemini down for this file only"))
@patch("ingest.is_relevant", return_value=True)
@patch("ingest.chunk_document")
@patch("ingest.parse_file", return_value="some content")
def test_ingest_directory_counts_failures_without_raising(
    mock_parse_file, mock_chunk_document, mock_is_relevant, mock_embed, mock_upsert, tmp_path
):
    (tmp_path / "a.md").write_text("some content")
    mock_chunk_document.return_value = [{"text": "some content", "metadata": {}}]

    # must not raise — a failure inside the try/except must be caught and
    # counted, not propagated out of ingest_directory
    result = ingest_directory(tmp_path, corpus_hasher=hashlib.sha256())

    assert result["ingested"] == 0
    assert result["failed"] == 1
