import pytest

from src.retrieval.embeddings import embed_texts


def test_embed_texts_empty_list_returns_empty_list():
    assert embed_texts([], task_type="RETRIEVAL_QUERY") == []


def test_embed_texts_rejects_empty_string():
    # regression test: Gemini's embed_content rejects an empty string with
    # an opaque "EmbedContentRequest.content contains an empty Part" 400,
    # several frames deep inside a tenacity retry stack — this should fail
    # fast and clearly instead, right where the empty text was handed in.
    with pytest.raises(ValueError, match="empty string at index 0"):
        embed_texts([""], task_type="SEMANTIC_SIMILARITY")


def test_embed_texts_rejects_whitespace_only_string():
    with pytest.raises(ValueError, match="empty string at index 0"):
        embed_texts(["   \n  "], task_type="SEMANTIC_SIMILARITY")


def test_embed_texts_rejects_empty_string_anywhere_in_batch():
    with pytest.raises(ValueError, match="empty string at index 1"):
        embed_texts(["a real question", "", "another real question"], task_type="RETRIEVAL_DOCUMENT")
