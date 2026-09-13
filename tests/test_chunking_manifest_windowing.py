from src.ingestion.chunking import MANIFEST_MAX_WORDS, split_by_manifest_blocks


def _large_configmap(word_count: int) -> str:
    # a ConfigMap embedding a large blob of config data is a realistic way
    # for a single manifest block to run well past MANIFEST_MAX_WORDS
    filler = " ".join(f"line{i}" for i in range(word_count))
    return (
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: big-config\n"
        f"data:\n  app.conf: |\n    {filler}\n"
    )


def test_small_manifest_block_stays_a_single_chunk():
    section = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: web\nspec:\n  containers: []\n"
    blocks = split_by_manifest_blocks(section)
    assert len(blocks) == 1
    assert blocks[0]["kind"] == "Pod"


def test_oversized_manifest_block_is_windowed_into_multiple_chunks():
    section = _large_configmap(word_count=900)
    blocks = split_by_manifest_blocks(section)
    assert len(blocks) > 1
    for block in blocks:
        assert len(block["text"].split()) <= MANIFEST_MAX_WORDS


def test_oversized_manifest_block_windows_keep_kind_and_name_metadata():
    section = _large_configmap(word_count=900)
    blocks = split_by_manifest_blocks(section)
    assert all(b["kind"] == "ConfigMap" for b in blocks)
    assert all(b["name"] == "big-config" for b in blocks)


def test_oversized_manifest_windows_overlap_like_prose_fallback():
    section = _large_configmap(word_count=900)
    blocks = split_by_manifest_blocks(section)
    first_words = set(blocks[0]["text"].split())
    second_words = set(blocks[1]["text"].split())
    assert first_words & second_words



