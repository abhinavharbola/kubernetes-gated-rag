import re

MARKDOWN_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

# A Kubernetes manifest conventionally begins with `apiVersion:` as its first
# top-level field, by the same near-universal convention that made a resource
# block's `resource "type" "name" {` a reliable start marker for HCL. Using
# it as the block-start marker means an object's boundary is simply "up to
# the next top-level apiVersion:" — no brace or indent matching needed, since
# YAML's own indentation already keeps everything belonging to one object
# nested under it.
MANIFEST_START_RE = re.compile(r"^apiVersion:\s*\S", re.MULTILINE)

# a standalone `---` is the YAML multi-document separator between manifests,
# not part of any object's own content
SEPARATOR_LINE_RE = re.compile(r"^---[ \t]*\r?\n?", re.MULTILINE)

FALLBACK_WINDOW_WORDS = 300
FALLBACK_OVERLAP_WORDS = 50


def split_by_markdown_headers(text: str) -> list[dict]:
    matches = list(MARKDOWN_HEADER_RE.finditer(text))
    if not matches:
        return [{"header": None, "text": text}]

    sections = []
    if matches[0].start() > 0:
        sections.append({"header": None, "text": text[: matches[0].start()]})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({"header": match.group(2).strip(), "text": text[start:end]})

    return sections


def _strip_separator_lines(text: str) -> str:
    return SEPARATOR_LINE_RE.sub("", text)


def _clean_scalar(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip("'\"")


def _parse_manifest_kind_and_name(block_text: str) -> tuple[str | None, str | None]:
    kind_match = re.search(r"^kind:\s*(\S+)", block_text, re.MULTILINE)
    kind = _clean_scalar(kind_match.group(1)) if kind_match else None

    name = None
    metadata_match = re.search(r"^metadata:\s*$", block_text, re.MULTILINE)
    if metadata_match:
        rest = block_text[metadata_match.end() :]
        # metadata's indented body runs until the next line back at column 0
        # (the next top-level key in this object, or the next manifest entirely)
        next_top_level = re.search(r"^\S", rest, re.MULTILINE)
        metadata_body = rest[: next_top_level.start()] if next_top_level else rest
        name_match = re.search(r"^\s+name:\s*(\S+)", metadata_body, re.MULTILINE)
        name = _clean_scalar(name_match.group(1)) if name_match else None

    return kind, name


def split_by_manifest_blocks(section_text: str) -> list[dict]:
    # Known limitation, same class as the old HCL chunker's brace-inside-a-
    # string edge case: this is a structural heuristic, not a real YAML
    # parser. A block scalar (`|` or `>`) whose literal content happens to
    # contain a line that is exactly `---` will be mis-split as if it were a
    # document boundary. Rare in practice for Kubernetes manifests, but a
    # real limitation worth knowing about rather than silently pretending
    # this is full YAML awareness.
    blocks = []
    cursor = 0
    starts = list(MANIFEST_START_RE.finditer(section_text))

    if not starts:
        return _fallback_window(section_text)

    for i, match in enumerate(starts):
        if match.start() > cursor:
            leftover = _strip_separator_lines(section_text[cursor : match.start()])
            blocks.extend(_fallback_window(leftover))

        end = starts[i + 1].start() if i + 1 < len(starts) else len(section_text)
        block_text = _strip_separator_lines(section_text[match.start() : end]).strip()

        if block_text:
            kind, name = _parse_manifest_kind_and_name(block_text)
            blocks.append({"text": block_text, "kind": kind, "name": name})
        cursor = end

    if cursor < len(section_text):
        leftover = _strip_separator_lines(section_text[cursor:])
        blocks.extend(_fallback_window(leftover))

    return blocks


def _fallback_window(text: str) -> list[dict]:
    words = text.split()
    if not words:
        return []
    if len(words) <= FALLBACK_WINDOW_WORDS:
        return [{"text": text.strip(), "kind": None, "name": None}] if text.strip() else []

    chunks = []
    step = FALLBACK_WINDOW_WORDS - FALLBACK_OVERLAP_WORDS
    for start in range(0, len(words), step):
        window = words[start : start + FALLBACK_WINDOW_WORDS]
        if window:
            chunks.append({"text": " ".join(window), "kind": None, "name": None})
        if start + FALLBACK_WINDOW_WORDS >= len(words):
            break
    return chunks


def chunk_document(text: str, base_metadata: dict) -> list[dict]:
    chunks = []
    for section in split_by_markdown_headers(text):
        for block in split_by_manifest_blocks(section["text"]):
            if not block["text"].strip():
                continue
            metadata = {
                **base_metadata,
                "section_header": section["header"],
                "manifest_kind": block["kind"],
                "manifest_name": block["name"],
            }
            chunks.append({"text": block["text"], "metadata": metadata})
    return chunks
