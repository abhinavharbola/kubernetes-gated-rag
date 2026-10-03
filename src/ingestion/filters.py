import logging
import re

from src.config import settings
from src.providers.llm import generate_planner

logger = logging.getLogger(__name__)

RELEVANCE_SYSTEM_PROMPT = (
    "You are a document relevance classifier for a Kubernetes documentation corpus. "
    "Given an excerpt from a document, classify whether its subject matter is about "
    "Kubernetes, container orchestration, or closely related cluster/infrastructure "
    "operations. Off-topic technical content (general CS theory, unrelated hardware, "
    "algorithms, compilers, data structures, etc.) is NOT relevant, even if it's "
    "technical and well-written. Respond with exactly one word: 'relevant' or "
    "'irrelevant'."
)

EXCERPT_CHARS = 2000

_VERDICT_RE = re.compile(r"\b(irrelevant|not\s+relevant|relevant)\b")


def is_relevant(document_text: str, fail_open: bool = True) -> bool:
    excerpt = document_text[:EXCERPT_CHARS]
    try:
        result = generate_planner(
            [
                {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                {"role": "user", "content": excerpt},
            ],
            timeout_seconds=settings.ingest_classifier_timeout_seconds,
        )
        verdict = result.content.strip().lower()
    except Exception as error:
        logger.warning("relevance classifier failed, %s: %s", "ingesting anyway" if fail_open else "skipping", error)
        return fail_open

    match = _VERDICT_RE.search(verdict)
    if match:
        return match.group(1) == "relevant"

    logger.warning("relevance classifier gave unparseable verdict %r, %s", verdict, "ingesting anyway" if fail_open else "skipping")
    return fail_open
