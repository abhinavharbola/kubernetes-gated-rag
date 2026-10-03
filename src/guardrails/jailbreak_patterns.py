import re
import unicodedata

_INVISIBLE = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

_TARGET = r"(?:(?:all|any)\s+)?(?:of\s+)?(?:(?:the|your|my|these|those)\s+)?"
_VERB = r"(?:ignore|forget|disregard|override)"

JAILBREAK_PATTERNS = (
    re.compile(
        rf"\b{_VERB}\s+{_TARGET}(?:previous|prior|above|earlier|preceding|system)\s+"
        r"(?:instructions|prompts?|rules|guidelines|directions)\b",
        re.I,
    ),
    re.compile(rf"\b{_VERB}\s+(?:all\s+)?your\s+(?:instructions|rules|guidelines|programming)\b", re.I),
    re.compile(r"\b(?:forget|reveal|show|repeat|print|leak)\s+(?:(?:me|us)\s+)?(?:(?:your|the)\s+)?system\s+prompt\b", re.I),
    re.compile(r"\b(?:you\s+are\s+now|switch\s+to|enable|enter)\s+(?:DAN|developer\s+mode)\b", re.I),
    re.compile(
        r"\b(?:pretend|act|behave)\s+(?:like\s+|as\s+if\s+|that\s+)?you\s+(?:have|are|had)\s+no\s+"
        r"(?:rules|restrictions|filters|limits)\b",
        re.I,
    ),
    re.compile(r"\b(?:bypass|disable|override)\s+(?:your\s+)?(?:safety|content)\s+filters\b", re.I),
    re.compile(r"\bunrestricted\s+(?:AI|assistant|mode)\b", re.I),
    re.compile(r"\b(?:your|the)\s+new\s+instructions\s+(?:are|say)\b", re.I),
    re.compile(r"\broleplay\s+as\s+(?:an?\s+)?AI\s+(?:with|without)\s+(?:no\s+)?(?:filters|restrictions)\b", re.I),
)


def _normalize(text: str) -> str:
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))


def deterministic_jailbreak_check(raw_message: str) -> bool:
    cleaned = _normalize(raw_message)
    return any(pattern.search(cleaned) for pattern in JAILBREAK_PATTERNS)
