import re

JAILBREAK_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.I),
    re.compile(r"\b(?:forget|disregard|override)\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules|guidelines)\b", re.I),
    re.compile(r"\b(?:forget|reveal|show|repeat)\s+(?:your\s+)?system\s+prompt\b", re.I),
    re.compile(r"\b(?:you are now|switch to)\s+(?:DAN|developer mode)\b", re.I),
    re.compile(r"\b(?:pretend|act)\s+(?:that\s+)?you\s+(?:have|are)\s+no\s+(?:rules|restrictions|filters)\b", re.I),
    re.compile(r"\b(?:bypass|disable|override)\s+(?:your\s+)?(?:safety|content)\s+filters\b", re.I),
    re.compile(r"\bunrestricted\s+(?:AI|assistant|mode)\b", re.I),
    re.compile(r"\b(?:your|the)\s+new\s+instructions\s+(?:are|say)\b", re.I),
    re.compile(r"\broleplay\s+as\s+(?:an?\s+)?AI\s+(?:with|without)\s+(?:no\s+)?(?:filters|restrictions)\b", re.I),
)


def deterministic_jailbreak_check(raw_message: str) -> bool:
    return any(pattern.search(raw_message) for pattern in JAILBREAK_PATTERNS)






