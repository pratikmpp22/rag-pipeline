import re

# PII detection patterns
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "name": re.compile(r"\b[A-Z][a-z]{1,}\s[A-Z][a-z]{1,}\b"),
}

# Prompt injection patterns
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|.*?\|>"),
]


def detect_pii(text):
    """Return list of (type, match) tuples found in text."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((pii_type, match.group()))
    return findings


def sanitize_input(query):
    """Strip injection patterns from query, return (cleaned string, list of blocked substrings)."""
    cleaned = query
    blocked_matches = []
    
    for pattern in INJECTION_PATTERNS:
        # Find all matches before replacing them
        for match in pattern.finditer(cleaned):
            blocked_matches.append(match.group().strip())
        # Replace
        cleaned = pattern.sub("", cleaned)
        
    return cleaned.strip(), blocked_matches


def filter_output_pii(text):
    """Replace PII matches with [TYPE_REDACTED] placeholders."""
    filtered = text
    for pii_type, pattern in PII_PATTERNS.items():
        tag = f"[{pii_type.upper()}_REDACTED]"
        filtered = pattern.sub(tag, filtered)
    return filtered
