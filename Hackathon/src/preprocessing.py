"""Text preprocessing for TruthLens Suite."""

import re
from typing import List, Optional


def clean_text(text: str) -> str:
    """
    Clean input text: normalize whitespace, strip extra chars.
    """
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple regex.
    Handles common punctuation: . ! ?
    """
    if not text:
        return []
    text = clean_text(text)
    if not text:
        return []
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def detect_language(text: str) -> str:
    """
    Simple heuristic: assume English if mostly ASCII.
    Optional: could add langdetect later.
    """
    if not text:
        return "unknown"
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return "en" if ascii_ratio > 0.9 else "unknown"


def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    if not text:
        return []
    url_pattern = r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
    return re.findall(url_pattern, text, re.IGNORECASE)


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text."""
    if not text:
        return []
    return re.findall(r"#\w+", text)
