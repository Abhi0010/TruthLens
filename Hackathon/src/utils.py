"""Utility functions for Clarion."""

from pathlib import Path
from typing import Any, Optional


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_kb_path() -> Path:
    """Return path to the knowledge base file."""
    return get_project_root() / "src" / "kb" / "seed_kb.md"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length with suffix."""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def is_empty_input(text: Optional[str]) -> bool:
    """Check if input text is empty or whitespace-only."""
    return not text or not str(text).strip()
