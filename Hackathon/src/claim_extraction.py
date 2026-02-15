"""Rule-based claim extraction without LLM."""

import re
from typing import List

from .preprocessing import split_sentences


# Strong verbs that often indicate factual claims
STRONG_VERBS = {
    "is", "are", "was", "were", "has", "have", "had", "will", "would",
    "said", "says", "claimed", "claims", "proved", "proves", "shows",
    "causes", "caused", "kills", "killed", "prevents", "prevented",
    "increases", "decreased", "reduces", "reduced", "found", "discovered",
    "confirmed", "denied", "revealed", "exposed", "linked", "contains",
}

# Patterns for numbers (statistics, dates, etc.)
NUMBER_PATTERN = re.compile(r"\d+([.,]\d+)?%?|\d{1,2}/\d{1,2}/\d{2,4}")

# Named entity-ish patterns: capitalized words, acronyms
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b")


def _has_strong_verb(sentence: str) -> bool:
    """Check if sentence contains strong claim-indicating verbs."""
    lower = sentence.lower()
    words = set(re.findall(r"\b\w+\b", lower))
    return bool(words & STRONG_VERBS)


def _has_number(sentence: str) -> bool:
    """Check if sentence contains numbers."""
    return bool(NUMBER_PATTERN.search(sentence))


def _has_entity(sentence: str) -> bool:
    """Check if sentence has entity-like tokens (capitalized, acronyms)."""
    return bool(ENTITY_PATTERN.search(sentence))


def _is_claim_like(sentence: str) -> bool:
    """Heuristic: sentence looks like a factual claim."""
    if len(sentence) < 10:
        return False
    # At least one of: strong verb, number, or entity
    return _has_strong_verb(sentence) or _has_number(sentence) or _has_entity(sentence)


def _similarity_simple(a: str, b: str) -> float:
    """Simple word overlap similarity 0-1."""
    wa = set(re.findall(r"\b\w+\b", a.lower()))
    wb = set(re.findall(r"\b\w+\b", b.lower()))
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


def _deduplicate_claims(claims: List[str], threshold: float = 0.7) -> List[str]:
    """Remove claims that are too similar to earlier ones."""
    result: List[str] = []
    for c in claims:
        if not c.strip():
            continue
        is_dup = False
        for r in result:
            if _similarity_simple(c, r) >= threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(c)
    return result


# Max characters per claim block so we don't lose focus (context preserved but bounded)
MAX_CLAIM_BLOCK_CHARS = 450


def _build_claim_blocks(
    sentences: List[str],
    claim_like_indices: List[int],
    add_context_for_singles: bool = True,
) -> List[str]:
    """
    Group consecutive claim-like sentences into blocks to preserve context.
    For a single claim-like sentence, optionally add one adjacent sentence for context.
    """
    if not sentences or not claim_like_indices:
        return []

    set_claim = set(claim_like_indices)
    blocks: List[str] = []

    # Group consecutive indices: [0,1,3,5,6] -> [[0,1], [3], [5,6]]
    groups: List[List[int]] = []
    current: List[int] = []
    for i in claim_like_indices:
        if not current or i == current[-1] + 1:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    if current:
        groups.append(current)

    for group in groups:
        start, end = group[0], group[-1]
        # Optionally add one context sentence for singletons so we don't verify a single line in isolation
        if add_context_for_singles and len(group) == 1:
            if start > 0:
                start = start - 1  # include sentence before
            elif end + 1 < len(sentences):
                end = end + 1  # include sentence after
        block_sentences = sentences[start : end + 1]
        block = " ".join(block_sentences)
        # Cap length so verification stays focused
        if len(block) > MAX_CLAIM_BLOCK_CHARS:
            block = block[: MAX_CLAIM_BLOCK_CHARS].rsplit(" ", 1)[0] or block[:MAX_CLAIM_BLOCK_CHARS]
        if block.strip():
            blocks.append(block.strip())

    return blocks


def extract_claims(text: str, max_claims: int = 6) -> List[str]:
    """
    Extract claim blocks from text while preserving context.
    - Splits by sentence but groups consecutive claim-like sentences into blocks
    - Single claim-like sentences get one adjacent sentence for context (avoids line-by-line loss of context)
    - Deduplicates and returns up to max_claims blocks
    """
    if not text or not str(text).strip():
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    claim_like_indices = [i for i, s in enumerate(sentences) if _is_claim_like(s)]

    if not claim_like_indices:
        # Fallback: use first few non-trivial sentences as one or two blocks (keep context)
        fallback = [s for s in sentences if len(s) > 15]
        if not fallback:
            fallback = sentences[:3]
        # Join into at most 2 blocks so we still have some context
        chunk_size = max(1, (len(fallback) + 1) // 2)
        blocks = [
            " ".join(fallback[i : i + chunk_size]).strip()
            for i in range(0, len(fallback), chunk_size)
        ][:max_claims]
        return blocks

    blocks = _build_claim_blocks(sentences, claim_like_indices, add_context_for_singles=True)
    deduped = _deduplicate_claims(blocks, threshold=0.6)
    return deduped[:max_claims]
