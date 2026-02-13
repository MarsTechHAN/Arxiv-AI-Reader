"""
English tokenization and search scoring for paper search.
Uses Unicode-aware word tokenization and optional stemming.
"""

import re
from collections import Counter
from typing import List, Set

# Unicode word boundary: \w = letters, digits, underscore. Splits on hyphen, punctuation.
_WORD_PATTERN = re.compile(r"(?u)\w+")

_stemmer = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            import snowballstemmer
            _stemmer = snowballstemmer.stemmer("english")
        except ImportError:
            _stemmer = False
    return _stemmer


def tokenize(text: str, stem: bool = True) -> List[str]:
    """Tokenize English text. Extracts words (letters, digits, apostrophe)."""
    if not text or not isinstance(text, str):
        return []
    tokens = _WORD_PATTERN.findall(text.lower())
    if stem and tokens:
        s = _get_stemmer()
        if s:
            tokens = s.stemWords(tokens)
    return tokens


def tokenize_query(query: str, stem: bool = True) -> List[str]:
    """Tokenize search query for matching."""
    return tokenize(query, stem=stem)


def tokens_to_set(tokens: List[str]) -> Set[str]:
    return set(tokens)


def score_text(
    query: str,
    text: str,
    *,
    exact_phrase_bonus: float = 0.4,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """
    BM25-like scoring: tokenize both, compute TF-based relevance.
    query: search query
    text: document text to score
    exact_phrase_bonus: bonus when full query appears as substring
    k1, b: BM25-style parameters (k1=term freq saturation, b unused for single-doc)
    """
    if not text:
        return 0.0
    q_tokens = tokenize_query(query)
    if not q_tokens:
        return 0.0
    text_tokens = tokenize(text)
    text_counter = Counter(text_tokens)
    text_lower = text.lower()

    score = 0.0
    for t in q_tokens:
        tf = text_counter.get(t, 0)
        if tf > 0:
            # BM25-like: (k1 + 1) * tf / (k1 + tf)
            score += ((k1 + 1) * tf) / (k1 + tf)

    if score <= 0:
        return 0.0

    # Normalize by query length to avoid long queries dominating
    norm = len(q_tokens)
    score = score / norm

    # Exact phrase bonus
    if query.strip().lower() in text_lower:
        score += exact_phrase_bonus

    return min(1.0, score)


def score_text_legacy(query: str, text: str) -> float:
    """Simpler scoring: token overlap + exact match bonus. Compatible fallback."""
    return score_text(query, text, exact_phrase_bonus=0.3)


def normalize_fts_query(query: str) -> str:
    """
    Normalize query for FTS5: tokenize and rejoin. Ensures hyphenated/phrase
    terms match FTS unicode61 tokenization. Escapes double-quotes.
    """
    if not query or not isinstance(query, str):
        return ""
    tokens = tokenize(query, stem=False)
    normalized = " ".join(tokens)
    return normalized.replace('"', '""')
