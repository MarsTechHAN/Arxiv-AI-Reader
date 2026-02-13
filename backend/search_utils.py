"""
English tokenization and search scoring for paper search.
Uses Unicode-aware word tokenization, optional stemming, phrase support, and ngrams.
"""

import re
from collections import Counter
from typing import List, Set, Tuple

# Unicode word boundary: \w = letters, digits, underscore. Splits on hyphen, punctuation.
_WORD_PATTERN = re.compile(r"(?u)\w+")
_QUOTED_PHRASE_PATTERN = re.compile(r'"([^"]*)"')

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


def get_ngrams(text: str, n: int = 3) -> List[str]:
    """Character ngrams for partial/fuzzy matching. Min length n."""
    if not text or len(text) < n:
        return []
    text = text.lower()
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def parse_query_parts(query: str) -> List[Tuple[str, str]]:
    """
    Parse query into phrases (quoted) and tokens (unquoted).
    Returns [("phrase", "machine learning"), ("token", "paper"), ...]
    """
    if not query or not isinstance(query, str):
        return []
    parts = []
    last_end = 0
    for m in _QUOTED_PHRASE_PATTERN.finditer(query):
        before = query[last_end : m.start()].strip()
        if before:
            tokens = tokenize(before, stem=False)
            for t in tokens:
                parts.append(("token", t))
        phrase = m.group(1).strip()
        if phrase:
            parts.append(("phrase", phrase))
        last_end = m.end()
    rest = query[last_end:].strip()
    if rest:
        for t in tokenize(rest, stem=False):
            parts.append(("token", t))
    return parts


def tokens_to_set(tokens: List[str]) -> Set[str]:
    return set(tokens)


def score_text(
    query: str,
    text: str,
    *,
    exact_phrase_bonus: float = 0.4,
    phrase_bonus: float = 0.35,
    ngram_bonus: float = 0.15,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """
    BM25-like scoring with phrase and ngram support.
    query: search query (use "phrase" for forced non-tokenization)
    text: document text to score
    exact_phrase_bonus: bonus when full query appears as substring
    phrase_bonus: bonus per quoted phrase match
    ngram_bonus: bonus for ngram overlap (partial/fuzzy)
    k1, b: BM25-style parameters
    """
    if not text:
        return 0.0
    parts = parse_query_parts(query)
    if not parts:
        q_tokens = tokenize_query(query)
        if not q_tokens:
            return 0.0
        parts = [("token", t) for t in q_tokens]

    text_tokens = tokenize(text)
    text_counter = Counter(text_tokens)
    text_lower = text.lower()
    text_ngrams = set(get_ngrams(text_lower, 3))

    score = 0.0
    token_count = 0
    for kind, val in parts:
        if kind == "phrase":
            if val.lower() in text_lower:
                score += phrase_bonus
        else:
            token_count += 1
            tf = text_counter.get(val, 0)
            if tf > 0:
                score += ((k1 + 1) * tf) / (k1 + tf)
            # ngram overlap for partial match
            q_ngrams = set(get_ngrams(val, 3))
            if q_ngrams and (q_ngrams & text_ngrams):
                overlap = len(q_ngrams & text_ngrams) / len(q_ngrams)
                score += ngram_bonus * overlap

    if score <= 0:
        return 0.0

    norm = max(1, token_count)
    score = score / norm

    if query.strip().lower() in text_lower:
        score += exact_phrase_bonus

    return min(1.0, score)


def score_text_legacy(query: str, text: str) -> float:
    """Simpler scoring: token overlap + exact match bonus. Compatible fallback."""
    return score_text(query, text, exact_phrase_bonus=0.3)


def normalize_fts_query(query: str) -> str:
    """
    Normalize query for FTS5: preserve quoted phrases, tokenize rest.
    Output like: "machine learning" transformer
    Escapes embedded double-quotes inside phrases.
    """
    if not query or not isinstance(query, str):
        return ""
    parts = parse_query_parts(query)
    if not parts:
        tokens = tokenize(query, stem=False)
        return " ".join(tokens).replace('"', '""')

    out = []
    for kind, val in parts:
        if kind == "phrase":
            escaped = val.replace('"', '""')
            out.append(f'"{escaped}"')
        else:
            out.append(val)
    return " ".join(out)
