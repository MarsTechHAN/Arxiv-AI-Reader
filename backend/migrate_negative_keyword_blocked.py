"""
One-time migration: clean historical papers with negative-keyword block in one_line_summary.
- Replace block message with one-sentence summary (from abstract)
- Set is_relevant by relevance_score instead of tag
- Also fix papers with bad summaries (arXiv:... or Announce Type)
Run: cd backend && python migrate_negative_keyword_blocked.py
"""
import re
import sys
from pathlib import Path

# Ensure backend is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetcher import ArxivFetcher
from models import Config

NEGATIVE_KEYWORD_BLOCK_PATTERN = "论文包含负面关键词"
BAD_SUMMARY_PATTERNS = ("arXiv:", "Announce Type:")
MAX_SUMMARY_LEN = 200


def _extract_real_abstract(abstract: str) -> str:
    """Strip arXiv metadata prefix to get real abstract content."""
    if not abstract or not abstract.strip():
        return ""
    text = abstract.strip()
    # Match "arXiv:xxx Announce Type: xxx" + optional newline
    m = re.match(r"^arXiv:\S+\s+Announce Type:\s*\S+\s*\n?", text, re.I)
    if m:
        text = text[m.end() :].strip()
    if text.lower().startswith("abstract:"):
        text = text[9:].strip()
    return text


def _abstract_to_one_line(abstract: str) -> str:
    """Extract one-sentence summary from abstract (no LLM)."""
    text = _extract_real_abstract(abstract or "")
    if not text:
        return "（无摘要）"
    # First sentence: end at . or 。 or newline (prefer sentence breaks)
    for sep in [". ", "。", ".\n", "\n"]:
        idx = text.find(sep)
        if 0 < idx <= MAX_SUMMARY_LEN:
            return text[: idx + len(sep)].strip()
    return text[:MAX_SUMMARY_LEN].rstrip() + ("…" if len(text) > MAX_SUMMARY_LEN else "")


def main():
    config_path = Path(__file__).parent / "data" / "config.json"
    config = Config.load(config_path)
    min_score = getattr(config, "min_relevance_score_for_stage2", 6.0)

    fetcher = ArxivFetcher()
    all_papers = fetcher.list_papers(skip=0, limit=100000)
    blocked = [
        p
        for p in all_papers
        if NEGATIVE_KEYWORD_BLOCK_PATTERN in (p.one_line_summary or "")
    ]
    bad_summaries = [
        p
        for p in all_papers
        if any(pat in (p.one_line_summary or "") for pat in BAD_SUMMARY_PATTERNS)
    ]
    to_fix = list({p.id: p for p in blocked + bad_summaries}.values())

    if not to_fix:
        print("✓ No papers to migrate")
        return

    print(f"🔄 Migrating {len(to_fix)} papers...")
    for paper in to_fix:
        old_summary = paper.one_line_summary or ""
        paper.one_line_summary = _abstract_to_one_line(paper.abstract or "")
        paper.is_relevant = paper.relevance_score >= min_score
        fetcher.save_paper(paper)
        print(f"  {paper.id}: score={paper.relevance_score} -> is_relevant={paper.is_relevant}")
        print(f"    one_line: ... -> {paper.one_line_summary[:80]}...")
    print(f"✓ Migrated {len(to_fix)} papers")


if __name__ == "__main__":
    main()
