"""
MCP server for arXiv AI Reader.
Exposes search and retrieval tools for papers.
"""

import re
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Use backend-relative data path
BACKEND_DIR = Path(__file__).parent
DATA_DIR = str(BACKEND_DIR / "data" / "papers")

# Lazy init fetcher
_fetcher = None


def _get_fetcher():
    global _fetcher
    if _fetcher is None:
        from fetcher import ArxivFetcher
        _fetcher = ArxivFetcher(data_dir=DATA_DIR)
    return _fetcher


def _calculate_similarity(query: str, text: str) -> float:
    """Calculate text similarity score (0-1)."""
    if not text:
        return 0.0
    query_lower = query.lower()
    text_lower = text.lower()
    exact_bonus = 0.3 if query_lower in text_lower else 0.0
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    if not query_words:
        return 0.0
    intersection = len(query_words & text_words)
    union = len(query_words | text_words)
    jaccard = intersection / union if union > 0 else 0.0
    word_freq_score = sum(1 for w in query_words if w in text_lower) / len(query_words)
    return min(1.0, jaccard * 0.4 + word_freq_score * 0.3 + exact_bonus)


def _do_search(q: str, fetcher, limit: int = 50, ids_only: bool = False,
               search_full_text: bool = True, search_generated_only: bool = False) -> list:
    """Core search logic."""
    q_lower = q.lower()
    metadata_list = fetcher.list_papers_metadata(max_files=5000, check_stale=True)
    results = []
    arxiv_id_pattern = r'^\d{4}\.\d{4,5}(v\d+)?$'

    if re.match(arxiv_id_pattern, q.strip()):
        arxiv_id = q.strip()
        for meta in metadata_list:
            if meta.get('id', '').startswith(arxiv_id.split('v')[0]):
                if meta.get('is_hidden', False):
                    continue
                try:
                    paper = fetcher.load_paper(meta['id'])
                    item = {"id": paper.id, "search_score": 1000.0}
                    if not ids_only:
                        item.update({
                            "title": paper.title,
                            "authors": paper.authors,
                            "abstract": paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract,
                            "url": paper.url,
                            "one_line_summary": paper.one_line_summary,
                            "detailed_summary": paper.detailed_summary,
                        })
                    results.append(item)
                    return results[:limit]
                except Exception:
                    pass
        return results

    for meta in metadata_list:
        if meta.get('is_hidden', False):
            continue
        paper_id = meta.get('id', '')
        title = meta.get('title', '')
        abstract = meta.get('abstract', '')
        detailed_summary = meta.get('detailed_summary', '')
        one_line_summary = meta.get('one_line_summary', '')
        preview_text = meta.get('preview_text', '')
        authors = meta.get('authors', [])
        tags = meta.get('tags', [])
        extracted_keywords = meta.get('extracted_keywords', [])

        if search_generated_only:
            searchable = f"{one_line_summary} {detailed_summary} {' '.join(tags + extracted_keywords)}"
            if not _calculate_similarity(q, searchable):
                continue
        elif not search_full_text:
            searchable = f"{title} {abstract} {one_line_summary} {detailed_summary} {' '.join(authors)} {' '.join(tags + extracted_keywords)}"
            if not _calculate_similarity(q, searchable):
                continue

        title_score = _calculate_similarity(q, title) * 2.0 if title else 0.0
        abstract_score = _calculate_similarity(q, abstract) if abstract else 0.0
        summary_score = _calculate_similarity(q, detailed_summary) * 1.5 if detailed_summary else 0.0
        one_line_score = _calculate_similarity(q, one_line_summary) * 1.2 if one_line_summary else 0.0
        fulltext_score = _calculate_similarity(q, preview_text) * 0.8 if preview_text and search_full_text else 0.0
        author_score = 0.8 if any(
            any(w in a.lower() for w in set(q_lower.split()))
            for a in authors
        ) else 0.0
        tags_text = ' '.join(tags + extracted_keywords).lower()
        tag_score = _calculate_similarity(q, tags_text) * 1.2 if tags_text else 0.0

        total_score = title_score + abstract_score + summary_score + one_line_score + fulltext_score + author_score + tag_score
        if total_score <= 0:
            continue

        try:
            paper = fetcher.load_paper(paper_id)
            item = {"id": paper.id, "search_score": total_score}
            if not ids_only:
                item.update({
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract,
                    "url": paper.url,
                    "one_line_summary": paper.one_line_summary,
                    "detailed_summary": paper.detailed_summary[:500] + "..." if len(paper.detailed_summary or "") > 500 else (paper.detailed_summary or ""),
                })
            results.append(item)
        except Exception:
            continue

    results.sort(key=lambda x: x["search_score"], reverse=True)
    return results[:limit]


mcp = FastMCP("arXiv AI Reader", json_response=True)


@mcp.tool()
def search_papers(
    query: str,
    limit: int = 50,
    ids_only: bool = False,
    search_full_text: bool = True,
) -> list:
    """
    Search papers by keyword, arXiv ID, title, author, abstract, or AI summaries.
    Uses metadata cache (fast). search_full_text=True uses preview (~2k chars); for
    real full-text search use search_full_text tool.
    """
    fetcher = _get_fetcher()
    return _do_search(query, fetcher, limit=limit, ids_only=ids_only, search_full_text=search_full_text, search_generated_only=False)


@mcp.tool()
def search_generated_content(query: str, limit: int = 50, ids_only: bool = False) -> list:
    """
    Search only within AI-generated content: one_line_summary, detailed_summary, tags, extracted_keywords.
    Use when you want to find papers by AI analysis, not original text.
    """
    fetcher = _get_fetcher()
    return _do_search(query, fetcher, limit=limit, ids_only=ids_only, search_full_text=False, search_generated_only=True)


def _do_search_full_text(q: str, fetcher, limit: int = 50, ids_only: bool = False, max_scan: int = 2000) -> list:
    """
    Search within actual full paper html_content.
    Loads each paper from disk - slower but searches entire paper text.
    """
    q_lower = q.lower()
    metadata_list = fetcher.list_papers_metadata(max_files=max_scan, check_stale=True)
    results = []

    for meta in metadata_list[:max_scan]:
        if meta.get('is_hidden', False):
            continue
        paper_id = meta.get('id', '')
        try:
            paper = fetcher.load_paper(paper_id)
            full_text = (paper.html_content or "") + " " + (paper.abstract or "")
            if not full_text.strip():
                continue
            score = _calculate_similarity(q, full_text)
            if q_lower in full_text.lower():
                score += 0.5  # Bonus for exact substring match
            if score <= 0:
                continue

            item = {"id": paper.id, "search_score": score}
            if not ids_only:
                item.update({
                    "title": paper.title,
                    "authors": paper.authors,
                    "abstract": paper.abstract[:300] + "..." if len(paper.abstract or "") > 300 else (paper.abstract or ""),
                    "url": paper.url,
                })
            results.append(item)
        except Exception:
            continue

    results.sort(key=lambda x: x["search_score"], reverse=True)
    return results[:limit]


@mcp.tool()
def search_full_text(query: str, limit: int = 50, ids_only: bool = False, max_scan: int = 2000) -> list:
    """
    Search within ACTUAL full paper text (html_content + abstract).
    Loads papers from disk - slower but searches entire content. Use max_scan to limit papers scanned.
    """
    fetcher = _get_fetcher()
    return _do_search_full_text(query, fetcher, limit=limit, ids_only=ids_only, max_scan=max_scan)


@mcp.tool()
async def get_paper(
    arxiv_id: str,
    include_abstract: bool = True,
    include_html_content: bool = False,
    include_one_line_summary: bool = True,
    include_detailed_summary: bool = True,
    include_qa_pairs: bool = False,
    include_tags: bool = True,
) -> dict:
    """
    Get a paper by arXiv ID (e.g. 2401.12345 or 2401.12345v1).
    Configurable: choose which content to include.
    If paper not local, fetches from arXiv.
    """
    fetcher = _get_fetcher()
    arxiv_id = arxiv_id.strip()

    try:
        if fetcher._paper_exists(arxiv_id):
            paper = fetcher.load_paper(arxiv_id)
        else:
            paper = await fetcher.fetch_single_paper(arxiv_id)
    except Exception as e:
        return {"error": str(e), "arxiv_id": arxiv_id}

    out = {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "url": paper.url,
        "published_date": paper.published_date,
        "is_relevant": paper.is_relevant,
        "relevance_score": paper.relevance_score,
    }
    if include_abstract:
        out["abstract"] = paper.abstract
    if include_one_line_summary:
        out["one_line_summary"] = paper.one_line_summary
    if include_detailed_summary:
        out["detailed_summary"] = paper.detailed_summary
    if include_tags:
        out["tags"] = getattr(paper, 'tags', [])
        out["extracted_keywords"] = paper.extracted_keywords
    if include_qa_pairs:
        out["qa_pairs"] = [
            {"question": qa.question, "answer": qa.answer}
            for qa in (paper.qa_pairs or [])
        ]
    if include_html_content:
        out["html_content"] = paper.html_content or ""
    return out


@mcp.tool()
async def get_paper_full_text(arxiv_id: str) -> dict:
    """
    Get a paper by arXiv ID with FULL text content (html_content) included.
    Returns: id, title, authors, abstract, url, one_line_summary, detailed_summary,
    tags, qa_pairs, and html_content (full paper text extracted from arXiv HTML).
    If paper not local, fetches from arXiv.
    """
    fetcher = _get_fetcher()
    arxiv_id = arxiv_id.strip()
    try:
        if fetcher._paper_exists(arxiv_id):
            paper = fetcher.load_paper(arxiv_id)
        else:
            paper = await fetcher.fetch_single_paper(arxiv_id)
    except Exception as e:
        return {"error": str(e), "arxiv_id": arxiv_id}

    return {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "url": paper.url,
        "published_date": paper.published_date,
        "one_line_summary": paper.one_line_summary,
        "detailed_summary": paper.detailed_summary,
        "tags": getattr(paper, 'tags', []),
        "extracted_keywords": paper.extracted_keywords,
        "qa_pairs": [{"question": qa.question, "answer": qa.answer} for qa in (paper.qa_pairs or [])],
        "html_content": paper.html_content or "",
    }


@mcp.tool()
def get_paper_ids_by_query(query: str, limit: int = 50) -> list:
    """
    Search papers and return only arXiv IDs. Fast way to get IDs for follow-up get_paper calls.
    """
    fetcher = _get_fetcher()
    results = _do_search(query, fetcher, limit=limit, ids_only=True)
    return [r["id"] for r in results]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
