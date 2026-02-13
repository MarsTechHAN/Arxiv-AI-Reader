"""
MCP server for arXiv AI Reader.
Exposes search and retrieval tools for papers.
"""

import re
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Optional date parser for date range filter
def _parse_date(s):
    if not s or not isinstance(s, str):
        return None
    try:
        from dateutil import parser as date_parser
        from datetime import timezone
        dt = date_parser.parse(s.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


def _paper_in_date_range(meta_or_item, from_date_dt, to_date_dt):
    """Check if paper's published_date is within [from_date, to_date]."""
    if from_date_dt is None and to_date_dt is None:
        return True
    pd = meta_or_item.get("published_date", "") or ""
    if not pd:
        return True
    try:
        from dateutil import parser as date_parser
        from datetime import timezone
        dt = date_parser.parse(pd)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        if from_date_dt and dt < from_date_dt:
            return False
        if to_date_dt and dt > to_date_dt:
            return False
        return True
    except Exception:
        return True

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
    """Tokenized BM25-like scoring via search_utils."""
    from search_utils import score_text
    return score_text(query, text, exact_phrase_bonus=0.3)


def _meta_matches_tab(meta: dict, category: Optional[str] = None, starred_only: bool = False) -> bool:
    if not category and not starred_only:
        return True
    if not meta.get("is_starred", False):
        return False
    if category and meta.get("star_category", "Other") != category:
        return False
    return True


def _do_search(q: str, fetcher, limit: int = 50, ids_only: bool = False,
               search_full_text: bool = True, search_generated_only: bool = False,
               from_date: Optional[str] = None, to_date: Optional[str] = None,
               sort_by: str = "relevance", skip: int = 0,
               category: Optional[str] = None, starred_only: bool = False) -> list:
    """Core search logic. Uses store FTS when available (SQLite), else in-memory scan."""
    from_date_dt = _parse_date(from_date) if from_date else None
    to_date_dt = _parse_date(to_date) if to_date else None

    tab_filter = category or starred_only
    fetch_limit = limit + skip
    store = getattr(fetcher, "store", None)
    if store and hasattr(store, "search") and not search_generated_only:
        fts_limit = fetch_limit * 5 if tab_filter else fetch_limit
        fts_results = store.search(q.strip(), limit=fts_limit * 3 if (from_date_dt or to_date_dt) else fts_limit, search_full_text=search_full_text)
        if fts_results:
            filtered = [r for r in fts_results if _paper_in_date_range(r, from_date_dt, to_date_dt)]
            if tab_filter:
                meta_list = fetcher.list_papers_metadata(max_files=999999, check_stale=False)
                id_to_meta = {m["id"]: m for m in meta_list}
                filtered = [r for r in filtered if _meta_matches_tab(id_to_meta.get(r["id"], {}), category, starred_only)]
            if from_date_dt or to_date_dt:
                for r in filtered:
                    if "published_date" not in r:
                        try:
                            p = fetcher.load_paper(r["id"])
                            r["published_date"] = getattr(p, "published_date", "") or ""
                        except Exception:
                            r["published_date"] = ""
            if sort_by == "latest":
                def _sort_key(r):
                    pd = r.get("published_date", "") or ""
                    try:
                        from dateutil import parser as date_parser
                        from datetime import timezone
                        dt = date_parser.parse(pd) if pd else None
                        if dt and dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt or __import__("datetime").datetime.fromtimestamp(0, tz=timezone.utc)
                    except Exception:
                        return __import__("datetime").datetime.fromtimestamp(0, tz=__import__("datetime").timezone.utc)
                filtered.sort(key=_sort_key, reverse=True)
            else:
                filtered.sort(key=lambda x: x.get("search_score", 0), reverse=True)
            filtered = filtered[skip:skip + limit]
            if ids_only:
                return [{"id": r["id"], "search_score": r.get("search_score", 0)} for r in filtered]
            return filtered

    metadata_list = fetcher.list_papers_metadata(max_files=999999, check_stale=True)
    if tab_filter:
        metadata_list = [m for m in metadata_list if _meta_matches_tab(m, category, starred_only)]
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
        if not _paper_in_date_range(meta, from_date_dt, to_date_dt):
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

        # Core fields: title, authors, abstract, AI summaries
        title_score = _calculate_similarity(q, title) * 2.0 if title else 0.0
        abstract_score = _calculate_similarity(q, abstract) if abstract else 0.0
        summary_score = _calculate_similarity(q, detailed_summary) * 1.5 if detailed_summary else 0.0
        one_line_score = _calculate_similarity(q, one_line_summary) * 1.2 if one_line_summary else 0.0
        authors_text = ' '.join(authors or [])
        author_score = _calculate_similarity(q, authors_text) * 1.2 if authors_text else 0.0
        fulltext_score = _calculate_similarity(q, preview_text) * 0.8 if preview_text and search_full_text else 0.0
        tags_text = ' '.join(tags + extracted_keywords).lower()
        tag_score = _calculate_similarity(q, tags_text) * 1.2 if tags_text else 0.0

        total_score = title_score + abstract_score + summary_score + one_line_score + fulltext_score + author_score + tag_score
        if total_score <= 0:
            continue

        try:
            paper = fetcher.load_paper(paper_id)
            item = {"id": paper.id, "search_score": total_score, "published_date": getattr(paper, "published_date", "") or ""}
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

    if sort_by == "latest":
        from datetime import datetime, timezone
        def _dt_key(x):
            return _parse_date(x.get("published_date", "")) or datetime.fromtimestamp(0, tz=timezone.utc)
        results.sort(key=_dt_key, reverse=True)
    else:
        results.sort(key=lambda x: x["search_score"], reverse=True)
    return results[skip:skip + limit]


mcp = FastMCP("arXiv AI Reader", json_response=True)


@mcp.tool()
def search_papers(
    query: str,
    limit: int = 50,
    ids_only: bool = False,
    search_full_text: bool = True,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    sort_by: str = "relevance",
    skip: int = 0,
) -> list:
    """
    Search papers by keyword, arXiv ID, title, author, abstract, or AI summaries.
    Uses metadata cache (fast). search_full_text=True uses preview (~2k chars); for
    real full-text search use search_full_text tool.
    from_date, to_date: optional ISO date (e.g. 2024-01-01) to filter by published_date.
    sort_by: "relevance" (default) or "latest".
    """
    fetcher = _get_fetcher()
    return _do_search(query, fetcher, limit=limit, ids_only=ids_only, search_full_text=search_full_text,
                     search_generated_only=False, from_date=from_date, to_date=to_date, sort_by=sort_by, skip=skip)


@mcp.tool()
def search_generated_content(
    query: str,
    limit: int = 50,
    ids_only: bool = False,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    sort_by: str = "relevance",
    skip: int = 0,
) -> list:
    """
    Search only within AI-generated content: one_line_summary, detailed_summary, tags, extracted_keywords.
    Use when you want to find papers by AI analysis, not original text.
    from_date, to_date: optional ISO date. sort_by: "relevance" or "latest".
    """
    fetcher = _get_fetcher()
    return _do_search(query, fetcher, limit=limit, ids_only=ids_only, search_full_text=False,
                     search_generated_only=True, from_date=from_date, to_date=to_date, sort_by=sort_by, skip=skip)


def _do_search_full_text(q: str, fetcher, limit: int = 50, ids_only: bool = False, max_scan: int = 999999,
                         from_date: Optional[str] = None, to_date: Optional[str] = None,
                         sort_by: str = "relevance", skip: int = 0,
                         category: Optional[str] = None, starred_only: bool = False) -> list:
    """
    Search within actual full paper html_content.
    Uses store FTS when available (SQLite), else loads each paper from disk.
    """
    from_date_dt = _parse_date(from_date) if from_date else None
    to_date_dt = _parse_date(to_date) if to_date else None

    tab_filter = category or starred_only
    fetch_limit = limit + skip
    store = getattr(fetcher, "store", None)
    if store and hasattr(store, "search"):
        fts_limit = fetch_limit * 5 if tab_filter else fetch_limit
        fts_results = store.search(q.strip(), limit=fts_limit * 3 if (from_date_dt or to_date_dt) else fts_limit, search_full_text=True)
        if fts_results:
            filtered = [r for r in fts_results if _paper_in_date_range(r, from_date_dt, to_date_dt)]
            if tab_filter:
                meta_list = fetcher.list_papers_metadata(max_files=999999, check_stale=False)
                id_to_meta = {m["id"]: m for m in meta_list}
                filtered = [r for r in filtered if _meta_matches_tab(id_to_meta.get(r["id"], {}), category, starred_only)]
            if sort_by == "latest":
                from datetime import datetime, timezone
                def _dt_key(r):
                    return _parse_date(r.get("published_date", "")) or datetime.fromtimestamp(0, tz=timezone.utc)
                filtered.sort(key=_dt_key, reverse=True)
            else:
                filtered.sort(key=lambda x: x.get("search_score", 0), reverse=True)
            filtered = filtered[skip:skip + limit]
            if ids_only:
                return [{"id": r["id"], "search_score": r.get("search_score", 0)} for r in filtered]
            return filtered

    q_lower = q.lower()
    metadata_list = fetcher.list_papers_metadata(max_files=max_scan, check_stale=True)
    if tab_filter:
        metadata_list = [m for m in metadata_list if _meta_matches_tab(m, category, starred_only)]
    results = []

    for meta in metadata_list[:max_scan]:
        if meta.get('is_hidden', False):
            continue
        if not _paper_in_date_range(meta, from_date_dt, to_date_dt):
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

            item = {"id": paper.id, "search_score": score, "published_date": getattr(paper, "published_date", "") or ""}
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

    if sort_by == "latest":
        from datetime import datetime, timezone
        def _dt_key(x):
            return _parse_date(x.get("published_date", "")) or datetime.fromtimestamp(0, tz=timezone.utc)
        results.sort(key=_dt_key, reverse=True)
    else:
        results.sort(key=lambda x: x["search_score"], reverse=True)
    return results[skip:skip + limit]


@mcp.tool()
def search_full_text(
    query: str,
    limit: int = 50,
    ids_only: bool = False,
    max_scan: int = 999999,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    sort_by: str = "relevance",
    skip: int = 0,
) -> list:
    """
    Search within ACTUAL full paper text (html_content + abstract).
    Loads papers from disk - slower but searches entire content.
    from_date, to_date: optional ISO date. sort_by: "relevance" or "latest".
    """
    fetcher = _get_fetcher()
    return _do_search_full_text(query, fetcher, limit=limit, ids_only=ids_only, max_scan=max_scan,
                               from_date=from_date, to_date=to_date, sort_by=sort_by, skip=skip)


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
def get_paper_ids_by_query(
    query: str,
    limit: int = 50,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    sort_by: str = "relevance",
    skip: int = 0,
) -> list:
    """
    Search papers and return only arXiv IDs.
    from_date, to_date: optional ISO date. sort_by: "relevance" or "latest".
    """
    fetcher = _get_fetcher()
    results = _do_search(query, fetcher, limit=limit, ids_only=True,
                        from_date=from_date, to_date=to_date, sort_by=sort_by, skip=skip)
    return [r["id"] for r in results]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
