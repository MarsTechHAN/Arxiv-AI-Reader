"""
Overlay paper_user_results onto base Paper for serving mode.
Base paper from papers.db; user-specific fields from serving.db paper_user_results.
"""

from typing import Optional
from models import Paper, QAPair

from .db import get_serving_db


def overlay_paper_from_dict(paper: Paper, overlay: dict) -> Paper:
    """Apply pre-fetched overlay dict (no DB call)."""
    if not overlay:
        return paper
    if overlay.get("is_relevant") is not None:
        paper.is_relevant = bool(overlay["is_relevant"])
    if overlay.get("relevance_score") is not None:
        paper.relevance_score = float(overlay["relevance_score"])
    if overlay.get("extracted_keywords") is not None:
        paper.extracted_keywords = overlay["extracted_keywords"]
    if overlay.get("one_line_summary"):
        paper.one_line_summary = overlay["one_line_summary"]
    if overlay.get("detailed_summary"):
        paper.detailed_summary = overlay["detailed_summary"]
    if overlay.get("tags") is not None:
        paper.tags = overlay["tags"]
    if overlay.get("qa_pairs"):
        paper.qa_pairs = [QAPair(**qa) if isinstance(qa, dict) else qa for qa in overlay["qa_pairs"]]
    if overlay.get("is_starred") is not None:
        paper.is_starred = bool(overlay["is_starred"])
    if overlay.get("is_hidden") is not None:
        paper.is_hidden = bool(overlay["is_hidden"])
    if overlay.get("star_category"):
        paper.star_category = overlay["star_category"]
    return paper


def overlay_paper(paper: Paper, user_id: int) -> Paper:
    """
    Overlay user-specific results onto paper. Returns paper with merged fields.
    """
    db = get_serving_db()
    r = db.get_paper_user_result(paper.id, user_id)
    if not r:
        return paper
    if r.get("is_relevant") is not None:
        paper.is_relevant = bool(r["is_relevant"])
    if r.get("relevance_score") is not None:
        paper.relevance_score = float(r["relevance_score"])
    if r.get("extracted_keywords") is not None:
        paper.extracted_keywords = r["extracted_keywords"]
    if r.get("one_line_summary"):
        paper.one_line_summary = r["one_line_summary"]
    if r.get("detailed_summary"):
        paper.detailed_summary = r["detailed_summary"]
    if r.get("tags") is not None:
        paper.tags = r["tags"]
    if r.get("qa_pairs"):
        paper.qa_pairs = [QAPair(**qa) if isinstance(qa, dict) else qa for qa in r["qa_pairs"]]
    if r.get("is_starred") is not None:
        paper.is_starred = bool(r["is_starred"])
    if r.get("is_hidden") is not None:
        paper.is_hidden = bool(r["is_hidden"])
    if r.get("star_category"):
        paper.star_category = r["star_category"]
    return paper


def save_paper_user_result_from_paper(paper: Paper, user_id: int) -> None:
    """Persist paper's user-relevant fields to paper_user_results."""
    db = get_serving_db()
    qa_json = None
    if paper.qa_pairs:
        from dataclasses import asdict
        qa_json = [asdict(qa) for qa in paper.qa_pairs]
    db.save_paper_user_result(
        paper_id=paper.id,
        user_id=user_id,
        is_relevant=paper.is_relevant,
        relevance_score=paper.relevance_score,
        extracted_keywords=paper.extracted_keywords,
        one_line_summary=paper.one_line_summary or "",
        detailed_summary=paper.detailed_summary or "",
        tags=paper.tags or [],
        qa_pairs=qa_json or [],
        is_starred=paper.is_starred,
        is_hidden=paper.is_hidden,
        star_category=paper.star_category or "Other",
    )
