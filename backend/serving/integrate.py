"""
Serving mode integration. Minimal api.py changes via helpers and app extension.
Enable: ARXIV_SERVING_MODE=1
"""

import asyncio
import os
from contextvars import ContextVar
from typing import Optional, Tuple
from starlette.requests import Request
from models import Config, Paper
from pathlib import Path

SERVING_MODE = os.environ.get("ARXIV_SERVING_MODE", "").lower() in ("1", "true", "yes")
_current_user_id: ContextVar[Optional[int]] = ContextVar("serving_user_id", default=None)


def set_serving_user_id(user_id: Optional[int]) -> None:
    _current_user_id.set(user_id)


def get_serving_user_id() -> Optional[int]:
    return _current_user_id.get(None)


from storage import DATA_ROOT
config_path = DATA_ROOT / "config.json"


def get_user_and_config(request: Request, fallback_config_path: Path = None) -> Tuple[Optional[int], Config]:
    """Get (user_id, config) for request. In non-serving mode, user_id is always None. Sync - use get_user_and_config_async in async handlers."""
    if not SERVING_MODE:
        return None, Config.load(str(fallback_config_path or config_path))
    from .middleware import get_current_user_id
    from .config_resolver import get_config_for_user
    user_id = get_current_user_id(request)
    return user_id, get_config_for_user(user_id, fallback_config_path or config_path)


async def get_user_and_config_async(request: Request, fallback_config_path: Path = None) -> Tuple[Optional[int], Config]:
    """Async wrapper - runs get_user_and_config in thread pool to avoid blocking."""
    return await asyncio.to_thread(get_user_and_config, request, fallback_config_path or config_path)


def overlay_paper_for_user(paper: Paper, user_id: Optional[int]) -> Paper:
    """Overlay user results on paper when serving mode."""
    if not SERVING_MODE or user_id is None:
        return paper
    from .paper_overlay import overlay_paper
    return overlay_paper(paper, user_id)


def save_paper_for_user(paper: Paper, user_id: Optional[int]) -> None:
    """Save paper user results when serving mode."""
    if not SERVING_MODE or user_id is None:
        return
    from .paper_overlay import save_paper_user_result_from_paper
    save_paper_user_result_from_paper(paper, user_id)


def ensure_one_line_tasks(papers: list, user_id: Optional[int], config: Config, analyzer, fetcher, overlays: Optional[dict] = None) -> list:
    """
    Return list of asyncio tasks to run stage1 for papers missing one_line for user.
    If overlays dict is provided (e.g. from get_user_paper_overlays), use it to avoid sync DB calls.
    """
    import asyncio
    if not SERVING_MODE or user_id is None:
        return []
    tasks = []
    for paper in papers:
        if overlays is not None:
            r = overlays.get(paper.id, {})
        else:
            from .db import get_serving_db
            r = get_serving_db().get_paper_user_result(paper.id, user_id) or {}
        if r and r.get("one_line_summary"):
            continue
        if paper.one_line_summary:
            continue
        p = paper  # capture for closure

        async def _run(pp=p):
            set_serving_user_id(user_id)
            try:
                analyzed = await analyzer.stage1_filter(pp, config)
                await asyncio.to_thread(save_paper_for_user, analyzed, user_id)
                await asyncio.to_thread(fetcher.save_paper, analyzed)
            finally:
                set_serving_user_id(None)

        tasks.append(asyncio.create_task(_run()))
    return tasks


def should_run_full_summary(paper: Paper, user_id: Optional[int], config: Config, user_requested: bool) -> bool:
    """Full summary when: (1) new + matches keywords, or (2) user clicked."""
    if user_requested:
        return True
    if not SERVING_MODE:
        return bool(paper.is_relevant) and (
            not (paper.detailed_summary or "").strip() or
            (config.preset_questions and len([q for q in (paper.qa_pairs or []) if q.question in config.preset_questions]) < len(config.preset_questions))
        )
    from .db import get_serving_db
    db = get_serving_db()
    r = db.get_paper_user_result(paper.id, user_id) if user_id else None
    is_rel = r.get("is_relevant") if r else paper.is_relevant
    has_sum = (r.get("detailed_summary") if r else paper.detailed_summary or "")
    return bool(is_rel) and not (has_sum and has_sum.strip())
