"""
arXiv fetcher - uses arXiv API directly (no RSS).

Fetches latest papers via API. When no new papers, backfills older papers
at low rate to expand library over time.
Uses PaperStore for persistence (SQLite+FTS5+compression or JSON fallback).
"""

import asyncio
import httpx
import feedparser
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Optional, Callable, Awaitable
import json
from datetime import datetime
from urllib.parse import quote
import os

from models import Paper
from storage import get_paper_store

ARXIV_API_RATE_DELAY = 3.0  # sec between API calls (arXiv recommends 3s)
BACKFILL_BATCH_SIZE = 20


class ArxivFetcher:
    """
    Fetches papers from arXiv via API.
    Latest first; backfills older papers when no new ones.
    Uses PaperStore (SQLite or JSON) for persistence.
    """

    def __init__(self, data_dir: str = "data/papers", store=None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir.parent / "fetcher_state.json"
        self.store = store or get_paper_store(data_dir=str(self.data_dir), db_path=str(self.data_dir.parent / "papers.db"))
        self._backfill_category_idx = self._load_backfill_idx()
        
        self.categories = [
            "cs.AI",
            "cs.CV",
            "cs.LG",
            "cs.CL",
            "cs.NE",
        ]
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    
    def _load_backfill_idx(self) -> int:
        """Load persisted backfill rotation index."""
        if not self.state_file.exists():
            return 0
        try:
            with open(self.state_file, 'r') as f:
                raw = json.load(f)
            return int(raw.get("_backfill_category_idx", 0))
        except Exception:
            return 0

    def _load_query_state(self) -> dict:
        """Load state: {cat: {"backfill_start": int, "backfill_done": bool}}"""
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, 'r') as f:
                raw = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Failed to load query state: {e}")
            return {}
        state = {}
        for cat, val in (raw or {}).items():
            if cat == "_backfill_category_idx":
                continue
            if isinstance(val, dict) and "backfill_start" in val:
                state[cat] = val
            else:
                state[cat] = {"backfill_start": 100, "backfill_done": False}
        return state
    
    def _save_query_state(self, state: dict):
        try:
            out = {**state, "_backfill_category_idx": self._backfill_category_idx}
            with open(self.state_file, 'w') as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            print(f"  ⚠️  Failed to save query state: {e}")
    
    async def fetch_latest(
        self,
        max_papers_per_category: int = 100,
        on_new_paper: Optional[Callable[[Paper], Awaitable[None]]] = None,
    ) -> List[Paper]:
        """
        Fetch latest papers via arXiv API. When no new papers, backfill older
        papers at low rate to expand library.
        If on_new_paper is provided, call it for each new paper as soon as it's
        saved (allows downstream to start Stage 1 before fetch completes).
        """
        papers = []
        async with httpx.AsyncClient(headers=self.headers, timeout=60.0, follow_redirects=True) as client:
            for i, category in enumerate(self.categories):
                if i > 0:
                    await asyncio.sleep(ARXIV_API_RATE_DELAY)
                cat_papers = await self._fetch_latest_api(
                    client, category, max_papers_per_category, on_new_paper
                )
                papers.extend(cat_papers)
            
            if not papers:
                backfill_papers = await self._fetch_backfill_batch(
                    client, max_papers_per_category, on_new_paper
                )
                papers.extend(backfill_papers)
        
        return papers
    
    async def _fetch_latest_api(
        self,
        client: httpx.AsyncClient,
        category: str,
        max_papers: int,
        on_new_paper: Optional[Callable[[Paper], Awaitable[None]]] = None,
    ) -> List[Paper]:
        """Fetch latest papers from arXiv API (start=0, newest first)."""
        search_query = f"cat:{category}"
        api_url = (
            f"https://export.arxiv.org/api/query?search_query={quote(search_query)}"
            f"&start=0&max_results={max_papers}&sortBy=submittedDate&sortOrder=descending"
        )
        print(f"  📂 {category} (API latest)")
        papers, _ = await self._query_api_and_save(
            client, api_url, category, on_new_paper=on_new_paper
        )
        return papers
    
    async def _fetch_backfill_batch(
        self,
        client: httpx.AsyncClient,
        max_papers_per_category: int,
        on_new_paper: Optional[Callable[[Paper], Awaitable[None]]] = None,
    ) -> List[Paper]:
        """Fetch one batch of older papers when no new latest. Low rate."""
        state = self._load_query_state()
        candidates = [
            c for c in self.categories
            if not state.get(c, {}).get("backfill_done", False)
        ]
        if not candidates:
            print("  ✓ All categories backfilled")
            return []
        
        idx = self._backfill_category_idx % len(candidates)
        self._backfill_category_idx += 1
        category = candidates[idx]
        
        cat_state = state.setdefault(category, {"backfill_start": max_papers_per_category, "backfill_done": False})
        start = int(cat_state.get("backfill_start", max_papers_per_category))
        
        search_query = f"cat:{category}"
        api_url = (
            f"https://export.arxiv.org/api/query?search_query={quote(search_query)}"
            f"&start={start}&max_results={BACKFILL_BATCH_SIZE}&sortBy=submittedDate&sortOrder=descending"
        )
        
        print(f"  📂 {category} (backfill start={start})")
        await asyncio.sleep(ARXIV_API_RATE_DELAY)
        papers, has_more = await self._query_api_and_save(
            client, api_url, category, is_backfill=True, on_new_paper=on_new_paper
        )
        
        cat_state["backfill_start"] = start + BACKFILL_BATCH_SIZE
        if not has_more:
            cat_state["backfill_done"] = True
            print(f"     Reached end for {category}")
        self._save_query_state(state)
        return papers
    
    async def _query_api_and_save(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        category: str,
        is_backfill: bool = False,
        on_new_paper: Optional[Callable[[Paper], Awaitable[None]]] = None,
    ) -> tuple[List[Paper], bool]:
        """Query arXiv API and save new papers. Returns (papers, has_more)."""
        papers = []
        entries = []
        try:
            response = await client.get(api_url)
            if response.status_code != 200:
                print(f"     ✗ HTTP {response.status_code}")
                return [], False
            if not response.text or len(response.text) < 100:
                print(f"     ✗ Empty/short response")
                return [], False
            
            feed = feedparser.parse(response.text)
            entries = getattr(feed, "entries", []) or []
            print(f"     Got {len(entries)} entries")
            
            for entry in entries:
                if not hasattr(entry, "id"):
                    continue
                arxiv_id = (
                    entry.id.split("/abs/")[-1]
                    if "/abs/" in entry.id
                    else entry.id.split("oai:arXiv.org:")[-1] if "oai:arXiv.org:" in entry.id else entry.id.split("/")[-1]
                )
                
                exists, existing_id = self.store.any_version_exists(arxiv_id)
                if exists:
                    if existing_id != arxiv_id:
                        ev = pv = 0
                        try:
                            ev = int(existing_id.rsplit("v", 1)[1]) if "v" in existing_id else 0
                        except Exception:
                            pass
                        try:
                            pv = int(arxiv_id.rsplit("v", 1)[1]) if "v" in arxiv_id else 0
                        except Exception:
                            pass
                        if pv > ev:
                            self.store.delete_paper(existing_id)
                            print(f"     🔄 Replaced {existing_id} with {arxiv_id}")
                        else:
                            continue
                    else:
                        continue
                
                published_date = getattr(entry, "published", "")
                html_content = await self._fetch_html(client, arxiv_id)
                preview_text = self._extract_preview(html_content, entry.summary)
                link = getattr(entry, "link", f"https://arxiv.org/abs/{arxiv_id}")
                
                paper = Paper(
                    id=arxiv_id,
                    title=entry.title,
                    authors=self._extract_authors(entry),
                    abstract=entry.summary,
                    url=link,
                    html_url=f"https://arxiv.org/html/{arxiv_id}",
                    html_content=html_content,
                    preview_text=preview_text,
                    published_date=published_date,
                )
                self.store.save_paper(paper)
                papers.append(paper)
                print(f"     ✓ {arxiv_id} - {paper.title[:60]}...")
                if on_new_paper:
                    await on_new_paper(paper)
            
            if papers:
                print(f"     Saved {len(papers)} new papers")
        except Exception as e:
            print(f"     ✗ Error: {e}")
        has_more = len(entries) >= BACKFILL_BATCH_SIZE if is_backfill else True
        return papers, has_more
    
    async def _fetch_html(self, client: httpx.AsyncClient, arxiv_id: str) -> str:
        """
        Download HTML version of paper.
        Falls back to abstract if HTML not available.
        """
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        
        try:
            response = await client.get(html_url)
            if response.status_code == 200:
                # Extract main content
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Try to find main article content
                article = soup.find('article') or soup.find('div', {'id': 'main'})
                if article:
                    return article.get_text(separator='\n', strip=True)
                
                return soup.get_text(separator='\n', strip=True)
        
        except Exception as e:
            print(f"  Warning: Could not fetch HTML for {arxiv_id}: {e}")
        
        # Fallback: return empty, will use abstract
        return ""
    
    def _extract_preview(self, html_content: str, abstract: str) -> str:
        """
        Extract preview text (first 2000 chars).
        Priority: abstract + beginning of paper
        """
        if html_content:
            # Combine abstract and paper start
            preview = f"{abstract}\n\n{html_content[:1500]}"
        else:
            preview = abstract
        
        return preview[:2000]
    
    def _extract_authors(self, entry) -> List[str]:
        """Extract author names from RSS entry"""
        if hasattr(entry, 'authors'):
            return [author.name for author in entry.authors]
        elif hasattr(entry, 'author'):
            return [entry.author]
        return []
    
    async def fetch_single_paper(self, arxiv_id: str) -> Paper:
        """
        Fetch a single paper by arXiv ID.
        Uses arXiv API to get metadata, then downloads HTML.
        Treats 2602.08426, 2602.08426v1, etc. as same paper; returns latest version.
        Raises exception if paper not found or fetch fails.
        """
        exists, latest_id = self.store.any_version_exists(arxiv_id)
        if exists:
            return self.store.load_paper(latest_id, resolve_version=False)
        
        base_id = self.store._get_base_id(arxiv_id)
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0, follow_redirects=True) as client:
            # Use base_id to always fetch latest version from arXiv
            api_url = f"https://export.arxiv.org/api/query?id_list={base_id}"
            
            try:
                response = await client.get(api_url)
                if response.status_code != 200:
                    raise Exception(f"arXiv API returned {response.status_code}")
                
                # Parse Atom feed
                feed = feedparser.parse(response.text)
                
                if not feed.entries or len(feed.entries) == 0:
                    raise Exception(f"Paper {arxiv_id} not found on arXiv")
                
                entry = feed.entries[0]
                # Extract actual id from response (arXiv returns latest, e.g. 2602.08426v2)
                fetched_id = (
                    entry.id.split("/abs/")[-1]
                    if "/abs/" in entry.id
                    else entry.id.split("oai:arXiv.org:")[-1] if "oai:arXiv.org:" in entry.id else entry.id.split("/")[-1]
                )
                # Download HTML version
                html_content = await self._fetch_html(client, fetched_id)
                
                # Extract preview text
                preview_text = self._extract_preview(html_content, entry.summary)
                
                # Extract published date
                published_date = getattr(entry, 'published', '')
                
                # Create Paper object (use fetched_id from API response = latest version)
                paper = Paper(
                    id=fetched_id,
                    title=entry.title,
                    authors=self._extract_authors(entry),
                    abstract=entry.summary,
                    url=entry.link,
                    html_url=f"https://arxiv.org/html/{fetched_id}",
                    html_content=html_content,
                    preview_text=preview_text,
                    published_date=published_date,
                )
                
                self.store.save_paper(paper)
                print(f"✓ Fetched single paper: {fetched_id} - {paper.title[:60]}...")
                
                return paper
            
            except Exception as e:
                print(f"✗ Error fetching paper {arxiv_id}: {e}")
                raise
    
    def _paper_exists(self, arxiv_id: str) -> bool:
        """Check if any version of paper exists"""
        return self.store.any_version_exists(arxiv_id)[0]
    
    def save_paper(self, paper: Paper):
        """Save paper to store"""
        self.store.save_paper(paper)

    def load_paper(self, arxiv_id: str, resolve_version: bool = True) -> Paper:
        """Load paper from store. resolve_version=True: 2602.08426 loads 2602.08426v2 if that's latest."""
        return self.store.load_paper(arxiv_id, resolve_version=resolve_version)
    
    def list_papers(self, skip: int = 0, limit: int = 20) -> List[Paper]:
        """List papers with pagination. limit<=0 loads all."""
        return self.store.list_papers(skip=skip, limit=limit if (limit and limit > 0) else None)
    
    def list_papers_metadata(self, max_files: int = 10000, check_stale: bool = True) -> List[dict]:
        """List paper metadata from store."""
        return self.store.list_papers_metadata(max_files=max_files, check_stale=check_stale)

    def _refresh_metadata_cache(self) -> None:
        """Force refresh store's metadata cache."""
        self.store.refresh_metadata_cache()


async def run_fetcher_loop(interval: int = 300):
    """
    Run fetcher in a loop.
    Simple: while True + sleep. No framework bullshit.
    """
    fetcher = ArxivFetcher()
    
    print(f"🚀 Starting arXiv fetcher (every {interval}s)...")
    
    while True:
        try:
            print(f"\n📡 Fetching latest papers... [{datetime.now().strftime('%H:%M:%S')}]")
            papers = await fetcher.fetch_latest()
            print(f"✓ Fetched {len(papers)} new papers")
        
        except Exception as e:
            print(f"✗ Fetcher error: {e}")
        
        await asyncio.sleep(interval)


if __name__ == "__main__":
    # Test the fetcher
    asyncio.run(run_fetcher_loop())

