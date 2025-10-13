"""
arXiv fetcher - dead simple, no bullshit.

Fetches latest papers every 5 minutes.
Downloads HTML version for DeepSeek analysis.
"""

import asyncio
import httpx
import feedparser
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List
import json
from datetime import datetime
import os

from models import Paper


class ArxivFetcher:
    """
    Fetches papers from arXiv.
    Simple and effective.
    """
    
    def __init__(self, data_dir: str = "data/papers"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # arXiv RSS feed URLs for different categories
        self.categories = [
            "cs.AI",  # Artificial Intelligence
            "cs.CV",  # Computer Vision
            "cs.LG",  # Machine Learning
            "cs.CL",  # Computation and Language
            "cs.NE",  # Neural and Evolutionary Computing
        ]
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
    
    async def fetch_latest(self, max_papers_per_category: int = 100) -> List[Paper]:
        """
        Fetch latest papers from arXiv.
        Always checks the most recent N papers (index 0 onwards) in each category.
        Skips papers that already exist locally.
        Returns list of Paper objects.
        """
        papers = []
        
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0, follow_redirects=True) as client:
            for category in self.categories:
                # IMPORTANT: Use HTTPS, not HTTP (http returns 301 redirect with empty body)
                rss_url = f"https://export.arxiv.org/rss/{category}"
                print(f"  📂 Fetching category: {category}")
                
                try:
                    response = await client.get(rss_url)
                    
                    if response.status_code != 200:
                        print(f"  ✗ HTTP {response.status_code} for {category}")
                        continue
                    
                    feed = feedparser.parse(response.text)
                    total_entries = len(feed.entries)
                    print(f"     Found {total_entries} papers in RSS feed")
                    
                    if total_entries == 0:
                        print(f"  ⚠️  No entries found in {category}")
                        continue
                    
                    # Always check from index 0 (latest papers)
                    check_count = min(max_papers_per_category, total_entries)
                    fetched_count = 0
                    
                    for idx in range(check_count):
                        entry = feed.entries[idx]
                        
                        # Extract arXiv ID
                        if not hasattr(entry, 'id'):
                            continue
                        
                        # RSS format: "oai:arXiv.org:2510.08619v1" or URL format
                        # API format: "http://arxiv.org/abs/2510.08619v1"
                        if 'oai:arXiv.org:' in entry.id:
                            arxiv_id = entry.id.split('oai:arXiv.org:')[-1]
                        else:
                            arxiv_id = entry.id.split('/abs/')[-1]
                        
                        # Skip if already exists
                        if self._paper_exists(arxiv_id):
                            continue
                        
                        # Download HTML version
                        html_content = await self._fetch_html(client, arxiv_id)
                        
                        # Extract preview text (first 2000 chars from abstract + intro)
                        preview_text = self._extract_preview(html_content, entry.summary)
                        
                        # Extract published date
                        published_date = getattr(entry, 'published', '')
                        
                        paper = Paper(
                            id=arxiv_id,
                            title=entry.title,
                            authors=self._extract_authors(entry),
                            abstract=entry.summary,
                            url=entry.link,
                            html_url=f"https://arxiv.org/html/{arxiv_id}",
                            html_content=html_content,
                            preview_text=preview_text,
                            published_date=published_date,
                        )
                        
                        # Save immediately
                        self._save_paper(paper)
                        papers.append(paper)
                        fetched_count += 1
                        
                        print(f"     ✓ {arxiv_id} - {paper.title[:60]}...")
                    
                    print(f"     Fetched {fetched_count} new papers (checked latest {check_count})")
                
                except Exception as e:
                    print(f"  ✗ Error fetching {category}: {e}")
                    continue
        
        return papers
    
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
    
    def _paper_exists(self, arxiv_id: str) -> bool:
        """Check if paper already exists"""
        return (self.data_dir / f"{arxiv_id}.json").exists()
    
    def save_paper(self, paper: Paper):
        """Save paper to JSON file"""
        file_path = self.data_dir / f"{paper.id}.json"
        with open(file_path, 'w') as f:
            json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Keep _save_paper for backward compatibility
    def _save_paper(self, paper: Paper):
        """Deprecated: use save_paper instead"""
        self.save_paper(paper)
    
    def load_paper(self, arxiv_id: str) -> Paper:
        """Load paper from JSON file"""
        file_path = self.data_dir / f"{arxiv_id}.json"
        with open(file_path) as f:
            return Paper.from_dict(json.load(f))
    
    def list_papers(self, skip: int = 0, limit: int = 20) -> List[Paper]:
        """List papers with pagination"""
        paper_files = sorted(
            self.data_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        papers = []
        for file_path in paper_files[skip:skip + limit]:
            with open(file_path) as f:
                papers.append(Paper.from_dict(json.load(f)))
        
        return papers


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

