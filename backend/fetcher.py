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
from typing import List, Dict, Optional
import json
from datetime import datetime
import os
from threading import Lock

from models import Paper


class ArxivFetcher:
    """
    Fetches papers from arXiv.
    Simple and effective.
    """
    
    def __init__(self, data_dir: str = "data/papers"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State file to track last query time for each category
        self.state_file = self.data_dir.parent / "fetcher_state.json"
        
        # In-memory metadata cache: {paper_id: {metadata dict}}
        # Only stores lightweight metadata, not full Paper objects
        self._metadata_cache: Dict[str, dict] = {}
        self._cache_lock = Lock()  # Thread-safe cache updates
        self._cache_initialized = False
        
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
        
        # Initialize cache on startup
        self._refresh_metadata_cache()
    
    def _load_query_state(self) -> Dict[str, str]:
        """
        Load query state from file.
        Returns dict mapping category to last query end time (ISO format).
        """
        if not self.state_file.exists():
            return {}
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️  Failed to load query state: {e}")
            return {}
    
    def _save_query_state(self, state: Dict[str, str]):
        """
        Save query state to file.
        state: dict mapping category to last query end time (ISO format).
        """
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"  ⚠️  Failed to save query state: {e}")
    
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
                    
                    # Check response content
                    if not response.text or len(response.text) < 100:
                        print(f"  ⚠️  Empty or too short response for {category} (length: {len(response.text) if response.text else 0})")
                        continue
                    
                    feed = feedparser.parse(response.text)
                    
                    # Check for parsing errors
                    if hasattr(feed, 'bozo') and feed.bozo:
                        bozo_exception = getattr(feed, 'bozo_exception', None)
                        if bozo_exception:
                            print(f"  ⚠️  RSS parsing error for {category}: {bozo_exception}")
                        else:
                            print(f"  ⚠️  RSS parsing error for {category} (unknown)")
                    
                    # Get entries from feedparser
                    # feedparser always provides 'entries' as a list
                    total_entries = len(feed.entries) if hasattr(feed, 'entries') and feed.entries else 0
                    
                    # If no entries, try using arXiv API as fallback
                    if total_entries == 0:
                        # Check if it's a weekend (RSS feed is empty on weekends)
                        feed_title = getattr(feed.feed, 'title', 'Unknown') if hasattr(feed, 'feed') else 'Unknown'
                        feed_type = getattr(feed, 'version', 'Unknown')
                        
                        # Check if response actually contains item tags
                        if '<item>' in response.text or '<entry>' in response.text:
                            # Feed has items but feedparser didn't parse them
                            item_count = response.text.count('<item>') + response.text.count('<entry>')
                            print(f"  ⚠️  Feed has {item_count} items but feedparser found 0 (feed title: {feed_title}, type: {feed_type})")
                            print(f"     Response length: {len(response.text)}, preview: {response.text[:300]}...")
                        else:
                            # RSS feed is empty, try arXiv API as fallback
                            print(f"  ⚠️  No entries in RSS feed for {category}, trying arXiv API...")
                            try:
                                api_papers = await self._fetch_from_api(client, category, max_papers_per_category)
                                if api_papers:
                                    papers.extend(api_papers)
                                    print(f"     ✓ Fetched {len(api_papers)} papers from arXiv API")
                                else:
                                    print(f"     ⚠️  No new papers found in {category} (RSS empty, API also empty)")
                            except Exception as api_error:
                                print(f"     ✗ Failed to fetch from arXiv API: {api_error}")
                        continue
                    else:
                        print(f"     Found {total_entries} papers in RSS feed")
                    
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
                        
                        # Check if any version exists
                        exists, existing_id = self._any_version_exists(arxiv_id)
                        
                        if exists:
                            # Check if this is a newer version
                            if existing_id != arxiv_id:
                                # Compare versions: extract version numbers
                                existing_version = 0
                                new_version = 0
                                
                                if 'v' in existing_id:
                                    try:
                                        existing_version = int(existing_id.rsplit('v', 1)[1])
                                    except:
                                        pass
                                
                                if 'v' in arxiv_id:
                                    try:
                                        new_version = int(arxiv_id.rsplit('v', 1)[1])
                                    except:
                                        pass
                                
                                # If new version is newer, delete old version and save new one
                                if new_version > existing_version:
                                    old_file = self.data_dir / f"{existing_id}.json"
                                    if old_file.exists():
                                        old_file.unlink()
                                        print(f"     🔄 Replaced {existing_id} with newer version {arxiv_id}")
                                    # Continue to save new version below
                                else:
                                    # Old version is same or newer, skip
                                    continue
                            else:
                                # Exact match, skip
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
    
    async def _fetch_from_api(self, client: httpx.AsyncClient, category: str, max_papers: int = 100) -> List[Paper]:
        """
        Fetch latest papers from arXiv API when RSS feed is empty.
        Uses incremental query: queries one day at a time, moving backward if no papers found.
        Saves query state to avoid duplicate queries.
        """
        from urllib.parse import quote
        from datetime import datetime, timedelta
        
        papers = []
        max_days_back = 30  # Maximum days to search back
        days_searched = 0
        
        # Load query state
        query_state = self._load_query_state()
        
        # Get last query end time for this category
        # State format: {category: "end_time_iso"} - last successful query end time
        if category in query_state:
            try:
                last_end_time = datetime.fromisoformat(query_state[category])
            except:
                last_end_time = datetime.now()
        else:
            # First time querying this category, start from now
            last_end_time = datetime.now()
        
        # Start querying from last_end_time backward (one day at a time)
        current_end_time = last_end_time
        current_start_time = current_end_time - timedelta(days=1)
        
        # Continue querying backward until we find papers or reach max_days_back
        while days_searched < max_days_back:
            # Format dates for arXiv API (YYYYMMDDHHMMSS)
            start_date_str = current_start_time.strftime("%Y%m%d%H%M%S")
            end_date_str = current_end_time.strftime("%Y%m%d%H%M%S")
            
            # Build search query with date range
            search_query = f"cat:{category} AND submittedDate:[{start_date_str} TO {end_date_str}]"
            api_url = f"https://export.arxiv.org/api/query?search_query={quote(search_query)}&start=0&max_results={max_papers}&sortBy=submittedDate&sortOrder=descending"
            
            print(f"     Querying {category} from {current_start_time.strftime('%Y-%m-%d %H:%M')} to {current_end_time.strftime('%Y-%m-%d %H:%M')}")
            
            try:
                response = await client.get(api_url)
                if response.status_code != 200:
                    if response.status_code == 429:
                        # Rate limited, try simpler query without date range
                        print(f"     ⚠️  Rate limited, using simpler query...")
                        search_query = f"cat:{category}"
                        api_url = f"https://export.arxiv.org/api/query?search_query={quote(search_query)}&start=0&max_results={max_papers}&sortBy=submittedDate&sortOrder=descending"
                        response = await client.get(api_url)
                        if response.status_code != 200:
                            print(f"     ✗ arXiv API returned {response.status_code}")
                            break
                    else:
                        print(f"     ✗ arXiv API returned {response.status_code}")
                        break
                
                feed = feedparser.parse(response.text)
                
                if not feed.entries:
                    print(f"     No papers found in this time range, moving backward 1 day...")
                    days_searched += 1
                    # Move backward: next query will be one day earlier
                    current_end_time = current_start_time
                    current_start_time = current_end_time - timedelta(days=1)
                    continue
                
                print(f"     Found {len(feed.entries)} papers in API response")
                
                new_papers_in_batch = 0
                
                for entry in feed.entries:
                    # Extract arXiv ID from entry.id (format: "http://arxiv.org/abs/2510.08619v1")
                    if not hasattr(entry, 'id'):
                        continue
                    
                    arxiv_id = entry.id.split('/abs/')[-1] if '/abs/' in entry.id else entry.id.split('/')[-1]
                    
                    # Check if any version exists
                    exists, existing_id = self._any_version_exists(arxiv_id)
                    
                    if exists:
                        # Check if this is a newer version
                        if existing_id != arxiv_id:
                            # Compare versions: extract version numbers
                            existing_version = 0
                            new_version = 0
                            
                            if 'v' in existing_id:
                                try:
                                    existing_version = int(existing_id.rsplit('v', 1)[1])
                                except:
                                    pass
                            
                            if 'v' in arxiv_id:
                                try:
                                    new_version = int(arxiv_id.rsplit('v', 1)[1])
                                except:
                                    pass
                            
                            # If new version is newer, delete old version and save new one
                            if new_version > existing_version:
                                old_file = self.data_dir / f"{existing_id}.json"
                                if old_file.exists():
                                    old_file.unlink()
                                    print(f"     🔄 Replaced {existing_id} with newer version {arxiv_id}")
                                # Continue to save new version below
                            else:
                                # Old version is same or newer, skip
                                continue
                        else:
                            # Exact match, skip
                            continue
                    
                    new_papers_in_batch += 1
                    
                    # Download HTML version
                    html_content = await self._fetch_html(client, arxiv_id)
                    
                    # Extract preview text
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
                    
                    print(f"     ✓ {arxiv_id} - {paper.title[:60]}...")
                
                # Update query state: record the end time of this query
                # Next time we'll start from current_start_time (one day earlier)
                query_state[category] = current_start_time.isoformat()
                self._save_query_state(query_state)
                
                # If found new papers, we can stop searching backward
                if new_papers_in_batch > 0:
                    print(f"     Found {new_papers_in_batch} new papers, stopping backward search")
                    break
                else:
                    # No new papers in this batch, continue searching backward
                    days_searched += 1
                    current_end_time = current_start_time
                    current_start_time = current_end_time - timedelta(days=1)
                    continue
                
            except Exception as e:
                print(f"     ✗ Error fetching from arXiv API: {e}")
                # On error, move to next day and continue
                days_searched += 1
                current_end_time = current_start_time
                current_start_time = current_end_time - timedelta(days=1)
                continue
        
        # If we've searched max_days_back without finding papers, update state to prevent re-querying
        if days_searched >= max_days_back and len(papers) == 0:
            print(f"     Reached max search limit ({max_days_back} days), updating state")
            query_state[category] = current_start_time.isoformat()
            self._save_query_state(query_state)
        
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
    
    async def fetch_single_paper(self, arxiv_id: str) -> Paper:
        """
        Fetch a single paper by arXiv ID.
        Uses arXiv API to get metadata, then downloads HTML.
        Returns Paper object.
        Raises exception if paper not found or fetch fails.
        """
        # Check if already exists
        if self._paper_exists(arxiv_id):
            paper = self.load_paper(arxiv_id)
            # Update cache if paper exists
            if paper.id not in self._metadata_cache:
                # Paper exists but not in cache, refresh this entry
                file_path = self.data_dir / f"{arxiv_id}.json"
                if file_path.exists():
                    try:
                        with open(file_path) as f:
                            data = json.load(f)
                            with self._cache_lock:
                                self._metadata_cache[paper.id] = {
                                    'id': data.get('id', arxiv_id),
                                    'file_path': file_path,
                                    'mtime': file_path.stat().st_mtime,
                                    'title': data.get('title', ''),
                                    'is_starred': data.get('is_starred', False),
                                    'is_hidden': data.get('is_hidden', False),
                                    'relevance_score': data.get('relevance_score', 0.0),
                                    'published_date': data.get('published_date', ''),
                                    'created_at': data.get('created_at', ''),
                                    'extracted_keywords': data.get('extracted_keywords', []),
                                    'detailed_summary': data.get('detailed_summary', ''),
                                    'abstract': data.get('abstract', ''),
                                    'authors': data.get('authors', []),
                                    'tags': data.get('tags', []),
                                }
                    except Exception as e:
                        print(f"Warning: Failed to update cache for {arxiv_id}: {e}")
            return paper
        
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0, follow_redirects=True) as client:
            # Use arXiv API to get paper metadata
            api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
            
            try:
                response = await client.get(api_url)
                if response.status_code != 200:
                    raise Exception(f"arXiv API returned {response.status_code}")
                
                # Parse Atom feed
                feed = feedparser.parse(response.text)
                
                if not feed.entries or len(feed.entries) == 0:
                    raise Exception(f"Paper {arxiv_id} not found on arXiv")
                
                entry = feed.entries[0]
                
                # Download HTML version
                html_content = await self._fetch_html(client, arxiv_id)
                
                # Extract preview text
                preview_text = self._extract_preview(html_content, entry.summary)
                
                # Extract published date
                published_date = getattr(entry, 'published', '')
                
                # Create Paper object
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
                self.save_paper(paper)
                print(f"✓ Fetched single paper: {arxiv_id} - {paper.title[:60]}...")
                
                return paper
            
            except Exception as e:
                print(f"✗ Error fetching paper {arxiv_id}: {e}")
                raise
    
    def _get_base_id(self, arxiv_id: str) -> str:
        """
        Extract base ID from arXiv ID (remove version suffix).
        Example: '2510.08619v1' -> '2510.08619'
        """
        if 'v' in arxiv_id:
            return arxiv_id.rsplit('v', 1)[0]
        return arxiv_id
    
    def _paper_exists(self, arxiv_id: str) -> bool:
        """Check if paper already exists"""
        return (self.data_dir / f"{arxiv_id}.json").exists()
    
    def _any_version_exists(self, arxiv_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if any version of this paper exists.
        Returns (exists, existing_id) where existing_id is the full ID if exists.
        """
        base_id = self._get_base_id(arxiv_id)
        
        # Check exact match first
        if self._paper_exists(arxiv_id):
            return (True, arxiv_id)
        
        # Check for any version (v1, v2, v3, etc.)
        for version_file in self.data_dir.glob(f"{base_id}v*.json"):
            existing_id = version_file.stem
            return (True, existing_id)
        
        # Also check base ID without version
        if self._paper_exists(base_id):
            return (True, base_id)
        
        return (False, None)
    
    def save_paper(self, paper: Paper):
        """Save paper to JSON file and update metadata cache"""
        file_path = self.data_dir / f"{paper.id}.json"
        with open(file_path, 'w') as f:
            json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Update metadata cache immediately (thread-safe)
        with self._cache_lock:
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = 0
            self._metadata_cache[paper.id] = {
                'id': paper.id,
                'file_path': file_path,  # Path object, will be used in _refresh_stale_cache_entries
                'mtime': mtime,
                'title': paper.title,
                'is_starred': paper.is_starred,
                'is_hidden': paper.is_hidden,
                'relevance_score': paper.relevance_score,
                'published_date': paper.published_date,
                'created_at': paper.created_at,
                'extracted_keywords': paper.extracted_keywords,
                'detailed_summary': paper.detailed_summary,
                'abstract': paper.abstract,
                'authors': paper.authors,
                'tags': getattr(paper, 'tags', []),
            }
    
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
        """
        List papers with pagination.
        If limit is None or <= 0, load all papers.
        """
        paper_files = sorted(
            self.data_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        papers = []
        
        # If limit is None or <= 0, load all papers
        if limit is None or limit <= 0:
            file_range = paper_files[skip:]
        else:
            file_range = paper_files[skip:skip + limit]
        
        for file_path in file_range:
            try:
                with open(file_path) as f:
                    papers.append(Paper.from_dict(json.load(f)))
            except Exception as e:
                print(f"Warning: Failed to load paper {file_path.name}: {e}")
                continue
        
        return papers
    
    def _refresh_metadata_cache(self):
        """
        Refresh in-memory metadata cache by scanning all paper files.
        This is called on startup and can be called periodically to sync with disk.
        """
        print("🔄 Refreshing metadata cache...")
        paper_files = list(self.data_dir.glob("*.json"))
        
        new_cache = {}
        for file_path in paper_files:
            try:
                # Check file modification time
                mtime = file_path.stat().st_mtime
                paper_id = file_path.stem
                
                # If already in cache and file hasn't changed, reuse cache
                if paper_id in self._metadata_cache:
                    cached = self._metadata_cache[paper_id]
                    if cached.get('mtime') == mtime:
                        new_cache[paper_id] = cached
                        continue
                
                # Read metadata from file (only minimal fields)
                with open(file_path) as f:
                    data = json.load(f)
                    new_cache[paper_id] = {
                        'id': data.get('id', paper_id),
                        'file_path': file_path,
                        'mtime': mtime,
                        'title': data.get('title', ''),  # Cache title for search
                        'is_starred': data.get('is_starred', False),
                        'is_hidden': data.get('is_hidden', False),
                        'relevance_score': data.get('relevance_score', 0.0),
                        'published_date': data.get('published_date', ''),
                        'created_at': data.get('created_at', ''),
                        'extracted_keywords': data.get('extracted_keywords', []),
                        'detailed_summary': data.get('detailed_summary', ''),
                        'abstract': data.get('abstract', ''),  # Cache abstract for search
                        'authors': data.get('authors', []),  # Cache authors for search
                        'tags': data.get('tags', []),  # Cache tags for search
                    }
            except Exception as e:
                print(f"Warning: Failed to read metadata from {file_path.name}: {e}")
                continue
        
        with self._cache_lock:
            self._metadata_cache = new_cache
        
        print(f"✓ Metadata cache refreshed: {len(self._metadata_cache)} papers")
        self._cache_initialized = True
    
    def list_papers_metadata(self, max_files: int = 10000, check_stale: bool = True) -> List[dict]:
        """
        List paper metadata from in-memory cache (FAST - no file I/O).
        Returns list of dicts with: id, file_path, mtime, is_starred, is_hidden, etc.
        
        If cache not initialized, refresh it first.
        If check_stale=True, verify file modification times and refresh stale entries.
        """
        if not self._cache_initialized:
            print("[DEBUG] Cache not initialized, refreshing...")
            self._refresh_metadata_cache()
        
        # Check for stale cache entries (files modified outside of save_paper)
        if check_stale:
            self._refresh_stale_cache_entries()
        
        # Get all metadata from cache
        with self._cache_lock:
            metadata_list = list(self._metadata_cache.values())
        
        # Sort by mtime (newest first) and limit
        metadata_list.sort(key=lambda m: m.get('mtime', 0), reverse=True)
        result = metadata_list[:max_files]
        
        # Debug: log cache stats
        if len(result) > 0:
            sample = result[0]
            print(f"[DEBUG] Cache: {len(self._metadata_cache)} total, returning {len(result)}, sample keys: {list(sample.keys())}")
        
        return result
    
    def _refresh_stale_cache_entries(self):
        """
        Check cache entries against disk files and refresh stale ones.
        This handles cases where files are modified outside of save_paper().
        """
        stale_count = 0
        with self._cache_lock:
            for paper_id, cached_meta in list(self._metadata_cache.items()):
                file_path = cached_meta.get('file_path')
                # Handle both Path objects and string paths
                if isinstance(file_path, str):
                    file_path = Path(file_path)
                if not file_path or not file_path.exists():
                    continue
                
                try:
                    # Check if file was modified
                    current_mtime = file_path.stat().st_mtime
                    cached_mtime = cached_meta.get('mtime', 0)
                    
                    if current_mtime > cached_mtime:
                        # File was modified, refresh this entry
                        try:
                            with open(file_path) as f:
                                data = json.load(f)
                                self._metadata_cache[paper_id] = {
                                    'id': data.get('id', paper_id),
                                    'file_path': file_path,  # Keep as Path object
                                    'mtime': current_mtime,
                                    'title': data.get('title', ''),
                                    'is_starred': data.get('is_starred', False),
                                    'is_hidden': data.get('is_hidden', False),
                                    'relevance_score': data.get('relevance_score', 0.0),
                                    'published_date': data.get('published_date', ''),
                                    'created_at': data.get('created_at', ''),
                                    'extracted_keywords': data.get('extracted_keywords', []),
                                    'detailed_summary': data.get('detailed_summary', ''),
                                    'abstract': data.get('abstract', ''),
                                    'authors': data.get('authors', []),
                                    'tags': data.get('tags', []),
                                }
                                stale_count += 1
                        except Exception as e:
                            print(f"Warning: Failed to refresh stale cache entry {paper_id}: {e}")
                except OSError:
                    # File doesn't exist or can't be accessed
                    continue
        
        if stale_count > 0:
            print(f"🔄 Refreshed {stale_count} stale cache entries")


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

