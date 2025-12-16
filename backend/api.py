"""
FastAPI backend - simple REST API.

Endpoints:
- GET /papers - list papers (timeline)
- GET /papers/{id} - get paper details
- POST /papers/{id}/ask - ask custom question
- GET /config - get config
- PUT /config - update config
- GET /search?q=query - search papers
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone
from dateutil import parser as date_parser
import json

from models import Paper, Config, QAPair
from fetcher import ArxivFetcher
from analyzer import DeepSeekAnalyzer
from default_config import DEFAULT_CONFIG


# Background task reference
background_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler - replaces deprecated on_event.
    Handles startup and shutdown.
    """
    # Startup
    config_path = Path("data/config.json")
    
    # Create default config if not exists
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = Config(**DEFAULT_CONFIG)
        config.save(config_path)
        print(f"✓ Created default config at {config_path}")
    
    # Initialize metadata cache (this happens in fetcher.__init__)
    # Cache is now ready for fast queries
    print("✓ Metadata cache initialized")
    
    # Start background fetcher (pending analysis handled there)
    global background_task
    background_task = asyncio.create_task(background_fetcher())
    print("🚀 Server ready - background tasks started")
    
    yield
    
    # Shutdown
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
    print("👋 Background fetcher stopped")


app = FastAPI(title="arXiv Paper Fetcher", lifespan=lifespan)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression - BUT exclude streaming endpoints to prevent buffering
# FastAPI's GZipMiddleware buffers StreamingResponse, which breaks SSE real-time delivery
# Solution: Only apply GZip to non-streaming endpoints
@app.middleware("http")
async def selective_gzip_middleware(request: Request, call_next):
    """Skip GZip for streaming endpoints to prevent buffering"""
    # Skip compression for streaming endpoints (critical for real-time SSE)
    if "/ask_stream" in str(request.url.path):
        response = await call_next(request)
        return response
    
    # For other endpoints, use standard GZip middleware behavior
    # But we need to apply it properly - use the middleware's dispatch method
    # Actually, the simplest: just bypass GZip for streaming, let others go through
    # We'll apply GZip via a wrapper that checks the response type
    
    response = await call_next(request)
    
    # Don't compress StreamingResponse - they need immediate delivery
    if isinstance(response, StreamingResponse):
        return response
    
    # For other responses, let GZip middleware handle it if configured
    # Since we're not using global GZipMiddleware, we rely on reverse proxy (nginx) for compression
    # The key fix: streaming endpoints never go through GZip
    return response

# Note: We don't add global GZipMiddleware because it buffers StreamingResponse
# Compression for non-streaming endpoints can be handled by reverse proxy (nginx)

# Global instances
fetcher = ArxivFetcher()
analyzer = DeepSeekAnalyzer()
config_path = Path("data/config.json")

# Serve frontend static files FIRST (before other routes)
# Try frontend_dist first (built assets), fallback to frontend (source)
frontend_dist = Path(__file__).parent.parent / "frontend_dist"
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_dist.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="static")
    frontend_path = frontend_dist  # Use dist for serving index.html too
elif frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# Request/Response models
class AskQuestionRequest(BaseModel):
    question: str
    parent_qa_id: Optional[int] = None  # For follow-up questions


class UpdateConfigRequest(BaseModel):
    filter_keywords: Optional[List[str]] = None
    negative_keywords: Optional[List[str]] = None
    preset_questions: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    fetch_interval: Optional[int] = None
    max_papers_per_fetch: Optional[int] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    concurrent_papers: Optional[int] = None
    min_relevance_score_for_stage2: Optional[float] = None


class UpdateRelevanceRequest(BaseModel):
    is_relevant: bool
    relevance_score: float


# ============ Frontend & Endpoints ============

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve frontend index.html"""
    # Try frontend_dist first, fallback to frontend
    frontend_dist = Path(__file__).parent.parent / "frontend_dist"
    frontend_source = Path(__file__).parent.parent / "frontend"
    
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        return FileResponse(str(frontend_dist / "index.html"))
    elif frontend_source.exists() and (frontend_source / "index.html").exists():
        return FileResponse(str(frontend_source / "index.html"))
    return {"message": "Frontend not found. Please check frontend directory."}


@app.get("/api/health")
async def health_check():
    """API health check"""
    return {"message": "arXiv Paper Fetcher API", "status": "running"}


@app.get("/papers", response_model=List[dict])
async def list_papers(skip: int = 0, limit: int = 20, sort_by: str = "relevance", keyword: str = None, starred_only: str = "false"):
    """
    List papers for timeline.
    sort_by: 'relevance' (default), 'latest'
    keyword: filter by keyword
    starred_only: 'true' to return only starred papers, 'false' to exclude starred papers
    
    PERFORMANCE OPTIMIZATION: Use metadata scanning first, then load only needed papers.
    """
    starred_only_bool = starred_only.lower() == 'true'
    
    # Step 1: Scan metadata first (fast - only reads minimal JSON fields)
    # This avoids loading all paper files into memory
    max_scan = 10000 if starred_only_bool else 2000  # Scan more for starred, less for normal
    metadata_list = fetcher.list_papers_metadata(max_files=max_scan, check_stale=True)
    
    # Debug: log metadata stats
    if starred_only_bool:
        total_starred = sum(1 for m in metadata_list if m.get('is_starred', False))
        total_hidden = sum(1 for m in metadata_list if m.get('is_hidden', False))
        print(f"[DEBUG] Total metadata: {len(metadata_list)}, starred: {total_starred}, hidden: {total_hidden}")
        # If no starred papers found, force refresh cache
        if total_starred == 0 and len(metadata_list) > 0:
            print("[DEBUG] No starred papers in cache, forcing cache refresh...")
            fetcher._refresh_metadata_cache()
            metadata_list = fetcher.list_papers_metadata(max_files=max_scan, check_stale=False)
            total_starred = sum(1 for m in metadata_list if m.get('is_starred', False))
            print(f"[DEBUG] After refresh: {len(metadata_list)} total, {total_starred} starred")
    
    # Step 2: Filter by starred/hidden status using metadata (no file loading)
    if starred_only_bool:
        filtered_metadata = [m for m in metadata_list if m.get('is_starred', False) and not m.get('is_hidden', False)]
        print(f"[DEBUG] Filtered starred metadata: {len(filtered_metadata)}")
    else:
        filtered_metadata = [m for m in metadata_list if not m.get('is_starred', False) and not m.get('is_hidden', False)]
    
    # Step 3: Filter by keyword if provided (still using metadata)
    if keyword:
        filtered_metadata = [
            m for m in filtered_metadata 
            if keyword.lower() in ' '.join(m.get('extracted_keywords', [])).lower()
        ]
    
    # Step 4: Sort by relevance or latest (using metadata)
    if sort_by == "relevance":
        filtered_metadata.sort(key=lambda m: (
            bool(m.get('detailed_summary', '') and m['detailed_summary'].strip()),
            m.get('relevance_score', 0.0)
        ), reverse=True)
    elif sort_by == "latest":
        def parse_date(date_str):
            """Parse date string to datetime object, normalize to UTC for comparison"""
            if not date_str or not isinstance(date_str, str):
                return None
            try:
                # Use dateutil.parser which handles all formats and timezones
                dt = date_parser.parse(date_str)
                # Normalize to UTC for consistent comparison
                # If datetime is naive (no timezone), assume UTC
                if dt.tzinfo is None:
                    # Naive datetime - assume UTC
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    # Convert to UTC
                    dt = dt.astimezone(timezone.utc)
                return dt
            except (ValueError, TypeError, AttributeError, OverflowError):
                return None
        
        def get_sort_date(m):
            """Get date for sorting: prefer published_date, fallback to created_at"""
            # Try published_date first
            pub_date = parse_date(m.get('published_date', ''))
            if pub_date:
                return pub_date
            
            # Fallback to created_at
            created_date = parse_date(m.get('created_at', ''))
            if created_date:
                return created_date
            
            # If both are invalid, use epoch (oldest) in UTC
            return datetime.fromtimestamp(0, tz=timezone.utc)
        
        filtered_metadata.sort(key=get_sort_date, reverse=True)
    
    # Step 5: Paginate (still using metadata)
    paginated_metadata = filtered_metadata[skip:skip + limit]
    
    # Step 6: NOW load only the papers we actually need (lazy loading)
    papers = []
    for meta in paginated_metadata:
        try:
            paper = fetcher.load_paper(meta['id'])
            papers.append(paper)
        except Exception as e:
            print(f"Warning: Failed to load paper {meta['id']}: {e}")
            continue
    
    # Return simplified data for timeline
    return [
        {
            "id": p.id,
            "title": p.title,
            "authors": p.authors,
            "abstract": p.abstract[:200] + "..." if len(p.abstract) > 200 else p.abstract,
            "url": p.url,
            "is_relevant": p.is_relevant,
            "relevance_score": p.relevance_score,
            "extracted_keywords": p.extracted_keywords,
            "one_line_summary": p.one_line_summary,
            "published_date": p.published_date,
            "is_starred": p.is_starred,
            "is_hidden": p.is_hidden,
            "created_at": p.created_at,
            "has_qa": len(p.qa_pairs) > 0,
            "detailed_summary": p.detailed_summary,  # For Stage 2 status detection
        }
        for p in papers
    ]


@app.get("/papers/{paper_id}", response_model=dict)
async def get_paper(paper_id: str):
    """
    Get full paper details including Q&A.
    """
    try:
        paper = fetcher.load_paper(paper_id)
        return paper.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/{paper_id}/ask")
async def ask_question(paper_id: str, request: AskQuestionRequest):
    """
    Ask a custom question about a paper.
    Uses KV cache for efficiency.
    """
    try:
        paper = fetcher.load_paper(paper_id)
        config = Config.load(config_path)
        
        answer = await analyzer.ask_custom_question(paper, request.question, config)
        
        return {
            "question": request.question,
            "answer": answer,
            "paper_id": paper_id
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/papers/{paper_id}/ask_stream")
async def ask_question_stream(paper_id: str, request: AskQuestionRequest):
    """
    Ask a custom question about a paper with streaming response.
    Uses Server-Sent Events (SSE) for real-time streaming.
    
    Supports:
    - Reasoning mode: prefix question with "think:" to use deepseek-reasoner
    - Follow-up: provide parent_qa_id to build conversation context
    """
    try:
        paper = fetcher.load_paper(paper_id)
        config = Config.load(config_path)
        
        async def event_generator():
            """Generate SSE events with streamed answer"""
            try:
                print(f"[Stream] Starting stream for paper {paper_id}, question: {request.question[:50]}...")
                chunk_count = 0
                last_yield_time = None
                
                async for chunk_data in analyzer.ask_custom_question_stream(
                    paper, 
                    request.question, 
                    config,
                    parent_qa_id=request.parent_qa_id
                ):
                    # chunk_data is now a dict: {"type": "thinking"/"content", "chunk": "..."}
                    chunk_count += 1
                    
                    if chunk_count <= 5 or chunk_count % 10 == 0:
                        import time
                        current_time = time.time()
                        time_since_last = current_time - last_yield_time if last_yield_time else 0
                        print(f"[Stream] Chunk {chunk_count}: type={chunk_data.get('type')}, len={len(chunk_data.get('chunk', ''))}, time_since_last={time_since_last:.3f}s")
                        last_yield_time = current_time
                    
                    # Yield immediately - don't buffer
                    sse_data = f"data: {json.dumps(chunk_data)}\n\n"
                    yield sse_data
                
                print(f"[Stream] Stream complete, total chunks: {chunk_count}")
                # Send completion event
                yield f"data: {json.dumps({'done': True})}\n\n"
            
            except Exception as e:
                import traceback
                error_msg = f"Stream error: {str(e)}\n{traceback.format_exc()}"
                print(f"[Stream] ERROR: {error_msg}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx/proxy buffering
                "Transfer-Encoding": "chunked",  # Explicit chunked encoding
            }
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current configuration"""
    config = Config.load(config_path)
    return config.to_dict()


@app.put("/config")
async def update_config(request: UpdateConfigRequest):
    """Update configuration - supports all config options"""
    config = Config.load(config_path)
    old_negative_keywords = set(config.negative_keywords or [])
    
    # Update all provided fields
    if request.filter_keywords is not None:
        config.filter_keywords = request.filter_keywords
    if request.negative_keywords is not None:
        config.negative_keywords = request.negative_keywords
    if request.preset_questions is not None:
        config.preset_questions = request.preset_questions
    if request.system_prompt is not None:
        config.system_prompt = request.system_prompt
    if request.fetch_interval is not None:
        config.fetch_interval = max(60, request.fetch_interval)  # Minimum 60 seconds
    if request.max_papers_per_fetch is not None:
        config.max_papers_per_fetch = max(1, min(500, request.max_papers_per_fetch))  # 1-500 range
    if request.model is not None:
        config.model = request.model
    if request.temperature is not None:
        config.temperature = max(0.0, min(2.0, request.temperature))  # 0-2 range
    if request.max_tokens is not None:
        config.max_tokens = max(100, min(8000, request.max_tokens))  # 100-8000 range
    if request.concurrent_papers is not None:
        config.concurrent_papers = max(1, min(50, request.concurrent_papers))  # 1-50 range
    if request.min_relevance_score_for_stage2 is not None:
        config.min_relevance_score_for_stage2 = max(0.0, min(10.0, request.min_relevance_score_for_stage2))  # 0-10 range
    
    config.save(config_path)
    
    # Check if negative keywords changed
    new_negative_keywords = set(config.negative_keywords or [])
    if new_negative_keywords != old_negative_keywords:
        # Trigger background task to recheck papers with new negative keywords
        asyncio.create_task(recheck_negative_keywords(config, new_negative_keywords))
        return {
            "message": "Config updated. Re-checking papers with new negative keywords in background...",
            "config": config.to_dict()
        }
    
    return {"message": "Config updated", "config": config.to_dict()}


@app.get("/search")
async def search_papers(q: str, limit: int = 50):
    """
    Search papers by keyword, full-text, or arXiv ID.
    If query looks like arXiv ID (e.g., 2510.09212), fetch if not exists.
    """
    import re
    
    # Check if query is an arXiv ID (format: YYMM.NNNNN or YYMM.NNNNNvN)
    arxiv_id_pattern = r'^\d{4}\.\d{4,5}(v\d+)?$'
    if re.match(arxiv_id_pattern, q.strip()):
        arxiv_id = q.strip()
        print(f"🔍 Detected arXiv ID: {arxiv_id}")
        
        try:
            # Fetch or load the paper
            paper = await fetcher.fetch_single_paper(arxiv_id)
            
            # Trigger analysis in background
            config = Config.load(config_path)
            
            # Check if Stage 1 is needed (is_relevant is None)
            needs_stage1 = paper.is_relevant is None
            
            # Check if Stage 2 is needed (is_relevant=True, score>=min, but no detailed_summary)
            min_score = getattr(config, 'min_relevance_score_for_stage2', 6.0)
            needs_stage2 = (
                paper.is_relevant is True 
                and paper.relevance_score >= min_score
                and (not paper.detailed_summary or paper.detailed_summary.strip() == '')
            )
            
            if needs_stage1 or needs_stage2:
                if needs_stage1:
                    print(f"📊 Started background Stage 1+2 analysis for {arxiv_id}")
                    asyncio.create_task(analyzer.process_papers([paper], config))
                else:
                    # Only Stage 2 needed
                    print(f"📚 Started background Stage 2 analysis for {arxiv_id}")
                    asyncio.create_task(analyzer.process_papers([paper], config, skip_stage1=True))
            
            # Return the paper
            return [{
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                "url": paper.url,
                "is_relevant": paper.is_relevant,
                "relevance_score": paper.relevance_score,
                "extracted_keywords": paper.extracted_keywords,
                "one_line_summary": paper.one_line_summary,
                "published_date": paper.published_date,
                "is_starred": paper.is_starred,
                "is_hidden": paper.is_hidden,
                "created_at": paper.created_at,
                "has_qa": len(paper.qa_pairs) > 0,
                "detailed_summary": paper.detailed_summary,
                "search_score": 1000,  # High score for direct ID match
            }]
        
        except Exception as e:
            print(f"✗ Failed to fetch arXiv paper {arxiv_id}: {e}")
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found on arXiv")
    
    # Normal keyword search
    papers = fetcher.list_papers(limit=1000)
    q_lower = q.lower()
    
    results = []
    for paper in papers:
        # Skip hidden papers
        if paper.is_hidden:
            continue
            
        # Search in title, abstract, keywords, summary
        searchable = (
            f"{paper.title} {paper.abstract} "
            f"{' '.join(paper.extracted_keywords)} {paper.one_line_summary}"
        ).lower()
        
        if q_lower in searchable:
            results.append({
                "id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract,
                "url": paper.url,
                "is_relevant": paper.is_relevant,
                "relevance_score": paper.relevance_score,
                "extracted_keywords": paper.extracted_keywords,
                "one_line_summary": paper.one_line_summary,
                "published_date": paper.published_date,
                "is_starred": paper.is_starred,
                "is_hidden": paper.is_hidden,
                "created_at": paper.created_at,
                "has_qa": len(paper.qa_pairs) > 0,
                "detailed_summary": paper.detailed_summary,  # For Stage 2 status detection
                "search_score": searchable.count(q_lower),  # Simple scoring
            })
    
    # Sort by search score
    results.sort(key=lambda x: x["search_score"], reverse=True)
    
    return results[:limit]


@app.post("/fetch")
async def trigger_fetch():
    """
    Manually trigger paper fetching.
    Fetch and analysis run in background (non-blocking).
    """
    async def fetch_and_analyze():
        try:
            config = Config.load(config_path)
            print(f"\n📡 Manual fetch triggered...")
            papers = await fetcher.fetch_latest(config.max_papers_per_fetch)
            if papers:
                print(f"✓ Fetched {len(papers)} papers, starting analysis...")
                await analyzer.process_papers(papers, config)
                print(f"✓ Manual fetch and analysis complete")
            else:
                print(f"✓ No new papers found")
        except Exception as e:
            print(f"✗ Manual fetch error: {e}")
            import traceback
            traceback.print_exc()
    
    # Start task in background
    asyncio.create_task(fetch_and_analyze())
    
    return {"message": "Fetch triggered", "status": "running"}


@app.post("/papers/{paper_id}/hide")
async def hide_paper(paper_id: str):
    """Hide a paper"""
    try:
        paper = fetcher.load_paper(paper_id)
        paper.is_hidden = True
        fetcher.save_paper(paper)
        return {"message": "Paper hidden", "paper_id": paper_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/{paper_id}/unhide")
async def unhide_paper(paper_id: str):
    """Unhide a paper"""
    try:
        paper = fetcher.load_paper(paper_id)
        paper.is_hidden = False
        fetcher.save_paper(paper)
        return {"message": "Paper unhidden", "paper_id": paper_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/{paper_id}/star")
async def star_paper(paper_id: str):
    """Star a paper"""
    try:
        paper = fetcher.load_paper(paper_id)
        paper.is_starred = not paper.is_starred  # Toggle
        fetcher.save_paper(paper)  # This updates cache automatically
        print(f"[DEBUG] Starred paper {paper_id}: is_starred={paper.is_starred}")
        return {"message": "论文已收藏" if paper.is_starred else "取消收藏", "is_starred": paper.is_starred}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/{paper_id}/update_relevance")
async def update_relevance(paper_id: str, request: UpdateRelevanceRequest):
    """Update paper relevance status and score manually"""
    try:
        paper = fetcher.load_paper(paper_id)
        paper.is_relevant = request.is_relevant
        paper.relevance_score = max(0, min(10, request.relevance_score))  # Clamp 0-10
        paper.updated_at = datetime.now().isoformat()
        fetcher.save_paper(paper)
        return {
            "message": "论文相关性已更新",
            "is_relevant": paper.is_relevant,
            "relevance_score": paper.relevance_score
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")




# ============ Background Tasks ============

async def recheck_negative_keywords(config: Config, negative_keywords: set):
    """
    Re-check papers with new negative keywords.
    Update papers that match new negative keywords to be marked as not relevant.
    Only check papers with relevance_score >= min_relevance_score_for_stage2.
    """
    try:
        min_score = getattr(config, 'min_relevance_score_for_stage2', 6.0)
        all_papers = fetcher.list_papers(limit=10000)
        
        # Filter papers: is_relevant=True and score >= threshold
        relevant_papers = [
            p for p in all_papers 
            if p.is_relevant is True and p.relevance_score >= min_score
        ]
        
        if not relevant_papers:
            print("✓ No papers to recheck")
            return
        
        print(f"\n🔍 Rechecking {len(relevant_papers)} papers with new negative keywords: {negative_keywords}")
        
        updated_count = 0
        for paper in relevant_papers:
            # Check if paper matches any negative keyword
            searchable_text = f"{paper.title} {paper.preview_text}".lower()
            
            for neg_kw in negative_keywords:
                if neg_kw.lower() in searchable_text:
                    # Update paper to not relevant
                    paper.is_relevant = False
                    paper.relevance_score = 1.0
                    paper.extracted_keywords = [f"❌ {neg_kw}"] + paper.extracted_keywords
                    paper.one_line_summary = f"论文包含负面关键词「{neg_kw}」，自动标记为不相关"
                    fetcher.save_paper(paper)
                    updated_count += 1
                    print(f"  ✗ Updated {paper.id}: matched negative keyword '{neg_kw}'")
                    break  # Only need to match one negative keyword
        
        print(f"✓ Recheck complete: {updated_count} papers updated to not relevant")
    
    except Exception as e:
        print(f"✗ Error rechecking negative keywords: {e}")
        import traceback
        traceback.print_exc()


async def check_pending_deep_analysis():
    """
    Check for papers marked as relevant but lacking deep analysis (Stage 2).
    Only process papers with score >= min_relevance_score_for_stage2.
    Process these papers with priority on startup.
    """
    try:
        config = Config.load(config_path)
        all_papers = fetcher.list_papers(limit=10000)
        
        min_score = getattr(config, 'min_relevance_score_for_stage2', 6.0)
        
        # Find papers: is_relevant=True, score >= min_score, but detailed_summary is empty
        pending_papers = [
            p for p in all_papers 
            if p.is_relevant is True 
            and p.relevance_score >= min_score
            and (not p.detailed_summary or p.detailed_summary.strip() == '')
        ]
        
        if pending_papers:
            print(f"\n🔍 Found {len(pending_papers)} papers pending deep analysis (score >= {min_score})")
            print(f"📚 Prioritizing deep analysis for these papers...")
            
            # Process with skip_stage1=True since they're already marked as relevant
            await analyzer.process_papers(pending_papers, config, skip_stage1=True)
            print(f"✓ Completed pending deep analysis for {len(pending_papers)} papers")
        else:
            print(f"✓ No pending deep analysis required (min score: {min_score})")
            
    except Exception as e:
        print(f"✗ Error checking pending deep analysis: {e}")
        import traceback
        traceback.print_exc()


async def analyze_papers_task(papers: List[Paper], config: Config):
    """
    Analyze papers in background (non-blocking).
    """
    try:
        print(f"📊 Starting analysis of {len(papers)} papers...")
        await analyzer.process_papers(papers, config)
        print(f"✓ Analysis complete")
    except Exception as e:
        print(f"✗ Analysis error: {e}")
        import traceback
        traceback.print_exc()


async def background_fetcher():
    """
    Background task: check pending analysis first, then fetch + analyze loop.
    Everything runs asynchronously (non-blocking).
    """
    # First run: check for pending deep analysis (non-blocking)
    asyncio.create_task(check_pending_deep_analysis())
    
    # Main fetch loop
    while True:
        try:
            config = Config.load(config_path)
            
            # Fetch new papers
            print(f"\n📡 Fetching papers... [{datetime.now().strftime('%H:%M:%S')}]")
            papers = await fetcher.fetch_latest(config.max_papers_per_fetch)
            
            # Start analysis asynchronously (don't wait)
            if papers:
                print(f"✓ Fetched {len(papers)} papers, starting analysis in background...")
                asyncio.create_task(analyze_papers_task(papers, config))
            else:
                print(f"✓ No new papers to analyze")
            
            # Sleep before next fetch (analysis runs in parallel)
            await asyncio.sleep(config.fetch_interval)
        
        except Exception as e:
            print(f"✗ Background fetcher error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)  # Wait 1 min on error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

