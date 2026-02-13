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
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
from dateutil import parser as date_parser
import json

from models import Paper, Config, QAPair
from fetcher import ArxivFetcher
from storage import DATA_ROOT
from analyzer import DeepSeekAnalyzer
from default_config import DEFAULT_CONFIG

# Scan all papers (no limit)
SCAN_ALL = 999999

# Background task reference
background_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler - replaces deprecated on_event.
    Handles startup and shutdown.
    """
    # Startup
    config_path = DATA_ROOT / "config.json"

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

# Serving mode: TOTP auth, multi-user (ARXIV_SERVING_MODE=1)
try:
    from serving.integrate import SERVING_MODE
    if SERVING_MODE:
        from serving.db import get_serving_db
        from serving.middleware import ServingAuthMiddleware
        from serving.views import router as auth_router, get_login_router
        get_serving_db()
        app.add_middleware(ServingAuthMiddleware)
        app.include_router(auth_router)
        app.include_router(get_login_router())
        print("✓ Serving mode enabled - TOTP auth, multi-user")
except ImportError:
    pass

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
    if "/ask_stream" in str(request.url.path) or "/search/ai/stream" in str(request.url.path):
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

# Global instances (analyzer uses fetcher.save_paper so writes go to SQLite when enabled)
fetcher = ArxivFetcher()

def _save_paper_sync(paper):
    """Sync save - run via asyncio.to_thread to avoid blocking event loop."""
    fetcher.save_paper(paper)
    try:
        from serving.integrate import SERVING_MODE, get_serving_user_id, save_paper_for_user
        if SERVING_MODE:
            uid = get_serving_user_id()
            if uid is not None:
                save_paper_for_user(paper, uid)
    except ImportError:
        pass

def _save_paper_cb(paper):
    """Non-blocking save: schedule in thread pool, return immediately."""
    asyncio.create_task(asyncio.to_thread(_save_paper_sync, paper))

analyzer = DeepSeekAnalyzer(save_paper=_save_paper_cb)
config_path = DATA_ROOT / "config.json"

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
    star_categories: Optional[List[str]] = None
    mcp_search_url: Optional[str] = None


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
async def list_papers(request: Request, skip: int = 0, limit: int = 20, sort_by: str = "relevance", keyword: str = None, starred_only: str = "false", category: str = None):
    """
    List papers for timeline.
    sort_by: 'relevance' (default), 'latest'
    keyword: filter by keyword
    starred_only: 'true' to return only starred papers (optionally filtered by category)
    category: when starred_only=true, filter by star_category (e.g. '高效视频生成', 'Other')
    When starred_only=false: show ALL papers (star is just categorization, starred papers remain in main list)
    
    PERFORMANCE OPTIMIZATION: Use metadata scanning first, then load only needed papers.
    All sync I/O runs in thread pool to avoid blocking event loop.
    """
    starred_only_bool = starred_only.lower() == 'true'
    user_id, config = None, None
    try:
        from serving.integrate import get_user_and_config_async, overlay_paper_for_user, ensure_one_line_tasks
        user_id, config = await get_user_and_config_async(request, config_path)
    except ImportError:
        user_id, config = None, await asyncio.to_thread(Config.load, config_path)

    # Step 1: Scan all metadata (no limit)
    metadata_list = await asyncio.to_thread(fetcher.list_papers_metadata, SCAN_ALL, True)
    overlays = {}
    if user_id:
        try:
            from serving.db import get_serving_db
            overlays = await asyncio.to_thread(get_serving_db().get_user_paper_overlays, user_id)
            for m in metadata_list:
                o = overlays.get(m.get("id"), {})
                if o:
                    m.update(o)
        except Exception:
            pass
    
    # Debug: log metadata stats
    if starred_only_bool:
        total_starred = sum(1 for m in metadata_list if m.get('is_starred', False))
        total_hidden = sum(1 for m in metadata_list if m.get('is_hidden', False))
        print(f"[DEBUG] Total metadata: {len(metadata_list)}, starred: {total_starred}, hidden: {total_hidden}")
        if total_starred == 0 and len(metadata_list) > 0:
            await asyncio.to_thread(fetcher._refresh_metadata_cache)
            metadata_list = await asyncio.to_thread(fetcher.list_papers_metadata, SCAN_ALL, False)
            total_starred = sum(1 for m in metadata_list if m.get('is_starred', False))
            print(f"[DEBUG] After refresh: {len(metadata_list)} total, {total_starred} starred")
    
    # Step 2: Filter - star is just categorization; main list shows ALL non-hidden papers
    if starred_only_bool:
        filtered_metadata = [m for m in metadata_list if m.get('is_starred', False) and not m.get('is_hidden', False)]
        if category:
            filtered_metadata = [m for m in filtered_metadata if m.get('star_category', 'Other') == category]
        print(f"[DEBUG] Filtered starred metadata: {len(filtered_metadata)} (category={category})")
    else:
        # Main list: show ALL papers (starred + non-starred), only exclude hidden
        filtered_metadata = [m for m in metadata_list if not m.get('is_hidden', False)]
    
    # Step 3: Filter by keyword if provided (match title, abstract, keywords, etc.)
    if keyword:
        kw_lower = keyword.lower()
        def _keyword_match(m):
            searchable = ' '.join([
                m.get('title', ''),
                m.get('abstract', ''),
                m.get('one_line_summary', ''),
                m.get('detailed_summary', ''),
                ' '.join(m.get('extracted_keywords', [])),
                ' '.join(m.get('tags', [])),
            ]).lower()
            return kw_lower in searchable
        filtered_metadata = [m for m in filtered_metadata if _keyword_match(m)]
    
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
    
    # Step 6: Load papers in parallel (non-blocking)
    async def _load_one(meta):
        try:
            paper = await asyncio.to_thread(fetcher.load_paper, meta['id'])
            if user_id and overlays:
                try:
                    from serving.paper_overlay import overlay_paper_from_dict
                    paper = overlay_paper_from_dict(paper, overlays.get(meta['id'], {}))
                except Exception:
                    pass
            return paper
        except Exception as e:
            print(f"Warning: Failed to load paper {meta['id']}: {e}")
            return None

    loaded = await asyncio.gather(*[_load_one(m) for m in paginated_metadata])
    papers = [p for p in loaded if p is not None]

    if user_id:
        try:
            for t in ensure_one_line_tasks(papers, user_id, config, analyzer, fetcher, overlays=overlays):
                asyncio.create_task(t)
        except (ImportError, NameError):
            pass

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
            "star_category": getattr(p, 'star_category', 'Other'),
            "created_at": p.created_at,
            "has_qa": len(p.qa_pairs) > 0,
            "detailed_summary": p.detailed_summary,
            "tags": getattr(p, 'tags', []),
            "stage2_pending": _stage2_status(p, config)[1],
        }
        for p in papers
    ]


def _stage2_status(paper: Paper, config: Config) -> tuple:
    """Returns (needs_stage2, stage2_pending)."""
    min_score = getattr(config, 'min_relevance_score_for_stage2', 6.0)
    preset = getattr(config, 'preset_questions', []) or []
    if not paper.is_relevant or paper.relevance_score < min_score:
        return False, False
    has_summary = bool(paper.detailed_summary and paper.detailed_summary.strip())
    preset_answered = sum(1 for qa in (paper.qa_pairs or []) if qa.question in preset)
    needs = not has_summary or preset_answered < len(preset)
    return needs, needs


@app.get("/papers/{paper_id}", response_model=dict)
async def get_paper(request: Request, paper_id: str):
    """Get full paper details including Q&A."""
    try:
        user_id, config = None, None
        try:
            from serving.integrate import get_user_and_config_async, overlay_paper_for_user
            user_id, config = await get_user_and_config_async(request, config_path)
        except ImportError:
            config = await asyncio.to_thread(Config.load, config_path)
        paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
        if user_id:
            paper = overlay_paper_for_user(paper, user_id)
        d = paper.to_dict()
        _, d["stage2_pending"] = _stage2_status(paper, config)
        return d
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/{paper_id}/request_full_summary")
async def request_full_summary(request: Request, paper_id: str):
    """User clicked to request full summary. Triggers Stage 2 on demand (serving mode)."""
    user_id, config = None, None
    try:
        from serving.integrate import get_user_and_config_async, overlay_paper_for_user, set_serving_user_id
        user_id, config = await get_user_and_config_async(request, config_path)
    except ImportError:
        config = await asyncio.to_thread(Config.load, config_path)
    try:
        paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
        if user_id:
            paper = overlay_paper_for_user(paper, user_id)
            set_serving_user_id(user_id)
        if paper.is_relevant is None:
            paper = await analyzer.stage1_filter(paper, config)
            if user_id:
                try:
                    from serving.paper_overlay import save_paper_user_result_from_paper
                    await asyncio.to_thread(save_paper_user_result_from_paper, paper, user_id)
                except ImportError:
                    pass
            await asyncio.to_thread(fetcher.save_paper, paper)
        if paper.is_relevant and (not paper.detailed_summary or not paper.detailed_summary.strip()):
            await analyzer.stage2_qa(paper, config)
            if user_id:
                try:
                    from serving.paper_overlay import save_paper_user_result_from_paper
                    await asyncio.to_thread(save_paper_user_result_from_paper, paper, user_id)
                except ImportError:
                    pass
            await asyncio.to_thread(fetcher.save_paper, paper)
        return {"ok": True, "paper_id": paper_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/papers/{paper_id}/ask")
async def ask_question(http_request: Request, paper_id: str, request: AskQuestionRequest):
    """Ask a custom question about a paper. Uses KV cache for efficiency."""
    try:
        user_id, config = None, None
        try:
            from serving.integrate import get_user_and_config_async, overlay_paper_for_user
            user_id, config = await get_user_and_config_async(http_request, config_path)
        except ImportError:
            config = await asyncio.to_thread(Config.load, config_path)
        paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
        if user_id:
            paper = overlay_paper_for_user(paper, user_id)
        
        answer = await analyzer.ask_custom_question(paper, request.question, config, fetcher=fetcher)
        
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
async def ask_question_stream(http_request: Request, paper_id: str, request: AskQuestionRequest):
    """Ask a custom question with SSE streaming. Supports think: prefix and follow-up."""
    try:
        user_id, config = None, None
        try:
            from serving.integrate import get_user_and_config_async, overlay_paper_for_user
            user_id, config = await get_user_and_config_async(http_request, config_path)
        except ImportError:
            config = await asyncio.to_thread(Config.load, config_path)
        paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
        if user_id:
            paper = overlay_paper_for_user(paper, user_id)
        
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
                    parent_qa_id=request.parent_qa_id,
                    fetcher=fetcher,
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
async def get_config(request: Request):
    """Get current configuration"""
    try:
        from serving.integrate import get_user_and_config_async
        _, config = await get_user_and_config_async(request, config_path)
    except ImportError:
        config = await asyncio.to_thread(Config.load, config_path)
    return config.to_dict()


@app.put("/config")
async def update_config(http_request: Request, request: UpdateConfigRequest):
    """Update configuration - supports all config options. All DB/file I/O runs in thread pool."""
    user_id, config = None, None
    try:
        from serving.integrate import get_user_and_config_async
        from serving.db import get_serving_db
        user_id, config = await get_user_and_config_async(http_request, config_path)
    except ImportError:
        config = await asyncio.to_thread(Config.load, config_path)
    if user_id:
        cfg = await asyncio.to_thread(get_serving_db().get_user_config, user_id)
        config = cfg if cfg else Config(**DEFAULT_CONFIG)
    else:
        config = await asyncio.to_thread(Config.load, config_path)

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
        config.min_relevance_score_for_stage2 = max(0.0, min(10.0, request.min_relevance_score_for_stage2))
    if request.star_categories is not None:
        old_categories = set(config.star_categories or [])
        config.star_categories = request.star_categories
        new_categories = set(config.star_categories)
        if old_categories != new_categories:
            asyncio.create_task(reclassify_all_starred_papers(config))
    if request.mcp_search_url is not None:
        config.mcp_search_url = request.mcp_search_url.strip() or None

    if user_id:
        try:
            from serving.db import get_serving_db
            await asyncio.to_thread(get_serving_db().save_user_config, user_id, config)
        except ImportError:
            await asyncio.to_thread(config.save, config_path)
    else:
        await asyncio.to_thread(config.save, config_path)
    return {"message": "Config updated", "config": config.to_dict()}


def _parse_pdf_to_paper(file_content: bytes, filename: str) -> Paper:
    """Extract text from PDF and create Paper object. ID prefix: local_"""
    from pypdf import PdfReader
    from io import BytesIO
    import uuid
    
    reader = PdfReader(BytesIO(file_content))
    full_text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            full_text += t + "\n"
    
    if not full_text.strip():
        raise ValueError("PDF contains no extractable text")
    
    lines = [ln.strip() for ln in full_text.split("\n") if ln.strip()]
    title = Path(filename).stem.replace("_", " ") if filename else "Uploaded Paper"
    if lines:
        first_line = lines[0]
        if len(first_line) > 10 and len(first_line) < 300 and not first_line.isdigit():
            title = first_line
    
    abstract = full_text[:1500] if len(full_text) > 1500 else full_text
    preview_text = f"{abstract}\n\n{full_text[1500:3500]}"[:2000] if len(full_text) > 1500 else full_text[:2000]
    
    paper_id = f"local_{uuid.uuid4().hex[:12]}"
    return Paper(
        id=paper_id,
        title=title,
        authors=[],
        abstract=abstract,
        url="",
        html_url="",
        html_content=full_text,
        preview_text=preview_text,
        published_date="",
    )


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract text, create Paper, and trigger analysis.
    Returns the created paper for immediate display.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    try:
        content = await file.read()
        if len(content) < 100:
            raise HTTPException(status_code=400, detail="PDF file is too small or empty")
        
        paper = _parse_pdf_to_paper(content, file.filename or "paper.pdf")
        await asyncio.to_thread(fetcher.save_paper, paper)

        config = await asyncio.to_thread(Config.load, config_path)
        asyncio.create_task(analyzer.process_papers([paper], config))
        
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
        }]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")


def _query_has_cjk(text: str) -> bool:
    """True if text contains CJK (Chinese/Japanese/Korean) characters."""
    if not text:
        return False
    for c in text:
        if '\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af':
            return True
    return False


def _calculate_similarity(query: str, text: str) -> float:
    """Tokenized BM25-like scoring via search_utils."""
    from search_utils import score_text
    return score_text(query, text, exact_phrase_bonus=0.3)


def _score_substring_only(query: str, meta: dict) -> float:
    """Fast scoring for CJK: substring match on title, authors, abstract, AI summary only (no full text)."""
    if not query:
        return 0.0
    q = query.strip().lower()
    if not q:
        return 0.0
    score = 0.0
    title = (meta.get("title") or "").lower()
    abstract = (meta.get("abstract") or "").lower()
    detailed = (meta.get("detailed_summary") or "").lower()
    one_line = (meta.get("one_line_summary") or "").lower()
    authors_text = " ".join(meta.get("authors") or []).lower()
    tags_text = " ".join((meta.get("tags") or []) + (meta.get("extracted_keywords") or [])).lower()
    if q in title:
        score += 2.0
    if q in abstract:
        score += 1.0
    if q in detailed:
        score += 1.5
    if q in one_line:
        score += 1.2
    if q in authors_text:
        score += 1.2
    if q in tags_text:
        score += 1.2
    return score


def _mcp_format_search_result(p: dict, include_all: bool = True) -> dict:
    """Format search result for AI (minimal tokens). Include papers even without summaries."""
    return {
        "id": p.get("id"),
        "title": p.get("title", ""),
        "search_score": round(p.get("search_score", 0), 2),
        "one_line_summary": p.get("one_line_summary", "")[:200],
        "detailed_summary": (p.get("detailed_summary", "") or "")[:300],
    }


async def _mcp_tool_executor(fetcher, name: str, args: dict,
                            from_date: str = None, to_date: str = None, sort_by: str = "relevance",
                            category: str = None, starred_only: bool = False) -> Union[dict, list]:
    """Execute MCP search tools. Returns minimal format for AI, or full dict for get_paper."""
    from mcp_server import _do_search, _do_search_full_text

    q = args.get("query", "")
    limit = min(int(args.get("limit", 25)), 30)
    skip_val = max(0, int(args.get("skip", 0)))
    fd = args.get("from_date") or from_date
    td = args.get("to_date") or to_date
    sb = args.get("sort_by") or sort_by

    if name == "search_papers":
        raw = await asyncio.to_thread(
            _do_search, q, fetcher, limit, False, True, False, fd, td, sb, skip_val,
            category=category, starred_only=starred_only
        )
        return [_mcp_format_search_result(p) for p in raw]
    elif name == "search_generated_content":
        raw = await asyncio.to_thread(
            _do_search, q, fetcher, limit, False, False, True, fd, td, sb, skip_val,
            category=category, starred_only=starred_only
        )
        return [_mcp_format_search_result(p) for p in raw]
    elif name == "search_full_text":
        max_scan = min(max(0, int(args.get("max_scan", SCAN_ALL))), SCAN_ALL)
        raw = await asyncio.to_thread(
            _do_search_full_text, q, fetcher, limit, False, max_scan, fd, td, sb, skip_val,
            category=category, starred_only=starred_only
        )
        return [_mcp_format_search_result(p) for p in raw]
    elif name == "get_paper_ids_by_query":
        raw = await asyncio.to_thread(
            _do_search, q, fetcher, min(limit, 30), True, True, False, fd, td, sb, skip_val,
            category=category, starred_only=starred_only
        )
        return [r["id"] for r in raw]
    elif name == "get_paper":
        arxiv_id = (args.get("arxiv_id", "") or "").strip()
        if not arxiv_id:
            return {"error": "arxiv_id required"}
        try:
            if await asyncio.to_thread(fetcher._paper_exists, arxiv_id):
                paper = await asyncio.to_thread(fetcher.load_paper, arxiv_id)
            else:
                paper = await fetcher.fetch_single_paper(arxiv_id)
        except Exception as e:
            return {"error": str(e), "arxiv_id": arxiv_id}
        return {
            "id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": (paper.abstract or "")[:400],
            "one_line_summary": paper.one_line_summary,
            "detailed_summary": (paper.detailed_summary or "")[:500],
            "tags": getattr(paper, "tags", []),
            "extracted_keywords": paper.extracted_keywords,
        }
    return []


@app.get("/search/ai/stream")
async def search_ai_stream(q: str, limit: int = 50, from_date: str = None, to_date: str = None, sort_by: str = "relevance",
                           category: str = None, starred_only: str = "false"):
    """AI search with streaming progress. category+starred_only restrict to tab content."""
    config = await asyncio.to_thread(Config.load, config_path)
    inner_q = q.strip()
    for prefix in ("ai:", "ai："):
        if inner_q.lower().startswith(prefix):
            inner_q = inner_q[len(prefix):].strip()
            break

    if not inner_q:
        async def empty_gen():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Empty query'})}\n\n"
        return StreamingResponse(empty_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    starred_only_bool = starred_only.lower() == "true"
    async def tool_executor(name, args):
        return await _mcp_tool_executor(fetcher, name, args, from_date=from_date, to_date=to_date, sort_by=sort_by,
                                        category=category, starred_only=starred_only_bool)

    async def event_gen():
        try:
            yield f"data: {json.dumps({'type': 'progress', 'message': 'AI 搜索中...'})}\n\n"
            progress_queue = asyncio.Queue()

            async def on_progress(msg):
                await progress_queue.put(msg)

            async def run_ai():
                return await analyzer.ai_search_with_mcp_tools(
                    inner_q, tool_executor, config, limit=limit, on_progress=on_progress
                )

            def to_event(msg):
                if isinstance(msg, dict) and "type" in msg:
                    return msg
                return {"type": "progress", "message": str(msg)}

            task = asyncio.create_task(run_ai())
            while not task.done():
                try:
                    msg = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                    yield f"data: {json.dumps(to_event(msg))}\n\n"
                except asyncio.TimeoutError:
                    pass
            while not progress_queue.empty():
                try:
                    msg = progress_queue.get_nowait()
                    yield f"data: {json.dumps(to_event(msg))}\n\n"
                except asyncio.QueueEmpty:
                    break
            task_result = await task
            if isinstance(task_result, tuple):
                final_ids, _, _ = task_result
            else:
                final_ids = task_result or []

            yield f"data: {json.dumps({'type': 'progress', 'message': '加载结果中...'})}\n\n"

            results = []
            _tab_filter = category or starred_only_bool
            for pid in final_ids:
                try:
                    full = await asyncio.to_thread(fetcher.load_paper, pid)
                    if _tab_filter and not _paper_matches_tab_filter(full, category, starred_only_bool):
                        continue
                    results.append({
                        "id": full.id,
                        "title": full.title,
                        "authors": full.authors,
                        "abstract": (full.abstract[:200] + "..." if len(full.abstract) > 200 else full.abstract),
                        "url": full.url,
                        "is_relevant": full.is_relevant,
                        "relevance_score": full.relevance_score,
                        "extracted_keywords": full.extracted_keywords,
                        "one_line_summary": full.one_line_summary,
                        "published_date": full.published_date,
                        "is_starred": full.is_starred,
                        "is_hidden": full.is_hidden,
                        "created_at": full.created_at,
                        "has_qa": len(full.qa_pairs) > 0,
                        "detailed_summary": full.detailed_summary,
                        "tags": getattr(full, "tags", []),
                        "search_score": len(final_ids) - final_ids.index(pid) if pid in final_ids else 0,
                        "stage2_pending": _stage2_status(full, config)[1],
                    })
                except Exception:
                    continue
            if sort_by == "latest":
                results.sort(key=lambda x: (_parse_sort_date(x.get("published_date", "")) or datetime.fromtimestamp(0, tz=timezone.utc),), reverse=True)
            else:
                results.sort(key=lambda x: (x.get("is_relevant") is True, x.get("search_score", 0)), reverse=True)
            yield f"data: {json.dumps({'type': 'done', 'results': results})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _run_ai_search(inner_q: str, limit: int, config, fetcher, from_date: str = None, to_date: str = None, sort_by: str = "relevance",
                         category: str = None, starred_only: bool = False):
    """Core AI search logic. Returns list of paper dicts."""

    async def tool_executor_sync(name, args):
        return await _mcp_tool_executor(fetcher, name, args, from_date=from_date, to_date=to_date, sort_by=sort_by,
                                       category=category, starred_only=starred_only)

    task_result = await analyzer.ai_search_with_mcp_tools(
        inner_q, tool_executor_sync, config, limit=limit, on_progress=None
    )
    if isinstance(task_result, tuple):
        final_ids, _, _ = task_result
    else:
        final_ids = task_result or []
    results = []
    _tab_filter = category or starred_only
    for pid in final_ids:
        try:
            full = await asyncio.to_thread(fetcher.load_paper, pid)
            if _tab_filter and not _paper_matches_tab_filter(full, category, starred_only):
                continue
            results.append({
                "id": full.id,
                "title": full.title,
                "authors": full.authors,
                "abstract": (full.abstract[:200] + "..." if len(full.abstract) > 200 else full.abstract),
                "url": full.url,
                "is_relevant": full.is_relevant,
                "relevance_score": full.relevance_score,
                "extracted_keywords": full.extracted_keywords,
                "one_line_summary": full.one_line_summary,
                "published_date": full.published_date,
                "is_starred": full.is_starred,
                "is_hidden": full.is_hidden,
                "created_at": full.created_at,
                "has_qa": len(full.qa_pairs) > 0,
                "detailed_summary": full.detailed_summary,
                "tags": getattr(full, "tags", []),
                "search_score": len(final_ids) - final_ids.index(pid) if pid in final_ids else 0,
                "stage2_pending": _stage2_status(full, config)[1],
            })
        except Exception:
            continue
    if sort_by == "latest":
        results.sort(key=lambda x: (_parse_sort_date(x.get("published_date", "")) or datetime.fromtimestamp(0, tz=timezone.utc),), reverse=True)
    else:
        results.sort(key=lambda x: (x.get("is_relevant") is True, x.get("search_score", 0)), reverse=True)
    return results


@app.get("/search/ai")
async def search_ai_nostream(q: str, limit: int = 50, from_date: str = None, to_date: str = None, sort_by: str = "relevance",
                             category: str = None, starred_only: str = "false"):
    """Non-streaming AI search. category+starred_only restrict to tab content."""
    config = await asyncio.to_thread(Config.load, config_path)
    inner_q = q.strip()
    for prefix in ("ai:", "ai："):
        if inner_q.lower().startswith(prefix):
            inner_q = inner_q[len(prefix):].strip()
            break
    if not inner_q:
        return []
    results = await _run_ai_search(inner_q, limit, config, fetcher, from_date=from_date, to_date=to_date, sort_by=sort_by,
                                   category=category, starred_only=starred_only.lower() == "true")
    return results


def _parse_sort_date(date_str):
    """Parse date string to datetime (UTC) for sorting."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        dt = date_parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except (ValueError, TypeError, AttributeError, OverflowError):
        return None


def _relevance_sort_rank(is_relevant) -> int:
    """0=relevant, 1=pending, 2=not_relevant. Not-relevant goes last."""
    return 0 if is_relevant is True else (1 if is_relevant is None else 2)


def _paper_matches_tab_filter(paper_or_meta, category: str = None, starred_only: bool = False) -> bool:
    """Check if paper/meta matches tab filter (category tab = starred + star_category)."""
    if not category and not starred_only:
        return True
    is_starred = paper_or_meta.get("is_starred", False) if isinstance(paper_or_meta, dict) else getattr(paper_or_meta, "is_starred", False)
    if not is_starred:
        return False
    if category:
        sc = paper_or_meta.get("star_category", "Other") if isinstance(paper_or_meta, dict) else getattr(paper_or_meta, "star_category", "Other")
        if sc != category:
            return False
    return True


@app.get("/search")
async def search_papers(q: str, limit: int = 50, skip: int = 0, sort_by: str = "relevance", category: str = None, starred_only: str = "false"):
    """Search papers by keyword. Searches ALL papers. Returns [skip:skip+limit]. category+starred_only restrict to tab content."""
    config = await asyncio.to_thread(Config.load, config_path)
    q = q.strip()

    # Check if query is an arXiv ID (format: YYMM.NNNNN or YYMM.NNNNNvN)
    arxiv_id_pattern = r'^\d{4}\.\d{4,5}(v\d+)?$'
    if re.match(arxiv_id_pattern, q.strip()):
        arxiv_id = q.strip()
        print(f"🔍 Detected arXiv ID: {arxiv_id}")
        
        try:
            # Fetch or load the paper
            paper = await fetcher.fetch_single_paper(arxiv_id)
            
            # Trigger analysis in background
            config = await asyncio.to_thread(Config.load, config_path)
            
            # Check if Stage 1 is needed (is_relevant is None)
            needs_stage1 = paper.is_relevant is None
            
            # Stage 2 needed if relevant, score>=min, and (no summary or incomplete preset Q&As)
            needs_stage2, _ = _stage2_status(paper, config)
            
            if needs_stage1 or needs_stage2:
                if needs_stage1:
                    print(f"📊 Started background Stage 1+2 analysis for {arxiv_id}")
                    asyncio.create_task(analyzer.process_papers([paper], config))
                else:
                    # Only Stage 2 needed
                    print(f"📚 Started background Stage 2 analysis for {arxiv_id}")
                    asyncio.create_task(analyzer.process_papers([paper], config, skip_stage1=True))
            
            # Apply tab filter
            if category or starred_only.lower() == "true":
                if not _paper_matches_tab_filter(paper, category, starred_only.lower() == "true"):
                    return []

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
                "stage2_pending": _stage2_status(paper, config)[1],
                "search_score": 1000.0,  # High score for direct ID match
            }]
        
        except Exception as e:
            print(f"✗ Failed to fetch arXiv paper {arxiv_id}: {e}")
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found on arXiv")
    
    starred_only_bool = starred_only.lower() == "true"
    tab_filter = category or starred_only_bool

    # Try store FTS first (SQLite). Skip FTS for CJK - FTS5 unicode61 doesn't match Chinese.
    store = getattr(fetcher, "store", None)
    use_fts = store and hasattr(store, "search") and not _query_has_cjk(q)
    if use_fts:
        # Request enough FTS results for pagination; search full corpus
        fts_limit = max(limit * 100, 1000, skip + limit)
        fts_results = await asyncio.to_thread(store.search, q, fts_limit, True)
        if fts_results:
            config = await asyncio.to_thread(Config.load, config_path)
            q_lower = q.strip().lower()
            # Use metadata for title boost and sort key (avoid loading all papers)
            metadata_list = await asyncio.to_thread(fetcher.list_papers_metadata, SCAN_ALL, False)
            id_to_meta = {m.get("id"): m for m in metadata_list}
            # Prioritize title matches (FTS bm25 underweights title)
            def _parse_dt(s):
                if not s:
                    return datetime.fromtimestamp(0, tz=timezone.utc)
                try:
                    dt = date_parser.parse(s)
                    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
                except Exception:
                    return datetime.fromtimestamp(0, tz=timezone.utc)
            scored = []
            for r in fts_results:
                if tab_filter:
                    meta = id_to_meta.get(r["id"], {})
                    if not _paper_matches_tab_filter(meta, category, starred_only_bool):
                        continue
                meta = id_to_meta.get(r["id"], {})
                score = r.get("search_score", 0)
                title_match = bool(q_lower and q_lower in (meta.get("title") or "").lower())
                if title_match:
                    score += 10.0
                pub_dt = _parse_dt(meta.get("published_date", ""))
                # Sort: relevant first, pending, not_relevant last; then title_match, date/score
                ir_rank = _relevance_sort_rank(meta.get("is_relevant"))
                if sort_by == "latest":
                    sort_key = (ir_rank, 0 if title_match else 1, -pub_dt.timestamp(), -score)
                else:
                    sort_key = (ir_rank, 0 if title_match else 1, -pub_dt.timestamp(), -score)
                scored.append((sort_key, score, r["id"]))
            scored.sort(key=lambda x: x[0])
            # Paginate: [skip:skip+limit]
            results = []
            for _sort_key, score, pid in scored[skip:skip + limit]:
                try:
                    paper = await asyncio.to_thread(fetcher.load_paper, pid)
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
                        "detailed_summary": paper.detailed_summary,
                        "tags": getattr(paper, "tags", []),
                        "stage2_pending": _stage2_status(paper, config)[1],
                        "search_score": score,
                    })
                except Exception as e:
                    print(f"Warning: Failed to load paper {pid}: {e}")
            # Re-sort: not_relevant last; then by date or score
            if sort_by == "latest":
                def _get_result_date(r):
                    d = _parse_sort_date(r.get("published_date", ""))
                    return d or _parse_sort_date(r.get("created_at", "")) or datetime.fromtimestamp(0, tz=timezone.utc)
                results.sort(key=lambda r: (_relevance_sort_rank(r.get("is_relevant")), -_get_result_date(r).timestamp()))
            else:
                results.sort(key=lambda r: (_relevance_sort_rank(r.get("is_relevant")), -r.get("search_score", 0)))
            return results

    # Fallback: metadata scan (JSON store or FTS returned empty). Optimized: collect scores first, load only top N.
    config = await asyncio.to_thread(Config.load, config_path)
    metadata_list = await asyncio.to_thread(fetcher.list_papers_metadata, SCAN_ALL, True)
    if tab_filter:
        metadata_list = [m for m in metadata_list if _paper_matches_tab_filter(m, category, starred_only_bool)]

    is_cjk = _query_has_cjk(q)
    scored: List[tuple] = []  # (paper_id, score, meta for sort key)

    for meta in metadata_list:
        if meta.get('is_hidden', False):
            continue
        paper_id = meta.get('id', '')
        if is_cjk:
            total_score = _score_substring_only(q, meta)
        else:
            title = meta.get('title', '')
            abstract = meta.get('abstract', '')
            detailed_summary = meta.get('detailed_summary', '')
            one_line_summary = meta.get('one_line_summary', '')
            authors_text = ' '.join(meta.get('authors') or [])
            tags_text = ' '.join((meta.get('tags') or []) + (meta.get('extracted_keywords') or []))
            title_score = _calculate_similarity(q, title) * 2.0 if title else 0.0
            abstract_score = _calculate_similarity(q, abstract) if abstract else 0.0
            summary_score = _calculate_similarity(q, detailed_summary) * 1.5 if detailed_summary else 0.0
            one_line_score = _calculate_similarity(q, one_line_summary) * 1.2 if one_line_summary else 0.0
            author_score = _calculate_similarity(q, authors_text) * 1.2 if authors_text else 0.0
            tag_score = _calculate_similarity(q, tags_text) * 1.2 if tags_text else 0.0
            q_lower = q.lower()
            substring_bonus = 1.5 if (title and q_lower in title.lower()) else (0.8 if (abstract and q_lower in abstract.lower()) else 0.0)
            total_score = title_score + abstract_score + summary_score + one_line_score + author_score + tag_score + substring_bonus

        if total_score > 0:
            pub_dt = _parse_sort_date(meta.get('published_date', '')) or _parse_sort_date(meta.get('created_at', '')) or datetime.fromtimestamp(0, tz=timezone.utc)
            scored.append((paper_id, total_score, meta.get('is_relevant'), pub_dt))

    # Sort: relevant first, pending, not_relevant last; then score, then date. Paginate [skip:skip+limit].
    scored.sort(key=lambda x: (_relevance_sort_rank(x[2]), -x[1], -(x[3].timestamp() if hasattr(x[3], 'timestamp') else 0)))
    top_scored = scored[skip:skip + limit]

    results = []
    for paper_id, total_score, _ir, _pub in top_scored:
        try:
            paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
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
                "detailed_summary": paper.detailed_summary,
                "tags": getattr(paper, 'tags', []),
                "stage2_pending": _stage2_status(paper, config)[1],
                "search_score": total_score,
            })
        except Exception as e:
            print(f"Warning: Failed to load paper {paper_id} for search: {e}")

    if sort_by == "latest":
        def _get_result_date(r):
            d = _parse_sort_date(r.get("published_date", ""))
            return d or _parse_sort_date(r.get("created_at", "")) or datetime.fromtimestamp(0, tz=timezone.utc)
        results.sort(key=lambda r: (_relevance_sort_rank(r.get("is_relevant")), -_get_result_date(r).timestamp()))
    else:
        results.sort(key=lambda r: (_relevance_sort_rank(r.get("is_relevant")), -r.get("search_score", 0)))
    return results[:limit]


@app.post("/fetch")
async def trigger_fetch():
    """
    Manually trigger paper fetching.
    Fetch and analysis run in background (non-blocking).
    """
    async def fetch_and_analyze():
        try:
            config = await asyncio.to_thread(Config.load, config_path)
            print(f"\n📡 Manual fetch triggered (streaming pipeline)...")
            n = await analyzer.run_streaming_fetch_and_analyze(
                fetcher, config, config.max_papers_per_fetch
            )
            print(f"✓ Manual fetch and analysis complete ({n} papers)")
        except Exception as e:
            print(f"✗ Manual fetch error: {e}")
            import traceback
            traceback.print_exc()
    
    # Start task in background
    asyncio.create_task(fetch_and_analyze())
    
    return {"message": "Fetch triggered", "status": "running"}


async def _maybe_save_user_paper(paper, request: Request):
    """Save paper for user in serving mode. Non-blocking."""
    try:
        from serving.integrate import get_user_and_config_async, save_paper_for_user
        user_id, _ = await get_user_and_config_async(request, config_path)
        if user_id:
            await asyncio.to_thread(save_paper_for_user, paper, user_id)
    except ImportError:
        pass


@app.post("/papers/{paper_id}/hide")
async def hide_paper(request: Request, paper_id: str):
    """Hide a paper"""
    try:
        paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
        paper.is_hidden = True
        await asyncio.to_thread(fetcher.save_paper, paper)
        await _maybe_save_user_paper(paper, request)
        return {"message": "Paper hidden", "paper_id": paper_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/{paper_id}/unhide")
async def unhide_paper(request: Request, paper_id: str):
    """Unhide a paper"""
    try:
        paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
        paper.is_hidden = False
        await asyncio.to_thread(fetcher.save_paper, paper)
        await _maybe_save_user_paper(paper, request)
        return {"message": "Paper unhidden", "paper_id": paper_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/{paper_id}/star")
async def star_paper(request: Request, paper_id: str):
    """Star a paper and classify it (or unstar)"""
    try:
        user_id, config = None, None
        try:
            from serving.integrate import get_user_and_config_async
            user_id, config = await get_user_and_config_async(request, config_path)
        except ImportError:
            config = await asyncio.to_thread(Config.load, config_path)
        paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
        if user_id:
            from serving.paper_overlay import overlay_paper
            paper = overlay_paper(paper, user_id)
        paper.is_starred = not paper.is_starred
        if paper.is_starred:
            paper.star_category = await analyzer.classify_starred_paper(paper, config)
            print(f"[DEBUG] Starred {paper_id} -> category: {paper.star_category}")
        else:
            paper.star_category = "Other"
        await asyncio.to_thread(fetcher.save_paper, paper)
        await _maybe_save_user_paper(paper, request)
        return {
            "message": "论文已收藏" if paper.is_starred else "取消收藏",
            "is_starred": paper.is_starred,
            "star_category": paper.star_category,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/{paper_id}/update_relevance")
async def update_relevance(http_request: Request, paper_id: str, body: UpdateRelevanceRequest):
    """Update paper relevance status and score manually"""
    try:
        paper = await asyncio.to_thread(fetcher.load_paper, paper_id)
        paper.is_relevant = body.is_relevant
        paper.relevance_score = max(0, min(10, body.relevance_score))  # Clamp 0-10
        paper.updated_at = datetime.now().isoformat()
        await asyncio.to_thread(fetcher.save_paper, paper)
        await _maybe_save_user_paper(paper, http_request)
        return {
            "message": "论文相关性已更新",
            "is_relevant": paper.is_relevant,
            "relevance_score": paper.relevance_score
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/papers/reprocess-negative-keyword-blocked")
async def reprocess_negative_keyword_blocked_endpoint(background_tasks: BackgroundTasks):
    """Re-run Stage 1 for papers that were auto-blocked by negative keywords. Uses LLM to re-score."""
    background_tasks.add_task(reprocess_negative_keyword_blocked)
    return {"message": "Reprocessing started. Check server logs for progress."}


# ============ Background Tasks ============

async def reclassify_all_starred_papers(config: Config):
    """Re-classify all starred papers when categories change. DB/IO runs in thread pool."""
    try:
        metadata_list = await asyncio.to_thread(fetcher.list_papers_metadata, SCAN_ALL, True)
        starred = [m for m in metadata_list if m.get('is_starred', False)]
        if not starred:
            print("✓ No starred papers to reclassify")
            return
        print(f"\n🏷️ Reclassifying {len(starred)} starred papers...")
        for meta in starred:
            try:
                paper = await asyncio.to_thread(fetcher.load_paper, meta['id'])
                paper.star_category = await analyzer.classify_starred_paper(paper, config)
                await asyncio.to_thread(fetcher.save_paper, paper)
                print(f"  ✓ {paper.id} -> {paper.star_category}")
            except Exception as e:
                print(f"  ✗ Failed {meta.get('id', '?')}: {e}")
        print("✓ Reclassification complete")
    except Exception as e:
        print(f"✗ Reclassification error: {e}")
        import traceback
        traceback.print_exc()


NEGATIVE_KEYWORD_BLOCK_PATTERN = "论文包含负面关键词"


async def reprocess_negative_keyword_blocked():
    """Re-run Stage 1 for papers auto-blocked by old negative keyword strategy."""
    try:
        config = await asyncio.to_thread(Config.load, config_path)
        all_papers = await asyncio.to_thread(fetcher.list_papers, 0, SCAN_ALL)
        blocked = [p for p in all_papers if NEGATIVE_KEYWORD_BLOCK_PATTERN in (p.one_line_summary or "")]
        if not blocked:
            print("✓ No papers to reprocess (none were auto-blocked by negative keywords)")
            return
        for p in blocked:
            if not p.preview_text:
                p.preview_text = (p.abstract or "")[:2000]
        print(f"\n🔄 Reprocessing {len(blocked)} papers that were auto-blocked by negative keywords...")
        await analyzer.process_papers(blocked, config)
        print(f"✓ Reprocess complete: {len(blocked)} papers re-scored by LLM")
    except Exception as e:
        print(f"✗ Reprocess error: {e}")
        import traceback
        traceback.print_exc()


async def check_pending_deep_analysis():
    """Check for papers needing Stage 2. Process with priority on startup."""
    try:
        config = await asyncio.to_thread(Config.load, config_path)
        all_papers = await asyncio.to_thread(fetcher.list_papers, 0, SCAN_ALL)

        # Find papers needing Stage 2 (no summary or incomplete preset Q&As)
        pending_papers = [
            p for p in all_papers 
            if _stage2_status(p, config)[0]
        ]
        
        if pending_papers:
            min_score = getattr(config, 'min_relevance_score_for_stage2', 6.0)
            print(f"\n🔍 Found {len(pending_papers)} papers pending deep analysis (score >= {min_score})")
            print(f"📚 Prioritizing deep analysis for these papers...")
            
            # Process with skip_stage1=True since they're already marked as relevant
            await analyzer.process_papers(pending_papers, config, skip_stage1=True)
            print(f"✓ Completed pending deep analysis for {len(pending_papers)} papers")
        else:
            min_score = getattr(config, 'min_relevance_score_for_stage2', 6.0)
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
    In serving mode: only fetch (no global analysis; analysis is on-demand per user).
    """
    try:
        from serving.integrate import SERVING_MODE
        serving = SERVING_MODE
    except ImportError:
        serving = False

    if not serving:
        asyncio.create_task(check_pending_deep_analysis())
    else:
        print("📡 Serving mode: skipping background analysis (on-demand per user)")

    while True:
        try:
            config = await asyncio.to_thread(Config.load, config_path)
            print(f"\n📡 Fetching papers... [{datetime.now().strftime('%H:%M:%S')}]")
            if serving:
                per_cat = max(10, config.max_papers_per_fetch // max(1, len(fetcher.categories)))
                papers = await fetcher.fetch_latest(max_papers_per_category=per_cat)
                n = len(papers)
            else:
                n = await analyzer.run_streaming_fetch_and_analyze(
                    fetcher, config, config.max_papers_per_fetch
                )
            if n == 0:
                print(f"✓ No new papers" + (" to analyze" if not serving else ""))
            await asyncio.sleep(config.fetch_interval)
        
        except Exception as e:
            print(f"✗ Background fetcher error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)  # Wait 1 min on error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

