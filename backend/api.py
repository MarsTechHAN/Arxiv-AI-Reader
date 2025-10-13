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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from datetime import datetime
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

# Global instances
fetcher = ArxivFetcher()
analyzer = DeepSeekAnalyzer()
config_path = Path("data/config.json")

# Serve frontend static files FIRST (before other routes)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# Request/Response models
class AskQuestionRequest(BaseModel):
    question: str


class UpdateConfigRequest(BaseModel):
    filter_keywords: Optional[List[str]] = None
    preset_questions: Optional[List[str]] = None
    system_prompt: Optional[str] = None


class UpdateRelevanceRequest(BaseModel):
    is_relevant: bool
    relevance_score: float


# ============ Frontend & Endpoints ============

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve frontend index.html"""
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        return FileResponse(str(frontend_path / "index.html"))
    return {"message": "Frontend not found. Please check frontend directory."}


@app.get("/api/health")
async def health_check():
    """API health check"""
    return {"message": "arXiv Paper Fetcher API", "status": "running"}


@app.get("/papers", response_model=List[dict])
async def list_papers(skip: int = 0, limit: int = 20, sort_by: str = "relevance", keyword: str = None):
    """
    List papers for timeline.
    sort_by: 'relevance' (default), 'latest', 'starred'
    keyword: filter by keyword
    """
    papers = fetcher.list_papers(skip=0, limit=1000)  # Load all first
    
    # Filter by keyword if provided
    if keyword:
        papers = [p for p in papers if keyword.lower() in ' '.join(p.extracted_keywords).lower()]
    
    # Filter out hidden papers
    papers = [p for p in papers if not p.is_hidden]
    
    # Sort with priority: Starred > Has deep analysis > Relevance score
    if sort_by == "relevance":
        # Multi-level sorting: 
        # 1. Starred first (True > False)
        # 2. Has deep analysis (True > False) 
        # 3. Relevance score (high > low)
        papers.sort(key=lambda p: (
            p.is_starred,
            bool(p.detailed_summary and p.detailed_summary.strip()),  # Has deep analysis
            p.relevance_score
        ), reverse=True)
    elif sort_by == "latest":
        papers.sort(key=lambda p: p.published_date or p.created_at, reverse=True)
    elif sort_by == "starred":
        papers.sort(key=lambda p: (p.is_starred, p.relevance_score), reverse=True)
    
    # Paginate
    papers = papers[skip:skip + limit]
    
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


@app.get("/config")
async def get_config():
    """Get current configuration"""
    config = Config.load(config_path)
    return config.to_dict()


@app.put("/config")
async def update_config(request: UpdateConfigRequest):
    """Update configuration"""
    config = Config.load(config_path)
    
    if request.filter_keywords is not None:
        config.filter_keywords = request.filter_keywords
    if request.preset_questions is not None:
        config.preset_questions = request.preset_questions
    if request.system_prompt is not None:
        config.system_prompt = request.system_prompt
    
    config.save(config_path)
    
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
            
            # Trigger analysis in background if not analyzed yet
            if paper.is_relevant is None:
                config = Config.load(config_path)
                asyncio.create_task(analyzer.process_papers([paper], config))
                print(f"📊 Started background analysis for {arxiv_id}")
            
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
        fetcher.save_paper(paper)
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


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    papers = fetcher.list_papers(limit=10000)
    
    total = len(papers)
    analyzed = len([p for p in papers if p.is_relevant is not None])
    relevant = len([p for p in papers if p.is_relevant])
    starred = len([p for p in papers if p.is_starred])
    hidden = len([p for p in papers if p.is_hidden])
    
    return {
        "total_papers": total,
        "analyzed_papers": analyzed,
        "relevant_papers": relevant,
        "starred_papers": starred,
        "hidden_papers": hidden,
        "pending_analysis": total - analyzed,
    }


# ============ Background Tasks ============

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

