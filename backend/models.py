"""
Data models - simple and clean.
No ORM bullshit, just Python dataclasses.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
from datetime import datetime
import json


@dataclass
class QAPair:
    """Question-answer pair with reasoning support"""
    question: str
    answer: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Reasoning mode (deepseek-reasoner)
    thinking: Optional[str] = None  # Thinking process (only for reasoning mode)
    is_reasoning: bool = False      # Whether this used reasoning model
    
    # Follow-up conversation support
    parent_qa_id: Optional[int] = None  # Index of parent QA in qa_pairs list (for follow-ups)


@dataclass
class Paper:
    """
    Core data structure.
    Everything revolves around this.
    """
    id: str  # arXiv ID (e.g., "2401.12345")
    title: str
    authors: List[str]
    abstract: str
    url: str
    html_url: str
    html_content: str = ""
    preview_text: str = ""  # First 2000 chars for stage 1
    
    # Stage 1 results (quick filter)
    is_relevant: Optional[bool] = None
    relevance_score: float = 0.0  # 0-10, DeepSeek scores relevance
    extracted_keywords: List[str] = field(default_factory=list)
    one_line_summary: str = ""
    
    # Stage 2 results (deep analysis)
    detailed_summary: str = ""  # Detailed AI-generated abstract (Chinese)
    tags: List[str] = field(default_factory=list)  # AI-generated tags (prioritize given tags)
    qa_pairs: List[QAPair] = field(default_factory=list)
    
    # User actions
    is_hidden: bool = False
    is_starred: bool = False
    star_category: str = "Other"  # AI-classified category for starred papers
    
    # Metadata
    published_date: str = ""  # arXiv submission date
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Paper':
        """Create Paper from dict"""
        data = dict(data)
        if 'star_category' not in data:
            data['star_category'] = 'Other'
        if 'qa_pairs' in data and data['qa_pairs']:
            data['qa_pairs'] = [
                QAPair(**qa) if isinstance(qa, dict) else qa
                for qa in data['qa_pairs']
            ]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Config:
    """
    System configuration.
    User controls everything here.
    """
    filter_keywords: List[str] = field(default_factory=list)
    negative_keywords: List[str] = field(default_factory=list)
    preset_questions: List[str] = field(default_factory=list)
    system_prompt: str = "You are a helpful AI assistant analyzing academic papers."
    
    # Fetcher settings
    fetch_interval: int = 300  # 5 minutes
    max_papers_per_fetch: int = 50
    
    # DeepSeek settings
    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 2000
    concurrent_papers: int = 3  # Legacy, used when not using pipeline
    stage1_concurrency: int = 256  # Max concurrent Stage 1 filter tasks
    stage2_concurrency: int = 128  # Max concurrent Stage 2 deep analysis tasks
    min_relevance_score_for_stage2: float = 6.0  # Minimum relevance score for Stage 2 deep analysis

    # Star categories for AI classification (narrowest first)
    star_categories: List[str] = field(default_factory=lambda: [
        "高效视频生成", "LLM稀疏注意力", "注意力机制", "Roll-out方法"
    ])

    # Optional MCP/external search URL for AI search candidates.
    # GET {url}?q=query&limit=N returns JSON array of paper dicts.
    mcp_search_url: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON file"""
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except FileNotFoundError:
            # Return default config
            return cls()

