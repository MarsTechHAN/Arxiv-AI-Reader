# arXiv AI Reader

**Automatically fetch, filter, and analyze latest arXiv papers using DeepSeek AI.**

A lightweight, file-based system that monitors arXiv for new papers, intelligently filters them based on your interests, performs deep analysis with Q&A, and provides an interactive timeline UI for exploration. This code is completely vibed, and I have no idea how it works. Don't ask me why.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Configuration](#️-configuration)
- [Data Structure](#-data-structure)
- [API Reference](#-api-reference)
- [Advanced Usage](#-advanced-usage)
- [Architecture](#-architecture)
- [Requirements](#-requirements)

---

## ✨ Features

### Core Functionality

#### 1. **Automatic Paper Fetching**
- Monitors arXiv RSS feeds every 5 minutes (configurable)
- Fetches papers from configurable categories (AI, CV, LG, CL, NE by default)
- Downloads HTML version for full-text analysis
- Deduplicates automatically - skips existing papers

#### 2. **Two-Stage AI Analysis**

**Stage 1: Quick Filter** (Abstract-based)
- Analyzes paper preview (~2000 chars: abstract + intro)
- Scores relevance (0-10 scale) based on your keywords
- Extracts key topics and generates one-line summary
- **Negative keyword filtering** - instantly rejects unwanted topics (medical, healthcare, etc.)
- Only papers scoring ≥ 6 proceed to Stage 2 (configurable via `min_relevance_score_for_stage2`)

**Stage 2: Deep Analysis** (Full-text)
- Generates detailed summary (200-300 words in Chinese)
- Answers preset questions about methodology, experiments, limitations
- Full paper content analyzed with DeepSeek

#### 3. **KV Cache Optimization**
- Keeps `system_prompt + paper_content` fixed across questions
- DeepSeek's KV cache drastically reduces API costs
- Ask multiple questions without re-processing the paper

#### 4. **Interactive Q&A**
- Ask custom questions about any paper
- **Streaming responses** - see answers in real-time
- **Cross-paper comparison** - reference other papers using `[arxiv_id]` syntax
  - Example: `"Compare this paper with [2510.09212]"`
  - System auto-fetches and analyzes referenced papers
- All Q&A saved to paper JSON

#### 5. **Advanced Paper Management**
- ⭐ **Star/Bookmark** papers for later
- 👁️ **Hide** irrelevant papers from timeline
- 🎯 **Manual relevance override** - adjust AI's relevance scoring
- Smart sorting: Starred → Deep analyzed → Relevance score

#### 6. **Full-Text Search**
- Search across title, abstract, keywords, summaries
- **arXiv ID lookup** - enter `2510.09212` to fetch specific paper
- Auto-triggers analysis for new papers

#### 7. **Beautiful Timeline UI**
- Clean, responsive interface
- Expand/collapse paper details
- Color-coded relevance indicators
- Real-time streaming Q&A
- Keyword filtering

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- DeepSeek API key ([get one here](https://platform.deepseek.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/arxiv_paper_fetcher.git
cd arxiv_paper_fetcher

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your DeepSeek API key
export DEEPSEEK_API_KEY="your-api-key-here"

# 4. Start the server
./start.sh
```

The server starts at `http://localhost:5000` and begins fetching papers automatically.

### Alternative: Manual Start

```bash
# Set API key
export DEEPSEEK_API_KEY="your-api-key"

# Run backend
cd backend
python api.py
```

### Using Docker (Coming Soon)

```bash
# Create .env file
echo "DEEPSEEK_API_KEY=your-api-key" > .env

# Start with Docker Compose
docker-compose up --build
```

---

## ⚙️ Configuration

All configuration is in `data/config.json`. On first run, default config is auto-generated from `backend/default_config.py`.

### Configuration File Structure

```json
{
  "filter_keywords": [
    "video diffusion",
    "multimodal generation",
    "efficient LLM"
  ],
  "negative_keywords": [
    "medical",
    "healthcare",
    "protein"
  ],
  "preset_questions": [
    "What is the core innovation of this paper?",
    "What datasets were used and what were the results?"
  ],
  "system_prompt": "You are a professional academic paper analyst...",
  "fetch_interval": 300,
  "max_papers_per_fetch": 100,
  "model": "deepseek-chat",
  "temperature": 0.3,
  "max_tokens": 2000,
  "concurrent_papers": 10,
  "min_relevance_score_for_stage2": 6.0
}
```

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `filter_keywords` | `List[str]` | See below | Keywords for relevance filtering (Stage 1) |
| `negative_keywords` | `List[str]` | See below | Auto-reject papers containing these terms |
| `preset_questions` | `List[str]` | See below | Questions to ask in Stage 2 analysis |
| `system_prompt` | `str` | - | Fixed prompt for DeepSeek (optimized for KV cache) |
| `fetch_interval` | `int` | `300` | Seconds between fetches (300 = 5 minutes) |
| `max_papers_per_fetch` | `int` | `100` | Max papers to check per category per fetch |
| `model` | `str` | `"deepseek-chat"` | DeepSeek model to use |
| `temperature` | `float` | `0.3` | LLM temperature (0-1, lower = more focused) |
| `max_tokens` | `int` | `2000` | Max tokens per response |
| `concurrent_papers` | `int` | `10` | Papers to analyze concurrently (Stage 2) |
| `min_relevance_score_for_stage2` | `float` | `6.0` | Minimum score (0-10) required for deep analysis |

### Default Filter Keywords
```python
[
    "video diffusion",
    "multimodal generation",
    "unified generation understanding",
    "efficient LLM",
    "efficient diffusion model",
    "diffusion language model",
    "autoregressive diffusion model"
]
```

### Default Negative Keywords
Papers containing these are auto-rejected (score=1):
```python
[
    "medical",
    "healthcare",
    "clinical",
    "protein",
    "molecule"
]
```

### Default Preset Questions
```python
[
    "这篇论文的核心创新点是什么，他想解决什么问题，怎么解决的？",
    "基于他的前作，梳理这个方向的整个发展脉络",
    "他的前作有哪些？使用表格讲讲他和前作的区别",
    "论文提出了哪些关键技术方法？详细说明技术细节",
    "使用了哪些评价指标与数据集？",
    "在哪些数据集上进行了实验？主要性能提升是多少？",
    "论文的主要局限性有哪些？未来改进方向是什么？"
]
```

### How to Modify Configuration

#### Method 1: Edit JSON directly
```bash
vim data/config.json
# Server auto-reloads on next fetch cycle
```

#### Method 2: Use API
```bash
# Get current config
curl http://localhost:8000/config

# Update config
curl -X PUT http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "filter_keywords": ["your", "new", "keywords"],
    "temperature": 0.5
  }'
```

#### Method 3: Use Web UI
Navigate to `http://localhost:8000` → Settings panel → Edit configuration

---

## 📦 Data Structure

All data stored as JSON files - **no database required**.

### Directory Structure
```
data/
├── config.json          # System configuration
└── papers/             
    ├── 2510.08582v1.json
    ├── 2510.08588v1.json
    └── ...
```

### Paper JSON Schema

Each paper is saved as `data/papers/{arxiv_id}.json`:

```json
{
  "id": "2510.08582v1",
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Full abstract text...",
  "url": "https://arxiv.org/abs/2510.08582v1",
  "html_url": "https://arxiv.org/html/2510.08582v1",
  "html_content": "Full paper content extracted from HTML...",
  "preview_text": "Abstract + first 2000 chars for Stage 1...",
  
  "// Stage 1 Results": "",
  "is_relevant": true,
  "relevance_score": 8.5,
  "extracted_keywords": ["keyword1", "keyword2"],
  "one_line_summary": "Brief summary in Chinese",
  
  "// Stage 2 Results": "",
  "detailed_summary": "Detailed 200-300 word summary in Chinese...",
  "qa_pairs": [
    {
      "question": "What is the core innovation?",
      "answer": "Detailed answer...",
      "timestamp": "2024-01-15T10:30:00"
    }
  ],
  
  "// User Actions": "",
  "is_starred": false,
  "is_hidden": false,
  
  "// Metadata": "",
  "published_date": "2024-01-15T08:00:00Z",
  "created_at": "2024-01-15T09:00:00",
  "updated_at": "2024-01-15T10:30:00"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| **Basic Info** | | |
| `id` | `string` | arXiv ID (e.g., `2510.08582v1`) |
| `title` | `string` | Paper title |
| `authors` | `string[]` | Author names |
| `abstract` | `string` | Official arXiv abstract |
| `url` | `string` | arXiv abstract page URL |
| `html_url` | `string` | arXiv HTML version URL |
| `html_content` | `string` | Full paper text (extracted from HTML) |
| `preview_text` | `string` | First ~2000 chars for Stage 1 |
| **Stage 1** | | |
| `is_relevant` | `bool?` | `null` = not analyzed, `true/false` = analyzed |
| `relevance_score` | `float` | 0-10 scale, DeepSeek's relevance rating |
| `extracted_keywords` | `string[]` | Keywords identified by AI |
| `one_line_summary` | `string` | Brief summary (Chinese) |
| **Stage 2** | | |
| `detailed_summary` | `string` | 200-300 word summary (Chinese) |
| `qa_pairs` | `QAPair[]` | Questions and answers |
| **User Actions** | | |
| `is_starred` | `bool` | User bookmarked this paper |
| `is_hidden` | `bool` | Hidden from timeline |
| **Metadata** | | |
| `published_date` | `string` | arXiv submission date |
| `created_at` | `string` | When fetched by system |
| `updated_at` | `string` | Last modification time |

---

## 📡 API Reference

Backend runs on `http://localhost:8000` (default).

### Paper Endpoints

#### `GET /papers`
List papers with pagination and filtering.

**Query Parameters:**
- `skip` (int, default: 0) - Pagination offset
- `limit` (int, default: 20) - Max papers to return
- `sort_by` (string, default: "relevance") - Sort mode: `relevance`, `latest`, `starred`
- `keyword` (string, optional) - Filter by keyword

**Response:**
```json
[
  {
    "id": "2510.08582v1",
    "title": "Paper Title",
    "abstract": "Truncated abstract...",
    "is_relevant": true,
    "relevance_score": 8.5,
    "extracted_keywords": ["keyword1", "keyword2"],
    "one_line_summary": "Summary...",
    "is_starred": false,
    "has_qa": true,
    "detailed_summary": "..."
  }
]
```

---

#### `GET /papers/{paper_id}`
Get full paper details including all Q&A.

**Response:** Full Paper JSON (see [Data Structure](#-data-structure))

---

#### `POST /papers/{paper_id}/ask`
Ask a custom question about a paper.

**Request Body:**
```json
{
  "question": "What is the core innovation?"
}
```

**Response:**
```json
{
  "question": "What is the core innovation?",
  "answer": "Detailed answer from DeepSeek...",
  "paper_id": "2510.08582v1"
}
```

---

#### `POST /papers/{paper_id}/ask_stream`
Ask a question with **streaming response** (Server-Sent Events).

**Request Body:** Same as `/ask`

**Response:** SSE stream
```
data: {"chunk": "The"}
data: {"chunk": " core"}
data: {"chunk": " innovation..."}
data: {"done": true}
```

**Frontend Example:**
```javascript
const evtSource = new EventSource(`/papers/${paperId}/ask_stream`);
evtSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.chunk) {
    answerDiv.textContent += data.chunk;
  } else if (data.done) {
    evtSource.close();
  }
};
```

---

#### `POST /papers/{paper_id}/star`
Star/unstar a paper (toggle).

**Response:**
```json
{
  "message": "论文已收藏",
  "is_starred": true
}
```

---

#### `POST /papers/{paper_id}/hide`
Hide a paper from timeline.

---

#### `POST /papers/{paper_id}/unhide`
Unhide a paper.

---

#### `POST /papers/{paper_id}/update_relevance`
Manually override AI's relevance assessment.

**Request Body:**
```json
{
  "is_relevant": true,
  "relevance_score": 9.0
}
```

---

### Search Endpoint

#### `GET /search?q={query}`
Search papers by keyword or arXiv ID.

**Special behavior:** If query matches arXiv ID format (e.g., `2510.08582`), fetches and analyzes that specific paper.

**Query Parameters:**
- `q` (string, required) - Search query or arXiv ID
- `limit` (int, default: 50) - Max results

---

### Configuration Endpoints

#### `GET /config`
Get current configuration.

**Response:** Full config JSON (see [Configuration](#️-configuration))

---

#### `PUT /config`
Update configuration (partial update supported).

**Request Body:**
```json
{
  "filter_keywords": ["new", "keywords"],
  "temperature": 0.5
}
```

---

### System Endpoints

#### `POST /fetch`
Manually trigger paper fetching (non-blocking).

**Response:**
```json
{
  "message": "Fetch triggered",
  "status": "running"
}
```

---

#### `GET /stats`
Get system statistics.

**Response:**
```json
{
  "total_papers": 150,
  "analyzed_papers": 145,
  "relevant_papers": 32,
  "starred_papers": 8,
  "hidden_papers": 5,
  "pending_analysis": 5
}
```

---

## 🔥 Advanced Usage

### Cross-Paper Comparison

Ask questions that reference other papers using `[arxiv_id]` syntax. System automatically fetches and analyzes referenced papers.

**Example:**
```bash
curl -X POST http://localhost:8000/papers/2510.08582v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare this paper with [2510.09212] and [2510.08590]. Which approach is more efficient?"
  }'
```

**What happens:**
1. System detects `[2510.09212]` and `[2510.08590]`
2. Fetches these papers from arXiv (if not already local)
3. Analyzes them (Stage 1 + Stage 2 if relevant)
4. Builds combined context: current paper + 2 referenced papers
5. DeepSeek answers with full context

**In Web UI:** Just type your question with `[arxiv_id]` - everything is automatic.

---

### Batch Processing New Papers

If you have many unanalyzed papers:

```bash
cd backend
python analyzer.py
```

This finds all papers with `is_relevant=null` and analyzes them in batches.

---

### Custom Analysis Script

```python
import asyncio
from fetcher import ArxivFetcher
from analyzer import DeepSeekAnalyzer
from models import Config

async def analyze_specific_papers():
    fetcher = ArxivFetcher()
    analyzer = DeepSeekAnalyzer()
    config = Config.load("data/config.json")
    
    # Fetch specific paper
    paper = await fetcher.fetch_single_paper("2510.08582v1")
    
    # Analyze
    await analyzer.stage1_filter(paper, config)
    if paper.is_relevant:
        await analyzer.stage2_qa(paper, config)
    
    print(f"Relevance: {paper.relevance_score}/10")
    print(f"Summary: {paper.one_line_summary}")

asyncio.run(analyze_specific_papers())
```

---

### Adjusting Analysis Threshold

By default, only papers with `relevance_score >= 6.0` get deep analysis (Stage 2). To change:

```json
{
  "min_relevance_score_for_stage2": 7.0
}
```

Lower = more papers analyzed (higher cost)  
Higher = only top papers analyzed (lower cost)

---

### Customizing arXiv Categories

Edit `backend/fetcher.py`:

```python
self.categories = [
    "cs.AI",   # Artificial Intelligence
    "cs.CV",   # Computer Vision
    "cs.LG",   # Machine Learning
    "cs.CL",   # Computation and Language
    "cs.NE",   # Neural and Evolutionary Computing
    "cs.RO",   # Robotics (add this)
    "stat.ML", # Statistics - Machine Learning (add this)
]
```

See [arXiv category list](https://arxiv.org/category_taxonomy) for all options.

---

## 🏗️ Architecture

**Philosophy:** Simple data structures, no bullshit.

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Vanilla JS)                │
│  - Timeline UI with expand/collapse                     │
│  - Real-time streaming Q&A                              │
│  - Search, filter, star, hide                           │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/SSE
┌────────────────────▼────────────────────────────────────┐
│                FastAPI Backend (api.py)                 │
│  - REST API + SSE streaming                             │
│  - Serves static files                                  │
│  - Background fetcher loop                              │
└─────┬──────────────┬──────────────┬─────────────────────┘
      │              │              │
┌─────▼──────┐  ┌────▼─────┐  ┌────▼──────────┐
│  Fetcher   │  │ Analyzer │  │ File System   │
│ (fetcher.py│  │(analyzer.│  │  (data/)      │
│            │  │    py)   │  │               │
│ - RSS poll │  │ - Stage1 │  │ *.json files  │
│ - HTML get │  │ - Stage2 │  │ No database   │
│ - Parse    │  │ - KV opt │  │               │
└─────┬──────┘  └────┬─────┘  └───────────────┘
      │              │
      │         ┌────▼─────────────────┐
      │         │   DeepSeek API       │
      │         │  - Chat completion   │
      │         │  - KV cache          │
      │         │  - Streaming         │
      │         └──────────────────────┘
      │
┌─────▼────────────────────────┐
│     arXiv.org                │
│  - RSS feeds (5 categories)  │
│  - HTML papers               │
└──────────────────────────────┘
```

### Data Flow

1. **Fetch Loop** (every 5 min)
   ```
   arXiv RSS → Fetcher → Save JSON → Trigger Analysis
   ```

2. **Stage 1 Analysis** (all new papers)
   ```
   Preview text → DeepSeek → Relevance score → Update JSON
   ```

3. **Stage 2 Analysis** (score ≥ 6)
   ```
   Full content → DeepSeek (KV cached) → Summary + Q&A → Update JSON
   ```

4. **Interactive Q&A**
   ```
   User question → DeepSeek (reuses cache) → Stream answer → Save to JSON
   ```

### Key Design Principles

1. **File System as Database**  
   - One JSON per paper  
   - No SQL/NoSQL overhead  
   - Trivial backup: `cp -r data/ backup/`

2. **KV Cache Optimization**  
   - Fixed prefix: `system_prompt + paper_content`  
   - Only question changes → 90% cache hit  
   - Dramatically cheaper API costs

3. **Two-Stage Pipeline**  
   - Stage 1: Cheap filter (preview only)  
   - Stage 2: Expensive analysis (relevant papers only)  
   - Saves money and time

4. **Async Everything**  
   - Concurrent fetching (all categories parallel)  
   - Concurrent analysis (10 papers at once)  
   - Non-blocking background tasks

5. **Zero Special Cases**  
   - Same data structure for all papers  
   - Same analysis flow for all papers  
   - No edge case handling (clean code)

---

## 📚 Requirements

### Python Packages

See `requirements.txt`:

```txt
fastapi==0.109.0        # Web framework
uvicorn[standard]==0.27.0  # ASGI server
httpx==0.26.0           # Async HTTP client
beautifulsoup4==4.12.3  # HTML parsing
lxml==5.1.0             # XML/HTML parser
python-multipart==0.0.6 # Form parsing
pydantic==2.5.3         # Data validation
openai==1.12.0          # DeepSeek API client
feedparser==6.0.11      # RSS parsing
aiofiles==23.2.1        # Async file I/O
```

### External APIs

- **DeepSeek API**  
  - Get key at: https://platform.deepseek.com  
  - Model: `deepseek-chat`  
  - Supports KV cache and streaming  

### System Requirements

- Python 3.8+
- 1GB RAM minimum (2GB+ recommended for concurrent analysis)
- ~100MB disk per 1000 papers (with full HTML content)

---

## 🤝 Contributing

Built following **Linus Torvalds' principles**:
- **Good taste**: Simple data structures, no special cases
- **Practical**: Solves real problems, not imaginary ones
- **No bullshit**: File system as database, dead simple code

Pull requests welcome. Keep it simple.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [arXiv](https://arxiv.org) for providing open access to research papers
- [DeepSeek](https://www.deepseek.com) for powerful and cost-effective AI
- Inspired by the Unix philosophy: do one thing and do it well
