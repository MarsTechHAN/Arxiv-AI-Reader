# arXiv Paper Fetcher with DeepSeek Analysis

A system that automatically fetches latest papers from arXiv, uses DeepSeek AI to filter and analyze them based on your keywords, and provides an interactive Q&A interface.

## Architecture

**Simple and effective - no bullshit.**

```
arxiv-paper-fetcher/
├── backend/
│   ├── fetcher.py          # arXiv fetcher (every 5 minutes)
│   ├── analyzer.py         # DeepSeek two-stage analysis
│   ├── api.py              # FastAPI backend
│   └── models.py           # Data models
├── frontend/              
│   ├── index.html          # Timeline UI
│   ├── app.js              # Frontend logic
│   └── style.css           # Styles
├── data/
│   ├── papers/             # {arxiv_id}.json
│   └── config.json         # Keywords, questions, system prompt
└── requirements.txt
```

## Features

- **Automatic Fetching**: Pulls latest papers from arXiv every 5 minutes
- **Two-Stage Analysis**:
  - Stage 1: Quick filter using abstract/first 2000 chars
  - Stage 2: Deep analysis with Q&A for relevant papers
- **KV Cache Optimization**: Reuses "system prompt + content" to minimize API costs
- **Interactive Q&A**: Ask custom questions on any paper
- **Timeline UI**: Clean, intuitive interface with expand/collapse
- **Full-text Search**: Search across all papers

## Quick Start

### Method 1: Quick Start (Recommended)

```bash
# 1. Set your DeepSeek API key
export DEEPSEEK_API_KEY="your-api-key-here"

# 2. Run the start script
./start.sh
```

The backend will start at `http://localhost:8000` and automatically begin fetching papers every 5 minutes.

### Method 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set DeepSeek API key
export DEEPSEEK_API_KEY="your-api-key"

# Run backend
cd backend
python api.py
```

### Method 3: Docker

```bash
# Set API key in .env file
echo "DEEPSEEK_API_KEY=your-api-key" > .env

# Start with Docker Compose
docker-compose up --build
```

### Access the UI

Open your browser and navigate to:
- Frontend: `http://localhost:8000/` (served by FastAPI)
- API Docs: `http://localhost:8000/docs`

Or open `frontend/index.html` directly in your browser.

## Configuration

Edit `data/config.json`:
- `filter_keywords`: Keywords for stage 1 filtering
- `preset_questions`: Questions to ask for relevant papers
- `system_prompt`: Fixed prompt for DeepSeek (optimized for KV cache)

## Data Structure

Papers stored as `data/papers/{arxiv_id}.json`:
```json
{
  "id": "2401.12345",
  "title": "Paper Title",
  "abstract": "...",
  "is_relevant": true,
  "extracted_keywords": ["keyword1", "keyword2"],
  "one_line_summary": "...",
  "qa_pairs": [
    {"question": "...", "answer": "..."}
  ]
}
```

## How It Works

1. **Fetcher** runs every 5 minutes, pulls latest papers from arXiv RSS feeds
2. **Stage 1 Filter**: DeepSeek analyzes abstract/preview (~2000 chars) and determines relevance
3. **Stage 2 Analysis**: For relevant papers, DeepSeek reads full content and answers preset questions
4. **KV Cache**: Reuses "system prompt + paper content" across questions to minimize API costs
5. **Interactive Q&A**: Users can ask custom questions, reusing the same KV cache

## API Endpoints

- `GET /papers` - List papers (paginated)
- `GET /papers/{id}` - Get paper details
- `POST /papers/{id}/ask` - Ask custom question
- `GET /config` - Get configuration
- `PUT /config` - Update configuration
- `GET /search?q=query` - Search papers
- `POST /fetch` - Manually trigger fetch
- `GET /stats` - System statistics

## Philosophy

Built following Linus Torvalds' principles:
- **Good taste**: Simple data structures, no special cases
- **Practical**: Solves real problems, not imaginary ones
- **No bullshit**: File system as database, dead simple code

## Notes

- Papers are stored in `data/papers/{arxiv_id}.json`
- Configuration in `data/config.json`
- DeepSeek API key required (get one at https://platform.deepseek.com)
- Fetches from categories: cs.AI, cs.CV, cs.LG, cs.CL, cs.NE (configurable in `fetcher.py`)

