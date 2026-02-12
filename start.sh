#!/bin/bash

# arXiv Paper Fetcher - Start Script
# Uses .venv (Python 3.10+) if exists, else common env

echo "🚀 Starting arXiv Paper Fetcher..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -f "/Users/bytedance/Works/envs/common/bin/activate" ]; then
    source /Users/bytedance/Works/envs/common/bin/activate
fi

# Check if DEEPSEEK_API_KEY is set
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  DEEPSEEK_API_KEY not set"
    echo "Please run: export DEEPSEEK_API_KEY='your-api-key'"
    exit 1
fi

# Create data directory
mkdir -p data/papers

# Build static assets with cache busting
echo "🔨 Building static assets..."
python3 build_static.py
if [ $? -ne 0 ]; then
    echo "⚠️  Static assets build failed, continuing with source files..."
fi

# Check if running with Docker
if [ "$1" == "docker" ]; then
    echo "🐳 Starting with Docker..."
    docker-compose up --build
else
    echo "✅ Starting backend server..."
    echo "📍 URL: http://localhost:8000"
    echo "📖 Features:"
    echo "   - Markdown 渲染 Q&A"
    echo "   - 中文回答"
    echo "   - 相关性打分 (0-10)"
    echo "   - 按相关性/最新/收藏排序"
    echo "   - Hide/Star 功能"
    echo "   - Keyword 筛选"
    echo ""
    cd backend && python api.py
fi

