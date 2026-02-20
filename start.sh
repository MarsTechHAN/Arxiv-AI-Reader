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

# Create data directory (project_root/data/ - canonical path)
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
    cd backend && python api.py
fi

