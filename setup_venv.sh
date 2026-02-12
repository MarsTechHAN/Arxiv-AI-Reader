#!/bin/bash
# Create Python 3.10+ venv for arXiv AI Reader (required for MCP)

set -e
cd "$(dirname "$0")"

PYTHON=""
for p in python3.12 python3.11 python3.10; do
    if command -v $p &>/dev/null; then
        ver=$($p -c "import sys; print(sys.version_info >= (3,10))" 2>/dev/null)
        if [ "$ver" = "True" ]; then
            PYTHON=$p
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.10+ not found."
    echo "Install: brew install python@3.10"
    echo "Or: brew install python@3.12"
    exit 1
fi

echo "Using $PYTHON: $($PYTHON --version)"
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Venv ready. Activate: source .venv/bin/activate"
