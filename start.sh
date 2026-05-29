#!/usr/bin/env bash
# ============================================================
#  RAG System — Linux / macOS Startup Script
# ============================================================
set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

# ── Resolve project root (directory this script lives in) ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Banner ──────────────────────────────────────────────────
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║           RAG Chatbot System  v1.0               ║"
echo "║   PDF Upload · Chunking · Embedding · Chat       ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Check Python version (≥ 3.10) ───────────────────────
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "${major:-0}" -ge 3 ] && [ "${minor:-0}" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo -e "${RED}✗ Python 3.10 or higher is required but was not found.${NC}"
    echo "  Please install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

PY_VERSION=$("$PYTHON" --version 2>&1)
echo -e "${GREEN}✓ Found ${PY_VERSION}${NC}"

# ── 2. Create virtual environment if it does not exist ──────
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}» Creating virtual environment …${NC}"
    "$PYTHON" -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created at ${VENV_DIR}${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# ── 3. Activate the virtual environment ─────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# ── 4. Install / update dependencies ───────────────────────
INSTALLED_FLAG="$VENV_DIR/.installed"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

needs_install=false
if [ ! -f "$INSTALLED_FLAG" ]; then
    needs_install=true
elif [ "$REQ_FILE" -nt "$INSTALLED_FLAG" ]; then
    echo -e "${YELLOW}» requirements.txt changed since last install${NC}"
    needs_install=true
fi

if $needs_install; then
    echo -e "${YELLOW}» Installing Python dependencies (this may take a few minutes the first time) …${NC}"
    pip install --upgrade pip setuptools wheel -q
    pip install -r "$REQ_FILE"
    touch "$INSTALLED_FLAG"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies up to date${NC}"
fi

# ── 5. Create required directories ─────────────────────────
for dir in uploads data models logs config; do
    mkdir -p "$SCRIPT_DIR/$dir"
done
echo -e "${GREEN}✓ Required directories verified${NC}"

# ── 6. Copy .env.example → .env if .env is missing ─────────
if [ ! -f "$SCRIPT_DIR/.env" ] && [ -f "$SCRIPT_DIR/.env.example" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo -e "${YELLOW}» Created .env from .env.example — edit it to add your API keys${NC}"
fi

# ── 7. Read configuration ──────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Source .env if it exists (simple key=value parsing)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Re-read after sourcing .env (env vars may override defaults)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# ── 8. Launch the server ────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}🚀 Starting RAG Chatbot Server …${NC}"
echo -e "${BOLD}────────────────────────────────────────────────${NC}"
echo -e "  ${BOLD}Local URL :${NC}  http://127.0.0.1:${PORT}"
echo -e "  ${BOLD}Network   :${NC}  http://${HOST}:${PORT}"
echo -e "  ${BOLD}API Docs  :${NC}  http://127.0.0.1:${PORT}/docs"
echo -e "  ${BOLD}Log Level :${NC}  ${LOG_LEVEL}"
echo -e "${BOLD}────────────────────────────────────────────────${NC}"
echo -e "  Press ${BOLD}Ctrl+C${NC} to stop the server"
echo ""

exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --reload
