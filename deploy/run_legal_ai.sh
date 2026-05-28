#!/bin/bash
# Khởi động Legal AI unified backend (string matching + text summarisation)
# Chạy trên worker1, port 9010
# Usage: bash run_legal_ai.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_DIR:$REPO_DIR/backend/text_sumarisation"
export APP_PORT="${APP_PORT:-9010}"
export APP_PREFIX="${APP_PREFIX:-}"

# Load .env nếu có
if [ -f "$REPO_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$REPO_DIR/.env"
    set +a
fi

cd "$REPO_DIR"
exec python3 -m backend.legal_ai_app
