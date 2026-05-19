#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

FRONTEND_DIR="$ROOT/frontend"
DIST_DIR="$FRONTEND_DIR/dist"

need_build=0
if [ ! -f "$DIST_DIR/index.html" ]; then
    need_build=1
else
    newest_src=$(find "$FRONTEND_DIR/src" "$FRONTEND_DIR/index.html" "$FRONTEND_DIR/package.json" "$FRONTEND_DIR/vite.config.js" -type f -newer "$DIST_DIR/index.html" 2>/dev/null | head -n 1)
    [ -n "$newest_src" ] && need_build=1
fi

if [ "$need_build" = "1" ]; then
    echo "Building frontend..."
    (cd "$FRONTEND_DIR" && npm run build)
else
    echo "Frontend dist is up to date — skipping build."
fi

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "Starting Flask on http://0.0.0.0:9010"
echo "  - Local:   http://localhost:9010"
[ -n "$HOST_IP" ] && echo "  - Network: http://$HOST_IP:9010"

python -m backend.app
