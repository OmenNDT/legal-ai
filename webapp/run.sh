#!/usr/bin/env bash
cd "$(dirname "$0")/backend"
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "Starting Flask on http://0.0.0.0:5050"
echo "  - Local:   http://localhost:5050"
[ -n "$HOST_IP" ] && echo "  - Network: http://$HOST_IP:5050"
python app.py