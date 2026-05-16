#!/usr/bin/env bash
# Start Flask backend then open frontend in browser
cd "$(dirname "$0")/backend"
echo "Starting Flask on http://localhost:5050 ..."
python app.py
