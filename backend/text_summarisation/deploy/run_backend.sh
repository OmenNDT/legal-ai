#!/bin/bash
# Khoi dong Flask backend local (port 9020)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export API_PORT="${API_PORT:-9020}"
python -m backend.app.server
