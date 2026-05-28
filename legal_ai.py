import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.legal_ai_app import app

if __name__ == "__main__":
    port = int(os.environ.get("APP_PORT", 9010))
    app.run(host = "0.0.0.0", port = port, debug = False)
