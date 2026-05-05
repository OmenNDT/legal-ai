"""
Entry point: python app.py
Launches Flask backend (port 8000) + Streamlit frontend (port 8501) in parallel.
ChromaDB is expected to already be running (systemd service).
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"


def start_flask():
    return subprocess.Popen(
        [sys.executable, "-m", "flask", "--app", "src.app", "run",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def start_streamlit():
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(SRC / "ui.py"),
         "--server.port", "8501",
         "--server.address", "0.0.0.0",
         "--server.headless", "true"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def main():
    print("Starting Flask backend on http://localhost:8000 ...")
    flask_proc = start_flask()

    # Wait for Flask to be ready before Streamlit tries to connect
    time.sleep(2)

    print("Starting Streamlit frontend on http://localhost:8501 ...")
    streamlit_proc = start_streamlit()

    print("\nAll services running. Press Ctrl+C to stop.\n")

    procs = [flask_proc, streamlit_proc]

    def shutdown(sig, frame):
        print("\nShutting down...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Restart any process that dies unexpectedly
    while True:
        for i, (proc, name, starter) in enumerate(zip(
            procs,
            ["Flask", "Streamlit"],
            [start_flask, start_streamlit],
        )):
            if proc.poll() is not None:
                print(f"[WARN] {name} exited (code {proc.returncode}), restarting...")
                procs[i] = starter()
        time.sleep(3)


if __name__ == "__main__":
    main()
