#!/usr/bin/env python3
"""Chạy đồng thời backend (Flask) + frontend (Vite). Ctrl+C để tắt cả 2."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_CWD = ROOT
FRONTEND_CWD = ROOT / "frontend"

IS_WINDOWS = os.name == "nt"


def spawn(cmd, cwd, name):
    print(f"[run_dev] starting {name}: {' '.join(cmd)}  (cwd={cwd})")
    kwargs = {"cwd": str(cwd)}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(cmd, **kwargs)


def terminate(proc, name):
    if proc.poll() is not None:
        return
    print(f"[run_dev] stopping {name} (pid={proc.pid})...")
    try:
        if IS_WINDOWS:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print(f"[run_dev] force killing {name}")
        try:
            if IS_WINDOWS:
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()


def main():
    frontend_only = "--frontend-only" in sys.argv or "-f" in sys.argv

    backend_cmd = [sys.executable, "-m", "backend.app"]
    npm = "npm.cmd" if IS_WINDOWS else "npm"
    frontend_cmd = [npm, "run", "dev"]

    if not (FRONTEND_CWD / "node_modules").exists():
        print("[run_dev] node_modules missing — running `npm install` first...")
        subprocess.check_call([npm, "install"], cwd=str(FRONTEND_CWD))

    procs = []
    if not frontend_only:
        backend = spawn(backend_cmd, BACKEND_CWD, "backend")
        procs.append(("backend", backend))
        time.sleep(1.0)
    frontend = spawn(frontend_cmd, FRONTEND_CWD, "frontend")
    procs.append(("frontend", frontend))

    print("[run_dev] running. Ctrl+C to stop.")
    if not frontend_only:
        print("[run_dev]   backend  → http://localhost:9010")
    print("[run_dev]   frontend → http://localhost:5173")

    exit_code = 0
    try:
        while True:
            for name, p in procs:
                rc = p.poll()
                if rc is not None:
                    print(f"[run_dev] {name} exited with code {rc} — shutting down the other.")
                    exit_code = rc or 1
                    raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run_dev] received shutdown signal.")
    finally:
        for name, p in procs:
            terminate(p, name)
        print("[run_dev] bye.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
