"""Start FastAPI backend + Streamlit UI. Ctrl+C to stop all.

Milvus runs in-process via milvus-lite (./milvus.db) — no external server needed.
"""
import os
import sys
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

processes = []

# pymilvus ORM reads MILVUS_URI from .env at import time and rejects file paths.
# Pass an empty value so the ORM skips validation; code falls back to ./milvus.db.
api_env = os.environ.copy()
api_env["MILVUS_URI"] = ""


def shutdown(signum, frame):
    print("\nShutting down...")
    for p in processes:
        p.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# Free ports if held by stale processes
for _port in (8000, 8501):
    subprocess.run(f"lsof -ti :{_port} | xargs kill -9", shell=True,
                   capture_output=True)

import time; time.sleep(1)

# 1 — FastAPI (milvus-lite starts in-process on first MilvusClient call)
print("Starting API ...")
api = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=ROOT,
    env=api_env,
)
processes.append(api)

# 2 — Streamlit
print("Starting UI ...")
ui = subprocess.Popen(
    [PYTHON, "-m", "streamlit", "run", "src/ui/app.py", "--server.port", "8501"],
    cwd=ROOT,
)
processes.append(ui)

print("API:    http://localhost:8000")
print("UI:     http://localhost:8501")
print("Press Ctrl+C to stop.\n")

api.wait()
ui.wait()
