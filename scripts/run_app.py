"""Start FastAPI backend + Streamlit UI. Ctrl+C to stop all.

Milvus runs in-process via milvus-lite (./milvus.db) — no external server needed.
"""
import os
import sys
import time
import signal
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable

processes = []

# pymilvus ORM reads MILVUS_URI from .env at import time and rejects file paths.
# Pass an empty value so the ORM skips validation; code falls back to ./milvus.db.
api_env = os.environ.copy()
api_env["MILVUS_URI"] = ""


def wait_for_api(url: str, timeout_seconds: int = 30) -> None:
    """Poll the API health endpoint until it returns 2xx or the timeout elapses.

    Polls every 500 ms.  This is called after starting the FastAPI subprocess so
    Streamlit is not launched until the API is ready to accept connections.

    Args:
        url: Full URL to poll (e.g. ``"http://localhost:8000/health"``).
        timeout_seconds: Maximum seconds to wait before raising.

    Raises:
        RuntimeError: When the API has not become ready within ``timeout_seconds``.
    """
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"API did not become ready at {url}: {last_error}")


def shutdown(signum, frame):
    """SIGINT / SIGTERM handler: terminate all child processes and exit cleanly.

    Registered for both signals so that ``Ctrl+C`` and ``kill`` behave identically.
    Each process receives ``terminate()`` (SIGTERM) — forceful kill is not used so
    processes can flush buffers and release the SQLite/Milvus file locks.

    Args:
        signum: Signal number (unused, required by the signal handler signature).
        frame: Current stack frame (unused).
    """
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

time.sleep(1)

# 1 — FastAPI (milvus-lite starts in-process on first MilvusClient call)
print("Starting API ...")
api = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=ROOT,
    env=api_env,
)
processes.append(api)

print("Waiting for API health check ...")
wait_for_api("http://localhost:8000/health")

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
