import os
import sys
import time
import socket
import signal
import subprocess
import webbrowser
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000

FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 5173

BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
FRONTEND_URL = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"


processes = []


# ============================================================
# Utilities
# ============================================================

def print_banner():
    print()
    print("=" * 58)
    print("                 TENDER RAG")
    print("=" * 58)
    print()
    print("  Backend  :", BACKEND_URL)
    print("  Frontend :", FRONTEND_URL)
    print()
    print("  Starting services...")
    print()


def port_is_open(host: str, port: int) -> bool:
    """Check whether a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_port(
    host: str,
    port: int,
    timeout: float = 30.0,
) -> bool:
    """Wait until a service starts listening on the given port."""
    start = time.time()

    while time.time() - start < timeout:
        if port_is_open(host, port):
            return True

        time.sleep(0.5)

    return False


def terminate_process(process):
    """Terminate a subprocess safely."""
    if process is None:
        return

    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    except Exception:
        pass


# ============================================================
# Start backend
# ============================================================

def start_backend():
    print("[1/2] Starting FastAPI backend...")

    python_executable = sys.executable

    command = [
        python_executable,
        "-m",
        "uvicorn",
        "api:app",
        "--host",
        BACKEND_HOST,
        "--port",
        str(BACKEND_PORT),
    ]

    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=None,
            stderr=None,
            stdin=None,
        )
    except Exception as exc:
        print()
        print(f"ERROR: Could not launch FastAPI: {exc}")
        return False

    processes.append(process)

    # Give Uvicorn a moment to initialize.
    for _ in range(60):
        if process.poll() is not None:
            print()
            print("ERROR: FastAPI process exited unexpectedly.")
            print(f"Exit code: {process.returncode}")
            return False

        if port_is_open(BACKEND_HOST, BACKEND_PORT):
            print("      Backend ready.")
            print(f"      {BACKEND_URL}")
            return True

        time.sleep(0.5)

    print()
    print("ERROR: FastAPI did not become ready within 30 seconds.")
    print(f"Process status: {process.poll()}")
    return False
# ============================================================
# Start frontend
# ============================================================

def start_frontend():
    print()
    print("[2/2] Starting React frontend...")

    if not FRONTEND_DIR.exists():
        print()
        print("ERROR: frontend directory was not found:")
        print(FRONTEND_DIR)
        return False

    # Windows uses npm.cmd
    npm_command = "npm.cmd" if os.name == "nt" else "npm"

    command = [
        npm_command,
        "run",
        "dev",
        "--",
        "--host",
        FRONTEND_HOST,
        "--port",
        str(FRONTEND_PORT),
    ]

    process = subprocess.Popen(
        command,
        cwd=FRONTEND_DIR,
    )

    processes.append(process)

    if not wait_for_port(FRONTEND_HOST, FRONTEND_PORT, timeout=30):
        print()
        print("ERROR: React frontend failed to start.")
        print("Check the frontend terminal/output for details.")
        return False

    print("      Frontend ready.")
    print(f"      {FRONTEND_URL}")

    return True


# ============================================================
# Cleanup
# ============================================================

def cleanup():
    print()
    print()
    print("Stopping Tender RAG services...")

    # Stop frontend first
    for process in reversed(processes):
        terminate_process(process)

    processes.clear()

    print("Services stopped.")


def handle_exit_signal(signum, frame):
    cleanup()
    sys.exit(0)


# ============================================================
# Main
# ============================================================

def main():
    signal.signal(signal.SIGINT, handle_exit_signal)

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_exit_signal)

    print_banner()

    # Make sure required files exist.
    if not (ROOT_DIR / "api.py").exists():
        print("ERROR: api.py was not found.")
        return 1

    if not (FRONTEND_DIR / "package.json").exists():
        print("ERROR: frontend/package.json was not found.")
        return 1

    # Start backend.
    if not start_backend():
        cleanup()
        return 1

    # Start frontend.
    if not start_frontend():
        cleanup()
        return 1

    print()
    print("=" * 58)
    print("              TENDER RAG IS READY")
    print("=" * 58)
    print()
    print(f"  Open: {FRONTEND_URL}")
    print()
    print("  Press Ctrl+C to stop both services.")
    print("=" * 58)
    print()

    # Give Vite a moment to finish rendering.
    time.sleep(1)

    # Open browser automatically.
    webbrowser.open(FRONTEND_URL)

    # Keep launcher alive while child processes run.
    try:
        while True:
            time.sleep(1)

            # Detect unexpected backend/frontend shutdown.
            for process in processes:
                if process.poll() is not None:
                    print()
                    print("A Tender RAG service has stopped.")
                    cleanup()
                    return 1

    except KeyboardInterrupt:
        cleanup()
        return 0


if __name__ == "__main__":
    sys.exit(main())