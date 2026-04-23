"""
SilentBridge — One-command launcher.

Starts the FastAPI backend with uvicorn, which also serves the frontend.
Usage: python run.py
"""

import os
import sys
import webbrowser
import threading
import time


def main():
    # Ensure we can import the backend package
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # Create data directories
    data_dir = os.path.join(project_root, "backend", "data")
    os.makedirs(os.path.join(data_dir, "collected"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "models"), exist_ok=True)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        url = f"http://localhost:{port}"
        print(f"\n  ┌────────────────────────────────────────┐")
        print(f"  │  SilentBridge is running!               │")
        print(f"  │  Open: {url:<31s} │")
        print(f"  │  API docs: {url + '/docs':<27s} │")
        print(f"  └────────────────────────────────────────┘\n")
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    # Start uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
        )
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install -r backend/requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
