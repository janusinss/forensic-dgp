"""Run the implementation locally; keep the server attached to this console."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser

from .setup_models import ModelSetupError, ROOT, verify_models

HOST = "127.0.0.1"
from .managed_paths import data_path

URL_RECORD = data_path(ROOT, 'scratch') / 'implementation_launch.json'


def _open_when_ready(url: str) -> None:
    deadline = time.monotonic() + 15
    # This probe is loopback HTTP only. Avoid proxy discovery and creating an
    # unrelated HTTPS context while waiting for the local application.
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.ProxyHandler({}))
    opener.add_handler(urllib.request.HTTPHandler())
    while time.monotonic() < deadline:
        try:
            with opener.open(url + "/health", timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(url, new=2)
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    print(f"Browser was not opened: the health check did not pass within 15 seconds. URL: {url}", flush=True)


def _record_url(url: str, port: int) -> None:
    URL_RECORD.parent.mkdir(parents=True, exist_ok=True)
    temporary = URL_RECORD.with_name(URL_RECORD.name + f".{os.getpid()}.tmp")
    record = {
        "url": url,
        "host": HOST,
        "port": port,
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(URL_RECORD)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None, help="loopback port; 0 asks Windows for an available port")
    parser.add_argument("--no-browser", action="store_true", help="print the URL without opening a browser")
    parser.add_argument("--no-pause", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        port = args.port if args.port is not None else int(os.environ.get("IMPLEMENTATION_PORT", "8000"))
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
    except ValueError as exc:
        print(f"[ERROR] Invalid IMPLEMENTATION_PORT/--port: {exc}", file=sys.stderr)
        return 1
    try:
        verify_models()
        import uvicorn
    except ModelSetupError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"[ERROR] A dependency is missing ({exc}). Run install.bat first.", file=sys.stderr)
        return 1

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            listener.bind((HOST, port))
            listener.listen(socket.SOMAXCONN)
        except OSError as exc:
            print(
                f"[ERROR] Cannot listen on {HOST}:{port}: {exc}. "
                "Close the application using that port or set IMPLEMENTATION_PORT to another port (0 selects an available one).",
                file=sys.stderr,
            )
            return 1
        actual_port = listener.getsockname()[1]
        url = f"http://{HOST}:{actual_port}"
        try:
            _record_url(url, actual_port)
        except OSError as exc:
            print(f"[ERROR] Could not record the application URL: {exc}", file=sys.stderr)
            return 1
        print(f"Application URL: {url}", flush=True)
        print(f"URL record: {URL_RECORD}", flush=True)
        print("Press Ctrl+C in this window to stop the server.", flush=True)
        if not args.no_browser and os.environ.get("IMPLEMENTATION_NO_BROWSER", "").lower() not in {"1", "true", "yes"}:
            threading.Thread(target=_open_when_ready, args=(url,), daemon=True, name="browser-after-health").start()
        config = uvicorn.Config("app:app", host=HOST, port=actual_port, reload=False, workers=1)
        server = uvicorn.Server(config)
        server.run(sockets=[listener])
        return 0 if server.started else 1


if __name__ == "__main__":
    raise SystemExit(main())
