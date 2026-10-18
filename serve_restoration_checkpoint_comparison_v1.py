"""Local-only report preview, with a queue for concurrent comparison images."""
import argparse
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "outputs/cctv_dgp_checkpoint_comparison_v1"


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    assert (REPORT / "comparison.html").is_file()
    server = Server(("127.0.0.1", args.port), partial(Handler, directory=str(REPORT)))
    record = {"pid": os.getpid(), "url": f"http://127.0.0.1:{server.server_port}/comparison.html",
              "root": str(REPORT), "listen_queue": 128, "http": "1.1"}
    path = ROOT / "scratch/restoration_checkpoint_comparison_server_r1.json"
    assert not path.exists(), "Retain existing server receipt"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
