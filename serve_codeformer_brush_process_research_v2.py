"""Separate loopback brush page with a disposable inference worker per request."""
import argparse
import json
import os
from pathlib import Path
import socket

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

import research_brush_process_restoration_v2 as L


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha', required=True)
    parser.add_argument('--lifecycle-manifest', type=Path, required=True)
    parser.add_argument('--lifecycle-manifest-sha', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    L.B.need(0 <= args.port <= 65535, 'Use port0 or a valid loopback port.')
    receipt = L.ordinary(args.receipt)
    L.B.need(receipt.is_relative_to(L.ROOT / 'scratch') and receipt.suffix == '.json' and not receipt.exists(),
             'Fresh workspace scratch JSON receipt required.')
    config = L.configure_sources(args.lifecycle_manifest, args.lifecycle_manifest_sha,
                                 args.manifest, args.manifest_sha, args.source_manifest, args.source_manifest_sha)
    L.B.engine = L.B.BrushFlow(restorer=L.ProcessSelectedRestorer())
    app = FastAPI(title='Reviewed Protect and Apply restoration research')
    app.include_router(L.B.router)
    app.mount('/static', StaticFiles(directory=str(L.ROOT / 'static')), name='static')
    @app.get('/')
    def home():
        return RedirectResponse(L.B.ROUTE)
    @app.get('/favicon.ico')
    def favicon():
        return Response(status_code=204)
    sock = socket.socket()
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind(('127.0.0.1', args.port))
        value = {'pid': os.getpid(), 'url': 'http://127.0.0.1:%d%s' % (sock.getsockname()[1], L.B.ROUTE),
                 'source_manifest_sha256': args.source_manifest_sha, 'parent_manifest_sha256': args.manifest_sha,
                 'lifecycle_manifest': config['manifest'], 'model_qualified': False, 'main_app_default_changed': False,
                 'neural_calls_at_startup': 0, 'maximum_compare_forwards': 2, 'maximum_bypass_forwards': 1,
                 'training_optimizer_VM_calls': 0, 'worker_request_deadline_seconds': L.DEADLINE_SECONDS,
                 'process_lifecycle': 'one-disposable-worker-per-request-v2'}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        L.ordinary(receipt.parent, True)
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        print(json.dumps(value), flush=True)
        uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=sock.getsockname()[1], log_level='warning')).run(sockets=[sock])
    finally:
        sock.close()


if __name__ == '__main__':
    main()
