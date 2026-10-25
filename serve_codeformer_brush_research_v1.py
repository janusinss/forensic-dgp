"""Separate loopback Protect/Apply page. No VM connection, training or app change."""
import argparse
import json
import os
from pathlib import Path
import socket
import stat

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

import research_brush_restoration_v1 as B


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    B.need(0 <= args.port <= 65535, 'Use port0 or a valid loopback port.')
    receipt = args.receipt.absolute()
    B.need(receipt.resolve(strict=False).is_relative_to((B.ROOT / 'scratch').resolve())
           and receipt.suffix == '.json' and not receipt.exists(), 'Fresh workspace scratch JSON receipt required.')
    for part in (receipt, *receipt.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            B.need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024, 'Linked receipt path refused.')
    B.P.configure_manifest(args.manifest, args.manifest_sha)
    B.configure_sources(args.source_manifest, args.source_manifest_sha)
    app = FastAPI(title='Reviewed Protect and Apply restoration research')
    app.include_router(B.router)
    app.mount('/static', StaticFiles(directory=str(B.ROOT / 'static')), name='static')
    @app.get('/')
    def home(): return RedirectResponse(B.ROUTE)
    @app.get('/favicon.ico')
    def favicon(): return Response(status_code=204)
    sock = socket.socket()
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'): sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind(('127.0.0.1', args.port))
        value = {'pid': os.getpid(), 'url': 'http://127.0.0.1:%d%s' % (sock.getsockname()[1], B.ROUTE),
                 'source_manifest_sha256': args.source_manifest_sha, 'parent_manifest_sha256': args.manifest_sha,
                 'model_qualified': False, 'main_app_default_changed': False, 'neural_calls_at_startup': 0,
                 'maximum_compare_forwards': 2, 'maximum_bypass_forwards': 1, 'training_optimizer_VM_calls': 0}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2); stream.write('\n')
        print(json.dumps(value), flush=True)
        uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=sock.getsockname()[1], log_level='warning')).run(sockets=[sock])
    finally: sock.close()


if __name__ == '__main__': main()
