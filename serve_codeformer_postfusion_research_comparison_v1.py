"""Explicit separate loopback comparison; no main application or training import."""
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

from research_postfusion_restoration_v1 import ROOT, ROUTE, ARMS, QUALITY_STATUS, configure_manifest, router


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--receipt', type=Path, default=ROOT / 'scratch/codeformer_postfusion_research_server_v1.json')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535: raise ValueError('Use port0 or an exact valid loopback port.')
    args.receipt = args.receipt.absolute()
    if ('..' in args.receipt.parts or not args.receipt.is_relative_to(ROOT / 'scratch')
            or not args.receipt.resolve(strict=False).is_relative_to((ROOT / 'scratch').resolve())
            or args.receipt.suffix != '.json'):
        raise ValueError('Exclusive server receipt must stay in this workspace scratch directory.')
    for part in (args.receipt, *args.receipt.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
                raise ValueError('Linked server receipt path refused.')
    if args.receipt.exists(): raise ValueError('Preserve the old receipt; use a fresh --receipt path.')
    configure_manifest(args.manifest, args.manifest_sha)  # No Torch/model construction.
    app = FastAPI(title='Experimental U96 and post-fusion u64 restoration comparison')
    app.include_router(router)
    app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')
    @app.get('/')
    def home(): return RedirectResponse(ROUTE)
    @app.get('/favicon.ico')
    def favicon(): return Response(status_code=204)
    sock = socket.socket()
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'): sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        sock.bind(('127.0.0.1', args.port))
        record = {'pid': os.getpid(), 'url': 'http://127.0.0.1:%d%s' % (sock.getsockname()[1], ROUTE),
                  'host': '127.0.0.1', 'port': sock.getsockname()[1], 'model_qualified': False,
                  'main_app_default_changed': False, 'experimental_route': ROUTE, 'models': list(ARMS),
                  'quality_status': QUALITY_STATUS, 'training_optimizer_VM_calls': 0,
                  'maximum_forwards_per_request': 2, 'CPU_threads': 4,
                  'test_manifest_sha256': args.manifest_sha}
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False); stream.write('\n')
        print(json.dumps(record), flush=True)
        uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=record['port'], log_level='warning')).run(sockets=[sock])
    finally: sock.close()


if __name__ == '__main__': main()
