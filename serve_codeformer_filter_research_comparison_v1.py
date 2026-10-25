"""Start the same opt-in app route on a measured loopback port; no training."""
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

from research_filter_restoration_web_v1 import ROOT, router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--receipt', type=Path, default=ROOT / 'scratch/codeformer_filter_research_server_v1.json')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        raise ValueError('Choose port0 for a measured available loopback port, or a valid exact port.')
    args.receipt = args.receipt.absolute()
    if not args.receipt.resolve(strict=False).is_relative_to((ROOT / 'scratch').resolve()) or args.receipt.suffix != '.json':
        raise ValueError('Write the exclusive server receipt under this workspace scratch directory.')
    for parent in (args.receipt, *args.receipt.parents):
        if parent.exists() or parent.is_symlink():
            item = parent.lstat()
            if stat.S_ISLNK(item.st_mode) or getattr(item, 'st_file_attributes', 0) & 1024:
                raise ValueError('Linked or reparse server-receipt paths are refused.')
    if args.receipt.exists():
        raise ValueError('Preserve the previous server receipt; select a new --receipt.')
    app = FastAPI(title='CodeFormer filter-fusion experimental restoration test')
    app.include_router(router)
    app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')
    @app.get('/')
    def home(): return RedirectResponse('/restoration-filter-test')
    @app.get('/favicon.ico')
    def favicon(): return Response(status_code=204)
    sock = socket.socket()
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    sock.bind(('127.0.0.1', args.port))
    record = {'pid': os.getpid(), 'url': 'http://127.0.0.1:%d/restoration-filter-test' % sock.getsockname()[1],
        'host': '127.0.0.1', 'port': sock.getsockname()[1], 'main_app_default_changed': False,
        'model_qualified': False, 'experimental_route': '/restoration-filter-test',
        'source_training_updates': 50, 'models': ['original', 'r2', 'fusion50'],
        'training_optimizer_VM_calls': 0}
    args.receipt.parent.mkdir(exist_ok=True)
    with args.receipt.open('x', encoding='utf-8') as f: json.dump(record, f, indent=2); f.write('\n')
    print(json.dumps(record), flush=True)
    uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=record['port'], log_level='warning')).run(sockets=[sock])


if __name__ == '__main__': main()
