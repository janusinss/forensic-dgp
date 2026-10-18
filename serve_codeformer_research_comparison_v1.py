"""Start the same opt-in app route on a measured loopback port; no training."""
import argparse
import json
import os
from pathlib import Path
import socket

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from research_restoration_web import ROOT, router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--receipt', type=Path, default=ROOT / 'scratch/codeformer_r2_research_server_v1.json')
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError('Preserve the previous server receipt; select a new --receipt.')
    app = FastAPI(title='CodeFormer R2 experimental restoration test')
    app.include_router(router)
    app.mount('/static', StaticFiles(directory=str(ROOT / 'static')), name='static')
    @app.get('/')
    def home(): return RedirectResponse('/restoration-test')
    @app.get('/favicon.ico')
    def favicon(): return Response(status_code=204)
    sock = socket.socket(); sock.bind(('127.0.0.1', args.port))
    record = {'pid': os.getpid(), 'url': 'http://127.0.0.1:%d/restoration-test' % sock.getsockname()[1],
        'host': '127.0.0.1', 'port': sock.getsockname()[1], 'main_app_default_changed': False}
    args.receipt.parent.mkdir(exist_ok=True)
    with args.receipt.open('x', encoding='utf-8') as f: json.dump(record, f, indent=2); f.write('\n')
    print(json.dumps(record), flush=True)
    uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=record['port'], log_level='warning')).run(sockets=[sock])


if __name__ == '__main__': main()
