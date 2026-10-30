"""Bounded, local capture and source-linked evidence for the restoration workflow."""
from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import quote, urlsplit, urlunsplit
import uuid

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
import psutil
from pydantic import BaseModel, ConfigDict, StrictInt
from starlette.concurrency import run_in_threadpool

from .managed_paths import data_path

ROOT = Path(__file__).resolve().parents[1]
STORE = data_path(ROOT, 'scratch', 'capture_requests')
TOKENS = set()
MAX_BYTES = 24 * 1024 * 1024
MAX_RETAINED_BYTES = 256 * 1024 * 1024
FRAMES = {}
CROPS = {}
STREAMS = {}
LOCK = threading.RLock()
DETECTION_LOCK = threading.Lock()
PRIVATE = tuple(ipaddress.ip_network(v) for v in ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def worker_python():
    from .environment import expected_environment
    path = expected_environment() / 'capture' / 'Scripts' / 'python.exe'
    if not path.is_file():
        raise RuntimeError('Capture runtime is unavailable. Run install.bat to install it.')
    return str(path)


def _reserve(extra):
    used = sum(path.stat().st_size for path in STORE.rglob('*') if path.is_file()) if STORE.exists() else 0
    if used + extra > MAX_RETAINED_BYTES:
        raise ValueError('Capture evidence storage is full. Preserve or move existing records before starting another capture.')


def _guard(request, token=True):
    host = request.headers.get('host', '')
    # The real app binds loopback. Reject DNS rebinding even when a hostile page can read its own origin.
    if host:
        try:
            hostname = urlsplit('http://' + host).hostname
        except ValueError:
            raise HTTPException(403, 'Invalid host.')
        if hostname not in ('localhost', '127.0.0.1', '::1'):
            raise HTTPException(403, 'Local access only.')
    origin = request.headers.get('origin')
    if origin:
        try:
            parsed = urlsplit(origin)
            if parsed.scheme not in ('http', 'https') or parsed.netloc.lower() != host.lower():
                raise ValueError
        except ValueError:
            raise HTTPException(403, 'Origin is not allowed.')
    if token:
        supplied = request.headers.get('x-capture-token', '')
        with LOCK:
            valid = any(secrets.compare_digest(supplied, known) for known in TOKENS)
        if not valid:
            raise HTTPException(403, 'Capture session token required.')
        return supplied


def _image(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError('Frame exceeds the allowed size.')
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != 'PNG' or im.mode not in ('RGB', 'RGBA') or getattr(im, 'n_frames', 1) != 1:
                raise ValueError('Use a single RGB or opaque RGBA PNG frame.')
            w, h = im.size
            if min(w, h) < 1 or max(w, h) > 4096 or w * h > 8388608:
                raise ValueError('Frame dimensions exceed the allowed size.')
            im.load()
            if im.mode == 'RGBA' and im.getchannel('A').getextrema() != (255, 255):
                raise ValueError('Frame transparency is not supported.')
            return im.convert('RGB')
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid image frame.') from exc


def detect_faces(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    try:
        result = subprocess.run([worker_python(), '-B', str(Path(__file__).with_name('capture_worker.py')), 'detect'],
                                input=buffer.getvalue(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=15, check=True, cwd=ROOT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if len(result.stdout) > 16384:
            raise ValueError
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        raise RuntimeError('Face detection failed or timed out.') from exc


def _freeze_inner(raw, source_kind, source_time, owner):
    if source_kind not in ('webcam', 'video', 'rtsp'):
        raise ValueError('Choose a valid source type.')
    if source_time is not None and (not math.isfinite(source_time) or source_time < 0 or source_time > 86400):
        raise ValueError('Invalid source time.')
    im = _image(raw)
    warning = None
    try:
        faces = detect_faces(im)
    except RuntimeError:
        faces = []
        warning = 'Face detection is unavailable. Select the face area manually.'
    with LOCK:
        if len(FRAMES) >= 64:
            raise ValueError('Capture session is full; restart the app to begin another session.')
        _reserve(len(raw))
        ident = uuid.uuid4().hex
        path = STORE / ident
        path.mkdir(parents=True, exist_ok=False)
        (path / 'frame.png').write_bytes(raw)
        FRAMES[ident] = {'path': path, 'sha': _sha(raw), 'size': im.size,
                         'source_kind': source_kind, 'source_time': source_time, 'owner': owner}
    return {'frame_id': ident, 'width': im.width, 'height': im.height,
            'faces': faces, 'detection_warning': warning, 'frame_url': '/capture/frames/' + ident + '/image'}


def _freeze(raw, source_kind, source_time, owner):
    if not DETECTION_LOCK.acquire(blocking=False):
        raise HTTPException(409, 'Another frame is being inspected.')
    try:
        return _freeze_inner(raw, source_kind, source_time, owner)
    finally:
        DETECTION_LOCK.release()


def _frame(ident, owner=None):
    record = FRAMES.get(ident)
    if not record or (owner is not None and record['owner'] != owner):
        raise HTTPException(404, 'Frame is unavailable.')
    raw = (record['path'] / 'frame.png').read_bytes()
    if _sha(raw) != record['sha']:
        raise HTTPException(409, 'Frame evidence changed; capture again.')
    return record, raw


class Rectangle(BaseModel):
    model_config = ConfigDict(extra='forbid')
    x: StrictInt
    y: StrictInt
    width: StrictInt
    height: StrictInt


class StreamRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str
    username: str = ''
    password: str = ''


def _crop(ident, rect, owner):
    with LOCK:
        record, raw = _frame(ident, owner)
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        if x < 0 or y < 0 or min(w, h) < 16 or max(w, h) > 1600 or x + w > record['size'][0] or y + h > record['size'][1]:
            raise ValueError('Choose an in-bounds crop between 16 and 1600 pixels per side.')
        if len(CROPS) >= 64:
            raise ValueError('Capture session is full; restart the app to begin another session.')
        image = _image(raw).crop((x, y, x + w, y + h))
        out = io.BytesIO()
        image.save(out, format='PNG')
        crop = out.getvalue()
        capture_id = uuid.uuid4().hex
        metadata = {'capture_id': capture_id, 'source_kind': record['source_kind'],
                    'source_time': record['source_time'], 'rectangle': [x, y, w, h],
                    'frame_sha256': record['sha'], 'crop_sha256': _sha(crop),
                    'provenance': 'Unaligned native-pixel crop of the frozen source frame.'}
        _reserve(len(crop) + 4096)
        target = record['path'] / capture_id
        target.mkdir()
        (target / 'crop.png').write_bytes(crop)
        (target / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True), encoding='utf8')
        CROPS[capture_id] = {'frame': ident, 'path': target, 'metadata': metadata}
    return {'capture_id': capture_id, 'image': 'data:image/png;base64,' + base64.b64encode(crop).decode('ascii'), 'metadata': metadata}


def evidence_for_crop(contents, capture_id):
    with LOCK:
        record = CROPS.get(capture_id)
        if not record:
            raise ValueError('Capture evidence is unavailable in this app session.')
        _, frame = _frame(record['frame'])
        crop = (record['path'] / 'crop.png').read_bytes()
        meta = (record['path'] / 'metadata.json').read_bytes()
        if _sha(crop) != record['metadata']['crop_sha256'] or _sha(contents) != _sha(crop) or json.loads(meta) != record['metadata']:
            raise ValueError('Capture evidence differs from the selected crop.')
        return {'capture/frame.png': frame, 'capture/crop.png': crop, 'capture/metadata.json': meta}


def _stream_url(value):
    if len(value.url) > 2048 or len(value.username) > 256 or len(value.password) > 256:
        raise ValueError('Stream details are too long.')
    try:
        u = urlsplit(value.url)
        if u.scheme.lower() != 'rtsp' or not u.hostname or u.username or u.password or u.fragment or not (1 <= (u.port or 554) <= 65535):
            raise ValueError
        if any(ord(c) < 32 or ord(c) == 127 for c in value.url + value.username + value.password):
            raise ValueError
        # Resolve in a disposable process so a stuck system resolver cannot strand a request.
        resolver = subprocess.run([sys.executable, '-B', '-c',
            'import json,socket,sys;print(json.dumps(sorted({x[4][0] for x in socket.getaddrinfo(sys.argv[1],None,socket.AF_INET,socket.SOCK_STREAM)})))',
            u.hostname], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5,
            check=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        addresses = set(json.loads(resolver.stdout))
        if not addresses or any(not any(ipaddress.ip_address(ip) in n for n in PRIVATE) for ip in addresses):
            raise ValueError
        ip = sorted(addresses)[0]
        auth = ''
        if value.username or value.password:
            auth = quote(value.username, safe='') + ':' + quote(value.password, safe='') + '@'
        return urlunsplit(('rtsp', auth + ip + ':' + str(u.port or 554), u.path or '/', u.query, ''))
    except (ValueError, OSError, subprocess.SubprocessError):
        raise ValueError('Use an RTSP address resolving only to a private LAN IPv4 address.')


def _stop(record):
    process = record['process']
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def _read_stream(record):
    process = record['process']
    try:
        while True:
            header = process.stdout.read(4)
            if len(header) != 4:
                break
            size = struct.unpack('!I', header)[0]
            if not 0 < size <= MAX_BYTES:
                break
            data = process.stdout.read(size)
            if len(data) != size:
                break
            _image(data)
            with LOCK:
                record['latest'] = data
    except (OSError, ValueError):
        pass
    finally:
        with LOCK:
            record['failed'] = True


def _watch_stream(ident, record):
    while True:
        time.sleep(1)
        with LOCK:
            if ident not in STREAMS:
                return
            expired = time.monotonic() - record['last_poll'] > 30 or time.monotonic() - record['started'] > 600
            failed = record['failed'] or record['process'].poll() is not None or (record['latest'] is None and time.monotonic() - record['started'] > 15)
            try:
                too_large = psutil.Process(record['process'].pid).memory_info().rss > 1024**3
            except psutil.NoSuchProcess:
                too_large = False
            if expired or failed or too_large:
                STREAMS.pop(ident, None)
                _stop(record)
                return


def _start_stream(value, owner):
    url = _stream_url(value)
    with LOCK:
        if STREAMS:
            raise HTTPException(409, 'Stop the current stream first.')
        worker = Path(__file__).with_name('capture_worker.py')
        proc = subprocess.Popen([worker_python(), '-B', str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, cwd=ROOT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            proc.stdin.write((json.dumps({'url': url}) + '\n').encode())
            proc.stdin.close()
        except BaseException:
            _stop({'process': proc})
            raise
        ident = uuid.uuid4().hex
        record = {'process': proc, 'latest': None, 'started': time.monotonic(), 'last_poll': time.monotonic(), 'failed': False, 'owner': owner}
        STREAMS[ident] = record
        threading.Thread(target=_read_stream, args=(record,), daemon=True).start()
        threading.Thread(target=_watch_stream, args=(ident, record), daemon=True).start()
    return {'stream_id': ident}


def register(app):
    @app.on_event('shutdown')
    def stop_owned_streams():
        with LOCK:
            records = list(STREAMS.values())
            STREAMS.clear()
        for record in records:
            _stop(record)

    @app.get('/capture/status')
    def status(request: Request):
        _guard(request, False)
        with LOCK:
            if len(TOKENS) >= 256:
                raise HTTPException(429, 'Capture sessions are full; restart the app.')
            token = secrets.token_urlsafe(32)
            TOKENS.add(token)
        return JSONResponse({'token': token, 'available': True, 'rtsp_available': True}, headers={'Cache-Control': 'no-store'})

    @app.post('/capture/frames')
    async def frame(request: Request, file: UploadFile = File(...), source_kind: str = Form('video'), source_time: float | None = Form(None)):
        owner = _guard(request)
        raw = await file.read(MAX_BYTES + 1)
        try:
            return await run_in_threadpool(_freeze, raw, source_kind, source_time, owner)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, OSError) as exc:
            raise HTTPException(503, 'Unable to inspect the frame.') from exc

    @app.get('/capture/frames/{ident}/image')
    def image(request: Request, ident: str):
        owner = _guard(request)
        with LOCK:
            _, raw = _frame(ident, owner)
        return Response(raw, media_type='image/png', headers={'Cache-Control': 'no-store'})

    @app.post('/capture/frames/{ident}/crop')
    def crop(request: Request, ident: str, rect: Rectangle):
        owner = _guard(request)
        try:
            return _crop(ident, rect, owner)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/capture/streams')
    async def stream(request: Request, value: StreamRequest):
        owner = _guard(request)
        try:
            return await run_in_threadpool(_start_stream, value, owner)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (OSError, RuntimeError) as exc:
            raise HTTPException(503, 'Unable to start the camera worker.') from exc

    @app.get('/capture/streams/{ident}/frame')
    def stream_frame(request: Request, ident: str):
        owner = _guard(request)
        with LOCK:
            record = STREAMS.get(ident)
            if not record or record['owner'] != owner or record['failed']:
                raise HTTPException(404, 'Stream stopped or could not connect.')
            record['last_poll'] = time.monotonic()
            raw = record['latest']
        return Response(raw or b'', status_code=200 if raw else 202, media_type='image/png', headers={'Cache-Control': 'no-store'})

    @app.delete('/capture/streams/{ident}')
    def stop(request: Request, ident: str):
        owner = _guard(request)
        with LOCK:
            record = STREAMS.get(ident)
            if record and record['owner'] == owner:
                STREAMS.pop(ident)
            else:
                record = None
        if record:
            _stop(record)
        return {'stopped': True}
