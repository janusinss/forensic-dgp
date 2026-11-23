"""Local reviewed-mask CodeFormer prototype; restoration and training are separate."""
from __future__ import annotations
import argparse
import base64
import binascii
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import numpy as np
from PIL import Image, ImageOps
from completion_process_guard_v1 import OwnedWorker
from pretrained_completion_finetune_r1 import validate_candidate_manifest

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(r'C:\xampp\htdocs\YEAR 4\Testing')
CHECKPOINT = ROOT / 'outputs/completion_codeformer_finetune_v1_r1_candidate/codeformer_completion_finetune_r1.pth'
MODEL_LABEL = 'Our trained completion R1 — experimental'
WORKER = ROOT / 'scripts/completion_codeformer_finetune_r1_worker.py'
OUTPUT = ROOT / 'outputs/completion_codeformer_finetune_r1_app'
UPLOAD_LIMIT = 8 * 1024**2
BODY_LIMIT = 12 * 1024**2
WORKER_CAP = 120
HEX_ID = re.compile(r'^[a-f0-9]{24}$')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def decode_data_url(value, *, mask=False):
    need(isinstance(value, str) and ',' in value, 'Choose a PNG, JPEG or WebP face crop.')
    prefix, encoded = value.split(',', 1)
    allowed = ('data:image/png;base64',) if mask else ('data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64')
    need(prefix in allowed, 'Use a PNG mask.' if mask else 'Use a PNG, JPEG or WebP image.')
    need(len(encoded) <= (UPLOAD_LIMIT * 4 // 3 + 8), 'Choose an image smaller than 8 MB.')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('Cannot read this image encoding.') from exc
    need(0 < len(raw) <= UPLOAD_LIMIT, 'Choose an image smaller than 8 MB.')
    try:
        with Image.open(io.BytesIO(raw)) as im:
            need(im.format in (('PNG',) if mask else ('PNG', 'JPEG', 'WEBP')), 'Unsupported image format.')
            if mask:
                need(im.size == (256, 256), 'Upload a matching 256 by 256 black-and-white PNG mask.')
                a = np.array(im.convert('RGB'))
                need(np.all(a[..., 0] == a[..., 1]) and np.all(a[..., 1] == a[..., 2]) and np.isin(a, [0, 255]).all(), 'The mask must contain only black and white pixels.')
                if 'A' in im.getbands():
                    need(np.array(im.getchannel('A')).min() == 255, 'Use an opaque black-and-white mask.')
                return raw, a[..., 0]
            need(min(im.size) >= 32 and max(im.size) <= 4096 and im.width * im.height <= 16_000_000, 'Use one already cropped face, 32 to 4096 pixels per side.')
            im = ImageOps.exif_transpose(im)
            if im.mode in ('RGBA', 'LA') or 'transparency' in im.info:
                rgba = im.convert('RGBA')
                background = Image.new('RGBA', rgba.size, (127, 127, 127, 255))
                im = Image.alpha_composite(background, rgba).convert('RGB')
            else:
                im = im.convert('RGB')
            im.load()
            return raw, im.copy()
    except (OSError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ValueError('Cannot read this image. Choose a valid face crop.') from exc


def visibility(mask, bounds):
    left, top, right, bottom = bounds
    unavailable = mask != 0
    unavailable = unavailable.copy()
    unavailable[:top] = True
    unavailable[bottom:] = True
    unavailable[:, :left] = True
    unavailable[:, right:] = True
    yy, xx = np.mgrid[:256, :256]
    ellipse = ((xx - 127.5) / 83)**2 + ((yy - 137) / 104)**2 <= 1
    bands = {'left_eye': (60, 84, 119, 133), 'right_eye': (137, 84, 196, 133), 'nose': (108, 119, 149, 166), 'mouth': (82, 165, 174, 208)}
    fractions = {key: float(unavailable[y1:y2, x1:x2].mean()) for key, (x1, y1, x2, y2) in bands.items()}
    covered = float(unavailable[ellipse].mean())
    rejected = covered >= .8 or unavailable.mean() >= .85 or (fractions['left_eye'] >= .85 and fractions['right_eye'] >= .85 and fractions['nose'] >= .9 and fractions['mouth'] >= .9)
    return {'rejected': bool(rejected), 'unavailable_face_fraction': covered, 'feature_unavailable_fractions': fractions, 'scope': 'Frontal-crop geometry heuristic conditional on reviewed mask; padding is unavailable. Not a face detector.'}


def audit_worker_result(path, expected_inputs, expected_mask):
    """Recompute delivered PNG/raw bindings without loading the neural model."""
    receipt = json.loads((path/'receipt.json').read_text(encoding='utf-8-sig'))
    need(receipt.get('complete') is True and receipt.get('neural_calls') == int(bool(expected_mask.any())), 'Completion receipt differs from the reviewed request.')
    need(receipt.get('inputs_sha256') == expected_inputs, 'Worker receipt is not bound to the submitted input and mask.')
    need(all(sha(path/name) == digest for name,digest in expected_inputs.items()), 'Submitted input or mask changed during completion.')
    need(receipt.get('outputs_sha256') == {name:sha(path/name) for name in ('result.png','stages.npz')}, 'Worker output fingerprints differ.')
    with Image.open(path/'result.png') as im:
        need(im.size == (256,256) and im.mode == 'RGB', 'Completion output must be RGB 256 by 256.')
        result = np.array(im)
    with Image.open(path/'input.png') as im:
        original = np.array(im)
    with Image.open(path/'mask.png') as im:
        need(np.array_equal(np.array(im),expected_mask), 'Reviewed mask changed.')
    need(np.array_equal(result[expected_mask==0],original[expected_mask==0]), 'Visible pixels changed outside the reviewed removal area.')
    with np.load(path/'stages.npz',allow_pickle=False) as stages:
        need(set(stages.files) == {'input_rgb','removal_mask','completed_rgb','output_rgb'}, 'Raw-stage fields differ.')
        raw = stages['completed_rgb']
        need(raw.shape == (256,256,3) and raw.dtype == np.float32 and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1, 'Invalid raw completion stage.')
        need(stages['input_rgb'].dtype==np.uint8 and np.array_equal(stages['input_rgb'],original), 'Raw prepared input differs.')
        need(stages['removal_mask'].dtype==np.uint8 and np.array_equal(stages['removal_mask'],expected_mask//255), 'Raw reviewed mask differs.')
        need(stages['output_rgb'].dtype==np.uint8 and np.array_equal(stages['output_rgb'],result), 'Saved PNG differs from raw output bytes.')
        expected = np.rint(raw*np.float32(255)).clip(0,255).astype(np.uint8)
        expected[expected_mask==0] = original[expected_mask==0]
        need(np.array_equal(expected,result), 'Saved PNG does not replay from the raw completion stage.')
        reference = original.astype(np.float32)/np.float32(255)
        need(np.array_equal(raw[expected_mask==0],reference[expected_mask==0]), 'Raw stage changed visible pixels.')
    if not expected_mask.any():
        need(np.array_equal(result,original), 'Empty-mask bypass changed the original input.')
    return receipt


def wait_owned_worker(process, shutdown_event, timeout):
    """Only the request thread touches worker lifecycle or member handles."""
    deadline = time.monotonic()+timeout
    while True:
        if shutdown_event.is_set():
            process.kill()
            process.wait_tree(timeout=15)
            raise RuntimeError('Completion service is shutting down; saved evidence is retained.')
        remaining = deadline-time.monotonic()
        if remaining<=0:
            raise subprocess.TimeoutExpired(process.args,timeout)
        try:
            return process.wait_tree(timeout=min(.25,remaining))
        except subprocess.TimeoutExpired:
            if time.monotonic()>=deadline:
                raise subprocess.TimeoutExpired(process.args,timeout)


class CompletionApp:
    def __init__(self, *, admission_sha, worker_sha, guard_sha, candidate_manifest_sha, storage=OUTPUT, checkpoint=CHECKPOINT, python=None):
        self.storage = Path(storage)
        self.checkpoint = Path(checkpoint)
        self.python = str(python or sys.executable)
        self.admission = self.storage / 'source_admission.json'
        self.admission_sha = admission_sha
        self.worker_sha = worker_sha
        self.guard_sha = guard_sha
        self.candidate_manifest_sha = candidate_manifest_sha
        self.sessions = {}
        self.registry_lock = threading.Lock()
        self.generation_lock = threading.Lock()
        self.owned_processes = set()
        self.process_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.check_sources(check_checkpoint=True)
        (self.storage / 'sessions').mkdir(parents=True, exist_ok=True)

    def check_sources(self, check_checkpoint=False):
        need(sha(self.admission) == self.admission_sha, 'CodeFormer source review fingerprint changed.')
        admission = json.loads(self.admission.read_text(encoding='utf-8-sig'))
        need(admission.get('complete') is True, 'CodeFormer source review is incomplete.')
        for item in admission['source_files']:
            path = Path(item['path'])
            need(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Reviewed CodeFormer source changed: ' + path.name)
        need(sha(WORKER) == self.worker_sha, 'CodeFormer worker fingerprint changed.')
        need(sha(ROOT/'completion_process_guard_v1.py') == self.guard_sha, 'Worker process-tree guard fingerprint changed.')
        self.candidate_metadata = validate_candidate_manifest(self.candidate_manifest_sha, checkpoint=self.checkpoint, check_checkpoint=check_checkpoint)

    def new_session(self):
        with self.registry_lock:
            need(len(self.sessions) < 100, 'This test session is full. Restart the page server to begin a new session; saved evidence stays retained.')
            ident = uuid.uuid4().hex[:24]
            path = self.storage / 'sessions' / ident
            path.mkdir(exist_ok=False)
            self.sessions[ident] = {'path': path, 'assets': {}, 'prepared': False}
            return ident, self.sessions[ident]

    def session(self, ident):
        need(isinstance(ident, str) and HEX_ID.fullmatch(ident) and ident in self.sessions, 'Upload this face again before generating.')
        return self.sessions[ident]

    def publish(self, session, names):
        for name in names:
            session['assets'][name] = pin(session['path'] / name)

    def prepare(self, data):
        raw, image = decode_data_url(data.get('image'))
        ident, session = self.new_session()
        path = session['path']
        (path / 'upload.bin').write_bytes(raw)
        image.save(path / 'original.png')
        w, h = image.size
        side = max(w, h)
        padded = Image.new('RGB', (side, side), (127, 127, 127))
        left, top = (side - w)//2, (side - h)//2
        padded.paste(image, (left, top))
        padded.resize((256, 256), Image.Resampling.LANCZOS).save(path / 'input.png')
        # Nearest support is conservative: exclude partly interpolated padding.
        bounds = [int(np.ceil(left*256/side)), int(np.ceil(top*256/side)), int(np.floor((left+w)*256/side)), int(np.floor((top+h)*256/side))]
        metadata = {'id': ident, 'original_size': [w, h], 'delivery_size': [256, 256], 'prepared_bounds': bounds, 'policy': 'Original decoded EXIF-oriented upload; neutral square padding then Lanczos resize once to 256. Visible preservation is relative to this prepared input.', 'restoration_calls': 0, 'training_calls': 0}
        save_json(path / 'request.json', metadata)
        session.update({'prepared': True, 'metadata': metadata})
        self.publish(session, ['input.png', 'original.png', 'upload.bin', 'request.json'])
        return {'id': ident, 'input': f'/asset/{ident}/input.png', 'width': 256, 'height': 256, 'preparation_note': metadata['policy']}

    def generate(self, data):
        need(data.get('mask_reviewed') is True and data.get('visible_structure_confirmed') is True, 'Confirm the removal area and sufficient visible facial structure first.')
        parent = self.session(data.get('id'))
        need(parent['prepared'], 'Start from the original uploaded face.')
        _, mask = decode_data_url(data.get('mask'), mask=True)
        bounds = parent['metadata']['prepared_bounds']
        left, top, right, bottom = bounds
        outside = mask.copy()
        outside[top:bottom, left:right] = 0
        need(not outside.any(), 'Keep the padding unmarked; paint only the covering on the face crop.')
        review = visibility(mask, bounds)
        need(not review['rejected'], 'Too little visible face remains. Choose a less-covered or clearer face crop.')
        if not self.generation_lock.acquire(blocking=False):
            raise BlockingIOError('One estimate is running. Wait for it to finish.')
        try:
            self.check_sources()
            ident, session = self.new_session()
            path = session['path']
            for name in ('upload.bin', 'original.png', 'input.png'):
                expected = parent['assets'][name]
                need(sha(parent['path']/name) == expected['sha256'], 'Original upload evidence changed.')
                shutil.copyfile(parent['path']/name, path/name)
            Image.fromarray(mask).save(path / 'mask.png')
            save_json(path / 'request.json', {**parent['metadata'], 'id': ident, 'source_id': data['id'], 'mask_reviewed': True, 'visible_structure_confirmed': True, 'assisted': True, 'automatic_detection_used': False, 'visibility_review': review, 'worker_sha256': self.worker_sha, 'source_admission_sha256': self.admission_sha, 'process_guard_sha256': self.guard_sha, 'candidate_manifest_sha256': self.candidate_manifest_sha, 'model_label': MODEL_LABEL})
            expected_inputs = {name:sha(path/name) for name in ('input.png','mask.png','request.json')}
            start = time.monotonic()
            command = [self.python, '-B', str(WORKER), '--request-dir', str(path), '--checkpoint', str(self.checkpoint), '--candidate-manifest-sha', self.candidate_manifest_sha]
            process = None
            with (path/'worker.log').open('wb') as log:
                process = OwnedWorker(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
                with self.process_lock:
                    self.owned_processes.add(process)
                save_json(path/'supervisor.json', {'child_pid': process.pid, 'cap_seconds': WORKER_CAP, 'status': 'running', 'process_ownership': process.ownership})
                try:
                    code = wait_owned_worker(process, self.shutdown_event, WORKER_CAP)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait_tree(timeout=15)
                    save_json(path/'supervisor.json', {'child_pid': process.pid, 'cap_seconds': WORKER_CAP, 'status': 'timeout', 'seconds': time.monotonic()-start, 'returncode': process.returncode, 'process_ownership': process.ownership, 'active_owned_processes_after_kill': process.active_processes()})
                    raise TimeoutError('Completion exceeded 120 seconds on this computer. Saved evidence is retained; try a clearer crop.')
            elapsed = time.monotonic()-start
            save_json(path/'supervisor.json', {'child_pid': process.pid, 'cap_seconds': WORKER_CAP, 'status': 'finished', 'seconds': elapsed, 'returncode': code, 'process_ownership': process.ownership, 'active_owned_processes': process.active_processes(), 'process_tree_memory': process.memory_usage()})
            need(code == 0, 'CodeFormer could not complete this face. Its error log was retained for review.')
            receipt = audit_worker_result(path, expected_inputs, mask)
            need(receipt.get("format") == "reviewed-codeformer-finetune-r1-completion-cpu-v1" and receipt.get("candidate_manifest_sha256") == self.candidate_manifest_sha, "Worker did not use the admitted R1 candidate.")
            if mask.any():
                need(receipt.get("provenance", {}).get("weights_sha256") == self.candidate_metadata["checkpoint"]["sha256"], "Worker candidate checkpoint differs.")
            save_json(path/'app_audit.json', {'complete': True, 'independent_of_worker_compositing': True, 'raw_png_replay_verified': True, 'seconds_including_worker_startup': elapsed, 'process_ownership': process.ownership, 'active_owned_processes': process.active_processes(), 'process_tree_memory': process.memory_usage(), 'outside_mask_changed_bytes': 0, 'prepared_input_sha256': sha(path/'input.png'), 'mask_sha256': sha(path/'mask.png'), 'result_sha256': sha(path/'result.png'), 'neural_calls': receipt['neural_calls'], 'model_qualified': False, 'automatic': False, 'restoration_calls': 0, 'training_calls': 0})
            names = ['upload.bin','original.png','input.png','mask.png','result.png','request.json','receipt.json','app_audit.json','supervisor.json','worker.log','stages.npz']
            save_json(path/'manifest.json', {'files': [pin(path/name) for name in names], 'note': 'One plausible completion; hidden identity is not verified. Original.png is decoded/oriented upload; input.png is prepared256. Black mask pixels preserve input.png exactly.'})
            with zipfile.ZipFile(path/'evidence.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                for name in names + ['manifest.json']:
                    archive.write(path/name, arcname=name)
                archive.writestr('README.txt', 'One plausible estimate, not verified hidden identity.\nWhite mask pixels are generated; black pixels exactly retain input.png.\noriginal.png is the decoded, EXIF-oriented upload; upload.bin preserves uploaded bytes.\nNo automatic detector, restoration model or training was used.\n')
            self.publish(session, names + ['manifest.json','evidence.zip'])
            message = 'Our trained R1 estimate ready. Inspect the generated anatomy and boundary; hidden features are plausible, not verified.' if mask.any() else 'The mask is empty: prepared input returned unchanged. No completion model ran.'
            return {'id': ident, 'source_id': data['id'], 'result': f'/asset/{ident}/result.png', 'mask': f'/asset/{ident}/mask.png', 'bundle': f'/bundle/{ident}', 'seconds': elapsed, 'neural_calls': receipt['neural_calls'], 'message': message}
        finally:
            try:
                if 'process' in locals() and process is not None:
                    try:
                        process.close()
                    finally:
                        with self.process_lock:
                            self.owned_processes.discard(process)
            finally:
                self.generation_lock.release()

    def close(self):
        # The request thread alone handles kill/wait/close; OS job closure also
        # terminates descendants if the application process exits unexpectedly.
        self.shutdown_event.set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'LocalCompletion/1'
    def setup(self):
        super().setup()
        self.connection.settimeout(30)
    def log_message(self, fmt, *args):
        pass
    def send(self, status, body, mime='application/json', attachment=None):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'")
        if attachment:
            self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
    def json(self, status, data):
        self.send(status, json.dumps(data, allow_nan=False).encode())
    def same_host(self):
        need(self.headers.get('Host') == self.server.authority, 'Use this application on its local address.')
        origin = self.headers.get('Origin')
        need(not origin or origin == 'http://' + self.server.authority, 'Open the local completion page directly.')
    def do_GET(self):
        try:
            self.same_host()
            route = urlsplit(self.path).path
            if route == '/':
                return self.send(200, (ROOT/'templates/completion_codeformer_finetune_r1.html').read_bytes(), 'text/html; charset=utf-8')
            if route == '/health':
                return self.json(200, {'model': MODEL_LABEL, 'delivery_size': 256, 'generation_enabled': True, 'manual_review_required': True, 'device': 'cpu', 'worker_cap_seconds': WORKER_CAP, 'process_tree_guard': 'windows-job-suspended-before-resume' if os.name=='nt' else 'posix-new-session', 'model_qualified': False, 'restoration_enabled': False})
            if route == '/favicon.ico':
                return self.send(204,b'','image/x-icon')
            match = re.fullmatch(r'/(asset|bundle)/([a-f0-9]{24})(?:/([a-z0-9_.-]+))?', route)
            need(match is not None, 'Not found.')
            kind, ident, name = match.groups()
            need((kind == 'asset' and name is not None) or (kind == 'bundle' and name is None), 'Not found.')
            if kind == 'bundle':
                name = 'evidence.zip'
            session = self.server.app.session(ident)
            need(name in session['assets'], 'Not found.')
            item = session['assets'][name]
            path = session['path']/name
            need(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Saved evidence changed; download refused.')
            mime = 'image/png' if name.endswith('.png') else 'application/zip' if name.endswith('.zip') else 'application/octet-stream'
            return self.send(200, path.read_bytes(), mime, 'codeformer-completion-r1-evidence.zip' if kind=='bundle' else None)
        except (ValueError, OSError, KeyError):
            self.json(404, {'error': 'Asset unavailable.'})
    def do_POST(self):
        try:
            self.same_host()
            need(self.headers.get('Content-Type','').split(';')[0] == 'application/json', 'Use the local upload form.')
            length = int(self.headers.get('Content-Length','0'))
            need(0 < length <= BODY_LIMIT, 'Upload is too large.')
            route = urlsplit(self.path).path
            need(route in ('/prepare','/generate'), 'Unknown request.')
            body = self.rfile.read(length)
            need(len(body) == length, 'Upload was interrupted.')
            data = json.loads(body)
            need(isinstance(data,dict), 'Invalid request.')
            result = self.server.app.prepare(data) if route=='/prepare' else self.server.app.generate(data)
            self.json(200,result)
        except BlockingIOError as exc:
            self.json(409,{'error':str(exc)})
        except TimeoutError as exc:
            self.json(503,{'error':str(exc)})
        except (ValueError, OSError, KeyError) as exc:
            self.json(422,{'error':str(exc) or 'Request could not be completed.'})
        except Exception:
            self.json(500,{'error':'Completion failed; retained evidence is available for inspection.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission-sha', required=True)
    parser.add_argument('--worker-sha', required=True)
    parser.add_argument('--guard-sha', required=True)
    parser.add_argument('--candidate-manifest-sha', required=True)
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    app = CompletionApp(admission_sha=args.admission_sha, worker_sha=args.worker_sha, guard_sha=args.guard_sha, candidate_manifest_sha=args.candidate_manifest_sha)
    server = ThreadingHTTPServer(('127.0.0.1',args.port), Handler)
    server.daemon_threads = True
    server.app = app
    server.authority = '127.0.0.1:' + str(server.server_port)
    runtime = {'url':'http://'+server.authority+'/', 'pid':os.getpid(), 'model':MODEL_LABEL, 'manual':True, 'model_qualified':False, 'worker_sha256':args.worker_sha, 'source_admission_sha256':args.admission_sha, 'process_guard_sha256':args.guard_sha, 'candidate_manifest_sha256':args.candidate_manifest_sha}
    (ROOT/'scratch').mkdir(exist_ok=True)
    save_json(ROOT/'scratch/completion-codeformer-finetune-r1-runtime.json', runtime)
    print(json.dumps(runtime), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()


if __name__=='__main__':
    main()
