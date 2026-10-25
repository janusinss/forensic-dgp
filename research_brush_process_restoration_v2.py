"""One owned disposable interpreter per sealed Protect/Apply request."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
import time
import uuid

import research_brush_restoration_v1 as B

ROOT = B.ROOT
SOURCES = ('research_brush_process_restoration_v2.py', 'research_brush_process_worker_v2.py',
           'serve_codeformer_brush_process_research_v2.py')
REQUEST_ROOT = ROOT / 'scratch' / 'protect-apply-process-v2'
JSON_LIMIT = 4 * 1024 ** 2
DIRECTORY_LIMIT = 32 * 1024 ** 2
DEADLINE_SECONDS = 360
CLEANUP_SECONDS = 10
MEMORY_MINIMUM = 5 * 1024 ** 3 // 2
TOKEN_RE = re.compile(r'run-[0-9a-f]{32}\Z')
_configured = None


def worker_python():
    # Windows venv python.exe is a redirector; supervising it can orphan its child.
    # CPython's launcher variable keeps the same venv while directly owning the interpreter PID.
    executable = sys._base_executable if os.name == 'nt' else sys.executable
    environment = dict(os.environ)
    if os.name == 'nt':
        environment['__PYVENV_LAUNCHER__'] = sys.executable
    return executable, environment


def ordinary(path, directory=False):
    path = Path(path)
    B.need('..' not in path.parts, 'Parent traversal in process evidence paths is refused.')
    path = path.absolute()
    B.need(path.is_relative_to(ROOT), 'Process evidence must stay in this workspace.')
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            B.need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
                   'Linked process evidence paths are refused.')
    resolved = path.resolve(strict=False)
    B.need(resolved.is_relative_to(ROOT.resolve()), 'Resolved process evidence must stay in this workspace.')
    path = resolved
    if path.exists():
        info = path.stat()
        B.need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
               'Ordinary owned process evidence required.')
    return path


def read_json(path, cap=JSON_LIMIT):
    path = ordinary(path)
    before = path.stat()
    B.need(0 < before.st_size <= cap, 'Bounded process JSON required.')
    with path.open('rb') as stream:
        raw = stream.read(cap + 1)
    after = path.stat()
    B.need(len(raw) <= cap and (before.st_size, before.st_mtime_ns, before.st_ino) ==
           (after.st_size, after.st_mtime_ns, after.st_ino), 'Process evidence changed while reading.')
    return B.P.strict_json(raw)


def publish(directory, name, value, cap=JSON_LIMIT):
    directory = ordinary(directory, True)
    target = ordinary(directory / name)
    temp = ordinary(directory / (name + '.new'))
    raw = (json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
    B.need(0 < len(raw) <= cap and not target.exists() and not temp.exists(), 'Fresh bounded process record required.')
    size = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    B.need(size + len(raw) <= DIRECTORY_LIMIT, 'Process request directory exceeds32MiB.')
    with temp.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)


def configure_sources(path, expected_sha, parent_path, parent_sha, brush_path, brush_sha):
    global _configured
    B.need(_configured is None, 'Fresh process launcher required for a changed lifecycle manifest.')
    row = B.P.pin(path, 1024 ** 2)
    B.need(row['sha256'] == expected_sha, 'Exact process source-manifest SHA required.')
    value = B.P.strict_json(B.P.bound(row).read_bytes())
    B.need(value.get('format') == 'brush-process-lifecycle-source-v2' and value.get('complete') is True
           and value.get('model_qualified') is False and value.get('main_app_default_changed') is False
           and value.get('training_optimizer_VM_calls') == 0, 'Reviewed process lifecycle manifest required.')
    sources = value.get('sources')
    B.need(type(sources) is list and len(sources) == 3 and {r['path'] for r in sources} == set(SOURCES),
           'Exact three-source process namespace required.')
    for source in sources:
        B.P.bound(source, 1024 ** 2)
    parent = B.P.pin(parent_path, 1024 ** 2)
    brush = B.P.pin(brush_path, 1024 ** 2)
    B.need(parent['sha256'] == parent_sha and brush['sha256'] == brush_sha
           and value.get('parent_manifest') == parent and value.get('brush_manifest') == brush,
           'Exact unchanged parent and brush manifests required.')
    review = B.P.read(value['source_review'])
    B.need(review.get('complete') is True and review.get('source_review_pass') is True
           and review.get('sources') == sources, 'Independent current lifecycle source review required.')
    B.P.configure_manifest(parent_path, parent_sha)
    B.configure_sources(brush_path, brush_sha)
    _configured = {'manifest': row, 'parent_manifest': parent, 'brush_manifest': brush, 'sources': sources,
                   'source_review': value['source_review']}
    return source_check()


def source_check():
    B.need(_configured is not None, 'Start the separate process launcher with reviewed source manifests.')
    for row in (_configured['manifest'], _configured['parent_manifest'], _configured['brush_manifest'],
                _configured['source_review'], *_configured['sources']):
        B.P.bound(row, 1024 ** 2 if row is not _configured['source_review'] else 8 * 1024 ** 2)
    B.source_check()
    return _configured


class ProcessSelectedRestorer:
    def __init__(self):
        self.counter_lock = threading.RLock()
        self.counters = {'neural_forward_attempts': 0, 'neural_forward_returns': 0,
                         'completed_comparisons': 0, 'operator_review_refusals': 0}
        self.active = None
        self.uncertain_requests = 0
        self.uncertain_forward_upper_bound = 0
        self.cleanup_failures = 0

    def count(self, name):
        with self.counter_lock:
            B.need(name in self.counters, 'Unknown process counter.')
            if name == 'completed_comparisons' and self.active is not None:
                B.need(not self.active['complete_pending'], 'Duplicate completed comparison.')
                self.active['complete_pending'] = True
            else:
                self.counters[name] += 1

    def _events(self):
        state = self.active
        if state is None:
            return
        while state['events'] < 4:
            sequence = state['events'] + 1
            path = state['directory'] / ('event-%d.json' % sequence)
            if not path.exists():
                break
            event = read_json(path, 4096)
            name = 'neural_forward_attempts' if sequence % 2 else 'neural_forward_returns'
            expected_attempts, expected_returns = (sequence + 1) // 2, sequence // 2
            B.need(type(event) is dict and type(event.get('sequence')) is int
                   and type(event.get('attempts')) is int and type(event.get('returns')) is int
                   and event == {'request_id': state['token'], 'sequence': sequence, 'name': name,
                             'attempts': expected_attempts, 'returns': expected_returns},
                   'Ordered exact worker counter events required.')
            self.counters[name] += 1
            state['events'] = sequence

    def measured_status(self):
        with self.counter_lock:
            try:
                self._events()
            except (ValueError, OSError):
                if self.active is not None:
                    self._uncertain()
            state = self.active
            return {**self.counters, 'counter_integrity': self.uncertain_requests == 0,
                    'uncertain_requests': self.uncertain_requests,
                    'uncertain_forward_attempts_upper_bound': self.uncertain_forward_upper_bound,
                    'active_worker': state is not None,
                    'active_forward_counts_known': state is None or state['call_known'],
                    'worker_cleanup_failures': self.cleanup_failures,
                    'process_lifecycle': 'one-disposable-worker-per-request-v2'}

    def _uncertain(self):
        state = self.active
        if state is not None and not state['uncertain']:
            state['uncertain'] = True
            self.uncertain_requests += 1
            self.uncertain_forward_upper_bound += 2

    def _start(self, selection):
        config = source_check()
        B.P.manifest()
        import psutil
        if psutil.virtual_memory().available < MEMORY_MINIMUM:
            raise RuntimeError('Local memory is busy. Retry after the current image task or audit finishes.')
        ordinary(REQUEST_ROOT, True)
        REQUEST_ROOT.mkdir(parents=True, exist_ok=True)
        ordinary(REQUEST_ROOT, True)
        token = 'run-' + uuid.uuid4().hex
        directory = REQUEST_ROOT / token
        directory.mkdir(exist_ok=False)
        ordinary(directory, True)
        startup = {'request_id': token, 'selection': selection, 'deadline_seconds': DEADLINE_SECONDS,
                   'manifest': config['manifest'], 'parent_manifest': config['parent_manifest'],
                   'brush_manifest': config['brush_manifest']}
        publish(directory, 'startup.json', startup, 16384)
        executable, environment = worker_python()
        process = subprocess.Popen([executable, '-B', str(ROOT / SOURCES[1]), '--worker', token], env=environment,
                                   cwd=str(ROOT), shell=False, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.active = {'token': token, 'directory': directory, 'process': process,
                       'manifest': config['manifest'],
                       'deadline': time.monotonic() + DEADLINE_SECONDS, 'selection': selection,
                       'slots': 0, 'events': 0, 'complete_pending': False, 'call_known': False,
                       'uncertain': False, 'identity_verified': False}

    def _identity(self):
        state = self.active
        if state['identity_verified']:
            return
        identity = read_json(state['directory'] / 'identity.json', 4096)
        B.need(type(identity) is dict and type(identity.get('worker_pid')) is int
               and identity == {'request_id': state['token'], 'worker_pid': state['process'].pid,
                                 'python_prefix': sys.prefix, 'python_executable': sys.executable},
               'Direct owned interpreter PID and unchanged Python environment required.')
        state['identity_verified'] = True

    def _wait_identity(self):
        state = self.active
        while not (state['directory'] / 'identity.json').exists():
            if state['process'].poll() is not None:
                raise RuntimeError('Restoration worker exited before its owned interpreter identity was verified.')
            if time.monotonic() >= state['deadline']:
                raise RuntimeError('Restoration worker identity exceeded the360second request deadline.')
            time.sleep(.1)
        self._identity()

    def run_one(self, contents, selection):
        B.need(type(contents) is bytes and 0 < len(contents) <= B.preparation.MAX_UPLOAD_BYTES,
               'Upload a PNG/JPEG face crop up to10MiB.')
        B.need(selection in B.P.ARMS, 'Choose retained U96 or experimental u64.')
        with self.counter_lock:
            if self.active is None:
                self._start(selection)
            state = self.active
            B.need(state['selection'] == selection and state['slots'] < 2 and not state['uncertain'],
                   'At most two ordered calls to the same selected worker required.')
            state['call_known'] = False
            slot = state['slots']
            try:
                self._wait_identity()  # No input command exists until exact PID and venv ownership pass.
                path = ordinary(state['directory'] / ('input-%d.bin' % slot))
                with path.open('xb') as stream:
                    stream.write(contents)
                publish(state['directory'], 'command-%d.json' % slot,
                        {'request_id': state['token'], 'slot': slot, 'selection': selection,
                         'input_bytes': len(contents), 'input_sha256': hashlib.sha256(contents).hexdigest()}, 4096)
            except BaseException:
                self._uncertain()
                raise
        try:
            result_path = state['directory'] / ('result-%d.json' % slot)
            while not result_path.exists():
                with self.counter_lock:
                    if (state['directory'] / 'identity.json').exists():
                        self._identity()
                    self._events()
                if state['process'].poll() is not None:
                    raise RuntimeError('Restoration worker exited before a verified result; do not automatically retry.')
                if time.monotonic() >= state['deadline']:
                    raise RuntimeError('Restoration worker exceeded the360second request deadline; do not automatically retry.')
                time.sleep(.1)
            result = read_json(result_path)
            with self.counter_lock:
                self._identity()
                self._events()
                B.need(not state['uncertain'] and type(result) is dict and set(result) == {'request_id', 'slot', 'selection', 'input_sha256',
                       'attempts', 'returns', 'ok', 'value'} and result['request_id'] == state['token']
                       and type(result['slot']) is int and result['slot'] == slot and result['selection'] == selection
                       and result['input_sha256'] == hashlib.sha256(contents).hexdigest(), 'Worker result binding differs.')
                attempts, returns = (state['events'] + 1) // 2, state['events'] // 2
                B.need(type(result['attempts']) is int and type(result['returns']) is int
                       and (result['attempts'], result['returns']) == (attempts, returns)
                       and type(result['ok']) is bool, 'Verified worker result counters required.')
                state['call_known'] = True
                state['slots'] += 1
                if not result['ok']:
                    error = result['value']
                    B.need(type(error) is dict and set(error) == {'kind', 'message'}
                           and error['kind'] in ('ValueError', 'OSError', 'RuntimeError')
                           and type(error['message']) is str and len(error['message']) <= 2048,
                           'Bounded worker failure required.')
                    raise {'ValueError': ValueError, 'OSError': OSError, 'RuntimeError': RuntimeError}[error['kind']](error['message'])
                B.need((attempts, returns) == (slot + 1, slot + 1), 'Exact selected worker call budget required.')
                value = result['value']
                B.need(type(value) is dict and set(value) == {'PNG', 'raw', 'prepared', 'support', 'metadata'}
                       and type(value['metadata']) is dict
                       and value['metadata'].get('selection') == selection
                       and value['metadata'].get('uploaded_bytes_sha256') == result['input_sha256']
                       and value['metadata'].get('worker_lifecycle') == {
                           'version': 'one-disposable-worker-per-request-v2', 'request_id': state['token'],
                           'worker_pid': state['process'].pid, 'source_manifest': state['manifest']},
                       'Exact selected worker output schema required.')
                output = {'metadata': value['metadata']}
                for key in ('PNG', 'raw', 'prepared', 'support'):
                    B.need(type(value[key]) is str and len(value[key]) <= JSON_LIMIT, 'Bounded encoded worker output required.')
                    output[key] = base64.b64decode(value[key], validate=True)
                    B.need(0 < len(output[key]) <= 1024 ** 2, 'Bounded binary worker output required.')
                return output
        except BaseException:
            with self.counter_lock:
                if not state['call_known']:
                    self._uncertain()
            raise

    def release_models(self):
        with self.counter_lock:
            state = self.active
            if state is None:
                return
        process = state['process']
        forced = False
        cleanup_deadline = time.monotonic() + CLEANUP_SECONDS
        try:
            stop = state['directory'] / 'stop.json'
            if process.poll() is None and not stop.exists():
                publish(state['directory'], 'stop.json', {'request_id': state['token']}, 4096)
            try:
                process.wait(timeout=max(0., min(CLEANUP_SECONDS / 2, cleanup_deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                forced = True
                process.terminate()
                try:
                    process.wait(timeout=max(0., min(CLEANUP_SECONDS / 4, cleanup_deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=max(0., cleanup_deadline - time.monotonic()))
            with self.counter_lock:
                self._events()
                self._identity()
                terminal = read_json(state['directory'] / 'terminal.json', 4096)
                B.need(not forced and process.returncode == 0 and type(terminal) is dict
                       and type(terminal.get('attempts')) is int and type(terminal.get('returns')) is int
                       and terminal.get('clean_exit') is True
                       and terminal == {'request_id': state['token'], 'clean_exit': True,
                                        'attempts': (state['events'] + 1) // 2, 'returns': state['events'] // 2},
                       'Worker clean exit and exact terminal counters required.')
                if state['complete_pending'] and not state['uncertain']:
                    self.counters['completed_comparisons'] += 1
        except BaseException as error:
            # A full disk or invalid control path must not skip reaping the owned process.
            if process.poll() is None:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=max(0., min(CLEANUP_SECONDS / 4, cleanup_deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=max(0., cleanup_deadline - time.monotonic()))
                except (OSError, subprocess.TimeoutExpired):
                    pass  # Keep the exact handle and fail closed if exit cannot be confirmed.
            with self.counter_lock:
                self._uncertain()
                self.cleanup_failures += 1
            raise RuntimeError('Worker cleanup or counter verification failed; request outcome is uncertain; do not automatically retry.') from error
        finally:
            with self.counter_lock:
                if process.poll() is not None:
                    self.active = None
                else:
                    self._uncertain()  # Retain exact owned handle; future run_one cannot spawn another child.
