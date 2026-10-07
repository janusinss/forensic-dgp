"""Resumable Windows backup of actual research cache files using one SSH stream."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1'
MANIFEST = ROOT / 'outputs/cctv_dgp_vm_migration_backup_20261007_v1/cache_source_inventory.json'
PIN = 'af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291'
READER = ROOT / 'scripts/stream_cctv_dgp_cache_backup_20261007_v1.py'
REMOTE_READER = '/home/janusdominic0/stream_cctv_dgp_cache_backup_20261007_v1_r2.py'
REMOTE_MANIFEST = '/home/janusdominic0/cctv_dgp_cache_backup_manifest_20261007_v1.json'
PARTIAL = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261006_v1/cache_backups/group0/anatomical_cache/features.bin'
GCLOUD = Path(os.environ['LOCALAPPDATA']) / 'Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
BASE = ['--project=forensic-dgp-thesis', '--zone=us-central1-a', '--quiet']
MAGIC = b'DGPCBK01'
TOTAL = 39448585279
CHUNK = 256 * 1024 ** 2


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment():
    result = dict(os.environ)
    result['CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE'] = str(ROOT / 'scratch/gcloud_windows_trust.pem')
    return result


def write_json(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


def run_gcloud(args, label):
    logs = OUT / 'transport'
    with (logs / (label + '.out')).open('xb') as stdout, (logs / (label + '.err')).open('xb') as stderr:
        result = subprocess.run([str(GCLOUD)] + args, env=environment(), stdout=stdout,
                                stderr=stderr, timeout=120)
    assert result.returncode == 0, 'Transport failed; retained logs: ' + label


def read_exact(stream, count):
    blocks = []
    remaining = count
    while remaining:
        block = stream.read(remaining)
        if not block:
            raise EOFError('SSH stream ended before a complete frame')
        blocks.append(block)
        remaining -= len(block)
    return b''.join(blocks)


def read_frame(stream):
    assert read_exact(stream, 8) == MAGIC, 'Binary transport header differs'
    length = struct.unpack('!I', read_exact(stream, 4))[0]
    assert 0 < length <= 65536
    return json.loads(read_exact(stream, length))


def progress(data):
    print(json.dumps(data), flush=True)
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data) + '\n')


class Session:
    def __init__(self, run_id):
        check = ('import hashlib;from pathlib import Path;assert hashlib.sha256(Path(' +
                 repr(REMOTE_READER) + ').read_bytes()).hexdigest()==' + repr(sha(READER)))
        command = 'python3 -B -c ' + shlex.quote(check) + ' && python3 -B ' + shlex.quote(REMOTE_READER)
        command += ' --manifest ' + shlex.quote(REMOTE_MANIFEST) + ' --manifest-sha ' + PIN + ' --seconds 14400'
        # The installed gcloud PuTTY wrapper injects y\n and replaces stdin.
        # A simple command makes gcloud select console Plink, not GUI PuTTY.
        # Replace this one token after parsing the authenticated connection;
        # the SDK does not quote complex remote commands for Windows parsing.
        label = run_id + '_authenticated_ssh_command'
        run_gcloud(['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + BASE +
                   ['--ssh-flag=-batch', '--ssh-flag=-T', '--command=true', '--dry-run'], label)
        command_line = (OUT / 'transport' / (label + '.out')).read_text(encoding='utf-8-sig').strip()
        count = ctypes.c_int()
        parser = ctypes.windll.shell32.CommandLineToArgvW
        parser.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
        parser.restype = ctypes.POINTER(ctypes.c_wchar_p)
        pointer = parser(command_line, ctypes.byref(count))
        assert pointer
        try:
            arguments = [pointer[i] for i in range(count.value)]
        finally:
            ctypes.windll.kernel32.LocalFree(ctypes.cast(pointer, ctypes.c_void_p))
        expected_exe = GCLOUD.parent / 'sdk/plink.exe'
        assert Path(arguments[0]).resolve() == expected_exe.resolve()
        assert '-batch' in arguments and '-T' in arguments
        assert '-t' not in arguments and '-legacy-stdio-prompts' not in arguments
        assert arguments.pop() == 'true'
        assert arguments[-1].startswith('janusdominic0@')
        assert len(arguments[-1].split('@')) == 2
        # Plink receives the POSIX remote command as one argument; stdin remains
        # our framed protocol, without gcloud's automatic host-key reply.
        arguments.append(command)
        self.stderr = (OUT / 'transport' / (run_id + '_stream.err')).open('xb')
        self.proc = subprocess.Popen(arguments,
                                     env=environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr)
        write_json(OUT / 'transport' / (run_id + '_process.json'),
                   {'pid': self.proc.pid, 'started_unix': time.time(), 'owned_download_process': True})
        self.arm(120)
        hello = read_frame(self.proc.stdout)
        assert hello == {'op': 'hello', 'manifest_sha256': PIN, 'bytes': TOTAL, 'files': 4431}
        self.disarm()

    def arm(self, seconds):
        self.timer = threading.Timer(seconds, self.abort)
        self.timer.daemon = True
        self.timer.start()

    def disarm(self):
        self.timer.cancel()

    def abort(self):
        if self.proc.poll() is None:
            # Only the subprocess tree created by this downloader, never VM training.
            subprocess.run(['taskkill', '/PID', str(self.proc.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)

    def request(self, data):
        self.proc.stdin.write(json.dumps(data).encode() + b'\n')
        self.proc.stdin.flush()
        return read_frame(self.proc.stdout)

    def close(self):
        self.disarm()
        if self.proc.poll() is None:
            self.arm(30)
            result = self.request({'op': 'close'})
            assert result['op'] == 'closed'
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
            self.disarm()
        self.stderr.close()


def download():
    assert sha(MANIFEST) == PIN
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['file_count'] == 4431 and manifest['bytes'] == TOTAL
    assert OUT.resolve().is_relative_to(ROOT / 'outputs')
    OUT.mkdir(exist_ok=True)
    (OUT / 'transport').mkdir(exist_ok=True)
    (OUT / 'chunks').mkdir(exist_ok=True)
    if (OUT / 'complete.json').exists():
        progress({'complete': True, 'status': 'Existing completed backup retained'})
        return
    lock = OUT / 'download.lock'
    assert not lock.exists(), 'Inspect the previous downloader before clearing its lock'
    run_id = uuid.uuid4().hex
    write_json(lock, {'pid': os.getpid(), 'run_id': run_id, 'started_unix': time.time()})
    session = None
    started = time.monotonic()
    try:
        staged = sum(p.stat().st_size for p in (OUT / 'chunks').glob('*.bin'))
        materialized = sum(p.stat().st_size for p in (OUT / 'cache_files').rglob('*')
                           if p.is_file() and not p.name.endswith('.partial'))
        required = max(0, TOTAL - staged) + max(0, TOTAL - materialized) + 5 * 1024 ** 3
        assert shutil.disk_usage(OUT).free >= required, 'Insufficient space for the remaining backup files'
        if not (OUT / 'cache_source_inventory.json').exists():
            with (OUT / 'cache_source_inventory.json').open('xb') as f:
                f.write(MANIFEST.read_bytes())
        assert sha(OUT / 'cache_source_inventory.json') == PIN
        # Never overwrite a different pre-existing helper or manifest on the VM.
        remote_check = 'import hashlib;from pathlib import Path;'
        for remote, local in [(REMOTE_READER, READER), (REMOTE_MANIFEST, MANIFEST)]:
            remote_check += ('p=Path(' + repr(remote) + ');assert not p.exists() or '
                             'hashlib.sha256(p.read_bytes()).hexdigest()==' + repr(sha(local)) + ';')
        run_gcloud(['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + BASE +
                   ['--ssh-flag=-batch', '--command=python3 -B -c ' + shlex.quote(remote_check)],
                   run_id + '_helper_collision_guard')
        # These are two small helper files in the source user's home, not research tensors.
        run_gcloud(['compute', 'scp'] + BASE + [str(READER),
                   'janusdominic0@forensic-dgp-thesis:' + REMOTE_READER], run_id + '_reader_upload')
        run_gcloud(['compute', 'scp'] + BASE + [str(MANIFEST),
                   'janusdominic0@forensic-dgp-thesis:' + REMOTE_MANIFEST], run_id + '_manifest_upload')
        session = Session(run_id)
        session.arm(120)
        header = session.request({'op': 'chunk', 'start': 0, 'bytes': 65536})
        assert header == {'op': 'chunk', 'start': 0, 'bytes': 65536}
        sample = read_exact(session.proc.stdout, 65536)
        probe = read_frame(session.proc.stdout)
        session.disarm()
        assert probe == {'op': 'receipt', 'start': 0, 'bytes': 65536,
                         'sha256': hashlib.sha256(sample).hexdigest()}
        if PARTIAL.is_file():
            with PARTIAL.open('rb') as existing:
                assert sample == existing.read(65536), 'Binary transport sample differs from the earlier prefix'
        write_json(OUT / 'transport' / (run_id + '_binary_transport_probe.json'),
                   {'complete': True, 'bytes': 65536, 'sha256': probe['sha256'],
                    'VM_and_Windows_bytes_match': True})
        progress({'stage': 'binary_transport_verified', 'bytes': 65536})
        if PARTIAL.is_file() and not (OUT / 'prefix_reuse.json').exists():
            session.arm(300)
            receipt = session.request({'op': 'prefix', 'bytes': PARTIAL.stat().st_size})
            session.disarm()
            assert receipt['sha256'] == sha(PARTIAL), 'Earlier partial is not a matching prefix; preserve it'
            write_json(OUT / 'prefix_reuse.json', {'complete': True, 'bytes': receipt['bytes'],
                       'sha256': receipt['sha256'], 'source_partial': str(PARTIAL), 'source_modified': False})
        prefix = json.loads((OUT / 'prefix_reuse.json').read_text()) if (OUT / 'prefix_reuse.json').exists() else None
        if prefix:
            assert PARTIAL.stat().st_size == prefix['bytes'] and sha(PARTIAL) == prefix['sha256']
        completed_bytes = 0
        for index in range(math.ceil(TOTAL / CHUNK)):
            assert time.monotonic() - started < 14400, 'Finite local download stop: four hours'
            start, count = index * CHUNK, min(CHUNK, TOTAL - index * CHUNK)
            path = OUT / 'chunks' / f'{index:04d}.bin'
            proof = path.with_suffix('.json')
            if path.exists() and proof.exists():
                previous = json.loads(proof.read_text())
                assert previous['manifest_sha256'] == PIN and previous['start'] == start and previous['bytes'] == count
                assert path.stat().st_size == count and sha(path) == previous['sha256']
                completed_bytes += count
                continue
            assert not path.exists() and not proof.exists(), 'Preserve interrupted evidence; inspect inconsistent chunk'
            temporary = OUT / 'chunks' / f'{index:04d}_{run_id}.partial'
            h = hashlib.sha256()
            if prefix and start + count <= prefix['bytes']:
                with PARTIAL.open('rb') as old, temporary.open('xb') as target:
                    old.seek(start)
                    remaining = count
                    while remaining:
                        block = old.read(min(1024 ** 2, remaining));assert block
                        target.write(block);h.update(block);remaining -= len(block)
                origin = 'verified_earlier_prefix'
            else:
                session.arm(600)
                header = session.request({'op': 'chunk', 'start': start, 'bytes': count})
                assert header == {'op': 'chunk', 'start': start, 'bytes': count}
                with temporary.open('xb') as target:
                    remaining = count
                    while remaining:
                        block = read_exact(session.proc.stdout, min(1024 ** 2, remaining))
                        target.write(block);h.update(block);remaining -= len(block)
                receipt = read_frame(session.proc.stdout)
                session.disarm()
                assert receipt == {'op': 'receipt', 'start': start, 'bytes': count, 'sha256': h.hexdigest()}
                origin = 'verified_VM_stream'
            assert temporary.stat().st_size == count and not path.exists()
            temporary.rename(path)
            write_json(proof, {'manifest_sha256': PIN, 'start': start, 'bytes': count,
                              'sha256': h.hexdigest(), 'origin': origin})
            completed_bytes += count
            progress({'downloaded_bytes': completed_bytes, 'total_bytes': TOTAL,
                      'percent': round(100 * completed_bytes / TOTAL, 2),
                      'chunk': index + 1, 'chunks': math.ceil(TOTAL / CHUNK),
                      'seconds': round(time.monotonic() - started, 1)})
        session.close();session = None
        progress({'stage': 'materialize_actual_cache_files', 'files': 4431})
        destination_root = OUT / 'cache_files'
        position = 0
        receipts = []
        for index, row in enumerate(manifest['files']):
            original = Path(row['path'])
            relative = row['path'].lstrip('/').split('/')
            assert '..' not in relative
            destination = destination_root.joinpath(*relative)
            assert destination.resolve().is_relative_to(destination_root.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert destination.stat().st_size == row['bytes'] and sha(destination) == row['sha256']
                position += row['bytes']
            else:
                h = hashlib.sha256();remaining = row['bytes']
                # Legacy Windows MAX_PATH permits the final names, but adding
                # a UUID to those names exceeds 260 characters for NPZ files.
                # Each directory has one sequential writer; rename frees this
                # short, run-owned temporary name before the next file.
                temporary = destination.parent / ('.restore_' + run_id + '.partial')
                with temporary.open('xb') as target:
                    while remaining:
                        chunk_index, offset = divmod(position, CHUNK)
                        amount = min(remaining, CHUNK - offset)
                        with (OUT / 'chunks' / f'{chunk_index:04d}.bin').open('rb') as stream:
                            stream.seek(offset)
                            rest = amount
                            while rest:
                                block = stream.read(min(1024 ** 2, rest));assert block
                                target.write(block);h.update(block)
                                remaining -= len(block);rest -= len(block);position += len(block)
                assert temporary.stat().st_size == row['bytes'] and h.hexdigest() == row['sha256']
                temporary.rename(destination)
            receipts.append({'source': row['path'], 'local': str(destination), 'bytes': row['bytes'], 'sha256': row['sha256']})
            if index == 0 or (index + 1) % 500 == 0:
                progress({'materialized_files': index + 1, 'total_files': len(manifest['files'])})
        assert position == TOTAL
        write_json(OUT / 'materialized_files.json', {'complete': True, 'files': receipts})
        # A separate read-only verifier re-reads every materialized file from Windows.
        from verify_cctv_dgp_restored_caches_20261007_v1 import verify
        verify(MANIFEST, PIN, destination_root, OUT / 'independent_local_cache_audit.json', 1800)
        result = {'complete': True, 'actual_cache_files': len(receipts), 'actual_cache_bytes': TOTAL,
                  'local_cache_root': str(destination_root), 'manifest_sha256': PIN,
                  'seconds': time.monotonic() - started, 'source_research_files_modified': 0,
                  'training_started': False,
                  'cloud_snapshot_removed': json.loads((OUT / 'snapshot_rollback/snapshot_rollback_receipt.json').read_text())['snapshot_no_longer_active']}
        write_json(OUT / 'complete.json', result)
        progress(result)
    except BaseException as exc:
        write_json(OUT / ('failure_' + run_id + '.json'), {'error': type(exc).__name__, 'message': str(exc),
                   'seconds': time.monotonic() - started, 'completed_chunks_preserved': True})
        if session:
            session.abort()
        raise
    finally:
        if lock.exists():
            assert json.loads(lock.read_text())['run_id'] == run_id
            lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', required=True)
    parser.parse_args()
    download()
