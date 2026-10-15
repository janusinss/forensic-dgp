"""Hash-bound local cold-backup removal after completed, audited Drive copies.

Only exact regular files selected by the frozen plan can be removed. No cloud
mutations, recursive deletions, VM connection, model or training operations.
"""
import argparse
from collections import Counter
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gzip
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/google_drive_offload_20261012_v1'
OUT = PARENT / 'closure_20261013_v1'
PLAN_PIN = '2390ae75ca39b6a60383931b2d5fc658459a6051b4a640ef7cf608a1f38f716f'
CURRENT_ACTIVE = PARENT / 'protected_active_bindings.json'
deadline = None


def finite():
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('Finite local offload closure limit; retain all receipts and ledger')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            finite()
            value.update(block)
    return value.hexdigest()


def bounded(relative):
    rel = Path(relative)
    assert relative and not rel.is_absolute() and '..' not in rel.parts and ':' not in relative
    path = ROOT / rel
    assert path.resolve(strict=True).is_relative_to(ROOT.resolve(strict=True))
    for ancestor in [path, *path.parents]:
        if ancestor == ROOT.parent:
            break
        assert not ancestor.is_symlink() and not getattr(ancestor.lstat(), 'st_file_attributes', 0) & 0x400, relative
    assert stat.S_ISREG(path.lstat().st_mode), relative
    return path


def metadata(path):
    value = path.lstat()
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, inode=value.st_ino,
                dev=value.st_dev, nlink=value.st_nlink, mode=value.st_mode,
                attributes=getattr(value, 'st_file_attributes', 0))


def guard_processes():
    command = ('Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, text=True, check=True, timeout=30)
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(rows, dict):
        rows = [rows]
    candidates = []
    for item in rows:
        if not re.match(r'^(python.*|rclone|gcloud|tar|7z|gzip|robocopy|DriveFS|gsutil|pscp)\.exe$', item['Name'], re.I):
            continue
        command_text = item.get('CommandLine')
        assert command_text, ('Cannot inspect a storage/research process', item['Name'], item['ProcessId'])
        normalized = command_text.replace('\\', '/').lower()
        owned = ('scripts/close_google_drive_offload_20261013_v1.py' in normalized and item['Name'].lower().startswith('python'))
        if owned:
            continue
        # A known local application uses only the retained active model paths.
        known_app = item['Name'].lower().startswith('python') and 'uvicorn face_workflow_web:app' in normalized
        if known_app:
            assert 'cache_files' not in normalized and 'dgp-image-output-recovery' not in normalized
            candidates.append(dict(name=item['Name'], pid=item['ProcessId'], command=command_text,
                                   allowed_disjoint_current_app=True))
            continue
        raise RuntimeError('Active storage/research process requires a dependency review: ' +
                           json.dumps(dict(name=item['Name'], pid=item['ProcessId'])))
    return candidates


def snapshot(name):
    rows, skipped, errors = {}, [], []
    started = time.monotonic()
    for folder, directories, files in os.walk(ROOT, followlinks=False, onerror=lambda error: errors.append(str(error))):
        finite()
        kept = []
        for directory in sorted(directories):
            path = Path(folder) / directory
            if path == OUT:
                continue
            try:
                value = path.lstat()
                if path.is_symlink() or getattr(value, 'st_file_attributes', 0) & 0x400:
                    skipped.append(path.relative_to(ROOT).as_posix())
                else:
                    kept.append(directory)
            except OSError as error:
                errors.append(str(error))
        directories[:] = kept
        for filename in sorted(files):
            path = Path(folder) / filename
            try:
                value = path.lstat()
                if path.is_symlink() or getattr(value, 'st_file_attributes', 0) & 0x400:
                    skipped.append(path.relative_to(ROOT).as_posix())
                    continue
                assert stat.S_ISREG(value.st_mode)
                rows[path.relative_to(ROOT).as_posix()] = metadata(path)
                if len(rows) % 100000 == 0:
                    print(json.dumps(dict(snapshot=name, files=len(rows), seconds=round(time.monotonic()-started, 1))), flush=True)
            except OSError as error:
                errors.append(str(error))
    with gzip.open(OUT / (name + '.json.gz'), 'xb', compresslevel=1) as stream:
        stream.write(json.dumps(rows, separators=(',', ':')).encode('utf-8'))
    receipt = dict(complete=not errors, file_count=len(rows), skipped_reparse_paths=skipped,
                   errors=errors, SHA256=sha(OUT / (name + '.json.gz')), free_bytes=shutil.disk_usage(ROOT).free,
                   seconds=time.monotonic()-started)
    write(OUT / (name + '_receipt.json'), receipt)
    assert not errors, 'Full namespace inventory is incomplete; do not remove anything'
    return rows, receipt


def active_bindings(plan):
    assert sha(CURRENT_ACTIVE) == plan['protected_manifest_sha256']
    frozen = read(CURRENT_ACTIVE)
    hashes = {}
    for relative, expected in frozen.items():
        path = bounded(relative)
        assert sha(path) == expected, 'Active file changed: ' + relative
        hashes[relative] = expected
    extra = set()
    for folder in ['checkpoints', 'weights', 'models', 'templates', 'static', 'docs']:
        for path in (ROOT / folder).rglob('*'):
            if path.is_file():
                extra.add(path.relative_to(ROOT).as_posix())
    extra.update(path.name for path in ROOT.iterdir() if path.is_file())
    extra.update(row['path'] for batch in plan['batches'].values() for row in batch['files'] if not row['role'].startswith('inactive'))
    for relative in sorted(extra - set(hashes)):
        hashes[relative] = sha(bounded(relative))
    return hashes


def audit_cloud_and_manual_receipts(plan):
    bindings, used_receipts = {}, {}
    listing = read(OUT / 'cloud_inventory_before.json')
    assert len(listing) == 4457
    cloud = {}
    ids = set()
    for item in listing:
        assert item['Path'] not in cloud and item['ID'] not in ids and item['ID']
        assert not item['IsDir'] and not item.get('IsBucket', False)
        hashes = {name.lower(): value for name, value in item['Hashes'].items()}
        assert hashes['sha256'] and len(hashes['sha256']) == 64
        cloud[item['Path']] = {**item, 'verified_sha256': hashes['sha256']}
        ids.add(item['ID'])
    expected_namespace = {batch_name + '/' + item['path'] for batch_name, batch in plan['batches'].items() for item in batch['files']}
    assert set(cloud) == expected_namespace
    for batch_name, batch in plan['batches'].items():
        for mode in ['Upload', 'Verify']:
            paths = sorted(PARENT.glob(mode.lower() + '_' + batch_name + '_*/receipt.json'))
            good = []
            for path in paths:
                receipt = read(path)
                if not receipt['Complete']:
                    continue
                assert receipt['Mode'] == mode and receipt['Batch'] == batch_name
                assert receipt['PlanSHA256'] == PLAN_PIN and receipt['RcloneExitCode'] == 0
                assert receipt['ExpectedFiles'] == batch['file_count'] and receipt['ExpectedBytes'] == batch['upload_bytes']
                assert receipt['Destination'] == plan['destination'] + '/' + batch_name
                assert receipt['FilesRemoved'] == receipt['TrainingCalls'] == 0
                assert Path(receipt['ReceiptDirectory']) == path.parent
                if mode == 'Verify':
                    assert receipt['DownloadedSHA256Verification']
                    lines = (path.parent / 'combined.txt').read_text(encoding='utf-8').splitlines()
                    assert Counter(lines) == Counter('= ' + item['path'] for item in batch['files'])
                    log = (path.parent / 'rclone.log').read_text(encoding='utf-8')
                    assert '0 differences found' in log and str(batch['file_count']) + ' matching files' in log
                good.append(path)
                for evidence in ['receipt.json', 'rclone.log'] + (['combined.txt'] if mode == 'Verify' else []):
                    full_path = path.parent / evidence
                    bindings[full_path.relative_to(ROOT).as_posix()] = sha(full_path)
            assert good, 'Completed manual operation evidence is absent'
            used_receipts[batch_name + '_' + mode.lower()] = [path.relative_to(ROOT).as_posix() for path in good]
        for item in batch['files']:
            actual = cloud[batch_name + '/' + item['path']]
            assert actual['Size'] == item['bytes'] and actual['verified_sha256'] == item['sha256'], item['path']
    return cloud, bindings, used_receipts


def allocated(path):
    function = ctypes.WinDLL('kernel32', use_last_error=True).GetCompressedFileSizeW
    function.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    function.restype = wintypes.DWORD
    high = wintypes.DWORD()
    ctypes.set_last_error(0)
    low = function(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    return (high.value << 32) | low


def prepare():
    assert sha(PARENT / 'plan.json') == PLAN_PIN
    plan = read(PARENT / 'plan.json')
    assert sha(ROOT / 'scripts/run_google_drive_offload_20261012_v1.ps1') == plan['driver_sha256']
    cloud, receipts, operations = audit_cloud_and_manual_receipts(plan)
    guard = guard_processes()
    protected = active_bindings(plan)
    rows, snapshot_receipt = snapshot('namespace_before')
    candidates = []
    for batch_name, batch in plan['batches'].items():
        for item in batch['files']:
            if not item['role'].startswith('inactive'):
                continue
            relative = item['path']
            assert relative not in protected
            assert relative == 'outputs/cctv_dgp_vm_output_offload_20261010_v1_r2/dgp-image-output-recovery-20261010-v1.tar' or relative.startswith('outputs/cctv_dgp_local_research_cache_backup_20261007_v1/cache_files/home/janusdominic0/forensic-dgp/')
            path = bounded(relative)
            current = metadata(path)
            assert current == rows[relative] and current['bytes'] == item['bytes'] and current['nlink'] == 1
            assert current['inode'] == item['inode'] and current['mtime_ns'] == item['mtime_ns']
            assert sha(path) == item['sha256'] and metadata(path) == current
            remote = cloud[batch_name + '/' + relative]
            candidates.append(dict(path=relative, metadata=current, sha256=item['sha256'],
                                  allocated_bytes=allocated(path), cloud_path=batch_name + '/' + relative,
                                  cloud_id=remote['ID'], cloud_sha256=remote['verified_sha256']))
    assert len(candidates) == 4432 and sum(row['metadata']['bytes'] for row in candidates) == 52028568639
    result = dict(complete=True, prepared_utc=datetime.now(timezone.utc).isoformat(), plan_sha256=PLAN_PIN,
                  closure_driver_sha256=sha(Path(__file__)), cloud_namespace_sha256=sha(OUT / 'cloud_inventory_before.json'),
                  uploaded_cloud_files=len(cloud), manual_operations=operations, receipt_hashes=receipts,
                  manual_downloaded_SHA256_pass=True, current_cloud_SHA256_all_match=True,
                  protected_hashes=protected, namespace_before_sha256=snapshot_receipt['SHA256'],
                  candidates=candidates, candidate_count=len(candidates),
                  logical_bytes=sum(row['metadata']['bytes'] for row in candidates),
                  singly_linked_allocated_bytes=sum(row['allocated_bytes'] for row in candidates),
                  process_guard=guard, files_removed=0, training_calls=0)
    write(OUT / 'removal_plan.json', result)
    (OUT / 'removal_plan.sha256').write_text(sha(OUT / 'removal_plan.json') + '  removal_plan.json\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(prepared=True, removal_plan_sha256=sha(OUT / 'removal_plan.json'),
        cloud_files=len(cloud), local_candidates=len(candidates), protected_hashes=len(protected),
        logical_bytes=result['logical_bytes'], allocated_bytes=result['singly_linked_allocated_bytes'],
        namespace_files=snapshot_receipt['file_count'], files_removed=0)), flush=True)


def exclusive_delete(path, expected_metadata, expected_sha, commit=None):
    assert metadata(path) == expected_metadata and expected_metadata['nlink'] == 1
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000 | 0x00010000, 0, None, 3, 0x80 | 0x08000000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(fd, 'rb') as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            finite()
            value.update(block)
        assert value.hexdigest() == expected_sha, 'Changed bytes; refuse removal'
        assert metadata(path) == expected_metadata
        if commit:
            commit()
        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [('DeleteFile', ctypes.c_ubyte)]
        info = FileDispositionInfo(1)
        disposition = kernel.SetFileInformationByHandle
        disposition.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        disposition.restype = wintypes.BOOL
        if not disposition(handle, 4, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
    assert not path.exists()


def fixture():
    directory = OUT / 'exclusive_handle_fixture'
    directory.mkdir()
    good = directory / 'exact.txt'
    changed = directory / 'changed.txt'
    busy = directory / 'busy.txt'
    for path in [good, changed, busy]:
        path.write_bytes(b'only owned fixture bytes\n')
    expected = sha(good)
    refused_changed = False
    try:
        exclusive_delete(changed, metadata(changed), '0'*64)
    except AssertionError:
        refused_changed = changed.exists()
    refused_busy = False
    with busy.open('rb'):
        try:
            exclusive_delete(busy, metadata(busy), expected)
        except OSError:
            refused_busy = busy.exists()
    exclusive_delete(good, metadata(good), expected)
    assert refused_changed and refused_busy and not good.exists()
    result = dict(complete=True, changed_file_refused=True, active_reader_refused=True,
                  only_owned_exact_fixture_removed=True, research_files_removed=0)
    write(OUT / 'exclusive_handle_fixture.json', result)
    print(json.dumps(result), flush=True)


def append_ledger(stream, row):
    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def apply(pin):
    assert sha(OUT / 'removal_plan.json') == pin and read(OUT / 'exclusive_handle_fixture.json')['complete']
    plan = read(OUT / 'removal_plan.json')
    assert plan['complete'] and plan['plan_sha256'] == PLAN_PIN
    assert plan['closure_driver_sha256'] == sha(Path(__file__))
    assert sha(PARENT / 'plan.json') == PLAN_PIN
    assert sha(OUT / 'cloud_inventory_before.json') == plan['cloud_namespace_sha256']
    assert time.time() - (OUT / 'cloud_inventory_before.json').stat().st_mtime < 3600, 'Refresh cloud inventory before a later removal'
    for relative, expected in {**plan['protected_hashes'], **plan['receipt_hashes']}.items():
        assert sha(bounded(relative)) == expected, relative
    guard = guard_processes()
    ledger_path = OUT / 'deletion_ledger.jsonl'
    removed = []
    if ledger_path.exists():
        events = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines()]
        completed = {row['path'] for row in events if row['event'] == 'removed'}
        pending = {row['path']: row for row in events if row['event'] == 'removal_intent'}
        for relative in completed:
            assert not (ROOT / relative).exists()
        for relative in set(pending) - completed:
            if not (ROOT / relative).exists():
                raise RuntimeError('A removal intent lacks final completion; independently reconcile before resume')
        removed = [row for row in events if row['event'] == 'removed']
    completed = {row['path'] for row in removed}
    free_before = shutil.disk_usage(ROOT).free
    with ledger_path.open('a', encoding='utf-8', newline='\n') as ledger:
        for index, item in enumerate(plan['candidates'], 1):
            finite()
            if item['path'] in completed:
                continue
            if index % 100 == 1:
                guard_processes()
            path = bounded(item['path'])
            assert item['cloud_sha256'] == item['sha256'] and item['cloud_id']
            assert metadata(path) == item['metadata']
            intent = dict(event='removal_intent', path=item['path'], bytes=item['metadata']['bytes'],
                          allocated_bytes=item['allocated_bytes'], sha256=item['sha256'],
                          cloud_id=item['cloud_id'], cloud_path=item['cloud_path'], removal_plan_sha256=pin)
            exclusive_delete(path, item['metadata'], item['sha256'], lambda: append_ledger(ledger, intent))
            record = dict(intent, event='removed', removed_utc=datetime.now(timezone.utc).isoformat(),
                          exclusive_handle_hash_verified=True, native_exact_file_removal=True)
            append_ledger(ledger, record)
            removed.append(record)
            if len(removed) % 500 == 0 or index == len(plan['candidates']):
                print(json.dumps(dict(removed=len(removed), of=plan['candidate_count'],
                                     logical_bytes_removed=sum(row['bytes'] for row in removed))), flush=True)
    assert len(removed) == plan['candidate_count']
    guard_processes()
    for relative, expected in plan['protected_hashes'].items():
        assert sha(bounded(relative)) == expected, relative
    free_after = shutil.disk_usage(ROOT).free
    write(OUT / 'apply_receipt.json', dict(complete=True, removal_plan_sha256=pin,
        files_removed=len(removed), logical_bytes_removed=sum(row['bytes'] for row in removed),
        allocated_bytes_removed=sum(row['allocated_bytes'] for row in removed), free_before_bytes=free_before,
        free_after_bytes=free_after, observed_free_increase_bytes=free_after-free_before,
        protected_hashes_checked_after=len(plan['protected_hashes']), deletion_ledger_sha256=sha(ledger_path),
        process_guard_before=guard, cloud_files_removed=0, directories_removed=0, VM_calls=0,
        model_gradient_or_training_calls=0, goal_complete=False))
    print(json.dumps(read(OUT / 'apply_receipt.json')), flush=True)


def main():
    global deadline
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'fixture', 'apply', 'snapshot_after'])
    parser.add_argument('--removal-plan-sha')
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    assert 30 <= args.max_seconds <= 3600
    deadline = time.monotonic() + args.max_seconds
    assert ROOT.resolve() == Path(r'C:\xampp\htdocs\YEAR 4\Testing').resolve()
    assert OUT.is_dir() and OUT.resolve().is_relative_to(ROOT.resolve())
    try:
        if args.phase == 'prepare':
            prepare()
        elif args.phase == 'fixture':
            fixture()
        elif args.phase == 'apply':
            assert args.removal_plan_sha and re.fullmatch('[0-9a-f]{64}', args.removal_plan_sha)
            apply(args.removal_plan_sha)
        else:
            snapshot('namespace_after')
    except Exception as error:
        write(OUT / ('stop_' + args.phase + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json'),
              dict(complete=False, phase=args.phase, error_type=type(error).__name__, error=str(error),
                   ledger_preserved=True, no_recursive_deletion=True))
        raise


if __name__ == '__main__':
    main()
