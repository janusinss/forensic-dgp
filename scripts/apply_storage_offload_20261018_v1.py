"""Separate exact-three-file removal; complete backup proof and peer closure required.

No network, VM, training, recursive deletion, checkpoint or directory removal.
Validation writes evidence only. Apply requires the independently reviewed validation.
"""
import argparse
import ctypes
from ctypes import wintypes
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
import sys
import time
import storage_offload_20261018_v1 as transfer
import protect_storage_offload_20261018_v1 as protected

ROOT, OUT = transfer.ROOT, transfer.OUT
KEYS = ('bytes', 'mtime_ns', 'inode', 'dev', 'nlink', 'mode', 'attributes')


def require(value, message):
    if not value:
        raise ValueError(message)


def process_guard():
    command = ('Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ParentProcessId,ExecutablePath,CommandLine '
               '| ConvertTo-Json -Depth 3 -Compress')
    run = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                         capture_output=True, text=True, timeout=30,
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    require(run.returncode == 0 and run.stdout.strip(), 'Fresh full process census required')
    rows = json.loads(run.stdout)
    rows = rows if isinstance(rows, list) else [rows]
    normalize = lambda value: str(value).replace('\\', '/').casefold().strip()
    marker = 'scripts/apply_storage_offload_20261018_v1.py'
    own_rows = [r for r in rows if r['ProcessId'] == os.getpid()]
    require(len(rows) >= 3 and len(own_rows) == 1, 'Exact current apply process required')
    own = own_rows[0]
    own_line, own_exe = own.get('CommandLine'), own.get('ExecutablePath')
    launcher = normalize((ROOT / 'venv/Scripts/python.exe').absolute())
    base = normalize(Path(getattr(sys, '_base_executable', sys.executable)).absolute())
    require(own_line and marker in normalize(own_line) and own_exe
            and normalize(own_exe) in (launcher, base) and normalize(sys.executable) == launcher
            and own.get('ParentProcessId') == os.getppid(), 'Exact apply CLI/interpreter/parent required')
    names = {Path(n).name.lower() for n in transfer.ARCHIVES}
    names.update(Path(r['retained_root']).name.lower() for r in transfer.ARCHIVES.values())
    transfer_exe = re.compile(r'^(rclone|gcloud|tar|7z|gzip|robocopy|DriveFS|gsutil|pscp)\.exe$', re.I)
    relevant, blocked = [], []
    for row in rows:
        name = row['Name']
        if not (transfer_exe.match(name) or re.match(r'^(python.*|powershell|pwsh)\.exe$', name, re.I)):
            continue
        line = row.get('CommandLine')
        require(line, 'Uninspectable relevant process')
        normalized = normalize(line)
        is_self = row['ProcessId'] == os.getpid()
        parent_launcher = (row['ProcessId'] == os.getppid() and normalize(line) == normalize(own_line)
                           and normalize(row.get('ExecutablePath')) == launcher and normalize(own_exe) == base)
        candidate_hits = sorted(n for n in names if n in normalized)
        other_helper = (name.lower().startswith('python') and
                        any(m in normalized for m in (marker, 'scripts/storage_offload_20261018_v1.py'))
                        and not is_self and not parent_launcher)
        if not is_self and not parent_launcher and (candidate_hits or transfer_exe.match(name) or other_helper):
            blocked.append({'name': name, 'pid': row['ProcessId'], 'candidate_matches': candidate_hits})
        relevant.append({'name': name, 'pid': row['ProcessId'], 'parent_pid': row.get('ParentProcessId'),
                         'command_sha256': hashlib.sha256(line.encode()).hexdigest(),
                         'candidate_matches': candidate_hits, 'own_process': is_self,
                         'verified_own_venv_parent': parent_launcher})
    require(not blocked, 'Competing candidate/transfer/apply process: ' + json.dumps(blocked))
    return {'complete': True, 'CIM_process_count': len(rows), 'relevant': relevant,
            'unrelated_disjoint_app_or_research_not_stopped': True}


def bound(name):
    path = ROOT / transfer.relative(name)
    require(path.absolute().is_relative_to(ROOT), 'Exact Testing scope required')
    return path


def document_pin(name, digest):
    path = bound(name)
    require(transfer.HEX.fullmatch(digest or '') and transfer.sha(path) == digest,
            'Exact evidence document hash required')
    return transfer.read(path), transfer.pin(path)


def remote_rows_check(plan, rows):
    require(type(rows) is list and len(rows) == 3, 'Exactly three remote objects required')
    expected = {r['path']: r for r in plan['files']}
    require({r.get('Path') for r in rows} == set(expected), 'Remote namespace differs')
    require(len({r.get('ID') for r in rows}) == 3 and all(r.get('ID') for r in rows),
            'Three distinct real Drive object IDs required')
    for row in rows:
        want = expected[row['Path']]
        hashes = {key.lower().replace('-', ''): value for key, value in row.get('Hashes', {}).items()}
        require(row.get('IsDir') is False and row.get('Size') == want['bytes']
                and hashes.get('sha256') == want['sha256'], 'Remote size/SHA256 differs')


def evidence(args):
    plan = transfer.checked_plan(args.plan_sha256)
    protected_source = transfer.pin(Path(protected.__file__))
    require(protected_source['sha256'] == '408470172152ff0852811ef3b1332c068f554d49ed890eb8bda6cd3ba64412b8',
            'Reviewed protected-hash implementation changed')
    require(plan['file_count'] == 3 and plan['upload_bytes'] == 15463424100, 'Exact-three fixed scope')
    proof = transfer.retention_proof(bound(args.proof), args.proof_sha256)
    verify, verify_pin = document_pin(args.verify_receipt, args.verify_sha256)
    require(verify.get('complete') is True and verify.get('mode') == 'verify'
            and verify.get('plan_sha256') == args.plan_sha256
            and verify.get('streamed_downloaded_SHA256_verified') is True
            and verify.get('child_terminal') is True and verify.get('rclone_exit_code') == 0
            and verify.get('destination') == plan['destination']
            and verify.get('expected_archives') == 3
            and verify.get('expected_bytes') == plan['upload_bytes'], 'Complete streamed backup proof required')
    remote, remote_pin = document_pin(args.remote_inventory, args.remote_sha256)
    remote_rows_check(plan, remote)
    before, before_pin = document_pin('outputs/storage_offload_20261018_v1/protected_before_v1.json', args.protected_sha256)
    require(before.get('complete') is True and before.get('files') == 3630
            and before.get('latest_return_full_hash_verified') is True
            and not set(before['protected']) & set(transfer.ARCHIVES), 'Protected scope differs')
    return plan, before, {'plan_sha256': args.plan_sha256, 'retention_proof': proof,
        'streamed_verify_receipt': verify_pin, 'remote_inventory': remote_pin,
        'protected_before': before_pin, 'apply_source': transfer.pin(Path(__file__)),
        'protected_hash_source': protected_source,
        'copy_source': plan['source_helper'], 'files': plan['files'],
        'candidate_bytes': plan['upload_bytes'], 'destination': plan['destination']}


def protected_check(before, phase):
    started = time.monotonic()
    def budget():
        require(time.monotonic() - started <= 900, 'Finite15minute protected verification cap')
    for index, (name, expected) in enumerate(before['protected'].items(), 1):
        require(protected.pin(bound(name), budget) == expected, 'Protected content/identity changed: ' + name)
        if index % 500 == 0:
            print(json.dumps({'protected_phase': phase, 'files': index, 'of': before['files']}), flush=True)
    return {'complete': True, 'files': before['files'], 'logical_bytes_hashed': before['logical_bytes_hashed'],
            'seconds': time.monotonic() - started, 'exact_before_pins_matched': True}


def remove_exact(row):
    require(row['path'] in transfer.ARCHIVES, 'Not an exact authorized archive')
    path = bound(row['path'])
    expected = {key: row[key] for key in KEYS}
    require(transfer.metadata(path) == expected and row['nlink'] == 1, 'Candidate identity changed')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000 | 0x00010000, 0, None, 3, 0x08000000, None)
    require(handle != ctypes.c_void_p(-1).value, 'Exclusive native file handle unavailable')
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(fd, 'rb') as stream:
        value, start = hashlib.sha256(), time.monotonic()
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            require(time.monotonic() - start <= 300, 'Finite exclusive candidate hash cap')
            value.update(chunk)
        require(value.hexdigest() == row['sha256'] and transfer.metadata(path) == expected,
                'Exclusive final content/identity check failed')
        class Disposition(ctypes.Structure):
            _fields_ = [('DeleteFile', ctypes.c_ubyte)]
        operation = kernel.SetFileInformationByHandle
        operation.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        operation.restype = wintypes.BOOL
        flag = Disposition(1)
        require(operation(handle, 4, ctypes.byref(flag), ctypes.sizeof(flag)), 'Native exact removal failed')
    require(not path.exists(), 'Candidate remained after exclusive removal')


def namespace_after():
    started, current, pending = time.monotonic(), {}, [ROOT / 'outputs']
    while pending:
        require(time.monotonic() - started <= 600, 'Finite10minute namespace cap')
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                p = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                        'Unexpected linked/reparse namespace entry')
                if stat.S_ISDIR(info.st_mode):
                    pending.append(p)
                else:
                    require(stat.S_ISREG(info.st_mode), 'Nonregular namespace entry')
                    current[p.relative_to(ROOT).as_posix()] = {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}
    with gzip.open(OUT / 'files_metadata_v1.json.gz', 'rt', encoding='utf8') as stream:
        before = {r['path']: {k: r[k] for k in ('bytes', 'mtime_ns')} for r in json.load(stream)}
    scope = 'outputs/storage_offload_20261018_v1/'
    missing = sorted(set(before) - set(current))
    changed = sorted(name for name in before.keys() & current.keys() if before[name] != current[name] and not name.startswith(scope))
    added = sorted(name for name in current.keys() - before.keys() if not name.startswith(scope))
    data = {'complete': missing == sorted(transfer.ARCHIVES) and not changed,
            'exact_missing': missing, 'retained_metadata_changes': changed, 'new_external_files': added,
            'all_retained_bytes_hash_claim': False, 'current_file_count': len(current),
            'seconds': time.monotonic() - started}
    path = OUT / 'namespace_after_v1.json.gz'
    with gzip.open(path, 'xb', compresslevel=1) as stream:
        stream.write(json.dumps(current, separators=(',', ':')).encode())
    transfer.save(OUT / 'namespace_after_receipt_v1.json', data)
    require(data['complete'], 'Postapply namespace mismatch; retain ledger and investigate')
    return {**data, 'snapshot': transfer.pin(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--proof', required=True)
    parser.add_argument('--proof-sha256', required=True)
    parser.add_argument('--verify-receipt', required=True)
    parser.add_argument('--verify-sha256', required=True)
    parser.add_argument('--remote-inventory', required=True)
    parser.add_argument('--remote-sha256', required=True)
    parser.add_argument('--protected-sha256', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--validation-sha256')
    parser.add_argument('--independent-review')
    parser.add_argument('--review-sha256')
    args = parser.parse_args()
    started = time.monotonic()
    plan, before, pins = evidence(args)
    if not args.apply:
        require(not (OUT / 'apply_validation_v1.json').exists(), 'Preserve previous validation')
        guard = process_guard()
        hashes = protected_check(before, 'prevalidate')
        receipt = {'complete': True, 'evidence': pins, 'process_guard': guard,
                   'protected_verified': hashes, 'files_removed': 0,
                   'independent_preapply_review_required': True}
        pin = transfer.save(OUT / 'apply_validation_v1.json', receipt)
        print(json.dumps({'validation': pin, 'files_removed': 0}), flush=True)
        return
    validation, validation_pin = document_pin('outputs/storage_offload_20261018_v1/apply_validation_v1.json', args.validation_sha256)
    require(validation.get('complete') is True and validation.get('evidence') == pins, 'Validated evidence changed')
    review, review_pin = document_pin(args.independent_review, args.review_sha256)
    require(review.get('complete') is True and review.get('approved_for_exact_three_file_apply') is True
            and review.get('validation_sha256') == args.validation_sha256
            and review.get('plan_sha256') == args.plan_sha256
            and review.get('apply_source_sha256') == pins['apply_source']['sha256'], 'Exact independent preapply closure required')
    require(not (OUT / 'deletion_ledger_v1.jsonl').exists() and not (OUT / 'apply_receipt_v1.json').exists(),
            'Existing apply evidence retained; never replay automatically')
    pre = protected_check(before, 'immediate_preapply')
    guard = process_guard()
    free_before, count, removed = shutil.disk_usage(ROOT).free, 0, 0
    transfer.save(OUT / 'apply_intent_v1.json', {'evidence': pins, 'validation': validation_pin,
        'review': review_pin, 'process_guard': guard, 'free_bytes_before': free_before})
    with (OUT / 'deletion_ledger_v1.jsonl').open('x', encoding='utf8', newline='\n') as ledger:
        def event(data):
            ledger.write(json.dumps(data, separators=(',', ':')) + '\n')
            ledger.flush(); os.fsync(ledger.fileno())
        for row in plan['files']:
            require(time.monotonic() - started <= 1800, 'Finite30minute exact apply cap')
            process_guard()
            base = {k: row[k] for k in ('path', 'bytes', 'sha256', 'inode', 'dev', 'nlink')}
            event({'event': 'removal_intent', **base})
            remove_exact(row)
            event({'event': 'removed', 'exclusive_native_handle_hash_verified': True, **base})
            count += 1; removed += row['bytes']
    require(count == 3 and removed == plan['upload_bytes'], 'Exact three removal totals differ')
    after = protected_check(before, 'postapply')
    namespace = namespace_after()
    free_after = shutil.disk_usage(ROOT).free
    receipt = {'complete': True, 'evidence': pins, 'files_removed': count,
        'logical_bytes_removed': removed, 'free_bytes_before': free_before,
        'free_bytes_after': free_after, 'observed_free_increase_bytes': free_after - free_before,
        'protected_preapply': pre, 'protected_postapply': after, 'namespace_after': namespace,
        'deletion_ledger': transfer.pin(OUT / 'deletion_ledger_v1.jsonl'),
        'VM_calls': 0, 'training_calls': 0, 'seconds': time.monotonic() - started,
        'storage_only_not_model_or_goal_completion': True}
    transfer.save(OUT / 'apply_receipt_v1.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('evidence', 'namespace_after')}), flush=True)


if __name__ == '__main__':
    main()
