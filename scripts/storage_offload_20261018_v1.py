"""Plan/copy/stream-verify exactly three cold archives; no local removal mode.

Default --plan-only writes an inert exact-file plan after source proof checks.
Upload/verify require root-reviewed plan SHA and repeat process/content guards.
There is no VM, training, move, sync, purge or deletion capability in this file.
"""
from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/storage_offload_20261018_v1'
OLD_PLAN = ROOT / 'outputs/google_drive_offload_20261012_v1/plan.json'
DESTINATION = 'thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261018_v1/pass1'
EXE = Path('C:/Users/janus/AppData/Local/Programs/rclone/rclone.exe')
CONFIG = Path('C:/Users/janus/AppData/Roaming/rclone/forensic-dgp-drive-manual.conf')
EXE_SHA = '41eab3e118d264920b62c3327966c5ece0dc4dd715ee617d9e30450d8ead95a4'
CLIENT_SHA = 'd5fd58eeb3f8d8d7be7db633e06b74fe9d917339c296db09913089ba664df147'
ARCHIVES = {
    'outputs/cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz': {
        'bytes': 5229234795, 'sha256': 'd49f30c01284fd4c27f45a161b1de4075ddf0a94b26eda4bd592a8a711bf4c6e',
        'retained_root': 'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'},
    'outputs/cctv-dgp-bank-photometric-guard-v1-results.tar.gz': {
        'bytes': 5217289325, 'sha256': 'f2ff55481a68d1690a3f7463384cd337c4dad238fcf8b8a600362bed54e2d356',
        'retained_root': 'outputs/cctv_dgp_bank_photometric_guard_v1_return'},
    'outputs/cctv-dgp-bank-comparison-v1-r3-A-results.tar.gz': {
        'bytes': 5016899980, 'sha256': '2720c5838e6a5da748783ae0e476c2bf2210693434f8b7d25f7e7404dbb26222',
        'retained_root': 'outputs/cctv_dgp_bank_comparison_v1_r3_A_return'}}
MAX_SECONDS = 8 * 3600
MAX_LOG_BYTES = 100 * 1024**2
HEX = re.compile(r'^[0-9a-f]{64}$')


def require(value, message):
    if not value:
        raise ValueError(message)


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Portable relative workspace path required')
    p = Path(name)
    require(not p.is_absolute() and all(part not in ('', '.', '..') for part in name.split('/')),
            'Unsafe workspace relative path')
    return p


def regular(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'Symlink/reparse path rejected')
    s = path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'Unique regular file required')
    return path


def metadata(path):
    s = regular(path).lstat()
    require(s.st_ino != 0, 'Real Path.lstat file ID required, not scandir placeholder')
    return {'bytes': s.st_size, 'inode': s.st_ino, 'dev': s.st_dev, 'nlink': s.st_nlink,
            'mtime_ns': s.st_mtime_ns, 'mode': s.st_mode,
            'attributes': getattr(s, 'st_file_attributes', 0)}


def sha(path, cap_seconds=300):
    path = regular(path)
    before, start, value = metadata(path), time.monotonic(), hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            require(time.monotonic() - start <= cap_seconds, 'Finite local hash cap exceeded')
            value.update(block)
    require(metadata(path) == before, 'File changed during full content hash')
    return value.hexdigest()


def pin(path):
    return {'path': Path(path).relative_to(ROOT).as_posix(), **metadata(path), 'sha256': sha(path)}


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key rejected')
            result[key] = value
        return result
    path = regular(path)
    require(path.stat().st_size <= 16 * 1024**2, 'Bounded metadata document required')
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def save(path, document):
    payload = (json.dumps(document, indent=2, allow_nan=False) + '\n').encode('utf-8')
    require(len(payload) <= 16 * 1024**2, 'Bounded receipt required')
    with Path(path).open('xb') as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    return pin(path)


def connection():
    """Inspect private profile in RAM; return only safe hashes and fixed paths.

    Rclone may refresh its token during authorized use. Stable application/scope
    identity is required every time; before/after complete-config SHA is evidence
    rather than a demand to freeze an expiring secret token forever.
    """
    old = read(OLD_PLAN)
    require(old.get('version') == 'exact-cold-backups-v1' and old.get('remote') == 'thesis_archive'
            and old.get('scope') == 'drive.file' and old.get('client_id_sha256') == CLIENT_SHA,
            'Historical Drive application binding differs')
    require(sha(EXE) == EXE_SHA, 'Exact reviewed rclone executable differs')
    require(not CONFIG.absolute().is_relative_to(ROOT), 'Private profile must stay outside project')
    profile = configparser.ConfigParser(interpolation=None)
    try:
        profile.read(regular(CONFIG), encoding='utf-8-sig')
        section = profile['thesis_archive']
        client_sha = hashlib.sha256(section.get('client_id', '').encode('utf-8')).hexdigest()
        okay = (section.get('type') == 'drive' and section.get('scope') == 'drive.file' and
                not section.get('team_drive', '').strip() and not section.get('root_folder_id', '').strip() and
                client_sha == CLIENT_SHA and bool(json.loads(section.get('token', '{}')).get('access_token')))
    except Exception:
        raise ValueError('Private Drive profile identity/scope check failed; secrets withheld') from None
    require(okay, 'Private Drive application, MyDrive destination or scope differs')
    overrides = sorted(name for name in os.environ if name.upper().startswith('RCLONE_CONFIG_THESIS_ARCHIVE_')
                       or name.upper().startswith('RCLONE_DRIVE_') or name.upper() in ('RCLONE_CONFIG',))
    require(not overrides, 'Conflicting rclone environment setting names present: ' + ','.join(overrides))
    return {'executable_path': str(EXE), 'executable_sha256': EXE_SHA,
            'private_config_path': str(CONFIG), 'private_config_sha256': sha(CONFIG),
            'client_id_sha256': client_sha, 'type': 'drive', 'scope': 'drive.file',
            'root_folder_id_empty': True, 'team_drive_empty': True,
            'historical_plan': pin(OLD_PLAN), 'private_body_or_token_exported': False}


def process_guard():
    """Full fresh CIM inspection; retain only hashes and matching public paths."""
    command = ('Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ParentProcessId,ExecutablePath,CommandLine '
               '| ConvertTo-Json -Depth 3 -Compress')
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    run = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                         capture_output=True, text=True, timeout=30, creationflags=flags)
    require(run.returncode == 0 and run.stdout.strip(), 'Complete fresh CIM process inspection unavailable')
    rows = json.loads(run.stdout)
    rows = rows if isinstance(rows, list) else [rows]
    require(len(rows) >= 3 and all('Name' in r and 'ProcessId' in r and 'CommandLine' in r for r in rows),
            'Incomplete process census rejected')
    names = {Path(name).name.lower() for name in ARCHIVES}
    groups = {Path(row['retained_root']).name.lower() for row in ARCHIVES.values()}
    transfer = re.compile(r'^(rclone|gcloud|tar|7z|gzip|robocopy|DriveFS|gsutil|pscp)\.exe$', re.I)
    normalize = lambda value: str(value).replace('\\', '/').casefold().strip()
    own_rows = [r for r in rows if r['ProcessId'] == os.getpid()]
    require(len(own_rows) == 1, 'Exact current process must appear once in full CIM census')
    own = own_rows[0]
    own_line = own.get('CommandLine')
    require(own_line and 'scripts/storage_offload_20261018_v1.py' in normalize(own_line),
            'Current process must be this exact invoked helper')
    launcher = (ROOT / 'venv/Scripts/python.exe').absolute()
    base = Path(getattr(sys, '_base_executable', sys.executable)).absolute()
    expected_exes = {normalize(launcher), normalize(base)}
    own_exe = own.get('ExecutablePath')
    require(own_exe and normalize(own_exe) in expected_exes and
            normalize(sys.executable) == normalize(launcher),
            'Current helper must use the expected venv launcher/base interpreter')
    actual_parent = os.getppid()
    require(own.get('ParentProcessId') == actual_parent, 'CIM/OS immediate parent relationship differs')
    relevant, blocked = [], []
    for row in rows:
        name = row['Name']
        if not (transfer.match(name) or re.match(r'^(python.*|powershell|pwsh)\.exe$', name, re.I)):
            continue
        line = row.get('CommandLine')
        require(line, 'Uninspectable relevant process command line: ' + str(row['ProcessId']))
        normalized = line.replace('\\', '/').lower()
        helper_marker = name.lower().startswith('python') and 'scripts/storage_offload_20261018_v1.py' in normalized
        is_self = helper_marker and row['ProcessId'] == os.getpid()
        # Windows venv's tiny python.exe launcher waits as our direct parent.
        # Only that exact launch pair may share the helper command. Equal text
        # alone is never a concurrency exemption, and no grandparent is allowed.
        own_parent_launcher = (helper_marker and row['ProcessId'] == actual_parent and
            normalize(line) == normalize(own_line) and row.get('ExecutablePath') and
            normalize(row['ExecutablePath']) == normalize(launcher) and
            normalize(own_exe) == normalize(base))
        other_helper = helper_marker and not is_self and not own_parent_launcher
        hits = sorted(value for value in names | groups if value in normalized)
        other_transfer = bool(transfer.match(name))
        if (hits and not is_self and not own_parent_launcher) or other_transfer or other_helper:
            blocked.append({'name': name, 'pid': row['ProcessId'], 'candidate_matches': hits,
                            'other_transfer_process': other_transfer, 'other_offload_helper': other_helper})
        relevant.append({'name': name, 'pid': row['ProcessId'], 'parent_pid': row.get('ParentProcessId'),
                         'command_line_sha256': hashlib.sha256(line.encode()).hexdigest(),
                         'candidate_matches': hits, 'this_exact_helper': is_self,
                         'verified_own_immediate_venv_parent_launcher': bool(own_parent_launcher),
                         'executable_is_expected_venv_or_base': bool(row.get('ExecutablePath') and
                             normalize(row['ExecutablePath']) in expected_exes)})
    require(not blocked, 'Competing candidate/transfer process; stop before mutation: ' + json.dumps(blocked))
    return {'complete': True, 'all_CIM_processes_inspected': len(rows),
            'relevant_processes': relevant, 'raw_command_lines_exported': False,
            'unrelated_disjoint_app_or_research_not_stopped': True}


def retention_proof(path, expected_sha):
    require(HEX.fullmatch(expected_sha or '') and sha(path) == expected_sha, 'Exact independent retention proof SHA required')
    proof = read(path)
    # The proof's exact row schema is intentionally narrow. It must establish
    # all current archive members still match their retained local files.
    require(proof.get('complete') is True and proof.get('full_member_retention_verified') is True,
            'Full independent archive/member retention proof required')
    rows = proof.get('archives')
    require(type(rows) is list and len(rows) == 3 and {r['archive_path'] for r in rows} == set(ARCHIVES),
            'Exact three archive proof rows required')
    for row in rows:
        expected = ARCHIVES[row['archive_path']]
        require(row.get('archive_sha256') == expected['sha256'] and row.get('archive_bytes') == expected['bytes']
                and row.get('retained_root') == expected['retained_root']
                and row.get('full_member_retention_verified') is True, 'Retained member/source join differs')
    return pin(path)


def plan_only(proof_name, proof_sha):
    require(not any((OUT / name).exists() for name in ('plan_v1.json', 'pass1_files.txt', 'pass1_sha256.txt')),
            'Preserve previous plan/lists; exclusive preparation required')
    require(bool(proof_name) == bool(proof_sha), 'Provide both proof path and SHA, or neither')
    proof = retention_proof(ROOT / relative(proof_name), proof_sha) if proof_name else None
    guard = process_guard()
    conn = connection()
    files = []
    for name, expected in sorted(ARCHIVES.items()):
        path = ROOT / relative(name)
        p = pin(path)
        require(p['bytes'] == expected['bytes'] and p['sha256'] == expected['sha256'], 'Fresh full archive bytes differ')
        require((ROOT / expected['retained_root']).is_dir(), 'Retained extracted return missing')
        files.append({**p, 'retained_root': expected['retained_root'],
                      'role': 'cold archive backup copy only; no deletion authorized'})
    file_payload = ''.join(row['path'] + '\n' for row in files).encode('ascii')
    sums = ''.join(row['sha256'] + '  ' + row['path'] + '\n' for row in files).encode('ascii')
    for name, payload in [('pass1_files.txt', file_payload), ('pass1_sha256.txt', sums)]:
        with (OUT / name).open('xb') as handle:
            handle.write(payload)
    plan = {'format': 'storage-offload-three-cold-archives-copy-only-20261018-v1', 'complete': True,
            'UTC': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT), 'destination': DESTINATION,
            'files': files, 'candidates': files, 'file_count': 3,
            'upload_bytes': sum(r['bytes'] for r in files), 'candidate_bytes': sum(r['bytes'] for r in files),
            'retention_proof': proof, 'connection': conn, 'process_guard': guard,
            'source_helper': pin(Path(__file__)), 'file_list': pin(OUT / 'pass1_files.txt'),
            'sha256_list': pin(OUT / 'pass1_sha256.txt'),
            'companions_models_outputs_caches_splits_provenance_failures_migration_all_retained': True,
            'filter_fusion_return_and_active_stage_all_protected': True,
            'member_retention_proof_pending': proof is None,
            'reversible_upload_allowed_while_member_proof_pending': True,
            'local_removal_requires_later_independent_complete_member_and_remote_proof': True,
            'limits': {'operation_seconds': MAX_SECONDS, 'rclone_log_and_stdout_bytes': MAX_LOG_BYTES,
                       'transfers': 2, 'drive_chunk_bytes': 32 * 1024**2},
            'files_deleted': 0, 'VM_training_calls': 0, 'prepared_only_not_uploaded_or_verified': True}
    pp = save(OUT / 'plan_v1.json', plan)
    print(json.dumps({'complete': True, 'mode': 'plan-only', 'plan': pp,
                      'archives': 3, 'bytes': plan['upload_bytes'], 'files_deleted': 0}))


def checked_plan(expected_sha):
    path = OUT / 'plan_v1.json'
    require(HEX.fullmatch(expected_sha or '') and sha(path) == expected_sha, 'Exact reviewed plan SHA required')
    plan = read(path)
    require(plan.get('complete') is True and plan.get('root') == str(ROOT) and
            plan.get('destination') == DESTINATION and plan.get('file_count') == 3 and
            {r['path'] for r in plan['files']} == set(ARCHIVES), 'Plan scope differs')
    require(pin(Path(__file__)) == plan['source_helper'], 'Reviewed transfer helper changed')
    for key in ('file_list', 'sha256_list'):
        recorded = plan[key]
        require(pin(ROOT / relative(recorded['path'])) == recorded, 'Frozen list/proof changed')
    if plan.get('retention_proof') is not None:
        recorded = plan['retention_proof']
        require(pin(ROOT / relative(recorded['path'])) == recorded, 'Frozen retention proof changed')
    require(plan['candidates'] == plan['files'] and plan.get('files_deleted') == 0,
            'Exact copy-only candidate list differs')
    for row in plan['files']:
        path = ROOT / relative(row['path'])
        require(metadata(path) == {k: row[k] for k in metadata(path)} and sha(path) == row['sha256'],
                'Current archive fileID/metadata/fullSHA differs')
    return plan


def operation_command(mode, plan, folder):
    """Pure exact command argument list; no shell interpolation or execution."""
    require(mode in ('upload', 'verify') and plan.get('destination') == DESTINATION,
            'Exact immutable copy/check destination required')
    require(plan['file_list']['path'] == (OUT / 'pass1_files.txt').relative_to(ROOT).as_posix()
            and plan['sha256_list']['path'] == (OUT / 'pass1_sha256.txt').relative_to(ROOT).as_posix(),
            'Exact frozen list paths required')
    log = folder / 'rclone.log'
    common = [str(EXE), '--config', str(CONFIG)]
    if mode == 'upload':
        command = common + ['copy', str(ROOT), DESTINATION, '--files-from-raw', str(ROOT / plan['file_list']['path']),
                            '--immutable', '--checksum', '--transfers', '2', '--checkers', '4', '--drive-chunk-size', '32M']
    else:
        command = common + ['check', str(ROOT / plan['sha256_list']['path']), DESTINATION,
                            '--checkfile', 'SHA256', '--download', '--one-way', '--combined', str(folder / 'combined.txt'),
                            '--checkers', '2']
    command += ['--tpslimit', '6', '--retries', '3' if mode == 'upload' else '1',
                '--low-level-retries', '5' if mode == 'upload' else '3', '--timeout', '2m', '--contimeout', '20s',
                '--max-duration', '8h', '--cutoff-mode', 'hard', '--stats', '60s',
                '--log-level', 'INFO', '--log-file', str(log)]
    return command


def operation(mode, expected_sha):
    require(mode in ('upload', 'verify'), 'Copy/verify mode only')
    plan = checked_plan(expected_sha)
    guard = process_guard()
    conn = connection()
    require(conn['executable_sha256'] == plan['connection']['executable_sha256'] and
            conn['client_id_sha256'] == plan['connection']['client_id_sha256'], 'Drive app/executable changed')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    folder = OUT / (mode + '_pass1_' + stamp)
    folder.mkdir()
    log, stdout = folder / 'rclone.log', folder / 'stdout_stderr.log'
    command = operation_command(mode, plan, folder)
    save(folder / 'intent.json', {'complete_intent': True, 'mode': mode, 'plan_sha256': expected_sha,
         'command': command, 'process_guard': guard, 'connection_before': conn,
         'private_profile_body_logged': False, 'max_seconds': MAX_SECONDS, 'max_log_bytes': MAX_LOG_BYTES,
         'files_deleted': 0, 'VM_training_calls': 0})
    start = time.monotonic()
    reason, code, process = None, None, None
    try:
        with stdout.open('xb') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
            save(folder / 'process_started.json', {'pid': process.pid, 'mode': mode,
                 'plan_sha256': expected_sha, 'started_UTC': datetime.now(timezone.utc).isoformat(),
                 'exact_own_child_only': True, 'max_seconds': MAX_SECONDS})
            while process.poll() is None:
                elapsed = time.monotonic() - start
                logbytes = sum(p.stat().st_size for p in (stdout, log) if p.exists())
                if elapsed > MAX_SECONDS or logbytes > MAX_LOG_BYTES:
                    reason = 'outer_time_cap' if elapsed > MAX_SECONDS else 'log_byte_cap'
                    process.kill(); process.wait(timeout=30); break
                time.sleep(1)
            code = process.wait(timeout=30)
        complete = code == 0 and reason is None
        require(sum(p.stat().st_size for p in (stdout, log) if p.exists()) <= MAX_LOG_BYTES,
                'Final log byte cap exceeded; retain evidence')
        combined = None
        if mode == 'verify' and complete:
            result = regular(folder / 'combined.txt')
            require(result.stat().st_size <= 65536, 'Bounded three-file comparison required')
            actual = result.read_text(encoding='utf-8-sig').splitlines()
            expected = sorted('= ' + row['path'] for row in plan['files'])
            require(sorted(actual) == expected and len(actual) == 3, 'All three streamed SHA256 comparisons must match')
            combined = pin(result)
        after = connection()
        receipt = {'complete': complete, 'mode': mode, 'plan_sha256': expected_sha, 'destination': DESTINATION,
            'rclone_exit_code': code, 'external_stop_reason': reason, 'seconds': time.monotonic() - start,
            'child_pid': process.pid, 'child_terminal': True, 'expected_archives': 3,
            'expected_bytes': plan['upload_bytes'], 'streamed_downloaded_SHA256_verified': mode == 'verify' and complete,
            'comparison': combined, 'connection_after': after, 'stdout': pin(stdout),
            'rclone_log': pin(log) if log.exists() else None, 'files_deleted': 0,
            'VM_training_calls': 0, 'cloud_copy_not_local_removal_authorization': True,
            'partial_remote_or_failure_evidence_retained': True}
        save(folder / 'receipt.json', receipt)
        print(json.dumps({'receipt': str(folder.relative_to(ROOT) / 'receipt.json'),
                          'complete': complete, 'mode': mode, 'pid': process.pid, 'exit': code, 'files_deleted': 0}))
        require(complete, 'Transfer/verification did not complete; retain local originals and operation evidence')
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.kill(); process.wait(timeout=30)
        save(folder / 'failure.json', {'complete': False, 'mode': mode, 'plan_sha256': expected_sha,
             'exception_type': type(error).__name__, 'reason': str(error)[:2048],
             'child_pid': process.pid if process else None, 'child_terminal': process is None or process.poll() is not None,
             'seconds': time.monotonic() - start, 'files_deleted': 0,
             'all_local_original_archives_retained': True, 'VM_training_calls': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--upload', action='store_true')
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--inspect', action='store_true')
    parser.add_argument('--proof')
    parser.add_argument('--proof-sha256')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    require(OUT.is_dir(), 'Fresh inventory folder must exist before this helper')
    if args.inspect:
        print(json.dumps({'complete': True, 'mode': 'inspect-only', 'connection': connection(),
                          'process_guard': process_guard(),
                          'candidates_metadata_only': {name: metadata(ROOT / name) for name in ARCHIVES},
                          'new_files_written': 0, 'network_transfer_calls': 0, 'files_deleted': 0}))
    elif args.upload or args.verify:
        operation('upload' if args.upload else 'verify', args.plan_sha256)
    else:
        plan_only(args.proof, args.proof_sha256)


if __name__ == '__main__':
    main()
