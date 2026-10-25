"""Exact six HOME archive-copy maintenance. No training, extraction or broad removal.

R1 report-only repair preserves the verified original source and payload lineage.
Only --apply can unlink, after a complete verify payload and an independently
reviewed exact plan. Every other research byte and namespace entry is retained.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import time
import urllib.request

HOME = Path('/home/janusdominic0')
RESEARCH = HOME / 'forensic-dgp'
AUDIT = HOME / 'vm_archive_offload_20261018_v1'
UID = 1001
INSTANCE = 'forensic-dgp-thesis'
INSTANCE_ID = '4410777042005672095'
WHOLE_SECONDS = 1800
HASH_SECONDS = 1200
MAX_JSON = 256 * 1024**2
MAX_ENTRIES = 1_000_000
CHUNK = 4 * 1024**2
MODEL_SUFFIXES = frozenset({'.pth', '.pt', '.ckpt', '.safetensors', '.onnx'})
RECORD_SUFFIXES = frozenset({'.py', '.md', '.json', '.jsonl', '.log', '.csv', '.txt', '.sh',
                             '.yaml', '.yml', '.sha256'})
SMALL_RECORD_BYTES = 4 * 1024**2
FULL_ROOTS = (RESEARCH / 'cctv_codeformer_filter_fusion_training_v1_vm',
              RESEARCH / 'cctv_codeformer_role_preserving_training_v1_r2_vm')
HOME_KEEP_PREFIXES = ('cctv-codeformer-filter-fusion-training-v1',
                      'cctv-codeformer-role-preserving-training-v1-r2')
META_KEYS = ('bytes', 'allocated_bytes', 'uid', 'gid', 'nlink', 'inode', 'device',
             'mtime_ns', 'ctime_ns', 'mode')
EXPECTED_NAMES = frozenset({
    'cctv-codeformer-adain-gradient-isolation-v1-results.tar.gz',
    'cctv-codeformer-tone-control-training-v1-results.tar.gz',
    'cctv-codeformer-normal-tone-training-v1-results.tar.gz',
    'cctv-codeformer-fp32-comparison-v1-results.tar.gz',
    'cctv-codeformer-adaptation-v2-results.tar.gz',
    'cctv-codeformer-adaptation-v1-results.tar.gz',
})
TOTAL_BYTES = 16_692_148_382


def require(okay, message):
    if not okay:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def canonical_sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def path_policy(name):
    require(isinstance(name, str), 'Candidate path must be text')
    p = PurePosixPath(name)
    require(p.parent == PurePosixPath('/home/janusdominic0') and p.name in EXPECTED_NAMES and str(p) == name,
            'Only the six exact HOME result archive copies are eligible')
    return Path(name)


def meta_from_stat(s):
    return {'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512,
            'uid': s.st_uid, 'gid': s.st_gid, 'nlink': s.st_nlink,
            'inode': s.st_ino, 'device': s.st_dev, 'mtime_ns': s.st_mtime_ns,
            'ctime_ns': s.st_ctime_ns, 'mode': s.st_mode}


def safe(path, regular=False):
    path = Path(path)
    require(path.is_absolute() and HOME in (path, *path.parents), 'Exact HOME scope required')
    for p in (path, *path.parents):
        s = p.lstat()
        require(not stat.S_ISLNK(s.st_mode), 'Symlink ancestor refused: ' + str(p))
        if p == HOME:
            break
    if regular:
        s = path.lstat()
        require(stat.S_ISREG(s.st_mode), 'Regular file required: ' + str(path))
    return path


def candidate_meta(name):
    p = safe(path_policy(name), regular=True)
    s = p.lstat()
    require(s.st_uid == UID and s.st_nlink == 1, 'Owned unique candidate required')
    return meta_from_stat(s)


def same_expected(actual, expected):
    require(all(k in expected for k in ('bytes', 'allocated_bytes', 'uid', 'nlink', 'inode',
                                       'device', 'mtime_ns', 'ctime_ns', 'mode')),
            'Full fresh candidate metadata required')
    return all(actual[k] == expected[k] for k in expected if k in META_KEYS)


def bounded_read_json(path, expected_sha=None):
    p = safe(path, regular=True)
    require(p.stat().st_size <= MAX_JSON, 'Bounded metadata JSON required')
    data = p.read_bytes()
    if expected_sha is not None:
        require(hashlib.sha256(data).hexdigest() == expected_sha, 'Exact JSON source hash differs')
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def save_new(path, value):
    data = encoded(value)
    require(len(data) <= MAX_JSON, 'Receipt byte cap')
    with Path(path).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def save_gzip_new(path, value):
    data = encoded(value)
    require(len(data) <= MAX_JSON, 'Compressed evidence decoded byte cap')
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as f:
            f.write(data)
        raw.flush(); os.fsync(raw.fileno())
    return file_hash_pin(path)


def file_hash_pin(path, budget=None):
    p = safe(path, regular=True)
    before = meta_from_stat(p.lstat())
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            if budget: budget()
            h.update(b)
    require(meta_from_stat(p.lstat()) == before, 'File changed during opaque hash: ' + str(p))
    return {'path': str(p), 'metadata': before, 'bytes': before['bytes'], 'sha256': h.hexdigest()}


def read_gzip_payload(path, expected_sha):
    p = safe(path, regular=True)
    require(file_hash_pin(p)['sha256'] == expected_sha, 'Prior verification payload changed')
    with gzip.open(p, 'rb') as f:
        data = f.read(MAX_JSON + 1)
        require(len(data) <= MAX_JSON and not f.read(1), 'Bounded decoded verification payload')
    return json.loads(data)


def validate_request(request):
    require(request.get('complete') is True and request.get('format') == 'vm-six-home-archive-maintenance-request-v1',
            'Exact reviewed request format required')
    rows = request.get('candidates')
    require(type(rows) is list and len(rows) == 6 and
            {Path(r['vm_path']).name for r in rows} == EXPECTED_NAMES,
            'All six exact candidates required')
    require(sum(r['expected_metadata']['bytes'] for r in rows) == TOTAL_BYTES,
            'Exact fixed archive byte total required')
    for r in rows:
        path_policy(r['vm_path'])
        require(r['expected_metadata']['uid'] == UID and r['expected_metadata']['nlink'] == 1
                and stat.S_ISREG(r['expected_metadata']['mode']), 'Candidate metadata not eligible')
        b = r['backup']
        require(r['sha256'] == b['sha256'] and b['bytes'] == r['expected_metadata']['bytes']
                and b.get('fresh_full_windows_hash_verified') is True,
                'Current complete Windows backup hash/bytes required')
        require(len(r['sha256']) == 64 and all(c in '0123456789abcdef' for c in r['sha256']),
                'Exact SHA256 required')
    return rows


def runtime_identity():
    require(os.name == 'posix' and os.getuid() == UID and Path.home() == HOME
            and socket.gethostname().split('.')[0] == INSTANCE, 'Exact Linux user/host/HOME required')
    safe(HOME); safe(RESEARCH); safe(AUDIT)
    require(stat.S_ISDIR(AUDIT.lstat().st_mode) and AUDIT.lstat().st_uid == UID
            and stat.S_IMODE(AUDIT.lstat().st_mode) == 0o700, 'Exclusive owned0700 audit namespace required')
    def metadata(key):
        req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/' + key,
                                     headers={'Metadata-Flavor': 'Google'})
        with urllib.request.urlopen(req, timeout=5) as reply:
            return reply.read(4096).decode()
    identity = {k: metadata(k) for k in ('project/project-id', 'instance/id', 'instance/name',
                                        'instance/zone', 'instance/machine-type')}
    validate_identity_values(identity, UID, socket.gethostname(), str(HOME))
    return {'metadata': identity, 'uid': UID, 'hostname': socket.gethostname(),
            'home': str(HOME), 'research_root': str(RESEARCH)}


def validate_identity_values(identity, uid, hostname, home):
    require(uid == UID and hostname.split('.')[0] == INSTANCE and home == '/home/janusdominic0'
            and identity['project/project-id'] == INSTANCE and identity['instance/id'] == INSTANCE_ID
            and identity['instance/name'] == INSTANCE and identity['instance/zone'].endswith('/us-central1-a')
            and identity['instance/machine-type'].endswith('/g2-standard-4'), 'Fresh metadata identity mismatch')


def call(args, cap=30):
    p = subprocess.run(args, capture_output=True, text=True, timeout=cap)
    require(len(p.stdout.encode()) + len(p.stderr.encode()) <= 8 * 1024**2, 'Bounded subprocess output required')
    require(p.returncode == 0, 'Read-only command failed: ' + args[0] + ': ' + p.stderr[:200])
    return {'argv': args, 'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def classify_process(row, own_pid, parent_pid, names):
    pid, ppid, comm, args = row
    if pid in (own_pid, parent_pid):
        return None
    lower = args.casefold()
    hits = sorted(n for n in names if n.casefold() in lower)
    if hits:
        return 'candidate_dependency:' + ','.join(hits)
    try: tokens = shlex.split(args)
    except ValueError: tokens = args.split()
    executable = comm.casefold()
    is_python = executable.startswith('python') or executable in ('torchrun', 'accelerate')
    flags = {t.casefold() for t in tokens if t.startswith('--')}
    script = [Path(t).name.casefold() for t in tokens if t.endswith('.py')]
    training_name = any(any(w in t for w in ('train', 'supervise', 'trainer')) for t in script)
    mutating_mode = bool(flags & {'--train', '--run', '--launch', '--resume'})
    if executable in ('torchrun', 'accelerate') or (is_python and str(RESEARCH) in args and (training_name or mutating_mode)):
        return 'active_research_training_or_execution'
    return None


def workload_guard(rows):
    gpu = call(['nvidia-smi', '--query-gpu=name,memory.used,memory.total', '--format=csv,noheader'])
    processes = call(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
    gpu_rows = gpu['stdout'].strip().splitlines()
    require(len(gpu_rows) == 1 and 'NVIDIA L4' in gpu_rows[0] and
            gpu_rows[0].split(',')[1].strip() == '0 MiB' and not processes['stdout'].strip(),
            'One L4 must have0MiB and no compute workload')
    tmux = call(['tmux', 'list-panes', '-a', '-F', '#S|#P|#{pane_current_command}|#{pane_pid}|#{pane_current_path}|#{pane_dead}'])
    for line in tmux['stdout'].splitlines():
        parts = line.split('|')
        require(len(parts) == 6, 'Complete tmux pane record required')
        if parts[4].startswith(str(RESEARCH)):
            require(parts[2] in ('bash', 'sh', 'zsh', 'fish') and parts[5] == '0',
                    'Active research tmux pane blocks maintenance')
    ps = call(['ps', '-eo', 'pid=,ppid=,comm=,args='])
    names = [r['vm_path'] for r in rows] + [Path(r['vm_path']).name for r in rows]
    blockers = []
    for line in ps['stdout'].splitlines():
        p = line.strip().split(None, 3)
        if len(p) != 4: continue
        reason = classify_process((int(p[0]), int(p[1]), p[2], p[3]), os.getpid(), os.getppid(), names)
        if reason: blockers.append({'pid': int(p[0]), 'comm': p[2], 'reason': reason})
    require(not blockers, 'Competing candidate/research workload: ' + json.dumps(blockers))
    return {'complete': True, 'UTC': utc(), 'gpu': gpu, 'gpu_processes': processes,
            'tmux': tmux, 'processes': ps, 'unrelated_disjoint_workflows_not_stopped': True}


# Privileged READ-ONLY scanner so root-owned readers cannot hide from uid1001.
# Args are JSON, not shell code. No fd/path is opened for writing or removed.
PROC_FD_SOURCE = '''import json,os,stat,sys,time
rows=json.loads(sys.argv[1]);skip=json.loads(sys.argv[2]);targets={(r['device'],r['inode']):r['path'] for r in rows};out=[];count=0;start=time.monotonic()
assert os.geteuid()==0
for entry in os.scandir('/proc'):
 if not entry.name.isdigit():continue
 assert time.monotonic()-start<45
 pid=int(entry.name);folder=entry.path+'/fd'
 try:fds=list(os.scandir(folder))
 except (FileNotFoundError,ProcessLookupError):continue
 for fd in fds:
  count+=1
  try:s=os.stat(fd.path);link=os.readlink(fd.path)
  except (FileNotFoundError,ProcessLookupError):continue
  key=(s.st_dev,s.st_ino)
  if key in targets and [pid,int(fd.name)] not in skip:out.append({'pid':pid,'fd':int(fd.name),'target':targets[key],'link':link})
print(json.dumps({'complete':True,'root_read_only_proc_fd_census':True,'fd_entries_inspected':count,'readers':out,'seconds':time.monotonic()-start}))
'''


def open_reader_guard(rows, allowed_own_fd=None):
    targets = [{'path': r['vm_path'], 'device': r['expected_metadata']['device'],
                'inode': r['expected_metadata']['inode']} for r in rows]
    skip = [] if allowed_own_fd is None else [[os.getpid(), allowed_own_fd]]
    data = call(['sudo', '-n', 'python3', '-B', '-c', PROC_FD_SOURCE,
                 json.dumps(targets, separators=(',', ':')), json.dumps(skip)], cap=60)
    census = json.loads(data['stdout'])
    require(census.get('complete') is True and census.get('root_read_only_proc_fd_census') is True
            and not census.get('readers'), 'Open archive reader blocks maintenance')
    return census


def namespace():
    pending, result = [RESEARCH], {}
    while pending:
        folder = pending.pop()
        s = folder.lstat()
        require(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode), 'Directory traversal without symlink following required')
        with os.scandir(folder) as entries:
            for entry in entries:
                p = Path(entry.path); info = entry.stat(follow_symlinks=False)
                row = meta_from_stat(info)
                if stat.S_ISDIR(info.st_mode):
                    row['kind'] = 'directory'; pending.append(p)
                elif stat.S_ISREG(info.st_mode): row['kind'] = 'regular'
                elif stat.S_ISLNK(info.st_mode): row.update(kind='symlink', target=os.readlink(p))
                else: raise ValueError('Unexpected special research entry: ' + str(p))
                result[str(p)] = row
                require(len(result) <= MAX_ENTRIES, 'Research namespace entry cap')
    return result


def protected_names(space):
    result = {name for name, row in space.items() if row['kind'] == 'regular'
              and (Path(name).suffix.casefold() in MODEL_SUFFIXES or
                   (Path(name).suffix.casefold() in RECORD_SUFFIXES and row['bytes'] <= SMALL_RECORD_BYTES) or
                   any(Path(name).is_relative_to(p) for p in FULL_ROOTS))}
    for folder in FULL_ROOTS:
        require(str(folder) in space and space[str(folder)]['kind'] == 'directory',
                'Latest fusion and current R2 dependency roots must remain')
    for p in HOME.iterdir():
        if p.name.startswith(HOME_KEEP_PREFIXES):
            safe(p, regular=True); result.add(str(p))
    require(any(Path(n).suffix.casefold() in MODEL_SUFFIXES for n in result), 'Protected model census cannot be empty')
    require(any(Path(n).parent == HOME and Path(n).name.startswith(HOME_KEEP_PREFIXES[0]) for n in result),
            'Latest fusion HOME evidence cannot be absent')
    return sorted(result)


def protected_hashes(space):
    start = time.monotonic()
    def budget(): require(time.monotonic() - start <= HASH_SECONDS, '1200s protected hash cap')
    result = {}
    for i, name in enumerate(protected_names(space), 1):
        budget(); result[name] = file_hash_pin(name, budget)
        if i % 1000 == 0: print(json.dumps({'protected_hashed': i}), flush=True)
    return result


def verify_candidates(rows):
    start = time.monotonic(); result = []
    def budget(): require(time.monotonic() - start <= HASH_SECONDS, '1200s candidate hash cap')
    for row in rows:
        actual = candidate_meta(row['vm_path'])
        require(same_expected(actual, row['expected_metadata']), 'Current candidate metadata differs')
        pin = file_hash_pin(row['vm_path'], budget)
        require(pin['sha256'] == row['sha256'] and pin['bytes'] == row['backup']['bytes'],
                'VM archive differs from current complete Windows backup')
        result.append({**row, 'verified_current_pin': pin})
    return result


def disk():
    d = shutil.disk_usage(HOME)
    result = {'total_bytes': d.total, 'used_bytes': d.used, 'free_bytes': d.free,
              'df': call(['df', '-B1', str(HOME)])}
    require(len(encoded(result)) <= 64 * 1024, 'Bounded64KiB disk/df evidence required')
    return result


def verify_only(request_path, request_sha):
    request = bounded_read_json(request_path, request_sha); rows = validate_request(request)
    require(not (AUDIT / 'verify_intent.json').exists(), 'Existing verification attempt retained; no automatic replay')
    identity = runtime_identity()
    save_new(AUDIT / 'verify_intent.json', {'request_sha256': request_sha, 'guest_source_sha256': file_hash_pin(__file__)['sha256'],
                                         'UTC': utc(), 'identity': identity, 'files_removed': 0})
    workload = workload_guard(rows); readers = open_reader_guard(rows)
    before_disk = disk(); space = namespace(); protection = protected_hashes(space)
    candidates = verify_candidates(rows)
    require(namespace() == space, 'Research namespace changed during read-only verification')
    require(workload_guard(rows)['complete'] and open_reader_guard(rows)['complete'], 'Fresh idle/reader closure required')
    result = {'format': 'vm-six-home-archive-verify-payload-v1', 'complete': True,
              'identity': identity, 'request_sha256': request_sha,
              'guest_source_sha256': file_hash_pin(__file__)['sha256'], 'UTC': utc(),
              'candidates': candidates, 'namespace_before': space, 'protected_before': protection,
              'protected_sha256': canonical_sha(protection), 'namespace_sha256': canonical_sha(space),
              'candidate_current_sha256': canonical_sha(candidates),
              'workload_before': workload, 'open_reader_before': readers, 'disk_before': before_disk,
              'files_removed': 0, 'training_calls': 0, 'all_research_namespace_metadata_unchanged': True}
    pin = save_gzip_new(AUDIT / 'verify_payload.json.gz', result)
    receipt = {'complete': True, 'mode': 'verify-only', 'payload': pin,
               'candidate_count': 6, 'candidate_logical_bytes': TOTAL_BYTES,
               'candidate_allocated_bytes': sum(r['verified_current_pin']['metadata']['allocated_bytes'] for r in candidates),
               'protected_files': len(protection), 'namespace_entries': len(space),
               'protected_sha256': result['protected_sha256'], 'namespace_sha256': result['namespace_sha256'],
               'candidate_current_sha256': result['candidate_current_sha256'],
               'files_removed': 0, 'training_calls': 0}
    save_new(AUDIT / 'verify_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def require_review(plan, plan_sha, review):
    require(plan.get('complete') is True and plan.get('format') == 'vm-six-home-archive-exact-apply-plan-v1',
            'Complete exact plan required')
    require(review.get('complete') is True and review.get('approved_for_exact_six_HOME_archive_apply') is True
            and review.get('plan_sha256') == plan_sha
            and review.get('guest_source_sha256') == plan['guest_source']['sha256']
            and review.get('windows_backup_scope_sha256') == plan['windows_backup_scope']['sha256']
            and review.get('protected_sha256') == plan['verified']['protected_sha256']
            and review.get('candidate_current_sha256') == plan['verified']['candidate_current_sha256'],
            'Independent source/Windows-backup/current-candidate/protected preapply closure required')


def remove_one(row, rows, ledger):
    import fcntl  # Linux-only mutation primitive; no import/runtime calls in pure CPU fixtures.
    name = row['vm_path']; path = path_policy(name)
    require(same_expected(candidate_meta(name), row['expected_metadata']), 'Immediate candidate identity differs')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(same_expected(meta_from_stat(os.fstat(fd)), row['expected_metadata']), 'Opened candidate identity differs')
        value = hashlib.sha256(); start = time.monotonic()
        while True:
            require(time.monotonic() - start <= HASH_SECONDS, '1200s immediate opaque hash cap')
            b = os.read(fd, CHUNK)
            if not b: break
            value.update(b)
        require(value.hexdigest() == row['sha256'], 'Immediate opened candidate SHA differs')
        workload_guard(rows); reader = open_reader_guard(rows, allowed_own_fd=fd)
        require(same_expected(candidate_meta(name), row['expected_metadata']) and
                same_expected(meta_from_stat(os.fstat(fd)), row['expected_metadata']), 'Preunlink identity changed')
        event = {'event': 'removal_intent', 'path': name, 'sha256': row['sha256'],
                 'metadata': row['expected_metadata'], 'actual_open_reader_guard': reader,
                 'directory_fd_scoped_unlink': True, 'advisory_flock_not_claimed_kernel_exclusive_delete': True, 'UTC': utc()}
        ledger.write(encoded(event).decode()); ledger.flush(); os.fsync(ledger.fileno())
        homefd = os.open(HOME, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.unlink(path.name, dir_fd=homefd)
        finally: os.close(homefd)
        require(not path.exists(), 'Exact file remained after scoped unlink')
        ledger.write(encoded({'event': 'removed', 'path': name, 'sha256': row['sha256'],
                              'logical_bytes': row['expected_metadata']['bytes'],
                              'allocated_bytes': row['expected_metadata']['allocated_bytes'], 'UTC': utc()}).decode())
        ledger.flush(); os.fsync(ledger.fileno())
    finally:
        os.close(fd)


def after_report_budget(space, protection):
    maps_bytes = len(encoded({'namespace_after': space, 'protected_after': protection}))
    reserve = 8 * 1024**2
    require(maps_bytes + reserve <= MAX_JSON, 'Prospective256MiB after-report budget fails before removal')
    return {'complete': True, 'maps_serialized_bytes': maps_bytes, 'nonmap_reservation_bytes': reserve,
            'prospective_total_upper_bytes': maps_bytes + reserve, 'maximum_decoded_report_bytes': MAX_JSON,
            'full_before_evidence_retained_in_exact_verified_payload': True,
            'actual_after_maps_required_byte_equal_to_before_maps': True}


def require_verified_lineage(verified, plan):
    require(verified['complete'] is True and verified['files_removed'] == 0
            and verified['guest_source_sha256'] == plan['verified']['guest_source_sha256']
            and plan['compatibility_lineage']['original_guest_source']['sha256'] == verified['guest_source_sha256']
            and verified['request_sha256'] == plan['request_sha256']
            and verified['protected_sha256'] == plan['verified']['protected_sha256']
            and verified['candidate_current_sha256'] == plan['verified']['candidate_current_sha256'],
            'Verified original source, exact plan/protection/candidate binding differs')


def apply(plan_path, plan_sha, review_path, review_sha):
    plan = bounded_read_json(plan_path, plan_sha); review = bounded_read_json(review_path, review_sha)
    require_review(plan, plan_sha, review)
    require(file_hash_pin(__file__)['sha256'] == plan['guest_source']['sha256'], 'Reviewed guest source changed')
    verified = read_gzip_payload(AUDIT / 'verify_payload.json.gz', plan['verified']['payload_sha256'])
    request = plan['request']; rows = validate_request(request)
    require_verified_lineage(verified, plan)
    identity = runtime_identity()
    # This exclusive durable marker is created before any apply work; failures are never automatically replayed.
    save_new(AUDIT / 'apply_intent.json', {'plan_sha256': plan_sha, 'review_sha256': review_sha,
              'identity': identity, 'UTC': utc(), 'failure_or_partial_attempt_must_not_auto_replay': True})
    workload_guard(rows); open_reader_guard(rows)
    space = namespace(); require(space == verified['namespace_before'], 'Research namespace changed before apply')
    protection = protected_hashes(space)
    require(protection == verified['protected_before'], 'Protected content/identity changed before apply')
    candidates = verify_candidates(rows)
    require(candidates == verified['candidates'], 'Current VM candidate pins differ from independent plan')
    report_budget = after_report_budget(space, protection)
    save_new(AUDIT / 'apply_report_budget.json', report_budget)
    before_disk = disk(); removed = []
    try:
        with (AUDIT / 'removed_ledger.jsonl').open('x', encoding='utf-8', newline='\n') as ledger:
            for row in rows:
                remove_one(row, rows, ledger); removed.append(row['vm_path'])
        after_space = namespace(); require(after_space == space, 'Research namespace changed during apply')
        after_protection = protected_hashes(after_space)
        require(after_protection == protection, 'Protected content changed during apply')
        require(all(not path_policy(r['vm_path']).exists() for r in rows), 'All exact six archive paths must be absent')
        after_disk = disk(); workload_after = workload_guard(rows)
        result = {'format': 'vm-six-home-archive-apply-payload-v1', 'complete': True,
                  'plan_sha256': plan_sha, 'review_sha256': review_sha, 'identity': identity,
                  'verified_payload_sha256': plan['verified']['payload_sha256'],
                  'files_removed': len(removed), 'exact_removed_paths': removed,
                  'logical_bytes_removed': TOTAL_BYTES,
                  'allocated_bytes_removed': sum(r['expected_metadata']['allocated_bytes'] for r in rows),
                  'disk_before': before_disk, 'disk_after': after_disk,
                  'observed_free_increase_bytes': after_disk['free_bytes'] - before_disk['free_bytes'],
                  'verified_before': {'payload_sha256': plan['verified']['payload_sha256'],
                       'guest_source_sha256': verified['guest_source_sha256'],
                       'namespace_sha256': verified['namespace_sha256'],
                       'protected_sha256': verified['protected_sha256']},
                  'namespace_after': after_space, 'protected_after': after_protection,
                  'all_research_namespace_and_protected_content_unchanged': True,
                  'workload_after': save_new(AUDIT / 'apply_workload_after.json', workload_after),
                  'report_budget': report_budget, 'training_calls': 0,
                  'no_cache_checkpoint_source_MD_or_research_root_removed': True,
                  'ledger': file_hash_pin(AUDIT / 'removed_ledger.jsonl'), 'UTC': utc()}
        remainder = {k: v for k, v in result.items() if k not in ('namespace_after', 'protected_after')}
        require(len(encoded(remainder)) <= 8 * 1024**2, 'Fixed8MiB nonmap report reservation exceeded')
        pin = save_gzip_new(AUDIT / 'apply_payload.json.gz', result)
        receipt = {k: result[k] for k in ('complete', 'files_removed', 'exact_removed_paths', 'logical_bytes_removed',
                    'allocated_bytes_removed', 'observed_free_increase_bytes', 'training_calls')}
        receipt.update(mode='apply', payload=pin, plan_sha256=plan_sha, review_sha256=review_sha,
                       all_research_namespace_and_protected_content_unchanged=True)
        save_new(AUDIT / 'apply_receipt.json', receipt); print(json.dumps(receipt), flush=True)
    except BaseException as error:
        save_new(AUDIT / 'apply_failure.json', {'complete': False, 'removed_paths_confirmed': removed,
                 'exception_type': type(error).__name__, 'reason': str(error)[:2048],
                 'plan_sha256': plan_sha, 'review_sha256': review_sha, 'UTC': utc(),
                 'no_automatic_replay': True, 'training_calls': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-only', action='store_true'); mode.add_argument('--apply', action='store_true')
    parser.add_argument('--request'); parser.add_argument('--request-sha256')
    parser.add_argument('--plan'); parser.add_argument('--plan-sha256')
    parser.add_argument('--review'); parser.add_argument('--review-sha256')
    args = parser.parse_args()
    def alarm(signum, frame): raise TimeoutError('Finite1800s whole guest operation stop')
    signal.signal(signal.SIGALRM, alarm); signal.alarm(WHOLE_SECONDS)
    try:
        if args.verify_only:
            require(args.request and args.request_sha256, 'Exact request SHA required')
            verify_only(args.request, args.request_sha256)
        else:
            require(args.plan and args.plan_sha256 and args.review and args.review_sha256, 'Exact plan and peer closure SHA required')
            apply(args.plan, args.plan_sha256, args.review, args.review_sha256)
    except BaseException as error:
        if AUDIT.is_dir():
            target = AUDIT / ('verify_terminal_failure.json' if args.verify_only else 'apply_terminal_failure.json')
            if not target.exists():
                save_new(target, {'complete': False, 'exception_type': type(error).__name__,
                         'reason': str(error)[:2048], 'UTC': utc(), 'no_automatic_replay': True,
                         'removed_ledger_if_present_is_authoritative': True, 'training_calls': 0})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
