"""Inert stdlib-only verifier, finite manual supervisor and own-stage exporter."""
import argparse
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time

PACKAGE = 'research/codeformer_u64_gate_aligned_pilot_v1'
STAGE = 'cctv_codeformer_u64_gate_aligned_pilot_v1_vm'
PARENT_STAGE = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
OUTPUT = 'outputs/u64_gate_aligned_pilot_v1'
PREFIX = 'cctv-codeformer-u64-gate-aligned-pilot-v1'
FORMAT = 'u64-gate-aligned-pilot-manual-L4-v1'
PARENT = {'stage_name': PARENT_STAGE, 'bytes': 279879,
          'sha256': '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'}
COUNTERS = {'full_forward_attempts': 540, 'full_forward_returns': 540,
            'gradient_query_attempts': 512, 'gradient_query_returns': 512,
            'parameter_application_attempts': 4, 'parameter_applications': 4,
            'initial_quality_forwards': 56, 'initial_repeat_forwards': 4,
            'optimization_forwards': 256, 'quality_forwards': 224}
CATEGORIES = ('initial_quality_forwards', 'initial_repeat_forwards', 'optimization_forwards', 'quality_forwards')
OPERATIONS = {'forward': ('full_forward_attempts', 'full_forward_returns'),
              'gradient_query': ('gradient_query_attempts', 'gradient_query_returns'),
              'parameter_application': ('parameter_application_attempts', 'parameter_applications')}
HELPER_PINS = {
 'research/codeformer_u64_gate_aligned_v1/__init__.py': (87, '9eb734f64e4e7871f2357f8af11cabaf3a4b4efea1bf4d148bfb6455eaafadca'),
 'research/codeformer_u64_role_gradients_v1/__init__.py': (82, 'b13cc93cf4d889fde3d255356e1c9e7e483aa845de75e2ea4a94a2c562554baf'),
 'research/codeformer_u64_role_gradients_v1/contract.py': (11189, '90a11da6be2e531648e4c5051471946469e287d2bc151177006554932395074e'),
 'research/codeformer_u64_gate_aligned_v1/solver.py': (7511, '0e68b9ab6d7fbd019dbdee6119530b718fa29f38e0ea80aefc1891ecccea91ba'),
 'research/codeformer_u64_gate_aligned_v1/contract.py': (10905, '18bfd58795380e1ff2b3c1fab2179c87e7dfb19bf4a46cd0a03e7f6c10dcf4ca')}
GRADIENT_PROTOCOL = {'bytes': 184633, 'sha256': '07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3'}
ENV = 'FORENSIC_U64_GATE_ALIGNED_PILOT_'
RUNTIME = frozenset({'supervisor.start.json', 'supervisor.worker.started.json',
    'supervisor.worker.json', 'supervisor.export.started.json', 'supervisor.export.json',
    'supervisor.json', 'supervisor_failure.json', 'supervisor.worker.resources.jsonl', 'supervisor.export.resources.jsonl',
    'run.log', 'export.log', 'export.json', 'export_failure.json'})
POST_EXPORT = frozenset({'export.log', 'supervisor.export.started.json', 'supervisor.export.json',
                        'supervisor.json', 'export.json', 'export_failure.json', 'supervisor_failure.json', 'supervisor.export.resources.jsonl'})
MAX_FILE = 2*1024**3


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON: '+value)
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=bad)
    def walk(x):
        if type(x) is float:
            import math
            need(math.isfinite(x), 'Finite JSON scalar')
        elif type(x) in (dict, list):
            for child in x.values() if type(x) is dict else x:
                walk(child)
    walk(value)
    return value


def relative(name):
    need(type(name) is str and name and '\\' not in name and ':' not in name, 'Canonical relative path')
    p = PurePosixPath(name)
    need(not p.is_absolute() and p.as_posix() == name and name != '.'
         and not any(x in ('', '.', '..') for x in p.parts), 'Relative path escape')
    need(all(piece.rstrip(' .') == piece and not re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', piece)
             for piece in p.parts), 'Cross-platform relative namespace required')
    return name


def directory(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        s = item.lstat()
        need(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode)
             and not getattr(s, 'st_file_attributes', 0) & 1024, 'Ordinary directory ancestry')
    return path.resolve(strict=True)


def regular(path):
    path = Path(path).absolute(); directory(path.parent); s = path.lstat()
    need(stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode) and s.st_nlink == 1
         and not getattr(s, 'st_file_attributes', 0) & 1024 and 0 <= s.st_size <= MAX_FILE,
         'Bounded single-link ordinary file')
    return path.resolve(strict=True)


def sha256(path, guard=lambda: None):
    path = regular(path); before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            guard(); h.update(block)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns)
         == (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during hashing')
    return h.hexdigest()


def pin(root, path, guard=lambda: None):
    root = directory(root); path = regular(path)
    need(path.is_relative_to(root), 'Pinned file escaped root')
    before = path.stat(); digest = sha256(path, guard); after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns)
         == (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during pinning')
    return {'path': path.relative_to(root).as_posix(), 'bytes': after.st_size, 'sha256': digest}


def read_json(path, maximum=2*1024**2):
    path = regular(path); need(path.stat().st_size <= maximum, 'Bounded JSON file')
    return strict_json(path.read_bytes())


def write_new(path, value):
    directory(Path(path).parent)
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def require_vm(root, parent):
    root, parent = directory(root), directory(parent); home = Path.home()/'forensic-dgp'
    need(platform.system() == 'Linux' and root == home/STAGE and parent == home/PARENT_STAGE,
         'Exact new manual Linux VM stage and immutable parent required')
    need(Path.cwd().resolve() in (root, parent) and bool(os.environ.get('TMUX'))
         and Path(sys.prefix).resolve() == home/'cctv_dgp_vm_bundle/.venv',
         'Exact retained venv, allowed stage cwd and tmux required')
    return {'new_stage': str(root), 'read_only_parent': str(parent), 'manual_tmux': True}


def _binding(root, binding, guard=lambda: None):
    need(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}, 'Exact file binding')
    relative(binding['path'])
    need(pin(root, root/binding['path'], guard) == binding, 'Pinned file differs: '+binding['path'])


def _load_contract(root):
    module = importlib.import_module('research.codeformer_u64_gate_aligned_pilot_v1.contract')
    need(Path(module.__file__).resolve() == root/PACKAGE/'contract.py', 'Use own sealed contract')
    return module


def own_namespace(root, files):
    static_dirs = {'outputs'}
    for name in files:
        static_dirs.update(p.as_posix() for p in PurePosixPath(name).parents if p.as_posix() != '.')
    result = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        directory(current)
        for name in dirs:
            path = directory(Path(current)/name); rel = path.relative_to(root).as_posix()
            need(rel in static_dirs or rel == OUTPUT or rel.startswith(OUTPUT+'/'), 'Unexpected own-stage directory: '+rel)
        for name in names:
            path = regular(Path(current)/name); rel = path.relative_to(root).as_posix()
            need(rel in files or rel == 'protocol.json' or rel in RUNTIME or rel.startswith(OUTPUT+'/'),
                 'Unexpected own-stage file: '+rel)
            result[rel] = path
            need(len(result) <= 6000, 'Bounded own-stage namespace')
    return result


def verify_static(root, parent, expected_sha, guard=lambda: None):
    root, parent = directory(root), directory(parent)
    need(root.name == STAGE and parent.name == PARENT_STAGE and root != parent,
         'Distinct new stage and exact parent names')
    need(Path(__file__).resolve() == root/PACKAGE/'runtime.py', 'Use own staged runtime')
    need(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha), 'Explicit protocol SHA256')
    protocol_file = regular(root/'protocol.json')
    need(protocol_file.stat().st_size <= 4*1024**2, 'Bounded protocol before read')
    raw = protocol_file.read_bytes()
    need(len(raw) <= 4*1024**2 and hashlib.sha256(raw).hexdigest() == expected_sha, 'Exact bounded protocol')
    p = strict_json(raw)
    need(p['format'] == FORMAT and p['stage_name'] == STAGE and p['parent_protocol'] == PARENT
         and p['manual_launch_only'] is True and p['automatic_retry_resume'] is False
         and p['model_qualified'] is False, 'Manual immutable pilot packet')
    need(type(p['files']) is dict and 1 <= len(p['files']) <= 100, 'Bounded own-stage static namespace')
    for name, value in p['files'].items():
        guard(); relative(name)
        need(name != 'protocol.json' and name not in RUNTIME and name != OUTPUT and not name.startswith(OUTPUT+'/')
             and type(value) is dict and set(value) == {'bytes', 'sha256'}
             and type(value['bytes']) is int and 0 <= value['bytes'] <= 16*1024**2
             and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Static/runtime namespace separation')
        need(pin(root, root/name, guard) == {'path': name, **value}, 'Own static file differs: '+name)
    needed = {PACKAGE+'/'+n for n in ('__init__.py', 'runtime.py', 'contract.py', 'arithmetic.py', 'quality.py', 'worker.py')}
    needed |= {'research/__init__.py', 'scripts/run_u64_gate_aligned_pilot_v1.py', 'scripts/run_gate_aligned.sh'}
    need(needed <= set(p['files']), 'Complete reviewed worker/bootstrap sources required')
    need(set(HELPER_PINS) <= set(p['files']), 'Exact reviewed helper packages required')
    for name, (size, digest) in HELPER_PINS.items():
        need(p['files'][name] == {'bytes': size, 'sha256': digest}, 'Reviewed helper changed: '+name)
    own_namespace(root, p['files'])
    for key, flag in (('source_review', 'source_review_pass'), ('source_tests', 'all_tests_pass')):
        _binding(root, p[key], guard); proof = read_json(root/p[key]['path'])
        need(proof['complete'] is True and proof[flag] is True, 'Complete source review/tests admission')
        if key == 'source_review':
            need(proof['unresolved_blockers'] == [], 'No unresolved source blockers')
        else:
            need(proof['neural_gradient_optimizer_training_VM_calls'] == 0, 'Pure source-test admission')
        need(type(proof['sources']) is list and len({v['path'] for v in proof['sources']}) == len(proof['sources']),
             'Unique source admission rows')
        sources = {v['path']: {'bytes': v['bytes'], 'sha256': v['sha256']} for v in proof['sources']}
        for name in p['files']:
            if name.startswith(('research/', 'scripts/')):
                source = PACKAGE+'/run_gate_aligned.sh' if name == 'scripts/run_gate_aligned.sh' else name
                need(sources.get(source) == p['files'][name], 'Admission/current source differs: '+source)
    parent_file = regular(parent/'protocol.json')
    need(parent_file.stat().st_size == PARENT['bytes'], 'Exact parent protocol size before read')
    parent_raw = parent_file.read_bytes()
    need(len(parent_raw) == PARENT['bytes'] and hashlib.sha256(parent_raw).hexdigest() == PARENT['sha256'], 'Exact immutable parent protocol')
    parent_p = strict_json(parent_raw)
    need(len(parent_p['files']) == 469, 'Exact469 parent static bindings including installed weights')
    for name, value in parent_p['files'].items():
        guard(); relative(name)
        need(pin(parent, parent/name, guard) == {'path': name, **value}, 'Consumed parent static differs: '+name)
    standard = parent/relative(parent_p['parent_protocol']['path']); _binding(parent, parent_p['parent_protocol'], guard)
    for key, original in (('parent_protocol_copy', parent_file), ('standard_protocol_copy', standard)):
        _binding(root, p[key], guard)
        need({k:p[key][k] for k in ('bytes','sha256')} ==
             {k:pin(parent, original, guard)[k] for k in ('bytes','sha256')}, 'Exact parent evidence copy required')
    _binding(root, p['gradient_protocol_copy'], guard)
    gradient = regular(root/p['gradient_protocol_copy']['path'])
    need({k:p['gradient_protocol_copy'][k] for k in ('bytes','sha256')} == GRADIENT_PROTOCOL,
         'Exact sealed original role-gradient protocol required')
    contract = _load_contract(root)
    selection = contract.select_cases(parent_raw, standard.read_bytes())
    need(p['selection'] == selection and p['plan'] == contract.operation_plan(selection)
         and p['pilot'] == contract.prospective_contract(parent_raw, standard.read_bytes(), gradient.read_bytes())
         and p['limits'] == contract.LIMITS,
         'Exact scientific selection/plan/contract/limits')
    return p


def operation_sequence(plan):
    """Expand the sealed544-row plan into1056 finite reserve/return operations."""
    sequence = []
    for row in plan:
        kind = row.get('kind')
        need(kind in ('forward', 'parameter_application'), 'Known sealed plan operation')
        sequence.append((kind, row, None))
        if kind == 'forward':
            sequence.extend(('gradient_query', row, query) for query in row['gradient_queries'])
    return sequence


def replay_journal(raw, plan):
    need(type(raw) is bytes and len(raw) <= 32*1024**2 and raw.endswith(b'\n'), 'Complete bounded journal required')
    sequence = operation_sequence(plan); counters = dict.fromkeys(COUNTERS, 0)
    pending = None; position = 0; token = 0; count = 0; elapsed = -1.
    for line in raw.splitlines():
        need(bool(line), 'No empty journal records')
        row = strict_json(line); need(type(row) is dict, 'Journal object required'); count += 1
        if 'elapsed_seconds' in row:
            need(type(row['elapsed_seconds']) in (int,float) and row['elapsed_seconds'] >= elapsed, 'Monotonic event time required')
            elapsed = row['elapsed_seconds']
        name = row.get('kind', '')
        reserved = name.endswith('_reserved') and name[:-9] in OPERATIONS
        returned = name.endswith('_returned') and name[:-9] in OPERATIONS
        need(not (('token' in row) and (name.endswith('_reserved') or name.endswith('_returned')))
             or reserved or returned, 'Unknown journal operation refused')
        if reserved:
            need(pending is None and position < len(sequence), 'Known nonoverlapping bounded operation')
            kind, call, query = sequence[position]; token += 1
            need(name == kind+'_reserved' and row.get('token') == token, 'Exact operation/token order')
            attempted, _ = OPERATIONS[kind]; counters[attempted] += 1
            if kind == 'forward':
                need(row.get('category') == call['category'] and call['category'] in CATEGORIES, 'Forward category differs')
                counters[call['category']] += 1
            pending = {'token':token,'kind':kind,'id':call['id'],'plan_slot':call['slot'],'new_update':call['new_update']}
            if query is not None: pending['query'] = query
        elif returned:
            need(pending is not None and name == pending['kind']+'_returned', 'Return without exact reservation')
            kind, call, query = sequence[position]
            _, finished = OPERATIONS[kind]; counters[finished] += 1
            need(row.get('synchronized_return') is True, 'Synchronized operation return required')
        if reserved or returned:
            need(all(row.get(k) == v for k,v in pending.items() if k != 'kind'), 'Journal operation identity differs')
            if kind == 'forward': need(row.get('category') == call['category'], 'Returned forward category differs')
            need(row.get('counters') == counters and all(type(v) is int and 0 <= v <= COUNTERS[k] for k,v in counters.items()),
                 'Exact finite journal counter replay')
            if returned: pending = None; position += 1
        elif 'counters' in row:
            need(row['counters'] == counters, 'Nonoperation counters differ from replay')
    return {'counters':counters,'unknown_reservations':[] if pending is None else [pending],
            'operation_outcome_unknown':pending is not None,'journal_events_replayed':count,
            'completed_operation_pairs':position,'maximum_operation_pairs':len(sequence),
            'full_exact_plan_completed':position == len(sequence) and pending is None,
            'valid_sealed_plan_prefix':True,'last_token':token}


def status_snapshot(root, expected_sha, p=None):
    result = {'status_snapshot_present':False,'counters':None,'operation_outcome_unknown':True,
              'journal_tail_reconciled':False,'counter_scope':'full durable journal replay against sealed operation plan'}
    out = root/OUTPUT
    if not (out/'status.json').exists(): return result
    try:
        value = read_json(out/'status.json')
        need(value['protocol_sha256'] == expected_sha and type(value['counters']) is dict
             and set(value['counters']) == set(COUNTERS), 'Valid bound status counters')
        need(all(type(v) is int and 0 <= v <= COUNTERS[k] for k,v in value['counters'].items()), 'Finite actual counters')
        need(type(value['unknown_reservations']) is list and type(value['operation_outcome_unknown']) is bool, 'Explicit pending outcome')
        if p is None: p = read_json(root/'protocol.json',4*1024**2)
        log = regular(out/'events.jsonl'); before = log.stat(); need(before.st_size <= 32*1024**2,'Bounded journal')
        replay = replay_journal(log.read_bytes(),p['plan']); after = log.stat()
        stable = (before.st_ino,before.st_size,before.st_mtime_ns) == (after.st_ino,after.st_size,after.st_mtime_ns)
        known = stable and replay['counters'] == value['counters']
        pending = bool(list(out.glob('status*.pending*')))
        return {**result,**replay,'status_snapshot_present':True,'declared_status_counters':value['counters'],
                'journal_tail_reconciled':known,'operation_outcome_unknown':bool(not known or pending or
                    value['operation_outcome_unknown'] or value['unknown_reservations'] or replay['operation_outcome_unknown'])}
    except TimeoutError: raise
    except Exception as error:
        return {**result,'status_read_error':type(error).__name__+': '+str(error)[:1000]}


def _group_alive(pid):
    try:
        os.killpg(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _stop_group(child):
    start = time.monotonic(); actions = []
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 20)):
        if _group_alive(child.pid):
            try:
                os.killpg(child.pid, sig); actions.append(sig.name)
            except ProcessLookupError:
                pass
        try:
            child.wait(timeout=max(.01, min(wait, 20-(time.monotonic()-start))))
        except subprocess.TimeoutExpired:
            pass
    return {'child_reaped': child.poll() is not None, 'process_group_absent': not _group_alive(child.pid),
            'actions': actions, 'seconds': time.monotonic()-start}


def external_usage(root, child_pid, phase):
    free = shutil.disk_usage(root).free; output = 0; files = 0
    base = root/OUTPUT
    if base.exists():
        for current, dirs, names in os.walk(base,followlinks=False):
            directory(current)
            for name in dirs: directory(Path(current)/name)
            for name in names:
                try: item = regular(Path(current)/name); size = item.stat().st_size
                except FileNotFoundError: continue  # Atomic worker rename raced this read-only sample.
                files += 1; need(files <= 6000,'External output namespace cap'); output += size
    try:
        words = (Path('/proc')/str(child_pid)/'statm').read_text().split()
        need(len(words) >= 2 and words[1].isdigit(),'Linux child resident pages required')
        rss = int(words[1])*int(os.sysconf('SC_PAGE_SIZE'))
    except FileNotFoundError:
        rss = None  # A reaped/finishing child has no proc entry; never invent a zero sample.
    archive = Path.home()/(PREFIX+'-results.tar.gz')
    archive_bytes = regular(archive).stat().st_size if phase == 'export' and archive.exists() else 0
    return {'free_disk_bytes':free,'output_bytes':output,'current_RSS_bytes':rss,'archive_bytes':archive_bytes}


def _child(root, command, phase, seconds, environment, limits=None):
    limits = limits or {'maximum_RSS_bytes':10*1024**3,'maximum_output_bytes':2*1024**3,
                        'maximum_archive_bytes':2*1024**3,'continuing_free_disk_bytes':4*1024**3}
    child = None; cleanup = None; code = 1; error = None; timeout = False; samples = 0
    maxima = dict.fromkeys(('output_bytes','current_RSS_bytes','archive_bytes'),0); minimum_free = None
    began = time.monotonic()
    with (root/('run.log' if phase == 'worker' else 'export.log')).open('xb') as log, \
         (root/('supervisor.'+phase+'.resources.jsonl')).open('xb') as watch:
        try:
            child = subprocess.Popen(command,cwd=str(root),env=environment,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            write_new(root/('supervisor.'+phase+'.started.json'),{'child_pid':child.pid,
                      'process_group':child.pid,'phase':phase,'external_seconds':seconds})
            print(json.dumps({'phase':phase,'child_pid':child.pid,'log':log.name}),flush=True)
            while child.poll() is None:
                elapsed = time.monotonic()-began
                if elapsed >= seconds: raise subprocess.TimeoutExpired(command,seconds)
                usage = external_usage(root,child.pid,phase); samples += 1
                row = {'elapsed_seconds':elapsed,'child_pid':child.pid,'phase':phase,**usage}
                watch.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode());watch.flush()
                for key in maxima:
                    if usage[key] is not None: maxima[key] = max(maxima[key],usage[key])
                minimum_free = usage['free_disk_bytes'] if minimum_free is None else min(minimum_free,usage['free_disk_bytes'])
                need(usage['free_disk_bytes'] >= limits['continuing_free_disk_bytes'],'External continuing4GiB stop')
                need(usage['current_RSS_bytes'] is None or usage['current_RSS_bytes'] <= limits['maximum_RSS_bytes'],'External RSS10GiB stop')
                need(usage['output_bytes'] <= limits['maximum_output_bytes'],'External output2GiB stop')
                need(usage['archive_bytes'] <= limits['maximum_archive_bytes'],'External archive2GiB stop')
                try:
                    child.wait(timeout=max(.01,min(1.,seconds-(time.monotonic()-began))))
                except subprocess.TimeoutExpired: pass
            code = child.wait(timeout=.01)
        except subprocess.TimeoutExpired:
            code = 124; timeout = True
        except (KeyboardInterrupt,SystemExit): code = 130
        except Exception as exc:
            code = 125; error = type(exc).__name__+': '+str(exc)[:2000]
        finally:
            if child is not None:
                try: cleanup = _stop_group(child)
                except BaseException as exc:
                    cleanup = {'child_reaped':False,'process_group_absent':False,'error':repr(exc)[:1000]}
    return {'complete':True,'phase':phase,'exit_code':code,'timeout':timeout,'spawn_returned':child is not None,
            'child_pid':None if child is None else child.pid,'cleanup':cleanup,'error':error,
            'external_seconds':seconds,'elapsed_seconds':time.monotonic()-began,
            'resource_samples':samples,'resource_maxima':maxima,'minimum_free_disk_bytes':minimum_free,
            'model_qualified':False}


def supervise(root, parent, p, sha):
    require_vm(root, parent)
    need(not any((root/n).exists() for n in RUNTIME) and not (root/OUTPUT).exists(), 'Fresh stage only; never retry/resume')
    need(shutil.disk_usage(root).free >= p['limits']['initial_free_disk_bytes'],
         'Initial8GiB free disk required before worker reservation/launch')
    command = [sys.executable, '-B', '-u', str(root/'scripts/run_u64_gate_aligned_pilot_v1.py'),
               '--root', str(root), '--protocol-sha', sha]
    write_new(root/'supervisor.start.json', {'protocol_sha256': sha, 'worker_spawn_reserved': True,
                                          'read_only_parent': str(parent), 'model_qualified': False})
    env = dict(os.environ, **{ENV+'SUPERVISED': '1', ENV+'PROTOCOL': sha,
                            ENV+'SUPERVISOR_PID': str(os.getpid()), 'PYTHONDONTWRITEBYTECODE': '1'})
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Stop owned child group; retain evidence')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        worker = _child(root, command+['--run'], 'worker', p['limits']['external_seconds'], env, p['limits'])
        write_new(root/'supervisor.worker.json', worker)
        cleanup = worker.get('cleanup') or {}
        unknown = not (worker['spawn_returned'] and cleanup.get('child_reaped') is True and cleanup.get('process_group_absent') is True)
        exported = ({'complete': False, 'exit_code': None, 'skipped': True,
                     'active_writer_outcome_unknown': True} if unknown else
                    _child(root, command+['--export'], 'export', p['limits']['export_external_seconds'], dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), p['limits']))
        write_new(root/'supervisor.export.json', exported)
        result = {'complete': True, 'protocol_sha256': sha, 'worker_exit': worker['exit_code'],
                  'export_exit': exported['exit_code'], 'export_attempted': not unknown,
                  'active_writer_outcome_unknown': unknown, **status_snapshot(root, sha, p), 'model_qualified': False}
        write_new(root/'supervisor.json', result); print(json.dumps(result), flush=True)
        return worker['exit_code'] or exported['exit_code'] or (1 if unknown else 0)
    except BaseException as error:
        if not (root/'supervisor_failure.json').exists():
            write_new(root/'supervisor_failure.json', {'complete': False, 'protocol_sha256': sha,
                      'error': repr(error)[:2000], **status_snapshot(root, sha, p), 'model_qualified': False})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def _export(root, parent, sha):
    require_vm(root, parent); began = time.monotonic()
    def guard():
        need(time.monotonic()-began < 300 and shutil.disk_usage(root).free >= 4*1024**3, 'Finite300s export/4GiB reserve')
    p = verify_static(root, parent, sha, guard)
    if (root/OUTPUT).exists() or (root/'run.log').exists() or (root/'supervisor.worker.started.json').exists():
        worker = read_json(root/'supervisor.worker.json'); cleanup = worker.get('cleanup') or {}
        need(worker.get('spawn_returned') is True and cleanup.get('child_reaped') is True
             and cleanup.get('process_group_absent') is True and type(worker.get('child_pid')) is int
             and worker['child_pid'] > 0 and not _group_alive(worker['child_pid']), 'Known reaped worker/group absence required before export')
    files = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        guard(); directory(current)
        for name in dirs:
            directory(Path(current)/name)
        for name in names:
            path = regular(Path(current)/name); rel = path.relative_to(root).as_posix()
            if rel in POST_EXPORT:
                continue
            need(rel in p['files'] or rel == 'protocol.json' or rel in RUNTIME or rel.startswith(OUTPUT+'/'), 'Unexpected own-stage file: '+rel)
            files[rel] = pin(root, path, guard)
            need(len(files) <= 6000, 'Bounded exported namespace')
    need(sum(v['bytes'] for k, v in files.items() if k.startswith(OUTPUT+'/')) <= 2*1024**3, '2GiB own output cap')
    home = Path.home(); archive = home/(PREFIX+'-results.tar.gz'); checksum = home/(PREFIX+'-results.tar.gz.sha256')
    receipt = home/(PREFIX+'-export.json')
    need(not any(path.exists() for path in (archive, checksum, receipt, root/'export.json', root/'export_failure.json')), 'Export complete/partial bytes immutable')
    class Sink:
        def __init__(self, file):
            self.file = file; self.bytes = 0; self.digest = hashlib.sha256()
        def write(self, data):
            guard(); need(self.bytes+len(data) <= 2*1024**3, '2GiB return archive cap')
            count = self.file.write(data); need(count == len(data), 'Complete archive write')
            self.bytes += count; self.digest.update(data); return count
        def __getattr__(self, name):
            return getattr(self.file, name)
    class Reader:
        def __init__(self, file):
            self.file = file; self.digest = hashlib.sha256()
        def read(self, size):
            guard(); value = self.file.read(size); self.digest.update(value); return value
    with archive.open('xb') as f:
        sink = Sink(f)
        with gzip.GzipFile(fileobj=sink, mode='wb', filename='', mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                for rel, binding in sorted(files.items()):
                    guard(); path = regular(root/rel); before = path.stat()
                    info = tarfile.TarInfo(STAGE+'/'+rel); info.size = before.st_size; info.mode = 0o644
                    with path.open('rb') as source:
                        reader = Reader(source); tar.addfile(info, reader)
                        need(reader.digest.hexdigest() == binding['sha256'], 'Archived source content changed')
                    after = path.stat()
                    need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'Archived source stat changed')
        f.flush(); os.fsync(f.fileno())
    need(sha256(archive, guard) == sink.digest.hexdigest(), 'Compressed archive fullhash readback')
    with checksum.open('xb') as f:
        f.write((sink.digest.hexdigest()+'  '+archive.name+'\n').encode()); f.flush(); os.fsync(f.fileno())
    consumed_raw = regular(parent/'protocol.json').read_bytes()
    need(len(consumed_raw) == PARENT['bytes'] and hashlib.sha256(consumed_raw).hexdigest() == PARENT['sha256'], 'Parent protocol stable through export')
    result = {'complete': True, 'protocol_sha256': sha, 'archive': {'path': str(archive), 'bytes': sink.bytes, 'sha256': sink.digest.hexdigest()},
              'files': files, 'parent_protocol': PARENT, 'parent_consumed_static_bindings': 469,
              'parent_consumed_bindings': strict_json(consumed_raw)['files'],
              'parent_payload_included': False, 'post_export_receipts_not_in_archive': sorted(POST_EXPORT),
              **status_snapshot(root, sha, p), 'model_qualified': False}
    write_new(root/'export.json', result); write_new(receipt, result); print(json.dumps(result), flush=True)
    return result


def export(root, parent, sha):
    require_vm(root, parent)
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'Do not replace active export timer')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('Absolute300s export stop includes all verification/inventory/write/readback')
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, 300)
    try:
        return _export(root, parent, sha)
    except BaseException as error:
        # Cancel the exhausted work timer before a separate bounded failure receipt.
        signal.setitimer(signal.ITIMER_REAL, 5)
        try:
            if not (root/'export_failure.json').exists():
                write_new(root/'export_failure.json', {'complete': False, 'error': repr(error)[:2000],
                          'protocol_sha256': sha, 'partial_files_retained': True,
                          'failure_receipt_scope': 'bounded5s metadata only; status may remain unavailable',
                          'model_qualified': False})
        except BaseException:
            pass  # Preserve partial files and the original error if receipt I/O also fails.
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--protocol-sha', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('verify-transfer', 'status', 'supervise', 'run', 'export'):
        group.add_argument('--'+name, action='store_true')
    args = parser.parse_args(); root = directory(args.root); parent = root.parent/PARENT_STAGE
    if args.export:
        export(root, parent, args.protocol_sha); return 0
    p = verify_static(root, parent, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps({'complete': True, 'own_static_bindings': len(p['files']), 'parent_static_bindings': 469,
                          'maximum_forwards': 540, 'maximum_gradient_queries': 512, 'maximum_parameter_applications': 4,
                          'neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False})); return 0
    if args.status:
        print(json.dumps(status_snapshot(root, args.protocol_sha, p))); return 0
    require_vm(root, parent)
    if args.run:
        need(os.environ.get(ENV+'SUPERVISED') == '1' and os.environ.get(ENV+'PROTOCOL') == args.protocol_sha
             and os.environ.get(ENV+'SUPERVISOR_PID') == str(os.getppid()), 'Actual finite supervisor parent required')
        worker = importlib.import_module('research.codeformer_u64_gate_aligned_pilot_v1.worker')
        need(Path(worker.__file__).resolve() == root/PACKAGE/'worker.py', 'Use own sealed worker')
        os.chdir(parent)  # Parent is a read-only import/runtime context; all writes use new root.
        return worker.run(root, parent, p, args.protocol_sha)
    return supervise(root, parent, p, args.protocol_sha)


if __name__ == '__main__':
    raise SystemExit(main())
