"""Independent stdlib-only returned transport audit; never loads models or arrays.

Audit archive framing and file pins before interpreting saved JSON evidence.
No extraction, VM contact, worker execution, neural import or gradient operation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import sys
import tarfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u64_role_gradients_v1_vm'
PARENT = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PREFIX = 'cctv-codeformer-u64-role-gradients-v1'
PROTOCOL_SHA = '07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3'
ARCHIVE_SHA = 'c052f2b8c09e42d6ee82c263e5a9b1777b3eeb114c6e111a82ba46b4b1862be7'
ARCHIVE_BYTES = 1226001016
OUT = ROOT / 'outputs/codeformer_u64_role_gradients_v1_return_transport'
OPREFIX = 'outputs/u64_role_gradients_v1/'
COUNTER_KEYS = ('full_forward_attempts', 'full_forward_returns', 'gradient_query_attempts',
                'gradient_query_returns', 'parity_forwards', 'repeat_forwards', 'diagnostic_forwards')
FINAL_COUNTERS = dict(zip(COUNTER_KEYS, (72, 72, 192, 192, 4, 4, 64)))
PHASE_CATEGORY = {'historical_u64_parity': 'parity_forwards',
                  'deterministic_same_u64_repeat': 'repeat_forwards',
                  'paired_TRAIN_gradient_diagnostic': 'diagnostic_forwards'}
QUERIES = ['objective_div4', 'gate_mae', 'gate_adjacent']


def need(ok, why):
    if not ok:
        raise ValueError(why)


def strict_json(raw):
    def pairs(rows):
        d = {}
        for k, v in rows:
            need(k not in d, 'Duplicate JSON key: '+k)
            d[k] = v
        return d
    def bad(value):
        raise ValueError('Nonfinite JSON scalar: '+value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    def finite(x):
        if type(x) is float:
            need(math.isfinite(x), 'Nonfinite numeric value')
        elif type(x) is dict:
            for v in x.values(): finite(v)
        elif type(x) is list:
            for v in x: finite(v)
    finite(value)
    return value


def confined(name):
    need(type(name) is str and name.isascii() and bool(name), 'Nonempty ASCII namespace required')
    need(not any(c in name for c in ('\\', ':', '\x00', '\r', '\n')), 'Unsafe namespace character')
    pieces = name.split('/')
    need(all(p not in ('', '.', '..') and p.rstrip(' .') == p for p in pieces), 'Noncanonical namespace')
    need(all(not re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', p) for p in pieces),
         'Reserved Windows namespace')
    return name


def pin(path):
    path = Path(path).absolute()
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not path.is_symlink() and
         not getattr(before, 'st_file_attributes', 0) & 1024, 'Ordinary file required: '+str(path))
    need(path.resolve().is_relative_to(ROOT.resolve()), 'Local audit file outside workspace')
    for ancestor in path.absolute().parents:
        mode = ancestor.lstat()
        need(stat.S_ISDIR(mode.st_mode) and not ancestor.is_symlink() and
             not getattr(mode,'st_file_attributes',0) & 1024, 'Nonordinary local ancestry')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed while hashing')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': after.st_size, 'sha256': h.hexdigest()}


class SingleGzipReader:
    """Bounded streaming inflater: CRC/ISIZE validated; concatenation/trailing bytes rejected."""
    def __init__(self, stream):
        self.stream = stream
        self.decoder = zlib.decompressobj(31)
        self.pending = b''
        self.buffer = bytearray()
        self.done = False
        self.compressed_digest = hashlib.sha256()
        self.compressed_bytes = 0
        self.uncompressed_bytes = 0

    def raw(self, size):
        data = self.stream.read(size)
        self.compressed_digest.update(data)
        self.compressed_bytes += len(data)
        return data

    def read(self, size):
        need(type(size) is int and 0 < size <= 1024**2, 'Bounded positive read')
        while len(self.buffer) < size and not self.done:
            data = self.pending or self.raw(256*1024)
            need(bool(data), 'Truncated gzip member')
            out = self.decoder.decompress(data, 1024**2)
            self.pending = self.decoder.unconsumed_tail
            self.buffer.extend(out)
            if self.decoder.eof:
                need(not self.decoder.unused_data and not self.pending and not self.raw(1),
                     'Trailing compressed bytes or concatenated gzip member')
                self.done = True
        out = bytes(self.buffer[:size]); del self.buffer[:size]
        self.uncompressed_bytes += len(out)
        return out


def exact(stream, size):
    parts = bytearray()
    while len(parts) < size:
        data = stream.read(min(1024**2, size-len(parts)))
        need(bool(data), 'Truncated tar member/header')
        parts.extend(data)
    return bytes(parts)


def stream_archive(stream, expected=None, root_name=STAGE):
    gzip = SingleGzipReader(stream)
    files, metadata, folded = {}, {}, set()
    header_count = 0
    while True:
        block = exact(gzip, 512)
        if block == bytes(512):
            trailer = 512
            while True:
                chunk = gzip.read(1024**2)
                if not chunk: break
                need(not any(chunk), 'Nonzero content after tar end marker')
                trailer += len(chunk)
            need(trailer >= 1024 and trailer % 512 == 0, 'Two complete tar EOF blocks required')
            need(gzip.uncompressed_bytes % 10240 == 0, 'Full tar record framing required')
            break
        need(block[257:265] == b'ustar\x0000', 'Exact USTAR framing required')
        member = tarfile.TarInfo.frombuf(block, 'utf-8', 'strict')
        need(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.linkname,
             'Only nonlink regular tar members allowed')
        full = confined(member.name)
        need(full.startswith(root_name+'/'), 'Member outside exact stage root')
        rel = confined(full[len(root_name)+1:])
        need(rel not in files and rel.casefold() not in folded, 'Duplicate/colliding tar namespace')
        need(0 <= member.size <= 2*1024**3, 'Tar payload size cap')
        if expected is not None:
            need(rel in expected, 'Unexported tar member: '+rel)
            need(member.size == expected[rel]['bytes'], 'Export/member size mismatch: '+rel)
        h = hashlib.sha256(); remain = member.size
        capture = rel.endswith(('.json', '.jsonl')) and member.size <= 2*1024**2
        raw = bytearray()
        while remain:
            data = exact(gzip, min(remain, 1024**2)); h.update(data); remain -= len(data)
            if capture: raw.extend(data)
        padding = (-member.size) % 512
        if padding: need(not any(exact(gzip, padding)), 'Nonzero member padding')
        binding = {'path': rel, 'bytes': member.size, 'sha256': h.hexdigest()}
        if expected is not None: need(binding == expected[rel], 'Export/member content mismatch: '+rel)
        files[rel] = binding; folded.add(rel.casefold()); header_count += 1
        if capture: metadata[rel] = bytes(raw)
    if expected is not None: need(set(files) == set(expected), 'Missing exported member')
    return {'files': files, 'metadata': metadata, 'framing': {
        'compressed_bytes': gzip.compressed_bytes, 'compressed_sha256': gzip.compressed_digest.hexdigest(),
        'uncompressed_bytes': gzip.uncompressed_bytes, 'payload_bytes': sum(x['bytes'] for x in files.values()),
        'regular_members': header_count, 'trailer_zero_bytes': trailer, 'single_gzip_CRC_ISIZE_pass': True,
        'USTAR_header_checksum_and_full_record_pass': True, 'unique_confined_regular_namespace_pass': True}}


def replay_events(rows, plan):
    counters = dict.fromkeys(COUNTER_KEYS, 0)
    operations = []
    for call in plan:
        operations.append(('forward', call, None))
        operations.extend(('gradient_query', call, q) for q in call['gradient_queries'])
    position, token, pending, previous_time = 0, 0, None, -1.0
    counts = Counter()
    for row in rows:
        need(type(row) is dict and type(row.get('elapsed_seconds')) in (int,float), 'Timed event required')
        need(row['elapsed_seconds'] >= previous_time, 'Event time reversed')
        previous_time = row['elapsed_seconds']; kind = row.get('kind'); counts[kind] += 1
        if kind in ('forward_reserved', 'gradient_query_reserved'):
            need(pending is None and position < len(operations), 'Nonoverlapping bounded reservation required')
            op, call, query = operations[position]
            need(kind == op+'_reserved', 'Operation order differs from sealed plan')
            token += 1
            need(row.get('token') == token and row.get('plan_slot') == call['slot'] and
                 row.get('id') == call['id'] and row.get('phase') == 'call_'+str(call['slot']).zfill(4),
                 'Reservation identity/order mismatch')
            if query is not None: need(row.get('query') == query, 'Query order mismatch')
            if op == 'forward':
                need(row.get('category') == PHASE_CATEGORY[call['phase']], 'Forward category mismatch')
                counters[row['category']] += 1
            counters[op+'_attempts' if op != 'forward' else 'full_forward_attempts'] += 1
            need(row.get('counters') == counters, 'Reserved counter replay mismatch')
            pending = (op, call, query, token)
        elif kind in ('forward_returned', 'gradient_query_returned'):
            need(pending is not None, 'Return without reservation')
            op, call, query, current = pending
            need(kind == op+'_returned' and row.get('token') == current and row.get('plan_slot') == call['slot']
                 and row.get('id') == call['id'] and row.get('phase') == 'call_'+str(call['slot']).zfill(4)
                 and row.get('synchronized_return') is True, 'Return identity/order mismatch')
            if query is not None: need(row.get('query') == query, 'Returned query identity mismatch')
            else: need(row.get('category') == PHASE_CATEGORY[call['phase']], 'Returned category mismatch')
            counters[op+'_returns' if op != 'forward' else 'full_forward_returns'] += 1
            need(row.get('counters') == counters, 'Returned counter replay mismatch')
            position += 1; pending = None
        elif 'counters' in row:
            need(row['counters'] == counters, 'Nonoperation counter replay mismatch')
    need(pending is None and position == len(operations), 'Incomplete/unknown operation outcome')
    need(counters == FINAL_COUNTERS, 'Final counters differ from exact finite plan')
    return {'all_events_replayed': len(rows), 'event_kinds': dict(counts), 'operation_pairs': position,
            'last_token': token, 'counters': counters, 'unknown_reservations': [],
            'exact_full_order_and_per_call_three_queries_after_forward_pass': True,
            'last_event_elapsed_seconds': previous_time}


def local_parent_audit(parent_protocol, consumed, readback):
    need(parent_protocol['files'] == consumed and len(consumed) == 469, 'Consumed parent bindings differ from sealed parent')
    known = {r['parent_relative']: r['actual_local'] for r in readback['files']}
    need(set(known) == set(consumed), 'Local parent provenance namespace differs')
    records, aliases = [], []
    for name, expected in consumed.items():
        confined(name)
        confined(known[name]['path'])
        primary = ROOT/'outputs'/PARENT/name
        source = primary if primary.is_file() else ROOT/known[name]['path']
        actual = pin(source)
        need({'bytes': actual['bytes'], 'sha256': actual['sha256']} == expected, 'Local consumed parent bytes differ: '+name)
        if source != primary:
            need(name in ('weights/codeformer_r2.pth', 'weights/codeformer_u96_adapter.pth'), 'Unexpected parent alias')
            aliases.append({'parent_relative': name, 'actual_local': actual})
        records.append({'parent_relative': name, 'actual_local': actual})
    need(len(aliases) == 2, 'Expected two model-only local aliases')
    return {'complete': True, 'static_bindings_rehashed': len(records), 'direct_parent_files': len(records)-len(aliases),
            'model_only_local_aliases': aliases, 'files': records,
            'VM_files_currently_read': False, 'scope': 'returned consumed pins match sealed parent and exact local bytes'}


def audit_runtime_evidence(docs, events, resources, protocol, files, export):
    """Cross-check counters, phase sequence, saved supervisor and resource evidence."""
    limits = protocol['limits']
    initial_phases = ['idle_GPU_before_framework', 'dependencies_and_FP32_policy',
                      'exact_trained_u64_initialization_no_forward', 'initial_full575_value_witness']
    phase_rows = [r for r in events if r['kind'] == 'phase_returned']
    phase_names = initial_phases+['call_'+str(c['slot']).zfill(4) for c in protocol['plan']]+[
        'closing_full575_value_witness_and_integrity']
    need([r['phase'] for r in phase_rows] == phase_names, 'Exact phase-return sequence differs')
    need(all(0 <= r['phase_seconds'] < limits['per_phase_seconds'] for r in phase_rows), 'Phase cap exceeded')
    allowed = {'static_VM_supervisor_guards_passed_before_Torch', 'phase_returned', 'run_result_saved',
               'forward_reserved', 'forward_returned', 'gradient_query_reserved', 'gradient_query_returned',
               'gradient_warning_scope'}
    need(all(r['kind'] in allowed for r in events), 'Unexpected event kind')
    need(events[0]['kind'] == 'static_VM_supervisor_guards_passed_before_Torch' and
         events[0]['Torch_imported'] is False and events[-2]['kind'] == 'run_result_saved', 'Missing boundary events')
    warnings = [r for r in events if r['kind'] == 'gradient_warning_scope']
    need(len(warnings) == 192 and all(r['unexpected'] == [] and r['strict_flags_restored'] is True and
         r['bitwise_CUDA_gradient_repeatability_claimed'] is False for r in warnings), 'Gradient warning scope differs')
    for row in warnings:
        for item in row['warnings']:
            need(re.match(r'^upsample_bilinear2d_backward_out_cuda\b',item['message']) is not None,
                 'Unexpected retained warning')
    expected_labels = ['before','after']*4
    for call in protocol['plan']:
        expected_labels += ['before','before_forward']
        for _ in call['gradient_queries']: expected_labels += ['before_gradient_query','after_gradient_query']
        expected_labels += ['after_forward_and_queries','after']
    expected_labels += ['before','after']
    need([r['label'] for r in resources] == expected_labels, 'Resource reservation/return sample sequence differs')
    previous = -1
    for row in resources:
        need(row['elapsed_seconds'] >= previous and row['elapsed_seconds'] < limits['worker_seconds'], 'Resource time/cap differs')
        previous = row['elapsed_seconds']
        for field in ('current_RSS_bytes','peak_RSS_bytes','CUDA_reserved_bytes','CUDA_peak_reserved_bytes','free_disk_bytes','output_bytes'):
            need(type(row[field]) is int and row[field] >= 0, 'Nonnegative integer resource sample required')
        need(row['free_disk_bytes'] >= limits['continuing_free_disk_bytes'], 'Continuing reserve exceeded')
        need(max(row['current_RSS_bytes'],row['peak_RSS_bytes']) <= limits['maximum_RSS_bytes'], 'RSS bound exceeded')
        need(max(row['CUDA_reserved_bytes'],row['CUDA_peak_reserved_bytes']) <= limits['maximum_CUDA_reserved_bytes'], 'CUDA bound exceeded')
        need(row['output_bytes'] <= limits['maximum_output_bytes'], 'Output sample bound exceeded')
    need(resources[0]['free_disk_bytes'] >= limits['initial_free_disk_bytes'], 'First resource sample below admission reserve')
    status = docs[OPREFIX+'status.json']; result = docs[OPREFIX+'result.json']
    for value in (status,result,export):
        need(value['complete'] is True and value['counters'] == FINAL_COUNTERS and value['protocol_sha256'] == PROTOCOL_SHA
             and value['operation_outcome_unknown'] is False and value['unknown_reservations'] == []
             and value['model_qualified'] is False, 'Replayed counters/boundary receipt differs')
    need(status['elapsed_seconds'] < limits['worker_seconds'] and status['active_slot'] == 72 and
         status['active_case'] == protocol['plan'][-1]['id'] and status['active_query'] is None, 'Final status identity/cap differs')
    need(result['optimizer_constructions'] == result['optimizer_steps'] == result['parameter_applications'] == 0,
         'Returned zero optimizer/parameter-application declarations differ')
    start = docs['supervisor.start.json']; spawned = docs['supervisor.worker.started.json']; worker = docs['supervisor.worker.json']
    need(start['protocol_sha256'] == PROTOCOL_SHA and start['worker_spawn_reserved'] is True and start['read_only_parent'].endswith('/'+PARENT),
         'Supervisor reservation identity differs')
    need(spawned['child_pid'] == spawned['process_group'] == worker['child_pid'] and type(worker['child_pid']) is int
         and worker['child_pid'] > 0 and spawned['phase'] == worker['phase'] == 'worker'
         and spawned['external_seconds'] == limits['external_seconds'], 'Worker PID/group/external cap differs')
    cleanup = worker['cleanup']
    need(worker['complete'] is True and worker['exit_code'] == 0 and worker['timeout'] is False and
         worker['spawn_returned'] is True and worker['error'] is None and cleanup['child_reaped'] is True
         and cleanup['process_group_absent'] is True and cleanup['actions'] == []
         and 0 <= cleanup['seconds'] <= limits['termination_grace_seconds'], 'Worker exit/cleanup receipt differs')
    need(files['run.log']['bytes'] == 0, 'Worker traceback/log output present')
    runtime_bytes = sum(v['bytes'] for k,v in files.items() if k.startswith(OPREFIX))
    need(runtime_bytes <= limits['maximum_output_bytes'], 'Actual archived output exceeds 2GiB cap')
    missing = [name for name in export['post_export_receipts_not_in_archive'] if name not in files]
    return {'complete':True,'resource_samples':len(resources),'resource_sample_labels_verified':True,
        'phase_returns_verified':len(phase_rows),'maximum_phase_seconds':max(r['phase_seconds'] for r in phase_rows),
        'worker_elapsed_seconds':status['elapsed_seconds'],'last_journal_elapsed_seconds':events[-1]['elapsed_seconds'],
        'resource_maxima':{k:max(r[k] for r in resources) for k in ('current_RSS_bytes','peak_RSS_bytes','CUDA_reserved_bytes','CUDA_peak_reserved_bytes','output_bytes')},
        'minimum_free_disk_bytes':min(r['free_disk_bytes'] for r in resources),'actual_archived_output_bytes':runtime_bytes,
        'worker_supervisor_receipt':worker,'gradient_warning_scopes_verified':len(warnings),
        'zero_optimizer_source_and_receipt_scope':'authenticated reviewed worker has autograd.grad only, returned zero declarations; saved-state value audit is separate',
        'post_export_receipts_absent_by_design':missing,'final_export_exit_code_independently_available':False,
        'export_completed_payload_evidence':True,
        'limitations':['Final supervisor/export phase receipts are excluded by the exporter and unavailable in the three-file return.',
                       'Resource samples and supervisor cleanup are authenticated retained records, not a current VM/process query.',
                       'Transport and operation replay do not establish model usefulness or finite-training quality.']}


def audit_worker_ast(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    def dotted(node):
        if isinstance(node,ast.Name): return node.id
        if isinstance(node,ast.Attribute): return dotted(node.value)+'.'+node.attr
        return ''
    calls = [dotted(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    forbidden = [c for c in calls if c.endswith(('.backward','.step','.copy_','.add_','.sub_','.mul_','.div_')) or '.optim.' in c]
    need(not forbidden and calls.count('torch.autograd.grad') == 1, 'Worker AST includes optimizer/update or differing query callsites')
    return {'stdlib_AST_only':True,'autograd_grad_callsites':1,'optimizer_step_backward_parameter_inplace_callsites':forbidden,
            'worker_source':pin(path),'no_runtime_source_imported':True}


def audit():
    OUT.mkdir(exist_ok=True)
    archive = ROOT/'outputs'/(PREFIX+'-results.tar.gz')
    export_path = ROOT/'outputs'/(PREFIX+'-export.json')
    sidecar = ROOT/'outputs'/(PREFIX+'-results.tar.gz.sha256')
    protocol_path = ROOT/'outputs'/STAGE/'protocol.json'
    export = strict_json(export_path.read_bytes()); protocol_raw = protocol_path.read_bytes()
    need(len(protocol_raw) == 184633 and hashlib.sha256(protocol_raw).hexdigest() == PROTOCOL_SHA, 'Independent protocol seal differs')
    protocol = strict_json(protocol_raw)
    need(sidecar.read_bytes() == (ARCHIVE_SHA+'  '+archive.name+'\n').encode('ascii'), 'Exact LF sidecar differs')
    need(archive.stat().st_size == ARCHIVE_BYTES and export['archive']['bytes'] == ARCHIVE_BYTES and
         export['archive']['sha256'] == ARCHIVE_SHA and export['protocol_sha256'] == PROTOCOL_SHA, 'Expected return/archive pins differ')
    expected = export['files']
    need(len(expected) == 796, 'Expected 796 return bindings')
    for name, value in expected.items():
        confined(name); need(value['path'] == name and set(value) == {'path','bytes','sha256'}, 'Exact export binding required')
    before = archive.stat()
    with archive.open('rb') as f: parsed = stream_archive(f, expected)
    after = archive.stat()
    need((before.st_ino,before.st_size,before.st_mtime_ns) == (after.st_ino,after.st_size,after.st_mtime_ns), 'Archive changed while streaming')
    need(parsed['framing']['compressed_sha256'] == ARCHIVE_SHA and parsed['framing']['compressed_bytes'] == ARCHIVE_BYTES,
         'Independent compressed pin mismatch')
    files, raw = parsed['files'], parsed['metadata']
    need(raw['protocol.json'] == protocol_raw, 'Returned sealed protocol differs from local seal')
    static = []
    for name, expected_static in protocol['files'].items():
        need(files[name] == {'path':name, **expected_static}, 'Returned static source differs: '+name)
        actual = pin(ROOT/'outputs'/STAGE/name)
        need({k:actual[k] for k in ('bytes','sha256')} == expected_static, 'Local staged source differs: '+name)
        static.append(files[name])
    source_proofs = {}
    for name, flag in [('evidence/source_review.json','source_review_pass'),('evidence/source_tests.json','all_tests_pass')]:
        proof = strict_json(raw[name]); need(proof['complete'] is True and proof[flag] is True, 'Unapproved source evidence')
        if 'review' in name: need(proof['unresolved_blockers'] == [], 'Unresolved source blockers')
        source_pins = {r['path']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in proof['sources']}
        need(len(source_pins) == len(proof['sources']), 'Duplicate approved source')
        for source, expected_static in protocol['files'].items():
            if source.startswith(('research/','scripts/')):
                approved_name = 'research/codeformer_u64_role_gradients_v1/run_role_gradients.sh' if source == 'scripts/run_role_gradients.sh' else source
                need(source_pins[approved_name] == expected_static, 'Approved/current source pin differs: '+source)
        source_proofs[name] = {'complete':True,'sources_checked':len(source_pins),'proof_pin':files[name]}
    parent_protocol = strict_json(raw['evidence/parent_protocol.json'])
    readback = strict_json(raw['evidence/local_parent_readback.json'])
    parent_protocol_pin = pin(ROOT/'outputs'/PARENT/'protocol.json')
    need({k:parent_protocol_pin[k] for k in ('bytes','sha256')} ==
         {k:protocol['parent_protocol'][k] for k in ('bytes','sha256')}, 'Local sealed parent protocol differs')
    need(export['parent_protocol'] == protocol['parent_protocol'] and export['parent_payload_included'] is False,
         'Returned parent identity/payload scope differs')
    parent_result = local_parent_audit(parent_protocol, export['parent_consumed_bindings'], readback)
    docs = {k: strict_json(v) for k,v in raw.items() if k.endswith('.json')}
    events = [strict_json(line) for line in raw[OPREFIX+'events.jsonl'].splitlines()]
    resources = [strict_json(line) for line in raw[OPREFIX+'resources.jsonl'].splitlines()]
    need(raw[OPREFIX+'events.jsonl'].endswith(b'\n') and raw[OPREFIX+'resources.jsonl'].endswith(b'\n'), 'Incomplete JSONL tail')
    need(docs[OPREFIX+'operation_plan.json']['calls'] == protocol['plan'], 'Returned operation plan differs from seal')
    replay = replay_events(events, protocol['plan'])
    runtime = audit_runtime_evidence(docs, events, resources, protocol, files, export)
    source_ast = audit_worker_ast(ROOT/'outputs'/STAGE/'research/codeformer_u64_role_gradients_v1/worker.py')
    # Diagnostic details are inspected after all payloads and source approvals authenticate.
    metadata_path = OUT/'authenticated_small_metadata_v1.json'
    metadata = {'source_archive_sha256':ARCHIVE_SHA,'documents':docs,'events':events,'resources':resources}
    metadata_path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    receipt = {'complete':True,'status':'TRANSPORT_AND_OPERATION_REPLAY_VERIFIED_NOT_MODEL_QUALIFICATION',
        'created_UTC':datetime.now(timezone.utc).isoformat(),'audit_source':pin(Path(__file__)),
        'archive':pin(archive),'export':pin(export_path),'sidecar':pin(sidecar),'protocol':pin(protocol_path),
        'framing':parsed['framing'],'export_all_file_pins_match':True,'all_member_pins':files,
        'own_static_files_verified':len(static),'source_admission':source_proofs,'parent_protocol_actual':parent_protocol_pin,'parent':parent_result,
        'journal':replay,'runtime_evidence':runtime,'worker_source_AST':source_ast,'authenticated_metadata':pin(metadata_path),
        'model_qualified':False,'integrity_does_not_imply_quality':True,
        'actual_local_worker_model_VM_gradient_operations':0}
    (OUT/'transport_receipt_v1.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'framing':parsed['framing'],'journal':replay,'runtime_evidence':runtime,
          'parent_aliases':parent_result['model_only_local_aliases'],'receipt':str(OUT/'transport_receipt_v1.json')},indent=2))


class PureTests(unittest.TestCase):
    def test_namespace_escape(self):
        for name in ('../a','a//b','a/./b','/a','a\\b','a:b','a/CON.txt','a.','a/..','a\x00b'):
            with self.subTest(name=name), self.assertRaises(ValueError): confined(name)

    def fixture(self, names=('file.json',)):
        import gzip
        out = io.BytesIO()
        with tarfile.open(fileobj=out,mode='w',format=tarfile.USTAR_FORMAT) as t:
            for name in names:
                info=tarfile.TarInfo(STAGE+'/'+name); info.size=2
                t.addfile(info,io.BytesIO(b'{}'))
        return gzip.compress(out.getvalue(),mtime=0)

    def test_stream_valid(self):
        p=stream_archive(io.BytesIO(self.fixture()))
        self.assertEqual(p['framing']['regular_members'],1)

    def test_duplicate(self):
        with self.assertRaises(ValueError): stream_archive(io.BytesIO(self.fixture(('x','x'))))

    def test_case_collision(self):
        with self.assertRaises(ValueError): stream_archive(io.BytesIO(self.fixture(('x','X'))))

    def test_crc_corruption(self):
        raw=bytearray(self.fixture()); raw[-8]^=1
        with self.assertRaises(zlib.error): stream_archive(io.BytesIO(raw))

    def test_truncation(self):
        with self.assertRaises(ValueError): stream_archive(io.BytesIO(self.fixture()[:-5]))

    def test_gzip_trailing(self):
        with self.assertRaises(ValueError): stream_archive(io.BytesIO(self.fixture()+b'x'))

    def test_concatenated_gzip(self):
        with self.assertRaises(ValueError): stream_archive(io.BytesIO(self.fixture()+self.fixture()))

    def test_header_checksum(self):
        import gzip
        raw=bytearray(gzip.decompress(self.fixture())); raw[0]^=1
        with self.assertRaises(tarfile.InvalidHeaderError): stream_archive(io.BytesIO(gzip.compress(raw)))

    def test_nonzero_tar_trailer(self):
        import gzip
        raw=bytearray(gzip.decompress(self.fixture())); raw[-1]=1
        with self.assertRaises(ValueError): stream_archive(io.BytesIO(gzip.compress(raw)))

    def test_link_member(self):
        import gzip
        out=io.BytesIO()
        with tarfile.open(fileobj=out,mode='w',format=tarfile.USTAR_FORMAT) as t:
            info=tarfile.TarInfo(STAGE+'/link'); info.type=tarfile.SYMTYPE; info.linkname='x'; t.addfile(info)
        with self.assertRaises(ValueError): stream_archive(io.BytesIO(gzip.compress(out.getvalue())))

    def test_duplicate_json_keys(self):
        with self.assertRaises(ValueError): strict_json(b'{"a":1,"a":2}')

    def replay_fixture(self):
        plan=[]
        for slot in range(1,73):
            phase = 'historical_u64_parity' if slot<=4 else ('deterministic_same_u64_repeat' if slot<=8 else 'paired_TRAIN_gradient_diagnostic')
            plan.append({'slot':slot,'id':'case'+str(slot),'phase':phase,'gradient_queries':[] if slot<=8 else QUERIES})
        counters=dict.fromkeys(COUNTER_KEYS,0); rows=[]; token=0
        for call in plan:
            for op,query in [('forward',None)]+[('gradient_query',q) for q in call['gradient_queries']]:
                token+=1
                if op=='forward': counters[PHASE_CATEGORY[call['phase']]]+=1
                counters['full_forward_attempts' if op=='forward' else 'gradient_query_attempts']+=1
                data={'plan_slot':call['slot'],'id':call['id'],'phase':'call_'+str(call['slot']).zfill(4),'token':token}
                data.update({'category':PHASE_CATEGORY[call['phase']]} if op=='forward' else {'query':query})
                rows.append(dict(data,kind=op+'_reserved',counters=dict(counters),elapsed_seconds=len(rows)))
                counters['full_forward_returns' if op=='forward' else 'gradient_query_returns']+=1
                rows.append(dict(data,kind=op+'_returned',counters=dict(counters),elapsed_seconds=len(rows),synchronized_return=True))
        return plan,rows

    def test_full_event_replay(self):
        plan,rows=self.replay_fixture()
        self.assertEqual(replay_events(rows,plan)['operation_pairs'],264)

    def test_return_without_reservation(self):
        plan,rows=self.replay_fixture()
        with self.assertRaises(ValueError): replay_events(rows[1:],plan)

    def test_nonoverlapping_query_violation(self):
        plan,rows=self.replay_fixture(); rows[18],rows[17]=rows[17],rows[18]
        with self.assertRaises(ValueError): replay_events(rows,plan)

    def test_query_order_violation(self):
        plan,rows=self.replay_fixture(); rows[18]['query']='gate_mae'
        with self.assertRaises(ValueError): replay_events(rows,plan)

    def test_counter_tamper(self):
        plan,rows=self.replay_fixture(); rows[-1]['counters']['gradient_query_returns']=191
        with self.assertRaises(ValueError): replay_events(rows,plan)

    def test_unreturned_reservation(self):
        plan,rows=self.replay_fixture()
        with self.assertRaises(ValueError): replay_events(rows[:-1],plan)

    def test_reversed_time(self):
        plan,rows=self.replay_fixture(); rows[8]['elapsed_seconds']=-1
        with self.assertRaises(ValueError): replay_events(rows,plan)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--self-test',action='store_true'); args=parser.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PureTests))
        OUT.mkdir(exist_ok=True)
        (OUT/'pure_tests_v1.json').write_text(json.dumps({'complete':True,'all_tests_pass':result.wasSuccessful(),
            'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'audit_source':pin(Path(__file__)),'scope':'synthetic tar/gzip namespace/framing and complete finite operation replay',
            'actual_worker_model_VM_gradient_operations':0},indent=2)+'\n',encoding='utf-8',newline='\n')
        sys.exit(0 if result.wasSuccessful() else 1)
    audit()
