"""Inert source-reviewed remaining511 continuation; never retry any carried205.

Import the exact reviewed v2 finite-worker core, binding only this fresh namespace,
the independent205 carry closure, the unchanged absolute clock and shared bytes.
No prepare, verify or source test performs a neural call. Root must launch manually.
"""
import argparse
import _thread
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import run_u96_postfusion_lr_broader_tail_v2 as V

O = V.O
ROOT = V.ROOT
SOURCE = Path(__file__).resolve()
V2_SOURCE = ROOT / 'scripts/run_u96_postfusion_lr_broader_tail_v2.py'
V2_SOURCE_SHA = '05e63c21cf32e24809c35d478eed228be63c493b11b15774d0c76df7caa31be9'
V2_PLAN = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_tail_preparation_v2/remaining655_tail_plan_v2.json'
V2_PLAN_SHA = '86c90677f9096019660c03081c9194a325c4b50cda57735a3781b07cde828cd4'
V2_OUT = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_tail_v2'
PREP = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_tail_preparation_v3'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_tail_v3'
PLAN = PREP / 'remaining511_tail_plan_v3.json'
OWN_TEST = ROOT / 'tests/test_u96_postfusion_lr_broader_tail_v3.py'
CLOCK = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_tail_interruption_observation_v1/root_clock_observation_v1_r2.json'
CLOCK_SHA = 'c3ce4c0dbbc6aed8fef8186fac8eb8532fc473af24a92045179168b20f87a1af'
ORIGINAL_SECONDS = 571.5643963
V2_ABSOLUTE_DEADLINE = 18385.42586
# V2 used13829s; original571.5643963+13829 exceeds literal14400 by0.5643963s.
# Preserve that history; this distinct continuation shortens the old deadline1s.
ABSOLUTE_DEADLINE = V2_ABSOLUTE_DEADLINE - 1.
BOOT_EPOCH = 1792661878.5
BOOT_MARKER = '2026-10-22T09:37:58.5000000Z'
BASE_LIMITS = dict(V.LIMITS)
LAUNCH_BYTES_RESERVE = 16384
LIMITS = {**BASE_LIMITS, 'new_attempts': 511, 'new_returns': 511, 'new_saved': 511,
          'per_arm': {O.ARMS[0]: 255, O.ARMS[1]: 256}, 'global_seconds': 13828,
          'retained_original_bytes': 381083149 + 911108033,
          'maximum_new_output_bytes': 6 * 1024 ** 3 - 381083149 - 911108033 - LAUNCH_BYTES_RESERVE}
need, pin, bound, load = V.need, V.pin, V.bound, V.load


def windows_boot_marker():
    """Stable OS boot marker; psutil's derived floating Windows value drifts."""
    remaining = ABSOLUTE_DEADLINE - time.monotonic()
    need(remaining > 0, 'No new boot query beyond original deadline')
    powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    result = subprocess.run([str(powershell), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
        "[Console]::Write((Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o'))"],
        capture_output=True, text=True, check=True, timeout=min(10., remaining), shell=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    need(len(result.stdout) <= 64 and len(result.stderr) <= 4096 and not result.stderr.strip(), 'Bounded clean CIM boot response')
    return result.stdout.strip()


def clock_remaining(check_boot=True):
    """Never restart the clock; every idle/verification/source-review second counts."""
    info = time.get_clock_info('monotonic')
    need(info.monotonic and not info.adjustable and info.implementation == 'QueryPerformanceCounter()',
         'Same Windows monotonic clock implementation')
    if check_boot:
        need(windows_boot_marker() == BOOT_MARKER, 'Same exact stable Windows CIM boot; never reuse monotonic after reboot')
    observation = load(bound({'path': CLOCK.relative_to(ROOT).as_posix(),
                              'bytes': 1093, 'sha256': CLOCK_SHA}))
    prior = load(bound(observation['prior_clock']))
    need(observation['complete'] is True and observation['Windows_CIM_boot_epoch'] == BOOT_EPOCH
         and observation['Windows_CIM_LastBootUpTime_UTC'] == BOOT_MARKER
         and observation['original_absolute_deadline'] == V2_ABSOLUTE_DEADLINE
         and observation['conservative_stricter_absolute_deadline'] == ABSOLUTE_DEADLINE
         and prior['same_absolute_deadline_monotonic'] == V2_ABSOLUTE_DEADLINE
         and prior['monotonic_seconds'] <= observation['monotonic_seconds']
         and observation['monotonic_seconds'] <= time.monotonic(), 'Pinned pre-continuation clock observation')
    remaining = ABSOLUTE_DEADLINE - time.monotonic()
    need(0 < remaining <= LIMITS['global_seconds'], 'Original shared14400s clock remains available')
    return remaining


def chunks(slots):
    need(type(slots) is list and len(slots) == 511, 'Exactly511 never-attempted slots')
    need([r['new_output_index'] for r in slots] == list(range(206, 717)), 'Exact disjoint original206..716')
    need(Counter(r['arm'] for r in slots) == Counter(LIMITS['per_arm']), 'Exact255U96/256u64 remaining')
    return [{'chunk': i // 16 + 1, 'slots': slots[i:i + 16]} for i in range(0, 511, 16)]


def retained_file(row):
    """Only nine intentional DEVNULL log artifacts may use a zero-byte pin."""
    need(type(row) is dict and type(row.get('bytes')) is int, 'Exact integer retained byte count')
    if row['bytes'] != 0:
        return bound(row)  # Historical strict nonempty binder remains unchanged.
    allowed = {(V2_OUT / 'chunks' / ('%03d' % i) / 'worker.log').relative_to(ROOT).as_posix()
               for i in range(1, 10)}
    need(set(row) == {'path', 'bytes', 'sha256'} and row.get('path') in allowed
         and row.get('sha256') == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
         'Only exact nine retained intentionally empty V2 worker logs')
    path = V.safe(ROOT / row['path'])  # Ordinary, single-link, no symlink or reparse point.
    before = path.stat()
    need(before.st_size == 0, 'Retained empty log is still empty')
    with path.open('rb') as stream:
        raw = stream.read(1)
    after = path.stat()
    need(raw == b'' and (before.st_size, before.st_mtime_ns, before.st_ino, before.st_nlink) ==
         (after.st_size, after.st_mtime_ns, after.st_ino, after.st_nlink), 'Stable ordinary empty log hash/size/identity')
    return path


def namespace_equal(folder, files, directories=None):
    need(type(files) is list, 'Explicit immutable retained file namespace')
    actual_files = []
    actual_dirs = []
    for path in sorted(V.safe(folder, True).rglob('*')):
        V.safe(path, path.is_dir())
        if path.is_file():
            actual_files.append(path.relative_to(ROOT).as_posix())
        else:
            actual_dirs.append(path.relative_to(ROOT).as_posix())
    need(sorted(actual_files) == sorted(r['path'] for r in files), 'Retained namespace has no added/missing files')
    for row in files:
        retained_file(row)
    if directories is not None:
        need(sorted(actual_dirs) == sorted(directories), 'Retained exact directory namespace including empty010')


def dependencies(audit_pin):
    need('torch' not in sys.modules, 'All continuation proof verification before Torch')
    need(pin(V2_SOURCE)['sha256'] == V2_SOURCE_SHA and pin(V2_PLAN)['sha256'] == V2_PLAN_SHA,
         'Exact unchanged independently source-reviewed V2 core and plan')
    original, retained61, original_proofs = V.dependency_contract()
    oldplan = load(V2_PLAN)
    need(oldplan['remaining_slots'] == original['forward_slots'][61:] and oldplan['limits'] == BASE_LIMITS,
         'Original sealed655 plan remains exact')
    audit = load(bound(audit_pin))
    need(audit['complete'] is True and audit['independent_interrupted_saved_audit_pass'] is True
         and audit['outputs_verified'] == 205 and audit['original_carried61'] == 61 and audit['new_tail144'] == 144
         and audit['forward_outcome_unknown'] is False
         and audit['first_unfilled_slot'] == original['forward_slots'][205]
         and audit['remaining_slots'] == original['forward_slots'][205:],
         'Independent205 proof required before any continuation plan')
    retained_pin = audit['verified205_output_manifest']
    namespace_pin = audit['retained_interrupted_namespaces']
    retained = load(bound(retained_pin)); namespace = load(bound(namespace_pin))
    need(retained['complete'] is True and len(retained['rows']) == 205
         and [r['slot'] for r in retained['rows']] == original['forward_slots'][:205], 'Exact audited205 ordered prefix')
    canonical = lambda rows: [{**{k: row[k] for k in ('slot', 'record', 'inputs')},
        'output': {k: row['output'][k] for k in (*O.OUTPUT_KEYS, 'metrics', 'provenance')}} for row in rows]
    oldrecords = V.carried_records(retained61)
    need(canonical(retained['rows'][:61]) == oldrecords, 'Original61 records exact, without regeneration')
    v2records = []
    for index in range(1, 10):
        terminal = load(V2_OUT / 'chunks' / ('%03d' % index) / 'terminal.json')
        need(terminal['complete'] is True and terminal['clean_exit'] is True
             and terminal['forward_outcome_unknown'] is False, 'Known complete carried V2 chunk')
        v2records.extend(terminal['records'])
    need(canonical(retained['rows'][61:]) == v2records, 'Exact all144 V2 saved records')
    need(len(namespace['original_files']) == 952 and len(namespace['v2_files']) == 2414
         and namespace['v2_bytes'] == 911108033
         and sum(r['bytes'] for r in namespace['v2_files']) == 911108033,
         'Exact original952 plus interrupted2414 files/shared byte charge')
    # The original closure was already verified by V.dependency_contract; compare pins too.
    old_worker = load(bound(original_proofs['retained_worker']))
    need(namespace['original_files'] == old_worker.get('files', old_worker.get('rows')),
         'Exact retained original namespace lineage')
    namespace_equal(V2_OUT, namespace['v2_files'], namespace['v2_directories'])
    need((V2_OUT / 'chunks/010').is_dir() and not any((V2_OUT / 'chunks/010').iterdir())
         and not (V2_OUT / 'supervisor.json').exists() and not (V2_OUT / 'comparison.json').exists(),
         'Interrupted empty010 preserved; no fabricated V2 terminal')
    clock_remaining()
    proofs = {'partial205_audit': audit_pin, 'retained205': retained_pin, 'retained_namespaces': namespace_pin,
              'old_plan': pin(V2_PLAN), 'old_runner': pin(V2_SOURCE), 'clock': pin(CLOCK),
              'original_prerequisites': original_proofs}
    return original, retained61, retained, proofs


def prepare(source_review, partial_audit, partial_sha):
    need(not PLAN.exists(), 'New immutable continuation plan; never overwrite/reselect')
    audit_pin = pin(V.safe(partial_audit))
    need(audit_pin['sha256'] == partial_sha, 'Root explicit independently audited205 SHA')
    original, retained61, retained205, proofs = dependencies(audit_pin)
    sources = [pin(SOURCE), pin(OWN_TEST)]
    review = pin(V.safe(source_review))
    O.reviewed_sources(load(bound(review)), {r['path'] for r in sources})
    plan = {'complete': True, 'format': 'broader-preservation-distinct-tail511-v3',
            'status': 'SEALED_PREPARED_UNRUN', 'prerequisites': proofs, 'sources': sources, 'source_review': review,
            'remaining_slots': original['forward_slots'][205:], 'chunks': chunks(original['forward_slots'][205:]),
            'limits': LIMITS, 'absolute_deadline_monotonic': ABSOLUTE_DEADLINE, 'boot_epoch': BOOT_EPOCH,
            'original_supervisor_seconds': ORIGINAL_SECONDS, 'v2_original_absolute_deadline': V2_ABSOLUTE_DEADLINE,
            'deadline_arithmetic_correction_seconds': 1., 'cumulative_total_seconds_cap': 14400,
            'carried61': 61, 'carried_v2': 144, 'carried205': 205, 'new_calls_maximum': 511,
            'combined_calls_maximum': 716, 'native_new_calls': 0, 'optimization_permitted': False,
            'no_automatic_retry_resume': True, 'FINAL_used': False, 'model_qualified': False,
            'app_default_changed': False, 'preparation_neural_calls': 0}
    PREP.mkdir(parents=True, exist_ok=True)
    with PLAN.open('xb') as stream:
        stream.write(O.B.json_bytes(plan))
    return pin(PLAN)


def verify(expected_sha):
    need(pin(PLAN)['sha256'] == expected_sha, 'Exact explicit remaining511 plan SHA')
    plan = load(PLAN)
    original, retained61, retained205, proofs = dependencies(plan['prerequisites']['partial205_audit'])
    need(plan['complete'] is True and plan['format'] == 'broader-preservation-distinct-tail511-v3'
         and plan['prerequisites'] == proofs and plan['sources'] == [pin(SOURCE), pin(OWN_TEST)]
         and plan['remaining_slots'] == original['forward_slots'][205:]
         and plan['chunks'] == chunks(original['forward_slots'][205:]) and plan['limits'] == LIMITS
         and plan['absolute_deadline_monotonic'] == ABSOLUTE_DEADLINE and plan['boot_epoch'] == BOOT_EPOCH
         and plan['no_automatic_retry_resume'] is True and plan['optimization_permitted'] is False
         and plan['FINAL_used'] is False and plan['model_qualified'] is False and plan['app_default_changed'] is False,
         'Exact disjoint continuation proof/source/clock/byte contract')
    O.reviewed_sources(load(bound(plan['source_review'])), {r['path'] for r in plan['sources']})
    return plan, original, retained61


def bind_core():
    """The existing finite worker/model/math implementation remains byte-for-byte sealed."""
    need(pin(V2_SOURCE)['sha256'] == V2_SOURCE_SHA, 'Bind only exact reviewed immutable core')
    V.OUT = OUT
    V.LIMITS = LIMITS
    V.verify = verify
    V.__file__ = str(SOURCE)  # run_chunk spawns this wrapper; --worker dispatches unchanged chunk_worker.


def aggregate(original, retained61, carried205, new_records, new_counts, writer, expected_sha):
    need(len(new_records) == 511 and [r['slot'] for r in new_records] == original['forward_slots'][205:],
         'Exactly511 never-regenerated outputs for aggregate')
    V.counts_valid(new_counts, original['forward_slots'][205:])
    need(new_counts['attempts'] == new_counts['returns'] == new_counts['saved'] == 511, 'Exact511 counters')
    v2counts = V.blank_counts()
    for row in carried205['rows'][61:]:
        for key in ('attempts', 'returns', 'saved'):
            v2counts[key] += 1; v2counts['per_arm'][row['slot']['arm']][key] += 1
    combined655 = V.combine_counts([v2counts, new_counts])
    class Capture:
        captured = None
        def json(self, path, value):
            if path == OUT / 'comparison.json':
                self.captured = value
                return None
            return writer.json(path, value)
    capture = Capture()
    V.aggregate(original, retained61, carried205['rows'][61:] + new_records, combined655, capture, expected_sha)
    value = capture.captured
    need(value is not None and value['counters']['saved_outputs'] == 716, 'Unchanged V2 full gate aggregation produced716')
    value.pop('paired_fresh_cases', None)  #358 complete combined cases;205 carried slots are not new forwards.
    value.update({'format': 'original61_plus_v2_144_plus_v3_511', 'new_tail_counters': new_counts,
                  'paired_cases': 358,
                  'carried_paired_output_slots': 205, 'carried_original61_slots': 61, 'carried_v2_slots': 144,
                  'new_tail_forward_returns': 511, 'prior_v2_plan_sha256': V2_PLAN_SHA,
                  'continuation_plan_sha256': expected_sha,
                  'absolute_deadline_monotonic': ABSOLUTE_DEADLINE,
                  'retained_partial205_audit': load(PLAN)['prerequisites']['partial205_audit']})
    for row in value['historical_membership406']:
        if row['status'] == 'SOURCE_BOUND_ORIGINAL61_PLUS_TAIL655_PAIRED_OUTPUTS':
            row['status'] = 'SOURCE_BOUND_ORIGINAL61_PLUS_V2_144_PLUS_V3_511_PAIRED_OUTPUTS'
    writer.json(OUT / 'comparison.json', value)
    return value


def launch_handshake(expected_sha, token):
    """Publish actual direct interpreter identity; await root's exact PID approval."""
    need(type(token) is str and len(token) == 32 and all(c in '0123456789abcdef' for c in token),
         'Root unique generated launch token')
    folder = V.safe(PREP / 'runtime_launch' / token, True)
    need(not folder.exists(), 'Fresh owned launch namespace; no restart')
    folder.mkdir(parents=True)
    writer = V.Writer(folder, LAUNCH_BYTES_RESERVE)
    import psutil
    process = psutil.Process()
    identity = {'complete': True, 'token': token, 'supervisor_pid': os.getpid(),
        'parent_pid': process.ppid(), 'create_time_epoch': process.create_time(),
        'python_executable': sys.executable, 'python_prefix': sys.prefix,
        'plan_sha256': expected_sha, 'runner': pin(SOURCE), 'boot_epoch': BOOT_EPOCH,
        'absolute_deadline_monotonic': ABSOLUTE_DEADLINE, 'new_neural_calls': 0}
    identity_pin = writer.json(folder / 'identity.json', identity)
    approval_path = folder / 'root_approval.json'
    while not approval_path.exists():
        clock_remaining(check_boot=False)  # Stable boot was checked before this owned handshake.
        time.sleep(.1)
    approval = V.json_file(approval_path, 4096)
    need(type(approval) is dict and approval == {'token': token, 'supervisor_pid': os.getpid(),
        'plan_sha256': expected_sha, 'actual_process_identity_verified': True},
        'Exact explicit root approval of actual supervisor identity before inference')
    need(type(approval['supervisor_pid']) is int and type(approval['actual_process_identity_verified']) is bool,
         'Strict root launch approval scalar types')
    files = sorted(p.name for p in folder.iterdir())
    need(files == ['identity.json', 'root_approval.json'] and V.tree_bytes(folder) <= LAUNCH_BYTES_RESERVE,
         'Launch metadata bound within reserved shared bytes')
    return {'identity': identity_pin, 'root_approval': pin(approval_path), 'reserved_bytes': LAUNCH_BYTES_RESERVE}


def supervise_body(expected_sha, launch_proof):
    start = time.monotonic()
    plan, original, retained61 = verify(expected_sha)
    carried205 = load(bound(plan['prerequisites']['retained205']))
    need(not OUT.exists(), 'Fresh continuation namespace; automatic retry/resume forbidden')
    V.safe(OUT, True).mkdir()
    writer = V.Writer(OUT, LIMITS['maximum_new_output_bytes'])
    writer.json(OUT / 'started.json', {'complete': True, 'plan': pin(PLAN), 'runner': pin(SOURCE),
        'supervisor_pid': os.getpid(), 'python_executable': sys.executable, 'python_prefix': sys.prefix,
        'boot_epoch': BOOT_EPOCH, 'started_monotonic': start, 'absolute_deadline_monotonic': ABSOLUTE_DEADLINE,
        'new_forward_cap': 511, 'carried205_new_calls': 0, 'original61_bytes': 381083149,
        'retained_v2_bytes': 911108033, 'new_bytes_maximum': LIMITS['maximum_new_output_bytes'],
        'launch_proof': launch_proof, 'model_qualified': False})
    records = []; counters = V.blank_counts(); completed = []
    try:
        for chunk in plan['chunks']:
            clock_remaining()
            terminal = V.run_chunk(plan, expected_sha, chunk, writer, ABSOLUTE_DEADLINE, counters)
            counters = V.combine_counts([counters, terminal['counters']]); records.extend(terminal['records'])
            completed.append({'chunk': chunk['chunk'], 'terminal': pin(OUT / 'chunks' / ('%03d' % chunk['chunk']) / 'terminal.json')})
            writer.json(OUT / 'progress' / ('%03d.json' % chunk['chunk']), {'complete': True,
                'completed_chunks': len(completed), 'new_counts': counters, 'carried205_new_calls': 0,
                'combined_attempts': 205 + counters['attempts'], 'combined_returns': 205 + counters['returns'],
                'combined_saved': 205 + counters['saved'], 'remaining_seconds': clock_remaining(),
                'actual_previous_worker_exit_confirmed': True})
        verify(expected_sha); writer.synchronize_external()
        comparison = aggregate(original, retained61, carried205, records, counters, writer, expected_sha)
        remaining = clock_remaining(); writer.synchronize_external()
        writer.json(OUT / 'supervisor.json', {'complete': True, 'all716_measurements_complete': True,
            'quality_status': comparison['quality_status'], 'new_tail_counts': counters,
            'carried61': 61, 'carried_v2': 144, 'carried205': 205, 'combined_counts': comparison['counters'],
            'completed_chunks': completed, 'forward_outcome_unknown': False, 'seconds': time.monotonic() - start,
            'remaining_original_deadline_seconds': remaining, 'absolute_deadline_monotonic': ABSOLUTE_DEADLINE,
            'model_qualified': False, 'app_default_changed': False, 'automatic_retry_resume': False,
            'training_optimizer_VM_calls': 0, 'final90_root_visual_review': 'pending', 'independent_saved_pixel_audit': 'pending'})
        return 0 if comparison['quality_status'] == 'BROAD_PRESERVATION_GATE_PASS_NOT_QUALIFIED' else 2
    except BaseException as error:
        writer.synchronize_external()
        stop = V.json_file(OUT / 'chunk_stop.json') if (OUT / 'chunk_stop.json').exists() else None
        writer.json(OUT / 'supervisor.json', {'complete': True, 'status': 'LOCAL_V3_INFRASTRUCTURE_STOP_NOT_QUALIFIED',
            'completed_chunk_counts': counters, 'completed_chunks': completed,
            'current_chunk_stop': pin(OUT / 'chunk_stop.json') if stop else None,
            'observed_total_new_counts': V.combine_counts([counters, stop['last_durable_counters']]) if stop else counters,
            'forward_outcome_unknown': stop['forward_outcome_unknown'] if stop else False,
            'unobserved_attempt_upper_bound': stop['unobserved_attempt_upper_bound'] if stop else 0,
            'all716_measurements_complete': False, 'carried205_outputs_unchanged': True,
            'seconds': time.monotonic() - start, 'error': type(error).__name__, 'message': str(error)[:4096],
            'automatic_retry_resume': False, 'model_qualified': False, 'app_default_changed': False,
            'training_optimizer_VM_calls': 0})
        return 1


def supervise(expected_sha, launch_token):
    need(threading.current_thread() is threading.main_thread(), 'Owned supervisor main thread')
    timer = threading.Timer(clock_remaining(), _thread.interrupt_main); timer.daemon = True; timer.start()
    try:
        launch_proof = launch_handshake(expected_sha, launch_token)
        return supervise_body(expected_sha, launch_proof)
    finally:
        timer.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-plan', action='store_true'); action.add_argument('--verify-only', action='store_true')
    action.add_argument('--run', action='store_true'); action.add_argument('--worker', action='store_true')
    parser.add_argument('--plan-sha'); parser.add_argument('--source-review')
    parser.add_argument('--partial-audit'); parser.add_argument('--partial-sha')
    parser.add_argument('--chunk', type=int); parser.add_argument('--token')
    parser.add_argument('--launch-token')
    args = parser.parse_args()
    if args.prepare_plan:
        need(all(type(v) is str and v for v in (args.source_review, args.partial_audit, args.partial_sha)), 'All independent proofs required')
        print(O.B.json_bytes(prepare(args.source_review, args.partial_audit, args.partial_sha)).decode(), end=''); return 0
    need(type(args.plan_sha) is str and len(args.plan_sha) == 64 and all(c in '0123456789abcdef' for c in args.plan_sha), 'Explicit plan SHA')
    bind_core()
    if args.verify_only:
        verify(args.plan_sha)
        print(O.B.json_bytes({'complete': True, 'remaining511': 511, 'chunks': 32, 'new_neural_calls': 0,
                             'remaining_same_deadline_seconds': clock_remaining()}).decode(), end=''); return 0
    if args.worker:
        return V.chunk_worker(args.plan_sha, args.chunk, args.token)
    return supervise(args.plan_sha, args.launch_token)


if __name__ == '__main__':
    raise SystemExit(main())
