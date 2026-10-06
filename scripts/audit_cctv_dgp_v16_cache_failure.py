"""Audit returned zero-update V16 cache failure and bootstrap provenance; no models."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit():
    started = time.monotonic()
    returned = ROOT / 'outputs/cctv_dgp_broader_codes_failure_return_v16'
    out = returned / 'outputs/broader_codes_v16'
    core = read(returned / 'local_failure_import.json')
    require(core['complete'] and core['success'] is False and core['completed_trace_records_checked'] == 0,
            'Require completed independent failed-run import')
    terminal = read(ROOT / 'outputs/failure_export.json')
    archive = ROOT / 'outputs/cctv-dgp-broader-codes-v16-failure.tar.gz'
    require(terminal['complete'] and sha(archive) == terminal['archive_sha256'] and
            archive.stat().st_size == terminal['bytes'], 'Export hash/size differs')
    preparation = read(ROOT / 'outputs/cctv_dgp_broader_codes_v16_transfer_audit.json')
    p = read(returned / 'broader_codes_protocol_v16.json')
    require(sha(returned / 'broader_codes_protocol_v16.json') == preparation['protocol_sha256'], 'Protocol differs')
    launch = read(returned / 'supervisor_launch.json')
    recovery = launch['preflight_recovery']
    repair_transfer = read(ROOT / 'outputs/cctv_dgp_v16_preflight_recovery_r1_transfer_audit.json')
    source = recovery['launcher_source'].encode('utf-8')
    source_hash = hashlib.sha256(source).hexdigest()
    require(source_hash == launch['launcher_sha256'] == recovery['launcher_sha256'] == repair_transfer['recovery_sha256']
            and source == (ROOT / 'outputs/recover_cctv_dgp_broader_codes_v16_preflight_r1.py').read_bytes(),
            'Recovery executable provenance differs')
    require(launch['complete'] and launch['protocol_sha256'] == preparation['protocol_sha256'] and
            launch['archive_sha256'] == preparation['archive_sha256'] and
            launch['frozen_bootstrap_sha256'] == preparation['bootstrap_sha256'] and
            launch['session'] == 'dgp_broader_codes_v16' and not launch['automatic_resume'] and
            not launch['production_promoted'], 'Launch binding/scope differs')
    require(all(recovery[key] is False for key in ['training_recipe_changed', 'frozen_bundle_changed', 'training_resume']),
            'Unexpected recovery scope')
    original, corrected = recovery['original_failure'], recovery['corrected_preflight']
    require(original['returncode'] == 1 and "TypeError: unsupported operand type(s) for /: 'str' and 'str'" in original['stderr']
            and corrected['returncode'] == 0 and corrected['stderr'] == '' and
            corrected['stdout'] == 'V16 source/data/CUDA availability preflight passed\n', 'Preflight transcript differs')
    for probe in [original, corrected]:
        require(probe['training_started'] is False and probe['timeout_seconds'] == 120 and
                0 <= probe['seconds'] < 120, 'Preflight bounds/scope differ')
    failure = read(out / 'failure.json')
    require(failure['error'] == 'Cache projection exceeds900s' and failure['optimizer_updates'] ==
            failure['backward_calls'] == 0 and not failure['resume_permitted'], 'Wrong failure/counters')
    require(not any((out / name).exists() for name in ['cuda_preflight.json', 'update_trace.jsonl', 'results.json']) and
            not (returned / 'supervisor_completion.json').exists(), 'Unexpected training/success evidence')
    execution = read(out / 'execution.json')
    require(execution['protocol_sha256'] == preparation['protocol_sha256'] and execution['design'] == p['design'] and
            execution['teacher_roles'] == ['train'] and 'L4' in execution['gpu'] and
            all(execution[key] is False for key in ['native_used', 'native_reserved_used', 'production_promoted']),
            'Execution binding/scope differs')
    timing = read(out / 'cache_timing.json')
    require(timing['references'] == p['design']['cache_timing_at_reference'] == 20 and
            timing['cap_seconds'] == p['design']['cache_cap_seconds'] == 900, 'Timing sample/limit differs')
    expected_refs = p['references'][:timing['references']]
    require(all(ref['role'] == 'train' for ref in expected_refs), 'Timing prefix roles differ')
    teacher_paths = list((out / 'teacher').glob('*.npy'))
    require({path.name for path in teacher_paths} == {ref['id'] + '_codes.npy' for ref in expected_refs},
            'Teacher files cross cache prefix/roles')
    for path in teacher_paths:
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.int64 and value.shape == (256,) and value.min() >= 0 and value.max() < 1024,
                'Invalid teacher code array')
    train = sum(ref['role'] == 'train' for ref in p['references'])
    remaining = train - timing['references'] + len(p['validation_cases'])
    factor = 1 + remaining * p['design']['timing_safety_factor'] / timing['references']
    rebuilt = timing['seconds'] * factor + 30
    require(math.isclose(rebuilt, timing['projected_seconds'], rel_tol=0, abs_tol=1e-9) and
            rebuilt > timing['cap_seconds'], 'Timing arithmetic/gate differs')
    partial = out / 'partial_conditioner.pth'
    require(partial.is_file(), 'Preserve missing partial checkpoint diagnosis')
    result = {'complete': True, 'success': False, 'failure_archive_sha256': terminal['archive_sha256'],
              'protocol_sha256': preparation['protocol_sha256'], 'recovery_source_sha256': source_hash,
              'original_path_failure_verified': True, 'corrected_preflight_verified': True,
              'optimizer_updates': 0, 'backward_calls': 0, 'teacher_arrays_checked': len(teacher_paths),
              'cache_reference_count': timing['references'], 'cache_elapsed_seconds': timing['seconds'],
              'projected_seconds': rebuilt, 'elapsed_multiplier': factor, 'projection_cap_seconds': 900,
              'startup_seconds_separately_recorded': None, 'steady_cache_rate_verified': False,
              'partial_checkpoint_sha256': sha(partial), 'partial_checkpoint_bytes': partial.stat().st_size,
              'supervisor_seconds': read(returned / 'supervisor_failure.json')['seconds'],
              'frozen_source_members_reverified_by_import': 23, 'resume_permitted': False,
              'neural_forwards': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
              'seconds': time.monotonic() - started,
              'limitation': 'Confirms serialized zero-update failure and timing bias; no learned output/usefulness, cache replay or startup/steady-time measurement. Original cache remains on VM.'}
    receipt = returned / 'local_cache_failure_audit.json'
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
