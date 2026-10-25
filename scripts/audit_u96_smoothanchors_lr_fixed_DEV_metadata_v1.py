"""Independent completed fixed68 saved-file/counter/resource readback only.

This script never imports model modules, Torch, or starts/resumes inference.
It requires the exact fresh completed comparison and owned supervisor receipts.
"""
import hashlib
import json
import math
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_preparation'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_run_v1'
AUDIT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_metadata_audit'
PLAN_SHA = 'bb8a60f29630ddb21a2e9cea7e43b30c689b78f6474e6671a28c516e21bf1a6d'
ARMS = ('incumbent_U96', 'postfusion_LR_u64')


def main():
    checks = 0
    def need(ok, message):
        nonlocal checks
        checks += 1
        if not ok: raise AssertionError(message)
    def load(path):
        def pairs(rows):
            value = {}
            for key, item in rows:
                need(key not in value, 'Duplicate JSON key'); value[key] = item
            return value
        def bad(value): raise ValueError('Nonfinite token ' + value)
        result = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=bad)
        def finite(value):
            if isinstance(value, float): need(math.isfinite(value), 'Nonfinite exponent')
            elif isinstance(value, dict):
                for child in value.values(): finite(child)
            elif isinstance(value, list):
                for child in value: finite(child)
        finite(result); return result
    def pin(path):
        path = Path(path); info = path.lstat()
        need(path.resolve().is_relative_to(ROOT) and stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary workspace evidence')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 ** 2), b''): h.update(block)
        return {'path': path.relative_to(ROOT).as_posix(), 'bytes': info.st_size, 'sha256': h.hexdigest()}
    def bound(row):
        actual = pin(ROOT / row['path'])
        need(actual == {key: row[key] for key in ('path', 'bytes', 'sha256')}, 'Exact saved pin ' + row['path'])
        return actual
    ppath = PREP / 'fixed_DEV_plan_v1.json'; plan = load(ppath)
    need(pin(ppath)['sha256'] == PLAN_SHA, 'Exact sealed fixed68 plan')
    for row in [plan['cohort'], plan['data_reuse'], plan['protocol'], plan['head_file'], *plan['sources'], *plan['model'].values(), *plan['prerequisites'].values()]: bound(row)
    need(not (OUT / 'status.pending').exists() and not (OUT / 'failure.json').exists(), 'No interrupted/failure evidence in completed return')
    status = load(OUT / 'status.json'); comparison = load(OUT / 'comparison.json')
    supervisor = load(PREP / 'fixed_DEV_supervisor_v1.json')
    need(supervisor['complete'] and supervisor['worker_exit'] == 0 and not supervisor['timeout'] and not supervisor['forward_outcome_unknown'] and not supervisor['status_pending_present'], 'Normal authoritative supervisor end')
    need(status['complete'] and status['phase'] == status['label'] == 'complete' and not status['forward_outcome_unknown'], 'Complete durable known status')
    need(comparison['complete'] and comparison['status'] == 'SAVED_FIXED_DEV_OUTPUTS_PENDING_INDEPENDENT_AUDIT_VISUAL_REVIEW' and not comparison['model_qualified'] and not comparison['app_default_changed'], 'Saved-output completion is not qualification')
    counters = {'full_forward_attempts': 136, 'full_forward_returns': 136, 'successful_saved_outputs': 136, 'completed_cases': 68,
                'per_arm': {arm: {'attempts': 68, 'returns': 68, 'saved': 68} for arm in ARMS},
                'gradient_queries': 0, 'backward_calls': 0, 'optimizer_constructions': 0, 'optimizer_updates': 0}
    need(status['counters'] == comparison['counters'] == supervisor['counters'] == counters, 'Exact136/68calls and zero optimization counters')
    need(status['plan_sha256'] == supervisor['plan_sha256'] == PLAN_SHA and comparison['plan'] == pin(ppath), 'All terminal plan joins')
    bound(supervisor['last_durable_status'])
    controls = load(OUT / 'controls_zero_calls.json')
    need(controls == {'complete': True, 'controls': plan['zero_call_controls'], 'neural_calls': 0}, 'All40originalcontrols zero neural calls')
    need([row['id'] for row in comparison['cases']] == plan['case_order'] and len(comparison['cases']) == 68, 'Exact68resultorder')
    owned_prefix = OUT.relative_to(ROOT).as_posix() + '/'
    expected_files = {owned_prefix + name for name in ('comparison.json', 'controls_zero_calls.json', 'resources.jsonl', 'started.json', 'status.json')}
    hashes, case_pins = [], []
    for index, (frozen, case) in enumerate(zip(plan['cases'], comparison['cases'])):
        need(case['complete'] and case['id'] == frozen['id'] and case['input_review'] == frozen['input_review'] and case['geometry'] == frozen['geometry'] and case['cohort'] == frozen['cohort'] and case['split'] == frozen['split'] and case['paired_target'] == frozen['target'] and case['appearance_flag'] == frozen.get('appearance_flag', False), 'Original labels/geometry/source/target exact per case')
        need(case['native_PSNR_SSIM_identity_metrics'] is None and set(case['outputs']) == set(ARMS), 'No invented native metrics and exact two arms')
        folder = OUT / 'cases' / case['id']; cp = pin(folder / 'case.json')
        need(load(folder / 'case.json') == case, 'Individual/result case exact JSON join')
        case_pins.append(cp); expected_files.add(cp['path'])
        need(set(case['inputs']) == ({'original', 'prepared', 'support', 'target'} if frozen['target'] else {'original', 'prepared', 'support'}), 'Exact common input/target namespace')
        for row in case['inputs'].values():
            hashes.append(bound(row)); expected_files.add(row['path'])
        for arm_index, arm in enumerate(ARMS):
            output = case['outputs'][arm]
            need(set(output) == {'native512', 'logits', 'LQ', 'unclipped', 'raw', 'png', 'provenance'}, 'Exact output schema')
            expected_provenance = {'arm': arm, 'R2_file_sha256': plan['model']['r2']['sha256'], 'U96_file_sha256': plan['model']['u96']['sha256'],
                'head_file_sha256': plan['head_file']['sha256'] if arm_index == 1 else None,
                'head_canonical_sha256': plan['candidate_head_canonical_sha256'] if arm_index == 1 else None,
                'w': 1., 'detach16': True, 'AdaIN': True, 'CPU_FP32': True, 'display_processing': 'none',
                'source_candidate_training_updates': 64 if arm_index == 1 else None, 'model_qualified': False,
                'same_observed_geometry': True, 'outside_support_exact_observed_pixels': True, 'forward_sites': ['32', '64', '128', '256']}
            need(output['provenance'] == expected_provenance, 'Exact fixed incumbent/candidate provenance')
            record = load(folder / arm / 'record.json')
            need(record['complete'] and record['case_id'] == frozen['id'] and
                 {key: record[key] for key in output} == output, 'Individual output receipt exact result join')
            expected_files.add((folder / arm / 'record.json').relative_to(ROOT).as_posix())
            rc = record['counters']
            # Record is saved before increasing the successful_saved_outputs
            # counter, while attempted/returned call counts are already durable.
            n = index * 2 + arm_index + 1
            need(rc['full_forward_attempts'] == rc['full_forward_returns'] == n and rc['successful_saved_outputs'] == n - 1 and rc['completed_cases'] == index,
                 'Durable per-output attempt/return/preincrement saved counters')
            for key in ('gradient_queries', 'backward_calls', 'optimizer_constructions', 'optimizer_updates'): need(rc[key] == 0, 'No optimization in per-output receipt')
            for key in ('native512', 'logits', 'LQ', 'unclipped', 'raw', 'png'):
                row = output[key]; hashes.append(bound(row)); expected_files.add(row['path'])
    actual_files = {q.relative_to(ROOT).as_posix() for q in OUT.rglob('*') if q.is_file()}
    need(actual_files == expected_files, 'Exact completed owned output namespace')
    need(not any(q.name in {row['id'] for row in plan['zero_call_controls']} for q in (OUT / 'cases').iterdir()), 'No output directories for40zero-callcontrols')
    resources = []
    with (OUT / 'resources.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            need(len(line) < 4096, 'Bounded resource line')
            resources.append(json.loads(line))
    need(len(resources) == 274 and resources[0]['label'] == 'before_model_load' and resources[-1]['label'] == 'final', 'Before-load+272before/after+final samples')
    lim = plan['limits']; previous_seconds = 0
    for row in resources:
        need(previous_seconds <= row['seconds'] <= lim['internal_seconds'] and row['worker_RSS_bytes'] <= lim['maximum_worker_RSS_bytes'] and row['available_RAM_bytes'] >= lim['minimum_available_RAM_bytes'] and row['free_disk_bytes'] >= lim['minimum_free_disk_bytes'] and row['output_bytes'] <= lim['maximum_output_bytes'], 'Saved finite resource samples')
        previous_seconds = row['seconds']
    payload_bytes = sum((ROOT / name).stat().st_size for name in actual_files)
    need(payload_bytes <= lim['maximum_output_bytes'] and comparison['seconds'] <= lim['internal_seconds'] and supervisor['seconds'] <= lim['external_seconds'], 'Final actual owned bytes/time below caps')
    report = {'complete': True, 'independent_saved_metadata_pass': True, 'checks': checks,
        'source': pin(Path(__file__).resolve()), 'plan': pin(ppath), 'comparison': pin(OUT / 'comparison.json'),
        'status': pin(OUT / 'status.json'), 'supervisor': pin(PREP / 'fixed_DEV_supervisor_v1.json'),
        'cases': 68, 'case_order': plan['case_order'], 'counters': counters, 'zero_call_controls': 40,
        'payload_files': len(actual_files), 'payload_bytes': payload_bytes, 'bound_output_file_pins': hashes,
        'case_receipts': case_pins, 'resource_samples': len(resources),
        'resources': {'peak_sampled_RSS_bytes': max(r['worker_RSS_bytes'] for r in resources), 'minimum_sampled_available_RAM_bytes': min(r['available_RAM_bytes'] for r in resources), 'minimum_sampled_free_disk_bytes': min(r['free_disk_bytes'] for r in resources), 'worker_seconds': comparison['seconds'], 'supervisor_seconds': supervisor['seconds'], 'output_cap_headroom_bytes': lim['maximum_output_bytes'] - payload_bytes},
        'scope': 'Exact saved namespace/hash/JSON/provenance/counter/resource audit. Pixel reconstruction/metrics and root visual review are separate. No new inference, model, training, VM or browser calls.',
        'paired36_vs_native32_separate': True, 'native_PSNR_SSIM_identity_claimed': False,
        'Torch_model_neural_gradient_optimizer_training_VM_HTTP_browser_calls': 0, 'model_qualified': False}
    AUDIT.mkdir(exist_ok=False)
    destination = AUDIT / 'independent_saved_metadata_v1.json'
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': True, 'checks': checks, 'receipt': pin(destination), 'model_qualified': False}))


if __name__ == '__main__':
    main()
