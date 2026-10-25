"""Independent saved-only terminal tail89/aggregate136 metadata audit.

No model, Torch, browser, HTTP, inference, optimizer, or VM calls. This audit
requires completed evidence and never retries an incomplete inference run.
Saved pixel/scientific and visual reviews are separate prerequisites.
"""
import hashlib
import json
import math
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_tail_preparation'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_tail_run_v1'
AUDIT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_tail_metadata_audit'
PLAN_SHA = '299a819c3016a2f338b75e1023feeb3f4598333cd2a08f1e3b4d9e6d946da694'
ARMS = ('incumbent_U96', 'postfusion_LR_u64')
OUTPUT_KEYS = ('native512', 'logits', 'LQ', 'unclipped', 'raw', 'png')


def main():
    checks = 0
    cache = {}

    def need(ok, message):
        nonlocal checks
        checks += 1
        if not ok:
            raise AssertionError(message)

    def load(path):
        path = Path(path)
        need(path.stat().st_size <= 2 * 1024 ** 2, 'Bounded metadata JSON')
        def pairs(rows):
            value = {}
            for key, item in rows:
                need(key not in value, 'Duplicate JSON key')
                value[key] = item
            return value
        def bad(value):
            raise ValueError('Nonfinite token ' + value)
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=bad)
        def finite(v):
            if isinstance(v, float):
                need(math.isfinite(v), 'Finite exponent')
            elif isinstance(v, dict):
                for x in v.values():
                    finite(x)
            elif isinstance(v, list):
                for x in v:
                    finite(x)
        finite(value)
        return value

    def pin(path):
        path = Path(path)
        name = path.relative_to(ROOT).as_posix()
        if name in cache:
            return cache[name]
        info = path.lstat()
        need(path.resolve().is_relative_to(ROOT) and stat.S_ISREG(info.st_mode)
             and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary workspace file')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 ** 2), b''):
                digest.update(block)
        value = {'path': name, 'bytes': info.st_size, 'sha256': digest.hexdigest()}
        cache[name] = value
        return value

    def bound(row):
        got = pin(ROOT / row['path'])
        need(got == {k: row[k] for k in ('path', 'bytes', 'sha256')}, 'Exact pin ' + row['path'])
        return got

    plan_path = PREP / 'tail_plan_v1.json'
    plan = load(plan_path)
    need(pin(plan_path)['sha256'] == PLAN_SHA, 'Exact sealed tail plan')
    for key in ('original_plan', 'original_failed_source', 'original_partial_metadata',
                'original_failure', 'original_pending', 'original_status', 'original_supervisor',
                'carried_admission', 'tail_source_review', 'tail_source', 'head_file'):
        bound(plan[key])
    for row in plan['original454files']:
        bound(row)
    original = load(ROOT / plan['original_plan']['path'])
    for row in [original['cohort'], original['data_reuse'], original['protocol'],
                *original['sources'], *original['prerequisites'].values(), *plan['model'].values()]:
        bound(row)
    admission = load(ROOT / plan['carried_admission']['path'])
    need(admission['complete'] and admission['carried47_pixel_admission_pass']
         and plan['carried_outputs'] == admission['carried_outputs']
         and len(plan['carried_outputs']) == 47, 'Exact independently admitted carried47')
    oldroot = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_run_v1'
    need({r['path'] for r in plan['original454files']} ==
         {p.relative_to(ROOT).as_posix() for p in oldroot.rglob('*') if p.is_file()}
         and len(plan['original454files']) == 454, 'Original454 exact immutable namespace')
    full = [{'case_id': c['id'], 'case_index': i, 'arm': arm, 'arm_index': j,
             'aggregate_output_index': 2 * i + j + 1}
            for i, c in enumerate(original['cases']) for j, arm in enumerate(ARMS)]
    need(plan['tail_slots'] == full[47:] and len(full) == 136, 'Exact remaining89 sequence')
    need(plan['all68_case_order'] == original['case_order'], 'Exact68 case order')
    need(not (OUT / 'failure.json').exists(), 'No terminal tail failure')
    comparison = load(OUT / 'comparison.json')
    supervisor = load(PREP / 'tail_supervisor_v1.json')
    need(supervisor['complete'] and supervisor['worker_exit'] == 0 and not supervisor['timeout']
         and supervisor['complete89_tail_observed'] and not supervisor['tail_forward_outcome_unknown']
         and not supervisor['partial_journal_files'] and supervisor['failure_receipt'] is None
         and not supervisor['failure_receipt_unreadable'], 'Authoritative known completed tail89')
    need(supervisor['original_unknown_reservation_retained'] and comparison['original_unknown_reservation_retained'],
         'Historical48th reservation uncertainty retained')
    need(comparison['complete'] and comparison['status'] ==
         'AGGREGATE_FIXED68_SAVED_OUTPUTS_PENDING_INDEPENDENT_AUDIT_VISUAL_REVIEW'
         and comparison['no_optimization'] and comparison['paired_and_native_evidence_separate']
         and not comparison['model_qualified'] and not comparison['app_default_changed'], 'No scientific promotion')
    need(comparison['tail_plan'] == pin(plan_path) and comparison['original_plan'] == plan['original_plan']
         and supervisor['tail_plan_sha256'] == PLAN_SHA, 'All terminal plan joins')
    need(comparison['new_saved_outputs'] == 89 and comparison['carried_saved_outputs'] == 47
         and comparison['aggregate_returned_outputs'] == 136
         and comparison['aggregate_reservations_including_failed_pre_call'] == 137, 'Actual aggregate counts')
    need(comparison['carried_output_records'] == [r['record'] for r in plan['carried_outputs']], 'All carried records exact')
    started = load(OUT / 'started.json')
    need(started['plan'] == pin(plan_path) and started['runner'] == plan['tail_source']
         and started['carried47_admission'] == plan['carried_admission'], 'Started binding')
    controls = load(OUT / 'controls_zero_calls.json')
    need(controls == {'complete': True, 'controls': original['zero_call_controls'], 'neural_calls': 0}
         and len(controls['controls']) == 40, 'Exact40 zero-call controls')
    counter = {'forward_attempts': 0, 'forward_returns': 0, 'saved_outputs': 0,
               'per_arm': {a: {'attempts': 0, 'returns': 0, 'saved': 0} for a in ARMS},
               'carried_saved_outputs': 47, 'completed_tail_cases': 0, 'optimizer_updates': 0,
               'gradient_queries': 0, 'backward_calls': 0, 'optimizer_constructions': 0}
    expected = []
    def state(label, slot):
        expected.append((label, slot, json.loads(json.dumps(counter))))
    state('claimed_before_Torch', None)
    state('before_model_load', None)
    returned_counters = {}
    for slot in plan['tail_slots']:
        arm = slot['arm']
        state('before_reservation', slot)
        counter['forward_attempts'] += 1
        counter['per_arm'][arm]['attempts'] += 1
        state('forward_reserved', slot)
        counter['forward_returns'] += 1
        counter['per_arm'][arm]['returns'] += 1
        state('forward_returned', slot)
        returned_counters[(slot['case_id'], arm)] = json.loads(json.dumps(counter))
        counter['saved_outputs'] += 1
        counter['per_arm'][arm]['saved'] += 1
        state('output_fully_saved', slot)
        if arm == ARMS[1]:
            counter['completed_tail_cases'] += 1
            state('case_complete', slot)
    state('complete', plan['tail_slots'][-1])
    journals = sorted((OUT / 'journal').glob('*.json'))
    need(len(journals) == len(expected) == 404, 'Exact404 durable journal transitions')
    previous_seconds = 0
    for seq, (path, (label, slot, cnt)) in enumerate(zip(journals, expected), 1):
        row = load(path)
        need(path.name == f'{seq:06d}.json' and row['sequence'] == seq and row['label'] == label
             and row['slot'] == slot and row['tail_counters'] == cnt, 'Independent ordered journal counters')
        need(row['tail_plan_sha256'] == PLAN_SHA and row['original_unknown_reservation_retained']
             and row['complete'] == (label == 'complete')
             and row['tail_forward_outcome_unknown'] == (cnt['forward_attempts'] != cnt['forward_returns'])
             and row['aggregate_returned_outputs'] == 47 + cnt['forward_returns']
             and row['aggregate_reservations_including_failed_pre_call'] == 48 + cnt['forward_attempts']
             and not row['model_qualified'], 'Journal state truth')
        need(previous_seconds <= row['seconds'] <= plan['limits']['internal_seconds'], 'Finite monotonic journal time')
        previous_seconds = row['seconds']
        pin(path)
    need(counter == comparison['tail_counters'] == supervisor['tail_counters'] ==
         supervisor['last_durable_tail_counters'] and supervisor['failure_tail_counters'] is None,
         'Final89 counters all terminal joins')
    need(supervisor['last_durable_status'] == pin(journals[-1]), 'Last immutable journal pin')
    prefix = OUT.relative_to(ROOT).as_posix() + '/'
    expected_files = {prefix + name for name in ('started.json', 'controls_zero_calls.json', 'comparison.json')}
    expected_files.update(p.relative_to(ROOT).as_posix() for p in journals)
    need([c['id'] for c in comparison['cases']] == original['case_order'], 'Exact68 aggregate case order')
    for i, (spec, case) in enumerate(zip(original['cases'], comparison['cases'])):
        need(case['complete'] and case['id'] == spec['id'] and case['input_review'] == spec['input_review']
             and case['geometry'] == spec['geometry'] and case['paired_target'] == spec['target']
             and case['cohort'] == spec['cohort'] and case['split'] == spec['split']
             and case['appearance_flag'] == spec.get('appearance_flag', False)
             and case['native_PSNR_SSIM_identity_metrics'] is None, 'Labels/geometry/paired-only claims unchanged')
        case_root = oldroot if i < 23 else OUT
        case_path = case_root / 'cases' / case['id'] / 'case.json'
        need(load(case_path) == case, 'Individual/aggregate exact case JSON')
        pin(case_path)
        if i >= 23:
            expected_files.add(case_path.relative_to(ROOT).as_posix())
        for q in case['inputs'].values():
            got = bound(q)
            if got['path'].startswith(prefix):
                expected_files.add(got['path'])
        need(set(case['outputs']) == set(ARMS), 'Both arms per case')
        for j, arm in enumerate(ARMS):
            out = case['outputs'][arm]
            need(set(out) == set(OUTPUT_KEYS) | {'provenance'}, 'Exact six saved output types')
            provenance = {'arm': arm, 'R2_file_sha256': plan['model']['r2']['sha256'],
                'U96_file_sha256': plan['model']['u96']['sha256'],
                'head_file_sha256': plan['head_file']['sha256'] if j else None,
                'head_canonical_sha256': original['candidate_head_canonical_sha256'] if j else None,
                'w': 1., 'detach16': True, 'AdaIN': True, 'CPU_FP32': True, 'display_processing': 'none',
                'source_candidate_training_updates': 64 if j else None, 'model_qualified': False,
                'same_observed_geometry': True, 'outside_support_exact_observed_pixels': True,
                'forward_sites': ['32', '64', '128', '256']}
            need(out['provenance'] == provenance, 'Exact unchanged model/delivery provenance')
            carried = 2 * i + j < 47
            record_root = oldroot if carried else OUT
            record_path = record_root / 'cases' / case['id'] / arm / 'record.json'
            record = load(record_path)
            need(record['complete'] and record['case_id'] == case['id'] and
                 {key: record[key] for key in out} == out, 'Exact individual output receipt join')
            pin(record_path)
            if not carried:
                expected_files.add(record_path.relative_to(ROOT).as_posix())
                need(record['tail_counters'] == returned_counters[(case['id'], arm)] and
                     record['slot'] == full[2 * i + j] and record['tail_plan_sha256'] == PLAN_SHA,
                     'Saved89 record exact preincrement counter/slot')
            else:
                old = plan['carried_outputs'][2 * i + j]
                need(old['record'] == pin(record_path) and old['outputs'] ==
                     {key: out[key] for key in OUTPUT_KEYS}, 'Exact old47 file association')
            for key in OUTPUT_KEYS:
                got = bound(out[key])
                if not carried:
                    need(got['path'].startswith(prefix), 'New89 output exclusively tail-owned')
                    expected_files.add(got['path'])
    resource_paths = sorted((OUT / 'resources').glob('*.json'))
    need(len(resource_paths) == 180, 'Before-load +178 forward boundary +final resource samples')
    samples = []
    previous_seconds = 0
    lim = plan['limits']
    for i, path in enumerate(resource_paths, 1):
        row = load(path)
        need(path.name == f'{i:06d}.json' and row['sequence'] == i, 'Exact resource numbering')
        need(previous_seconds <= row['seconds'] <= lim['internal_seconds']
             and row['worker_RSS_bytes'] <= lim['maximum_worker_RSS_bytes']
             and row['available_RAM_bytes'] >= lim['minimum_available_RAM_bytes']
             and row['free_disk_bytes'] >= lim['minimum_free_disk_bytes']
             and row['output_bytes'] <= lim['maximum_output_bytes'], 'Every saved resource cap')
        previous_seconds = row['seconds']
        samples.append(row)
        expected_files.add(pin(path)['path'])
    need(samples[0]['label'] == 'before_model_load' and samples[-1]['label'] == 'final', 'Resource endpoints')
    actual_paths = {p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    need(actual_paths == expected_files, 'Exact completed tail namespace; no hidden retry/pending files')
    files = [pin(ROOT / name) for name in sorted(actual_paths)]
    size = sum(q['bytes'] for q in files)
    need(size <= lim['maximum_output_bytes'] and comparison['seconds'] <= lim['internal_seconds']
         and supervisor['seconds'] <= lim['external_seconds'], 'Final actual output/time caps')
    receipt = {'complete': True, 'independent_terminal_metadata_audit_pass': True,
        'checks': checks, 'auditor': pin(Path(__file__).resolve()), 'tail_plan': pin(plan_path),
        'original_plan': plan['original_plan'], 'carried_admission': plan['carried_admission'],
        'comparison': pin(OUT / 'comparison.json'), 'supervisor': pin(PREP / 'tail_supervisor_v1.json'),
        'terminal_journal': pin(journals[-1]), 'tail_counters': counter,
        'original454files_unchanged': True, 'original_failed_reservation_retained': True,
        'original_forward_unknown_retained': True, 'tail_forward_unknown': False,
        'aggregate_saved_outputs': 136, 'aggregate_returns': 136, 'aggregate_reservations': 137,
        'paired_filter_cases': 36, 'unpaired_native_cases': 32, 'zero_call_controls': 40,
        'append_only_journal_count': 404, 'resource_samples': len(samples),
        'worker_seconds': comparison['seconds'], 'supervisor_seconds': supervisor['seconds'],
        'peak_sampled_RSS_bytes': max(r['worker_RSS_bytes'] for r in samples),
        'minimum_sampled_available_RAM_bytes': min(r['available_RAM_bytes'] for r in samples),
        'minimum_sampled_free_disk_bytes': min(r['free_disk_bytes'] for r in samples),
        'tail_file_count': len(files), 'tail_file_bytes': size, 'tail_files': files,
        'saved_pixel_audit_performed_by_this_script': False, 'model_qualified': False,
        'main_app_default_changed': False, 'optimization_performed': False,
        'scope': 'Saved-only terminal namespace, receipt, counter, provenance and resource readback. '
                 'Carried47 saved pixels are bound to prior independent admission; all136 pixel '
                 'metrics and all17 visual sheets require separate current review. '
                 'Observed resource samples do not prove an unsampled continuous peak.',
        'neural_Torch_optimizer_VM_HTTP_browser_calls': 0}
    AUDIT.mkdir(exist_ok=True)
    path = AUDIT / 'independent_terminal_metadata_v1.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'complete': True, 'metadata_pass': True, 'checks': checks, 'receipt': pin(path),
                      'actual_saved_outputs': 136, 'aggregate_reservations': 137, 'model_qualified': False}))


if __name__ == '__main__':
    main()
