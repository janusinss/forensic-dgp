"""Independent data-only sealed plan/pin readback; never imports model runtimes."""
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_DEV_preparation'
PLAN = PREP / 'broader_preservation_plan_v1.json'
SHA = '33a95897ed91c7f59a0704de65410f6093dd54a35f13a91933113775d8703c34'
ARMS = ['incumbent_U96', 'postfusion_LR_u64']
PINS = {}
CHECKS = 0


def need(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    name = path.relative_to(ROOT).as_posix()
    if name in PINS:
        return PINS[name]
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
         not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary pinned file: ' + name)
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            h.update(block)
    after = path.stat()
    need((info.st_size, info.st_mtime_ns, info.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Stable file: ' + name)
    PINS[name] = {'path': name, 'bytes': info.st_size, 'sha256': h.hexdigest()}
    return PINS[name]


def bound(row):
    name = row['path']
    rel = PurePosixPath(name)
    need(name == rel.as_posix() and not rel.is_absolute() and '\\' not in name and ':' not in name and
         all(x not in ('', '.', '..') for x in name.split('/')), 'Canonical direct pin')
    need(pin(name) == {k: row[k] for k in ('path', 'bytes', 'sha256')}, 'Exact direct pin: ' + name)
    return ROOT / name


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def memory():
    class Status(ctypes.Structure):
        _fields_ = [('size', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    *[(x, ctypes.c_ulonglong) for x in ('total', 'available', 'page', 'available_page',
                                                       'virtual', 'available_virtual', 'extended')]]
    status = Status(); status.size = ctypes.sizeof(status)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {'UTC': datetime.now(timezone.utc).isoformat(), 'method': 'Windows GlobalMemoryStatusEx',
            'RAM_query_success': bool(ok), 'total_RAM_bytes': status.total if ok else None,
            'available_RAM_bytes': status.available if ok else None,
            'workspace_free_disk_bytes': shutil.disk_usage(ROOT).free,
            'availability_is_not_launch_authorization_or_concurrency_proof': True}


def main():
    destination = PREP / 'independent_sealed_plan_readback_v1.json'
    need(not destination.exists(), 'Preserve earlier readback')
    plan_pin = pin(PLAN); need(plan_pin['sha256'] == SHA and plan_pin['bytes'] == 4698738, 'Exact sealed plan')
    plan = read(PLAN)
    need(plan['complete'] and plan['status'] == 'SEALED_PREPARED_UNRUN', 'Prepared plan status is historical readiness only')
    for row in [*plan['sources'], *plan['prerequisites'].values(), plan['evaluation_bundle'], plan['data_audit'],
                plan['teacher_reference_audit'], plan['independent_runner_source_review'], plan['protocol'],
                plan['head_file'], *plan['model'].values(), plan['prior_fixed68_plan'], plan['prior_aggregate_comparison']]:
        bound(row)
    peer = read(bound(plan['independent_runner_source_review']))
    need(peer['complete'] and peer['source_review_pass'], 'Actual source peer pass')
    for row in peer['sources']:
        bound(row)
    need(any(r['sha256'] == '9739c49f0a34ccc62f31968cf3b6d956bef08755d04d3cc9ac41518fc6d096ae'
             for r in peer['sources']), 'Held reviewed runner9739c49f')
    data = read(bound(plan['data_audit'])); teacher = read(bound(plan['teacher_reference_audit']))
    bundle = read(bound(plan['evaluation_bundle']))
    need(data['complete'] and data['independent_broader_data_audit_pass'] and
         teacher['complete'] and teacher['independent_teacher_reference_audit_pass'], 'Real data/teacher receipts')
    for row in data['verified_file_pins']:
        bound(row)
    for row in teacher['sources']:
        bound(row)
    for row in teacher['rows']:
        for entry in row['teachers'].values():
            bound(entry['png'])
    history = bundle['cases'][:406]
    paired = [r for r in history if r['evidence'] == 'paired_synthetic']
    native = [r for r in history if r['evidence'] == 'native_unpaired']
    need(plan['historical_cases'] == history and plan['historical_case_order'] == data['historical_case_order'] ==
         [r['id'] for r in history] and len(history) == 406 and len(paired) == 358 and len(native) == 48,
         'Exact406 historical membership/order')
    need(plan['paired_cases'] == paired and [r['id'] for r in teacher['rows']] == [r['id'] for r in paired], 'Ordered actual358 paired/teacher cases')
    need(all(r['split'] == 'DEV' and not r['reserved_FINAL'] and not r['optimization_permitted_now'] for r in history),
         'No FINAL/TRAIN optimization case admitted')
    expected_slots = [{'case_id': r['id'], 'paired_case_index': i, 'arm': arm,
                       'arm_index': j, 'new_output_index': 2 * i + j + 1}
                      for i, r in enumerate(paired) for j, arm in enumerate(ARMS)]
    need(plan['forward_slots'] == expected_slots and len(expected_slots) == 716, 'Exactly716 new model-call slots')
    need(Counter(x['arm'] for x in expected_slots) == dict.fromkeys(ARMS, 358), 'Exactly358 each arm')
    for row in paired:
        for field in ('input', 'target', 'support'):
            if row[field] is not None:
                bound(row[field])
    by_teacher = {r['id']: r for r in teacher['rows']}
    for kind, key in [('original', 'original_teacher_bank'), ('r2', 'R2_dark_reference_bank')]:
        bank = plan[key]
        need(set(bank) == {r['id'] for r in history} and len(bank) == 406, 'Full genuine reference membership')
        for row in history:
            actual = bank[row['id']]
            need(all(actual[k] == row[k] for k in ('id', 'source', 'profile', 'evidence')), 'Unchanged reference labels')
            expected = by_teacher[row['id']]['teachers'][kind]['retained_recorded_metrics'] if row in paired else None
            need(actual['metrics'] == expected, 'Exact recorded worker RAW and PNG references; no future candidate metrics')
    comparison = read(bound(plan['prior_aggregate_comparison']))
    old = {r['id']: r for r in comparison['cases']}
    need(len(plan['native_carry32']) == 32 and len(plan['native_excluded16']) == 16, '32carry16excluded native')
    for carried in plan['native_carry32']:
        case = carried['case']; need(case == old[carried['fixed_id']] and carried['new_forward_calls'] == 0,
                                    'Exact prior native case and zero new calls')
        for arm in ARMS:
            for key in ('native512', 'logits', 'LQ', 'unclipped', 'raw', 'png'):
                bound(case['outputs'][arm][key])
            provenance = case['outputs'][arm]['provenance']
            need(provenance['R2_file_sha256'] == plan['model']['r2']['sha256'] and
                 provenance['U96_file_sha256'] == plan['model']['u96']['sha256'] and
                 provenance['head_file_sha256'] == (plan['head_file']['sha256'] if arm == ARMS[1] else None),
                 'Carried full frozen model provenance')
    need(plan['native_excluded16'] == data['native16_excluded_zero_call_controls'] and
         all(x['model_outputs'] is None and x['reference_metrics'] is None and x['neural_calls_allowed'] == 0
             for x in plan['native_excluded16']), 'All16 explicit no-output/no-call exclusions')
    represented = {r['historical_id'] for r in plan['native_carry32']} | {r['historical_id'] for r in plan['native_excluded16']}
    need(represented == {r['id'] for r in native} and len(represented) == 48, 'All48 native members accounted without fabricated outputs')
    for key, value in {'membership_rows': 406, 'fresh_paired_cases': 358, 'fresh_output_slots': 716,
                       'carried_native_output_slots': 64, 'native_excluded_zero_call_rows': 16,
                       'represented_output_slots': 780, 'CPU_threads': 4, 'preparation_neural_calls': 0}.items():
        need(plan[key] == value, 'Honest count: ' + key)
    need(all(plan[k] is False for k in ('optimization_permitted', 'automatic_retry_resume',
         'app_default_changed', 'model_qualified', 'FINAL_used')), 'No training/qualification/default/FINAL permission')
    need(len(plan['planned_paired_visual_sheets']) == 90 and
         sum(r['generated_PNG_cells'] for r in plan['planned_paired_visual_sheets']) == 716 and
         [k for s in plan['planned_paired_visual_sheets'] for k in s['case_ids']] == [r['id'] for r in paired],
         'All90 planned paired sheets/716 output cells')
    visual = read(bound(plan['prerequisites']['fixed_DEV_root_visual_review']))
    need(visual['complete'] and visual['root_personal_visual_review_complete'] and visual['sheets_reviewed'] == 17 and
         visual['case_rows_reviewed'] == 68 and visual['generated_PNG_cells_reviewed'] == 136, 'Current root all17 visual closure')
    for review in visual['reviews']:
        need(review['personally_viewed_by_root'], 'Root review record'); bound(review['file'])
    witness = read(bound(plan['prerequisites']['CPU_runtime_witness']))
    accept = read(bound(plan['prerequisites']['CPU_runtime_root_acceptance']))
    need(witness['complete'] and witness['runtime_integrity_pass'] and witness['CPU_threads'] == 4 and
         witness['forward_calls'] == 4 and witness['CPU_RNG_preserved'] and not witness['CUDA_initialized'] and
         witness['frozen_invariants_before'] == witness['frozen_invariants_after'] and
         witness['frozen_invariants_after']['frozen_tensors'] == 575, 'Actual frozen CPU4 witness')
    need(accept['complete'] and accept['runtime_readback_pass'] and accept['accepted_for_fixed_DEV_inference'] and
         not accept['training_quality_tolerance_changed'] and not accept['accepted_for_model_promotion'], 'Root measured CPU/L4 readback acceptance only')
    bound(accept['independent_readback'])
    need(plan['head_file']['sha256'] == '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17' and
         plan['candidate_head_canonical_sha256'] == witness['frozen_invariants_after']['head_canonical_sha256'] ==
         '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be', 'Terminal36head file/canonical pins')
    expected_limits = {'full_forward_attempts': 716, 'full_forward_returns': 716,
        'per_arm_forward_attempts': dict.fromkeys(ARMS, 358), 'per_arm_forward_returns': dict.fromkeys(ARMS, 358),
        'internal_seconds': 14400, 'external_seconds': 14460, 'individual_forward_watchdog_seconds': 120,
        'model_load_watchdog_seconds': 240, 'maximum_output_bytes': 6442450944,
        'minimum_free_disk_bytes': 8589934592, 'maximum_worker_RSS_bytes': 6442450944,
        'minimum_available_RAM_bytes': 536870912}
    need(plan['limits'] == expected_limits, 'Exact strict reviewed caps')
    source = (ROOT / 'scripts/run_u96_postfusion_lr_broader_preservation_v1.py').read_text()
    need('M.sum = ordered_sum' in source and 'os.replace' not in source, 'Isolated ordered gate arithmetic and append-only writes')
    need('retained_recorded_metrics' in plan['teacher_metric_policy'] and
         'without modifying retained source/module' in source, 'Exact worker references and isolated reduction')
    resources = memory()
    # These are observations only, never proof that another process has stopped.
    resources['meets_sealed_min_available_RAM_at_sample'] = resources['RAM_query_success'] and resources['available_RAM_bytes'] >= expected_limits['minimum_available_RAM_bytes']
    resources['meets_root_separate_2p5GiB_preload_guard_at_sample'] = resources['RAM_query_success'] and resources['available_RAM_bytes'] >= 2684354560
    resources['meets_sealed_min_free_disk_at_sample'] = resources['workspace_free_disk_bytes'] >= expected_limits['minimum_free_disk_bytes']
    need('torch' not in sys.modules and 'numpy' not in sys.modules and 'PIL' not in sys.modules, 'No runtime/pixel decode imports')
    pin_source = pin(Path(__file__))
    # Independently re-read sealed bytes at the end, bypassing the hash cache.
    need(hashlib.sha256(PLAN.read_bytes()).hexdigest() == SHA, 'Sealed plan unchanged after readback')
    receipt = {'format': 'U96-LR-broader-preservation-independent-sealed-plan-readback-v1',
        'complete': True, 'independent_sealed_plan_readback_pass': True, 'plan': plan_pin, 'source': pin_source,
        'checks_passed': CHECKS, 'current_byte_pins_verified': list(PINS.values()),
        'counts': {'historical_membership': 406, 'fresh_paired_cases': 358, 'new_model_slots': 716,
                   'new_slots_each_arm': 358, 'carried_native_cases': 32, 'carried_native_outputs': 64,
                   'excluded_native_zero_call': 16, 'represented_outputs': 780, 'planned_paired_sheets': 90},
        'nominal_complete_run_resource_checks': 1434,
        'nominal_resource_check_derivation': '1before_model_load +716before_forward +716after_save +1final',
        'nominal_complete_run_status_journal_rows': 3225,
        'nominal_status_derivation': '1claimed +1before_model_load +4*716slot events +358case_complete +1complete_measurement',
        'strict_limits': expected_limits, 'resource_observation': resources,
        'current_CPU_witness_and_root_acceptance_bound': True, 'actual_root17_sheet_closure_and_publication_bound': True,
        'recorded_worker_RAW_PNG_teacher_refs_exact_no_new_tolerance': True,
        'reference_RAW_limitation': '716historical teacher RAW files absent, retained recorded scalar references remain supported by separately pinned independent conversion/import/state proofs; current PNG corroboration is not a new baseline.',
        'native_references_null_not_false406_inference_completion': True,
        'reserved_FINAL_image_or_identity_review': False, 'image_or_array_decodes': 0,
        'no_new_training': True, 'model_imports_or_forward_calls': 0, 'gradient_optimizer_training_VM_browser_calls': 0,
        'plan_changes': 0, 'launch_claimed_or_attempted': False,
        'fresh_runtime_success_or_gate_pass_established': False, 'model_qualified': False, 'app_default_changed': False}
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'receipt': pin(destination), 'checks': CHECKS, 'resource_observation': resources}))


if __name__ == '__main__':
    main()
