"""Source-reviewed finite broader DEV preservation comparison; inert by default.

716 new frozen CPU forwards, 64 exact prior native outputs and 16 explicit
zero-call exclusions retain 406 historical membership rows. No training,
automatic retry/resume, FINAL inspection, context blur or app promotion.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import run_u96_smoothanchors_lr_fixed_DEV_v1 as B

ROOT = B.ROOT
PREP = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_DEV_preparation'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_lr_broader_preservation_v1'
PLAN = PREP / 'broader_preservation_plan_v1.json'
DATA_AUDIT = PREP / 'independent_broader_cohort_data_audit_v1.json'
BUNDLE = ROOT / 'outputs/codeformer_filter_fusion_training_v1_preparation/evaluation_bundle_v1.json'
FIXED_PLAN = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_preparation/fixed_DEV_plan_v1.json'
FIXED_TAIL = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_tail_run_v1'
FIXED_AUDIT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_aggregate_audit_v1'
ARMS = B.ARMS
METRICS_PATH = 'research/codeformer_filter_fusion_training_v1/metrics.py'
METRICS_SHA = '3e9a129f41c7b62e2b7cda89b2bf05b7cf8478930b50ce5611a2c17ef741107c'
BUNDLE_SHA = '1fd8feba1fb8f7475b3022d8e906da59d7a979525d1b0a10e2e200603674b72b'
DATA_SHA = 'b6961301272263720c148642b63e5c94a8b1b766b39964bb703b600d902ef2dd'
TEACHER_SHA = 'e9619d162ee5baf8a50e209587056f112b6bec8b28e6be6eeb37de3339c0c1e1'
TEACHER_DELIVERY_SHA = 'df0e164cc386ad288e60f30b8efc48e811654b2de39bf13dcd2ead7f8fb551f1'
FIXED_SHA = 'bb8a60f29630ddb21a2e9cea7e43b30c689b78f6474e6671a28c516e21bf1a6d'
OWN_TEST = 'tests/test_u96_postfusion_lr_broader_preservation_v1.py'
LIMITS = {'full_forward_attempts': 716, 'full_forward_returns': 716,
          'per_arm_forward_attempts': {a: 358 for a in ARMS},
          'per_arm_forward_returns': {a: 358 for a in ARMS},
          'internal_seconds': 14400, 'external_seconds': 14460,
          'individual_forward_watchdog_seconds': 120, 'model_load_watchdog_seconds': 240,
          'maximum_output_bytes': 6 * 1024 ** 3, 'minimum_free_disk_bytes': 8 * 1024 ** 3,
          'maximum_worker_RSS_bytes': 6 * 1024 ** 3, 'minimum_available_RAM_bytes': 512 * 1024 ** 2}
OUTPUT_KEYS = ('native512', 'logits', 'LQ', 'unclipped', 'raw', 'png')
NULL_FIELDS = ('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'signed_luminance_bias',
               'coarse_rgb_mae', 'adjacent_rgb_error')
need, pin, bound, load = B.need, B.pin, B.bound, B.load


def blank_native_metrics():
    """Metadata-only nulls required by legacy banks; never invented images."""
    return None


def ordered_sum(values, start=0):
    """Match the historical Python3.10 left-to-right scalar gate reduction."""
    total = start
    for value in values:
        total += value
    return total


def slots(paired):
    return [{'case_id': row['id'], 'paired_case_index': i, 'arm': arm,
             'arm_index': j, 'new_output_index': 2 * i + j + 1}
            for i, row in enumerate(paired) for j, arm in enumerate(ARMS)]


def sheet_plan(paired):
    answer = []
    for source in dict.fromkeys(row['source'] for row in paired):
        ids = [row['id'] for row in paired if row['source'] == source]
        for start in range(0, len(ids), 4):
            answer.append({'source': source, 'case_ids': ids[start:start + 4],
                           'rows': len(ids[start:start + 4]), 'exact_image_size': [256, 256],
                           'columns': ['input', 'target', *ARMS],
                           'generated_PNG_cells': 2 * len(ids[start:start + 4])})
    return answer


def require_teacher(teacher, historical):
    need(teacher['complete'] and teacher['independent_teacher_reference_audit_pass'],
         'Actual independently verified saved teacher receipt required')
    paired = [r for r in historical if r['evidence'] == 'paired_synthetic']
    need(len(teacher['rows']) == 358 and [r['id'] for r in teacher['rows']] == [r['id'] for r in paired],
         'Teachers exact ordered358 historical paired cases')
    for row, spec in zip(teacher['rows'], paired):
        need(all(row[k] == spec[k] for k in ('id', 'source', 'profile', 'evidence'))
             and set(row['teachers']) == {'original', 'r2'}, 'Original/R2 teacher identity and role')
        for entry in row['teachers'].values():
            need(set(entry['retained_recorded_metrics']) == {'raw', 'native_png'}, 'Both historical reference views required')
            # Historical RAW files can be offloaded. This is a sealed scalar
            # reference, bound to the prior exact-conversion/import receipts.
            # Current PNG readback corroborates it; no baseline is recomputed
            # or replaced by a new image/model inference here.
            for view in ('raw', 'native_png'):
                need(set(entry['retained_recorded_metrics'][view]) ==
                     {'mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'coarse_rgb_mae'},
                     'Exact six original worker metric fields; no invented teacher measurements')
    expected = {
        'clear61': sorted(r['id'] for r in paired if r['profile'] == 'clear'),
        'normal_blur53': sorted(r['id'] for r in paired if r['source'] == 'FFHQ validation' and r['profile'] == 'blur'),
        'lowlight53': sorted(r['id'] for r in paired if r['source'] == 'FFHQ validation' and r['profile'] == 'lowlight'),
        'compound53': sorted(r['id'] for r in paired if r['source'] == 'FFHQ validation' and r['profile'] == 'compound')}
    need([len(expected[k]) for k in ('clear61', 'normal_blur53', 'lowlight53', 'compound53')] == [61, 53, 53, 53],
         'Unchanged paired guard denominators')
    need(teacher['gate_case_ids'] == expected, 'Teacher exact guard case membership')
    need(teacher['gate_reference_policy']['no_reference_rebaselining_or_rounding'] is True and
         'retained_recorded_metrics.raw' in teacher['gate_reference_policy']['raw'] and
         'retained_recorded_metrics.native_png' in teacher['gate_reference_policy']['native_png'],
         'Explicit historical scalar references; current PNG audit is corroboration only')
    return paired


def reference_bank(historical, teacher_rows, kind):
    by_id = {r['id']: r for r in teacher_rows}
    need(kind in ('original', 'r2'), 'Declared reference arm')
    return {r['id']: {'id': r['id'], 'evidence': r['evidence'], 'source': r['source'],
                     'profile': r['profile'],
                     'metrics': by_id[r['id']]['teachers'][kind]['retained_recorded_metrics']
                     if r['evidence'] == 'paired_synthetic' else blank_native_metrics(),
                     'native_metadata_only_not_a_teacher_output': r['evidence'] == 'native_unpaired'}
            for r in historical}


def reviewed_sources(receipt, required):
    need(receipt['complete'] and receipt['source_review_pass'] and
         required <= {row['path'] for row in receipt['sources']}, 'Independent exact source review required')
    for row in receipt['sources']: bound(row)


def data_contract(data_path, teacher_path):
    need('torch' not in sys.modules, 'All preparation/data proof checks precede Torch import')
    need(pin(BUNDLE)['sha256'] == BUNDLE_SHA and pin(data_path)['sha256'] == DATA_SHA,
         'Exact historical evaluation bundle and fresh independent data audit')
    need(pin(teacher_path)['sha256'] == TEACHER_SHA, 'Exact actual independently audited historical teacher receipt')
    data, bundle, teacher = load(data_path), load(BUNDLE), load(teacher_path)
    need(data['complete'] and data['independent_broader_data_audit_pass'] and
         data['historical_membership_count'] == 406 and data['paired_synthetic_cases'] == 358 and
         data['native_unpaired_membership'] == 48 and not data['optimization_permitted'] and
         not data['reserved_FINAL_image_or_identity_review'], 'Data-only358/48 audit')
    historical = bundle['cases'][:406]
    need([r['id'] for r in historical] == data['historical_case_order'] and
         all(r['cohort'] == 'historical_DEV406' and r['split'] == 'DEV' and not r['reserved_FINAL'] and
             not r['optimization_permitted_now'] for r in historical), 'Exact406 original DEV membership/order')
    for row in data['verified_file_pins']: bound(row)
    for row in bundle['metadata_dependencies'].values(): bound(row)
    paired = require_teacher(teacher, historical)
    for row in teacher['sources']: bound(row)
    for row in teacher['rows']:
        for entry in row['teachers'].values(): bound(entry['png'])
    delivery_path = PREP / 'independent_teacher_delivery_source_readback_v1.json'
    need(pin(delivery_path)['sha256'] == TEACHER_DELIVERY_SHA, 'Exact actual historical teacher delivery-source readback')
    delivery = load(delivery_path)
    need(delivery['complete'] and delivery['independent_source_readback_pass'] and
         delivery['teacher_audit'] == pin(teacher_path) and delivery['no_new_gate_tolerance'] and
         all(delivery['v1_and_actual_R2_AST_exact'].values()), 'Actual teacher source/conversion readback')
    for row in delivery['actual_R2_sources_exact_export_members']: bound(row)
    bound(delivery['historical_base_delivery_source'])
    need(pin(ROOT / METRICS_PATH)['sha256'] == METRICS_SHA, 'Unchanged historical saved metrics/gates source')
    fixed = B.verify_plan(FIXED_SHA)
    need(pin(FIXED_PLAN)['sha256'] == FIXED_SHA, 'Exact prior fixed68 plan')
    prereq_paths = {
        'fixed_DEV_root_visual_review': FIXED_AUDIT / 'root_visual_review_v1.json',
        'fixed_DEV_saved_pixel_audit': FIXED_AUDIT / 'saved_pixel_audit_v1.json',
        'fixed_DEV_publication': FIXED_AUDIT / 'DEV_publication_v1.json',
        'fixed_DEV_terminal_metadata': ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_DEV_tail_metadata_audit/independent_terminal_metadata_v1.json'}
    prerequisites = {k: pin(p) for k, p in prereq_paths.items()}
    prerequisites['teacher_delivery_source_readback'] = pin(delivery_path)
    visual = load(prereq_paths['fixed_DEV_root_visual_review'])
    need(visual['complete'] and visual['root_personal_visual_review_complete'] and
         visual['sheets_reviewed'] == 17 and visual['case_rows_reviewed'] == 68 and
         visual['generated_PNG_cells_reviewed'] == 136 and not visual['model_qualified'],
         'Actual root all17/136 DEV visual closure required')
    for row in visual['reviews']:
        need(row['personally_viewed_by_root'], 'Root personally viewed fixed DEV sheet')
        bound(row['file'])
    bound(visual['visual_manifest']); bound(visual['pixel_audit']); bound(visual['metadata_audit'])
    pixel_audit = load(prereq_paths['fixed_DEV_saved_pixel_audit'])
    need(pixel_audit['complete'] and pixel_audit['independent_saved_pixel_integrity_pass'] and
         pixel_audit['outputs_verified'] == 136 and pixel_audit['all136_native_to_unclipped_RAW_floorPNG_exact']
         and pixel_audit['all136_outside_support_exact'], 'Actual136 saved-pixel integrity required')
    publication = load(prereq_paths['fixed_DEV_publication'])
    need(publication['complete'] and publication['root_visual_review'] == prerequisites['fixed_DEV_root_visual_review']
         and publication['saved_pixel_audit'] == prerequisites['fixed_DEV_saved_pixel_audit'] and
         not publication['model_qualified'] and not publication['goal_complete'], 'Published fixed DEV scope, not qualification')
    bound(publication['report'])
    metadata = load(prereq_paths['fixed_DEV_terminal_metadata'])
    need(metadata['complete'] and metadata['independent_terminal_metadata_audit_pass'] and
         metadata['aggregate_saved_outputs'] == 136 and metadata['original_forward_unknown_retained'],
         'Prior complete known136 saved outputs and old failure preserved')
    comparison = load(FIXED_TAIL / 'comparison.json')
    need(comparison['complete'] and len(comparison['cases']) == 68 and not comparison['model_qualified'], 'Prior actual aggregate136 result')
    current_native = {r['id']: r for r in comparison['cases'] if r['paired_target'] is None}
    need(len(current_native) == 32, 'Native32 prior saved cases')
    native_carry = []
    for row in data['native32_existing_output_mapping']:
        old = current_native[row['fixed_id']]
        need(old['geometry'] == row['geometry'] and old['input_review'] == row['input_only_review'] and
             old['inputs']['prepared']['sha256'] == row['input']['sha256'] and
             old['inputs']['support']['sha256'] == row['support']['sha256'], 'Exact prior native input/geometry/label')
        for q in old['inputs'].values(): bound(q)
        for arm in ARMS:
            need(old['outputs'][arm]['provenance']['R2_file_sha256'] == B.R2_SHA and
                 old['outputs'][arm]['provenance']['U96_file_sha256'] == B.U96_SHA and
                 old['outputs'][arm]['provenance']['head_file_sha256'] == (B.HEAD_SHA if arm == ARMS[1] else None),
                 'Exact previous native arm model/head provenance')
            for key in OUTPUT_KEYS: bound(old['outputs'][arm][key])
        native_carry.append({'historical_id': row['historical_id'], 'fixed_id': row['fixed_id'],
                             'case': old, 'status': 'CARRIED_SOURCE_BOUND_NATIVE_UNPAIRED', 'new_forward_calls': 0})
    need(len(native_carry) == 32 and len(data['native16_excluded_zero_call_controls']) == 16, 'Native32carry16excluded')
    for name, sha in B.STATIC.items(): need(pin(ROOT / name)['sha256'] == sha, 'Exact immutable model/input source')
    prerequisites.update({k: v for k, v in fixed['prerequisites'].items()
                          if k in ('CPU_runtime_witness', 'CPU_runtime_root_acceptance',
                                   'head_export', 'frozen_adapter_source_review', 'TRAIN_visual_closure')})
    return data, bundle, teacher, historical, paired, fixed, native_carry, prerequisites


def prepare(args):
    need(not PLAN.exists(), 'Fresh sealed plan; never overwrite/reselect')
    data_path = Path(args.data_audit).absolute(); teacher_path = Path(args.teacher_audit).absolute()
    data, bundle, teacher, historical, paired, fixed, carry, prerequisites = data_contract(data_path, teacher_path)
    source_review = pin(Path(args.source_review).absolute())
    reviewed_sources(load(bound(source_review)), {Path(__file__).resolve().relative_to(ROOT).as_posix(), OWN_TEST,
                                                  METRICS_PATH, 'scripts/run_u96_smoothanchors_lr_fixed_DEV_v1.py'})
    need(len(sheet_plan(paired)) == 90, '90 source-group paired sheets planned, at most4 rows')
    plan = {'complete': True, 'format': 'U96-terminal-LR-broader-preservation-CPU716-v1',
        'status': 'SEALED_PREPARED_UNRUN', 'evaluation_bundle': pin(BUNDLE), 'data_audit': pin(data_path),
        'teacher_reference_audit': pin(teacher_path), 'independent_runner_source_review': source_review,
        'prerequisites': prerequisites, 'model': fixed['model'], 'head_file': fixed['head_file'],
        'protocol': fixed['protocol'], 'source_archive_sha256': B.ARCHIVE_SHA,
        'candidate_head_canonical_sha256': B.HEAD_STATE, 'prior_fixed68_plan': pin(FIXED_PLAN),
        'prior_aggregate_comparison': pin(FIXED_TAIL / 'comparison.json'),
        'sources': [pin(Path(__file__).resolve()), pin(ROOT / OWN_TEST), pin(ROOT / METRICS_PATH),
                    pin(ROOT / 'scripts/run_u96_smoothanchors_lr_fixed_DEV_v1.py'),
                    *[pin(ROOT / name) for name in B.STATIC]],
        'historical_case_order': [r['id'] for r in historical], 'historical_cases': historical,
        'paired_cases': paired, 'forward_slots': slots(paired), 'native_carry32': carry,
        'native_excluded16': data['native16_excluded_zero_call_controls'],
        'membership_rows': 406, 'fresh_paired_cases': 358, 'fresh_output_slots': 716,
        'carried_native_output_slots': 64, 'represented_output_slots': 780,
        'native_excluded_zero_call_rows': 16, 'mixed72_and_refused24_excluded_from_this_phase': True,
        'original_teacher_bank': reference_bank(historical, teacher['rows'], 'original'),
        'R2_dark_reference_bank': reference_bank(historical, teacher['rows'], 'r2'),
        'teacher_metric_policy': 'Both RAW and PNG use exact retained_recorded_metrics from pinned prior workers. Current saved-PNG recomputation corroborates; no baseline replacement. RAW teacher arrays may be offloaded and are not represented as freshly decoded.',
        'gate_policy': 'Exact unchanged metrics.py preservation61clear/53blur limits0.01mean/0.03worst bothviews; dark53lowlight/53compound PNG meanMAE<=R2 and meanSSIM>=R2. No epsilon or quality threshold change.',
        'scalar_gate_reduction': 'Exact sorted-ID left-to-right Pythonfloat64 accumulation, preserving historicalPython3.10; isolated metrics source namespace binds ordered_sum instead of Python3.13 compensated builtins.sum.',
        'native_gate_representation': 'All48 remain metadata rows with null reference metrics;32source-bound saved-output cases,16explicitzero-call outputs:null. No invented native output or PSNR/SSIM.',
        'CPU_threads': 4, 'input_policy': 'Exact retainedRGB256 and binarysupport; no recrop/rescale/alignment/noise/contextblur; nullsupport means allpixels observed.',
        'delivery_policy': 'Same immutable fixedDEV/tail: RGB256/255→bilinear512*2-1; native unclipped and clipped before bilinear256; exact outside-support input RGB/255; floorRGB8.',
        'planned_paired_visual_sheets': sheet_plan(paired), 'native32_visuals': 'Already personally reviewed in original17sheets; bind proof; no new native output calls.',
        'limits': LIMITS, 'automatic_retry_resume': False, 'optimization_permitted': False,
        'app_default_changed': False, 'model_qualified': False, 'FINAL_used': False,
        'preparation_neural_calls': 0}
    PREP.mkdir(exist_ok=True)
    with PLAN.open('xb') as stream:
        stream.write(B.json_bytes(plan)); stream.flush(); os.fsync(stream.fileno())
    print(B.json_bytes({'complete': True, 'plan': pin(PLAN), 'new_neural_calls': 0, 'training_UNRUN': True}).decode(), end='')
    return 0


def verify(expected_sha):
    need('torch' not in sys.modules, 'Bounded plan verification precedes Torch')
    need(pin(PLAN)['sha256'] == expected_sha, 'Exact explicit sealed plan SHA')
    plan = load(PLAN)
    data, bundle, teacher, historical, paired, fixed, carry, prerequisites = data_contract(
        bound(plan['data_audit']), bound(plan['teacher_reference_audit']))
    need(plan['limits'] == LIMITS and plan['historical_cases'] == historical and plan['paired_cases'] == paired
         and plan['forward_slots'] == slots(paired) and plan['native_carry32'] == carry
         and plan['native_excluded16'] == data['native16_excluded_zero_call_controls']
         and plan['prerequisites'] == prerequisites and plan['model'] == fixed['model']
         and plan['head_file'] == fixed['head_file'] and plan['planned_paired_visual_sheets'] == sheet_plan(paired),
         'Exact data/order/geometry/teacher/native/resource contract')
    need(plan['original_teacher_bank'] == reference_bank(historical, teacher['rows'], 'original') and
         plan['R2_dark_reference_bank'] == reference_bank(historical, teacher['rows'], 'r2'), 'Unchanged historical scalar reference banks')
    for row in [*plan['sources'], *plan['prerequisites'].values(), plan['protocol'], plan['head_file'],
                *plan['model'].values(), plan['independent_runner_source_review'], plan['prior_aggregate_comparison']]: bound(row)
    reviewed_sources(load(bound(plan['independent_runner_source_review'])),
                     {Path(__file__).resolve().relative_to(ROOT).as_posix(), OWN_TEST, METRICS_PATH,
                      'scripts/run_u96_smoothanchors_lr_fixed_DEV_v1.py'})
    need(not plan['optimization_permitted'] and not plan['automatic_retry_resume'] and not plan['model_qualified']
         and not plan['app_default_changed'] and not plan['FINAL_used'], 'No optimization/promotion/FINAL')
    return plan


class Writer:
    """Owned exclusive writes with finite pre-reservation and no hidden retry."""
    def __init__(self, directory, cap):
        self.directory = Path(directory).resolve(); self.cap = cap
        self.reserved_bytes = 0; self.successful_bytes = 0; self.successful_files = 0
    def write(self, path, raw):
        path = Path(path)
        need(path.resolve().is_relative_to(self.directory), 'Owned worker write namespace')
        need(type(raw) is bytes and self.reserved_bytes + len(raw) <= self.cap, 'Reserve finite byte budget before writing')
        path.parent.mkdir(parents=True, exist_ok=True)
        for part in (path.parent, *path.parent.parents):
            info = part.lstat()
            need(not B.stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024, 'No reparse output parent')
        self.reserved_bytes += len(raw)
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self.successful_bytes += len(raw); self.successful_files += 1
        return pin(path)
    def json(self, path, value):
        return self.write(path, B.json_bytes(value))


class Journal:
    def __init__(self, writer, directory):
        self.writer = writer; self.directory = directory; self.sequence = 0
    def append(self, row):
        self.sequence += 1
        return self.writer.json(self.directory / f'{self.sequence:06d}.json', {'sequence': self.sequence, **row})


def supervisor_outcome(code, timeout, last, partial, failure, unreadable_failure, comparison):
    durable = last.get('counters') if last else None
    failed = failure.get('counters') if failure else None
    counters = failed if failed is not None else durable
    gate_rows = comparison.get('gates', {}).get(ARMS[1]) if isinstance(comparison, dict) else None
    gates_well_formed = bool(isinstance(gate_rows, dict) and
        set(gate_rows) == {'preservation_vs_original', 'dark_retention_vs_R2'} and
        all(isinstance(v, dict) and type(v.get('pass')) is bool for v in gate_rows.values()))
    measurement_complete = bool(code in (0, 2) and not timeout and not failure and not unreadable_failure
        and not partial and last and last.get('measurement_complete') and comparison and comparison.get('complete')
        and counters and counters['forward_attempts'] == counters['forward_returns'] == counters['saved_outputs'] == 716
        and counters['completed_paired_cases'] == 358 and comparison.get('counters') == counters
        and last.get('quality_status') == comparison.get('quality_status') and gates_well_formed)
    candidate_pass = bool(measurement_complete and all(
        row['pass'] for row in comparison['gates'][ARMS[1]].values()))
    quality_stop = bool(measurement_complete and not candidate_pass and code == 2
                        and comparison.get('quality_status') == 'QUALITY_STOP_NOT_QUALIFIED')
    known_pass = bool(measurement_complete and candidate_pass and code == 0
                     and comparison.get('quality_status') == 'BROAD_PRESERVATION_GATE_PASS_NOT_QUALIFIED')
    unknown = bool(partial or unreadable_failure or counters is None or
        (counters and counters['forward_attempts'] != counters['forward_returns']) or
        (last and last.get('forward_outcome_unknown')) or (failure and failure.get('forward_outcome_unknown')) or
        (not measurement_complete and failure is None))
    return {'last_durable_counters': durable, 'failure_counters': failed, 'counters': counters,
            'all716_measurements_complete': measurement_complete, 'quality_stop_recorded': quality_stop,
            'finite_preservation_gate_pass_not_qualification': known_pass, 'forward_outcome_unknown': unknown}


def worker(expected_sha):
    plan = verify(expected_sha)
    need(not OUT.exists(), 'Fresh exclusive output namespace; no retry/resume')
    OUT.mkdir(); writer = Writer(OUT, LIMITS['maximum_output_bytes'])
    journal = Journal(writer, OUT / 'journal'); resources = Journal(writer, OUT / 'resources')
    counters = {'forward_attempts': 0, 'forward_returns': 0, 'saved_outputs': 0, 'completed_paired_cases': 0,
        'per_arm': {a: {'attempts': 0, 'returns': 0, 'saved': 0} for a in ARMS},
        'carried_native_output_slots': 64, 'excluded_native_zero_call_rows': 16,
        'optimizer_updates': 0, 'backward_calls': 0, 'gradient_queries': 0, 'optimizer_constructions': 0}
    start = time.monotonic(); phase = 'claimed_before_Torch'; active = None
    measurement_complete = False; quality_status = None; last_status = None
    previous_handler = signal.getsignal(signal.SIGTERM)
    def termination(signum, frame): raise TimeoutError('Owned worker termination signal')
    def status(label):
        nonlocal last_status
        last_status = journal.append({'label': label, 'phase': phase, 'slot': active,
            'measurement_complete': measurement_complete, 'quality_status': quality_status,
            'counters': counters, 'forward_outcome_unknown': counters['forward_attempts'] != counters['forward_returns'],
            'seconds': time.monotonic() - start, 'plan_sha256': expected_sha, 'model_qualified': False,
            'no_automatic_retry_resume': True, 'prior_fixed68_original_unknown_evidence_retained': True})
    def resource(label):
        import psutil
        row = {'label': label, 'seconds': time.monotonic() - start,
            'worker_RSS_bytes': psutil.Process().memory_info().rss,
            'available_RAM_bytes': psutil.virtual_memory().available,
            'free_disk_bytes': shutil.disk_usage(ROOT).free,
            'reserved_output_bytes': writer.reserved_bytes, 'successful_output_bytes': writer.successful_bytes,
            'successful_output_files': writer.successful_files}
        resources.append(row)
        need(row['seconds'] <= LIMITS['internal_seconds'] and row['worker_RSS_bytes'] <= LIMITS['maximum_worker_RSS_bytes']
             and row['available_RAM_bytes'] >= LIMITS['minimum_available_RAM_bytes'] and
             row['free_disk_bytes'] >= LIMITS['minimum_free_disk_bytes'] and
             row['reserved_output_bytes'] <= LIMITS['maximum_output_bytes'], 'Finite worker resource cap')
    def watchdog(seconds):
        timer = threading.Timer(seconds, lambda: os._exit(124)); timer.daemon = True; timer.start(); return timer
    try:
        signal.signal(signal.SIGTERM, termination)
        status('claimed_before_Torch')
        writer.json(OUT / 'started.json', {'complete': True, 'plan': pin(PLAN), 'runner': pin(Path(__file__).resolve()),
            'new_forwards_maximum': 716, 'native_carried_output_slots': 64, 'model_qualified': False})
        writer.json(OUT / 'zero_call_controls.json', {'complete': True, 'native_excluded16': plan['native_excluded16'],
            'native32_carried_new_calls': 0, 'mixed72_and_filter24_new_calls': 0, 'new_neural_calls': 0})
        need(ROOT.resolve() == Path.cwd().resolve() and (ROOT / 'third_party/codeformer/codeformer_arch.py').is_file(),
             'Validated ROOT before retained factory imports')
        sys.path.insert(0, str(ROOT))
        import torch
        import numpy as np
        from PIL import Image
        from torch.nn import functional as F
        torch.set_num_threads(4); need(torch.get_default_dtype() == torch.float32, 'CPU FP32 default')
        source = importlib.import_module('research.codeformer_u96_postfusion_lr_inference_v1')
        legacy = importlib.import_module('research_spatial_restoration_v1')
        metric_spec = importlib.util.spec_from_file_location('_owned_broader_preservation_metrics_v1', ROOT / METRICS_PATH)
        M = importlib.util.module_from_spec(metric_spec); metric_spec.loader.exec_module(M)
        M.sum = ordered_sum  # Preserve historical scalar arithmetic without modifying retained source/module.
        from third_party.codeformer.codeformer_arch import CodeFormer
        phase = 'load_two_fresh_owned_frozen_CPU_models'; status('before_model_load'); resource('before_model_load')
        timer = watchdog(LIMITS['model_load_watchdog_seconds'])
        try:
            candidate = source.load_frozen_cpu_restorer(bound(plan['model']['r2']), bound(plan['model']['u96']),
                bound(plan['head_file']), inference_source_sha256=B.INFERENCE_SHA)
            with torch.no_grad(), torch.device('cpu'), torch.random.fork_rng(devices=[]):
                base = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(source.SITES))
                r2 = torch.load(bound(plan['model']['r2']), map_location='cpu', weights_only=True, mmap=True)
                base.load_state_dict(r2['params_ema'], strict=True)
                incumbent = legacy.inference_class(torch)(base).float().eval().requires_grad_(False)
                u96 = torch.load(bound(plan['model']['u96']), map_location='cpu', weights_only=True, mmap=True)
                incumbent.adapters.load_state_dict({k.removeprefix('adapters.'): v for k, v in u96['adapter_state'].items()}, strict=True)
                del r2, u96
            need(incumbent.base is not candidate.base, 'Distinct owned frozen model bases')
        finally: timer.cancel()
        def invariants():
            need(len(incumbent.state_dict()) == 539 and all(not m.training for m in incumbent.modules()) and
                 all(v.device.type == 'cpu' and v.dtype == torch.float32 and not v.requires_grad and v.grad is None
                     for v in incumbent.parameters()), 'Frozen539 incumbent CPUFP32')
            need(legacy.base.state_sha(incumbent.base.state_dict(), torch) == source.R2_STATE and
                 legacy.base.state_sha({'adapters.' + k: v for k, v in incumbent.adapters.state_dict().items()}, torch) == source.U96_STATE,
                 'Exact retained515+24 canonical U96')
            need(all(not m._forward_hooks and not m._forward_pre_hooks for m in incumbent.modules()), 'Clean incumbent hooks')
            candidate.validate_invariants()
        invariants(); models = dict(zip(ARMS, (incumbent, candidate)))
        def png(path, value):
            stream = io.BytesIO(); Image.fromarray(value).save(stream, format='PNG'); return writer.write(path, stream.getvalue())
        def array(path, value):
            need(value.dtype == np.float32 and np.isfinite(value).all(), 'Finite FP32 saved tensor')
            stream = io.BytesIO(); np.save(stream, value, allow_pickle=False); return writer.write(path, stream.getvalue())
        banks = {a: {} for a in ARMS}; case_rows = []
        for index, spec in enumerate(plan['paired_cases']):
            cid = spec['id']; folder = OUT / 'cases' / cid
            with Image.open(bound(spec['input'])) as image:
                need(image.mode == 'RGB' and image.size == (256, 256), 'ExactRGB256 observed input')
                canvas = np.asarray(image).copy()
            if spec['support']:
                with Image.open(bound(spec['support'])) as image:
                    need(image.mode == 'L' and image.size == (256, 256), 'ExactL256 support')
                    observed = np.asarray(image).copy() > 0
            else: observed = np.ones((256, 256), dtype=np.bool_)
            with Image.open(bound(spec['target'])) as image:
                need(image.mode == 'RGB' and image.size == (256, 256), 'ExactRGB256 paired target')
                truth = np.asarray(image).astype(np.float32) / np.float32(255)
            need(B.digest(canvas.tobytes()) == spec['input_pixels']['RGB8_sha256'], 'Exact input pixels')
            inputs = {'original': writer.write(folder / 'input.png', bound(spec['input']).read_bytes()),
                      'target': writer.write(folder / 'target.png', bound(spec['target']).read_bytes()),
                      'support': png(folder / 'observed_support.png', observed.astype(np.uint8) * 255)}
            tensor = torch.from_numpy(canvas.astype(np.float32).transpose(2, 0, 1).copy())[None] / 255
            sample = F.interpolate(tensor, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
            case = {'complete': False, 'id': cid, 'evidence': spec['evidence'], 'source': spec['source'],
                    'profile': spec['profile'], 'split': 'DEV', 'inputs': inputs, 'outputs': {},
                    'historical_spec': spec, 'input_readability_not_newly_qualified': True, 'model_qualified': False}
            for arm_index, arm in enumerate(ARMS):
                active = plan['forward_slots'][index * 2 + arm_index]
                phase = 'before_forward'; resource(arm + '_before'); status('before_reservation')
                need(counters['forward_attempts'] < 716 and counters['per_arm'][arm]['attempts'] < 358, 'Finite716 forward reservations')
                counters['forward_attempts'] += 1; counters['per_arm'][arm]['attempts'] += 1
                phase = 'forward_reserved'; status('forward_reserved')
                timer = watchdog(LIMITS['individual_forward_watchdog_seconds'])
                try:
                    with torch.no_grad(): triple = models[arm](sample, w=1., detach_16=True, adain=True)
                finally: timer.cancel()
                counters['forward_returns'] += 1; counters['per_arm'][arm]['returns'] += 1
                phase = 'forward_returned'; status('forward_returned')
                need(isinstance(triple, tuple) and len(triple) == 3, 'Stock retained output tuple')
                for value, shape in zip(triple, ((1, 3, 512, 512), (1, 256, 1024), (1, 256, 16, 16))):
                    need(value.shape == shape and value.dtype == torch.float32 and value.device.type == 'cpu'
                         and not value.requires_grad and torch.isfinite(value).all().item(), 'Finite exactCPU output shape')
                with torch.no_grad():
                    native, logits, lq = triple
                    unclipped = F.interpolate((native + 1) / 2, (256, 256), mode='bilinear', align_corners=False)
                    raw = F.interpolate(((native + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
                raw_array = raw[0].permute(1, 2, 0).contiguous().numpy().copy()
                raw_array[~observed] = canvas.astype(np.float32)[~observed] / 255
                pixels = np.floor(raw_array * np.float32(255)).astype(np.uint8)
                armfolder = folder / arm
                output = {'native512': array(armfolder / 'native512.npy', native.numpy()),
                    'logits': array(armfolder / 'logits.npy', logits.numpy()), 'LQ': array(armfolder / 'LQ.npy', lq.numpy()),
                    'unclipped': array(armfolder / 'unclipped256.npy', unclipped[0].permute(1, 2, 0).contiguous().numpy()),
                    'raw': array(armfolder / 'raw.npy', raw_array), 'png': png(armfolder / 'output.png', pixels),
                    'metrics': M.metrics(raw_array, pixels, truth, observed, 'paired_synthetic'),
                    'provenance': {'arm': arm, 'R2_file_sha256': B.R2_SHA, 'U96_file_sha256': B.U96_SHA,
                        'head_file_sha256': B.HEAD_SHA if arm_index else None,
                        'head_canonical_sha256': B.HEAD_STATE if arm_index else None,
                        'w': 1., 'detach16': True, 'AdaIN': True, 'CPU_FP32': True, 'CPU_threads': 4,
                        'alignment': 'none', 'display_processing': 'none', 'context_blur_or_noise': False,
                        'source_candidate_training_updates': 64 if arm_index else None,
                        'outside_support_exact_observed_pixels': True, 'model_qualified': False,
                        'forward_sites': list(models[arm].last_forward_sites)}}
                invariants(); need(models[arm].last_forward_sites == source.SITES, 'All4 orderedfeature sites')
                writer.json(armfolder / 'record.json', {'complete': True, 'case_id': cid, **output,
                    'counters': counters, 'slot': active, 'plan_sha256': expected_sha})
                case['outputs'][arm] = output
                banks[arm][cid] = {**{k: spec[k] for k in ('id', 'evidence', 'source', 'profile')}, 'metrics': output['metrics']}
                counters['saved_outputs'] += 1; counters['per_arm'][arm]['saved'] += 1
                phase = 'output_fully_saved'; status('output_fully_saved'); resource(arm + '_saved')
                del triple, native, logits, lq, raw, unclipped
            case['complete'] = True; writer.json(folder / 'case.json', case); case_rows.append(case)
            counters['completed_paired_cases'] += 1; phase = 'case_complete'; status('case_complete')
            print(B.json_bytes({'case': index + 1, 'of': 358, 'new_returns': counters['forward_returns']}).decode(), end='', flush=True)
        need(counters['forward_attempts'] == counters['forward_returns'] == counters['saved_outputs'] == 716
             and counters['completed_paired_cases'] == 358, 'Exactly358paired cases/716new forwards saved')
        native_cases = {r['historical_id']: r for r in plan['native_carry32']}
        excluded = {r['historical_id']: r for r in plan['native_excluded16']}
        membership = []
        by_pair = {r['id']: r for r in case_rows}
        for spec in plan['historical_cases']:
            cid = spec['id']
            if cid in by_pair:
                membership.append({'id': cid, 'status': 'FRESH_PAIRED_OUTPUTS', 'case': pin(OUT / 'cases' / cid / 'case.json')})
                continue
            for arm in ARMS:
                banks[arm][cid] = {**{k: spec[k] for k in ('id', 'evidence', 'source', 'profile')}, 'metrics': blank_native_metrics()}
            if cid in native_cases:
                membership.append({'id': cid, 'status': 'CARRIED_NATIVE_UNPAIRED_OUTPUTS',
                    'prior_case': native_cases[cid], 'native_reference_metrics': None, 'new_forward_calls': 0})
            else:
                need(cid in excluded, 'All excluded native membership accounted')
                membership.append({'id': cid, 'status': 'EXCLUDED_INPUT_ONLY_ZERO_CALL',
                    'input_only_review': excluded[cid]['input_only_review'], 'outputs': None,
                    'reference_metrics': None, 'new_forward_calls': 0})
        phase = 'final_integrity'; invariants(); resource('final')
        for row in plan['sources']: bound(row)
        for row in [plan['head_file'], *plan['model'].values(), *plan['prerequisites'].values()]: bound(row)
        gates = {arm: {'preservation_vs_original': M.preservation(banks[arm], plan['original_teacher_bank']),
                       'dark_retention_vs_R2': M.dark_retention(banks[arm], plan['R2_dark_reference_bank'])} for arm in ARMS}
        candidate_pass = all(v['pass'] for v in gates[ARMS[1]].values())
        quality_status = 'BROAD_PRESERVATION_GATE_PASS_NOT_QUALIFIED' if candidate_pass else 'QUALITY_STOP_NOT_QUALIFIED'
        comparison = {'complete': True, 'quality_status': quality_status, 'plan': pin(PLAN), 'counters': counters,
            'cases': case_rows, 'historical_membership406': membership, 'banks406': banks, 'gates': gates,
            'paired_fresh_cases': 358, 'new_forward_returns': 716, 'native_carried_output_slots': 64,
            'native_excluded_rows_zero_call': 16, 'represented_output_slots': 780,
            'teacher_metric_policy': plan['teacher_metric_policy'], 'native_reference_metrics': None,
            'paired_and_native_evidence_separate': True, 'model_qualified': False, 'app_default_changed': False,
            'no_optimization': True, 'seconds': time.monotonic() - start}
        writer.json(OUT / 'comparison.json', comparison)
        measurement_complete = True; phase = 'complete_measurement'; status('complete_measurement')
        actual_files = [q for q in OUT.rglob('*') if q.is_file()]
        need(sum(q.stat().st_size for q in actual_files) == writer.successful_bytes == writer.reserved_bytes,
             'Incremental byte ledger equals actual fully saved output namespace')
        return 0 if candidate_pass else 2
    except BaseException as error:
        writer.json(OUT / 'failure.json', {'complete': True, 'status': 'LOCAL_INFERENCE_INFRASTRUCTURE_STOP_NOT_QUALIFIED',
            'phase': phase, 'slot': active, 'counters': counters,
            'forward_outcome_unknown': counters['forward_attempts'] != counters['forward_returns'],
            'last_returned_status_receipt': last_status, 'error': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(), 'seconds': time.monotonic() - start,
            'reserved_output_bytes': writer.reserved_bytes, 'successful_output_bytes': writer.successful_bytes,
            'automatic_retry_resume': False, 'model_qualified': False})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def read_journal():
    parsed = []; partial = []
    for path in sorted((OUT / 'journal').glob('*.json')):
        try:
            value = load(path)
            need(isinstance(value, dict) and isinstance(value.get('counters'), dict) and
                 type(value.get('forward_outcome_unknown')) is bool and
                 {'forward_attempts', 'forward_returns', 'saved_outputs', 'completed_paired_cases'} <= value['counters'].keys(),
                 'Journal receipt schema')
            need(value['sequence'] == int(path.stem), 'Journal sequence identity')
            parsed.append((pin(path), value))
        except (ValueError, OSError, KeyError):
            info = path.lstat()
            need(path.resolve().is_relative_to(OUT) and B.stat.S_ISREG(info.st_mode) and
                 not B.stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024
                 and 0 <= info.st_size <= 1024 ** 2, 'Bounded ordinary partial receipt')
            raw = path.read_bytes(); partial.append({'path': path.relative_to(ROOT).as_posix(),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    return parsed, partial


def supervise(expected_sha):
    verify(expected_sha); need(not OUT.exists(), 'Fresh worker namespace, no automatic retry')
    log = PREP / 'broader_worker_v1.log'; result_path = PREP / 'broader_supervisor_v1.json'
    need(not log.exists() and not result_path.exists(), 'Fresh supervisor evidence')
    start = time.monotonic(); timeout = False
    with log.open('xb') as stream:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--plan-sha', expected_sha],
                                 cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT)
        try: code = child.wait(timeout=LIMITS['external_seconds'])
        except subprocess.TimeoutExpired:
            timeout = True; child.terminate()
            try: code = child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); code = child.wait(timeout=10)
    parsed, partial = read_journal(); last_pin, last = parsed[-1] if parsed else (None, None)
    failure = None; failure_pin = None; unreadable = False; failure_error = None
    if (OUT / 'failure.json').exists():
        try:
            path = OUT / 'failure.json'; info = path.lstat()
            need(path.resolve().is_relative_to(OUT) and B.stat.S_ISREG(info.st_mode) and
                 not B.stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024
                 and 0 <= info.st_size <= 1024 ** 2, 'Bounded owned failure receipt')
            raw = path.read_bytes(); failure_pin = {'path': path.relative_to(ROOT).as_posix(),
                'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            parsed_failure = B.strict_json(raw)
            need(isinstance(parsed_failure, dict) and isinstance(parsed_failure.get('counters'), dict) and
                 type(parsed_failure.get('forward_outcome_unknown')) is bool and
                 {'forward_attempts', 'forward_returns', 'saved_outputs', 'completed_paired_cases'} <= parsed_failure['counters'].keys(),
                 'Owned failure receipt schema')
            failure = parsed_failure
        except (ValueError, OSError) as error:
            unreadable = True; failure_error = {'path': str(OUT / 'failure.json'), 'error': type(error).__name__,
                'message': str(error), 'unreadable_bytes_not_given_invented_hash': failure_pin is None}
    comparison = None; comparison_pin = None
    if (OUT / 'comparison.json').exists():
        try:
            comparison = load(OUT / 'comparison.json'); comparison_pin = pin(OUT / 'comparison.json')
        except (ValueError, OSError): pass
    result = {'complete': True, 'worker_exit': code, 'timeout': timeout, 'seconds': time.monotonic() - start,
        'plan_sha256': expected_sha, 'last_durable_status': last_pin, 'partial_journal_files': partial,
        'failure_receipt': failure_pin, 'failure_receipt_unreadable': unreadable, 'failure_receipt_error': failure_error,
        'comparison_receipt': comparison_pin, **supervisor_outcome(code, timeout, last, partial, failure, unreadable, comparison),
        'prior_fixed68_original_unknown_evidence_retained': True, 'model_qualified': False,
        'automatic_retry_resume': False, 'training_calls': 0}
    with result_path.open('xb') as stream:
        stream.write(B.json_bytes(result)); stream.flush(); os.fsync(stream.fileno())
    print(B.json_bytes(result).decode(), end='')
    return 0 if result['finite_preservation_gate_pass_not_qualification'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-plan', action='store_true'); action.add_argument('--verify-only', action='store_true')
    action.add_argument('--run', action='store_true'); action.add_argument('--worker', action='store_true')
    parser.add_argument('--plan-sha'); parser.add_argument('--data-audit'); parser.add_argument('--teacher-audit')
    parser.add_argument('--source-review'); args = parser.parse_args()
    if args.prepare_plan:
        need(all((args.data_audit, args.teacher_audit, args.source_review)), 'Actual data/teacher/source reviews required')
        return prepare(args)
    need(type(args.plan_sha) is str and len(args.plan_sha) == 64, 'Explicit sealed SHA required')
    if args.verify_only:
        verify(args.plan_sha); print(B.json_bytes({'complete': True, 'new716slots': 716, 'neural_calls': 0}).decode(), end=''); return 0
    return worker(args.plan_sha) if args.worker else supervise(args.plan_sha)


if __name__ == '__main__': raise SystemExit(main())
