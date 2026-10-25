"""Pure saved-metric joins and unchanged gates for a future u64 continuation.

No model, tensor framework, optimizer, worker or runtime entry point exists.
The caller supplies actual pixel arrays to record_metrics and immutable parent/
historical-audit bytes to the bank evaluator. Scalar evidence does not establish
pixel integrity; a later worker and return auditor must bind both explicitly.
"""
import copy
import hashlib
import json
from pathlib import Path

from research.codeformer_u96_spatial_feasibility_v1 import metrics as _inherited
from . import contract


INHERITED_METRICS_BYTES = 17480
INHERITED_METRICS_SHA256 = '1ec77a3cf3621494d0dbe36f8f46a2a8db06c4006bfab41be639399156e08b05'
SAVED_AUDIT_BYTES = 2677742
SAVED_AUDIT_SHA256 = '8b73cbb901452faff8b3679bcf9d077d4d8c98a90e461c18af260565fd04c18d'
SAVED_PROTOCOL_SHA256 = '8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9'
VIEWS = contract.VIEWS
ROLES = contract.ROLES
FIELDS = contract.FIELDS
REGIONS = ('whole', 'central128', 'outside')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def verify_inherited_source():
    """Join the reused scalar/pixel implementation to its reviewed exact bytes."""
    raw = Path(_inherited.__file__).read_bytes()
    need((len(raw), hashlib.sha256(raw).hexdigest()) ==
         (INHERITED_METRICS_BYTES, INHERITED_METRICS_SHA256),
         'Exact reviewed inherited metric source required')


def _scope(parent_bytes):
    selection = contract.select_parent_cases(parent_bytes)
    protocol = json.loads(parent_bytes)
    rows = selection['monitors'] + selection['guards']
    return protocol, selection, {row['id']: row for row in rows}


def _bank(value, expected):
    bank = _inherited._bank(value)
    _inherited._validate_bank(bank, expected)
    # Tighten schemas for exact saved-metric equality, including the distances
    # on legacy rows which are not consumed by every inherited scalar gate.
    for key, reference in expected.items():
        row = bank[key]
        for field in ('metrics', 'regional_metrics', 'U96_behavior_distance',
                      'U96_illumination_distance'):
            need(type(row.get(field)) is dict and set(row[field]) == set(VIEWS),
                 'Exact two-view metric schema required: ' + field)
        for view in VIEWS:
            need(set(row['metrics'][view]) == set(FIELDS), 'Exact whole metric fields required')
            if reference.get('target') is not None:
                regions = row['regional_metrics'][view]
                need(type(regions) is dict and set(regions) == set(REGIONS),
                     'Exact whole/central/outside metric regions required')
                for region in REGIONS:
                    need(type(regions[region]) is dict and set(regions[region]) == set(FIELDS),
                         'Exact regional metric fields required')
            for field in ('U96_behavior_distance', 'U96_illumination_distance'):
                _inherited._number(row[field][view])
        teacher = reference.get('behavior_teacher') is not None
        need(('behavior_distance' in row) == teacher, 'Exact original teacher-distance presence required')
        if teacher:
            need(type(row['behavior_distance']) is dict and set(row['behavior_distance']) == set(VIEWS),
                 'Both original teacher-distance views required')
    return bank


def _metric_payload(row):
    fields = ('id', 'reference_id', 'role', 'split', 'metrics', 'regional_metrics',
              'U96_behavior_distance', 'U96_illumination_distance')
    value = {field: row[field] for field in fields}
    if 'behavior_distance' in row:
        value['behavior_distance'] = row['behavior_distance']
    return value


def historical_controls(parent_bytes, saved_audit_bytes):
    """Read exact saved scalar evidence, not cached image/tensor payloads.

    Initial u64 and frozen U96 banks come from the independently audited same
    historical run. The returned output bindings identify the actual cached
    RAW/PNG files whose bytes a later worker must verify and recompute.
    """
    verify_inherited_source()
    protocol, selection, expected = _scope(parent_bytes)
    need(type(saved_audit_bytes) is bytes and
         (len(saved_audit_bytes), hashlib.sha256(saved_audit_bytes).hexdigest()) ==
         (SAVED_AUDIT_BYTES, SAVED_AUDIT_SHA256), 'Exact independently audited u64 pixel receipt required')
    audit = json.loads(saved_audit_bytes)
    need(audit['complete'] is True and audit['independent_saved_pixel_integrity_pass'] is True
         and audit['pilot_quality_gates_pass'] is True and audit['model_qualified'] is False
         and audit['protocol']['sha256'] == SAVED_PROTOCOL_SHA256,
         'Original completed finite u64 screen required')
    banks, bindings = {}, {}
    for label, kind, update in (('frozen_U96', 'reference', 0), ('saved_u64', 'snapshot', 64)):
        found = [row for row in audit['all_saved_pixel_metrics']
                 if row['id'] in expected and row['kind'] == kind and row['update'] == update]
        need(len(found) == 56 and len({row['id'] for row in found}) == 56,
             'Exactly one saved historical record for all56 quality IDs required')
        bank, bound = {}, {}
        for source in found:
            reference = expected[source['id']]
            need(source['role'] == reference['role'], 'Saved role differs from immutable parent')
            row = {field: reference[field] for field in ('id', 'reference_id', 'role', 'split')}
            row['metrics'] = copy.deepcopy(source['metrics'])
            row['regional_metrics'] = copy.deepcopy(source['regional_metrics'])
            for field in ('U96_behavior_distance', 'U96_illumination_distance'):
                row[field] = (dict.fromkeys(VIEWS, 0.0) if label == 'frozen_U96'
                              else copy.deepcopy(source[field]))
            if reference.get('behavior_teacher') is not None:
                row['behavior_distance'] = copy.deepcopy(source['behavior_distance'])
            bank[row['id']] = row
            bound[row['id']] = {'raw': copy.deepcopy(source['raw_file']),
                                'native_png': copy.deepcopy(source['png_file'])}
        banks[label] = _bank(bank, expected)
        bindings[label] = bound
    return {**banks, 'output_bindings': bindings,
            'saved_audit': {'bytes': len(saved_audit_bytes), 'sha256': SAVED_AUDIT_SHA256},
            'scalar_evidence_only': True, 'cached_pixels_redecoded': False,
            'model_qualified': False}


def cohort_means(bank, parent_bytes):
    """Ordered eight-photo means: local central128, all other roles whole256."""
    verify_inherited_source()
    protocol, selection, expected = _scope(parent_bytes)
    bank = _bank(bank, expected)
    result = {view: {} for view in VIEWS}
    for role in ROLES:
        ids = sorted(row['id'] for row in selection['monitors'] if row['role'] == role)
        need(len(ids) == 8, 'Exactly eight admitted monitor rows per role required')
        region = 'central128' if role == ROLES[0] else 'whole'
        for view in VIEWS:
            result[view][role] = {
                field: _inherited.ordered_float_sum(
                    bank[key]['regional_metrics'][view][region][field] for key in ids) / 8
                for field in FIELDS}
    return result


def record_metrics(raw, png, target, row, *, np, frozen_U96_raw, frozen_U96_png,
                   teacher_raw=None, teacher_png=None):
    """Recompute actual delivered metrics with an explicit original U96 control.

    Both baseline arrays are mandatory: omitted controls cannot silently turn
    actual u64 movement/illumination into a zero initializer self-distance.
    """
    verify_inherited_source()
    need(frozen_U96_raw is not None and frozen_U96_png is not None,
         'Actual immutable frozen U96 control arrays required')
    return _inherited.record_metrics(raw, png, target, row, np=np,
        teacher_raw=teacher_raw, teacher_png=teacher_png,
        baseline_raw=frozen_U96_raw, baseline_png=frozen_U96_png)


def record_training_metrics(raw, png, target, row, *, np):
    """Photographic metrics for an admitted optimization call without a cache.

    Only the56 quality cases have frozen-U96 cached arrays in this finite
    design. The64 optimizing calls therefore report no behavior/illumination
    movement distances, rather than inventing an initializer self-distance.
    """
    verify_inherited_source()
    need(type(row) is dict and row.get('cohort') == 'optimize'
         and row.get('optimization_permitted') is True and row.get('split') == 'TRAIN'
         and row.get('role') in ROLES and row.get('target') is not None
         and row.get('behavior_teacher') is None and target is not None,
         'Admitted paired TRAIN optimization target without behavior teacher required')
    measured = _inherited.record_metrics(raw, png, target, row, np=np)
    return {'metrics': measured['metrics'], 'regional_metrics': measured['regional_metrics'],
            'behavior_movement_reference_available': False}


def evaluate_bank(current, initial_u64, frozen_U96, parent_bytes, *, phase, saved_audit_bytes):
    """Require exact56 banks, original gates, and the new retention requirements.

    The original gate helper is always called in initializer mode against
    frozen U96, avoiding that diagnostic's unrelated proposal-only extras.
    Original terminal1%/dark absolute gates bind at u16. Additional contract
    checks compare with exact historical u64; no inherited baseline is reset.
    """
    need(phase in ('initializer', 'u4', 'u8', 'u16'), 'Explicit finite continuation phase required')
    controls = historical_controls(parent_bytes, saved_audit_bytes)
    protocol, selection, expected = _scope(parent_bytes)
    initial_u64 = _bank(initial_u64, expected)
    frozen_U96 = _bank(frozen_U96, expected)
    current = _bank(current, expected)
    for label, actual, saved in (('initial_u64', initial_u64, controls['saved_u64']),
                                 ('frozen_U96', frozen_U96, controls['frozen_U96'])):
        need(all(_metric_payload(actual[key]) == _metric_payload(saved[key]) for key in expected),
             'Complete actual bank metrics differ from exact saved historical ' + label)
    if phase == 'initializer':
        need(all(_metric_payload(current[key]) == _metric_payload(initial_u64[key]) for key in expected),
             'Initializer phase must be the exact admitted u64 bank')
    original = _inherited.evaluate_bank(current, frozen_U96, protocol, state='initializer')
    means = {'initial_u64': cohort_means(initial_u64, parent_bytes),
             'current': cohort_means(current, parent_bytes),
             'frozen_U96': cohort_means(frozen_U96, parent_bytes)}
    extra = contract.retention_checks(means['initial_u64'], means['current'],
                                     means['frozen_U96'], phase=phase)
    terminal = original['full_pilot_endpoints']
    terminal_binding = phase == 'u16'
    passed = original['pass'] and extra['pass'] and (not terminal_binding or terminal['pass'])
    return {'pass': passed, 'phase': phase, 'complete_quality_cases': 56,
            'legacy': original['legacy'], 'ordinary_monitor': original['monitor'],
            'original_terminal': {'pass': terminal['pass'], 'binding': terminal_binding,
                                  'groups': terminal['groups']},
            'additional_retention': extra, 'cohort_means': means,
            'initial_u64_full_metric_join': True, 'frozen_U96_full_metric_join': True,
            'original_quality_baseline': 'frozen_U96_and_original_legacy_teachers',
            'inherited_proposal_extras_applied': False, 'quality_tolerance_added': False,
            'historical_broader_failures_cleared': False, 'model_qualified': False,
            'TRAIN_only_not_DEV_FINAL_or_native_qualification': True}
