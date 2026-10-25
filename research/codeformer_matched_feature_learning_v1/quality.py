"""Pure matched-TRAIN quality checks; no model, image or worker operations.

The inherited 56-case bank is evaluated by its unchanged, byte-pinned helper.
The new 40-case bank adds checks against its own original u64 measurements;
its result cannot replace the inherited bank or establish pixel integrity.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path

from .objective import ROLES


ROOT = Path(__file__).resolve().parents[2]
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'adjacent_rgb_error', 'coarse_rgb_mae')
SNAPSHOTS = (0, 8, 24, 48)
INHERITED_PHASES = {0: 'initializer', 8: 'u8', 24: 'u16', 48: 'u16'}
INHERITED_SOURCE = 'research/codeformer_u64_cleananchors_continuation_v1/metrics.py'
INHERITED_BYTES = 11967
INHERITED_SHA256 = '030649b25df131750460ae51767bdc0254d0017bcf7e9597bc169b7d4c8f386e'
INHERITED_DEPENDENCIES = {
    'research.codeformer_u64_cleananchors_continuation_v1.contract': {
        'path': 'research/codeformer_u64_cleananchors_continuation_v1/contract.py',
        'bytes': 10844, 'sha256': '99e1bbef911750c5ef0622d350e203bcafed58559b3c2a2c500744ef76102bb0'},
    'research.codeformer_u64_cleananchors_continuation_v1.objective': {
        'path': 'research/codeformer_u64_cleananchors_continuation_v1/objective.py',
        'bytes': 8205, 'sha256': 'cffafed3feae10ce91890cfd68ee5de96c8db8edad00041420615cd071f2b7ad'},
    'research.codeformer_u96_spatial_feasibility_v1.metrics': {
        'path': 'research/codeformer_u96_spatial_feasibility_v1/metrics.py',
        'bytes': 17480, 'sha256': '1ec77a3cf3621494d0dbe36f8f46a2a8db06c4006bfab41be639399156e08b05'},
}
CANDIDATE_SOURCE = 'outputs/codeformer_matched_feature_learning_v1_preparation/candidate_data_v1.json'
CANDIDATE_BYTES = 323833
CANDIDATE_SHA256 = '23c402f82c9dea0b52f32a828446b5f961c52d249a7c9881e260459e0147e9ce'
MONITOR_REFERENCES = (
    'tr_ffhq_02147', 'tr_ffhq_05179', 'tr_ffhq_08064', 'tr_ffhq_08094',
    'tr_ffhq_08698', 'tr_ffhq_08949', 'tr_ffhq_11039', 'tr_ffhq_11154',
)
METRIC_PAYLOAD_FIELDS = ('id', 'reference_id', 'role', 'split', 'metrics')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _update(update):
    need(type(update) is int and update in SNAPSHOTS,
         'Explicit matched checkpoint 0, 8, 24 or 48 required')
    return update


def _pinned_bytes(relative, size, digest, label):
    path = ROOT / relative
    need(path.resolve(strict=True) == path.absolute(),
         'Exact unlinked ' + label + ' source required')
    raw = path.read_bytes()
    need(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
         'Exact unchanged ' + label + ' source bytes required')
    return raw


def inherited_metrics():
    """Authenticate the original scalar evaluator before importing it."""
    _pinned_bytes(INHERITED_SOURCE, INHERITED_BYTES, INHERITED_SHA256, 'inherited quality')
    for pin in INHERITED_DEPENDENCIES.values():
        _pinned_bytes(pin['path'], pin['bytes'], pin['sha256'], 'inherited quality dependency')
    module = importlib.import_module('research.codeformer_u64_cleananchors_continuation_v1.metrics')
    need(Path(module.__file__).resolve() == ROOT / INHERITED_SOURCE,
         'Inherited quality import origin differs')
    for name, pin in INHERITED_DEPENDENCIES.items():
        dependency = importlib.import_module(name)
        need(Path(dependency.__file__).resolve() == ROOT / pin['path'],
             'Inherited quality dependency import origin differs')
    return module


def evaluate_inherited(current, initial_u64, frozen_U96, parent_bytes,
                       saved_audit_bytes, *, update):
    """Apply the exact56 bank, original references and original thresholds.

    Checkpoints24 and48 both bind the original terminal1% requirements and
    strict u64 retention. No checkpoint is compared to a moving reference.
    The unchanged inherited result is returned without replacing any gate.
    """
    phase = INHERITED_PHASES[_update(update)]
    return inherited_metrics().evaluate_bank(
        current, initial_u64, frozen_U96, parent_bytes,
        phase=phase, saved_audit_bytes=saved_audit_bytes)


def _selection(selection):
    """Validate the fixed40 identities after the contract binds source bytes.

    The worker must rederive selection through contract.select_cases using the
    pinned admission, candidate and parent bytes. These fixed identities avoid
    a runtime dependency on a local preparation-file path; they do not prove
    admission, input/target bytes or saved output pixels by themselves.
    """
    expected = {
        reference + '__' + role: {'reference_id': reference, 'role': role, 'split': 'TRAIN'}
        for reference in MONITOR_REFERENCES for role in ROLES}
    need(type(selection) is dict and type(selection.get('new_monitors')) is list,
         'Selection must contain the exact new_monitors list')
    rows = selection['new_monitors']
    need(len(rows) == 40 and all(type(row) is dict and type(row.get('id')) is str for row in rows),
         'Exactly40 identified selected monitor rows required')
    by_id = {row['id']: row for row in rows}
    need(len(by_id) == len(rows), 'Duplicate selected monitor ID refused')
    need(set(by_id) == set(expected), 'Complete exact40 matched monitor membership required')
    for key, row in by_id.items():
        source = expected[key]
        need(row.get('reference_id') == source['reference_id']
             and row.get('role') == source['role'] and row.get('role') in ROLES,
             'Selected monitor reference or role differs from original candidate')
        need(row.get('split') == source['split'] == 'TRAIN'
             and row.get('cohort') == 'monitor' and row.get('optimization_permitted') is False,
             'Selected matched monitors must remain evaluation-only TRAIN monitor rows')
    references = {row['reference_id'] for row in rows}
    need(len(references) == 8 and
         {(row['reference_id'], row['role']) for row in rows} ==
         {(reference, role) for reference in references for role in ROLES},
         'Exact eight-reference five-role monitor cross product required')
    return by_id


def _number(value):
    need(type(value) in (float, int), 'Finite nonnegative scalar metric required')
    try:
        scalar = float(value)
    except OverflowError:
        raise ValueError('Finite nonnegative scalar metric required') from None
    need(math.isfinite(scalar) and scalar >= 0, 'Finite nonnegative scalar metric required')
    return scalar


def _bank(value, expected):
    if type(value) is list:
        need(all(type(row) is dict and type(row.get('id')) is str for row in value),
             'Identified new monitor metric records required')
        converted = {row['id']: row for row in value}
        need(len(converted) == len(value), 'Duplicate new monitor metric ID refused')
        value = converted
    need(type(value) is dict and all(type(key) is str and type(row) is dict
         and row.get('id') == key for key, row in value.items()),
         'ID-bound new monitor metric bank required')
    need(set(value) == set(expected), 'Complete exact40 new monitor metric bank required')
    for key, row in value.items():
        reference = expected[key]
        need(all(row.get(field) == reference[field]
                 for field in ('id', 'reference_id', 'role', 'split')),
             'New monitor metric identity, role or TRAIN split differs')
        need(type(row.get('metrics')) is dict and set(row['metrics']) == set(VIEWS),
             'Exact RAW/native_png metric views required')
        for view in VIEWS:
            metrics = row['metrics'][view]
            need(type(metrics) is dict and set(metrics) == set(FIELDS),
                 'Exact MAE, adjacent RGB and coarse RGB metric fields required')
            for field in FIELDS:
                _number(metrics[field])
    return value


def _metric_payload(row):
    """Canonical JSON bytes preserve numeric representation at initializer0.

    These are metric-payload bytes, never RAW image or delivered PNG bytes.
    Extra row bookkeeping is excluded; all identity and exact metric fields
    are included. Integer1, float1.0 and signed zero remain distinguishable.
    """
    return json.dumps({field: row[field] for field in METRIC_PAYLOAD_FIELDS},
                      sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _cohort_means(bank, expected):
    result = {view: {} for view in VIEWS}
    for role in ROLES:
        ids = sorted(key for key, row in expected.items() if row['role'] == role)
        need(len(ids) == 8, 'Exactly eight ordered metrics per matched role required')
        for view in VIEWS:
            result[view][role] = {}
            for field in FIELDS:
                total = 0.0
                for key in ids:
                    total = total + float(bank[key]['metrics'][view][field])
                result[view][role][field] = _number(total / 8)
    return result


def evaluate_new_bank(current, initial, selection, *, update):
    """Supplemental five-profile monitor gates against original u64 only.

    Every checkpoint validates all three fields in both views. At8 all five
    MAE means must be nonincreasing; at24/48 the adjacent and coarse RGB means
    must also be nonincreasing. Means use sorted IDs and sequential Python
    float arithmetic. No added epsilon or previous-checkpoint rebasing exists.
    """
    _update(update)
    expected = _selection(selection)
    start, now = _bank(initial, expected), _bank(current, expected)
    equal = all(_metric_payload(now[key]) == _metric_payload(start[key]) for key in expected)
    if update == 0:
        need(equal, 'Checkpoint0 must have byte-equal canonical initial metric payloads')
    means = {'initial_u64': _cohort_means(start, expected),
             'current': _cohort_means(now, expected)}
    bound_fields = () if update == 0 else (('mae',) if update == 8 else FIELDS)
    checks = []
    for view in VIEWS:
        for role in ROLES:
            for field in bound_fields:
                original = means['initial_u64'][view][role][field]
                actual = means['current'][view][role][field]
                checks.append({'view': view, 'role': role, 'field': field,
                    'initial_u64': original, 'current': actual,
                    'pass': actual <= original})
    passed = all(row['pass'] for row in checks)
    return {'pass': passed, 'update': update, 'complete_new_monitor_cases': 40,
        'monitor_photographs': 8, 'roles': list(ROLES), 'cohort_means': means,
        'role_mean_nonincrease_checks': checks,
        'initializer_canonical_metric_payloads_equal': equal if update == 0 else None,
        'additional_gates_only': True, 'inherited_complete_bank_gates_required': True,
        'inherited_quality_cases_required': 56, 'full_quality_pass_established': False,
        'original_new_bank_reference': 'initial_u64', 'quality_tolerance_added': False,
        'zero_added_tolerance': True, 'scalar_evidence_only': True,
        'pixels_or_model_state_verified_here': False, 'independent_holdout': False,
        'historical_broader_failures_cleared': False, 'model_qualified': False,
        'TRAIN_only_not_DEV_FINAL_or_native_qualification': True,
        'quality_stop': not passed, 'earlier_state_fallback_or_retry_authorized': False}
