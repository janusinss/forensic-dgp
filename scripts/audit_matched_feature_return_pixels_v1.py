"""Saved-array arithmetic and unchanged gates only; no neural execution.

Consumes the completed independently retained return. It never imports a
returned source, framework, model or worker and never alters model/data/gates.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.abc
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_matched_feature_learning_v1_return_audit_v1'
STAGE = 'cctv_codeformer_matched_feature_learning_v1_vm'
STATIC = ROOT / 'outputs' / STAGE
RETAINED = OUT / 'retained' / STAGE
RUN = 'outputs/matched_feature_learning_v1'
PROTOCOL_SHA = 'afd59b90951138f94f0fcf0dea6ef308f8860042ae3c4f25b43c279d13d0fbd8'
ARCHIVE_SHA = 'de7639af214d80be78ca73b24a3942afbf41223dae94591ce0e14076c079b860'
MAX_SECONDS = 600
MAX_MEMBER = 384 * 1024**2
MAX_ARRAY_BYTES = 4 * 1024**2
MAX_ARRAY_HEADER = 16 * 1024
ARRAY_SHAPES = frozenset(((256,256,3), (1,3,256,256), (1,3,512,512),
                          (1,256,1024), (1,256,16,16)))
FORBIDDEN = {'torch', 'torchvision', 'torchaudio', 'tensorflow', 'jax', 'jaxlib', 'keras'}


class ImportBlocker(importlib.abc.MetaPathFinder):
    attempted = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN or fullname.endswith(('.model', '.worker')):
            self.attempted.append(fullname)
            raise RuntimeError('Forbidden pixel-audit import: ' + fullname)
        return None


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def strict_json(raw):
    need(type(raw) is bytes and len(raw) <= 32 * 1024**2, 'Bounded JSON bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key refused')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON token refused: ' + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    def walk(item, depth=0):
        need(depth <= 64, 'Bounded JSON nesting required')
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON scalar required')
        elif type(item) is dict:
            for child in item.values(): walk(child, depth + 1)
        elif type(item) is list:
            for child in item: walk(child, depth + 1)
    walk(result)
    return result


def relative(value):
    need(type(value) is str and value and '\\' not in value and ':' not in value and
         all(ord(c) >= 32 for c in value), 'Canonical relative path required')
    parsed = PurePosixPath(value)
    need(not parsed.is_absolute() and parsed.as_posix() == value and value != '.' and
         all(p not in ('', '.', '..') for p in parsed.parts), 'Path escape or alias refused')
    return value


def ordinary(path):
    path = Path(path).absolute()
    need(path.is_relative_to(ROOT), 'Audit input outside workspace')
    for parent in path.parents:
        info = parent.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
             not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary input ancestry required')
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and info.st_nlink == 1 and
         not getattr(info, 'st_file_attributes', 0) & 1024 and 0 <= info.st_size <= MAX_MEMBER,
         'Bounded ordinary single-link audit input required')
    return path


def pixel_summary(raw, png, *, np):
    """Bind saved RAW/PNG geometry and floor delivery without repairing pixels."""
    need(isinstance(raw, np.ndarray) and raw.dtype == np.float32 and
         raw.shape == (256, 256, 3) and np.isfinite(raw).all() and
         raw.min() >= 0 and raw.max() <= 1, 'Finite FP32 RAW HWC256 in [0,1] required')
    need(isinstance(png, np.ndarray) and png.dtype == np.uint8 and
         png.shape == raw.shape, 'Actual RGB8 HWC256 PNG required')
    need(np.array_equal((np.clip(raw, 0, 1) * 255).astype(np.uint8), png),
         'Delivered PNG is not exact floor-quantized RAW')
    return {'RAW_pixel_bytes_sha256': sha(raw.tobytes()), 'PNG_RGB8_sha256': sha(png.tobytes())}


def unclip_shape(value, *, np):
    need(isinstance(value, np.ndarray) and value.dtype == np.float32 and
         value.shape == (1, 3, 256, 256) and np.isfinite(value).all(),
         'Finite actual unclipped FP32 NCHW256 required')


def decode_array_bytes(raw, *, np):
    """Bound NPY version/header/schema/payload before NumPy allocates an array."""
    need(type(raw) is bytes and 10 <= len(raw) <= MAX_ARRAY_BYTES + MAX_ARRAY_HEADER + 12 and
         raw[:6] == b'\x93NUMPY', 'Bounded numerical NPY bytes required')
    version = tuple(raw[6:8])
    need(version in ((1,0),(2,0)), 'Only bounded NPY1/2 formats are admitted')
    prefix = 10 if version == (1,0) else 12
    need(len(raw) >= prefix, 'Complete NPY header prefix required')
    header_size = struct.unpack('<H' if version == (1,0) else '<I', raw[8:prefix])[0]
    need(0 < header_size <= MAX_ARRAY_HEADER and prefix + header_size <= len(raw),
         'Bounded complete NPY header required before parsing')
    stream = io.BytesIO(raw)
    need(np.lib.format.read_magic(stream) == version, 'Exact NPY magic version required')
    reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream, max_header_size=MAX_ARRAY_HEADER)
    need(type(shape) is tuple and shape in ARRAY_SHAPES and all(type(x) is int for x in shape) and
         fortran is False and dtype == np.dtype('float32') and not dtype.hasobject,
         'Exact permitted C-order floating array schema required before allocation')
    expected_bytes = math.prod(shape) * dtype.itemsize
    need(0 < expected_bytes <= MAX_ARRAY_BYTES and stream.tell() == prefix + header_size and
         len(raw) == stream.tell() + expected_bytes, 'Exact bounded NPY payload, no trailing bytes, required')
    stream.seek(0)
    value = np.load(stream, allow_pickle=False)
    need(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype and
         stream.read(1) == b'', 'Exactly one non-pickled NumPy array required')
    return value


def arithmetic_close(actual, reference):
    return (type(actual) in (float, int) and type(reference) in (float, int) and
            math.isfinite(actual) and math.isfinite(reference) and
            abs(actual - reference) <= 2e-7 + 2e-6 * abs(reference))


def assert_subset(actual, saved, label):
    need(type(actual) is dict and type(saved) is dict, label + ': exact record objects required')
    for key, value in actual.items():
        need(key in saved and canonical(value) == canonical(saved[key]),
             label + ': recomputed field differs: ' + key)


def parity_equal(current, reference):
    return (current['stock'] == reference['stock'] and
            current['RAW_pixel_bytes_sha256'] == reference['RAW_pixel_bytes_sha256'] and
            current['PNG_RGB8_sha256'] == reference['PNG_RGB8_sha256'] and
            current['outputs']['unclipped']['sha256'] == reference['outputs']['unclipped']['sha256'] and
            current['input'] == reference['input'] and current['target'] == reference['target'])


def failed_checks(value, path=''):
    """Report failed leaves only; ancestor summaries are not separate failures."""
    results = []
    if type(value) is dict:
        children = []
        for key, child in value.items():
            if type(child) in (dict, list):
                children.extend(failed_checks(child, path + '/' + key))
        if value.get('pass') is False and not children:
            results.append({'path': path, **copy.deepcopy(value)})
        results.extend(children)
    elif type(value) is list:
        for position, child in enumerate(value):
            results.extend(failed_checks(child, path + '/' + str(position)))
    return results


def audit():
    started = time.monotonic()
    blocker = ImportBlocker(); sys.meta_path.insert(0, blocker)
    consumed = {}
    members = None
    def check_clock():
        need(time.monotonic() - started < MAX_SECONDS, 'Finite600s saved-pixel audit stop')
    def pin(path):
        check_clock(); path = ordinary(path); before = path.stat(); raw = path.read_bytes(); after = path.stat()
        need((before.st_size, before.st_ino, before.st_mtime_ns) ==
             (after.st_size, after.st_ino, after.st_mtime_ns), 'Consumed file changed during read')
        result = {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': sha(raw)}
        if members is not None and path.is_relative_to(RETAINED):
            name = path.relative_to(RETAINED).as_posix()
            need(name in members and members[name] == {key: result[key] for key in ('bytes','sha256')},
                 'Retained pixel-audit input differs from authenticated archive member')
        old = consumed.get(result['path']); need(old is None or old == result, 'Consumed input changed between uses')
        consumed[result['path']] = result
        return raw, result
    def doc(path): return strict_json(pin(path)[0])
    transport = doc(OUT / 'transport.json')
    need(transport.get('complete') is True and transport.get('archive', {}).get('sha256') == ARCHIVE_SHA and
         transport.get('protocol', {}).get('sha256') == PROTOCOL_SHA,
         'Completed exact return transport audit required before any array decode')
    index_raw, index_pin = pin(OUT / 'member_index.json')
    need(index_pin == transport['member_index'], 'Exact transport member-index binding required')
    index = strict_json(index_raw)
    need(index['complete'] is True and index['archive_sha256'] == ARCHIVE_SHA, 'Authenticated complete archive index required')
    members = index['members']
    # The independent transport owns archive authentication, extraction namespace
    # and static hashes. This audit separately hashes every actually consumed file.
    protocol_raw, protocol_pin = pin(STATIC / 'protocol.json')
    need(protocol_pin['sha256'] == PROTOCOL_SHA, 'Exact original sealed protocol required')
    protocol = strict_json(protocol_raw)
    for relative_path, fixed in protocol['files'].items():
        if relative_path.startswith(('research/', 'third_party/')) and relative_path.endswith('.py'):
            workspace = ROOT / relative_path
            if relative_path in ('research/__init__.py', 'third_party/__init__.py', 'third_party/codeformer/__init__.py'):
                # Workspace package docstrings differ from their retained stage
                # counterparts and are separately authenticated by joined review.
                continue
            _, measured = pin(workspace)
            need({key: measured[key] for key in ('bytes', 'sha256')} == fixed,
                 'Trusted local source differs from sealed helper: ' + relative_path)
    review = doc(ROOT / 'outputs/codeformer_matched_feature_learning_v1_source_preparation/runner_review/complete_runner_source_review_v1.json')
    for name in ('research/__init__.py', 'third_party/__init__.py', 'third_party/codeformer/__init__.py'):
        _, measured = pin(ROOT / name)
        need({key: measured[key] for key in ('bytes', 'sha256')} == review['source_pins'][name],
             'Workspace package initializer differs from joined review')
    sys.path.insert(0, str(ROOT))
    from research.codeformer_matched_feature_learning_v1 import contract, records, quality, objective
    import numpy as np
    from PIL import Image
    for module in (contract, records, quality, objective):
        need(Path(module.__file__).resolve() == ROOT / 'research/codeformer_matched_feature_learning_v1' /
             (module.__name__.split('.')[-1] + '.py'), 'Unexpected trusted local module origin')
    def resolve(binding):
        need(type(binding) is dict and set(binding) == {'path', 'bytes', 'sha256'}, 'Exact saved artifact binding required')
        name = relative(binding['path'])
        path = (STATIC if name in protocol['files'] else RETAINED) / name
        raw, measured = pin(path)
        need((measured['bytes'], measured['sha256']) == (binding['bytes'], binding['sha256']),
             'Saved artifact hash mismatch: ' + name)
        return raw
    def static_doc(field): return resolve(protocol[field])
    selection = contract.select_cases(static_doc('admission'), static_doc('candidate_data'), static_doc('parent_protocol'))
    need(selection == protocol['selection'], 'Exact admitted selection must be rederived')
    plan = contract.operation_plan(selection)
    def rgb(binding):
        with Image.open(io.BytesIO(resolve(binding))) as image:
            need(image.mode == 'RGB' and image.size == (256, 256) and getattr(image, 'n_frames', 1) == 1,
                 'Exact decoded single-frame RGB256 required')
            value = np.asarray(image).copy()
        return value
    def array(binding):
        need(type(binding.get('bytes')) is int and 0 < binding['bytes'] <=
             MAX_ARRAY_BYTES + MAX_ARRAY_HEADER + 12, 'Bounded array file pin required before read')
        return decode_array_bytes(resolve(binding), np=np)
    controls = {row['id']: row for row in protocol['controls']}
    parity = {row['id']: row for row in protocol['u64_parity']}
    frozen = {}; frozen_pixels = {}; teacher_pixels = {}
    for source in selection['old_quality']:
        row = dict(source, bank='old_quality', case_key='old_quality:' + source['id'])
        control = controls[row['id']]
        raw, png = array(control['raw']), rgb(control['png']); pixel_summary(raw, png, np=np)
        target = None if row['target'] is None else rgb(row['target']).astype(np.float32) / np.float32(255)
        frozen_pixels[row['id']] = raw, png
        kwargs = {}
        if row.get('behavior_teacher') is not None:
            teacher = row['behavior_teacher']; pair = array(teacher['raw']), rgb(teacher['png'])
            teacher_pixels[row['id']] = pair; kwargs = {'teacher_raw': pair[0], 'teacher_png': pair[1]}
        measured = records.record_inherited(raw, png, target, row, np=np,
            frozen_U96_raw=raw, frozen_U96_png=png, **kwargs)
        saved = strict_json(resolve(control['record']))
        need(saved['metrics'] == measured['metrics'] and saved['regional_metrics'] == measured['regional_metrics'],
             'Frozen U96 saved original scalar metrics differ from actual pixels')
        frozen[row['id']] = measured
    frozen_return = doc(RETAINED / RUN / 'frozen_U96_control_metrics.json')
    for key, record in frozen.items(): assert_subset(record, frozen_return['records'][key], 'Frozen U96 ' + key)
    output_root = RETAINED / RUN
    call_paths = sorted((output_root / 'calls').glob('*/record.json'))
    need(len(call_paths) == 712, 'Exact returned712 call records required for this run')
    calls = {}; by_scope = {}; summary = {'all_RAW_PNG_floor_pairs': 0, 'actual_stock_arrays_verified': 0,
        'A0_B0_exact_pairs': 0, 'self_repeat_exact_pairs': 0, 'historical_RAW_PNG_exact_pairs': 0,
        'objective_references_recomputed': 0, 'connected_objective_arithmetic_comparisons': 0}
    objective_rows = []
    for path in call_paths:
        saved = doc(path); call = saved['call']; slot = call['slot']
        need(type(slot) is int and 1 <= slot <= len(plan) and call == plan[slot - 1] and
             path.parent.name == str(slot).zfill(4) and slot not in calls, 'Exact unique planned scope-qualified call required')
        row = contract.resolve_case(selection, call); row.update(bank=call['bank'], case_key=call['case_key'])
        assert_subset({key: row[key] for key in ('id','reference_id','role','split','input','target','bank','case_key')}, saved,
                      'Saved call metadata ' + str(slot))
        raw, png, unclipped = array(saved['outputs']['raw']), rgb(saved['outputs']['png']), array(saved['outputs']['unclipped'])
        expected_stem = RUN + '/calls/' + str(slot).zfill(4)
        need({key: binding['path'] for key, binding in saved['outputs'].items()} ==
             {'raw': expected_stem + '/raw.npy', 'png': expected_stem + '/result.png',
              'unclipped': expected_stem + '/unclipped.npy'}, 'Exact slot-owned output namespace required')
        hashes = pixel_summary(raw, png, np=np); unclip_shape(unclipped, np=np)
        assert_subset(hashes, saved, 'Saved pixel digest ' + str(slot)); summary['all_RAW_PNG_floor_pairs'] += 1
        observed = rgb(row['input'])
        need(sha(observed.tobytes()) == row.get('prepared_input_RGB8_sha256', row.get('input_RGB8_sha256')),
             'Exact admitted source input RGB digest required')
        target = None if row['target'] is None else rgb(row['target'])
        if target is not None:
            need(sha(target.tobytes()) == row.get('prepared_target_RGB8_sha256', row.get('target_RGB8_sha256')),
                 'Exact photographic source target RGB digest required')
            target = target.astype(np.float32) / np.float32(255)
        if call['bank'] == 'old_quality':
            kwargs = {}; fr, fp = frozen_pixels[row['id']]
            if row['id'] in teacher_pixels:
                pair = teacher_pixels[row['id']]; kwargs = {'teacher_raw': pair[0], 'teacher_png': pair[1]}
            measured = records.record_inherited(raw, png, target, row, np=np, frozen_U96_raw=fr, frozen_U96_png=fp, **kwargs)
        else:
            measured = records.record_new(raw, png, target, row, np=np)
        assert_subset(measured, saved, 'Actual recomputed metrics ' + str(slot))
        for name, binding in saved['stock_files'].items():
            value = array(binding); stock = saved['stock'][name]
            need(value.dtype == np.float32 and np.isfinite(value).all() and
                 list(value.shape) == stock['shape'] and str(value.dtype) == stock['dtype'] and
                 value.nbytes == stock['bytes'] and sha(value.tobytes()) == stock['bytes_sha256'],
                 'Actual saved native/logits/LQ witness differs')
            summary['actual_stock_arrays_verified'] += 1
        copied = copy.deepcopy(saved)
        # Only actual recomputed metric fields enter independent gate banks.
        for key, value in measured.items(): copied[key] = value
        calls[slot] = copied
        if call['phase'] != 'same_state_deterministic_repeat':
            key = (call['arm'], call['phase'], call['new_update'], call['bank'], call['id'])
            need(key not in by_scope, 'Duplicate arm/bank/phase record refused')
            by_scope[key] = copied
        if call['backward']:
            obj = doc(path.parent / 'objective.json')
            need(obj['call'] == call and obj['arm'] == call['arm'] and obj['update'] == call['new_update'] and
                 obj['divisor_applied_once'] == 5 and obj['quality_tolerance_added'] is False,
                 'Objective must bind the exact paired optimization call')
            objective_value, terms = objective.objective_numpy(raw, target, row['role'], np=np)
            need(obj['NumPy_objective'] == objective_value and obj['NumPy_terms'] == terms,
                 'Saved NumPy objective differs from saved RAW photographic target')
            need(arithmetic_close(obj['objective'], objective_value) and
                 all(arithmetic_close(obj['terms'][key], terms[key]) for key in objective.TERM_NAMES) and
                 arithmetic_close(obj['divided_objective'], obj['objective'] / 5),
                 'Connected FP32 objective exceeds declared arithmetic-only comparison policy')
            objective_rows.append({'slot': slot, 'arm': call['arm'], 'update': call['new_update'],
                'case_key': call['case_key'], 'unscaled_NumPy_objective': objective_value,
                'connected_reported_objective': obj['objective'], 'components': terms})
            summary['objective_references_recomputed'] += 1
            summary['connected_objective_arithmetic_comparisons'] += 1
        if call['phase'] == 'initial_quality' and call['bank'] == 'old_quality' and row['id'] in parity:
            original = parity[row['id']]
            need(np.array_equal(raw, array(original['raw'])) and np.array_equal(png, rgb(original['png'])),
                 'Historical four terminal-u64 RAW/PNG witness parity differs')
            summary['historical_RAW_PNG_exact_pairs'] += 1
        if call['phase'] == 'same_state_deterministic_repeat':
            initial = by_scope[(call['arm'], 'initial_quality', 0, call['bank'], call['id'])]
            need(parity_equal(saved, initial), 'Actual current self-repeat parity differs')
            summary['self_repeat_exact_pairs'] += 1
        elif call['arm'] == 'B' and call['new_update'] == 0:
            initial = by_scope[('A', call['phase'], 0, call['bank'], call['id'])]
            need(parity_equal(saved, initial), 'Actual A0/B0 scope-matched saved parity differs')
            summary['A0_B0_exact_pairs'] += 1
        if len(calls) % 100 == 0: print(json.dumps({'pixels_audited': len(calls), 'of': len(call_paths)}), flush=True)
    need(summary['self_repeat_exact_pairs'] == 8 and summary['historical_RAW_PNG_exact_pairs'] == 8 and
         summary['A0_B0_exact_pairs'] == 216 and summary['objective_references_recomputed'] == 80,
         'Exact returned parity/objective scope differs')
    gates = {}; bank_checks = []
    for arm in contract.ARMS:
        initial = {bank: {key[-1]: record for key, record in by_scope.items() if
            key[:4] == (arm, 'initial_quality', 0, bank)} for bank in ('old_quality','new_monitors')}
        for update in (0, 8):
            bank = initial if update == 0 else {scope: {key[-1]: record for key, record in by_scope.items()
                if key[:4] == (arm, 'quality_snapshot', update, scope)} for scope in ('old_quality','new_monitors')}
            inherited = quality.evaluate_inherited(bank['old_quality'], initial['old_quality'], frozen,
                static_doc('parent_protocol'), static_doc('historical_pixel_audit'), update=update)
            supplemental = quality.evaluate_new_bank(bank['new_monitors'], initial['new_monitors'], selection, update=update)
            gate = {'pass': inherited['pass'] and supplemental['pass'], 'update': update,
                'inherited': inherited, 'supplemental': supplemental,
                'same_original_references_and_zero_added_quality_tolerance': True, 'model_qualified': False}
            saved_bank = doc(output_root / 'arms' / arm / 'banks' / ('u' + str(update).zfill(4)) / 'bank.json')
            need(canonical(saved_bank['gate']) == canonical(gate), 'Recomputed complete gate differs from saved terminal gate')
            for scope, values in bank.items():
                need(set(saved_bank['records'][scope]) == set(values), 'Complete saved bank membership differs')
                for key, record in values.items():
                    need(canonical({k:v for k,v in record.items() if k != 'receipt'}) ==
                         canonical({k:v for k,v in saved_bank['records'][scope][key].items() if k != 'receipt'}),
                         'Bank record differs from actual independently read call record')
            gates[arm + ':' + str(update)] = gate
            bank_checks.append({'arm': arm, 'update': update, 'pass': gate['pass'],
                'inherited_pass': inherited['pass'], 'supplemental_pass': supplemental['pass'],
                'inherited_failures': failed_checks(inherited), 'supplemental_failures': failed_checks(supplemental),
                'unbound_original_terminal_1percent_status': inherited['original_terminal']})
    snapshots = {'A': [0, 8], 'B': [0, 8]}
    need(all(gates[arm + ':0']['pass'] is True and gates[arm + ':8']['pass'] is False for arm in contract.ARMS),
         'Independent actual gates must retain the observed initial pass/update8 stops')
    expected_slots = {call['slot'] for call in plan if call['new_update'] == 0 or
        call['backward'] and call['new_update'] <= 8 or call['phase'] == 'quality_snapshot' and call['new_update'] == 8}
    need(set(calls) == expected_slots, 'Saved call coverage differs from exact both-arm stop-at8 scope')
    for name, binding in list(consumed.items()):
        _, measured = pin(ROOT / name); need(measured == binding, 'Consumed file changed at final readback')
    need(not blocker.attempted and not FORBIDDEN & {name.split('.')[0] for name in sys.modules},
         'No forbidden neural framework/model/worker may be imported')
    return {'complete': True, 'status': 'INDEPENDENT_SAVED_PIXELS_AND_GATES_VERIFIED_BOTH_ARMS_STOP_AT8',
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'seconds': time.monotonic() - started,
        'source': {'path': Path(__file__).relative_to(ROOT).as_posix(), 'bytes': len(Path(__file__).read_bytes()),
                   'sha256': sha(Path(__file__).read_bytes())},
        'transport_required_complete': True, 'archive_expected_sha256': ARCHIVE_SHA,
        'protocol_sha256': PROTOCOL_SHA, 'saved_call_count': len(calls), 'checks': summary,
        'actual_quality_snapshots': snapshots, 'unexecuted_checkpoints': [24, 48],
        'no_terminal_optimization120_readouts_per_arm': True,
        'gates': gates, 'gate_summary': bank_checks, 'training_objectives': objective_rows,
        'consumed_files': consumed, 'all_consumed_before_after_exact': True,
        'quality_gate_tolerance_added': False, 'arithmetic_only_objective_policy': {'atol': 2e-7, 'rtol': 2e-6},
        'stored_stock_metadata_and_witnesses_not_model_reexecution': True,
        'native_DEV_FINAL_or_independent_identity_evidence': False,
        'forbidden_import_attempts': blocker.attempted, 'new_neural_gradient_optimizer_training_VM_calls': 0,
        'model_qualified': False, 'promotion_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--audit', action='store_true')
    args = parser.parse_args(); need(args.audit, 'Explicit saved-pixel --audit required')
    target = OUT / 'pixels.json'; need(not target.exists(), 'Prior pixel audit is immutable')
    result = audit()
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': result['complete'], 'saved_calls': result['saved_call_count'],
                      'receipt': str(target), 'model_qualified': False}))


if __name__ == '__main__':
    main()
