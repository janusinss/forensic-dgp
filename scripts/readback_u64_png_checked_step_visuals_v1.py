"""Read back PNG-checked exact visual crops and saved endpoint diagnosis only."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat

import importlib.abc
import math
import os
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_audit'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_png_checked_step_v1_vm'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
OUT = 'outputs/u64_png_checked_step_v1'
PROTOCOL_SHA = '7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def relative(name):
    need(type(name) is str and name and '\\' not in name and ':' not in name, 'Canonical relative path')
    p = PurePosixPath(name)
    need(not p.is_absolute() and p.as_posix() == name and not any(v in ('', '.', '..') for v in p.parts), 'Confined path')
    return name


class Reader:
    def __init__(self):
        self.files = {}
    def read(self, path, expected=None, maximum=8 * 1024**2):
        path = path.absolute()
        need(path.is_relative_to(ROOT), 'Workspace-bound read')
        for parent in (path.parent, *path.parent.parents):
            value = parent.lstat()
            need(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode)
                 and not getattr(value, 'st_file_attributes', 0) & 1024, 'Ordinary ancestry')
        before = path.lstat()
        need(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode) and before.st_nlink == 1
             and not getattr(before, 'st_file_attributes', 0) & 1024 and 0 <= before.st_size <= maximum, 'Bounded single-link file')
        raw = path.read_bytes(); after = path.stat()
        need((before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns), 'Changed while read')
        measured = {'bytes': len(raw), 'sha256': sha(raw)}
        if expected is not None:
            need(measured == {key: expected[key] for key in ('bytes', 'sha256')}, 'Exact byte pin differs')
        name = path.relative_to(ROOT).as_posix()
        need(name not in self.files or self.files[name] == measured, 'Consumed file changed')
        self.files[name] = measured
        return raw
    def after(self):
        files = dict(self.files)
        for name, value in files.items():
            self.read(ROOT / name, value)
        need(files == self.files, 'Consumed namespace stable')
        return [{'path': name, **value} for name, value in sorted(files.items())]


def image(raw, size):
    with Image.open(io.BytesIO(raw)) as value:
        need(value.format == 'PNG' and value.mode == 'RGB' and value.size == size
             and getattr(value, 'n_frames', 1) == 1, 'Exact RGB PNG size')
        return value.copy()


class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        need(fullname.split('.')[0] not in {'torch', 'torchvision', 'tensorflow', 'jax', 'scipy'},
             'Neural/scientific framework import forbidden')
        return None


def compare(actual, expected, path='', bound=5e-17):
    """Rounding bound for diagnostic readback only; quality remains zero-tolerance."""
    if type(expected) is dict:
        need(type(actual) is dict and set(actual) == set(expected), 'Exact diagnostic keys: ' + path)
        for key in expected:
            compare(actual[key], expected[key], path + '/' + key, bound)
    elif type(expected) is list:
        need(type(actual) is list and len(actual) == len(expected), 'Exact diagnostic list: ' + path)
        for i, value in enumerate(expected):
            compare(actual[i], value, path + '/' + str(i), bound)
    elif type(expected) is float:
        need(type(actual) in (int, float) and math.isfinite(actual)
             and abs(actual - expected) <= bound, 'Diagnostic floating readback: ' + path)
    else:
        need(type(actual) is type(expected) and actual == expected, 'Exact diagnostic value: ' + path)


def ordered_mean(values):
    total = 0.
    for value in values:
        total += float(value)
    return total / len(values)


def adjacent(value, target):
    residual = value.astype(np.float64) - target.astype(np.float64)
    horizontal = np.abs(residual[:, 1:, :] - residual[:, :-1, :])
    vertical = np.abs(residual[1:, :, :] - residual[:-1, :, :])
    return float((horizontal.sum(dtype=np.float64) + vertical.sum(dtype=np.float64)) /
                 (horizontal.size + vertical.size))


def run():
    sys.meta_path.insert(0, BlockFrameworks())
    r = Reader(); r.read(Path(__file__).resolve())
    bindings_path, diagnosis_path = AUDIT / 'visual_bindings_v1.json', AUDIT / 'endpoint_diagnosis_v1.json'
    bindings_raw, diagnosis_raw = r.read(bindings_path), r.read(diagnosis_path)
    bindings, diagnosis = json.loads(bindings_raw), json.loads(diagnosis_raw)
    inventory = json.loads(r.read(AUDIT / 'inventory_v1.json'))
    need(inventory['complete'] is True and inventory['protocol_sha256'] == PROTOCOL_SHA, 'Bound complete returned inventory')
    protocol_raw = r.read(STAGE / 'protocol.json'); need(sha(protocol_raw) == PROTOCOL_SHA, 'Exact sealed protocol')
    protocol = json.loads(protocol_raw)
    parent_raw = r.read(PARENT / 'protocol.json'); need(sha(parent_raw) == PARENT_SHA, 'Exact parent protocol')
    parent = json.loads(parent_raw)
    need(bindings['complete'] is True and bindings['planned_pages'] == len(bindings['pages']) == 21
         and len(bindings['cells']) == 664 and len(bindings['exact_repeated_witness_aliases']) == 4,
         'Exact21-page664-cell4-alias visual census')
    need(bindings['selected_trial_index'] == 0 and bindings['selected_trial_quality_failed'] is True
         and bindings['monitor_based_candidate_reselection_permitted'] is False and bindings['model_qualified'] is False,
         'Failed selected trial remains binding')
    need(diagnosis['complete'] is True and diagnosis['saved_endpoint_diagnosis_pass'] is True
         and diagnosis['quality_threshold_or_quantizer_changed'] is False
         and diagnosis['monitor_fallback_authorized'] is False
         and diagnosis['actual_new_model_gradient_parameter_VM_calls'] == 0,
         'Saved diagnosis scope and unchanged policy')
    r.read(ROOT / relative(bindings['source']['path']), bindings['source'])
    r.read(ROOT / relative(diagnosis['source']['path']), diagnosis['source'])
    protected = {
        'outputs/codeformer_u64_png_checked_step_v1_return_pixels/receipt_v1.json':
            '7d325406da4a2a62dcbc9f0003f1412bf096c5a5f0edc66c9db0450319e80edf',
        'scripts/audit_u64_png_checked_step_return_pixels_v1.py':
            'beed3d45f5c197debc9f2a2ff8acb45f9de9cfa8bb0c50ba5f4add1f73ed433a'}
    for name, digest in protected.items():
        need(sha(r.read(ROOT / name)) == digest, 'Earlier independent saved-pixel artifact unchanged')
    evidence = Path(inventory['evidence_root']).resolve(strict=True)
    need(evidence == (AUDIT / 'extracted').resolve(strict=True), 'Exact inventory extraction root')
    def returned(name):
        relative(name)
        need(inventory['files'][name] == inventory['extracted_files'][name], 'Extracted/inventory exact pin join')
        return r.read(evidence / name, inventory['extracted_files'][name])
    records, state_records, quality, optimization = {}, {}, {}, {}
    for call in protocol['plan']:
        if call['kind'] != 'forward':
            continue
        name = OUT + '/calls/' + str(call['slot']).zfill(4) + '/record.json'
        need(name in inventory['extracted_files'], 'All436 actual record slots present')
        record = json.loads(returned(name)); need(record['call'] == call, 'Exact record/call join')
        records[call['slot']] = record
        phase = call['phase']
        if phase == 'deterministic_same_u64_repeat':
            continue
        if phase == 'initial_u64_quality_bank':
            state = 'original'; group = quality
        elif phase == 'selected_quality_bank':
            state = 'selected'; group = quality
        elif phase == 'initial_optimization_bank':
            state = 'original'; group = optimization
        else:
            need(phase == 'trial_optimization_bank', 'Known actual forward phase')
            state = 'trial_' + str(call['trial_index']); group = optimization
        need((call['id'], phase, call['trial_index']) not in state_records, 'Unique case/state output')
        state_records[(call['id'], phase, call['trial_index'])] = record
        group.setdefault(call['id'], {})[state] = record
    need(len(records) == 436 and len(quality) == 56 and len(optimization) == 64, 'Exact actual output/case census')
    specs = {row['id']: row for row in protocol['selection']['optimization'] +
             protocol['selection']['monitors'] + protocol['selection']['guards']}
    # Derive exact grid placements from immutable cohort order, independently of
    # the renderer's manifest. This also checks absent targets have no fake cell.
    expected_cells, expected_widths = {}, {}; page_count = 0
    groups = (
        ('monitor_guard', protocol['selection']['monitors'] + protocol['selection']['guards'],
         quality, ('original', 'selected')),
        ('optimization', protocol['selection']['optimization'], optimization,
         ('original', 'trial_0', 'trial_1', 'trial_2', 'trial_3')))
    for group_name, ordered, outputs, states in groups:
        for start in range(0, len(ordered), 6):
            page_count += 1; expected_widths[page_count] = 10 + 270 * (len(states) + 2)
            for line, row in enumerate(ordered[start:start + 6]):
                y = 100 + 288 * line
                cells = [(0, 'input', None, row['input'])]
                if row.get('target') is not None:
                    cells.append((1, 'target', None, row['target']))
                for col, state in enumerate(states, 2):
                    record = outputs[row['id']][state]
                    cells.append((col, state, record['call']['slot'], record['outputs']['png']))
                for col, state, slot, source in cells:
                    base = PARENT if slot is None else evidence
                    expected_cells[(page_count, col, y)] = {
                        'group': group_name, 'id': row['id'], 'state': state, 'slot': slot,
                        'source': {'path': (base / source['path']).relative_to(ROOT).as_posix(),
                                   **{key: source[key] for key in ('bytes', 'sha256')}}}
    need(page_count == 21 and len(expected_cells) == 664, 'Independent exact visual grid census')
    pages = {}
    for number, source in enumerate(bindings['pages'], 1):
        need(source['path'] == (AUDIT / 'visuals_v1' / ('page_' + str(number).zfill(2) + '.png')).relative_to(ROOT).as_posix(),
             'Exact page ordering and saved namespace')
        pages[number] = image(r.read(ROOT / relative(source['path']), source), (expected_widths[number], 1838))
    cache, cell_checks, output_slots, geometries = {}, [], [], set()
    for cell in bindings['cells']:
        number, col = cell['page'], cell['column']; x, y = cell['xy']
        geometry = (number, col, y)
        need(geometry in expected_cells and geometry not in geometries and x == 10 + 270 * col,
             'Exact unique immutable-grid cell coordinate')
        geometries.add(geometry); expected = expected_cells[geometry]
        need(all(cell[key] == expected[key] for key in ('group', 'id', 'state', 'slot', 'source')),
             'Exact case/state/source-to-grid binding')
        source = cell['source']; name = relative(source['path'])
        if name not in cache:
            cache[name] = image(r.read(ROOT / name, source), (256, 256))
        observed = cache[name]; crop = pages[number].crop((x, y, x + 256, y + 256))
        need(crop.tobytes() == observed.tobytes() and sha(crop.tobytes()) == cell['RGB8_sha256'],
             'Exact256 source RGB crop without resampling/enhancement')
        if cell['slot'] is None:
            reference = specs[cell['id']]; bound = reference[cell['state']]
            need(parent['files'][bound['path']] == {key: source[key] for key in ('bytes', 'sha256')}, 'Exact parent asset static pin')
            rgb_key = 'input_RGB8_sha256' if cell['state'] == 'input' else 'target_RGB8_sha256'
            need(sha(observed.tobytes()) == reference[rgb_key], 'Input/target RGB digest')
        else:
            output_slots.append(cell['slot'])
            record = records[cell['slot']]
            need(record['id'] == cell['id'] and record['outputs']['png']['sha256'] == source['sha256'], 'Exact actual returned output binding')
            returned(record['outputs']['png']['path'])
        cell_checks.append({'page': number, 'group': cell['group'], 'id': cell['id'], 'column': col,
            'state': cell['state'], 'slot': cell['slot'], 'xy': cell['xy'],
            'exact_bound_RGB256_crop': True, 'RGB8_sha256': cell['RGB8_sha256']})
    need(geometries == set(expected_cells) and len(output_slots) == len(set(output_slots)) == 432,
         'All664 expected cells and432 distinct actual outputs')
    aliases = []
    for row in bindings['exact_repeated_witness_aliases']:
        slot, previous = row['slot'], row['same_RGB_cell_as_slot']
        need(slot in (57, 58, 59, 60) and previous in output_slots, 'Exact four repeat slots and represented originals')
        now, original = records[slot], records[previous]
        need(now['id'] == original['id'] and now['call']['phase'] == 'deterministic_same_u64_repeat'
             and original['call']['phase'] == 'initial_u64_quality_bank', 'Repeat same-case phase binding')
        need(now['outputs']['png']['sha256'] == original['outputs']['png']['sha256'] == row['exact_same_PNG_sha256'],
             'Exact repeat PNG digest binding')
        need(returned(now['outputs']['png']['path']) == returned(original['outputs']['png']['path']), 'Actual repeat PNG byte identity')
        aliases.append({**row, 'actual_PNG_bytes_identical': True})
    need(set(output_slots) | {row['slot'] for row in aliases} == set(records), 'All436 output slots covered once or exact alias')
    for row in bindings['consumed_files']:
        r.read(ROOT / relative(row['path']), row)

    # Independently recompute the eight local-monitor central128 endpoints and
    # the integer change in absolute residual-edge code units.
    original_bank = json.loads(returned(OUT + '/banks/initial_quality.json'))
    selected_bank = json.loads(returned(OUT + '/banks/selected_quality.json'))
    cases = []
    local_rows = sorted((row for row in protocol['selection']['monitors'] if row['role'] == 'local_gaussian_1p2'), key=lambda row: row['id'])
    need(len(local_rows) == 8, 'Exactly eight immutable local monitors')
    for reference in local_rows:
        cid = reference['id']; target_binding = reference['target']
        target_full = np.asarray(image(r.read(PARENT / target_binding['path'], target_binding), (256, 256))).copy()
        target_codes = target_full[64:192, 64:192].astype(np.int32)
        target = target_codes.astype(np.float32) / np.float32(255)
        codes, scalars = [], []
        for bank, state in ((original_bank, 'original'), (selected_bank, 'selected')):
            record = quality[cid][state]; saved = bank['records'][cid]
            need(saved['outputs'] == record['outputs'] and saved['call'] == record['call'], 'Diagnosis bank/actual record output join')
            raw_bytes = returned(record['outputs']['raw']['path'])
            value = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
            need(value.dtype == np.float32 and value.shape == (256, 256, 3) and np.isfinite(value).all(), 'Saved FP32 RAW256 schema')
            png_value = np.asarray(image(returned(record['outputs']['png']['path']), (256, 256))).copy()
            need(np.array_equal((np.clip(value, 0, 1) * np.float32(255)).astype(np.uint8), png_value), 'Actual RAW floorPNG pair')
            code = png_value[64:192, 64:192]
            raw_scalar = adjacent(value[64:192, 64:192], target)
            png_scalar = adjacent(code.astype(np.float32) / np.float32(255), target)
            compare(raw_scalar, saved['regional_metrics']['raw']['central128']['adjacent_rgb_error'])
            compare(png_scalar, saved['regional_metrics']['native_png']['central128']['adjacent_rgb_error'])
            scalars.append((raw_scalar, png_scalar)); codes.append(code.astype(np.int32))
        code_change = codes[1] - codes[0]
        res0, res1 = codes[0] - target_codes, codes[1] - target_codes
        h0, h1 = np.abs(res0[:, 1:] - res0[:, :-1]), np.abs(res1[:, 1:] - res1[:, :-1])
        v0, v1 = np.abs(res0[1:] - res0[:-1]), np.abs(res1[1:] - res1[:-1])
        edge_changes = (h1 - h0, v1 - v0)
        cases.append({'id': cid, 'original_RAW_adjacent': scalars[0][0], 'selected_RAW_adjacent': scalars[1][0],
            'original_PNG_adjacent': scalars[0][1], 'selected_PNG_adjacent': scalars[1][1],
            'RAW_delta': scalars[1][0] - scalars[0][0], 'PNG_delta': scalars[1][1] - scalars[0][1],
            'changed_RGB_components': int(np.count_nonzero(code_change)), 'RGB_components': int(code_change.size),
            'positive_code_crossings': int(np.count_nonzero(code_change > 0)), 'negative_code_crossings': int(np.count_nonzero(code_change < 0)),
            'max_absolute_code_delta': int(np.abs(code_change).max()),
            'integer_residual_edge_increased_units': int(sum(value[value > 0].sum(dtype=np.int64) for value in edge_changes)),
            'integer_residual_edge_decreased_units': int(sum(value[value < 0].sum(dtype=np.int64) for value in edge_changes)),
            'integer_residual_edge_net_units': int(sum(value.sum(dtype=np.int64) for value in edge_changes)),
            'edge_components': int(sum(value.size for value in edge_changes))})
    p0 = ordered_mean([row['original_PNG_adjacent'] for row in cases])
    p1 = ordered_mean([row['selected_PNG_adjacent'] for row in cases])
    net = sum(row['integer_residual_edge_net_units'] for row in cases)
    prediction = net / (255 * sum(row['edge_components'] for row in cases))
    summary = {'original_PNG_adjacent': p0, 'selected_PNG_adjacent': p1, 'PNG_delta': p1 - p0,
        'PNG_relative_error_increase_percent': (p1 / p0 - 1) * 100,
        'RAW_delta': ordered_mean([row['selected_RAW_adjacent'] for row in cases]) - ordered_mean([row['original_RAW_adjacent'] for row in cases]),
        'changed_RGB_components': sum(row['changed_RGB_components'] for row in cases),
        'RGB_components': sum(row['RGB_components'] for row in cases),
        'integer_residual_edge_increased_units': sum(row['integer_residual_edge_increased_units'] for row in cases),
        'integer_residual_edge_decreased_units': sum(row['integer_residual_edge_decreased_units'] for row in cases),
        'integer_residual_edge_net_units': net, 'integer_code_predicted_delta': prediction,
        'FP32_normalization_minus_integer_prediction': p1 - p0 - prediction,
        'monitor_cases_RAW_improved': sum(row['RAW_delta'] < 0 for row in cases),
        'monitor_cases_PNG_worsened': sum(row['PNG_delta'] > 0 for row in cases)}
    compare(cases, diagnosis['cases']); compare(summary, diagnosis['summary'])
    need(summary['changed_RGB_components'] == 882 and summary['integer_residual_edge_increased_units'] == 1752
         and summary['integer_residual_edge_decreased_units'] == -1727 and net == 25
         and summary['PNG_delta'] == 1.2564461624939294e-7
         and summary['monitor_cases_RAW_improved'] == 8 and summary['monitor_cases_PNG_worsened'] == 5,
         'Independent actual integer/endpoint diagnosis counts')
    for row in diagnosis['consumed_before_after_exact']:
        r.read(ROOT / relative(row['path']), row)
    need(not any(name.split('.')[0] in {'torch', 'torchvision', 'tensorflow', 'jax', 'scipy'} for name in sys.modules), 'No neural framework loaded')
    files = r.after()
    result = {'complete': True, 'visual_readback_pass': True, 'unresolved_blockers': [],
        'protocol_sha256': PROTOCOL_SHA, 'planned_pages': 21, 'readback_pages': 21,
        'exact_RGB256_cells': len(cell_checks), 'distinct_actual_output_cells': 432,
        'repeated_witness_aliases': aliases, 'all436_actual_outputs_covered_once_or_exact_alias': True,
        'all_input_target_output_cells_match_immutable_case_state_grid': True,
        'all_cells_without_resampling_or_enhancement': True,
        'endpoint_diagnosis_independently_recomputed': True, 'endpoint_summary': summary,
        'eight_local_monitor_cases': cases, 'integer_code_units_compared_exactly': True,
        'diagnostic_float_readback_bound_only': 5e-17, 'quality_tolerance': 0,
        'earlier_saved_pixel_receipt_and_source_unchanged': True,
        'visual_bindings_bound': {'path': bindings_path.relative_to(ROOT).as_posix(), 'bytes': len(bindings_raw), 'sha256': sha(bindings_raw)},
        'endpoint_diagnosis_bound': {'path': diagnosis_path.relative_to(ROOT).as_posix(), 'bytes': len(diagnosis_raw), 'sha256': sha(diagnosis_raw)},
        'cell_checks': cell_checks, 'consumed_files': files, 'all_consumed_before_after_exact': True,
        'root_subjective_visual_review_performed_by_this_audit': False,
        'root_visual_review_receipt_consumed': False, 'subjective_review_claimed': False,
        'readback_scope': 'Artifact-only independent21-page664 exact RGB256 crop/grid/output-pin coverage and eight local-monitor RAW/PNG/absolute residual-edge integer diagnosis. No subjective sheet review is claimed, regardless of concurrent root review. No root_visual_review receipt is consumed.',
        'monitor_fallback_or_new_candidate_authorized': False,
        'threshold_quantizer_or_renderer_changed': False, 'model_qualified': False,
        'no_model_inference_neural_gradient_optimizer_parameter_training_VM_browser_calls': True,
        'source': {'path': Path(__file__).relative_to(ROOT).as_posix(), **r.files[Path(__file__).relative_to(ROOT).as_posix()]}}
    destination = AUDIT / 'visual_readback_v1.json'
    with destination.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'visual_readback_pass': True, 'pages': 21, 'cells': len(cell_checks), 'outputs': 432,
        'aliases': len(aliases), 'consumed_files': len(files), 'endpoint_summary': summary,
        'subjective_review_claimed': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(run())


