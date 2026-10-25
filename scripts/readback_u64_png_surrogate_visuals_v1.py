"""Read back exact RGB visual cells and output coverage, with no inference."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
OUT = 'outputs/u64_png_surrogate_measurement_v1'
PROTOCOL_SHA = '0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'
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


def run():
    r = Reader()
    r.read(Path(__file__).resolve())
    bindings_path = AUDIT / 'visual_bindings_v1.json'
    bindings_raw = r.read(bindings_path); bindings = json.loads(bindings_raw)
    review_path = AUDIT / 'root_visual_review_v1.json'
    review_raw = r.read(review_path); review = json.loads(review_raw)
    inventory = json.loads(r.read(AUDIT / 'inventory_v1.json'))
    protocol_raw = r.read(STAGE / 'protocol.json'); need(sha(protocol_raw) == PROTOCOL_SHA, 'Exact sealed protocol')
    protocol = json.loads(protocol_raw)
    parent_raw = r.read(PARENT / 'protocol.json'); need(sha(parent_raw) == PARENT_SHA, 'Exact parent protocol')
    parent = json.loads(parent_raw)
    need(bindings['complete'] and bindings['planned_pages'] == 12 and len(bindings['pages']) == 12
         and len(bindings['cells']) == 204 and len(bindings['exact_repeated_witness_aliases']) == 4, 'Exact12/204/4 visual census')
    need(review['complete'] and review['root_personally_inspected_all_planned_sheets'] is True
         and review['planned_pages'] == 12 and len(review['reviewed_pages']) == 12, 'Root full12-page review receipt')
    need(review['binding'] == {'path': bindings_path.relative_to(ROOT).as_posix(), 'bytes': len(bindings_raw), 'sha256': sha(bindings_raw)}, 'Review binds exact visual manifest')
    r.read(ROOT / relative(bindings['source']['path']), bindings['source'])
    # Verify and retain the earlier independent pixel receipt unchanged.
    pixel_path = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_pixels/receipt_v1.json'
    pixel_raw = r.read(pixel_path)
    need(sha(pixel_raw) == '24e24647ce297fe3a8cd076e70ffe01c6b159740e2cfe54e1f7273b51d340ae4', 'Earlier pixel receipt unchanged')
    evidence = Path(inventory['evidence_root']).resolve(strict=True)
    def returned(name):
        relative(name)
        need(inventory['files'][name] == inventory['extracted_files'][name], 'Exact extracted/index join')
        return r.read(evidence / name, inventory['extracted_files'][name])
    records = {}
    for call in protocol['plan']:
        name = OUT + '/calls/' + str(call['slot']).zfill(4) + '/record.json'
        record = json.loads(returned(name)); need(record['call'] == call, 'Exact record/call join')
        records[call['slot']] = record
    pages, review_rows = {}, {row['page']: row for row in review['reviewed_pages']}
    need(set(review_rows) == set(range(1, 13)), 'All12 page numbers reviewed once')
    for number, source in enumerate(bindings['pages'], 1):
        need(review_rows[number]['source'] == source and type(review_rows[number]['observation']) is str
             and review_rows[number]['observation'].strip(), 'Review exact page pin and observation')
        pages[number] = image(r.read(ROOT / relative(source['path']), source), (820, 1838))
    cache, checks, output_slots, case_order = {}, [], [], []
    seen_geometry = set()
    for cell in bindings['cells']:
        number, column, cid = cell['page'], cell['column'], cell['id']
        need(number in pages and column in (0, 1, 2), 'Known page/column')
        x, y = cell['xy']
        need(x == 10 + 270 * column and 100 <= y <= 1540 and (y - 100) % 288 == 0, 'Exact cell layout')
        geometry = (number, x, y)
        need(geometry not in seen_geometry, 'Unique cell coordinate')
        seen_geometry.add(geometry)
        source = cell['source']; name = relative(source['path'])
        if name not in cache:
            cache[name] = image(r.read(ROOT / name, source), (256, 256))
        observed = cache[name]
        crop = pages[number].crop((x, y, x + 256, y + 256))
        need(crop.tobytes() == observed.tobytes() and sha(crop.tobytes()) == cell['RGB8_sha256'], 'Exact source-to-cell RGB256 no resampling/enhancement')
        row_slot = next((slot for slot, value in records.items() if value['id'] == cid and value['call']['phase'] != 'deterministic_same_u64_repeat'), None)
        need(row_slot is not None, 'Rendered case exists in plan')
        record = records[row_slot]
        if column in (0, 1):
            need(cell['slot'] is None, 'Input/target not counted as output')
            bound = record['input' if column == 0 else 'target']
            need(name == (PARENT / bound['path']).relative_to(ROOT).as_posix()
                 and {key: source[key] for key in ('bytes', 'sha256')} == parent['files'][bound['path']]
                 == {key: bound[key] for key in ('bytes', 'sha256')}, 'Exact parent input/target case binding')
            need(sha(observed.tobytes()) == next(row for row in protocol['selection']['optimization'] + protocol['selection']['monitors'] if row['id'] == cid)['input_RGB8_sha256' if column == 0 else 'target_RGB8_sha256'], 'Parent RGB pixel digest')
        else:
            need(cell['slot'] == row_slot, 'Rendered output slot matches case')
            bound = record['outputs']['png']
            need(name == (evidence / bound['path']).relative_to(ROOT).as_posix()
                 and {key: source[key] for key in ('bytes', 'sha256')} == {key: bound[key] for key in ('bytes', 'sha256')}, 'Actual returned output cell binding')
            returned(bound['path'])
            output_slots.append(row_slot); case_order.append(cid)
        checks.append({'page': number, 'id': cid, 'column': column, 'slot': cell['slot'],
                       'xy': cell['xy'], 'exact_RGB256_source_cell': True, 'RGB8_sha256': cell['RGB8_sha256']})
    need(len(output_slots) == len(set(output_slots)) == len(set(case_order)) == 68, 'Exact68 distinct output cells')
    aliases = []
    for value in bindings['exact_repeated_witness_aliases']:
        slot, previous = value['slot'], value['same_RGB_cell_as_slot']
        need(slot in (5, 6, 7, 8) and previous == slot - 4 and previous in output_slots, 'Exact repeated4 alias slots')
        current_record, original_record = records[slot], records[previous]
        need(current_record['id'] == original_record['id'] and current_record['call']['phase'] == 'deterministic_same_u64_repeat', 'Same-case repeated phase')
        current = current_record['outputs']['png']; original = original_record['outputs']['png']
        need(current['sha256'] == original['sha256'] == value['exact_same_PNG_sha256'], 'Alias exact PNG hash')
        a, b = returned(current['path']), returned(original['path'])
        need(a == b, 'Alias actual PNG byte identity')
        aliases.append({**value, 'exact_current_file_bytes': True})
    need(set(output_slots) | {row['slot'] for row in aliases} == set(range(1, 73)), 'All72 outputs represented once or exact repeated alias')
    for value in bindings['consumed_files']:
        r.read(ROOT / relative(value['path']), value)
    files = r.after()
    result = {'complete': True, 'visual_readback_pass': True, 'unresolved_blockers': [],
              'protocol_sha256': PROTOCOL_SHA, 'planned_pages': 12, 'readback_pages': 12,
              'root_review_pages_bound': 12, 'exact_RGB256_cells': len(checks), 'distinct_output_cells': 68,
              'repeated_witness_aliases': aliases, 'all72_outputs_covered_once_or_exact_alias': True,
              'all_input_target_output_cells_match_actual_bound_sources': True,
              'all_cells_without_resampling_or_enhancement': True, 'earlier_saved_pixel_receipt_unchanged': True,
              'root_visual_review_bound': {'path': review_path.relative_to(ROOT).as_posix(), 'bytes': len(review_raw), 'sha256': sha(review_raw)},
              'visual_bindings_bound': {'path': bindings_path.relative_to(ROOT).as_posix(), 'bytes': len(bindings_raw), 'sha256': sha(bindings_raw)},
              'cell_checks': checks, 'consumed_files': files, 'all_consumed_before_after_exact': True,
              'readback_scope': 'Independent pixel/file/case/slot/coverage binding of12savedpages204exactRGBcells androotreviewreceipt. This does not substitute for root visual judgement or independent human FINAL review.',
              'zero_updates_not_an_improved_model': True, 'model_qualified': False,
              'no_model_inference_neural_gradient_optimizer_parameter_training_VM_browser_calls': True}
    path = AUDIT / 'visual_readback_v1.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in ('cell_checks', 'consumed_files')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
