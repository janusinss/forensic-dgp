"""Export only the reviewed terminal LR head NPZ; no model or Torch execution.

The full archive is streamed under the original caps. Only the one requested
NPZ is retained; no extract/extractall call is used. Root visual closure and a
fresh independent review of these export/runtime sources are prerequisites.
"""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import tarfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
RETURN = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit'
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_export'
ARCHIVE = ROOT / 'outputs/cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-results.tar.gz'
RUN = 'outputs/u96_postfusion_smoothanchors_lr_v1'
ARCHIVE_SHA = '7e5208d2ff69713c834c21fb466020ae54d7ee68b84795e2c044ffc11847041e'
ARCHIVE_BYTES = 1166628159
PROTOCOL_SHA = '8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9'
INFERENCE = ROOT / 'research/codeformer_u96_postfusion_lr_inference_v1.py'
INFERENCE_SHA = '56408052cb13616aa3365cd1bb211c70c54826331d1ca2bb0171a7a6334da803'
HEAD_SHA = '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'
HEAD_CANONICAL = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'
PAYLOAD_CAP = 1536 * 1024**2
OVERHEAD_CAP = 16 * 1024**2
ROLES = ['local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    def bad(value):
        raise ValueError('Nonfinite JSON ' + value)
    result = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=bad)
    def finite(value):
        if type(value) is float:
            need(math.isfinite(value), 'Nonfinite exponent-overflow JSON')
        elif isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
    finite(result)
    return result


def ordinary(path, maximum=32 * 1024**2):
    path = Path(path).absolute()
    need(path.is_relative_to(ROOT), 'Input must stay inside workspace')
    for parent in path.parents:
        info = parent.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary input parent')
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not path.is_symlink() and info.st_nlink == 1
         and not getattr(info, 'st_file_attributes', 0) & 1024 and 0 < info.st_size <= maximum,
         'Bounded ordinary unlinked input required')
    return path


def pin(path, maximum=PAYLOAD_CAP):
    path = ordinary(path, maximum)
    before = path.stat(); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Input changed during read')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': before.st_size,
            'sha256': digest.hexdigest()}


def read(path):
    return strict_json(ordinary(path).read_bytes())


def require_pin(row):
    need(set(row) >= {'path', 'bytes', 'sha256'}, 'Complete pin required')
    need(pin(ROOT / row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')},
         'Current evidence pin differs: ' + row['path'])


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    return pin(path)


def reviewed_source(path):
    review = read(path)
    need(review['complete'] is True and review['source_review_pass'] is True,
         'Independent fresh export/runtime source review required')
    rows = review['sources']
    required = {
        'scripts/export_u96_postfusion_lr_inference_v1.py',
        'scripts/audit_u96_postfusion_lr_cpu_runtime_v1.py',
        'research/codeformer_u96_postfusion_lr_inference_v1.py'}
    need(required <= {row['path'] for row in rows}, 'Peer review must bind all export/runtime/inference sources')
    for row in rows: require_pin(row)
    return review


def visual_closure(path):
    visual = read(path); manifest_path = RETURN / 'visual_manifest_v1.json'; manifest = read(manifest_path)
    need(visual['complete'] is True and visual['all_planned_sheets_reviewed'] is True
         and visual['generated_PNG_cells_reviewed'] == 712 and visual['sheets_reviewed'] == 30,
         'Root personal all30/712 visual closure required before export')
    need(visual['visual_manifest'] == pin(manifest_path) and manifest['planned_sheets'] == 30,
         'Root closure must bind exact immutable visual manifest')
    observed = visual['observations']
    expected = {sheet['file']['path']: sheet['file'] for sheet in manifest['sheets']}
    need(len(expected) == len(observed) == 30 and
         {row['file']['path'] for row in observed} == set(expected), 'Every planned sheet reviewed exactly once')
    need(sum(row['generated_cells_reviewed'] for row in observed) == 712, 'Root generated-cell review count differs')
    for row in observed:
        need(row['personally_viewed_by_root'] is True and row['file'] == expected[row['file']['path']],
             'Personal visual observation/pin missing')
        require_pin(row['file'])
    return visual


def prerequisites(visual_path, review_path):
    reviewed_source(review_path); visual_closure(visual_path)
    need(pin(INFERENCE)['sha256'] == INFERENCE_SHA, 'Held frozen inference source differs')
    protocol_path = STAGE / 'protocol.json'; protocol = read(protocol_path)
    need(pin(protocol_path)['sha256'] == PROTOCOL_SHA, 'Exact sealed LR protocol required')
    audit_path = RETURN / 'saved_pixel_audit_v1.json'; audit = read(audit_path)
    need(audit['complete'] and audit['independent_saved_pixel_integrity_pass'] is True
         and audit['pilot_quality_gates_pass'] is True and audit['scientific_trial_pass'] is True
         and audit['archive']['sha256'] == ARCHIVE_SHA and audit['protocol']['sha256'] == PROTOCOL_SHA,
         'Actual audited successful finite TRAIN return required')
    cross_path = RETURN / 'peer_state_and_metadata_crossread_v1.json'; cross = read(cross_path)
    need(cross['complete'] and cross['independent_crossread_pass'] is True, 'Actual crossread prerequisite')
    state_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_state_audit/saved_state_audit_v1.json'
    state = read(state_path)
    need(state['complete'] and state['durable_committed_trajectory_fully_covered'] is True
         and state['artifact_outcome']['confirmed_optimizer_updates'] == 64
         and state['artifact_outcome']['optimizer_outcome_unknown'] is False
         and state['archive']['sha256'] == ARCHIVE_SHA and state['protocol_sha256'] == PROTOCOL_SHA,
         'Actual saved-state coverage prerequisite')
    peer_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_peer/independent_return_review_v1.json'
    peer = read(peer_path)
    need(peer['complete'] and peer['independent_return_metadata_audit_pass'] is True
         and peer['authoritative_status'] == 'FINITE_B64_LEARNING_SCREEN_PASSED_NOT_QUALIFIED'
         and peer['optimizer_updates'] == peer['optimizer_update_attempts'] == 64
         and peer['optimizer_outcome_unknown'] is False, 'Actual independent metadata prerequisite')
    inference_review_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_preparation/independent_source_review_v1_r1.json'
    inference_review = read(inference_review_path)
    need(inference_review['complete'] and inference_review['source_review_pass'] is True,
         'Frozen inference source review prerequisite')
    for row in inference_review['sources']: require_pin(row)
    index_path = RETURN / 'archive_index_v1.json'; index = read(index_path)
    need(index['complete'] and index['archive_SHA256'] == ARCHIVE_SHA, 'Exact prior archive index')
    cp_path = RETURN / 'returned_metadata/checkpoints/u0064/checkpoint.json'; checkpoint = read(cp_path)
    need(checkpoint['complete'] and checkpoint['update'] == 64
         and checkpoint['head_canonical_sha256'] == HEAD_CANONICAL, 'Exact terminal checkpoint only')
    head = checkpoint['heads']['file']
    need(head['path'] == RUN + '/checkpoints/u0064/heads.npz'
         and head['bytes'] == 685818 and head['sha256'] == HEAD_SHA
         and index['members'][head['path']] == head, 'Exact unchanged terminal head NPZ binding')
    evidence = [protocol_path, audit_path, cross_path, state_path, peer_path,
                inference_review_path, index_path, cp_path, Path(visual_path), Path(review_path)]
    return protocol, index, head, [pin(path) for path in evidence]


def stream_selected(archive_path, expected_members, selected, seconds=120):
    """One bounded gzip/TAR read, exact every-member pins, selected bytes only."""
    need(0 < len(selected) <= 4 and set(selected) <= set(expected_members), 'Bounded known selection required')
    path = ordinary(archive_path, PAYLOAD_CAP); before = path.stat()
    need(before.st_size == ARCHIVE_BYTES, 'Exact original LR archive bytes required')
    started = time.monotonic(); count = 0; payload = 0; members = {}; kept = {}
    class HashReader:
        def __init__(self, stream): self.stream = stream; self.hash = hashlib.sha256(); self.count = 0
        def read(self, size=-1):
            need(size >= 0, 'Unbounded compressed read refused')
            data = self.stream.read(size); self.hash.update(data); self.count += len(data); return data
    class Bounded:
        def __init__(self, stream): self.stream = stream; self.count = 0; self.tail = b''
        def read(self, size=-1):
            need(size >= 0 and time.monotonic() - started < seconds, 'Finite bounded archive read required')
            data = self.stream.read(size); self.count += len(data)
            self.tail = (self.tail + data)[-65536:]
            need(self.count <= PAYLOAD_CAP + OVERHEAD_CAP, 'Expanded archive cap exceeded'); return data
    with path.open('rb') as source:
        hashed = HashReader(source)
        with gzip.GzipFile(fileobj=hashed, mode='rb') as zipped:
            bounded = Bounded(zipped)
            with tarfile.open(fileobj=bounded, mode='r|') as archive:
                for member in archive:
                    name = member.name; pp = PurePosixPath(name)
                    need(name.startswith(STAGE.name + '/') and '\\' not in name and ':' not in name
                         and '..' not in pp.parts and pp.as_posix() == name.rstrip('/'), 'Canonical exact-stage TAR name required')
                    if member.isdir(): continue
                    relative = PurePosixPath(*pp.parts[1:]).as_posix()
                    need(member.isfile() and member.type != tarfile.GNUTYPE_SPARSE and not member.issparse()
                         and not any(k.startswith('GNU.sparse') for k in member.pax_headers), 'Only ordinary nonsparse files')
                    need(relative in expected_members and relative not in members, 'Unexpected or duplicate TAR file')
                    need(0 <= member.size <= 64 * 1024**2, 'Bounded TAR member')
                    payload += member.size; count += 1
                    need(payload <= PAYLOAD_CAP and count <= 6000, 'Payload/member cap exceeded')
                    stream = archive.extractfile(member); digest = hashlib.sha256(); size = 0; pieces = []
                    while True:
                        block = stream.read(1024**2)
                        if not block: break
                        size += len(block); need(size <= member.size, 'TAR member length overflow'); digest.update(block)
                        if relative in selected:
                            need(size <= 1024**2, 'Selected NPZ/RAW exceeds1MiB'); pieces.append(block)
                    row = {'path': relative, 'bytes': size, 'sha256': digest.hexdigest()}
                    need(size == member.size and row == expected_members[relative], 'Actual returned member pin differs')
                    members[relative] = row
                    if relative in selected: kept[relative] = b''.join(pieces)
                logical_end = archive.offset
                # TarFile's stream may already have read past its zero header.
                # Include that bounded read-ahead, not only later drain reads.
                ahead = bounded.count - logical_end
                need(0 <= ahead <= len(bounded.tail), 'Bounded TAR read-ahead suffix required')
                need(not ahead or not bounded.tail[-ahead:].strip(b'\0'),
                     'Nonzero buffered TAR suffix after logical terminator')
            while True:
                block = bounded.read(1024**2)
                if not block: break
                need(not block.strip(b'\0'), 'Nonzero trailing TAR bytes')
        while hashed.read(1024**2): pass
    after = path.stat()
    need((before.st_size, before.st_mtime_ns, before.st_ino) ==
         (after.st_size, after.st_mtime_ns, after.st_ino), 'Original archive changed')
    need(hashed.count == ARCHIVE_BYTES and hashed.hash.hexdigest() == ARCHIVE_SHA
         and set(members) == set(expected_members) and set(kept) == set(selected), 'Complete exact archive read required')
    return kept, {'bytes': hashed.count, 'sha256': hashed.hash.hexdigest(), 'members': count,
                  'payload_bytes': payload, 'expanded_bytes': bounded.count,
                  'gzip_CRC_footer_drained': True, 'only_zero_TAR_suffix': True,
                  'buffered_TAR_suffix_bytes_verified': ahead,
                  'selected_paths': sorted(selected), 'unsafe_extract_used': False,
                  'seconds': time.monotonic() - started}


def four_cases(protocol, audit):
    first = next(row['reference_id'] for row in protocol['cases'] if row['cohort'] == 'monitor')
    rows = [next(row for row in protocol['cases'] if row['reference_id'] == first and
                 row['cohort'] == 'monitor' and row['role'] == role) for role in ROLES]
    result = []
    for row in rows:
        saved = next(x for x in audit['all_saved_pixel_metrics'] if x['kind'] == 'snapshot'
                     and x['update'] == 64 and x['id'] == row['id'])
        result.append({'case': row, 'saved_L4_RAW': saved['raw_file'],
                       'saved_L4_PNG': saved['png_file'], 'saved_L4_regional_metrics': saved['regional_metrics']})
    need(len({x['case']['reference_id'] for x in result}) == 1, 'Same first monitor photograph required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-visual', type=Path, required=True)
    parser.add_argument('--source-review', type=Path, required=True)
    args = parser.parse_args()
    need(not OUT.exists(), 'Export namespace immutable and must not preexist')
    protocol, index, head, evidence = prerequisites(args.root_visual, args.source_review)
    checksum = Path(str(ARCHIVE) + '.sha256'); exported = ARCHIVE.parent / 'cctv-codeformer-u96-postfusion-smoothanchors-lr-v1-export.json'
    need(ordinary(checksum, 4096).read_text(encoding='ascii').splitlines() == [ARCHIVE_SHA + '  ' + ARCHIVE.name], 'Exact archive checksum sidecar')
    ex = read(exported)
    need(ex['complete'] and ex['archive_sha256'] == ARCHIVE_SHA and ex['bytes'] == ARCHIVE_BYTES
         and ex['protocol_sha256'] == PROTOCOL_SHA, 'Exact archive export sidecar')
    kept, transport = stream_selected(ARCHIVE, index['members'], {head['path']})
    spec = importlib.util.spec_from_file_location('reviewed_inert_frozen_source', INFERENCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    arrays = module.inspect_head_npz(kept[head['path']])
    need(module.numpy_state_sha(arrays) == HEAD_CANONICAL, 'Exact terminal36 canonical values required')
    import sys
    need('torch' not in sys.modules, 'Export must not import Torch')
    audit = read(RETURN / 'saved_pixel_audit_v1.json')
    plan = {'complete': True, 'format': 'terminal-LR-four-case-CPU-witness-plan-v1',
            'runtime_executed': False, 'runtime_HOLD_until_root_review': True,
            'cases': four_cases(protocol, audit), 'maximum_cases': 4, 'CPU_threads': 4,
            'input_recipe': 'RGB256 uint8 -> float32 CHW/255 -> bilinear512 align_corners=False ->*2-1',
            'delivery_recipe': 'native.float()+1 divided2; clip native to[0,1] BEFORE bilinear256; float32 floor(raw*255)',
            'runtime_policy': 'Separate new owned frozen CPU FP32 575-tensor model; caller no_grad, no optimizer/backward/training; no app default change.',
            'comparison_policy': 'Read four exact saved L4 RAWs into bounded memory from pinned archive; quantify CPU float differences and delivered PNG differences. No bitwise equality promise, parity tolerance, quality-gate relaxation or qualification.',
            'runtime_limits': {'wall_seconds': 600, 'maximum_forward_calls': 4, 'output_bytes': 32 * 1024**2},
            'inference_source': pin(INFERENCE), 'archive': {'path': ARCHIVE.relative_to(ROOT).as_posix(), 'bytes': ARCHIVE_BYTES, 'sha256': ARCHIVE_SHA},
            'protocol': pin(STAGE / 'protocol.json'), 'model_files': protocol['model'],
            'canonical_states': module.source_contract()['canonical_states'],
            'model_qualified': False, 'main_app_default_changed': False}
    OUT.mkdir()
    destination = OUT / 'terminal_u0064_heads.npz'
    with destination.open('xb') as stream: stream.write(kept[head['path']])
    plan_pin = write_json(OUT / 'cpu_witness_plan_v1.json', plan)
    receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
               'inference_export_pass': True, 'head_file': pin(destination),
               'source_member': head, 'canonical_head_sha256': HEAD_CANONICAL, 'source_archive_sha256': ARCHIVE_SHA,
               'head_tensors': 36, 'head_parameters': 168960, 'source_update': 64,
               'archive': {'path': ARCHIVE.relative_to(ROOT).as_posix(), **transport},
               'protocol': pin(STAGE / 'protocol.json'), 'inference_source': pin(INFERENCE),
               'evidence': evidence + [pin(checksum), pin(exported)], 'model_files': protocol['model'],
               'canonical_states': module.source_contract()['canonical_states'],
               'runtime_plan': plan_pin, 'only_archive_artifact_written': 'terminal_u0064_heads.npz',
               'Torch_model_neural_optimizer_training_VM_calls': 0, 'CPU_runtime_executed': False,
               'root_visual_closure': pin(args.root_visual), 'model_qualified': False,
               'main_app_default_changed': False, 'script': pin(Path(__file__))}
    result = write_json(OUT / 'inference_export_v1.json', receipt)
    print(json.dumps({'complete': True, 'export': result, 'runtime_HOLD': True}))


if __name__ == '__main__':
    main()
