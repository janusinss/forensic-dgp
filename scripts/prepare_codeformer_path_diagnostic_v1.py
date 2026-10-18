"""Stage eight exact TRAIN observations; seal sources only after root review.

Pure CPU preparation. No neural/gradient/training/VM call. Parent weights and
stopped checkpoint stay external read-only dependencies; no large reupload.
"""
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_path_diagnostic_v1 import contract as c

PARENT = 'outputs/cctv_codeformer_adaptation_v2_vm'
RETURNED = 'outputs/codeformer_adaptation_v2_return'
REVIEW = 'outputs/codeformer_adaptation_v2_return_review'
EVIDENCE = 'outputs/codeformer_path_diagnostic_v1_preparation'
DESIGN = REVIEW + '/next_diagnostic_review_v1.md'
DESIGN_SHA = '872caeaad27e73247c6b5f65aba87accdb49c77c466743b4321d8d973ad64eea'
PREREQUISITES = {
    REVIEW + '/archive_import_v1.json': '3f85cf121e7b3ad764c3afabd53722dfc56674aaa95921d61b3b6f762ad19b96',
    REVIEW + '/trajectory_v2/trajectory_audit.json': 'e5a42b0f5a495fe377c81789fa9d13825ecb7e8092a5531877c6d8ef5c431d5c',
    REVIEW + '/publication_readback_v1.json': '4f61ba8e95a32343c0c2e427a446cd465ccbfe28afc608af6316d05bd32b6cc0',
    REVIEW + '/metrics_v1/metrics_audit.json': '9a819c016018b7d99621d10c455f8908a9f0f1657de59e06b50dc02dc6775dee',
}
SHARED = ('research/__init__.py', 'research/codeformer_adaptation_v1/__init__.py',
    'research/codeformer_adaptation_v1/common.py', 'research/codeformer_adaptation_v1/data.py',
    'research/codeformer_adaptation_v1/perceptual.py',
    'research/codeformer_adaptation_v2/__init__.py', 'research/codeformer_adaptation_v2/model.py',
    'third_party/__init__.py', 'third_party/codeformer/__init__.py',
    'third_party/codeformer/codeformer_arch.py', 'third_party/codeformer/vqgan_arch.py',
    'third_party/codeformer/registry.py', 'third_party/codeformer/LICENSE', 'third_party/codeformer/NOTICE.md')


def write_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as target:
        target.write(payload); target.flush(); os.fsync(target.fileno())


def exact_files(root):
    files, folded = set(), set()
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            p = Path(folder) / name
            c.require(not p.is_symlink() and not getattr(p.lstat(), 'st_file_attributes', 0) & 0x400,
                      'Staging directory link forbidden')
        for name in names:
            relative = (Path(folder) / name).relative_to(root).as_posix()
            c.bounded(root, relative)
            c.require(relative.casefold() not in folded, 'Staging case collision')
            files.add(relative); folded.add(relative.casefold())
    return files


def stage_inputs(root=ROOT):
    root = Path(root).resolve(strict=True)
    packet = root / 'outputs' / c.PACKAGE_DIR
    evidence = root / EVIDENCE
    c.require(not packet.exists() and not (evidence / 'stage_inputs.json').exists() and
              not (evidence / 'stage_failure.json').exists(), 'Retain prior diagnostic stage; no automatic repeat')
    parent = root / PARENT
    returned = root / RETURNED
    c.require(c.sha256(c.bounded(parent, 'protocol.json')) == c.PARENT_PROTOCOL_SHA256 ==
              c.sha256(c.bounded(returned, 'protocol.json')), 'Exact parent/imported V2 protocol differs')
    p = c.read_json(parent / 'protocol.json')
    c.require(c.sha256(c.bounded(root, DESIGN)) == DESIGN_SHA, 'Prospective diagnostic review changed')
    for name, expected in PREREQUISITES.items():
        c.require(c.sha256(c.bounded(root, name)) == expected, 'Independent V2 audit/publication prerequisite changed')
    trajectory = c.read_json(root / REVIEW / 'trajectory_v2/trajectory_audit.json')
    c.require(trajectory['audit_finished'] and trajectory['evidence_consistency_pass'] and
              trajectory['completed_updates'] == 50 and not trajectory['model_qualified'],
              'The stopped50 trajectory must be independently verified, not qualified')
    c.require(c.read_json(root / REVIEW / 'archive_import_v1.json')['complete'] is True and
              c.read_json(root / REVIEW / 'publication_readback_v1.json')['complete'] is True,
              'Closed import and publication review required')
    export = c.read_json(returned / 'outputs/adaptation_v2/export_manifest_20261015T140245Z-40fec745.json')
    refs = c.read_json(parent / p['data_manifest'])['references']
    c.require(c.sha256(parent / p['data_manifest']) == p['files_sha256'][p['data_manifest']],
              'Frozen reference-role manifest changed')
    by_id = {r['id']: r for r in refs}
    history = [json.loads(line) for line in (returned / 'outputs/adaptation_v2/training.jsonl').read_text().splitlines()]
    c.require(len(history) == 50 and [r['update'] for r in history] == list(range(1, 51)), 'TRAIN history prefix differs')
    selected = []
    for update, slot in c.POSITIONS:
        sample = history[update - 1]['microbatches'][slot - 1]
        pair = sample['pair_record']
        ref = by_id[sample['id']]
        c.require(ref['historical_role'] == pair['role'] == 'train' and pair['update'] == update and
                  pair['slot'] == slot and pair['reference_id'] == ref['id'] and
                  pair['source_sha256'] == ref['source_sha256'] and
                  pair['reference_manifest_sha256'] == p['files_sha256'][p['data_manifest']],
                  'Only exact chronologically selected TRAIN records may be staged')
        input_path = c.bounded(returned, pair['saved_input_path'])
        record_relative = str(Path(pair['saved_input_path']).with_name('pair_record.json')).replace('\\', '/')
        pair_path = c.bounded(returned, record_relative)
        c.require(c.sha256(input_path) == pair['saved_input_sha256'] == export['files_sha256'][pair['saved_input_path']]
                  and c.sha256(pair_path) == export['files_sha256'][record_relative]
                  and c.read_json(pair_path) == pair, 'Saved TRAIN input/record differs')
        c.require(c.sha256(c.bounded(parent, ref['source_path'])) == ref['source_sha256'], 'Genuine source bytes changed')
        if slot == 3:
            anchor = sample['behavior_anchor']
            c.require(anchor['reference_id'] == ref['id'] and anchor['update'] == update and
                      anchor['slot'] == slot and anchor['pair_record'] == pair,
                      'Behavior anchor is not the exact normal-B slot')
            for path_key, hash_key in [('delivered_path', 'delivered_sha256'), ('logits_path', 'logits_sha256')]:
                c.require(c.sha256(c.bounded(returned, anchor[path_key])) == anchor[hash_key] ==
                          export['files_sha256'][anchor[path_key]], 'Saved behavior anchor changed before stage')
        selected.append((sample, ref, input_path, pair_path, record_relative))
    c.require(tuple(ref['id'] for _, ref, _, _, _ in selected) == c.REFERENCE_IDS,
              'The already audited fixed TRAIN photograph choices differ')
    external = {}
    for asset in c.ASSETS.values():
        source_root = returned if asset['path'] == c.STOPPED_CHECKPOINT['path'] else parent
        source = c.bounded(source_root, asset['path'])
        c.require(c.sha256(source) == asset['sha256'] and source.stat().st_size == asset['bytes'], 'External parent asset differs')
        external[asset['path']] = asset['sha256']
    for name in SHARED:
        c.require(c.sha256(c.bounded(parent, name)) == p['files_sha256'][name] ==
                  c.sha256(c.bounded(root, name)), 'Shared upstream/frozen helper changed')
        external[name] = p['files_sha256'][name]
    # All checks above precede staging mutation. Derivation is CPU-only and only
    # for the four approved selected sources; never regenerate camera pixels.
    packet.mkdir(); evidence.mkdir(parents=True, exist_ok=True)
    files, copies, generated, cases = {}, {}, {}, []

    def copy(source, relative, expected):
        c.require(c.sha256(source) == expected, 'Copy source changed')
        dest = c.bounded(packet, relative, must_exist=False)
        write_bytes(dest, source.read_bytes())
        c.require(c.sha256(dest) == expected, 'Copied payload hash differs')
        files[relative] = expected
        copies[relative] = {'source': source.relative_to(root).as_posix(), 'sha256': expected,
                            'bytes': source.stat().st_size}

    try:
        import numpy as np
        from research.codeformer_adaptation_v1.data import load_reference_manifest, derive_clean_target
        references = load_reference_manifest(parent, 'train', p['data_manifest'],
            expected_manifest_sha256=p['files_sha256'][p['data_manifest']])
        refs_by_id = {r.id: r for r in references}
        targets = {}
        for _, ref, _, _, _ in selected:
            rid = ref['id']
            if rid in targets:
                continue
            clean = derive_clean_target(refs_by_id[rid])
            c.require(clean.dtype == np.float32 and clean.shape == (512, 512, 3) and
                      np.isfinite(clean).all() and clean.min() >= 0 and clean.max() <= 1,
                      'Direct clean512 derivation differs')
            clean_hash = __import__('hashlib').sha256(clean.tobytes()).hexdigest()
            c.require(all(sample['pair_record']['clean512_float32_sha256'] == clean_hash and
                          sample['pair_record']['clean512_rgb8_sha256'] == ref['proposed_512_rgb_sha256']
                          for sample, source_ref, _, _, _ in selected if source_ref['id'] == rid),
                      'Derived clean target differs from exact saved TRAIN target')
            relative = f'data/clean512/{rid}.npy'
            destination = c.bounded(packet, relative, must_exist=False)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                np.save(stream, clean, allow_pickle=False)
            files[relative] = c.sha256(destination)
            generated[relative] = {'kind': 'direct_original1024_Pillow_LANCZOS512_float32',
                'reference_id': rid, 'source_sha256': ref['source_sha256'],
                'clean512_float32_sha256': clean_hash,
                'clean512_rgb8_sha256': ref['proposed_512_rgb_sha256'],
                'sha256': files[relative], 'bytes': destination.stat().st_size}
            targets[rid] = relative
        for (update, slot), (sample, ref, input_path, pair_path, old_record) in zip(c.POSITIONS, selected):
            rid, identifier = ref['id'], c.case_id(update, slot)
            inp, record = f'data/cases/{identifier}/input.png', f'data/cases/{identifier}/pair_record.json'
            copy(input_path, inp, sample['pair_record']['saved_input_sha256'])
            copy(pair_path, record, export['files_sha256'][old_record])
            row = {'id': identifier, 'update': update, 'slot': slot, 'reference_id': rid,
                   'image_eligible': sample['image_eligible'], 'input_path': inp, 'input_sha256': files[inp],
                   'clean512_path': targets[rid], 'clean512_sha256': files[targets[rid]],
                   'pair_record_path': record, 'pair_record_sha256': files[record], 'behavior': None}
            if slot == 3:
                anchor = sample['behavior_anchor']
                row['behavior'] = {}
                for label in ('delivered', 'logits'):
                    path_key, hash_key = label + '_path', label + '_sha256'
                    source = c.bounded(returned, anchor[path_key])
                    c.require(c.sha256(source) == anchor[hash_key] == export['files_sha256'][anchor[path_key]], 'Saved behavior anchor differs')
                    relative = f'data/behavior/update_{update:04d}/{label}.npy'
                    copy(source, relative, anchor[hash_key]); row['behavior'][path_key] = relative
                    row['behavior'][hash_key] = anchor[hash_key]
            c.verify_case_pixels(packet, row, sample['pair_record'])
            cases.append(row)
        for name in SHARED:
            copy(c.bounded(parent, name), name, p['files_sha256'][name])
        notices = [n for n in p['files_sha256'] if n.startswith('perceptual/') and
                   Path(n).suffix.lower() in ('.txt', '.md', '.json') and n != 'perceptual/manifest.json']
        for name in notices:
            copy(c.bounded(parent, name), 'provenance/parent/' + name, p['files_sha256'][name])
        source_record = {'complete': True, 'format': 'codeformer-path-selected-TRAIN-provenance-v1',
            'parent_protocol_sha256': c.PARENT_PROTOCOL_SHA256, 'diagnostic_review_sha256': DESIGN_SHA,
            'independent_audit_sha256': PREREQUISITES,
            'selected_references': [by_id[rid] for rid in sorted(targets)],
            'exact_original_VM_inputs_preserved': True, 'camera_regeneration': False,
            'clean_targets': generated, 'parent_checkpoint_retained_external': c.STOPPED_CHECKPOINT,
            'pretraining_identity_overlap_unknown': True, 'TRAIN_only': True,
            'DEV_final_native_inputs': 0, 'neural_calls': 0, 'gradient_queries': 0}
        rel = 'provenance/selected_sources.json'; c.write_new_json(packet / rel, source_record); files[rel] = c.sha256(packet / rel)
        copy(c.bounded(root, DESIGN), 'provenance/next_diagnostic_review_v1.md', DESIGN_SHA)
        receipt = {'complete': True, 'phase': 'inputs_staged_not_executable_not_sealed',
            'packet': packet.relative_to(root).as_posix(), 'cases': cases, 'assets': c.ASSETS,
            'parent_files_sha256': external, 'files_sha256': files, 'copy_bindings': copies,
            'generated_targets': generated, 'selected_photographs': sorted(targets),
            'files': len(files), 'bytes': sum((packet / n).stat().st_size for n in files),
            'optimizer_updates': 0, 'neural_calls': 0, 'VM_calls': 0}
        c.require(exact_files(packet) == set(files), 'Unlisted staged artifact')
        c.write_new_json(evidence / 'stage_inputs.json', receipt)
        return receipt
    except BaseException as error:
        c.write_new_json(evidence / 'stage_failure.json', {'complete': False, 'error': repr(error),
            'staging_retained': True, 'optimizer_updates': 0, 'neural_calls': 0})
        raise


def seal_sources(root, readiness, readiness_sha, instruction, instruction_sha):
    """Freeze a reviewed small packet; no archive or model execution occurs."""
    root = Path(root).resolve(strict=True)
    packet, evidence = root / 'outputs' / c.PACKAGE_DIR, root / EVIDENCE
    c.require(not (packet / 'protocol.json').exists(), 'Retain frozen diagnostic protocol')
    source_receipt = c.bounded(root, readiness)
    c.require(c.sha256(source_receipt) == readiness_sha, 'Readiness SHA256 differs')
    reviewed = c.read_json(source_receipt)
    c.require(reviewed.get('complete') is True and reviewed.get('approved_for_protocol_freeze') is True and
              reviewed.get('training_authorized') is False, 'Independent reviewed-source approval required')
    c.require(reviewed.get('design_sha256') == instruction_sha,
              'The instruction must match the independently reviewed design SHA256')
    staged = c.read_json(evidence / 'stage_inputs.json')
    c.require(staged['complete'] and exact_files(packet) == set(staged['files_sha256']), 'Inputs-only stage changed')
    files = deepcopy(staged['files_sha256'])
    for name, expected in files.items():
        c.require(c.sha256(c.bounded(packet, name)) == expected, 'Staged input hash differs')
    sources = reviewed.get('source_sha256', {})
    c.require(set(c.REQUIRED_SOURCES) <= set(sources), 'Incomplete reviewed diagnostic source list')
    source_payloads = {}
    for name, expected in sources.items():
        source = c.bounded(root, name)
        c.require(name.startswith(('research/codeformer_path_diagnostic_v1/', 'scripts/', 'tests/')) and
                  source.suffix == '.py' and c.sha256(source) == expected, 'Unreviewed diagnostic source')
        ast.parse(source.read_text(encoding='utf-8'), feature_version=(3, 10))
        c.require(name not in files, 'Reviewed source collides with staged payload')
        source_payloads[name] = source.read_bytes()
    c.require(c.sha256(c.bounded(root, instruction)) == instruction_sha, 'Reviewed diagnostic instruction changed')
    # Missing, stale or invalid source/instruction files stop before sealing writes.
    for name, payload in source_payloads.items():
        write_bytes(c.bounded(packet, name, must_exist=False), payload); files[name] = sources[name]
    for relative, source, expected in [('instructions/DIAGNOSTIC_DESIGN.md', root / instruction, instruction_sha),
            ('provenance/source_readiness.json', source_receipt, readiness_sha)]:
        write_bytes(c.bounded(packet, relative, must_exist=False), source.read_bytes()); files[relative] = expected
    protocol = deepcopy(c.SPEC)
    protocol.update(parent_files_sha256=staged['parent_files_sha256'], files_sha256=files,
        cases=staged['cases'], optimizer_updates=0, no_automatic_training=True,
        instruction={'path': 'instructions/DIAGNOSTIC_DESIGN.md', 'sha256': instruction_sha},
        export_files=sorted(n for n in files if not n.startswith('data/')))
    c.validate_protocol(protocol)
    c.write_new_json(packet / 'protocol.json', protocol)
    pin = c.sha256(packet / 'protocol.json')
    c.verify_transfer(packet, pin, verify_parent=False)
    receipt = {'complete': True, 'phase': 'reviewed_sources_sealed_not_archived', 'protocol_sha256': pin,
        'packet': packet.relative_to(root).as_posix(), 'files': len(files) + 1,
        'bytes': sum((packet / n).stat().st_size for n in exact_files(packet)),
        'optimizer_updates': 0, 'neural_calls': 0, 'VM_calls': 0}
    c.write_new_json(evidence / 'sealed_sources.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage-inputs', action='store_true')
    mode.add_argument('--seal-sources', action='store_true')
    parser.add_argument('--readiness'); parser.add_argument('--readiness-sha')
    parser.add_argument('--instruction'); parser.add_argument('--instruction-sha')
    args = parser.parse_args()
    if args.stage_inputs:
        c.require(not any((args.readiness, args.readiness_sha, args.instruction, args.instruction_sha)), 'Seal fields require seal mode')
        receipt = stage_inputs(args.root)
    else:
        c.require(all((args.readiness, args.readiness_sha, args.instruction, args.instruction_sha)), 'Sealing needs reviewed source/instruction pins')
        receipt = seal_sources(args.root, args.readiness, args.readiness_sha, args.instruction, args.instruction_sha)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('cases','files_sha256','copy_bindings','generated_targets','parent_files_sha256')}, indent=2))


if __name__ == '__main__':
    main()
