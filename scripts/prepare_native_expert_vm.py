"""Audit immutable local inputs and package a new bounded VM recipe; no models."""
import ast
from collections import Counter
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_expert import MODEL_SHA, OPTIMIZER_SHA, PARENT_SHA, fixed_schedule, validate_source, validate_source_optimizer, require
from supported_real_data import SupportedMasks
from scripts.train_native_expert_vm import sha, read_json, write_json, safe_path, verify_fixture

DEST = ROOT / 'outputs/native_expert_protocol_v1'
ARCHIVE = ROOT / 'outputs/native-expert-vm-code.tar.gz'
PREFIX = 'native_expert_vm_bundle'
REGISTRY = 'dataset/detector_supported_review_v1/manifest.json'
FIXTURES = 'outputs/reflection_coverage_data_v1'
SPLIT = 'outputs/downloaded_phase4/outputs/phase4_with_progress/split.json'
BOUND = {
    REGISTRY: '860ea1dfba8fa442cab19bf3ab41f6a678109dcdb067c38ec0bd79ce43b51ace',
    SPLIT: '5c71bc358a351d50e3c0a7d76abe749cb4412aca80d2c7eb4de5fb33dbd29071',
    FIXTURES + '/manifest.json': '6696cee3a3acf48049f2f121a2a6328625c4351f558deabb5475da7fe948baf9',
    FIXTURES + '/pixels.pth': 'ab47a88d79fe8d300ca92d572a2c1a05b92fa273b9577cb209df99e692009cc1',
    'outputs/supported_real_dataset_validation_v1/verification.json': '308e5ca5b00bf8c52b36eba78605e41d188b754044307fd54d4f39edf16cbfe9',
    'outputs/frozen_complementarity_v1/results.json': '2880c85f067c013f6998faf1744177be79b8ed0a50d949d6e35ae5da14563928',
    'outputs/frozen_complementarity_validation_v1/verification.json': '449ddc0f85908f653099cb229bfdbce16d152df1eb0ec7a7f4c881bd18d8134b',
}
ASSETS = {
    'model': ('outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/epoch_42.pth',
              'outputs/reflection_coverage_vm/reflective/epoch_42.pth', MODEL_SHA),
    'optimizer': ('outputs/downloaded_reflection_coverage/outputs/reflection_coverage_vm/reflective/final_optimizer.pth',
                  'outputs/reflection_coverage_vm/reflective/final_optimizer.pth', OPTIMIZER_SHA),
    'parent': ('outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth',
               'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth', PARENT_SHA),
    'fixture_manifest': (FIXTURES + '/manifest.json', FIXTURES + '/manifest.json', BOUND[FIXTURES + '/manifest.json']),
    'fixture_pixels': (FIXTURES + '/pixels.pth', FIXTURES + '/pixels.pth', BOUND[FIXTURES + '/pixels.pth']),
}


def code_closure(initial):
    """Conservative static local-import closure; never import/execute models."""
    found, pending = set(), list(initial)
    while pending:
        name = pending.pop()
        if name in found: continue
        path = safe_path(ROOT, name); require(path.is_file(), 'Missing explicit code: ' + name)
        found.add(name); tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
        parent_parts = list(Path(name).parent.parts)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import): modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = parent_parts[:len(parent_parts)-node.level+1] if node.level else []
                suffix = node.module.split('.') if node.module else []
                prefix = '.'.join(base + suffix)
                modules = [prefix] + [(prefix + '.' if prefix else '') + alias.name for alias in node.names]
            for module in modules:
                parts = module.split('.')
                if not module or any(p in ('', '*', '..') for p in parts): continue
                local = ROOT.joinpath(*parts)
                candidates = [local.with_suffix('.py'), local / '__init__.py']
                for length in range(1, len(parts)): candidates.append(ROOT.joinpath(*parts[:length]) / '__init__.py')
                pending.extend(p.relative_to(ROOT).as_posix() for p in candidates if p.is_file())
    return sorted(found)


def tensor_parameter_groups(payload):
    state = payload['model']
    require(state and all(isinstance(t, torch.Tensor) and t.device.type == 'cpu' and torch.isfinite(t).all()
                          for t in state.values()), 'Source checkpoint tensors invalid')
    tensors = []
    for prefix in ('network.encoder.', ('network.decoder.', 'network.segmentation_head.')):
        tensors.append([value for name, value in state.items() if name.startswith(prefix)
                        and not name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))])
    require([len(group) for group in tensors] == [60, 32], 'Source tensor group order/count differs')
    return [{'params': tensors[0], 'lr': 1e-5}, {'params': tensors[1], 'lr': 1e-4}]


def main():
    require(not DEST.exists() and not ARCHIVE.exists() and not ARCHIVE.with_name(ARCHIVE.name+'.sha256').exists(),
            'Preserve prepared/partial native expert evidence; use a new version for changes')
    torch.set_num_threads(4)
    for name, expected in BOUND.items(): require(sha(ROOT / name) == expected, 'Audited bound input changed: ' + name)
    for _, (name, _, expected) in ASSETS.items(): require(sha(ROOT / name) == expected, 'Source asset changed: ' + name)
    frozen = read_json(ROOT / 'outputs/frozen_complementarity_v1/results.json')
    for name, expected in frozen['mask_sha256'].items():
        require(sha(ROOT / 'outputs/frozen_complementarity_v1' / name) == expected, 'Audited frozen mask changed')
    real = SupportedMasks(ROOT / REGISTRY, split='train')
    require(len(real) == 83 and Counter(r['kind'] for r in real.rows) == {'covered': 51, 'uncovered': 32}, 'Training membership differs')
    fixture_manifest = read_json(ROOT / FIXTURES / 'manifest.json'); fixture_rows = fixture_manifest['cases']
    schedule = fixed_schedule(real.rows, fixture_rows)
    split = read_json(ROOT / SPLIT); training_paths = set(split['train'])
    require(not (training_paths & set(split['validation']))
            and all(row['source'] in training_paths for row in real.rows[73:] + fixture_rows), 'Native/fixture Phase4 membership changed')
    cache = torch.load(ROOT / FIXTURES / 'pixels.pth', map_location='cpu', weights_only=True)
    require(cache['format'] == 'dgp-reflection-coverage-pixels-v1' and set(cache['pixels']) == set(range(280)), 'Fixture cache format differs')
    for i, row in enumerate(fixture_rows): verify_fixture(cache['pixels'][i], row)
    payload = torch.load(ROOT / ASSETS['model'][0], map_location='cpu', weights_only=True); validate_source(payload)
    moments = torch.load(ROOT / ASSETS['optimizer'][0], map_location='cpu', weights_only=True)
    moment_check = validate_source_optimizer(moments, tensor_parameter_groups(payload), MODEL_SHA)
    old_path = ROOT / 'dataset/detector_expanded_review_v2/manifest.json'
    old = {r['image_sha256']: r for r in read_json(old_path)['records']}
    lens_arrays = []
    for i, row in enumerate(real.rows):
        if row.get('glare_stratum') == 'strong_lens_reflection':
            _, mask, _ = real[i]
            with Image.open(old_path.parent / old[row['image_sha256']]['mask']) as image: original = np.array(image).astype(bool)
            lens_arrays.append((i, mask[0].numpy().astype(bool) & ~original))
        elif i >= 73 and row.get('source_index') in (171, 216, 348, 374):
            lens_arrays.append((i, real[i][1][0].numpy().astype(bool)))
    require(len(lens_arrays) == 6 and sum(int(mask.sum()) for _, mask in lens_arrays) == 21853
            and sum(int(mask.sum()) for i, mask in lens_arrays if i < 73) == 2031, 'Reviewed lens cohort differs')
    explicit = ['native_expert.py', 'supported_real_data.py', 'reflection_coverage.py',
                'face_occlusion_adapter.py', 'scripts/train_native_expert_vm.py',
                'scripts/prepare_native_expert_vm.py', 'tests/test_native_expert.py', 'tests/test_native_expert_vm.py']
    code = code_closure(explicit)
    DEST.mkdir(); (DEST / 'lens').mkdir()
    maps = []
    for index, array in lens_arrays:
        path = DEST / 'lens' / f'real_{index:04}.png'; Image.fromarray(array.astype('uint8') * 255).save(path)
        maps.append({'real_index': index, 'path': 'inputs/lens/' + path.name, 'sha256': sha(path), 'pixels': int(array.sum())})
    preview_ids = [f'real/{i:04}' for i, r in enumerate(real.rows) if i >= 73 and r.get('source_index') in (171,216,348,374)]
    preview_ids += [f'real/{i:04}' for i, r in enumerate(real.rows) if r.get('glare_stratum') == 'strong_lens_reflection']
    preview_ids += [f'real/{i:04}' for i, r in enumerate(real.rows) if i >= 73 and r.get('source_index') in (207,208)]
    for covered in (True, False):
        i = next(r['case_id'] for r in fixture_rows if r['degraded'] is True and (r['style'] != 'clear') is covered)
        preview_ids.append(f'reflection/{i:04}')
    require(len(set(preview_ids)) == 10, 'Fixed preview cohort differs')
    protocol = {'format': 'dgp-native-expert-pilot-v1', 'date': '2026-10-02', 'epochs': 6,
        'steps_per_epoch': 56, 'updates_budget': 336, 'batch_size': 8, 'real_batch_size': 3, 'fixture_batch_size': 5,
        'domain_weights': {'real': .5, 'reflection': .5}, 'seed': 42, 'encoder_lr': 1e-5, 'decoder_head_lr': 1e-4,
        'weight_decay': 1e-4, 'clip': 1., 'background_weight': .25, 'hard_fraction': .1, 'threshold': .5,
        'native_source_order': [r['source_index'] for r in real.rows[73:]], 'schedule': schedule,
        'phase4_split_sha256': BOUND[SPLIT], 'registry_sha256': BOUND[REGISTRY],
        'local_audited_inputs': BOUND, 'existing_assets': {role: {'vm_path': vm, 'sha256': digest} for role, (_, vm, digest) in ASSETS.items()},
        'setup_sha256': '6af41c0662a0a91d55871c4faadb6d0810d3e366a964a4da23c5b52183ef210a',
        'source_counters': {'epoch': 42, 'model_updates': 882, 'moment_step': 672}, 'moment_check': moment_check,
        'code_sha256': {name: sha(ROOT / name) for name in code}, 'lens_maps': maps, 'preview_ids': preview_ids,
        'held_out_forward_images': 0, 'local_model_forward_images': 0, 'local_optimizer_constructed': False,
        'local_optimizer_updates': 0, 'stage': 'A native expert training-fit feasibility only', 'promoted': False,
        'development_gates_unchanged': True, 'stage_b_implemented': False,
        'limitations': ['Approximate assistant-reviewed labels; unknown source171 pixels excluded only from supervision.',
                       'Changed objective and sampling; not a causal native-addition or replay-removal comparison.',
                       'No identity/pretraining-overlap certification or held-out generalization claim.',
                       'Useful expert fit does not qualify a router, detector or completed facial output.']}
    write_json(DEST / 'protocol.json', protocol)
    members = {name: ROOT / name for name in code}
    for path in sorted((ROOT / Path(REGISTRY).parent).rglob('*')):
        if path.is_file(): members[path.relative_to(ROOT).as_posix()] = path
    members['inputs/native_expert_protocol.json'] = DEST / 'protocol.json'
    members['inputs/phase4_split.json'] = ROOT / SPLIT
    for row in maps: members[row['path']] = DEST / 'lens' / Path(row['path']).name
    for name in ('NATIVE_EXPERT_VM.md', 'FROZEN_COMPLEMENTARITY_RESULTS.md', 'SUPPORTED_REAL_DATA.md'):
        members[name] = ROOT / name
    inventory = {name: sha(path) for name, path in sorted(members.items())}
    write_json(DEST / 'inventory.json', inventory)
    with tarfile.open(ARCHIVE, 'x:gz') as tar:
        for name, path in sorted(members.items()): tar.add(path, arcname=PREFIX + '/' + name, recursive=False)
        tar.add(DEST / 'inventory.json', arcname=PREFIX + '/inventory.json', recursive=False)
    with ARCHIVE.with_name(ARCHIVE.name + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(sha(ARCHIVE) + '  ' + ARCHIVE.name + '\n')
    report = {'complete': True, 'protocol_sha256': sha(DEST / 'protocol.json'), 'inventory_sha256': sha(DEST / 'inventory.json'),
        'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size, 'files_in_archive': len(members) + 1,
        'code_files': len(code), 'supported_dataset_files': 252, 'source_moments': moment_check,
        'frozen_mask_hashes_rechecked': len(frozen['mask_sha256']), 'fixture_cases_verified': 280,
        'local_model_forward_images': 0, 'local_optimizer_constructed': False, 'local_optimizer_updates': 0,
        'actual_cuda_execution_verified': False, 'stage_a_packaged': True, 'model_promoted': False,
        'vm_reuse_bytes_not_uploaded': sum((ROOT / ASSETS[role][0]).stat().st_size for role in ('model','optimizer','fixture_pixels'))}
    write_json(DEST / 'build.json', report); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
