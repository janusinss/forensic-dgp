"""Independent initializer/file/composition/replay audit; CPU inference only.

Does not import the proof runner. Initial output parity is not learned quality.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_bank_comparison_v1_model import CorrectedCurrentDGP, ConditionedBankDGP
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def rgb(path):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.asarray(im.convert('RGB')).copy()


def audit(root, receipt):
    root = root.resolve()
    assert root.is_relative_to(ROOT / 'outputs') and not receipt.exists()
    stamp = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    p = json.loads((root / 'plan.json').read_text())
    r = json.loads((root / 'results.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    assert r['complete'] and r['plan_sha256'] == sha(root / 'plan.json')
    assert not (root / 'failure.json').exists()
    actual = {f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file()}
    assert actual == set(manifest) | {'manifest.json'}
    for name, expected in manifest.items():
        assert sha(root / name) == expected, name
    for name, expected in p['source_bindings'].items():
        assert sha(ROOT / name) == expected, name
    assert p['paired_TRAIN_cases'] == 100 and p['native_unpaired_DEV_cases'] == 24
    assert p['reserved_final_pixels'] == p['gradient_queries'] == p['backwards'] == p['optimizer_updates'] == 0
    parent = Path(p['parent'])
    refs = {item['id']: item for item in p['references']}
    entries = [{'id': c['id'], 'input': c['input'], 'mask': refs[c['source_person_or_reference']]['observed'],
                'target': refs[c['source_person_or_reference']]['target']} for c in p['cases']]
    entries += [{'id': c['id'], 'input': c['input'], 'mask': c['observed'], 'target': None} for c in p['native_development']]
    assert [c['id'] for c in entries] == [row['id'] for row in r['rows']]
    for entry, row in zip(entries, r['rows']):
        camera = rgb(parent / entry['input'])
        assert np.array_equal(camera, rgb(root / 'inputs' / (entry['id'] + '.png')))
        mask = np.array(Image.open(parent / entry['mask']).convert('L')) > 0
        raw = np.load(root / 'raw' / (entry['id'] + '.npy'), allow_pickle=False)
        assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
        assert 0 <= raw.min() <= raw.max() <= 1
        assert np.array_equal(raw[~mask], (camera.astype(np.float32) / np.float32(255))[~mask])
        png = np.floor(raw * np.float32(255)).astype(np.uint8)
        png[~mask] = camera[~mask]
        assert np.array_equal(png, rgb(root / 'png' / (entry['id'] + '.png')))
        assert row['raw_hash'] == hashlib.sha256(raw.tobytes()).hexdigest()
        assert row['A_raw_max_error'] == row['B_raw_max_error'] == 0 and row['A_B_PNG_equal']
        if entry['target'] is not None:
            assert np.array_equal(rgb(parent / entry['target']), rgb(root / 'inputs' / (entry['id'] + '_target.png')))
    # Check every declared sheet cell against its saved source; labels lie outside.
    cells = 0
    for reference in p['visual_plan']['paired_refs']:
        selected = [c for c in p['cases'] if c['source_person_or_reference'] == reference]
        sheet = np.array(Image.open(root / 'sheets' / (reference + '.png')).convert('RGB'))
        assert sheet.shape == (1390, 1280, 3) and len(selected) == 5
        for i, c in enumerate(selected):
            tiles = [rgb(root / 'inputs' / (c['id'] + '.png')), rgb(parent / refs[reference]['target'])]
            tiles += [rgb(root / 'png' / (c['id'] + '.png'))] * 3
            for j, tile in enumerate(tiles):
                assert np.array_equal(sheet[i*278+22:i*278+278, j*256:(j+1)*256], tile)
                cells += 1
    sheet = np.array(Image.open(root / 'sheets/native_initial_parity.png').convert('RGB'))
    assert sheet.shape == (1112, 1024, 3)
    for i, name in enumerate(p['visual_plan']['native_ids']):
        tiles = [rgb(root / 'inputs' / (name + '.png'))] + [rgb(root / 'png' / (name + '.png'))] * 3
        for j, tile in enumerate(tiles):
            assert np.array_equal(sheet[i*278+22:i*278+278, j*256:(j+1)*256], tile)
            cells += 1
    implementation = ROOT / 'outputs/cctv_dgp_generative_bank_probe_v1/implementation'
    package = '_conditioned_bank_independent_initializer_v1'
    spec = importlib.util.spec_from_file_location(package, implementation / '__init__.py',
        submodule_search_locations=[str(implementation)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    generator = importlib.import_module(package + '.generator')
    prior = generator.StyleGAN2Generator(512, num_style_feat=512, num_mlp=8, channel_multiplier=1,
        resample_kernel=(1, 3, 3, 1), lr_mlp=.01).eval().requires_grad_(False)
    source = ROOT / 'outputs/cctv_dgp_generative_bank_source_v1/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth'
    prior.load_state_dict(torch.load(source, map_location='cpu', weights_only=True)['params_ema'], strict=True)
    mean = np.load(root / 'mean_style.npy', allow_pickle=False)
    with torch.inference_mode():
        z = torch.randn(256, 512, generator=torch.Generator().manual_seed(20261010))
        recomputed_mean = prior.get_latent(z).mean(0).numpy()
    assert np.array_equal(mean, recomputed_mean)
    original, _ = load_frozen_dgp_restorer(parent / 'weights/dgp_v2.pth',
        expected_sha256='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b')
    A = CorrectedCurrentDGP(original.net).eval().requires_grad_(False)
    B = ConditionedBankDGP(original.net, prior, torch.from_numpy(mean)).eval().requires_grad_(False)
    initial = {}
    for label, model in [('A', A), ('B', B)]:
        saved = torch.load(root / ('initial_' + label + '.pth'), map_location='cpu', weights_only=True)
        assert state_digest(model.state_dict()) == state_digest(saved) == r['state_hashes'][label]
        model.load_state_dict(saved, strict=True)
        assert len(model.learning_parameters()) == r[label + '_parameter_tensors']
        assert sum(v.numel() for v in model.learning_parameters()) == r[label + '_parameter_elements']
        assert all(not v.requires_grad for v in model.parameters())
        initial[label] = state_digest(model.state_dict())
    assert state_digest(original.state_dict()) == r['state_hashes']['original']
    assert all(torch.count_nonzero(v) == 0 for m in [B.fusions, B.spatial_conditioners] for v in m.parameters())
    assert state_digest(B.prior.state_dict()) == state_digest(torch.load(source, map_location='cpu', weights_only=True)['params_ema'])
    replay_ids = [p['cases'][0]['id'], p['cases'][1]['id'], p['cases'][50]['id'], p['native_development'][0]['id']]
    replay = []
    with torch.inference_mode():
        for name in replay_ids:
            c = next(item for item in entries if item['id'] == name)
            camera = rgb(parent / c['input'])
            x = torch.from_numpy(camera.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
            mask = np.array(Image.open(parent / c['mask']).convert('L')) > 0
            old, a, b = original(x), A(x), B(x)
            assert torch.equal(old, a) and torch.equal(old, b)
            raw = old[0].permute(1, 2, 0).numpy().copy()
            raw[~mask] = (camera.astype(np.float32) / np.float32(255))[~mask]
            assert np.array_equal(raw, np.load(root / 'raw' / (name + '.npy'), allow_pickle=False))
            assert torch.equal(B(x, bank_enabled=False), b)
            replay.append({'id': name, 'maximum_error': 0.0})
    for label, model in [('A', A), ('B', B)]:
        assert state_digest(model.state_dict()) == initial[label]
        try:
            model(torch.zeros(1, 3, 256, 256))
        except AssertionError:
            pass
        else:
            raise AssertionError('Local gradient-enabled model API was not refused')
    for name, expected in p['source_bindings'].items():
        assert sha(ROOT / name) == expected
    result = {'complete': True, 'checker_sha256': sha(__file__), 'manifest_sha256': sha(root / 'manifest.json'),
        'results_sha256': sha(root / 'results.json'), 'artifact_bindings': len(manifest),
        'source_bindings': len(p['source_bindings']), 'saved_raw_PNG_compositions_checked': len(entries),
        'sheet_cells_checked': cells, 'independent_CPU_triplet_replays': replay,
        'mean_style_independently_recomputed': True, 'initializers_strictly_reconstructed': True,
        'A_parameter_elements': r['A_parameter_elements'], 'B_parameter_elements': r['B_parameter_elements'],
        'frozen_prior_weights_unchanged': True, 'local_gradient_APIs_refused': 2,
        'initial_zero_fusions_and_spatial_conditioners': True, 'seconds': time.monotonic() - stamp,
        'optimizer_updates': 0, 'backwards': 0, 'gradient_queries': 0,
        'CUDA_and_gradient_parity_pending': True, 'restoration_model_qualified': False, 'goal_complete': False}
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.receipt)
