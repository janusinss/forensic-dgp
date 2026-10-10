"""Finite CPU inference initialization proof; no training, gradients or VM use."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_bank_comparison_v1_model import CorrectedCurrentDGP, ConditionedBankDGP
from cctv_dgp_generative_bank_v1 import load_prior, sha
from dgp_frozen_inference_v2 import load_frozen_dgp_restorer


def write(path, obj):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return digest.hexdigest()


def run(root):
    root = root.resolve()
    assert root.is_relative_to(ROOT / 'outputs') and not root.exists()
    root.mkdir(parents=True)
    parent = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1'
    protocol = json.loads((parent / 'protocol.json').read_text())
    source = ROOT / 'outputs/cctv_dgp_generative_bank_source_v1'
    implementation = ROOT / 'outputs/cctv_dgp_generative_bank_probe_v1/implementation'
    old_probe = json.loads((ROOT / 'outputs/cctv_dgp_generative_bank_probe_v1_independent_audit.json').read_text())
    assert old_probe['complete']
    source_paths = [ROOT / 'scripts/cctv_dgp_bank_comparison_v1_model.py', ROOT / 'scripts/cctv_dgp_generative_bank_v1.py',
        ROOT / 'scripts/verify_cctv_dgp_conditioned_bank_initialization_v1.py', parent / 'protocol.json',
        parent / 'weights/dgp_v2.pth', source / 'StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth',
        implementation / 'generator.py', implementation / 'native_ops.py',
        ROOT / 'models/dgp_synthesizer.py', ROOT / 'models/fpn_mobilenet.py', ROOT / 'cctv_dgp_frozen_norm.py']
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    cases = protocol['cases']
    assert len(cases) == 100 and len(protocol['native_development']) == 24
    for name in {c['input'] for c in cases} | {r[k] for r in protocol['references'] for k in ['target', 'observed']}:
        assert sha(parent / name) == protocol['assets_sha256'][name]
    plan = {'format': 'conditioned-bank-initial-parity-v1', 'source_bindings': bindings,
        'parent': str(parent), 'cases': cases, 'references': protocol['references'],
        'native_development': protocol['native_development'], 'CPU_threads': 4,
        'worker_seconds_cap': 900, 'output_bytes_cap': 512 * 1024**2,
        'minimum_disk_reserve_bytes': 1024**3, 'paired_TRAIN_cases': 100,
        'native_unpaired_DEV_cases': 24, 'reserved_final_pixels': 0,
        'prior_native_checkpoint_size': 512, 'prior_feature_path_stops_at': 256,
        'mean_style_seed': 20261010, 'mean_style_Z_samples': 256, 'mean_style_mapping_calls': 1,
        'new_conditioner_seed': 20261010, 'model_forwards': {'original': 124, 'A': 124, 'B': 124},
        'initial_B_zero_fusion_expected': True, 'initial_conditioning_gradient_expected_zero': True,
        'conditioning_gradient_must_be_checked_after_first_fusion_update_on_VM': True,
        'gradient_queries': 0, 'backwards': 0, 'optimizer_updates': 0,
        'initializers_not_new_trained_checkpoints': True, 'app_promotion': False,
        'visual_plan': {'paired_sheets': 2, 'native_sheet': 1,
            'paired_refs': [protocol['references'][i]['id'] for i in [0, 50 // 5]],
            'native_ids': [r['id'] for r in protocol['native_development'][:4]],
            'initial_parity_not_quality_improvement': True}}
    write(root / 'plan.json', plan)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    def clock():
        assert time.monotonic() - start < 900, 'Finite900s local inference stop'
        assert shutil.disk_usage(root).free >= 1024**3, 'Local1GiB reserve stop'
        assert sum(f.stat().st_size for f in root.rglob('*') if f.is_file()) < 512 * 1024**2, 'Local512MiB output stop'

    def pixels(path, mode='RGB'):
        with Image.open(path) as im:
            assert im.size == (256, 256)
            return np.asarray(im.convert(mode)).copy()

    def tensor(a):
        return torch.from_numpy(a.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]

    try:
        original, _ = load_frozen_dgp_restorer(parent / 'weights/dgp_v2.pth', expected_sha256=protocol['original_checkpoint_sha256'])
        prior = load_prior(source, implementation, '05f5d33d79b32a3355cae3ede30e7ee06a90e56c60b1c2efe4ddd0d0e5a2959f')
        with torch.inference_mode():
            z = torch.randn(256, 512, generator=torch.Generator().manual_seed(20261010))
            mean_style = prior.get_latent(z).mean(0).clone()
        np.save(root / 'mean_style.npy', mean_style.numpy(), allow_pickle=False)
        A = CorrectedCurrentDGP(original.net).eval().requires_grad_(False)
        B = ConditionedBankDGP(original.net, prior, mean_style).eval().requires_grad_(False)
        before = {label: state_digest(model.state_dict()) for label, model in [('original', original), ('A', A), ('B', B)]}
        torch.save({k: v.detach().cpu().clone() for k, v in A.state_dict().items()}, root / 'initial_A.pth')
        torch.save({k: v.detach().cpu().clone() for k, v in B.state_dict().items()}, root / 'initial_B.pth')
        refs = {r['id']: r for r in protocol['references']}
        rows = []
        features_first = None
        feature_differences = []
        entries = [{'id': c['id'], 'role': 'TRAIN', 'input': c['input'],
            'observed': refs[c['source_person_or_reference']]['observed'],
            'target': refs[c['source_person_or_reference']]['target'],
            'source': c['source'], 'reference': c['source_person_or_reference'], 'profile': c['profile']} for c in cases]
        entries += [{'id': c['id'], 'role': 'native_unpaired_DEV', 'input': c['input'],
            'observed': c['observed'], 'target': None} for c in protocol['native_development']]
        for folder in ['raw', 'png', 'inputs', 'sheets']:
            (root / folder).mkdir()
        with torch.inference_mode():
            for index, c in enumerate(entries):
                clock()
                camera = pixels(parent / c['input'])
                mask = pixels(parent / c['observed'], 'L') > 0
                x = tensor(camera)
                baseline = original(x)
                a = A(x)
                b, features = B(x, return_features=True)
                assert torch.equal(a, baseline) and torch.equal(b, baseline), c['id']
                if index == 0:
                    features_first = features['64'].clone()
                elif index < 5:
                    feature_differences.append(float((features['64'] - features_first).abs().max()))
                raw = baseline[0].permute(1, 2, 0).numpy().copy()
                raw[~mask] = camera[~mask].astype(np.float32) / np.float32(255)
                np.save(root / 'raw' / (c['id'] + '.npy'), raw, allow_pickle=False)
                png = np.floor(raw * np.float32(255)).astype(np.uint8)
                png[~mask] = camera[~mask]
                Image.fromarray(camera).save(root / 'inputs' / (c['id'] + '.png'))
                Image.fromarray(png).save(root / 'png' / (c['id'] + '.png'))
                if c['target'] is not None:
                    Image.fromarray(pixels(parent / c['target'])).save(root / 'inputs' / (c['id'] + '_target.png'))
                rows.append({'id': c['id'], 'role': c['role'], 'A_raw_max_error': 0., 'B_raw_max_error': 0.,
                    'A_B_PNG_equal': True, 'raw_hash': hashlib.sha256(raw.tobytes()).hexdigest(),
                    'feature_shapes': {n: list(v.shape) for n, v in features.items()}})
                if index == 0 or (index + 1) % 20 == 0:
                    print(json.dumps({'initial_parity': index + 1, 'of': 124, 'seconds': time.monotonic() - start}), flush=True)
        assert max(feature_differences) > 0, 'Conditioned bank internal features must respond to input'
        # Initial output aliases are exact and explicit, not claims of trained gains.
        for name in plan['visual_plan']['paired_refs']:
            selected = [c for c in entries if c.get('reference') == name]
            assert len(selected) == 5
            sheet = Image.new('RGB', (5 * 256, 5 * 278), 'white')
            draw = ImageDraw.Draw(sheet)
            for row, c in enumerate(selected):
                camera = Image.open(root / 'inputs' / (c['id'] + '.png'))
                target = Image.open(root / 'inputs' / (c['id'] + '_target.png'))
                output = Image.open(root / 'png' / (c['id'] + '.png'))
                for column, (tile, label) in enumerate(zip([camera, target, output, output, output],
                        ['input', 'paired target', 'retained DGP', 'A initializer', 'B initializer'])):
                    sheet.paste(tile, (column * 256, row * 278 + 22))
                    draw.text((column * 256 + 2, row * 278 + 3), c['profile'] + ': ' + label, fill='black')
            sheet.save(root / 'sheets' / (name + '.png'))
        sheet = Image.new('RGB', (4 * 256, 4 * 278), 'white')
        draw = ImageDraw.Draw(sheet)
        for row, id_ in enumerate(plan['visual_plan']['native_ids']):
            camera = Image.open(root / 'inputs' / (id_ + '.png'))
            output = Image.open(root / 'png' / (id_ + '.png'))
            for column, (tile, label) in enumerate(zip([camera, output, output, output],
                    ['native input', 'retained DGP', 'A initializer', 'B initializer'])):
                sheet.paste(tile, (column * 256, row * 278 + 22))
                draw.text((column * 256 + 2, row * 278 + 3), id_ + ': ' + label, fill='black')
        sheet.save(root / 'sheets/native_initial_parity.png')
        after = {label: state_digest(model.state_dict()) for label, model in [('original', original), ('A', A), ('B', B)]}
        assert before == after
        for n, expected in bindings.items():
            assert sha(ROOT / n) == expected
        clock()
        result = {'complete': True, 'plan_sha256': sha(root / 'plan.json'), 'seconds': time.monotonic() - start,
            'rows': rows, 'state_hashes': before, 'state_unchanged': True,
            'mean_style_sha256': sha(root / 'mean_style.npy'),
            'initial_A_sha256': sha(root / 'initial_A.pth'), 'initial_B_sha256': sha(root / 'initial_B.pth'),
            'A_parameter_tensors': len(A.learning_parameters()), 'A_parameter_elements': sum(p.numel() for p in A.learning_parameters()),
            'B_parameter_tensors': len(B.learning_parameters()), 'B_parameter_elements': sum(p.numel() for p in B.learning_parameters()),
            'B_input_feature_max_differences': feature_differences,
            'model_forwards': plan['model_forwards'], 'style_mapping_calls': 1,
            'neural_inference_only': True, 'optimizer_updates': 0, 'backwards': 0, 'gradient_queries': 0,
            'new_conditioning_untrained': True, 'CUDA_and_gradient_parity_pending': True,
            'model_qualification': False, 'goal_complete': False}
        write(root / 'results.json', result)
        write(root / 'manifest.json', {f.relative_to(root).as_posix(): sha(f) for f in sorted(root.rglob('*')) if f.is_file()})
        print(json.dumps({k: result[k] for k in ['complete', 'seconds', 'A_parameter_elements', 'B_parameter_elements', 'optimizer_updates']}), flush=True)
    except BaseException as exc:
        write(root / 'failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'seconds': time.monotonic() - start, 'optimizer_updates': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
