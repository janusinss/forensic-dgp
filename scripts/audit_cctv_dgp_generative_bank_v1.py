"""Independent saved-artifact/operator/one-seed replay prerequisite audit.

Does not import the preparation runner. No gradients, optimizer or CCTV pixels.
This receipt qualifies component mechanics only, not restored-face quality.
"""
import argparse
import ast
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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name in sorted(state):
        a = state[name].detach().cpu().numpy()
        h.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def numpy_filter(x, kernel, up, down, pad):
    b, c, h, w = x.shape
    expanded = np.zeros((b, c, h * up, w * up), dtype=x.dtype)
    expanded[:, :, ::up, ::up] = x
    expanded = np.pad(expanded, ((0, 0), (0, 0), (max(pad[0], 0), max(pad[1], 0)),
                                (max(pad[0], 0), max(pad[1], 0))))
    if pad[0] < 0:
        expanded = expanded[:, :, -pad[0]:, -pad[0]:]
    if pad[1] < 0:
        expanded = expanded[:, :, :pad[1], :pad[1]]
    windows = np.lib.stride_tricks.sliding_window_view(expanded, kernel.shape, axis=(-2, -1))
    return np.einsum('bcyxij,ij->bcyx', windows, kernel[::-1, ::-1])[:, :, ::down, ::down]


def audit(root, receipt):
    root = root.resolve()
    assert not receipt.exists()
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    p = json.loads((root / 'plan.json').read_text())
    r = json.loads((root / 'results.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    assert r['complete'] and r['plan_sha256'] == digest(root / 'plan.json')
    actual_files = {f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file()}
    assert actual_files == set(manifest) | {'manifest.json'}
    for name, expected in manifest.items():
        assert digest(root / name) == expected, name
    source = Path(p['source_root'])
    assert digest(source / 'acquisition.json') == p['source_acquisition_sha256']
    acquisition = json.loads((source / 'acquisition.json').read_text())
    for name, record in acquisition['bindings'].items():
        assert digest(source / name) == record['sha256'] and (source / name).stat().st_size == record['bytes']
    assert p['standalone_prior_sha256'] == acquisition['bindings']['StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth']['sha256']
    upstream = ast.parse((source / 'stylegan2_arch.py').read_text())
    adapted = ast.parse((root / 'implementation/generator.py').read_text())
    ref = {n.name: n for n in upstream.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    source_bodies = 0
    for node in adapted.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            old = ref[node.name]
            if isinstance(old, ast.ClassDef):
                old.decorator_list = []
            assert ast.dump(old, include_attributes=False) == ast.dump(node, include_attributes=False), node.name
            source_bodies += 1
    native_ref = next(n for n in ast.parse((source / 'upfirdn2d.py').read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'upfirdn2d_native')
    native_new = next(n for n in ast.parse((root / 'implementation/native_ops.py').read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'upfirdn2d_native')
    assert ast.dump(native_ref) == ast.dump(native_new)
    package = '_independent_bank_audit_v1'
    spec = importlib.util.spec_from_file_location(package, root / 'implementation/__init__.py',
        submodule_search_locations=[str(root / 'implementation')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    ops = importlib.import_module(package + '.native_ops')
    gen = importlib.import_module(package + '.generator')
    operator_errors = []
    x = (np.arange(2 * 3 * 5 * 7, dtype=np.float64).reshape(2, 3, 5, 7) - 90) / 120
    kernel = np.array([[.1, .2, .1], [.1, .3, .05], [.02, .03, .1]], dtype=np.float64)
    with torch.inference_mode():
        for up in [1, 2]:
            for down in [1, 2]:
                for pad in [(0, 0), (1, 2), (-1, 2)]:
                    expected = numpy_filter(x, kernel, up, down, pad)
                    actual = ops.upfirdn2d(torch.from_numpy(x), torch.from_numpy(kernel), up, down, pad).numpy()
                    error = float(np.abs(expected - actual).max())
                    assert expected.shape == actual.shape and error <= 1e-12
                    operator_errors.append(error)
        for shape in [(2, 3), (2, 3, 5, 7)]:
            a = np.linspace(-1, 1, np.prod(shape), dtype=np.float64).reshape(shape)
            bias = np.array([.1, -.2, .3], dtype=np.float64)
            value = a + bias.reshape((1, 3) + (1,) * (len(shape) - 2))
            expected = np.where(value >= 0, value, .2 * value) * np.sqrt(2)
            actual = ops.fused_leaky_relu(torch.from_numpy(a), torch.from_numpy(bias)).numpy()
            error = float(np.abs(expected - actual).max())
            assert error <= 1e-12
            operator_errors.append(error)
    # Every saved RGB composition is checked with independent factor-two averaging.
    saved_raws, tiles = [], []
    bilinear_error = 0.0
    for i, seed in enumerate([0, 1, 0]):
        raw = np.load(root / f'call{i}_raw512.npy', allow_pickle=False)
        display = np.load(root / f'call{i}_display256.npy', allow_pickle=False)
        assert raw.shape == (512, 512, 3) and raw.dtype == display.dtype == np.float32
        assert np.isfinite(raw).all() and np.isfinite(display).all()
        unit = (np.clip(raw, -1, 1) + 1) / 2
        reconstructed = (unit[::2, ::2].astype(np.float64) + unit[1::2, ::2] +
                         unit[::2, 1::2] + unit[1::2, 1::2]) / 4
        error = float(np.abs(reconstructed - display).max())
        assert error <= 2e-7
        bilinear_error = max(bilinear_error, error)
        expected_png = np.floor(display * 255).astype(np.uint8)
        assert np.array_equal(np.array(Image.open(root / f'call{i}_display256.png')), expected_png)
        saved_raws.append(raw)
        if i < 2:
            tiles.append(expected_png)
    assert np.array_equal(saved_raws[0], saved_raws[2])
    assert np.array_equal(np.array(Image.open(root / 'generated_prior_seeds_not_restoration.png')), np.concatenate(tiles, 1))
    # Independent reference generator, hooks rather than the runner's feature loop.
    weights = torch.load(source / 'StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth', map_location='cpu', weights_only=True)
    assert state_digest(weights['params_ema']) == r['source_state_sha256']
    prior = gen.StyleGAN2Generator(512, num_style_feat=512, num_mlp=8, channel_multiplier=1,
                                  resample_kernel=(1, 3, 3, 1), lr_mlp=.01)
    prior.load_state_dict(weights['params_ema'], strict=True)
    prior.eval().requires_grad_(False)
    features = {}
    hooks = []
    for resolution, index in [(16, 3), (32, 5), (64, 7), (128, 9), (256, 11)]:
        hooks.append(prior.style_convs[index].register_forward_hook(
            lambda module, args, output, res=resolution: features.__setitem__(res, output.detach().cpu().numpy())))
    z = torch.randn(1, 512, generator=torch.Generator().manual_seed(0))
    assert np.array_equal(z.numpy(), np.load(root / 'call0_z.npy', allow_pickle=False))
    with torch.inference_mode():
        raw = prior([z], randomize_noise=False)[0][0].permute(1, 2, 0).numpy()
    for hook in hooks:
        hook.remove()
    assert np.array_equal(raw, saved_raws[0]), 'Independent CPU reference replay differs'
    for res, a in features.items():
        assert np.array_equal(a, np.load(root / f'seed0_feature{res}.npy', allow_pickle=False))
    assert state_digest(prior.state_dict()) == r['source_state_sha256']
    assert r['prior_generator_forwards'] == p['prior_generator_forward_limit'] == 6
    assert r['trainable_parameter_elements'] == 0 and p['reserved_final_images'] == 0
    workspace = Path(__file__).resolve().parents[1]
    for name, expected in p['protected_before'].items():
        assert digest(workspace / name) == expected and r['protected_after'][name] == expected
    # Local gradient-enabled public API is refused before any network evaluation.
    import cctv_dgp_generative_bank_v1 as public
    try:
        public.bank_features(prior, z)
    except RuntimeError as exc:
        assert 'manual VM tmux' in str(exc)
    else:
        raise AssertionError('Local gradient-enabled public API was not refused')
    result = {'complete': True, 'manifest_sha256': digest(root / 'manifest.json'),
        'results_sha256': digest(root / 'results.json'), 'checker_sha256': digest(__file__),
        'artifact_bindings': len(manifest), 'source_bindings': len(acquisition['bindings']),
        'unchanged_generator_source_bodies': source_bodies, 'upstream_native_filter_body_exact': True,
        'independent_numpy_operator_fixtures': len(operator_errors), 'operator_max_error': max(operator_errors),
        'RGB_compositions_checked': 3, 'maximum_bilinear_float_error': bilinear_error,
        'independent_CPU_generator_replays': 1, 'independent_CPU_feature_replays': 5,
        'replay_max_error': 0.0, 'unchanged_protected_files': len(p['protected_before']),
        'local_gradient_API_refused': True, 'seconds': time.monotonic() - start,
        'optimizer_updates': 0, 'backwards': 0, 'gradient_queries': 0,
        'CCTV_or_clean_face_inputs': 0, 'CUDA_parity_verified': False,
        'restoration_model_qualified': False, 'goal_complete': False}
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.receipt)
