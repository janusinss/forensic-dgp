"""Human-manual L4 precision/activation diagnostic. No optimizer or restorer."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request

NAME = 'cctv_dgp_recognizer_precision_v1_vm'
STEM = 'cctv-dgp-recognizer-precision-v1'
STAGE = 'outputs/recognizer_precision_v1'
STATE = '9ee66c2faefe0b82224c2cea6811073fbf70e8036808e293878109f078ea3963'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(root, inputs, response, bank, pin):
    assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json')
    assert p['format'] == 'recognizer-precision-v1' and p['name'] == NAME
    for folder, field in [(root, 'assets_sha256'), (inputs, 'input_required_sha256'),
                          (response, 'response_required_sha256'), (bank, 'bank_required_sha256')]:
        for n, h in p[field].items():
            path = folder / n
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder)
            assert sha(path) == h, n
    assert p['recognizer_state'] == STATE and p['expected_torch'] == '2.9.1+cu129'
    assert p['forward_cap'] == 16 and p['gradient_cap'] == 40 and p['optimizer_updates'] == p['full_epochs'] == 0
    assert p['labels'] == ['initial', 'plus02', 'minus02', 'repeat']
    assert p['minimum_free_bytes'] == 3 * 1024**3 and p['disk_reserve_bytes'] == 1024**3
    assert p['output_byte_cap'] == 256 * 1024**2 and p['measurement_seconds'] == 360
    assert p['scientific_gates_unchanged'] and not p['app_promotion'] and not p['automatic_follow_on']
    assert p['scientific_thresholds'] == read(bank / 'protocol.json')['scientific_thresholds']
    sys.path.insert(0, str(root))
    from precision_contract import select_references
    choices = select_references(read(root / 'evidence/boundary_review.json')['case_records'],
        read(bank / 'protocol.json')['cases'], p['sources'])
    assert choices == p['selected_references'] and len(p['reference_blocks']) == 2 and len(p['case_ids']) == 10
    assert p['reference_blocks'] == [b for b in read(response / 'protocol.json')['reference_blocks']
                                   if b['case_ids'][0] in p['case_ids']]
    assert p['case_ids'] == [c for b in p['reference_blocks'] for c in b['case_ids']]
    return p


def require_vm(root, inputs, response, bank, idle=True):
    assert platform.system() == 'Linux', 'Manual Linux L4 precision diagnostic only'
    assert platform.node().split('.')[0] == 'forensic-dgp-thesis'
    names = [(root, NAME), (inputs, 'cctv_dgp_appearance_input_response_v1_vm'),
             (response, 'cctv_dgp_conditioning_response_v1_vm'), (bank, 'cctv_dgp_bank_photometric_guard_v1_vm')]
    for folder, name in names:
        assert folder == (Path.home() / 'forensic-dgp' / name).resolve()
    assert Path.cwd().resolve() == root and os.environ.get('TMUX'), 'Human manual tmux required'
    assert sys.prefix == str(Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type', headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(req, timeout=3) as f:
        assert f.read().decode().strip().endswith('/g2-standard-4')
    if idle:
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=10, check=True)
        assert not r.stdout.strip(), 'Competing GPU workload; preserve it'


def measure(root, inputs, response, bank, p, pin):
    assert not (root / 'outputs').exists(), 'Used diagnostic; retain it'
    out = root / STAGE; out.mkdir(parents=True)
    start = time.monotonic(); progress = dict(stage='storage preflight')
    counts = dict(recognizer=0, gradient_queries=0, restorer=0, optimizer_updates=0,
                  accepted_updates=0, learned_weight_updates=0)
    models = {}; initial_rng = None; torch = None
    try:
        assert shutil.disk_usage(root).free >= p['minimum_free_bytes'], 'Need3GiB free; retain stop'
        import resource
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, buffer_hash
        from precision_contract import project
        assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L4'
        assert str(torch.__version__) == p['expected_torch']
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        torch.manual_seed(20261014); torch.cuda.manual_seed_all(20261014)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        def stop(_sig, _frame):
            raise TimeoutError('Finite precision diagnostic stop')
        signal.signal(signal.SIGTERM, stop)
        def rng():
            return dict(torch_CPU_RNG=torch.get_rng_state().numpy(),
                **{f'torch_CUDA_RNG_{i}': v.cpu().numpy() for i, v in enumerate(torch.cuda.get_rng_state_all())})
        def restore_rng(values):
            torch.set_rng_state(torch.from_numpy(values['torch_CPU_RNG']))
            torch.cuda.set_rng_state_all([torch.from_numpy(values[f'torch_CUDA_RNG_{i}']) for i in range(torch.cuda.device_count())])
        def save(path, value):
            with path.open('xb') as f:
                np.save(f, value, allow_pickle=False)
        def clock():
            torch.cuda.synchronize()
            assert time.monotonic() - start < 360, 'Measurement time stop'
            assert shutil.disk_usage(root).free >= p['disk_reserve_bytes'], 'Storage reserve stop'
            assert torch.cuda.max_memory_allocated() <= p['peak_VRAM_bytes'], 'VRAM stop'
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= p['peak_RSS_bytes'], 'RSS stop'
            assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) <= p['output_byte_cap'], 'Output cap stop'
            assert counts['recognizer'] <= 16 and counts['gradient_queries'] <= 40
            assert all(counts[k] == 0 for k in ['restorer', 'optimizer_updates', 'accepted_updates', 'learned_weight_updates'])
        original = FixedObservedIdentity(bank / 'weights/w600k_r50.onnx', 'cuda')
        assert state_hash(original) == STATE
        models = dict(FP32=original, FP64=copy.deepcopy(original).double())
        states = {k: state_hash(m) for k, m in models.items()}
        buffers = {k: buffer_hash(m) for k, m in models.items()}
        def roundtrip_state(model):
            return state_hash({n: v.float() if v.is_floating_point() else v for n, v in model.state_dict().items()})
        assert roundtrip_state(models['FP64']) == STATE
        initial_rng = rng(); np.savez_compressed(out / 'initial_RNG.npz', **initial_rng)
        write(out / 'environment.json', dict(GPU=torch.cuda.get_device_name(0), torch=str(torch.__version__),
            host=platform.node(), manual_tmux=True, states=states, buffers=buffers,
            FP64_state_roundtrip_to_original_FP32=roundtrip_state(models['FP64']),
            FP64_is_disposable_precision_copy_not_training=True, optimizer_constructed=False,
            matmul_tf32=False, cudnn_tf32=False, cudnn_benchmark=False, cudnn_deterministic=True))
        old = read(bank / 'protocol.json'); refs = {r['id']: r for r in old['references']}
        parent = response / 'outputs/conditioning_response_v1'
        for precision, model in models.items():
            dtype = torch.float32 if precision == 'FP32' else torch.float64
            folder = out / precision; folder.mkdir()
            assert all(not v.requires_grad and v.grad is None for v in model.parameters())
            floating = [v for v in model.state_dict().values() if v.is_floating_point()]
            assert floating and all(v.dtype == dtype for v in floating)
            for block in p['reference_blocks']:
                clock(); stem = f"b{block['begin']//5:03d}"
                cases = old['cases'][block['begin']:block['begin'] + 5]
                def pixels(path, mode='RGB'):
                    with Image.open(path) as im:
                        return np.asarray(im.convert(mode)).copy()
                target8 = np.stack([pixels(bank / refs[c['source_person_or_reference']]['target']) for c in cases])
                masks = np.stack([pixels(bank / refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in cases])
                fixed = np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in cases])
                target32 = np.asarray(target8, dtype=np.float32) / np.float32(255)
                target = torch.from_numpy(target32).permute(0, 3, 1, 2).contiguous().cuda().to(dtype)
                mask = torch.from_numpy(masks.astype(np.float32)).cuda()[:, None].to(dtype)
                grids = torch.from_numpy(fixed).cuda().to(dtype)
                before = np.load(parent / 'initial' / (stem + '.npy'), allow_pickle=False)
                baseline = torch.from_numpy(before).permute(0, 3, 1, 2).contiguous().cuda().to(dtype)
                changes = {label: np.load(parent / label / (stem + '.npy'), allow_pickle=False) for label in ['plus02', 'minus02']}
                for label in p['labels']:
                    clock(); phase_start = time.monotonic(); progress = dict(precision=precision, begin=block['begin'], label=label)
                    tag = stem + '_' + label; activation = {}; handles = []; floating_outputs = []
                    array = before if label in ['initial', 'repeat'] else changes[label]
                    rgb = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().cuda().to(dtype)
                    assert np.array_equal(rgb.detach().permute(0, 2, 3, 1).cpu().numpy().astype(np.float32), array)
                    rgb.requires_grad_(label in ['initial', 'repeat']); assert rgb.is_leaf
                    def check_output(name):
                        def hook(_m, _i, value):
                            values = value if isinstance(value, (list, tuple)) else [value]
                            for v in values:
                                if isinstance(v, torch.Tensor) and v.is_floating_point():
                                    assert v.dtype == dtype, 'Mixed precision inside encoder: ' + name
                                    floating_outputs.append(dict(module=name, dtype=str(v.dtype)))
                        return hook
                    def activation_hook(name):
                        def hook(_module, values):
                            value = values[0]; assert value.dtype == dtype and value.shape[0] == 15
                            a = value[:5].detach().cpu().numpy()
                            flags = np.stack([a > 0, a == 0]); packed = np.packbits(flags.reshape(-1), bitorder='little')
                            name_hash = hashlib.sha256(name.encode()).hexdigest()[:16]
                            path = folder / (tag + '_activation_' + name_hash + '.npy'); save(path, packed)
                            activation[name] = dict(path=path.name, shape=list(a.shape), packed_sha256=sha(path),
                                units=int(a.size), positive=int((a > 0).sum()), zero=int((a == 0).sum()),
                                negative=int((a < 0).sum()), module_type=type(_module).__name__, dtype=str(value.dtype))
                        return hook
                    for name, module in model.encoder.named_modules():
                        handles.append(module.register_forward_hook(check_output(name)))
                        if name == '':
                            def count_forward(_module, _values):
                                counts['recognizer'] += 1
                                assert counts['recognizer'] <= 16, 'Recognizer forward count stop'
                            handles.append(module.register_forward_pre_hook(count_forward))
                        if 'RELU' in type(module).__name__.upper():
                            handles.append(module.register_forward_pre_hook(activation_hook(name)))
                    try:
                        image_batch = torch.cat([rgb, baseline, target]); mask_batch = torch.cat([mask] * 3); grid_batch = torch.cat([grids] * 3)
                        if precision == 'FP32':
                            vectors = model.embedding(image_batch, mask_batch, grid_batch)
                        else:
                            crop = F.grid_sample(image_batch, grid_batch, mode='bilinear', padding_mode='zeros', align_corners=False)
                            support = F.grid_sample(mask_batch, grid_batch, mode='nearest', padding_mode='zeros', align_corners=False)
                            neutral = float(np.float32(128 / 255))
                            crop = crop * support + neutral * (1 - support)
                            vectors = F.normalize(model.encoder(crop * 2 - 1), dim=1)
                            assert vectors.dtype == torch.float64 and torch.isfinite(vectors).all()
                        assert activation and floating_outputs, 'No activation trace; retain stop'
                        v = vectors.detach().cpu().numpy().copy()
                        np.savez_compressed(folder / (tag + '_embeddings.npz'), raw=v)
                        if precision == 'FP32':
                            actual_label = 'initial' if label == 'repeat' else label
                            with np.load(parent / actual_label / (stem + '_embeddings.npz'), allow_pickle=False) as f:
                                np.testing.assert_allclose(v, f['raw'], rtol=1e-6, atol=1e-6)
                        loss = 1 - (vectors[:5].double() * vectors[10:15].detach().double()).sum(1)
                        rows = []; gradients = []
                        for i, case in enumerate(cases):
                            row = dict(id=case['id'], source=case['source'], profile=case['profile'], loss=float(loss[i].detach()))
                            if label in ['initial', 'repeat']:
                                all_g = torch.autograd.grad(loss[i], rgb, retain_graph=i < 4, allow_unused=False)[0]
                                counts['gradient_queries'] += 1
                                whole = all_g.detach().permute(0, 2, 3, 1).cpu().numpy().copy()
                                assert np.max(np.abs(np.delete(whole, i, axis=0))) == 0
                                g = whole[i]; assert np.isfinite(g).all(); gradients.append(g)
                                row.update(gradient_norm=float(np.linalg.norm(g.astype(np.float64))), cross_case_gradient_maximum=0,
                                    gradient_bytes_sha256=hashlib.sha256(g.tobytes()).hexdigest(),
                                    projections={k: project(g, before[i], a[i]) for k, a in changes.items()})
                                del all_g, whole, g
                            rows.append(row)
                        if gradients:
                            save(folder / (tag + '_RGB_gradients.npy'), np.stack(gradients))
                        write(folder / (tag + '.json'), dict(precision=precision, begin=block['begin'], label=label,
                            rows=rows, activations=activation, encoder_floating_outputs=floating_outputs))
                    finally:
                        for handle in handles:
                            handle.remove()
                    del rgb, vectors, loss, gradients, image_batch, mask_batch, grid_batch
                    if precision == 'FP64':
                        del crop, support
                    assert state_hash(model) == states[precision] and buffer_hash(model) == buffers[precision]
                    assert all(v.grad is None and not v.requires_grad for v in model.parameters())
                    clock(); seconds = time.monotonic() - phase_start
                    if precision == 'FP64' and block == p['reference_blocks'][0] and label == 'initial':
                        projection = seconds * 8 * 1.5
                        write(out / 'FP64_timing_projection.json', dict(first_phase_seconds=seconds, projected_seconds=projection, cap_seconds=300))
                        assert projection <= 300, 'FP64 timing projection stop'
                    print(json.dumps(dict(precision=precision, begin=block['begin'], label=label, seconds=seconds, counts=counts)), flush=True)
                del target, baseline, mask, grids, changes
        restore_rng(initial_rng); np.savez_compressed(out / 'restored_RNG.npz', **rng()); clock()
        assert counts['recognizer'] == 16 and counts['gradient_queries'] == 40
        assert state_hash(original) == STATE and roundtrip_state(models['FP64']) == STATE
        write(out / 'run_results.json', dict(complete_diagnostic=True, protocol_sha256=pin, counts=counts,
            seconds=time.monotonic() - start, peak_VRAM=torch.cuda.max_memory_allocated(),
            peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            states=states, buffers=buffers, FP64_state_roundtrip_to_original_FP32=STATE, Torch_RNG_restored=True,
            native_pixels_used=0, final_identity_pixels=0, full_epochs=0, model_qualified=False))
    except BaseException as error:
        rollback_error = None
        try:
            if initial_rng is not None:
                restore_rng(initial_rng)
            if models:
                assert state_hash(models['FP32']) == STATE and roundtrip_state(models['FP64']) == STATE
        except BaseException as second:
            rollback_error = repr(second)
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            progress=progress, counts=counts, seconds=time.monotonic() - start, rollback_error=rollback_error,
            optimizer_updates=0, accepted_updates=0, model_qualified=False))
        raise


def export(root, p, pin):
    start = time.monotonic(); assert not (root / 'export_manifest.json').exists()
    files = sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    payload = sum(f.stat().st_size for f in files)
    assert payload <= p['output_byte_cap'] + 16 * 1024**2
    assert shutil.disk_usage(root).free >= payload + p['disk_reserve_bytes']
    manifest = root / 'export_manifest.json'
    write(manifest, dict(protocol_sha256=pin, files={f.relative_to(root).as_posix(): sha(f) for f in files},
        complete_diagnostic=(root / STAGE / 'run_results.json').exists(), failure_present=(root / STAGE / 'failure.json').exists(),
        optimizer_updates=0, accepted_updates=0, complete_pilot=False, training_success_not_implied=True))
    archive = Path.home() / (STEM + '-results.tar.gz'); assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for f in files + [manifest]:
            assert time.monotonic() - start < 180 and shutil.disk_usage(root).free >= p['disk_reserve_bytes'] + f.stat().st_size
            tar.add(f, arcname=NAME + '/' + f.relative_to(root).as_posix(), recursive=False)
            assert archive.stat().st_size <= p['output_byte_cap'] + 16 * 1024**2
    digest = sha(archive)
    assert time.monotonic() - start < 180 and shutil.disk_usage(root).free >= p['disk_reserve_bytes']
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as f:
        f.write(digest + '  ' + archive.name + '\n')
    receipt = dict(complete=True, archive_sha256=digest, bytes=archive.stat().st_size, seconds=time.monotonic() - start,
        optimizer_updates=0, accepted_updates=0, training_success_not_implied=True)
    write(Path.home() / (STEM + '-export.json'), receipt); print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for flag in ['root', 'input-root', 'response-root', 'bank-root']:
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    for flag in ['verify-transfer', 'measure', 'export']:
        actions.add_argument('--' + flag, action='store_true')
    a = parser.parse_args()
    root, inputs, response, bank = [v.resolve() for v in [a.root, a.input_root, a.response_root, a.bank_root]]
    if a.measure:
        require_vm(root, inputs, response, bank)
    p = verify(root, inputs, response, bank, a.protocol_sha)
    if a.verify_transfer:
        print(json.dumps(dict(complete=True, assets=len(p['assets_sha256']), cases=10,
            actual_neural_or_gradient_calls=0, optimizer_updates=0, available_free_bytes=shutil.disk_usage(root).free)), flush=True)
    elif a.measure:
        measure(root, inputs, response, bank, p, a.protocol_sha)
    else:
        require_vm(root, inputs, response, bank, idle=False); export(root, p, a.protocol_sha)


if __name__ == '__main__':
    main()
