"""Manual finite frozen B readout ablation; no training or optimizer imports."""
import argparse
from datetime import datetime, timezone
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

NAME = 'cctv_dgp_bank_path_ablation_v1_vm'
STEM = 'cctv-dgp-bank-path-ablation-v1'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
PARENT_PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
VARIANTS = ['enabled', 'disabled', 'drop16', 'drop32', 'drop64', 'drop128', 'drop256']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def verify(root, parent, pin):
    assert sha(root/'protocol.json') == pin
    p = read(root/'protocol.json')
    assert p['name'] == NAME and p['parent_protocol_sha256'] == PARENT_PIN
    assert sha(parent/'protocol.json') == PARENT_PIN
    assert p['variants'] == VARIANTS and p['optimizer_updates'] == p['gradient_queries'] == 0
    assert p['paired_cases'] == 100 and p['native_cases'] == 24 and p['final_identity_pixels'] == 0
    assert p['frozen_rejected_checkpoint_measurement_only'] and not p['app_promotion']
    assert p['worker_seconds'] == 1200 and p['external_seconds'] == 1230
    assert p['model_forward_cap'] == 312 and p['recognizer_forward_cap'] == 140
    assert p['inference_seconds'] == 900 and p['output_byte_cap'] == 1024**3
    assert p['minimum_free_bytes'] == 4*1024**3 and p['disk_reserve_bytes'] == 1024**3
    for folder, items in [(root, p['assets_sha256']), (parent, p['parent_required_sha256'])]:
        for name, expected in items.items():
            path = folder/name
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder)
            assert sha(path) == expected, name
    old = read(parent/'protocol.json')
    assert p['case_ids'] == old['preview_case_ids']
    assert p['native_case_ids'] == [c['id'] for c in old['native_development']]
    assert not read(parent/'outputs/bank_comparison_v1/outcome_B.json')['model_qualified']
    assert read(parent/'outputs/bank_comparison_v1/outcome_B.json')['updates'] == 50
    return p, old


def require_vm(root, parent):
    assert platform.system() == 'Linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root == (Path.home()/'forensic-dgp'/NAME).resolve()
    assert parent == (Path.home()/'forensic-dgp'/PARENT_NAME).resolve()
    assert os.environ.get('TMUX'), 'Launch manually inside tmux'
    assert sys.prefix == str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                                    headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=10)
    assert not result.stdout.strip(), 'Competing GPU workload; preserve it and do not launch'


def run(root, parent, p, old, pin):
    require_vm(root, parent)
    assert not (root/'outputs').exists(), 'Immutable diagnostic; no unchanged rerun'
    assert shutil.disk_usage(root).free >= p['minimum_free_bytes'], 'Need4GiB free'
    sys.path.insert(0, str(parent))
    import numpy as np
    from PIL import Image
    import torch
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_lossless_v1 import pack, unpack
    from frozen_capacity_contract import feature_support, exported_pixel_metrics, detail_metric
    from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L4'
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(20261010); torch.cuda.manual_seed_all(20261010)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    # Grad-enabled forward paths are prohibited throughout this diagnostic.
    torch.set_grad_enabled(False)
    start = time.monotonic()
    out = root/'outputs/path_ablation_v1'; out.mkdir(parents=True)
    counts = dict(B=0, recognizer=0, original=0, gradient_queries=0, backwards=0, optimizer_updates=0)
    progress = dict(stage='initialize', counts=counts)
    def stop(_signal, _frame):
        raise TimeoutError('External diagnostic timing stop; retain evidence')
    signal.signal(signal.SIGTERM, stop)
    write(out/'environment.json', dict(python=sys.version, torch=torch.__version__, CUDA=torch.version.cuda,
        GPU=torch.cuda.get_device_name(0), host=platform.node(), manual_tmux=True,
        frozen_rejected_state_for_diagnosis_only=True, no_new_checkpoint=True))

    def clock():
        torch.cuda.synchronize()
        assert time.monotonic()-start < p['worker_seconds'], 'Finite worker stop'
        assert shutil.disk_usage(root).free >= p['disk_reserve_bytes'], 'Disk reserve stop'
        assert torch.cuda.max_memory_allocated() <= 20*1024**3, 'VRAM stop'
        assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) <= p['output_byte_cap'], 'Output stop'
        assert counts['B'] <= p['model_forward_cap'] and counts['recognizer'] <= p['recognizer_forward_cap']
        assert not torch.is_grad_enabled()

    def pixels(path, mode='RGB'):
        with Image.open(path) as im:
            assert im.size == (256, 256)
            return np.asarray(im.convert(mode)).copy()

    def tensor(a):
        return torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None].cuda()

    try:
        original, _ = load_frozen_dgp_restorer(parent/'weights/dgp_v2.pth',
            expected_sha256=old['original_checkpoint_sha256'], device='cuda')
        identity = FixedObservedIdentity(parent/'weights/w600k_r50.onnx', 'cuda')
        prior = load_prior(parent/'prior_source', parent/'bank_implementation', old['standalone_prior_sha256']).cuda()
        mean = torch.from_numpy(np.load(parent/'initializers/mean_style.npy', allow_pickle=False)).cuda()
        candidate = ConditionedBankDGP(original.net, prior, mean).cuda().eval().requires_grad_(False)
        stored = torch.load(parent/'outputs/bank_comparison_v1/state_B50/full_state.pth',
                            map_location='cpu', weights_only=True)
        assert stored['updates'] == 50 and not stored['manual_failed_resume_allowed']
        candidate.load_state_dict(stored['model'], strict=True)
        del stored
        initial_state = state_hash(candidate); initial_buffers = buffer_hash(candidate)
        assert initial_state == p['rejected_B50_model_state']
        assert state_hash(original.net) == old['original_state'] and state_hash(identity) == old['recognizer_state']
        candidate.register_forward_hook(lambda *_: counts.__setitem__('B', counts['B']+1))
        identity.encoder.register_forward_hook(lambda *_: counts.__setitem__('recognizer', counts['recognizer']+1))
        active = {'drop': set()}
        handles = []
        for name, layer in candidate.fusions.items():
            def drop(_module, _args, value, name=name):
                return torch.zeros_like(value) if name in active['drop'] else value
            handles.append(layer.register_forward_hook(drop))

        def generated(x, variant):
            active['drop'] = {variant[4:]} if variant.startswith('drop') else set()
            return candidate(x, bank_enabled=variant != 'disabled')

        selected = [begin for begin in range(0, 3905, 5) if old['cases'][begin]['id'] in p['case_ids']]
        assert len(selected) == 20
        parent_out = parent/'outputs/bank_comparison_v1'
        refs = {r['id']: r for r in old['references']}
        # Prove that zero readouts equal skipping the entire bank on two sources.
        parity = []
        for begin in p['parity_begins']:
            block = old['cases'][begin:begin+5]
            x = torch.cat([tensor(pixels(parent/c['input'])) for c in block])
            active['drop'] = set(candidate.fusions)
            zeroed = candidate(x)
            active['drop'] = set()
            disabled = candidate(x, bank_enabled=False)
            assert torch.equal(zeroed, disabled), 'Bank readout ablation parity failed'
            parity.append(dict(begin=begin, source=block[0]['source'], maximum_error=0., cases=5))
        write(out/'parity.json', dict(complete=True, rows=parity, gradient_queries=0))
        clock()
        inference_start = time.monotonic()
        for variant in VARIANTS:
            progress.update(stage='inference', variant=variant)
            folder = out/variant
            for child in ['packs', 'previews', 'native']:
                (folder/child).mkdir(parents=True, exist_ok=True)
            rows, native = [], []
            for begin in selected:
                clock(); assert time.monotonic()-inference_start < p['inference_seconds'], 'Inference15-minute stop'
                block = old['cases'][begin:begin+5]; ids = [c['id'] for c in block]
                cameras = [pixels(parent/c['input']) for c in block]
                targets = [pixels(parent/refs[c['source_person_or_reference']]['target']) for c in block]
                masks = [pixels(parent/refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in block]
                supports = [feature_support(mask, c['landmarks5_canvas_xy']) for mask, c in zip(masks, block)]
                grids = torch.from_numpy(np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in block])).cuda()
                mask = torch.from_numpy(np.stack(masks).astype(np.float32)).cuda()[:, None]
                x = torch.cat([tensor(a) for a in cameras])
                pred = torch.where(mask.bool(), generated(x, variant), x)
                raw = pred.permute(0, 2, 3, 1).cpu().numpy().copy()
                base, _ = unpack(parent_out/'baseline/packs'/f'b{begin//5:03d}.npz', ids)
                pngs = [deliver(a, c, m) for a, c, m in zip(raw, cameras, masks)]
                vectors = identity.embedding(torch.cat([pred, torch.cat([tensor(q) for q in pngs]),
                    torch.cat([tensor(q) for q in targets])]), torch.cat([mask]*3), torch.cat([grids]*3)).cpu().numpy().copy()
                embeddings = np.stack([vectors[:5], vectors[5:10], vectors[10:15]], axis=1)
                pack(folder/'packs'/f'b{begin//5:03d}.npz', raw, ids, embeddings)
                if variant in ['enabled', 'disabled']:
                    old_label = 'B50' if variant == 'enabled' else 'B50_bank_disabled'
                    previous, previous_emb = unpack(parent_out/old_label/'packs'/f'b{begin//5:03d}.npz', ids, baseline=base)
                    assert np.array_equal(raw, previous), 'Same L4 saved raw replay differs'
                    assert np.array_equal(embeddings, previous_emb), 'Same L4 embedding replay differs'
                for i, c in enumerate(block):
                    a, png, target, observed = raw[i], pngs[i], targets[i], masks[i]
                    mraw, mpng, shift = mean_only(a, base[i], cameras[i], observed)
                    rm = pixel_metrics(a, target, observed); pm = exported_pixel_metrics(png, target, observed)
                    rm.update(ArcFace_observed_fixed=float(embeddings[i, 0]@embeddings[i, 2]),
                        landmark_high_frequency_MSE=detail_float(a, target, supports[i]),
                        constant_mean_shift_only_MSE=pixel_metrics(mraw, target, observed)['MSE'])
                    pm.update(ArcFace_observed_fixed=float(embeddings[i, 1]@embeddings[i, 2]),
                        landmark_high_frequency_MSE=detail_metric(png, target, supports[i]),
                        constant_mean_shift_only_MSE=exported_pixel_metrics(mpng, target, observed)['MSE'])
                    Image.fromarray(png).save(folder/'previews'/(c['id']+'.png'))
                    rows.append(dict(id=c['id'], source=c['source'], profile=c['profile'], raw=rm, png=pm,
                        postclip_mean_RGB_shift=shift.tolist(), raw_RGB_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                        PNG_RGB_sha256=hashlib.sha256(png.tobytes()).hexdigest()))
            for c in old['native_development']:
                clock(); assert time.monotonic()-inference_start < p['inference_seconds']
                camera = pixels(parent/c['input']); observed = pixels(parent/c['observed'], 'L') > 0
                x = tensor(camera)
                pred = torch.where(torch.from_numpy(observed).cuda()[None, None], generated(x, variant), x)
                raw = pred[0].permute(1, 2, 0).cpu().numpy().copy(); png = deliver(raw, camera, observed)
                np.save(folder/'native'/(c['id']+'.npy'), raw, allow_pickle=False)
                Image.fromarray(png).save(folder/'native'/(c['id']+'.png'))
                if variant in ['enabled', 'disabled']:
                    old_label = 'B50' if variant == 'enabled' else 'B50_bank_disabled'
                    assert np.array_equal(raw, np.load(parent_out/old_label/'native'/(c['id']+'.npy'), allow_pickle=False))
                native.append(dict(id=c['id'], raw_RGB_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    PNG_RGB_sha256=hashlib.sha256(png.tobytes()).hexdigest(), unpaired_no_clean_metrics=True))
            write(folder/'metrics.json', dict(complete=True, variant=variant, rows=rows, native_unpaired_rows=native,
                groups={stage: review_groups(rows, stage) for stage in ['raw', 'png']}, model_state=initial_state,
                source_and_profile_metrics_TRAIN_only=True, full_3905_gate_not_tested=True, model_qualified=False))
            print(dict(variant=variant, paired_cases=len(rows), native_cases=len(native), seconds=time.monotonic()-start), flush=True)
        for handle in handles:
            handle.remove()
        assert state_hash(candidate) == initial_state and buffer_hash(candidate) == initial_buffers
        assert state_hash(original.net) == old['original_state'] and state_hash(identity) == old['recognizer_state']
        assert all(not m.training for model in [candidate, original, identity] for m in model.modules())
        assert all(not v.requires_grad and v.grad is None for model in [candidate, original, identity] for v in model.parameters())
        assert counts['B'] == 312, counts
        # 20 five-case paired batches + 24 single native calls, seven variants,
        # plus four zero-readout parity calls = 312 total B calls.
        assert counts['recognizer'] == 140
        clock()
        verify(root, parent, pin)
        write(out/'run_results.json', dict(complete=True, counts=counts, model_state_unchanged=initial_state,
            paired_variant_compositions=700, native_variant_compositions=168, full_epochs=0,
            local_app_promotion=False, frozen_inference_only=True, diagnostic_not_model_qualification=True,
            seconds=time.monotonic()-start, peak_VRAM=torch.cuda.max_memory_allocated()))
    except BaseException as exc:
        write(out/'failure.json', dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
            progress=progress, seconds=time.monotonic()-start, optimizer_updates=0,
            training_success_not_implied=True, prior_quality_failure_retained=True))
        raise


def export(root, parent, p, pin):
    assert platform.system() == 'Linux' and root == (Path.home()/'forensic-dgp'/NAME).resolve()
    assert os.environ.get('TMUX') and sys.prefix == str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    start = time.monotonic()
    manifest = root/'export_manifest.json'
    assert not manifest.exists()
    files = sorted(f for f in root.rglob('*') if f.is_file() and f.name not in
                   {'export.log', 'export_exit_code.txt', 'supervisor_receipt.json'})
    assert all(not f.is_symlink() and f.resolve().is_relative_to(root) for f in files)
    assert sum(f.stat().st_size for f in files) <= p['output_byte_cap']+1024**2
    write(manifest, dict(protocol_sha256=pin, parent_protocol_sha256=PARENT_PIN,
        parent_required_sha256=p['parent_required_sha256'],
        files={f.relative_to(root).as_posix():sha(f) for f in files},
        uncompressed_bytes=sum(f.stat().st_size for f in files), diagnostic_only=True,
        complete_diagnostic=(root/'outputs/path_ablation_v1/run_results.json').exists(),
        failure_present=(root/'outputs/path_ablation_v1/failure.json').exists(), optimizer_updates=0))
    archive = Path.home()/(STEM+'-results.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for f in files+[manifest]:
            assert time.monotonic()-start < 300, 'Export5-minute stop'
            tar.add(f, arcname=NAME+'/'+f.relative_to(root).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    write(Path.home()/(STEM+'-export.json'), dict(complete=True, archive_sha256=digest,
        bytes=archive.stat().st_size, seconds=time.monotonic()-start,
        optimizer_updates=0, diagnostic_success_not_implied=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-transfer', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--export', action='store_true')
    a = parser.parse_args(); root = a.root.resolve(); parent = a.parent_root.resolve()
    p, old = verify(root, parent, a.protocol_sha)
    if a.verify_transfer:
        if platform.system() == 'Linux':
            require_vm(root, parent)
            assert shutil.disk_usage(root).free >= p['minimum_free_bytes'], 'Need4GiB free before diagnostic logs'
        print(dict(complete=True, packet_assets=len(p['assets_sha256']),
            parent_assets=len(p['parent_required_sha256']), neural_calls=0, optimizer_updates=0))
    elif a.export:
        export(root, parent, p, a.protocol_sha)
    else:
        run(root, parent, p, old, a.protocol_sha)
