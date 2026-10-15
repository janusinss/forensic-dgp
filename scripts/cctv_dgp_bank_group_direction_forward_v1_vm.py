"""Manual L4 finite forward proposals from saved derivatives; no autograd/optimizer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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

NAME = 'cctv_dgp_bank_group_direction_forward_v1_vm'
STEM = 'cctv-dgp-bank-group-direction-forward-v1'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
PARENT_PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def fixed_reference_matrices(cases, references):
    """Use frozen target->112 affines; five landmarks are not affine coefficients."""
    matrices = []
    for case in cases:
        matrix = references[case['source_person_or_reference']]['matrix112']
        if len(matrix) != 2 or any(len(row) != 3 for row in matrix):
            raise ValueError('Fixed target affine must have shape2x3')
        if not all(math.isfinite(float(value)) for row in matrix for value in row):
            raise ValueError('Nonfinite fixed target affine')
        determinant = float(matrix[0][0])*float(matrix[1][1])-float(matrix[0][1])*float(matrix[1][0])
        if abs(determinant) < 1e-8:
            raise ValueError('Singular fixed target affine')
        matrices.append(matrix)
    return matrices


def verify(root, parent, pin):
    assert sha(root/'protocol.json') == pin and sha(parent/'protocol.json') == PARENT_PIN
    p, old = read(root/'protocol.json'), read(parent/'protocol.json')
    assert p['name'] == NAME and p['parent_protocol_sha256'] == PARENT_PIN
    assert p['case_ids'] == old['preview_case_ids'] and len(p['case_ids']) == 100
    assert p['optimizer_updates'] == p['gradient_queries'] == p['backwards'] == p['full_epochs'] == 0
    assert p['maximum_temporary_parameter_proposals'] == 4
    assert p['proposals'] == [{'label': 'balanced_'+rate, 'direction': 'balanced', 'rate': float(rate)}
        for rate in ['0.005', '0.01', '0.02', '0.04']]
    assert p['model_forward_cap'] == 100 and p['recognizer_forward_cap'] == 160
    assert p['preserved_scientific_thresholds'] == old['scientific_thresholds']
    assert p['minimum_free_bytes'] == 4*1024**3 and p['disk_reserve_bytes'] == 1024**3
    assert p['output_byte_cap'] == 768*1024**2 and p['peak_VRAM_bytes'] == 20*1024**3
    assert p['worker_seconds'] == 1200 and p['external_seconds'] == 1230
    assert p['peak_RSS_bytes'] == 12*1024**3
    assert p['gradient_groups_used_in_direction'] == 32 and p['proposal_derivation_TRAIN_references'] == 20
    assert p['measurement_seconds'] == 900 and p['export_seconds'] == 300
    assert not p['failed_run_resume_allowed'] and not p['app_promotion'] and not p['automatic_follow_on']
    for folder, bindings in [(root, p['assets_sha256']), (parent, p['parent_required_sha256'])]:
        for name, digest in bindings.items():
            file = folder/name
            assert file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(folder)
            assert sha(file) == digest, name
    direction = read(root/'direction_review.json')
    assert direction['complete'] and direction['direction_file_sha256'] == sha(root/'directions.npz')
    assert direction['names'] == p['learning_names'] and direction['initial_file_sha256'] == sha(parent/'initializers/initial_B.pth')
    assert direction['no_local_neural_calls_gradients_optimizer_or_parameter_displacement']
    assert direction['rate_candidates'] == [.005, .01, .02, .04]
    assert direction['maximum_prospective_displacement_relative_to_parameter_L2'] < .001
    assert p['native_pixels_used'] == p['final_identity_pixels'] == 0
    refs = {reference['id']: reference for reference in old['references']}
    cases = [case for case in old['cases'] if case['id'] in p['case_ids']]
    assert len(cases) == 100
    assert len(fixed_reference_matrices(cases, refs)) == 100
    return p, old


def require_vm(root, parent):
    assert platform.system() == 'Linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root == (Path.home()/'forensic-dgp'/NAME).resolve()
    assert parent == (Path.home()/'forensic-dgp'/PARENT_NAME).resolve()
    assert os.environ.get('TMUX'), 'Human manual tmux is required'
    assert sys.prefix == str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
                                    headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=3) as response:
        assert response.read().decode().rstrip().endswith('/g2-standard-4')
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=10)
    assert not result.stdout.strip(), 'Competing GPU workload; preserve it'


def run(root, parent, p, old, pin):
    require_vm(root, parent)
    assert not (root/'outputs').exists(), 'Immutable used forward diagnostic; no unchanged rerun'
    available = shutil.disk_usage(root).free
    if available < p['minimum_free_bytes']:
        early = root/'outputs/group_direction_forward_v1'
        early.mkdir(parents=True)
        write(early/'failure.json', dict(stage='pre-model storage preflight',
            available_free_bytes=available, required_free_bytes=p['minimum_free_bytes'],
            neural_calls=0, gradient_queries=0, optimizer_updates=0, parameter_displacements=0))
        raise AssertionError('Need4GiB free after install; structured storage stop retained')
    sys.path.insert(0, str(parent))
    import numpy as np
    from PIL import Image
    import torch
    import resource
    from cctv_dgp_bank_comparison_v1_model import ConditionedBankDGP
    from cctv_dgp_generative_bank_v1 import load_prior
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_raw_metrics import deliver
    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L4'
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261010)
    torch.cuda.manual_seed_all(20261010)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    out = root/'outputs/group_direction_forward_v1'
    out.mkdir(parents=True)
    counts = dict(B=0, recognizer=0, original=0, gradient_queries=0, backwards=0,
                  optimizer_updates=0, temporary_parameter_proposals=0)
    progress = dict(stage='initialize', counts=counts)
    candidate = None
    initial = None
    def stop(_signal, _frame):
        raise TimeoutError('External finite forward stop; preserve evidence')
    signal.signal(signal.SIGTERM, stop)
    def clock():
        torch.cuda.synchronize()
        assert not torch.is_grad_enabled()
        assert time.monotonic()-start < p['worker_seconds'], 'Worker stop'
        assert shutil.disk_usage(root).free >= p['disk_reserve_bytes'], 'Disk reserve stop'
        assert torch.cuda.max_memory_allocated() <= p['peak_VRAM_bytes'], 'VRAM stop'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= p['peak_RSS_bytes'], 'RSS stop'
        assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) <= p['output_byte_cap'], 'Output stop'
        assert counts['B'] <= p['model_forward_cap'] and counts['recognizer'] <= p['recognizer_forward_cap']
    def pixels(path, mode='RGB'):
        with Image.open(path) as value:
            assert value.size == (256, 256)
            return np.asarray(value.convert(mode)).copy()
    def tensor(a):
        return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2).contiguous().cuda()
    try:
        original, _ = load_frozen_dgp_restorer(parent/'weights/dgp_v2.pth',
            expected_sha256=old['original_checkpoint_sha256'], device='cuda')
        identity = FixedObservedIdentity(parent/'weights/w600k_r50.onnx', 'cuda')
        prior = load_prior(parent/'prior_source', parent/'bank_implementation', old['standalone_prior_sha256']).cuda()
        candidate = ConditionedBankDGP(original.net, prior,
            torch.from_numpy(np.load(parent/'initializers/mean_style.npy', allow_pickle=False)).cuda()).cuda()
        initial = torch.load(parent/'initializers/initial_B.pth', map_location='cpu', weights_only=True)
        candidate.load_state_dict(initial, strict=True)
        candidate.eval().requires_grad_(False)
        assert state_hash(candidate.state_dict()) == p['initial_model_state']
        names = p['learning_names']
        assert candidate.learning_names() == names and len(names) == 37
        params = dict(candidate.named_parameters())
        frozen_before = state_hash({n:v for n,v in candidate.state_dict().items() if n not in names})
        original_before = state_hash(original.net.state_dict())
        identity_before = state_hash(identity.state_dict())
        with np.load(root/'directions.npz', allow_pickle=False) as archive:
            assert set(archive.files) == {f'{label}_p{i:02d}' for label in ['balanced'] for i in range(37)}
            directions = {label: [archive[f'{label}_p{i:02d}'] for i in range(37)] for label in ['balanced']}
        assert all(a.dtype == np.float32 and a.shape == tuple(initial[names[i]].shape) and np.isfinite(a).all()
                   for values in directions.values() for i,a in enumerate(values))
        begins = [i for i in range(0, len(old['cases']), 5) if old['cases'][i]['id'] in p['case_ids']]
        assert len(begins) == 20
        refs = {r['id']:r for r in old['references']}
        blocks = []
        for begin in begins:
            cs = old['cases'][begin:begin+5]
            ids = [c['id'] for c in cs]
            ref = refs[cs[0]['source_person_or_reference']]
            x8 = np.stack([pixels(parent/c['input']) for c in cs])
            target8 = np.stack([pixels(parent/ref['target'])]*5)
            observed8 = np.stack([pixels(parent/ref['observed'], 'L')]*5)
            baseline, embeddings = unpack(parent/f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz', ids)
            blocks.append((begin, cs, x8, target8, observed8, baseline, embeddings))
        progress['stage'] = 'exact_initial_controls'
        controls = []
        for begin, cs, x8, target8, observed8, baseline, embeddings in blocks:
            clock()
            x = tensor(x8)
            mask = tensor(observed8[..., None])
            result = torch.where(mask.bool(), candidate(x), x)
            counts['B'] += 1
            raw = result.permute(0, 2, 3, 1).cpu().numpy().copy()
            error = float(np.abs(raw.astype(np.float64)-baseline).max())
            record = dict(begin=begin, case_ids=[c['id'] for c in cs], exact_raw_replay=np.array_equal(raw, baseline),
                maximum_absolute_error=error, changed_values=int(np.count_nonzero(raw != baseline)),
                actual_RGB_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                expected_RGB_float32_sha256=hashlib.sha256(baseline.tobytes()).hexdigest())
            write(out/f'control_b{begin//5:03d}.json', record)
            assert record['exact_raw_replay'], 'Initial raw differs; retain stop'
            controls.append(record)
            del result, x, mask
        write(out/'environment.json', dict(python=sys.version, torch=torch.__version__, CUDA=torch.version.cuda,
            GPU=torch.cuda.get_device_name(0), host=platform.node(), manual_tmux=True,
            optimizer_constructed=False, gradient_enabled=False, inference_of_disposable_parameter_proposals=True))
        records = []
        for proposal in p['proposals']:
            clock()
            candidate.load_state_dict(initial, strict=True)
            rate = np.float32(proposal['rate'])
            updates = {}
            for i, name in enumerate(names):
                # NumPy float32 arithmetic is independently reproducible on return.
                value = initial[name].numpy()+rate*directions[proposal['direction']][i]
                params[name].copy_(torch.from_numpy(value).to(params[name].device))
                updates[name] = sha_array(value)
            assert state_hash({n:v for n,v in candidate.state_dict().items() if n not in names}) == frozen_before
            proposal_state = state_hash(candidate.state_dict())
            assert proposal_state != p['initial_model_state']
            counts['temporary_parameter_proposals'] += 1
            folder = out/proposal['label']
            folder.mkdir()
            (folder/'png').mkdir()
            progress.update(stage='temporary_forward', proposal=proposal['label'])
            write(folder/'parameter_receipt.json', dict(label=proposal['label'], proposal=proposal,
                model_state=proposal_state, selected_parameter_value_sha256=updates,
                original_frozen_tensors_unchanged=True, optimizer_updates=0, checkpoint_saved=False))
            for serial, (begin, cs, x8, target8, observed8, baseline, expected_emb) in enumerate(blocks):
                clock()
                x, target, mask = tensor(x8), tensor(target8), tensor(observed8[..., None])
                pred = torch.where(mask.bool(), candidate(x), x)
                counts['B'] += 1
                raw = pred.permute(0, 2, 3, 1).cpu().numpy().copy()
                assert raw.shape == (5, 256, 256, 3) and np.isfinite(raw).all()
                observed = observed8 > 0
                assert np.array_equal(raw[~observed], (x8.astype(np.float32)/np.float32(255))[~observed])
                stem = f'b{begin//5:03d}'
                np.save(folder/(stem+'.npy'), raw, allow_pickle=False)
                png = np.stack([deliver(a, camera, area) for a, camera, area in zip(raw, x8, observed)])
                for c, image in zip(cs, png):
                    Image.fromarray(image).save(folder/'png'/(c['id']+'.png'))
                grids = torch.from_numpy(np.stack([grid112(matrix) for matrix in fixed_reference_matrices(cs, refs)])).cuda()
                base = torch.from_numpy(baseline).permute(0, 3, 1, 2).contiguous().cuda()
                base_png = tensor(np.stack([deliver(a, camera, area) for a, camera, area in zip(baseline, x8, observed)]))
                vectors_raw = identity.embedding(torch.cat([pred, base, target]), torch.cat([mask]*3), torch.cat([grids]*3))
                counts['recognizer'] += 1
                vectors_png = identity.embedding(torch.cat([tensor(png), base_png, target]), torch.cat([mask]*3), torch.cat([grids]*3))
                counts['recognizer'] += 1
                raw_emb, png_emb = vectors_raw.cpu().numpy().copy(), vectors_png.cpu().numpy().copy()
                assert np.allclose(raw_emb[5:10], expected_emb[:, 0], rtol=1e-6, atol=1e-6)
                assert np.allclose(png_emb[5:10], expected_emb[:, 1], rtol=1e-6, atol=1e-6)
                assert np.allclose(raw_emb[10:15], expected_emb[:, 2], rtol=1e-6, atol=1e-6)
                assert np.allclose(png_emb[10:15], expected_emb[:, 2], rtol=1e-6, atol=1e-6)
                np.savez_compressed(folder/(stem+'_embeddings.npz'), raw=raw_emb, png=png_emb)
                entry = dict(proposal=proposal['label'], begin=begin, case_ids=[c['id'] for c in cs],
                    raw_file=str((folder/(stem+'.npy')).relative_to(out)), raw_sha256=sha(folder/(stem+'.npy')),
                    embeddings_file=str((folder/(stem+'_embeddings.npz')).relative_to(out)),
                    embeddings_sha256=sha(folder/(stem+'_embeddings.npz')), maximum_target_embedding_error=float(max(
                        abs(raw_emb[10:15]-expected_emb[:, 2]).max(), abs(png_emb[10:15]-expected_emb[:, 2]).max())),
                    state=proposal_state, seconds=time.monotonic()-start)
                records.append(entry)
                print(dict(proposal=proposal['label'], reference_batches=serial+1, of=20, seconds=entry['seconds']), flush=True)
                if len(records) == 2:
                    projection = (time.monotonic()-start)/counts['B']*p['model_forward_cap']*1.5
                    write(out/'timing_projection.json', dict(projection_seconds=projection, cap_seconds=p['measurement_seconds'],
                        measured_B_batches=counts['B'], candidate_reference_batches=2, safety_factor=1.5))
                    assert projection < p['measurement_seconds'], 'Timing projection stop'
                assert time.monotonic()-start < p['measurement_seconds'], 'Measurement stop'
                del x, target, mask, pred, base, base_png, vectors_raw, vectors_png
            assert state_hash(candidate.state_dict()) == proposal_state
        candidate.load_state_dict(initial, strict=True)
        assert state_hash(candidate.state_dict()) == p['initial_model_state']
        assert state_hash(original.net.state_dict()) == original_before and state_hash(identity.state_dict()) == identity_before
        assert counts == dict(B=100, recognizer=160, original=0, gradient_queries=0, backwards=0,
                              optimizer_updates=0, temporary_parameter_proposals=4)
        assert all(not a.requires_grad and a.grad is None for net in [candidate, original.net, identity] for a in net.parameters())
        clock()
        verify(root, parent, pin)
        write(out/'run_results.json', dict(complete=True, counts=counts, controls=controls, records=records,
            seconds=time.monotonic()-start, peak_VRAM=torch.cuda.max_memory_allocated(),
            peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            candidate_initial_state_restored=True, original_and_recognizer_unchanged=True,
            four_temporary_model_displacements_are_not_optimizer_updates=True, checkpoint_created=False,
            native_pixels_used=0, final_identity_pixels=0, raw_PNG_independent_metrics_and_visual_review_pending=True,
            more_epochs_not_qualified=True, model_qualified=False))
    except BaseException as error:
        restore_error = None
        if candidate is not None and initial is not None:
            try:
                candidate.load_state_dict(initial, strict=True)
            except BaseException as failure:
                restore_error = repr(failure)
        write(out/'failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            progress=progress, seconds=time.monotonic()-start, disposable_state_restore_error=restore_error,
            more_epochs_not_qualified=True))
        raise


def sha_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def export(root, parent, p, pin):
    stamp = time.monotonic()
    assert platform.system() == 'Linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert root == (Path.home()/'forensic-dgp'/NAME).resolve() and os.environ.get('TMUX')
    manifest = root/'export_manifest.json'
    assert not manifest.exists()
    files = sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    assert not any(f.suffix == '.pth' for f in files)
    assert sum(f.stat().st_size for f in files) < p['output_byte_cap']+64*1024**2
    write(manifest, dict(protocol_sha256=pin, parent_protocol_sha256=PARENT_PIN,
        parent_required_sha256=p['parent_required_sha256'], files={f.relative_to(root).as_posix():sha(f) for f in files},
        optimizer_updates=0, complete_diagnostic=(root/'outputs/group_direction_forward_v1/run_results.json').exists(),
        failure_present=(root/'outputs/group_direction_forward_v1/failure.json').exists()))
    archive = Path.home()/(STEM+'-results.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for file in files+[manifest]:
            assert time.monotonic()-stamp < p['export_seconds'], 'Finite export stop'
            tar.add(file, arcname=NAME+'/'+file.relative_to(root).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive)+'.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    record = dict(complete=True, archive_sha256=digest, bytes=archive.stat().st_size,
        seconds=time.monotonic()-stamp, optimizer_updates=0, diagnostic_success_not_implied=True)
    write(Path.home()/(STEM+'-export.json'), record)
    print(record, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-transfer', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--export', action='store_true')
    a = parser.parse_args()
    root, parent = a.root.resolve(), a.parent_root.resolve()
    if a.run:
        require_vm(root, parent)
    p, old = verify(root, parent, a.protocol_sha)
    if a.verify_transfer:
        print(dict(complete=True, packet_assets=len(p['assets_sha256']), parent_assets=len(p['parent_required_sha256']),
                   neural_calls=0, gradient_queries=0, optimizer_updates=0, fixed_target_affines_checked=100,
                   available_free_bytes=shutil.disk_usage(root).free, required_free_bytes=p['minimum_free_bytes']), flush=True)
    elif a.run:
        run(root, parent, p, old, a.protocol_sha)
    else:
        export(root, parent, p, a.protocol_sha)


if __name__ == '__main__':
    main()
