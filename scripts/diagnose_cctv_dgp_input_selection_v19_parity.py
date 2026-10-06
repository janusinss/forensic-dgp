"""One-time, three-case L4 normalization diagnostic; no training or V19 resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time

PIN = '2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2'
FAILURE_SHA = '5c7c6bea942a8ff6fc81f6a861be05f4f671e8d2a154685f987e22de5021e4de'
LOG_SHA = '22c42badaea6566e3b411a6399a653b4579ae2b535775eb705b46184739c142d'
PARITY_SHA = 'a792935127b6e5e389a44bc5ba82dd86bce9a22deeb4c5248a1b3507318c5840'
NAME = 'cctv_dgp_input_selection_v19_parity_diagnostic_r1'
SESSION = 'dgp_v19_parity_diagnostic_r1'
ARCHIVE = 'cctv-dgp-input-selection-v19-parity-diagnostic-r1.tar.gz'
SCRIPT = 'diagnose_cctv_dgp_input_selection_v19_parity.py'
DESIGN = {'training': False, 'optimizer_updates': 0, 'backward_calls': 0,
          'cases': 3, 'DGP_forwards': 8, 'other_network_forwards': 0,
          'worker_cap_seconds': 120, 'supervisor_cap_seconds': 240,
          'export_cap_seconds': 60, 'peak_vram_cap_bytes': 20 * 1024**3,
          'minimum_free_disk_bytes': 256 * 1024**2,
          'checkpoint_search': False, 'threshold_refit': False,
          'production_promoted': False, 'native_used': False,
          'reserved_used': False, 'V19_resumed': False,
          'scope': 'Failed first development case, first fixed clear preview, first training parity case'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def safe(root, name):
    part = PurePosixPath(name)
    require(bool(name) and not part.is_absolute() and '..' not in part.parts
            and ':' not in name and '\\' not in name and str(part) == name, 'Unsafe diagnostic path')
    target = (Path(root) / name).resolve()
    require(target.is_relative_to(Path(root).resolve()), 'Diagnostic path escapes root')
    return target


def original_binding(original):
    require(sha(original / 'input_selection_protocol_v19.json') == PIN
            and sha(original / 'supervisor_failure.json') == FAILURE_SHA
            and sha(original / 'inference.log') == LOG_SHA
            and sha(original / 'outputs/input_selection_v19/fresh_training_parity.json') == PARITY_SHA,
            'Exact original V19 failure differs; do not run this diagnostic')


def live_tmux_tasks(output):
    allowed = {'bash', 'sh', 'zsh', 'fish', 'tmux', 'tail', 'less', 'watch', 'htop', 'top'}
    tasks = []
    for line in output.splitlines():
        fields = line.split('\t')
        require(len(fields) == 3, 'Unknown tmux process state')
        session, command, dead = fields
        if dead != '1' and command not in allowed:
            tasks.append({'session': session, 'command': command})
    return tasks


def classify_normalization(rows, scalar, stability):
    """Confirmation requires exact baseline restoration, not a relaxed PNG gate."""
    byrole = {r['role']: r for r in rows}
    require(len(rows) == 3 and set(byrole) ==
            {'failed_development_case', 'fixed_clear_preview', 'training_parity_case'}, 'Three fixed roles required')
    failed = byrole['failed_development_case']['routes']
    preview = byrole['fixed_clear_preview']['routes']
    training = byrole['training_parity_case']['routes']
    return (scalar['CUDA_matches_reciprocal_simulation'] is True
            and scalar['different_values_from_CPU_division'] > 0
            and stability == {'numpy_before_GPU': True, 'CUDA_scalar_division': True}
            and failed['numpy_before_GPU']['reference_PNG_equal'] is True
            and failed['CUDA_scalar_division']['reference_PNG_equal'] is False
            and preview['numpy_before_GPU']['reference_PNG_equal'] is True
            and preview['numpy_before_GPU']['reference_raw_max_difference'] is not None
            and preview['numpy_before_GPU']['reference_raw_max_difference'] <= 2e-6
            and training['CUDA_scalar_division']['reference_PNG_equal'] is True
            and training['CUDA_scalar_division']['reference_raw_max_difference'] <= 2e-6)


def launch(expected_sha):
    require(sys.platform == 'linux' and os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis',
            'Launch only on the existing forensic-dgp-thesis Linux VM')
    require(sha(Path(__file__)) == expected_sha, 'Diagnostic executable checksum differs')
    upload = Path(__file__).resolve()
    require(Path(str(upload) + '.sha256').read_text().split() == [expected_sha, SCRIPT],
            'Uploaded checksum differs')
    repo = (Path.home() / 'forensic-dgp').resolve()
    original = repo / 'cctv_dgp_input_selection_vm_v19'
    root = repo / NAME
    original_binding(original)
    require(not root.exists(), 'Preserve existing diagnostic root; no repeat/resume')
    require(shutil.disk_usage(repo).free >= DESIGN['minimum_free_disk_bytes'], 'Need256MiB free disk')
    python = repo / 'cctv_dgp_vm_bundle/.venv/bin/python'
    require(python.is_file(), 'Existing runtime missing; do not reinstall')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True, timeout=20)
    panes = subprocess.run(['tmux', 'list-panes', '-a', '-F', '#{session_name}\t#{pane_current_command}\t#{pane_dead}'],
                           capture_output=True, text=True, timeout=20)
    require(panes.returncode in [0, 1], 'Cannot verify tmux state')
    tasks = live_tmux_tasks(panes.stdout) if panes.returncode == 0 else []
    require(not apps.stdout.strip() and not tasks, 'Competing GPU/task; preserve it:' + repr(tasks))
    exists = subprocess.run(['tmux', 'has-session', '-t', SESSION], capture_output=True, timeout=20)
    require(exists.returncode == 1, 'Preserve existing diagnostic session')
    root.mkdir()
    shutil.copyfile(upload, root / SCRIPT)
    write(root / 'diagnostic_protocol.json', {'design': DESIGN, 'script_sha256': expected_sha,
        'original_protocol_sha256': PIN, 'original_failure_sha256': FAILURE_SHA,
        'original_log_sha256': LOG_SHA, 'original_training_parity_sha256': PARITY_SHA})
    args = [str(python), '-B', '-u', str(root / SCRIPT), '--supervise', '--root', str(root),
            '--expected-sha', expected_sha]
    shell = 'set -C; exec ' + shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1'] + args)
    shell += ' > ' + shlex.quote(str(root / 'supervisor.log')) + ' 2>&1'
    write(root / 'launch.json', {'complete': True, 'session': SESSION, 'script_sha256': expected_sha,
                               'root': str(root), 'V19_resumed': False, 'training': False})
    subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, '-c', str(root), shell], check=True, timeout=20)
    print(json.dumps({'tmux_launched': True, 'session': SESSION, 'root': str(root), 'budget_seconds': 240}), flush=True)


def worker(root, expected_sha):
    started = time.monotonic()
    repo = root.parent
    parent = repo / 'cctv_dgp_face_code_fit_vm_v12_r2'
    original = repo / 'cctv_dgp_input_selection_vm_v19'
    baseline = repo / 'cctv_dgp_generalization_vm_v15/outputs/generalization_v15'
    mixed = repo / 'cctv_dgp_mixed_vm_v9_r2'
    original_binding(original)
    require(sha(Path(__file__)) == expected_sha, 'Worker executable differs')
    protocol = read(root / 'diagnostic_protocol.json')
    require(protocol['design'] == DESIGN and protocol['script_sha256'] == expected_sha
            and protocol['original_protocol_sha256'] == PIN, 'Diagnostic finite design differs')
    sys.path.insert(0, str(parent))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)
    sys.path.insert(0, str(original))
    import cctv_dgp_input_selection_v19 as q
    p, v, _ = q.verify(original, parent, repo / 'cctv_dgp_broader_codes_vm_v16_r2', mixed, baseline, PIN)
    import numpy as np
    from PIL import Image
    import torch, torchvision
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_pilot import state_hash
    require((repo / 'cctv_dgp_vm_bundle/cuda_runtime_before.txt').read_text().splitlines()
            == [torch.__version__, torchvision.__version__], 'Existing runtime differs; no reinstall')
    torch.set_num_threads(4)
    torch.manual_seed(20261005)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def clock():
        torch.cuda.synchronize()
        require(time.monotonic() - started <= 120, 'Diagnostic worker exceeds120s')
        require(torch.cuda.max_memory_allocated() <= DESIGN['peak_vram_cap_bytes'], 'VRAM exceeds20GiB')
    clock()
    pp = v.read(parent / 'face_code_fit_protocol_v12.json')
    dgp, provenance = load_frozen_dgp_restorer(parent / pp['weights']['dgp'],
        expected_sha256=pp['assets_sha256'][pp['weights']['dgp']], device='cuda')
    dgp.net.eval().requires_grad_(False)
    before = state_hash(dgp.net)
    require(before == read(original / 'outputs/input_selection_v19/execution.json')['states_before']['dgp'],
            'Frozen DGP state differs')
    protected_names = list(p['assets_sha256']) + ['input_selection_protocol_v19.json',
        'supervisor_failure.json', 'inference.log', 'outputs/input_selection_v19/fresh_training_parity.json']
    protected_before = {name: sha(original / name) for name in protected_names}
    counter = {'DGP': 0}
    def count(_module, _args, _output):
        counter['DGP'] += 1
    handle = dgp.net.register_forward_hook(count)
    b = v.read(baseline / 'results.json')
    base_rows = {r['id']: r for r in b['rows']}
    refmap = {r['id']: r for r in p['references'] + p['training_parity_references']}
    preview = next(c for c in p['cases'] if c['reference_id'] == p['preview_reference_ids'][0] and c['profile'] == 'clear')
    cases = [('failed_development_case', p['cases'][0]), ('fixed_clear_preview', preview),
             ('training_parity_case', p['training_parity_cases'][0])]
    require(len({c['id'] for _, c in cases}) == 3, 'Three distinct cases required')
    artifacts = {}
    def save(name, value):
        path = safe(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.npy'):
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
        else:
            require(not path.exists(), 'No diagnostic PNG overwrite')
            Image.fromarray(value).save(path)
        artifacts[name] = sha(path)
        return name
    rows, stability = [], {}
    with torch.inference_mode():
        codes = np.arange(256, dtype=np.uint8).astype(np.float32)
        cuda_codes = (torch.arange(256, device='cuda', dtype=torch.float32) / 255).cpu().numpy().copy()
        divided = codes / 255
        reciprocal = codes * np.float32(1 / 255)
        scalar = {'CUDA_matches_reciprocal_simulation': bool(np.array_equal(cuda_codes, reciprocal)),
            'different_values_from_CPU_division': int(np.count_nonzero(cuda_codes != divided)),
            'maximum_difference_from_CPU_division': float(np.max(np.abs(cuda_codes - divided))),
            'CUDA_values': save('scalar/CUDA_values.npy', cuda_codes),
            'CPU_values': save('scalar/CPU_values.npy', divided)}
        for role, c in cases:
            clock()
            cid = c['id']; ref = refmap[c['reference_id']]
            camera = v.rgb(mixed / c['input'])
            with Image.open(mixed / ref['observed']) as im:
                support = np.asarray(im).copy() > 0
            if role == 'training_parity_case':
                reference_raw_path = original / ('parity/dgp_base/' + cid + '.npy')
                reference_raw = np.load(reference_raw_path, allow_pickle=False)
                reference_png = v.png(reference_raw, camera, support)
            else:
                reference = base_rows[cid]['arms']['retained_dgp_v2']
                reference_png = v.rgb(baseline / reference['prediction'])
                reference_raw_path = baseline / reference['raw'] if 'raw' in reference else None
                reference_raw = np.load(reference_raw_path, allow_pickle=False) if reference_raw_path else None
            routes, input_arrays, raw_arrays = {}, {}, {}
            for route in ['numpy_before_GPU', 'CUDA_scalar_division']:
                if route == 'numpy_before_GPU':
                    x = torch.from_numpy(camera.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()
                else:
                    x = torch.from_numpy(camera.copy()).permute(2, 0, 1).float()[None].cuda() / 255
                normalized = x[0].permute(1, 2, 0).cpu().numpy().copy()
                result = dgp.net(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy().copy()
                require(result.dtype == np.float32 and result.shape == (256, 256, 3)
                        and np.isfinite(result).all() and 0 <= result.min() <= result.max() <= 1,
                        'Invalid diagnostic DGP raw output')
                png = v.png(result, camera, support)
                delta = np.abs(png.astype(np.int16) - reference_png.astype(np.int16))
                routes[route] = {'input': save('inputs/' + route + '/' + cid + '.npy', normalized),
                    'raw': save('raw/' + route + '/' + cid + '.npy', result),
                    'PNG': save('predictions/' + route + '/' + cid + '.png', png),
                    'tensor_stride': list(x.stride()), 'tensor_dtype': str(x.dtype),
                    'reference_PNG_equal': bool(np.array_equal(png, reference_png)),
                    'reference_PNG_differing_channels': int(np.count_nonzero(delta)),
                    'reference_PNG_max_abs_channel_difference': int(delta.max()),
                    'reference_raw_max_difference': float(np.max(np.abs(result - reference_raw))) if reference_raw is not None else None}
                input_arrays[route] = normalized
                raw_arrays[route] = result
                if role == 'failed_development_case':
                    repeat = dgp.net(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy().copy()
                    stability[route] = bool(np.array_equal(result, repeat))
                    save('repeat_raw/' + route + '/' + cid + '.npy', repeat)
                clock()
            rows.append({'id': cid, 'role': role, 'routes': routes,
                'reference_PNG_sha256': hashlib.sha256(reference_png.tobytes()).hexdigest(),
                'reference_raw_sha256': sha(reference_raw_path) if reference_raw_path else None,
                'camera_sha256': sha(mixed / c['input']), 'support_sha256': sha(mixed / ref['observed']),
                'maximum_normalized_input_difference': float(np.max(np.abs(input_arrays['numpy_before_GPU'] - input_arrays['CUDA_scalar_division']))),
                'normalized_input_different_values': int(np.count_nonzero(input_arrays['numpy_before_GPU'] != input_arrays['CUDA_scalar_division'])),
                'maximum_DGP_raw_difference_between_routes': float(np.max(np.abs(raw_arrays['numpy_before_GPU'] - raw_arrays['CUDA_scalar_division'])))})
            print({'diagnosed': cid, 'role': role, 'routes': {k: r['reference_PNG_equal'] for k, r in routes.items()}}, flush=True)
    handle.remove()
    after = state_hash(dgp.net)
    protected_after = {name: sha(original / name) for name in protected_names}
    require(before == after and protected_before == protected_after and counter == {'DGP': 8},
            'State/source/forward scope differs')
    require(not dgp.net.training and all(not p.requires_grad and p.grad is None for p in dgp.net.parameters()),
            'Frozen inference/gradient invariant differs')
    confirmed = classify_normalization(rows, scalar, stability)
    clock()
    write(root / 'results.json', {'complete': True, 'design': DESIGN, 'script_sha256': expected_sha,
        'original_protocol_sha256': PIN, 'scalar_normalization': scalar, 'rows': rows,
        'repeat_stability': stability, 'normalization_hypothesis_confirmed': confirmed,
        'DGP_state_before': before, 'DGP_state_after': after, 'forward_counts': counter,
        'protected_original_fingerprints_before': protected_before,
        'protected_original_fingerprints_after': protected_after,
        'artifacts_sha256': artifacts, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'gpu': torch.cuda.get_device_name(0), 'provenance': provenance,
        'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(),
        'optimizer_constructed': False, 'optimizer_updates': 0, 'backward_calls': 0,
        'seconds': time.monotonic() - started,
        'interpretation': 'Diagnostic evidence only; no model quality, relaxed gate, recovery launch, V19 resume or production promotion'})
    print({'complete': True, 'normalization_hypothesis_confirmed': confirmed, 'DGP_forwards': 8}, flush=True)


def supervise(root, expected_sha):
    require(root.resolve() == (Path.home() / 'forensic-dgp' / NAME).resolve(), 'Exact diagnostic root required')
    require(sha(Path(__file__)) == expected_sha, 'Supervisor executable differs')
    require(not (root / 'supervision.json').exists(), 'No diagnostic repeat/resume')
    started = time.monotonic(); deadline = started + 240
    write(root / 'supervision.json', {'complete': True, 'started_unix': time.time(), 'budget_seconds': 240})
    error_text = None
    args = [sys.executable, '-B', '-u', str(root / SCRIPT), '--worker', '--root', str(root), '--expected-sha', expected_sha]
    try:
        with (root / 'worker.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(args, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                raise TimeoutError('Finite120s diagnostic worker stop')
        if code:
            raise subprocess.CalledProcessError(code, args)
    except BaseException as error:
        error_text = str(error)
        write(root / 'worker_failure.json', {'complete': False, 'error_type': type(error).__name__,
                                           'error': error_text, 'resume_permitted': False})
    print((root / 'worker.log').read_text()[-3000:], flush=True)
    archive = root / ARCHIVE
    export_deadline = min(deadline, time.monotonic() + 60)
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob('*')
                   if path.is_file() and path.name != 'supervisor.log')
    require(sum((root / name).stat().st_size for name in names) <= 32 * 1024**2, 'Diagnostic export exceeds32MiB')
    with tarfile.open(archive, 'x:gz', compresslevel=3) as stream:
        for name in names:
            require(time.monotonic() < export_deadline, 'Finite60s export stop; preserve partial archive')
            stream.add(safe(root, name), arcname=name, recursive=False)
    digest = sha(archive)
    require(time.monotonic() < export_deadline, 'Finite export/hash stop')
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + ARCHIVE + '\n')
    write(root / 'export.json', {'complete': True, 'archive_sha256': digest, 'bytes': archive.stat().st_size,
        'script_sha256': expected_sha, 'original_protocol_sha256': PIN, 'diagnostic_succeeded': error_text is None,
        'results_sha256': sha(root / 'results.json') if (root / 'results.json').is_file() else None,
        'seconds': time.monotonic() - started, 'training': False, 'V19_resumed': False})
    print((root / 'export.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ['launch', 'supervise', 'worker']:
        mode.add_argument('--' + name, action='store_true')
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if args.launch:
        launch(args.expected_sha)
    else:
        require(args.root is not None and args.root.resolve() == (Path.home() / 'forensic-dgp' / NAME).resolve(),
                'Exact diagnostic root required')
        if args.supervise:
            supervise(args.root.resolve(), args.expected_sha)
        else:
            worker(args.root.resolve(), args.expected_sha)
