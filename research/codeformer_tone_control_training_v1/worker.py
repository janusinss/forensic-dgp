"""Finite manual L4 training of the input-only controller; static import never trains."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import shlex
import subprocess
import sys
import time
import traceback

from . import contract as c


def checkpoint_safe(completed, pending, inflight, state_valid):
    return type(completed) is int and 0 <= completed <= 50 and not inflight and state_valid and 0 <= pending <= 4


def idle_conflicts(rows, own_pid, root, protocol_sha):
    """Classify full pid/ppid/argv inventory; admit only exact current launch ancestors."""
    from scripts.supervise_codeformer_tone_control_training_v1_vm import competing_commands
    table = {pid: (ppid, command) for pid, ppid, command in rows}
    c.require(own_pid in table and len(table) == len(rows), 'Unique current worker pid required')
    accepted = {own_pid}
    parent_id = table[own_pid][0]
    if parent_id in table:
        ppid, command = table[parent_id]; words = shlex.split(command)
        if words and Path(words[0]).name == 'timeout':
            expected_tail = [sys.executable, '-B', '-u', '-m', c.MODULE, '--root', str(root), '--protocol-sha', protocol_sha, '--run']
            c.require(words[1:4] == ['--signal=TERM', '--kill-after=60', '3000'] and words[4:] == expected_tail,
                      'Current timeout ancestor differs from exact supervised worker invocation')
            accepted.add(parent_id); parent_id = ppid
        if parent_id in table:
            ppid, command = table[parent_id]; words = shlex.split(command)
            script = str(root / 'scripts/supervise_codeformer_tone_control_training_v1_vm.py')
            if script in words:
                c.require(words == [sys.executable, '-B', '-u', script, '--root', str(root), '--protocol-sha', protocol_sha] or
                          words == [sys.executable, '-B', script, '--root', str(root), '--protocol-sha', protocol_sha] or
                          words == [sys.executable, script, '--root', str(root), '--protocol-sha', protocol_sha],
                          'Exact current ancestor supervisor identity differs')
                accepted.add(parent_id)
    conflicts = competing_commands([(pid, command) for pid, _, command in rows if pid not in accepted], own_pid)
    return conflicts, sorted(accepted)


def require_idle(root, protocol_sha):
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                          '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    c.require(not gpu.stdout.strip(), 'GPU occupied; preserve every existing workload')
    inventory = subprocess.run(['ps', '-eo', 'pid=,ppid=,args='], check=True, capture_output=True, text=True, timeout=20)
    rows = []
    for line in inventory.stdout.splitlines():
        fields = line.strip().split(None, 2)
        c.require(len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit(), 'Full process ancestry cannot be classified')
        rows.append((int(fields[0]), int(fields[1]), fields[2]))
    conflicts, accepted = idle_conflicts(rows, os.getpid(), root, protocol_sha)
    c.require(not conflicts, 'Competing training process; preserve it: ' + json.dumps(conflicts))
    return {'complete': True, 'GPU_processes': [], 'competing_commands': [],
            'exact_launch_ancestry_excluded': accepted, 'full_process_inventory': rows, 'other_processes_killed': False}


def finite_counts(counts, completed, *, terminal=False):
    c.require(set(counts) == set(c.COUNTS), 'Exact operation namespace required')
    c.require(all(type(counts[k]) is int and 0 <= counts[k] <= cap for k, cap in c.COUNTS.items()),
              'Declared operation cap exceeded')
    c.require(type(completed) is int and 0 <= completed <= 50 and
              completed <= counts['head_optimizer_step_attempts'] <= completed + 1,
              'Committed/attempted head updates inconsistent')
    c.require(counts['head_backward_calls'] <= counts['controller_TRAIN'] <= 200,
              'Backward/forward chronology inconsistent')
    if terminal:
        c.require(completed == 50 and counts == c.COUNTS, 'Full finite run must close every operation')


def delivered_png(raw, pixels, observed):
    import numpy as np
    c.require(raw.shape == pixels.shape == (256, 256, 3) and raw.dtype == np.float32 and
              pixels.dtype == np.uint8 and observed.shape == (256, 256) and observed.dtype == np.bool_ and
              observed.any() and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1,
              'Finite bounded RGB256/support required')
    value = np.floor(np.clip(raw, 0, 1) * np.float32(255)).astype(np.uint8)
    value[~observed] = pixels[~observed]
    return value


def pixel_metrics(raw, png, target, observed, *, evidence, ssim_fn=None):
    import numpy as np
    if evidence == 'native_unpaired':
        c.require(target is None, 'Native target forbidden')
        return {view: dict.fromkeys(('mae', 'mse', 'psnr', 'ssim', 'luminance_bias', 'coarse_rgb_mae'))
                for view in ('raw', 'native_png')}
    c.require(target is not None and observed.any(), 'Paired target required')
    result = {}
    for view, image in [('raw', raw), ('native_png', png.astype(np.float32) / np.float32(255))]:
        predicted = image.astype(np.float64)
        truth = target.astype(np.float64)
        diff = predicted[observed] - truth[observed]
        mse = float(np.square(diff).mean())
        row = {'mae': float(np.abs(diff).mean()), 'mse': mse,
               'psnr': None if mse == 0 else -10 * math.log10(mse),
               'ssim': None, 'luminance_bias': float(abs((diff @ np.asarray([.2126, .7152, .0722])).mean())),
               'coarse_rgb_mae': None}
        if observed.all():
            pool = lambda a: a.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3), dtype=np.float64)
            row['coarse_rgb_mae'] = float(np.abs(pool(predicted) - pool(truth)).mean())
            if ssim_fn is not None:
                row['ssim'] = float(ssim_fn(predicted, truth, data_range=1.0, channel_axis=2))
        result[view] = row
    return result


def preservation_guards(candidate, original):
    c.require(set(candidate) == set(original) and len(candidate) == 406, 'Full406 guard banks required')
    clear = [key for key, row in candidate.items() if row['evidence'] == 'paired_synthetic' and row['profile'] == 'clear']
    blur = [key for key, row in candidate.items() if row['source'] == 'FFHQ validation' and row['profile'] == 'blur']
    c.require(len(clear) == 61 and len(blur) == 53, 'Unchanged61/53 guard namespace required')
    reports = {}
    for label, cases, metrics in [('clear', clear, ['mae']), ('normal_blur', blur, ['luminance_bias', 'coarse_rgb_mae'])]:
        report = {'pass': True, 'cases': len(cases), 'mean_limit': .01, 'worst_limit': .03, 'views': {}}
        for view in ('raw', 'native_png'):
            report['views'][view] = {}
            for metric in metrics:
                delta = {key: candidate[key]['metrics'][view][metric] - original[key]['metrics'][view][metric]
                         for key in sorted(cases)}
                c.require(all(math.isfinite(v) for v in delta.values()), 'Finite paired guard metrics required')
                mean, worst = sum(delta.values()) / len(delta), max(delta.values())
                passed = mean <= .01 and worst <= .03
                report['views'][view][metric] = {'mean_delta': mean, 'worst_delta': worst,
                    'deltas': delta, 'cases_over_0_03': sum(v > .03 for v in delta.values()), 'pass': passed}
                report['pass'] = report['pass'] and passed
        reports[label] = report
    reports['pass'] = reports['clear']['pass'] and reports['normal_blur']['pass']
    return reports


def dark_retention(candidate, base):
    report = {'pass': True, 'profiles': {}, 'prospective_additional_gate': True}
    for profile in ('lowlight', 'compound'):
        keys = sorted(key for key, row in candidate.items() if row['source'] == 'FFHQ validation' and row['profile'] == profile)
        c.require(len(keys) == 53, 'Full53 dark profile required')
        means = {state: {metric: sum(rows[k]['metrics']['native_png'][metric] for k in keys) / 53
                        for metric in ('mae', 'ssim')} for state, rows in [('candidate', candidate), ('base', base)]}
        passed = means['candidate']['mae'] <= means['base']['mae'] and means['candidate']['ssim'] >= means['base']['ssim']
        report['profiles'][profile] = {**means, 'cases': keys, 'pass': passed}
        report['pass'] = report['pass'] and passed
    return report


@contextmanager
def worker_lock(root, protocol_sha):
    path = root / c.LOCK_NAME
    with path.open('xb') as stream:
        stream.write((json.dumps({'pid': os.getpid(), 'protocol_sha256': protocol_sha}) + '\n').encode())
    try:
        yield
    finally:
        value = c.read_json(path)
        c.require(value['pid'] == os.getpid() and value['protocol_sha256'] == protocol_sha, 'Worker lock ownership changed')
        path.unlink()


def run(root, protocol_sha):
    root = c.manual_root(root)  # Before every framework import or model construction.
    from . import transport, data, model
    verified = transport.verify_transfer(root, protocol_sha)
    p = verified['protocol']
    parent = c.resolve_parent(root)
    out = root / c.OUTPUTS
    c.require(not out.exists(), 'Retain prior evidence; automatic repeat/resume is forbidden')
    c.require(shutil.disk_usage(root).free >= c.LIMITS['initial_free_bytes_after_install'], '16GiB free required after installation')
    idle_receipt = require_idle(root, protocol_sha)
    import numpy as np
    from PIL import Image, __version__ as pillow_version
    # Verified frozen parent sources are appended, never substituted for this namespace.
    import research
    research.__path__.append(str(parent / 'research'))
    sys.path.append(str(parent))
    out.mkdir(parents=True)
    start = time.monotonic()
    counts = {key: 0 for key in c.COUNTS}
    completed, pending, inflight, state_valid = 0, 0, False, True
    net = original = head = optimizer = torch = None
    committed_rng = None
    stopped_signal = {'signal': None}
    saved_handlers = {}
    total_bytes, total_files, metadata_bytes = 0, 0, 0
    peak_rss = peak_cuda = 0
    minimum_free = shutil.disk_usage(root).free
    checkpoint_receipts = {}
    training_seconds = 0.0
    last_safe = None
    active_step_start = None
    base_digest = original_digest = None
    all_banks = {}
    static_bytes = sum(p['files_bytes'].values()) + (root / 'protocol.json').stat().st_size
    static_files = len(p['files_bytes']) + 1

    def external_accounting():
        files = list((root / 'outputs/supervision').rglob('*')) if (root / 'outputs/supervision').exists() else []
        sizes = []
        for path in files:
            c.no_links(path)
            if path.is_file(): sizes.append(path.stat().st_size)
        return static_bytes + sum(sizes), static_files + len(sizes)

    def budget(stage_start=None, seconds=None, additional=0):
        nonlocal peak_rss, peak_cuda, minimum_free
        import resource
        current = time.monotonic()
        free = shutil.disk_usage(root).free
        minimum_free = min(minimum_free, free)
        peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if torch is not None:
            peak_cuda = max(peak_cuda, torch.cuda.max_memory_reserved(), torch.cuda.memory_reserved())
        c.require(stopped_signal['signal'] is None, 'Manual signal stop requested')
        c.require(current - start <= c.LIMITS['worker_seconds'], 'Worker3000s cap reached')
        c.require(stage_start is None or current - stage_start <= seconds, 'Finite stage timing cap reached')
        active_seconds = 0 if active_step_start is None else current - active_step_start
        c.require(training_seconds + active_seconds <= c.LIMITS['head_optimization_seconds'], 'Head optimization300s cap reached')
        c.require(free >= c.LIMITS['reserve_free_bytes'] + additional, '4GiB reserve/write admission reached')
        c.require(peak_rss <= c.LIMITS['process_RSS_bytes'] and peak_cuda <= c.LIMITS['CUDA_reserved_bytes'], 'RSS/CUDA stop bound reached')
        external_bytes, external_files = external_accounting()
        c.require(total_bytes + external_bytes + additional <= c.LIMITS['result_payload_bytes'] and
                  total_files + external_files < c.LIMITS['regular_files'], 'Whole-root payload/file bound reached')
        c.require(metadata_bytes + external_bytes <= c.LIMITS['metadata_bytes'], 'Whole-root metadata128MiB bound reached')

    def write_bytes(relative, payload, *, limit, metadata=False, stop_safe=False):
        nonlocal total_bytes, total_files, metadata_bytes
        c.safe_relative(relative)
        c.require(len(payload) <= limit, 'Per-file writer bound exceeded: ' + relative)
        external_bytes, external_files = external_accounting()
        c.require(total_files + external_files + 1 <= c.LIMITS['regular_files'] and
                  metadata_bytes + external_bytes + (len(payload) if metadata else 0) <= c.LIMITS['metadata_bytes'],
                  'Prospective whole-root file/metadata admission failed')
        if not stop_safe:
            budget(additional=len(payload))
        else:
            c.require(shutil.disk_usage(root).free >= c.LIMITS['reserve_free_bytes'] + len(payload) and
                      total_bytes + external_bytes + len(payload) <= c.LIMITS['result_payload_bytes'], 'Failure evidence reserve unavailable')
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        total_bytes += len(payload); total_files += 1
        if metadata:
            metadata_bytes += len(payload)
        return {'path': path.relative_to(root).as_posix(), 'sha256': c.sha256(path), 'bytes': len(payload)}

    def write_json(relative, value, *, stop_safe=False):
        return write_bytes(relative, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode(),
                           limit=c.LIMITS['metadata_bytes'], metadata=True, stop_safe=stop_safe)

    def write_array(relative, value):
        from io import BytesIO
        value = np.ascontiguousarray(value)
        stream = BytesIO(); np.save(stream, value, allow_pickle=False)
        c.require(len(stream.getvalue()) <= value.nbytes + 256, 'NPY header bound exceeded')
        return write_bytes(relative, stream.getvalue(), limit=c.LIMITS['one_array_bytes'])

    def write_png(relative, pixels):
        from io import BytesIO
        stream = BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
        return write_bytes(relative, stream.getvalue(), limit=c.LIMITS['one256PNG_bytes'])

    def reserve(name):
        c.require(name in counts and counts[name] < c.COUNTS[name], 'Prospective operation cap reached: ' + name)
        budget()
        counts[name] += 1
        finite_counts(counts, completed)
        write_json('call_trace/%05d.json' % sum(counts.values()), {'event': sum(counts.values()),
            'before_call': name, 'completed_head_updates': completed, 'partial_microbatches': pending,
            'head_optimizer_inflight': inflight, 'counts': dict(counts), 'seconds': time.monotonic() - start})

    def frozen_check():
        nonlocal state_valid
        status = all(not module.training and all(not q.requires_grad and q.grad is None for q in module.parameters())
                     for module in (original, net))
        digests = {'original': model.state_digest(original), 'restorer': model.state_digest(net)}
        state_valid = status and digests == {'original': original_digest, 'restorer': base_digest}
        c.require(state_valid, 'Frozen515 parent state/mode/gradient changed')
        return digests

    def cpu_tree(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
        if isinstance(value, list): return [cpu_tree(v) for v in value]
        if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
        return value

    def checkpoint(slot, reason=None):
        nonlocal total_bytes, total_files, last_safe
        if head is None or optimizer is None or net is None:
            return None
        cap = c.DESIGN['state_and_return']['checkpoint_write_cap_bytes_each']
        # Each slot is write-once: an interrupted temp occupies its own slot.
        directory = out / slot
        c.require(not directory.exists(), 'Preserve existing checkpoint slot')
        external_bytes, external_files = external_accounting()
        c.require(shutil.disk_usage(root).free >= cap + c.LIMITS['reserve_free_bytes'] and
                  total_bytes + external_bytes + cap <= c.LIMITS['result_payload_bytes'] and
                  total_files + external_files + 1 <= c.LIMITS['regular_files'], 'Checkpoint cap plus reserve unavailable')
        directory.mkdir()
        temporary = directory / ('checkpoint.pt.tmp-' + os.urandom(16).hex())
        path = directory / 'checkpoint.pt'
        safe = checkpoint_safe(completed, pending, inflight, state_valid) and committed_rng is not None
        value = {'format': 'codeformer-tone-control-composite-state-v1',
            'protocol_sha256': protocol_sha, 'design_sha256': c.DESIGN_SHA256,
            'base_model': model.cpu_state(net), 'head': model.cpu_state(head),
            'head_optimizer': cpu_tree(optimizer.state_dict()), 'rng': committed_rng,
            'sampler': {'next_update': completed + 1, 'retained_training_parity_sha256': p['training_parity']['sha256']},
            'schedule': {'completed_updates': completed, 'training': c.DESIGN['training']},
            'completed_head_updates': completed, 'restorer_updates_this_run': 0,
            'historical_restorer_updates': 50, 'counts': dict(counts),
            'partial_microbatches': pending, 'head_optimizer_inflight': inflight,
            'head_state_safe': safe, 'no_automatic_resume_supported': True,
            'parent_checkpoint_sha256': c.ASSETS['base']['sha256'],
            'original_checkpoint_sha256': c.ASSETS['initializer']['sha256'],
            'base_model_tensor_digest': base_digest, 'head_tensor_digest': model.state_digest(head),
            'last_safe_checkpoint': last_safe, 'reason': reason, 'model_qualified': False,
            'composite_specification': c.DESIGN['composite_and_loss'], 'controller': c.DESIGN['controller']}
        class CappedWriter:
            def __init__(self, stream): self.stream, self.count = stream, 0
            def write(self, block):
                c.require(self.count + len(block) <= cap, 'Composite384MiB writer cap reached')
                count = self.stream.write(block); self.count += count; return count
            def flush(self): return self.stream.flush()
            def tell(self): return self.stream.tell()
        try:
            with temporary.open('xb') as stream:
                writer = CappedWriter(stream)
                torch.save(value, writer); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            if temporary.exists():
                total_bytes += temporary.stat().st_size; total_files += 1
            raise
        del value
        os.replace(temporary, path)
        total_bytes += path.stat().st_size; total_files += 1
        receipt = {'path': path.relative_to(root).as_posix(), 'sha256': c.sha256(path), 'bytes': path.stat().st_size,
                   'completed_head_updates': completed, 'head_state_safe': safe}
        checkpoint_receipts[slot] = receipt
        if safe: last_safe = receipt['path']
        return receipt

    def retained_parity(row, raw, png, state):
        entry = index['retained_outputs'][state][row['id']]
        for field in ('raw', 'png'):
            c.require(c.sha256(entry[field + '_path']) == entry[field + '_sha256'], 'Retained parity file changed')
        old = np.load(entry['raw_path'], allow_pickle=False)
        with Image.open(entry['png_path']) as image: old_png = np.asarray(image).copy()
        c.require(raw.dtype == old.dtype == np.float32 and np.array_equal(raw, old) and np.array_equal(png, old_png),
                  'Fresh frozen parent RAW/PNG differs from retained ' + state + ': ' + row['id'])

    def as_rgb(tensor):
        return np.ascontiguousarray(tensor[0].permute(1, 2, 0).detach().cpu().numpy(), dtype=np.float32)

    def controlled(row, base, *, training_target=None, reserve_name=None):
        feature = model.ordinary_tensor(data.features(row['input256'], row['observed']), 'features')
        bt = model.ordinary_tensor(base, 'rgb')
        it = model.ordinary_tensor(row['input256'], 'rgb')
        mt = model.ordinary_tensor(row['observed'], 'mask')
        if reserve_name: reserve(reserve_name)
        if training_target is None:
            with torch.no_grad(): corrected, delta, mean = model.apply_head(head, feature, bt, it, mt)
            return as_rgb(corrected), float(delta[0]), float(mean[0])
        target = model.ordinary_tensor(training_target, 'target')
        loss, corrected, delta, mean = model.training_loss(head, feature, bt, it, mt, target)
        return loss, float(delta[0].detach()), float(mean[0].detach())

    def bank(name, module, count_key, head_key=None):
        began = time.monotonic(); rows = {}
        cap = c.LIMITS['original_DEV_bank_seconds'] if head_key is None else c.LIMITS['each_controller_DEV_bank_seconds']
        for number, row in enumerate(index['dev'], 1):
            budget(began, cap)
            reserve(count_key)
            base = model.infer_rgb(module, row['input256'])
            old_png = delivered_png(base, row['input_rgb8'], row['observed'])
            if name in ('original0', 'head0000'):
                retained_parity(row, base, old_png, 'original' if name == 'original0' else 'base')
            if head_key:
                rgb, delta, mean = controlled(row, base, reserve_name=head_key)
                if name == 'head0000':
                    c.require(delta == 0 and np.array_equal(rgb[row['observed']], base[row['observed']]) and
                              np.array_equal(rgb[~row['observed']], row['input256'][~row['observed']]),
                              'Zero head must preserve observed base and protected input')
                write_array(name + '/' + row['id'] + '/base_raw.npy', base)
            else:
                rgb, delta, mean = base, None, model.numpy_mean_y(base, row['observed'])
            png = delivered_png(rgb, row['input_rgb8'], row['observed'])
            if name == 'head0000': c.require(np.array_equal(png, old_png), 'Zero-head deliveredPNG parity differs')
            raw_file = write_array(name + '/' + row['id'] + '/raw.npy', rgb)
            png_file = write_png(name + '/' + row['id'] + '/native.png', png)
            metric = pixel_metrics(rgb, png, row['target'], row['observed'], evidence=row['evidence'], ssim_fn=ssim_fn)
            item = {'id': row['id'], 'source': row['source'], 'profile': row['profile'], 'evidence': row['evidence'],
                    'metrics': metric, 'raw': raw_file, 'png': png_file, 'delta': delta, 'mean_y': mean,
                    'observed_pixels': int(row['observed'].sum()), 'sourcebindings': row['sourcebindings'],
                    'features': data.features(row['input256'], row['observed']).tolist(),
                    'clipped_scalar_fraction_observed': None if delta is None else
                        float(((base[row['observed']] + np.float32(delta) < 0) | (base[row['observed']] + np.float32(delta) > 1)).mean())}
            write_json(name + '/' + row['id'] + '/evidence.json', item)
            rows[row['id']] = item
            if number % 50 == 0 or number == 406:
                print({'bank': name, 'cases': number, 'of': 406}, flush=True)
        write_json(name + '_metrics.json', {'complete': True, 'cases': rows, 'seconds': time.monotonic() - began})
        budget(began, cap)
        return rows

    def terminal(reason=None):
        nonlocal peak_rss, peak_cuda, minimum_free
        import resource
        peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        minimum_free = min(minimum_free, shutil.disk_usage(root).free)
        if torch is not None:
            peak_cuda = max(peak_cuda, torch.cuda.max_memory_reserved(), torch.cuda.memory_reserved())
        return {'protocol_sha256': protocol_sha, 'counts': dict(counts), 'completed_head_updates': completed,
            'head_optimizer_inflight': inflight, 'partial_microbatches': pending, 'restorer_updates_this_run': 0,
            'historical_restorer_updates': 50, 'reason': reason, 'checkpoints': checkpoint_receipts,
            'last_safe_checkpoint': last_safe, 'head_optimization_seconds': training_seconds,
            'seconds': time.monotonic() - start, 'resources': {'peak_RSS_bytes': peak_rss,
                'peak_CUDA_reserved_bytes': peak_cuda, 'minimum_sampled_root_free_bytes': minimum_free,
                'disk_minimum_is_sampled_not_continuous': True},
            'model_qualified': False, 'app_promoted': False, 'automatic_resume': False}

    with worker_lock(root, protocol_sha):
        for sig in (signal.SIGTERM, signal.SIGINT):
            saved_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda value, frame: stopped_signal.__setitem__('signal', value))
        try:
            write_json('transfer_verified.json', {k: v for k, v in verified.items() if k != 'protocol'})
            write_json('idle_verified.json', idle_receipt)
            index = data.load_index(parent, p)
            write_json('input_index.json', index['binding_receipt'])
            torch = model.activate_runtime(root, protocol_sha)
            # Exact retained backend policy; no package upgrades or precision fallback.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            random.seed(20261017); np.random.seed(20261017); torch.manual_seed(20261017); torch.cuda.manual_seed_all(20261017)
            from skimage.metrics import structural_similarity as ssim_fn
            import cv2
            parity = c.read_json(parent / p['training_parity']['path'])
            runtime = {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'Pillow': pillow_version,
                       'OpenCV': cv2.__version__, 'CUDA': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
                       'capability': list(torch.cuda.get_device_capability(0))}
            required_runtime = ('python', 'torch', 'numpy', 'Pillow', 'OpenCV', 'CUDA', 'device', 'capability')
            c.require(all(key in parity['runtime_expected'] and runtime[key] == parity['runtime_expected'][key]
                          for key in required_runtime),
                      'Retain the exact audited parent runtime; do not upgrade or substitute')
            freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True, capture_output=True, text=True, timeout=30)
            c.require(len(freeze.stdout.encode()) <= c.LIMITS['one_log_bytes'], 'Package inventory exceeds bound')
            actual_packages = sorted(line.strip() for line in freeze.stdout.splitlines() if line.strip())
            retained_packages = sorted(line.strip() for line in parity['runtime_expected']['pip_freeze'].splitlines() if line.strip())
            c.require(actual_packages == retained_packages, 'Retained package inventory changed; do not upgrade the frozen environment')
            write_json('runtime.json', {**runtime, 'python': sys.version,
                'pip_freeze': freeze.stdout, 'pip_freeze_returncode': freeze.returncode,
                'checked_runtime_keys': list(required_runtime), 'package_inventory_exact_line_set': True,
                'NVIDIA_driver_and_UUID': subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                    '--format=csv,noheader'], check=True, capture_output=True, text=True, timeout=20).stdout,
                'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'autocast': False, 'TF32': False, 'cudnn_benchmark': False, 'universal_determinism_claim': False})
            began = time.monotonic()
            original, net, factory = model.load_frozen_parents(parent / c.ASSETS['initializer']['path'], parent / c.ASSETS['base']['path'])
            budget(began, c.LIMITS['initialization_seconds'])
            base_digest, original_digest = model.state_digest(net), model.state_digest(original)
            write_json('factory.json', factory)
            head = model.create_head()
            for identifier in c.DESIGN['prerequisites']['frozen_parity_cases']:
                row = next(row for row in index['dev'] if row['id'] == identifier)
                for state, module in [('original', original), ('base', net)]:
                    reserve('CodeFormer_parity'); rgb = model.infer_rgb(module, row['input256'])
                    retained_parity(row, rgb, delivered_png(rgb, row['input_rgb8'], row['observed']), state)
            # Reserve two MiB for the separately capped shared prerequisite writer.
            budget(additional=2 * 1024**2)
            model.head_prerequisites(head, out, root, reserve, budget)
            prerequisite_files = [path for path in (out / 'head_derivative_prerequisites').rglob('*') if path.is_file()]
            prerequisite_bytes = sum(path.stat().st_size for path in prerequisite_files)
            c.require(len(prerequisite_files) == 25 and prerequisite_bytes <= 2 * 1024**2, 'Shared prerequisite artifact census/cap differs')
            total_bytes += prerequisite_bytes; total_files += len(prerequisite_files)
            metadata_bytes += sum(path.stat().st_size for path in prerequisite_files if path.suffix == '.json')
            frozen_check()
            optimizer = torch.optim.Adam(head.parameters(), lr=.001, betas=(.9, .99), eps=1e-8, weight_decay=0)
            c.require(len(list(head.parameters())) == 4 and sum(q.numel() for q in head.parameters()) == 8257,
                      'Only8257 head parameters in fresh optimizer')
            committed_rng = model.capture_rng()
            checkpoint('composite_head0000')
            all_banks['original0'] = bank('original0', original, 'CodeFormer_original_DEV_bank')
            all_banks['head0000'] = bank('head0000', net, 'CodeFormer_zero_head_DEV_bank', 'controller_DEV_zero_and50')
            write_json('zero_head_guards.json', preservation_guards(all_banks['head0000'], all_banks['original0']))
            began = time.monotonic(); caches = {}
            for row in index['train']:
                budget(began, c.LIMITS['TRAIN_cache_seconds'])
                reserve('TRAIN_input_writes'); write_png('TRAIN/' + row['id'] + '/input.png', row['input_rgb8'])
                reserve('CodeFormer_TRAIN_original'); o = model.infer_rgb(original, row['input256'])
                reserve('CodeFormer_TRAIN_full_coarse'); b = model.infer_rgb(net, row['input256'])
                original_file = write_array('TRAIN/' + row['id'] + '/original_raw.npy', o)
                base_file = write_array('TRAIN/' + row['id'] + '/base_raw.npy', b)
                source = o if row['role'] == 'normal' else b
                target = float(model.mean_y(model.ordinary_tensor(source, 'rgb'), model.ordinary_tensor(row['observed'], 'mask'))[0])
                lower = model.numpy_mean_y(np.clip(b - np.float32(.25), 0, 1), row['observed'])
                upper = model.numpy_mean_y(np.clip(b + np.float32(.25), 0, 1), row['observed'])
                write_json('TRAIN/' + row['id'] + '/cache.json', {'id': row['id'], 'role': row['role'],
                    'update': row['update'], 'slot': row['slot'], 'target_mean_y': target,
                    'target_source': 'original' if row['role'] == 'normal' else 'full_coarse_base',
                    'reachable_mean_interval': [lower, upper], 'features': data.features(row['input256'], row['observed']).tolist(),
                    'sourcebindings': row['sourcebindings'], 'original_raw': original_file, 'base_raw': base_file})
                c.require(lower <= target <= upper, 'Fixed tone range cannot reach this TRAIN behavioral mean; retain stop')
                caches[row['id']] = {'base': b, 'target': target}
            write_json('TRAIN_cache_timing.json', {'complete': True, 'cases': 200, 'seconds': time.monotonic() - began,
                'cap_seconds': c.LIMITS['TRAIN_cache_seconds']})
            budget(began, c.LIMITS['TRAIN_cache_seconds'])
            head.train()
            for update in range(1, 51):
                budget(); step_start = active_step_start = time.monotonic(); optimizer.zero_grad(set_to_none=True); history = []
                for row in index['train'][(update - 1) * 4:update * 4]:
                    cache = caches[row['id']]
                    loss, delta, mean = controlled(row, cache['base'], training_target=cache['target'], reserve_name='controller_TRAIN')
                    reserve('head_backward_calls'); (loss / 4).backward(); pending += 1
                    history.append({'id': row['id'], 'role': row['role'], 'delta': delta, 'target_mean_y': cache['target'],
                                    'predicted_mean_y': mean, 'loss': float(loss.detach()), 'actual_divisor': 4})
                c.require(all(q.grad is not None and torch.isfinite(q.grad).all().item() for q in head.parameters()),
                          'Every connected head gradient must be finite; zeros are allowed')
                write_json('history/u%04d_attempt.json' % update, {'update': update, 'completed_before': completed, 'observations': history})
                reserve('head_optimizer_step_attempts'); inflight = True
                optimizer.step()
                c.require(all(torch.isfinite(q).all().item() for q in head.parameters()), 'Nonfinite updated head')
                completed = update; inflight = False; pending = 0
                optimizer.zero_grad(set_to_none=True); committed_rng = model.capture_rng()
                training_seconds += time.monotonic() - step_start
                active_step_start = None
                budget()
                finite_counts(counts, completed)
                write_json('history/u%04d_committed.json' % update, {'update': update, 'completed_head_updates': completed,
                    'observations': history, 'head_tensor_digest': model.state_digest(head), 'head_optimization_seconds': training_seconds})
                if update == 1 or update % 10 == 0:
                    print({'head_update': update, 'of': 50, 'mean_loss': sum(r['loss'] for r in history) / 4}, flush=True)
            head.eval(); frozen_check()
            checkpoint('composite_head0050')  # Genuine state is retained before every quality decision.
            all_banks['head0050'] = bank('head0050', net, 'CodeFormer_head50_DEV_bank', 'controller_DEV_zero_and50')
            stress = []
            for identifier in c.DESIGN['evaluation']['known_overlap_stress']['cases']:
                row = next(row for row in index['train'] if row['id'] == identifier)
                cache = caches[identifier]; base = cache['base']
                rgb, delta, mean = controlled(row, base, reserve_name='controller_final_known_stress')
                png = delivered_png(rgb, row['input_rgb8'], row['observed'])
                target = data.metric_target256(row['source_path'], row['sourcebindings']['source_sha256'])
                before = pixel_metrics(base, delivered_png(base, row['input_rgb8'], row['observed']), target, row['observed'], evidence='paired_synthetic', ssim_fn=ssim_fn)
                after = pixel_metrics(rgb, png, target, row['observed'], evidence='paired_synthetic', ssim_fn=ssim_fn)
                passed = (after['native_png']['luminance_bias'] < before['native_png']['luminance_bias']) if row['role'] == 'normal' else \
                    (after['native_png']['mae'] <= before['native_png']['mae'] and after['native_png']['ssim'] >= before['native_png']['ssim'])
                stress.append({'id': identifier, 'role': row['role'], 'before': before, 'after': after, 'pass': passed,
                    'delta': delta, 'mean_y': mean, 'RAW': write_array('known_stress/' + identifier + '/raw.npy', rgb),
                    'PNG': write_png('known_stress/' + identifier + '/native.png', png), 'independent_generalization': False})
            write_json('known_overlap_stress.json', {'complete': True, 'cases': stress, 'pass': all(r['pass'] for r in stress)})
            gates = preservation_guards(all_banks['head0050'], all_banks['original0'])
            dark = dark_retention(all_banks['head0050'], all_banks['head0000'])
            write_json('preservation_guards.json', gates); write_json('dark_retention.json', dark)
            frozen_check(); finite_counts(counts, completed, terminal=True)
            write_json('final_frozen_state.json', {'complete': True, 'digests': frozen_check(), 'CodeFormer_updates_this_run': 0})
            c.require(gates['pass'] and dark['pass'] and all(r['pass'] for r in stress),
                      'Declared preservation, dark retention or known overlap requirement failed; retain trained state')
            checkpoint('stopped_composite', reason='finite_complete; no resume or additional updates authorized')
            write_json('run_results.json', {**terminal(), 'complete': True, 'quality_gates_pass': True,
                'user_acceptance_and_independent_review_pending': True})
        except BaseException as exc:
            reason = type(exc).__name__ + ': ' + str(exc)
            checkpoint_error = None
            try:
                if optimizer is not None: optimizer.zero_grad(set_to_none=True)
                if head is not None: head.eval()
                if net is not None and original is not None: frozen_check()
                checkpoint('stopped_composite', reason=reason)
            except BaseException as checkpoint_exc:
                checkpoint_error = type(checkpoint_exc).__name__ + ': ' + str(checkpoint_exc)
            write_json('failure.json', {**terminal(reason), 'complete': True, 'failure': True,
                'traceback': traceback.format_exc(), 'stopped_checkpoint_error': checkpoint_error,
                'training_success_not_implied': True}, stop_safe=True)
            raise
        finally:
            for sig, handler in saved_handlers.items(): signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-transfer', action='store_true')
    action.add_argument('--run', action='store_true')
    action.add_argument('--export', action='store_true')
    args = parser.parse_args(argv)
    root = c.manual_root(args.root)
    from . import transport
    if args.verify_transfer:
        receipt = transport.verify_transfer(root, args.protocol_sha)
        print({'complete': True, 'dependencies': receipt['total_readonly_bindings_verified'], 'neural_calls': 0})
    elif args.export:
        print(transport.export_results(root, args.protocol_sha))
    else:
        run(root, args.protocol_sha)


if __name__ == '__main__':
    main()
