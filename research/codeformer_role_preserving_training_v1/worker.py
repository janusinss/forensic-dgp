"""Finite manual L4 role-preserving CodeFormer training; static import never trains."""
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


def checkpoint_slot_admission(out, relative):
    """Physical final or partial updated slots are mutually exclusive before writes."""
    c.require(relative in ('snapshot_0000/checkpoint.pt', 'snapshot_0050/checkpoint.pt', 'stopped_checkpoint.pt'),
              'Undeclared checkpoint destination')
    path = c.no_links(Path(out) / relative)
    c.require(not path.exists() and not any(path.parent.glob(path.name + '.tmp-*')), 'Preserve occupied checkpoint slot or partial')
    if relative != 'snapshot_0000/checkpoint.pt':
        other = Path(out) / ('stopped_checkpoint.pt' if relative == 'snapshot_0050/checkpoint.pt' else 'snapshot_0050/checkpoint.pt')
        c.no_links(other)
        c.require(not other.exists() and not any(other.parent.glob(other.name + '.tmp-*')),
                  'Updated checkpoint50/stopped final or partial slots are mutually exclusive')
    return path


def idle_conflicts(rows, own_pid, root, protocol_sha):
    """Classify full pid/ppid/argv inventory; admit only exact current launch ancestors."""
    from scripts.supervise_codeformer_role_preserving_training_v1_vm import competing_commands
    table = {pid: (ppid, command) for pid, ppid, command in rows}
    c.require(own_pid in table and len(table) == len(rows), 'Unique current worker pid required')
    accepted = {own_pid}
    parent_id = table[own_pid][0]
    if parent_id in table:
        ppid, command = table[parent_id]; words = shlex.split(command)
        if words and Path(words[0]).name == 'timeout':
            expected_tail = [sys.executable, '-B', '-u', '-m', c.MODULE, '--root', str(root), '--protocol-sha', protocol_sha, '--run']
            c.require(words[1:4] == ['--signal=TERM', '--kill-after=30', '3000'] and words[4:] == expected_tail,
                      'Current timeout ancestor differs from exact supervised worker invocation')
            accepted.add(parent_id); parent_id = ppid
        if parent_id in table:
            ppid, command = table[parent_id]; words = shlex.split(command)
            script = str(root / 'scripts/supervise_codeformer_role_preserving_training_v1_vm.py')
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
    c.require(set(counts) == set(c.COUNTS) and all(type(counts[k]) is int and 0 <= counts[k] <= cap
              for k, cap in c.COUNTS.items()), 'Exact finite operation keys/caps required')
    c.require(type(completed) is int and 0 <= completed <= 50 and
              completed <= counts['optimizer_step_attempts'] <= completed + 1,
              'Committed versus attempted Adam updates differs')
    c.require(counts['backward_calls'] <= counts['candidate_training_forwards'] <= 200,
              'Candidate/backward chronology differs')
    if terminal:
        c.require(completed == 50 and counts == c.COUNTS and sum(counts.values()) == c.RESERVATION_CAP,
                  'Every finite reservation must close at fifty committed updates')


def learning_rate(update, base):
    c.require(type(update) is int and 1 <= update <= 50 and base in (1e-5, 2e-5), 'Finite new warmup position required')
    return base * update / 50


@contextmanager
def capture_forward(net):
    """Observe one existing forward and its actual dictionary indices; no call added."""
    captured = {}; lookup = net.quantize.get_codebook_feat
    def observe(module, arguments, output):
        c.require(isinstance(output, tuple) and len(output) == 3, 'Exact CodeFormer return required')
        captured['logits'], captured['latent'] = output[1], output[2]
    def dictionary(indices, shape):
        c.require('tokens' not in captured, 'Exactly one candidate dictionary lookup required')
        captured['tokens'] = indices.detach().clone()
        return lookup(indices, shape)
    hook = net.register_forward_hook(observe)
    try:
        net.quantize.get_codebook_feat = dictionary
        yield captured
    finally:
        net.quantize.get_codebook_feat = lookup
        hook.remove()


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


def run(root, protocol_sha):
    root = c.manual_root(root)  # Always before any framework import.
    from . import transport, data, model
    verified = transport.verify_transfer(root, protocol_sha)
    p = verified['protocol']; parent = c.resolve_parent(root); out = root / c.OUTPUTS
    c.require(not out.exists() and not (root / c.LOCK_NAME).exists(), 'Preserve previous/partial pilot; no repeat or resume')
    c.require(shutil.disk_usage(root).free >= c.LIMITS['initial_free_bytes'], '16GiB free required after installation')
    idle = require_idle(root, protocol_sha)
    import numpy as np
    from PIL import Image, __version__ as pillow_version
    import resource
    import research
    c.require(list(research.__path__) == [str(root / 'research')], 'Fresh packet research namespace required')
    research.__path__.append(str(parent / 'research'))
    sys.path.append(str(parent))  # Only the fully verified immutable model parent.
    with (root / c.LOCK_NAME).open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'protocol_sha256': protocol_sha, 'automatic_resume': False}, stream)
        stream.flush(); os.fsync(stream.fileno())
    out.mkdir(parents=True)
    started = time.monotonic(); phase_started = started; phase = 'initialization'
    phase_seconds = {}; counts = {k: 0 for k in c.COUNTS}; own = {}
    completed = pending = 0; inflight = False; state_valid = True
    torch = net = teacher = original = parent_teacher = perceptual = optimizer = None
    committed_rng = None; latest_safe = None; checkpoint_receipts = {}; frozen_before = None
    peak_rss = peak_cuda = 0; minimum_free = shutil.disk_usage(root).free
    training_seconds = 0.; step_started = None; stop_requested = None; committed_state = None
    module_modes = None; index = None
    static = {}
    for folder, dirs, files in os.walk(root, followlinks=False):
        for directory in dirs:
            c.no_links(Path(folder) / directory)
        for filename in files:
            path = c.no_links(Path(folder) / filename)
            relative = path.relative_to(root).as_posix()
            if not relative.startswith('outputs/supervision/'):
                static[relative] = path.stat().st_size

    def external():
        directory = root / 'outputs/supervision'
        values = {}
        if directory.exists():
            for path in directory.iterdir():
                c.no_links(path)
                c.require(path.is_file(), 'Only regular supervision outputs permitted')
                values[path.name] = path.stat().st_size
        return values

    def usage():
        extras = external()
        return (sum(static.values()) + sum(v['bytes'] for v in own.values()) + sum(extras.values()),
                len(static) + len(own) + len(extras),
                sum(static.values()) + sum(v['bytes'] for v in own.values() if v['metadata']) + sum(extras.values()))

    def sample_resources():
        nonlocal peak_rss, peak_cuda, minimum_free
        peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        minimum_free = min(minimum_free, shutil.disk_usage(root).free, shutil.disk_usage(Path.home()).free)
        if torch is not None:
            peak_cuda = max(peak_cuda, torch.cuda.max_memory_reserved(), torch.cuda.memory_reserved())

    def budget(extra=0, files=0, metadata=0, *, stop_safe=False):
        sample_resources(); total, number, meta = usage(); now = time.monotonic()
        c.require(total + extra <= c.LIMITS['result_payload_bytes'] and number + files <= c.LIMITS['regular_file_cap']
                  and meta + metadata <= c.LIMITS['metadata_bytes'], 'Whole-root payload/file/metadata admission reached')
        c.require(min(shutil.disk_usage(root).free, shutil.disk_usage(Path.home()).free) >= c.LIMITS['reserve_free_bytes'] + extra,
                  '4GiB disk reserve/write admission reached')
        if not stop_safe:
            c.require(stop_requested is None, 'Manual stop signal received')
            c.require(now - started <= c.LIMITS['worker_seconds'], '3000second worker limit reached')
            cap = c.LIMITS['snapshot_seconds'] if phase.startswith('snapshot_') else c.LIMITS[phase + '_seconds']
            c.require(phase_seconds.get(phase, 0.) + now - phase_started <= cap, 'Finite phase time limit reached: ' + phase)
            active = 0. if step_started is None else now - step_started
            c.require(training_seconds + active <= c.LIMITS['training_seconds'], '900second actual optimization limit reached')
            c.require(peak_rss <= c.LIMITS['RSS_bytes'] and peak_cuda <= c.LIMITS['CUDA_reserved_bytes'], 'RSS/CUDA memory limit reached')

    def write_bytes(relative, payload, *, cap, metadata=False, stop_safe=False):
        path = c.safe_path(out, relative, must_exist=False)
        c.require(relative not in own and not path.exists() and len(payload) <= cap, 'Exclusive bounded output required: ' + relative)
        budget(len(payload), 1, len(payload) if metadata else 0, stop_safe=stop_safe)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        own[relative] = {'bytes': len(payload), 'metadata': metadata}
        return {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}

    def write_json(relative, value, *, stop_safe=False):
        payload = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        cap = c.LIMITS['training_history_line_bytes'] if relative.startswith('history/') else c.LIMITS['one_JSON_bytes']
        return write_bytes(relative, payload, cap=cap, metadata=True, stop_safe=stop_safe)

    def write_array(relative, array):
        from io import BytesIO
        array = np.ascontiguousarray(array); stream = BytesIO(); np.save(stream, array, allow_pickle=False)
        payload = stream.getvalue()
        c.require(len(payload) <= array.nbytes + 256, 'Bounded NPY header required')
        return write_bytes(relative, payload, cap=c.LIMITS['array_file_bytes'])

    def write_png(relative, pixels):
        from io import BytesIO
        c.require(pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8, 'Exact RGB256 image required')
        stream = BytesIO(); Image.fromarray(pixels).save(stream, format='PNG')
        return write_bytes(relative, stream.getvalue(), cap=c.LIMITS['PNG_file_bytes'])

    def reserve(name):
        budget(); finite_counts(counts, completed)
        c.require(name in counts and counts[name] < c.COUNTS[name], 'Undeclared or exhausted operation: ' + name)
        counts[name] += 1
        write_json('call_trace/%05d.json' % sum(counts.values()), {'event': sum(counts.values()), 'before_call': name,
            'completed_training_updates': completed, 'partial_microbatches': pending, 'optimizer_inflight': inflight,
            'counts': dict(counts), 'seconds': time.monotonic() - started, 'reservation_is_not_completion_evidence': True})

    def alarm(signum, frame):
        nonlocal stop_requested
        stop_requested = signum
        raise TimeoutError('Finite manual signal or phase limit reached')

    def next_phase(name):
        nonlocal phase, phase_started
        budget(); phase_seconds[phase] = phase_seconds.get(phase, 0.) + time.monotonic() - phase_started
        phase, phase_started = name, time.monotonic()
        cap = c.LIMITS['snapshot_seconds'] if name.startswith('snapshot_') else c.LIMITS[name + '_seconds']
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(cap - phase_seconds.get(name, 0.), c.LIMITS['worker_seconds'] - (time.monotonic() - started))))

    def tensor(array):
        c.require(not torch.is_inference_mode_enabled(), 'Training/target tensor construction must be ordinary')
        value = torch.from_numpy(np.ascontiguousarray(array).copy())
        if array.ndim == 3:
            value = value.permute(2, 0, 1).unsqueeze(0)
        return value.to('cuda')

    def as_rgb(value):
        return np.ascontiguousarray(value[0].detach().float().cpu().permute(1, 2, 0).numpy())

    def array_digest(value):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    def cpu_tree(value):
        if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
        if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
        if isinstance(value, list): return [cpu_tree(v) for v in value]
        if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
        return value

    def tree_digest(value):
        """Canonical recursive state proof, independent of Torch serialization bytes."""
        h = hashlib.sha256()
        def visit(item):
            if isinstance(item, torch.Tensor):
                array = item.detach().cpu().contiguous()
                header = json.dumps(['tensor', str(array.dtype), list(array.shape)], separators=(',', ':')).encode()
                h.update(len(header).to_bytes(8, 'big')); h.update(header)
                h.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(item, dict):
                h.update(b'dict[')
                for key in sorted(item, key=lambda x: (type(x).__name__, str(x))): visit(key); visit(item[key])
                h.update(b']')
            elif isinstance(item, (tuple, list)):
                h.update((type(item).__name__ + '[').encode())
                for child in item: visit(child)
                h.update(b']')
            else:
                payload = json.dumps([type(item).__name__, item], separators=(',', ':'), allow_nan=False).encode()
                h.update(len(payload).to_bytes(8, 'big')); h.update(payload)
        visit(value); return h.hexdigest()

    def committed_payload():
        return {'model': model.cpu_state(net), 'optimizer': cpu_tree(optimizer.state_dict()), 'rng': cpu_tree(committed_rng),
            'sampler': {'next_update': completed + 1, 'data_bundle_sha256': p['files_sha256'][p['data_bundle_file']],
                'retained_training_parity_sha256': p['training_parity']['sha256'],
                'policy': 'fixed chronological first50 retained TRAIN updates; no random sampler is advanced',
                'completed_data_updates': completed, 'historical_sampler_state_installed': False},
            'schedule': {'completed_updates': completed, 'training': c.TRAINING}, 'counts': dict(counts),
            'completed_training_updates': completed, 'partial_microbatches': pending, 'optimizer_inflight': inflight}

    def rng():
        value = np.random.get_state()
        return {'python': random.getstate(), 'numpy': {'algorithm': value[0], 'keys': value[1].tolist(),
            'position': value[2], 'has_gauss': value[3], 'cached_gaussian': value[4]},
            'torch_cpu': torch.get_rng_state().clone(), 'torch_cuda': [x.clone() for x in torch.cuda.get_rng_state_all()]}

    def restore_rng(value):
        random.setstate(value['python']); n = value['numpy']
        np.random.set_state((n['algorithm'], np.asarray(n['keys'], np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
        torch.set_rng_state(value['torch_cpu']); torch.cuda.set_rng_state_all(value['torch_cuda'])

    def rng_equal(a, b):
        return a['python'] == b['python'] and a['numpy'] == b['numpy'] and torch.equal(a['torch_cpu'], b['torch_cpu']) and len(
            a['torch_cuda']) == len(b['torch_cuda']) and all(torch.equal(x, y) for x, y in zip(a['torch_cuda'], b['torch_cuda']))

    def save_rng(label, value):
        doc = {'python': value['python'], 'numpy': value['numpy'], 'torch': {}}
        for name, state in [('CPU', value['torch_cpu']), *[('CUDA%d' % i, s) for i, s in enumerate(value['torch_cuda'])]]:
            doc['torch'][name] = write_array('gradient_route_witness/rng_' + label + '_' + name + '.npy', state.cpu().numpy())
        return write_json('gradient_route_witness/rng_' + label + '.json', doc)

    def frozen_check():
        nonlocal state_valid
        if net is None: return None
        rows = {'candidate_frozen': model.frozen_state_digest(net)}
        for label, module in [('original', original), ('parent', parent_teacher), ('HQ', teacher), ('perceptual', perceptual)]:
            c.require(module is not None and all(not part.training for part in module.modules()) and all(
                not q.requires_grad and q.grad is None for q in module.parameters()), 'Frozen teacher mode/gradient changed: ' + label)
            rows[label] = model.state_digest(module)
        state_valid = frozen_before is None or rows == frozen_before
        c.require(state_valid, 'Frozen teacher/generator/codebook tensor state changed')
        return rows

    def checkpoint(relative, reason=None, *, stop_safe=False):
        nonlocal latest_safe
        if net is None or optimizer is None:
            return None
        c.require(relative not in checkpoint_receipts, 'Preserve existing checkpoint receipt')
        checkpoint_slot_admission(out, relative)
        cap = c.LIMITS['initial_checkpoint_file_bytes'] if completed == 0 and not optimizer.state else c.LIMITS['trained_checkpoint_file_bytes']
        budget(cap, 1, stop_safe=stop_safe)
        path = c.safe_path(out, relative, must_exist=False)
        c.require(not path.exists() and not any(path.parent.glob(path.name + '.tmp-*')), 'Preserve checkpoint slot and partial temporary')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp-' + os.urandom(16).hex())
        safe = checkpoint_safe(completed, pending, inflight, state_valid) and committed_rng is not None
        actual_rng = rng()
        if relative == 'snapshot_0050/checkpoint.pt':
            c.require(rng_equal(committed_rng, actual_rng), 'Evaluation advanced actual RNG; exact committed50 capture cannot be asserted')
        committed = committed_payload(); full_digest = tree_digest(committed)
        value = {'format': 'codeformer-role-preserving-training-state-v1', 'protocol_sha256': protocol_sha,
            **committed, 'committed_state_digest': full_digest,
            'completed_training_updates': completed, 'historical_parent_updates': 50, 'partial_microbatches': pending,
            'partial_microbatches_discarded': pending, 'optimizer_inflight': inflight, 'state_safe': safe,
            'resume_safe': safe, 'automatic_resume_supported': False, 'last_safe_checkpoint': latest_safe,
            'actual_rng_at_capture': cpu_tree(actual_rng), 'actual_rng_equals_committed_rng': rng_equal(committed_rng, actual_rng),
            'candidate_tensor_digest': model.state_digest(net), 'frozen_state': frozen_check(), 'reason': reason,
            'parent_checkpoint_sha256': c.ASSETS['full_coarse_checkpoint']['sha256'],
            'original_checkpoint_sha256': c.ASSETS['initializer']['sha256'], 'objective': c.OBJECTIVE,
            'model_qualified': False, 'EMA': False}
        class Capped:
            def __init__(self, stream): self.stream, self.count = stream, 0
            def write(self, block):
                c.require(self.count + len(block) <= cap, 'Checkpoint writer cap reached')
                n = self.stream.write(block); self.count += n; return n
            def flush(self): return self.stream.flush()
            def tell(self): return self.stream.tell()
        try:
            with temporary.open('xb') as stream:
                torch.save(value, Capped(stream)); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            if temporary.exists():
                own[temporary.relative_to(out).as_posix()] = {'bytes': temporary.stat().st_size, 'metadata': False}
            raise
        del value, committed
        try:
            os.replace(temporary, path)
        except BaseException:
            if temporary.exists(): own[temporary.relative_to(out).as_posix()] = {'bytes': temporary.stat().st_size, 'metadata': False}
            raise
        own[relative] = {'bytes': path.stat().st_size, 'metadata': False}
        receipt = {'path': path.relative_to(root).as_posix(), 'sha256': c.sha256(path), 'bytes': path.stat().st_size,
            'completed_training_updates': completed, 'state_safe': safe, 'partial_microbatches': pending,
            'optimizer_inflight': inflight, 'model_tensor_sha256': model.state_digest(net), 'committed_state_digest': full_digest}
        checkpoint_receipts[relative] = receipt
        if safe: latest_safe = receipt['path']
        sample_resources()
        return receipt

    def stop_checkpoint(reason):
        if net is None or optimizer is None: return None
        saved50 = checkpoint_receipts.get('snapshot_0050/checkpoint.pt')
        if saved50 and completed == 50 and pending == 0 and not inflight and state_valid:
            c.require(rng_equal(committed_rng, rng()), 'Current actual RNG differs from retained committed50 state')
            actual = model.state_digest(net)
            c.require(actual == saved50['model_tensor_sha256'], 'Committed50 model differs from retained checkpoint')
            committed = committed_payload(); digest = tree_digest(committed); del committed
            c.require(digest == saved50['committed_state_digest'], 'Full model/Adam/RNG/sampler/rates/counters state differs from retained50 checkpoint')
            return write_json('stopped_checkpoint_alias.json', {'complete': True, 'protocol_sha256': protocol_sha,
                'completed_training_updates': 50, 'partial_microbatches': 0, 'optimizer_inflight': False,
                'source_checkpoint': saved50['path'], 'source_sha256': saved50['sha256'], 'source_bytes': saved50['bytes'],
                'committed_state_digest_equal': True, 'committed_state_digest': digest,
                'checkpoint': saved50, 'current_model_tensor_sha256': actual, 'same_committed_state': True,
                'duplicate_checkpoint_written': False, 'reason': reason, 'model_qualified': False}, stop_safe=True)
        return checkpoint('stopped_checkpoint.pt', reason, stop_safe=True)

    def retained_arrays(row, state):
        saved = index['retained_outputs'][state][row['id']]
        for key in ('raw', 'png'):
            path = Path(saved[key + '_path'])
            c.no_links(path)
            c.require(c.sha256(path) == saved[key + '_sha256'], 'Retained metric witness bytes changed')
        with Path(saved['raw_path']).open('rb') as stream:
            raw = np.load(stream, allow_pickle=False); c.require(not stream.read(1), 'Trailing retained NPY bytes')
        with Image.open(saved['png_path']) as image:
            c.require(image.mode == 'RGB' and image.size == (256, 256), 'Retained RGB256 required')
            png = np.asarray(image).copy()
        c.require(raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all(), 'Retained RAW precision/geometry differs')
        return raw, png

    def baseline_metrics():
        banks = {'original': {}, 'base': {}}
        for state in banks:
            for row in index['dev']:
                budget()
                raw, png = retained_arrays(row, state)
                c.require(np.array_equal(png, delivered_png(raw, row['input_rgb8'], row['observed'])), 'Retained RAW/PNG delivery differs')
                banks[state][row['id']] = {k: row[k] for k in ('id', 'source', 'profile', 'evidence')}
                banks[state][row['id']]['metrics'] = pixel_metrics(raw, png, row['target'], row['observed'], evidence=row['evidence'], ssim_fn=ssim_fn)
            write_json('retained_' + state + '_metrics.json', {'complete': True, 'cases': banks[state], 'neural_calls': 0,
                'metrics_recomputed_from_bound_RAW_PNG': True})
        return banks

    def snapshot(update):
        label = 'snapshot_%04d' % update
        next_phase(label); modes = [(m, m.training) for m in net.modules()]; net.eval(); rows = {}
        try:
            for number, row in enumerate(index['dev'], 1):
                budget(); reserve('snapshot_forwards')
                with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
                    native = model.infer(net, tensor_for_inference(row['input256']))
                    raw = as_rgb(model.delivered_rgb(native))
                png = delivered_png(raw, row['input_rgb8'], row['observed'])
                if update == 0:
                    old_raw, old_png = retained_arrays(row, 'base')
                    c.require(np.array_equal(raw, old_raw) and np.array_equal(png, old_png), 'All406 full-coarse initializer RAW/PNG parity failed: ' + row['id'])
                raw_pin = write_array(label + '/' + row['id'] + '/raw.npy', raw)
                png_pin = write_png(label + '/' + row['id'] + '/native.png', png)
                metric = pixel_metrics(raw, png, row['target'], row['observed'], evidence=row['evidence'], ssim_fn=ssim_fn)
                value = {k: row[k] for k in ('id', 'source', 'profile', 'evidence', 'sourcebindings')}
                value.update(metrics=metric, raw=raw_pin, png=png_pin, observed_pixels=int(row['observed'].sum()),
                    support_is_not_covering_mask=True, checkpoint_updates=update, independent_FINAL=False)
                write_json(label + '/' + row['id'] + '/evidence.json', value); rows[row['id']] = value
                if number % 50 == 0 or number == 406:
                    print(json.dumps({'snapshot_updates': update, 'cases': number, 'of': 406}), flush=True)
            write_json(label + '/metrics.json', {'complete': True, 'cases': rows, 'completed_training_updates': update,
                'native_clean_metrics_null': True, 'quality_acceptance_not_implied': True})
        finally:
            for module, mode in modes: module.training = mode
        frozen_check(); budget()
        return rows

    def tensor_for_inference(array):
        # Evaluation input only: safe to create inside inference_mode, never reused in backward.
        return torch.from_numpy(np.ascontiguousarray(array).copy()).permute(2, 0, 1).unsqueeze(0).to('cuda', dtype=torch.float32)

    def terminal(reason=None):
        sample_resources()
        return {'complete': True, 'protocol_sha256': protocol_sha, 'counts': dict(counts),
            'completed_training_updates': completed, 'partial_microbatches': pending, 'optimizer_inflight': inflight,
            'candidate_state_valid': state_valid, 'historical_parent_updates': 50, 'new_restorer_updates': completed,
            'gradient_query_reservations': counts['gradient_queries'], 'backward_call_reservations': counts['backward_calls'],
            'optimizer_constructions_completed': 1 if optimizer is not None else 0, 'optimizer_steps_completed': completed,
            'optimizer_step_attempts': counts['optimizer_step_attempts'], 'committed_training_updates': completed,
            'call_counters_are_reservations_not_completed_profiler_counts': True,
            'checkpoints': checkpoint_receipts, 'last_safe_checkpoint': latest_safe, 'reason': reason,
            'training_seconds': training_seconds, 'seconds': time.monotonic() - started,
            'phase_seconds': {**phase_seconds, phase: phase_seconds.get(phase, 0.) + time.monotonic() - phase_started},
            'resources': {'peak_RSS_bytes': peak_rss, 'peak_CUDA_reserved_bytes': peak_cuda,
                'minimum_sampled_free_bytes': minimum_free, 'disk_minimum_is_sampled_not_continuous': True},
            'model_qualified': False, 'app_promoted': False, 'automatic_resume': False, 'automatic_retry': False}

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers: signal.signal(sig, alarm)
    signal.setitimer(signal.ITIMER_REAL, c.LIMITS['initialization_seconds'])
    try:
        write_json('transfer_verified.json', {k: v for k, v in verified.items() if k not in ('protocol', 'data_bundle')})
        write_json('idle_verified.json', idle)
        bundle = c.read_json(c.safe_path(root, p['data_bundle_file']))
        index = data.load_index(parent, p, bundle)
        c.require(len(index['train']) == 200 and len(index['dev']) == 406 and len(index['hq']) == 100, 'Exact200/406/100 data census required')
        write_json('input_index.json', index['binding_receipt'])
        torch = model.activate_runtime(root, protocol_sha)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        random.seed(c.TRAINING['seed']); np.random.seed(c.TRAINING['seed'])
        torch.manual_seed(c.TRAINING['seed']); torch.cuda.manual_seed_all(c.TRAINING['seed'])
        from skimage.metrics import structural_similarity as ssim_fn
        import cv2
        expected_runtime = c.read_json(parent / 'outputs/adain_gradient_isolation_v1/runtime.json')
        runtime = {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'Pillow': pillow_version,
            'OpenCV': cv2.__version__, 'CUDA': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0))}
        # The old returned runtime uses Python/torch key names; never infer missing controls.
        c.require(all(k in expected_runtime and runtime[k] == expected_runtime[k] for k in runtime), 'Exact historical runtime differs')
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True, timeout=30)
        c.require(len(freeze.stdout.encode()) <= c.LIMITS['log_bytes'] and sorted(freeze.stdout.splitlines()) ==
            sorted(expected_runtime['pip_freeze'].splitlines()), 'Exact package inventory differs')
        runtime.update(pip_freeze=freeze.stdout, TF32=False, autocast=False, cudnn_benchmark=False,
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(), cudnn_deterministic=torch.backends.cudnn.deterministic,
            NVIDIA_driver_and_UUID=subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=20).stdout,
            universal_determinism_claim=False)
        old_algorithms = expected_runtime['algorithm_settings']
        c.require(runtime['deterministic_algorithms_enabled'] == old_algorithms['deterministic_algorithms_enabled'] and
            runtime['cudnn_deterministic'] == old_algorithms['cudnn_deterministic'] and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == old_algorithms['CUBLAS_WORKSPACE_CONFIG'], 'Exact historical algorithm policy differs')
        write_json('runtime.json', runtime)
        assets = {k: parent / value['path'] for k, value in c.ASSETS.items()}
        net, teacher, original, parent_teacher, census, factory = model.load_role_models(
            assets['initializer'], assets['full_coarse_checkpoint'], assets['hq_teacher'])
        perceptual = model.load_perceptual(assets['perceptual_features'].parent)
        base_model, architecture, route = model.historical_modules()
        write_json('factory.json', factory); write_json('parameter_census.json', census)
        write_json('codebook_compatibility.json', model.assert_shared_codebook(net, teacher))
        frozen_before = frozen_check(); write_json('initial_frozen_state.json', frozen_before)
        teacher_digest = model.state_digest(teacher); codebook_digest = model.state_digest(net.quantize)
        clean_cache = {}
        for rid, row in sorted(index['hq'].items()):
            budget(); reserve('teacher_forwards')
            codes = model.hq_targets(teacher, tensor(row['clean512']), codebook=net.quantize)
            indices = codes.indices.detach().cpu().numpy().copy(); features = codes.features.detach().cpu().numpy().copy()
            c.require(np.array_equal(indices, row['indices']) and np.array_equal(features, row['features']), 'Fresh100 HQ targets differ from retained exact cache: ' + rid)
            clean_cache[rid] = (indices, features)
            pins = {'indices': write_array('HQ_cache/' + rid + '/indices.npy', indices),
                    'features': write_array('HQ_cache/' + rid + '/features.npy', features)}
            write_json('HQ_cache/' + rid + '/record.json', {'reference_id': rid, 'sourcebindings': row['sourcebindings'],
                **pins, 'fresh_HQ_forward': True, 'exact_retained_arrays': True,
                'teacher_state_sha256': teacher_digest, 'restoration_codebook_state_sha256': codebook_digest})
            del codes
        c.require(len(clean_cache) == 100, 'Exact100 fresh HQ cache required')

        def codes_for(row):
            if row['role'] == 'normal': return None
            indices, features = clean_cache[row['reference_id']]
            return model.CleanCodes(tensor(indices), tensor(features))

        def capture(module, row):
            with capture_forward(module) as captured, torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
                native, logits, feature = module(model.prepare_input(tensor(row['input256'])), w=1., detach_16=True, adain=True)
                return {'native': native.detach().cpu().numpy().copy(), 'raw': as_rgb(model.delivered_rgb(native)),
                    'logits': logits.detach().cpu().numpy().copy(), 'latent': feature.detach().cpu().numpy().copy(),
                    'tokens': captured['tokens'].reshape(256).cpu().numpy().copy()}

        def save_capture(prefix, saved):
            return {key: write_array(prefix + '/' + key + '.npy', value) for key, value in saved.items() if key != 'terms'}

        # Both parity and route witnesses use the same declared u3 A normal/dark pair.
        witnesses = [next(r for r in index['train'] if r['id'] == identifier) for identifier in c.INITIALIZER_PARITY['cases']]
        c.require([r['slot'] for r in witnesses] == [1, 2] and all(r['update'] == 3 and r['parity']['image_eligible'] for r in witnesses), 'Declared eligible u3 witness pair required')
        model.set_training_mode(net); prereq_rng = rng(); prereq_modes = [(m, m.training) for m in net.modules()]
        initial_captures = {}; initial_totals = {}; witness_targets = {}; per_case_rng = {}
        for row in witnesses:
            budget(); restore_rng(prereq_rng)
            reserve('route_witness_behavior_teacher_forwards')
            behavior = model.behavior_targets(original if row['role'] == 'normal' else parent_teacher, tensor(row['input256']))
            witness_targets[row['id']] = behavior
            target_pins = {'delivered256': write_array('initializer_parity/' + row['id'] + '/teacher_delivered.npy', behavior.delivered256.cpu().numpy()),
                'logits': write_array('initializer_parity/' + row['id'] + '/teacher_logits.npy', behavior.logits.cpu().numpy())}
            per_case_rng[row['id']] = rng()
            reserve('initializer_parity_parent_forwards'); parent_capture = capture(parent_teacher, row)
            restore_rng(per_case_rng[row['id']])
            reserve('initializer_parity_candidate_forwards')
            if row['role'] == 'dark': reserve('initializer_parity_perceptual_forwards')
            with capture_forward(net) as captured, torch.autocast(device_type='cuda', enabled=False):
                total, terms, native = model.forward_role_terms(net, teacher, perceptual, tensor(row['input256']),
                    None if row['role'] == 'normal' else tensor(row['clean512']), role=row['role'], image_eligible=True,
                    codes=codes_for(row), behavior=behavior)
            initial_captures[row['id']] = {'native': native.detach().cpu().numpy().copy(),
                'raw': as_rgb(model.delivered_rgb(native)), 'terms': {k: float(v.detach().cpu()) for k, v in terms.items()},
                'logits': captured['logits'].detach().cpu().numpy().copy(),
                'latent': captured['latent'].detach().cpu().numpy().copy(),
                'tokens': captured['tokens'].reshape(256).cpu().numpy().copy()}
            initial_totals[row['id']] = float(total.detach())
            c.require(all(np.array_equal(initial_captures[row['id']][key], parent_capture[key]) for key in parent_capture),
                      'Actual candidate/full-coarse initialization native/RAW/logits/features/tokens differ')
            pins = save_capture('initializer_parity/' + row['id'], initial_captures[row['id']])
            parent_pins = save_capture('initializer_parity/' + row['id'] + '/parent', parent_capture)
            write_json('initializer_parity/' + row['id'] + '/record.json', {'id': row['id'], 'role': row['role'],
                'sourcebindings': row['sourcebindings'], 'targets': target_pins, 'captured': pins, 'parent_captured': parent_pins,
                'terms': initial_captures[row['id']]['terms'], 'total_loss': float(total.detach()),
                'candidate_parent_native_RAW_logits_features_tokens_exact': True})
            del total, terms, native, captured, parent_capture

        next_phase('witness'); save_rng('before', prereq_rng)
        trainable = {n: q for n, q in net.named_parameters() if q.requires_grad}
        names, values = tuple(trainable), tuple(trainable.values()); witness_records = []
        try:
            for row in witnesses:
                restore_rng(per_case_rng[row['id']])
                budget(); reserve('route_witness_forwards')
                if row['role'] == 'dark': reserve('route_witness_perceptual_forwards')
                with capture_forward(net) as captured:
                    with route.isolated_style(architecture) as route_capture, torch.autocast(device_type='cuda', enabled=False):
                        total, terms, native = model.forward_role_terms(net, teacher, perceptual, tensor(row['input256']),
                            None if row['role'] == 'normal' else tensor(row['clean512']), role=row['role'], image_eligible=True,
                            codes=codes_for(row), behavior=witness_targets[row['id']])
                route.validate_capture(route_capture)
                saved = {'native': native.detach().cpu().numpy().copy(), 'raw': as_rgb(model.delivered_rgb(native)),
                    'logits': captured['logits'].detach().cpu().numpy().copy(), 'latent': captured['latent'].detach().cpu().numpy().copy(),
                    'tokens': captured['tokens'].reshape(256).cpu().numpy().copy(),
                    'terms': {k: float(v.detach().cpu()) for k, v in terms.items()}}
                reference = initial_captures[row['id']]
                route.validate_forward_parity(reference, saved, np)
                c.require(float(total.detach()) == initial_totals[row['id']], 'Actual complete unscoped/scoped role total differs')
                pins = save_capture('gradient_route_witness/' + row['id'], saved)
                partitions = {}
                for label, selected, retain in [('image', model.IMAGE_TERMS, True), ('token_HQ', model.TOKEN_TERMS, False)]:
                    loss = sum(model.LOSS_WEIGHTS[k] * v for k, v in terms.items() if k in selected)
                    reserve('gradient_queries')
                    gradients = torch.autograd.grad(loss, values, allow_unused=True, create_graph=False, retain_graph=retain)
                    partitions[label] = route.summarize_gradients(names, gradients, np); del gradients
                connectivity = route.validate_witness_gradients('isolated', partitions['image'], partitions['token_HQ'])
                c.require(all(q.grad is None for q in net.parameters()), 'Witness must not populate candidate.grad')
                witness_records.append({'id': row['id'], 'role': row['role'], 'terms': saved['terms'], 'arrays': pins,
                    'route': route_capture, 'partitions': partitions, 'connectivity': connectivity,
                    'unscoped_full_role_forward_parity_exact': True, 'optimizer_constructed': False})
                if row['role'] == 'dark':
                    # Actual shared LPIPS source on a detached witness output, no extra CodeFormer invocation.
                    fixture = native.detach().clone().requires_grad_(True); truth = tensor(row['clean512']) * 2 - 1
                    reserve('perceptual_prerequisite_forwards'); ploss = perceptual(fixture, truth)
                    reserve('gradient_queries'); gradient, = torch.autograd.grad(ploss, fixture, create_graph=False, retain_graph=False)
                    c.require(torch.isfinite(gradient).all().item() and gradient.abs().max().item() > 0, 'LPIPS input gradient is detached/nonfinite/zero')
                    write_json('perceptual_prerequisite.json', {'complete': True, 'prediction_source': row['id'],
                        'forward_calls': 1, 'gradient_queries': 1, 'loss': float(ploss.detach()),
                        'input_gradient_max_abs': float(gradient.abs().max()), 'parameter_grads_None': all(q.grad is None for q in perceptual.parameters()),
                        'optimizer_constructed': False})
                    del fixture, truth, ploss, gradient
                del total, terms, native, captured
        finally:
            for module, mode in prereq_modes: module.training = mode
            restore_rng(prereq_rng)
        c.require(rng_equal(prereq_rng, rng()) and all(q.grad is None for q in net.parameters()), 'Witness RNG/None-gradient restore differs')
        save_rng('after', rng()); frozen_check()
        write_json('gradient_route_witness/report.json', {'complete': True, 'pass': True, 'cases': witness_records,
            'candidate_forwards': 2, 'component_gradient_queries': 4, 'all_gradient_queries_with_perceptual': 5,
            'CPU_CUDA_Python_NumPy_RNG_restored': True, 'all_parameter_grad_None': True, 'optimizer_constructed': False,
            'norms_are_source_bound_VM_readouts_not_independent_replay': True})
        initial_captures.clear(); initial_totals.clear(); witness_targets.clear()
        next_phase('initialization')
        groups = model.optimizer_groups(net)
        optimizer = torch.optim.Adam(groups, betas=tuple(c.TRAINING['betas']), eps=c.TRAINING['eps'], weight_decay=0.)
        c.require(not optimizer.state and len(optimizer.param_groups) == 2, 'Fresh empty Adam state required')
        committed_rng = rng(); committed_state = model.state_digest(net)
        checkpoint('snapshot_0000/checkpoint.pt')
        baselines = baseline_metrics()
        bank0 = snapshot(0)
        write_json('snapshot_0000/preservation_guards.json', {**preservation_guards(bank0, baselines['original']),
            'initial_known_failed_parent_recorded_without_abort': True})
        next_phase('training'); model.set_training_mode(net)
        coverage = {n: {'gradient_observations': 0, 'nonzero_observations': 0, 'max_abs': 0.} for n in trainable}
        for update in range(1, 51):
            budget(); step_started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups: group['lr'] = learning_rate(update, group['base_lr'])
            history = []
            batch_rows = index['train'][(update - 1)*4:update*4]
            c.require([r['slot'] for r in batch_rows] == [1, 2, 3, 4] and all(r['update'] == update for r in batch_rows), 'Fixed update/slot order differs')
            for row in batch_rows:
                budget(); prefix = 'TRAIN/' + row['id']
                reserve('training_input_writes'); input_pin = write_png(prefix + '/input.png', row['input_rgb8'])
                key = 'original_behavior_teacher_forwards' if row['role'] == 'normal' else 'parent_behavior_teacher_forwards'
                reserve(key)
                behavior = model.behavior_targets(original if row['role'] == 'normal' else parent_teacher, tensor(row['input256']))
                target_pins = {'delivered256': write_array(prefix + '/teacher_delivered.npy', behavior.delivered256.detach().cpu().numpy()),
                    'logits': write_array(prefix + '/teacher_logits.npy', behavior.logits.detach().cpu().numpy())}
                reserve('candidate_training_forwards')
                eligible = row['parity']['image_eligible']
                if row['role'] == 'dark' and eligible: reserve('perceptual_training_forwards')
                with capture_forward(net) as captured:
                    with route.isolated_style(architecture) as route_capture, torch.autocast(device_type='cuda', enabled=False):
                        total, terms, native = model.forward_role_terms(net, teacher, perceptual, tensor(row['input256']),
                            None if row['role'] == 'normal' else tensor(row['clean512']), role=row['role'], image_eligible=eligible,
                            codes=codes_for(row), behavior=behavior)
                route.validate_capture(route_capture)
                raw = as_rgb(model.delivered_rgb(native))
                native_rgb = as_rgb((native.detach().float() + 1) / 2)
                tokens = captured['tokens'].reshape(256).cpu().numpy().copy()
                pins = {'native': write_array(prefix + '/candidate_native.npy', native_rgb),
                    'raw': write_array(prefix + '/candidate_raw.npy', raw), 'tokens': write_array(prefix + '/tokens.npy', tokens)}
                term_values = {k: float(v.detach().cpu()) for k, v in terms.items()}
                item = {'id': row['id'], 'update': update, 'slot': row['slot'], 'role': row['role'],
                    'reference_id': row['reference_id'], 'sourcebindings': row['sourcebindings'], 'input': input_pin,
                    'teacher_kind': behavior.teacher_kind, 'teacher_targets': target_pins, 'candidate_arrays': pins,
                    'candidate_logits_array_sha256': array_digest(captured['logits'].detach().cpu().numpy()),
                    'terms': term_values, 'total_loss': float(total.detach()), 'actual_accumulation_divisor': 4,
                    'image_eligible': eligible, 'normal_clean_supervision_called': False,
                    'route': route_capture, 'record_captured_before_optimizer_step': True,
                    'optimizer_committed_at_record_time': False}
                write_json(prefix + '/record.json', item)
                reserve('backward_calls'); (total / 4).backward(); pending += 1
                history.append(item)
                c.require(all(q.grad is None for module in (teacher, original, parent_teacher, perceptual) for q in module.parameters()), 'Frozen teacher has gradients')
                del total, terms, native, captured, behavior, raw, native_rgb
            c.require(pending == 4, 'Exactly4 accumulated microbatches required')
            c.require(all(parameter.grad is not None for parameter in trainable.values()),
                      'Every350 trainable tensor must be connected by the complete four-slot objective; connected zeros are allowed')
            for name, parameter in trainable.items():
                gradient = parameter.grad
                if gradient is not None:
                    c.require(torch.isfinite(gradient).all().item(), 'Nonfinite accumulated candidate gradient')
                    maximum = float(gradient.abs().max())
                    coverage[name]['gradient_observations'] += 1; coverage[name]['nonzero_observations'] += maximum > 0
                    coverage[name]['max_abs'] = max(coverage[name]['max_abs'], maximum)
            c.require(any(q.grad is not None and q.grad.abs().max().item() > 0 for q in trainable.values()), 'All candidate gradients absent/zero')
            write_json('history/update_%04d_attempt.json' % update, {'update': update, 'completed_before': completed,
                'observations': history, 'group_lrs': {g['name']: g['lr'] for g in optimizer.param_groups}, 'optimizer_committed': False})
            reserve('optimizer_step_attempts'); inflight = True
            optimizer.step()
            c.require(all(torch.isfinite(q).all().item() for q in net.parameters()), 'Nonfinite candidate after Adam')
            completed = update; pending = 0; inflight = False; optimizer.zero_grad(set_to_none=True)
            committed_rng = rng(); committed_state = model.state_digest(net)
            training_seconds += time.monotonic() - step_started; step_started = None
            finite_counts(counts, completed); budget()
            write_json('history/update_%04d_committed.json' % update, {'update': update, 'completed_training_updates': completed,
                'observations': [{k: x[k] for k in ('id', 'role', 'terms', 'total_loss')} for x in history],
                'group_lrs': {g['name']: g['lr'] for g in optimizer.param_groups}, 'model_tensor_sha256': committed_state,
                'retained_data_selection': index['train_updates'][update - 1]['TRAIN_selection'],
                'partial_microbatches': 0, 'optimizer_inflight': False, 'optimizer_committed': True, 'training_seconds': training_seconds})
            if update == 1 or update % 10 == 0:
                frozen_check(); print(json.dumps({'committed_training_updates': completed, 'of': 50,
                    'mean_microbatch_loss': sum(x['total_loss'] for x in history) / 4}), flush=True)
        net.eval(); frozen_check()
        write_json('gradient_coverage.json', {'complete': True, 'tensors': coverage,
            '350_trainable_tensors': len(coverage) == 350, '50_actual_Adam_updates': completed == 50})
        bank50 = snapshot(50)
        guards = preservation_guards(bank50, baselines['original']); dark = dark_retention(bank50, baselines['base'])
        write_json('preservation_guards.json', guards); write_json('dark_retention.json', dark)
        write_json('final_frozen_state.json', {'complete': True, 'digests': frozen_check(), 'same_as_initial': frozen_check() == frozen_before})
        finite_counts(counts, completed, terminal=True)
        checkpoint('snapshot_0050/checkpoint.pt')  # Full final counters/state retained before any quality decision.
        stop_checkpoint('finite_complete' if guards['pass'] and dark['pass'] else 'quality_gate_failure_after_actual50')
        c.require(guards['pass'] and dark['pass'], 'Preservation or dark-retention quality gate failed; actual trained50state retained')
        write_json('run_results.json', {**terminal(), 'quality_gates_pass': True, 'independent_review_and_user_acceptance_pending': True})
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        reason = type(exc).__name__ + ': ' + str(exc); checkpoint_error = None
        try:
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            if net is not None: net.eval()
            if frozen_before is not None: frozen_check()
            if not (out / 'stopped_checkpoint_alias.json').exists() and not (out / 'stopped_checkpoint.pt').exists():
                stop_checkpoint(reason)
        except BaseException as cp_error:
            checkpoint_error = type(cp_error).__name__ + ': ' + str(cp_error)
        write_json('failure.json', {**terminal(reason), 'failure': True, 'traceback': traceback.format_exc(),
            'stopped_checkpoint_error': checkpoint_error, 'optimizer_state_uncertain': inflight,
            'pending_microbatches_not_committed': pending, 'trained_candidate_not_qualified': True}, stop_safe=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--protocol-sha', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ('--verify-transfer', '--run', '--export'): action.add_argument(flag, action='store_true')
    args = parser.parse_args(argv); root = c.manual_root(args.root)
    from . import transport
    if args.verify_transfer:
        value = transport.verify_transfer(root, args.protocol_sha)
        print(json.dumps({'complete': value['complete'], 'protocol_sha256': args.protocol_sha,
            'dependencies': len(value['protocol']['dependencies']), 'actual_neural_calls': 0}), flush=True)
    elif args.export:
        print(json.dumps(transport.export_results(root, args.protocol_sha)), flush=True)
    else:
        run(root, args.protocol_sha)


if __name__ == '__main__': main()


