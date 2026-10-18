"""Manual finite frozen-response measurement; no optimization or checkpoint writes."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

from . import contract as c


def counts_valid(counts, terminal=False):
    c.require(set(counts) == set(c.COUNTS), 'Exact finite operation keys required')
    c.require(all(type(counts[k]) is int and 0 <= counts[k] <= cap for k, cap in c.COUNTS.items()),
              'Declared operation reservation cap exceeded')
    if terminal:
        c.require(counts == c.COUNTS and sum(counts.values()) == 656, 'Every finite measurement call must close')
    return True


def idle_conflicts(rows, own_pid, root, protocol_sha):
    from scripts.supervise_codeformer_fixed_state_response_v1_vm import competing_commands
    table = {pid: (ppid, cmd) for pid, ppid, cmd in rows}
    c.require(own_pid in table and len(rows) == len(table), 'Unique full own-process inventory required')
    accepted = {own_pid}; parent = table[own_pid][0]
    if parent in table:
        ppid, cmd = table[parent]; words = shlex.split(cmd)
        if words and Path(words[0]).name == 'timeout':
            c.require(words[1:] == ['--signal=TERM', '--kill-after=30', '900', sys.executable,
                '-B', '-u', '-m', c.MODULE, '--root', str(root), '--protocol-sha', protocol_sha, '--run'],
                'Current timeout ancestor differs from exact frozen launch')
            accepted.add(parent); parent = ppid
        if parent in table:
            _, cmd = table[parent]; words = shlex.split(cmd)
            script = str(root / 'scripts/supervise_codeformer_fixed_state_response_v1_vm.py')
            if script in words:
                c.require(words == [sys.executable, '-B', '-u', script, '--root', str(root), '--protocol-sha', protocol_sha],
                          'Exact absolute supervisor ancestry required')
                accepted.add(parent)
    return competing_commands([(p, cmd) for p, _, cmd in rows if p not in accepted], own_pid), sorted(accepted)


def require_idle(root, protocol_sha):
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=20)
    c.require(not gpu.stdout.strip(), 'GPU is occupied; preserve existing workloads')
    ps = subprocess.run(['ps', '-eo', 'pid=,ppid=,args='], check=True, capture_output=True, text=True, timeout=20)
    rows = []
    for line in ps.stdout.splitlines():
        fields = line.strip().split(None, 2)
        c.require(len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit(), 'Cannot classify full process inventory')
        rows.append((int(fields[0]), int(fields[1]), fields[2]))
    conflicts, accepted = idle_conflicts(rows, os.getpid(), root, protocol_sha)
    c.require(not conflicts, 'Competing workload; preserve it: ' + json.dumps(conflicts))
    return {'complete': True, 'GPU_processes': [], 'full_process_inventory': rows,
            'own_launch_ancestry': accepted, 'other_processes_killed': False}


def summarize(records):
    """Continuous fixed-state responses; no new quality threshold or selected best arm."""
    grouped = {}
    for row in records:
        key = (row['cohort'], row['role'], row['head'])
        grouped.setdefault(key, []).append(row)
    result = []
    for (cohort, role, head), rows in sorted(grouped.items()):
        fields = {name: [r[name] for r in rows] for name in ('delta', 'loss', 'signed_residual')}
        result.append({'cohort': cohort, 'role': role, 'head': head, 'cases': len(rows),
            **{name: {'minimum': min(values), 'mean': math.fsum(values) / len(values), 'maximum': max(values)}
               for name, values in fields.items()},
            'negative_offsets': sum(r['delta'] < 0 for r in rows),
            'positive_offsets': sum(r['delta'] > 0 for r in rows),
            'zero_offsets': sum(r['delta'] == 0 for r in rows)})
    by_id = {}
    for row in records:
        by_id.setdefault(row['id'], {})[row['head']] = row
    pairs = []
    for identifier, rows in sorted(by_id.items()):
        if set(rows) != {'initial', 'final'}:
            continue
        a, b = rows['initial'], rows['final']
        pairs.append({'id': identifier, 'reference_id': a['reference_id'], 'cohort': a['cohort'], 'role': a['role'],
            'initial_loss': a['loss'], 'final_loss': b['loss'], 'loss_change': b['loss'] - a['loss'],
            'initial_signed_residual': a['signed_residual'], 'final_signed_residual': b['signed_residual'],
            'final_delta': b['delta'], 'target_mean_y': a['target_mean_y'],
            'desired_mean_change': a['target_mean_y'] - a['mean_y']})
    return {'complete': True, 'groups': result, 'paired_fixed_state_comparisons': pairs,
            'new_acceptance_threshold_or_selection': False, 'model_qualified': False,
            'scope': 'Fitted versus held-out TRAIN reference responses; not independent FINAL or native CCTV evidence'}


def run(root, protocol_sha):
    root = c.manual_root(root)
    import resource  # Linux-only; pure CPU audit imports remain framework-free on Windows.
    from . import transport
    c.require(not (root / c.LOCK_NAME).exists() and not (root / c.OUTPUTS).exists(),
              'Preserve previous or partial measurement; no relaunch')
    start = time.monotonic()
    verified = transport.verify_transfer(root, protocol_sha)
    parents = c.resolve_parents(root); p = verified['protocol']
    idle = require_idle(root, protocol_sha)
    c.require(shutil.disk_usage(root).free >= c.LIMITS['initial_free_bytes_after_install'], '8GiB free after install required')
    c.write_new_json(root / c.LOCK_NAME, {'pid': os.getpid(), 'protocol_sha256': protocol_sha, 'manual_tmux': True})
    out = root / c.OUTPUTS; out.mkdir(parents=True)
    own = {}; counts = {k: 0 for k in c.COUNTS}; last_reserved = None; finished_calls = 0
    torch = None; frozen = None; initial_state = None
    records = []; minimum_free = shutil.disk_usage(root).free; peak_rss = peak_cuda = 0
    phase = 'data'; phase_started = start; phase_seconds = {}; terminal_written = False
    # New static inputs, source, protocol and lock are counted with every result.
    initial_files = transport._walk(root)
    static_sizes = {name: path.stat().st_size for name, path in initial_files.items()
                    if not name.startswith('outputs/supervision/')}
    metadata_static = sum(size for name, size in static_sizes.items() if Path(name).suffix != '.png')

    def external_sizes():
        directory = root / 'outputs/supervision'
        return {path.name: path.stat().st_size for path in directory.iterdir() if path.is_file()} if directory.exists() else {}

    def usage():
        external = external_sizes()
        return (sum(static_sizes.values()) + sum(v['bytes'] for v in own.values()) + sum(external.values()),
                len(static_sizes) + len(own) + len(external),
                metadata_static + sum(v['bytes'] for v in own.values() if v['metadata']) + sum(external.values()))

    def budget(extra=0, files=0, metadata=0, stop_safe=False):
        nonlocal minimum_free, peak_rss, peak_cuda
        total, count, meta = usage()
        free = min(shutil.disk_usage(root).free, shutil.disk_usage(Path.home()).free)
        minimum_free = min(minimum_free, free)
        peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if torch is not None:
            peak_cuda = max(peak_cuda, torch.cuda.max_memory_reserved(), torch.cuda.memory_reserved())
        c.require(total + extra <= c.LIMITS['result_payload_bytes'] and count + files <= c.LIMITS['regular_files'] and
                  meta + metadata <= c.LIMITS['metadata_bytes'], 'Whole-root result/file/metadata admission cap reached')
        c.require(free >= c.LIMITS['reserve_free_bytes'] + extra, '4GiB disk reserve reached')
        if not stop_safe:
            c.require(time.monotonic() - start <= c.LIMITS['worker_seconds'], '900second measurement stop reached')
            c.require(time.monotonic() - phase_started <= c.LIMITS[phase + '_seconds'], 'Declared phase stop reached: ' + phase)
            c.require(peak_rss <= c.LIMITS['process_RSS_bytes'] and peak_cuda <= c.LIMITS['CUDA_reserved_bytes'],
                      'RSS/CUDA resource stop reached')

    def write(relative, payload, metadata, stop_safe=False):
        path = c.safe_path(out, relative, must_exist=False)
        c.require(not path.exists() and relative not in own, 'Preserve existing output')
        budget(len(payload), 1, len(payload) if metadata else 0, stop_safe)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        own[relative] = {'bytes': len(payload), 'metadata': metadata}
        return {'path': path.relative_to(root).as_posix(), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}

    def write_json(name, value, stop_safe=False):
        payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
        c.require(len(payload) <= c.LIMITS['one_JSON_bytes'], 'One JSON record exceeds declared cap')
        return write(name, payload, True, stop_safe)

    def reserve(name):
        nonlocal last_reserved
        budget(); counts_valid(counts)
        c.require(name in counts and counts[name] < c.COUNTS[name], 'No undeclared or zero-cap operation permitted')
        counts[name] += 1
        last_reserved = {'sequence': sum(counts.values()), 'name': name, 'counts': dict(counts), 'call_completion_proved': False}
        write_json('trace/' + str(last_reserved['sequence']).zfill(4) + '.json', last_reserved)

    def terminal(measurement_complete, reason=None):
        budget(stop_safe=True)
        total, count, meta = usage()
        return {'complete': True, 'measurement_complete': measurement_complete, 'protocol_sha256': protocol_sha,
            'counts': dict(counts), 'finished_neural_calls': finished_calls, 'last_reserved_call': last_reserved,
            'head_updates_this_run': 0, 'restorer_updates_this_run': 0, 'gradient_queries': 0, 'backward_calls': 0,
            'optimizer_constructions': 0, 'optimizer_steps': 0, 'committed_training_updates': 0,
            'reason': reason, 'seconds': time.monotonic() - start,
            'phase_seconds': {**phase_seconds, phase: time.monotonic() - phase_started},
            'resources': {'peak_RSS_bytes': peak_rss, 'peak_CUDA_reserved_bytes': peak_cuda,
                'minimum_sampled_free_bytes': minimum_free, 'free_space_is_sampled': True,
                'root_payload_bytes_before_terminal': total, 'regular_files_before_terminal': count,
                'metadata_bytes_before_terminal': meta},
            'frozen_state_verified': frozen, 'new_checkpoints_written': False, 'model_qualified': False,
            'app_promoted': False, 'automatic_retry': False}

    def timed_out(signum, frame):
        raise TimeoutError('Finite overall or ' + phase + ' phase alarm reached')

    def next_phase(name):
        nonlocal phase, phase_started
        budget()
        phase_seconds[phase] = time.monotonic() - phase_started
        phase, phase_started = name, time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(c.LIMITS[name + '_seconds'],
            c.LIMITS['worker_seconds'] - (time.monotonic() - start))))

    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers:
        signal.signal(sig, timed_out)
    signal.setitimer(signal.ITIMER_REAL, max(.001, min(c.LIMITS['data_seconds'], c.LIMITS['worker_seconds']) - (time.monotonic() - start)))
    try:
        write_json('transfer_verified.json', verified); write_json('idle_verified.json', idle)
        from . import data, model
        index = data.load_index(root, p, parents)
        c.require(len(index['fitted']) == 200 and len(index['heldout']) == 64, 'Exact fixed index required')
        write_json('input_index.json', index['binding_receipt'])
        next_phase('initialization')
        torch = model.activate_runtime(root, protocol_sha)
        import numpy as np
        from PIL import Image, __version__ as pillow_version
        import cv2
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        expected_runtime = c.read_json(c.safe_path(parents['tone_training'], 'outputs/tone_control_training_v1/runtime.json'))
        runtime = {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'Pillow': pillow_version,
            'OpenCV': cv2.__version__, 'CUDA': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0))}
        c.require(all(runtime[k] == expected_runtime[k] for k in runtime), 'Recorded parent runtime versions/device differ')
        pip = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True, timeout=30)
        c.require(sorted(pip.stdout.splitlines()) == sorted(expected_runtime['pip_freeze'].splitlines()),
                  'Installed package line set differs from fitted-state runtime')
        runtime.update(pip_freeze=pip.stdout, TF32=False, autocast=False, cudnn_benchmark=False,
            deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic, universal_determinism_claim=False)
        c.require(runtime['deterministic_algorithms_enabled'] == expected_runtime['deterministic_algorithms_enabled'] and
                  runtime['cudnn_deterministic'] == expected_runtime['cudnn_deterministic'], 'Determinism policy changed')
        write_json('runtime.json', runtime)
        # Extend only verified historical import roots; never activate the old VM root.
        import research
        old_research = str(parents['adain'] / 'research')
        c.require(list(research.__path__) == [str(root / 'research')], 'Unexpected preloaded research namespace')
        research.__path__.append(old_research)
        sys.path.append(str(parents['adain']))
        paths = {name: c.safe_path(parents[pin['root']], pin['path']) for name, pin in c.ASSETS.items()}
        original, base, heads, factory = model.load_frozen_parents_and_heads(
            paths['initializer'], paths['initial_composite'], paths['stopped_composite'])
        initial_state = model.frozen_state_receipt(original, base, heads)
        write_json('factory.json', factory); write_json('initial_state.json', initial_state)
        next_phase('measurement')

        def write_array(name, value):
            c.require(value.dtype == np.float32 and value.shape == (256, 256, 3) and np.isfinite(value).all() and
                      value.min() >= 0 and value.max() <= 1, 'Finite bounded FP32 RAW256 required')
            buffer = io.BytesIO(); np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
            return write(name, buffer.getvalue(), False)

        def write_png(name, raw, row):
            pixels = np.floor(np.clip(raw, 0, 1) * np.float32(255)).astype(np.uint8)
            pixels[~row['observed']] = row['input_rgb8'][~row['observed']]
            buffer = io.BytesIO(); Image.fromarray(pixels, mode='RGB').save(buffer, format='PNG', compress_level=6)
            c.require(buffer.tell() <= c.LIMITS['one256PNG_bytes'], 'Exact PNG per-file cap exceeded')
            return write(name, buffer.getvalue(), False)

        for cohort in ('fitted', 'heldout'):
            for number, row in enumerate(index[cohort], 1):
                budget()
                if cohort == 'heldout':
                    reserve('CodeFormer_new_original'); o = model.infer_rgb(original, row['input256']); finished_calls += 1
                    reserve('CodeFormer_new_full_coarse'); b = model.infer_rgb(base, row['input256']); finished_calls += 1
                    parent_outputs = {'original_raw': write_array('new/' + row['id'] + '/original_raw.npy', o),
                                      'base_raw': write_array('new/' + row['id'] + '/base_raw.npy', b)}
                    target = model.mean_target(o if row['role'] == 'normal' else b, row['observed'])
                    target_policy = 'fresh_frozen_original_mean' if row['role'] == 'normal' else 'fresh_frozen_base_mean'
                else:
                    o, b, target = row['original_raw'], row['base_raw'], row['target']
                    parent_outputs = None
                    target_policy = 'exact_retained_fitted_target_mean'
                for label, head_name in (('initial', 'head0000'), ('final', 'head0050')):
                    key = 'controller_' + cohort + '_' + label
                    reserve(key)
                    response = model.head_response(heads[head_name], row['features'], b,
                        row['input256'], row['observed'], target)
                    finished_calls += 1
                    raw = response.pop('raw')
                    c.require(np.array_equal(raw[~row['observed']], row['input256'][~row['observed']]),
                              'Protected input pixels changed')
                    if label == 'initial':
                        expected = b.copy(); expected[~row['observed']] = row['input256'][~row['observed']]
                        c.require(response['delta'] == 0 and np.array_equal(raw, expected), 'Actual initial zero-head parity differs')
                    destination = label + '/' + row['id']
                    output_raw = write_array(destination + '/raw.npy', raw)
                    output_png = write_png(destination + '/native.png', raw, row)
                    record = {'id': row['id'], 'cohort': cohort, 'role': row['role'], 'head': label,
                        'reference_id': row['reference_id'], 'source_sha256': row['source_sha256'],
                        'pair_id': row['pair_id'], 'sourcebindings': row['sourcebindings'],
                        'features': row['features'].tolist(), 'target_mean_y': float(target), 'target_policy': target_policy,
                        **response, 'signed_residual': response['mean_y'] - float(target),
                        'original_mean_y': model.mean_target(o, row['observed']), 'base_mean_y': model.mean_target(b, row['observed']),
                        'RAW': output_raw, 'PNG': output_png, 'new_parent_RAW': parent_outputs,
                        'fitted_parent_arrays_reused_without_new_CodeFormer_calls': cohort == 'fitted',
                        'support_is_not_covering_mask': True, 'clean_pixels_used_for_target': False,
                        'independent_FINAL_or_native_evaluation': False}
                    write_json(destination + '/record.json', record); records.append(record)
                if number % 25 == 0 or number == len(index[cohort]):
                    print(json.dumps({'cohort': cohort, 'rows': number, 'of': len(index[cohort]),
                                      'neural_calls_completed': finished_calls}), flush=True)
        counts_valid(counts, terminal=True)
        c.require(finished_calls == 656 and len(records) == 528, 'Exact completed measurement census required')
        frozen = model.frozen_state_receipt(original, base, heads)
        c.require(c.equal(frozen, initial_state), 'Frozen state changed during measurement')
        write_json('final_state.json', frozen)
        write_json('response_summary.json', summarize(records))
        write_json('run_results.json', terminal(True))
        terminal_written = True
    except BaseException as exc:
        # Keep every partial output. No optimizer state or model checkpoint is created.
        signal.setitimer(signal.ITIMER_REAL, 0)
        if terminal_written or (out / 'run_results.json').exists():
            raise  # A late interrupt cannot manufacture conflicting terminal records.
        if torch is not None and initial_state is not None:
            try:
                frozen = model.frozen_state_receipt(original, base, heads)
            except BaseException as check_error:
                frozen = {'complete': False, 'error': str(check_error)}
        failure = terminal(False, type(exc).__name__ + ': ' + str(exc))
        failure.update(failure=True, traceback=traceback.format_exc(), training_success_not_implied=True)
        write_json('failure.json', failure, stop_safe=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ('--verify-transfer', '--run', '--export'): action.add_argument(flag, action='store_true')
    args = parser.parse_args(argv); root = c.manual_root(args.root)
    from . import transport
    if args.verify_transfer:
        value = transport.verify_transfer(root, args.protocol_sha)
        print(json.dumps({'complete': value['complete'], 'dependencies': value['dependencies_verified'],
                          'neural_calls': 0, 'training_updates': 0}), flush=True)
    elif args.export:
        print(json.dumps(transport.export_results(root, args.protocol_sha)), flush=True)
    else:
        run(root, args.protocol_sha)


if __name__ == '__main__': main()
