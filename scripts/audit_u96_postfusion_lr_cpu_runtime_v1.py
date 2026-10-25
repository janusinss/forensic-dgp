"""HELD four-case CPU inference witness. No action without explicit execute flag.

This is inference only. CPU/L4 numerical differences are measurements, not a
new training tolerance or broader quality screen. No optimizer is constructed.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

import export_u96_postfusion_lr_inference_v1 as E

OUT = E.ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_cpu_runtime'


class RuntimeLedger:
    """Durable reservations and returns; interrupted forwards stay unknown."""
    def __init__(self, directory):
        self.directory = Path(directory); self.lock = threading.Lock()
        self.phase = 'STARTED_BEFORE_MODEL_CONSTRUCTION'
        self.counters = {'model_construction_attempts': 0, 'model_construction_returns': 0,
                         'forward_attempts': 0, 'forward_returns': 0,
                         'backward_calls': 0, 'optimizer_constructions': 0, 'optimizer_steps': 0}
        self.directory.mkdir()
        E.write_json(self.directory / 'started_v1.json', {'complete': True,
            'UTC': datetime.now(timezone.utc).isoformat(), 'purpose': 'Four held CPU inference witnesses only',
            'model_qualified': False, 'main_app_default_changed': False})
        self.update(self.phase)

    def snapshot(self):
        counters = dict(self.counters)
        unknown = (counters['forward_attempts'] != counters['forward_returns'] or
                   counters['model_construction_attempts'] != counters['model_construction_returns'])
        return {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
                'phase': self.phase, 'counters': counters,
                'reserved_model_or_forward_outcome_unknown': unknown,
                'model_qualified': False, 'main_app_default_changed': False}

    def _save(self):
        pending = self.directory / 'status.pending.json'
        raw = (json.dumps(self.snapshot(), indent=2, allow_nan=False) + '\n').encode()
        with pending.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(pending, self.directory / 'status.json')

    def update(self, phase, increment=None):
        with self.lock:
            if increment is not None:
                E.need(increment in self.counters, 'Known ledger counter required')
                self.counters[increment] += 1
                E.need(self.counters['forward_attempts'] <= 4 and self.counters['model_construction_attempts'] <= 1,
                       'Four-forward/one-construction runtime cap')
            self.phase = phase; self._save()

    def fail(self, kind, error):
        with self.lock:
            self.phase = kind
            path = self.directory / 'failure_v1.json'
            if not path.exists():
                E.write_json(path, {**self.snapshot(), 'status': kind,
                                   'error_type': type(error).__name__, 'error': str(error)[:1500]})
            self._save()


def decode_raw(raw):
    import numpy as np
    stream = io.BytesIO(raw); version = np.lib.format.read_magic(stream)
    E.need(version in ((1, 0), (2, 0)), 'Known NPY1/2 required')
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream, max_header_size=4096)
    E.need(shape == (256, 256, 3) and dtype == np.dtype('float32') and not fortran
           and stream.tell() + 256 * 256 * 3 * 4 == len(raw), 'Exact bounded HWC256 FP32 RAW')
    value = np.frombuffer(raw, dtype=np.float32, offset=stream.tell()).reshape(shape)
    E.need(np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1, 'Finite clipped L4 RAW')
    return value


def difference(cpu, saved):
    import numpy as np
    E.need(cpu.shape == saved.shape == (256, 256, 3), 'Same delivered geometry')
    delta = cpu.astype(np.float64) - saved.astype(np.float64); absolute = np.abs(delta)
    png_cpu = np.floor(cpu * np.float32(255)).astype(np.uint8)
    png_saved = np.floor(saved * np.float32(255)).astype(np.uint8)
    return {'mean_absolute_RAW_difference': float(absolute.mean()), 'maximum_absolute_RAW_difference': float(absolute.max()),
            'RAW_RMSE': float(np.sqrt(np.square(delta).mean())), 'signed_RAW_mean_difference': float(delta.mean()),
            'RAW_components_different': int(np.count_nonzero(cpu != saved)),
            'total_RGB_components': int(cpu.size), 'floored_PNG_components_different': int(np.count_nonzero(png_cpu != png_saved)),
            'floored_PNG_pixels_different': int(np.any(png_cpu != png_saved, axis=2).sum()),
            'no_bitwise_parity_or_quality_tolerance_invented': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-reviewed-cpu-runtime', action='store_true')
    parser.add_argument('--source-review', type=Path, required=True)
    parser.add_argument('--export-receipt', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    parser.add_argument('--r2-artifact', type=Path, required=True)
    parser.add_argument('--u96-artifact', type=Path, required=True)
    args = parser.parse_args()
    E.need(args.execute_reviewed_cpu_runtime, 'Runtime HOLD: root must explicitly execute the reviewed four-case plan')
    E.reviewed_source(args.source_review)
    export = E.read(args.export_receipt)
    E.need(export['complete'] and export['inference_export_pass'] and export['CPU_runtime_executed'] is False,
           'Exact reviewed inference export prerequisite')
    for row in export['evidence']: E.require_pin(row)
    for key in ('head_file', 'inference_source', 'protocol', 'runtime_plan'): E.require_pin(export[key])
    E.need(export['inference_source']['sha256'] == E.INFERENCE_SHA and export['runtime_plan']['sha256'] == args.plan_sha
           and export['canonical_head_sha256'] == E.HEAD_CANONICAL and export['archive']['sha256'] == E.ARCHIVE_SHA,
           'Exact runtime source/plan/head/archive pins required')
    E.visual_closure(E.ROOT / export['root_visual_closure']['path'])
    plan = E.read(E.ROOT / export['runtime_plan']['path'])
    E.need(plan['runtime_executed'] is False and plan['maximum_cases'] == 4 and len(plan['cases']) == 4
           and [x['case']['role'] for x in plan['cases']] == E.ROLES and plan['CPU_threads'] == 4,
           'Exact four-role CPU plan only')
    protocol = E.read(E.STAGE / 'protocol.json'); audit = E.read(E.RETURN / 'saved_pixel_audit_v1.json')
    E.need(plan['cases'] == E.four_cases(protocol, audit), 'Plan cases must equal unchanged first monitor photograph')
    E.need(not OUT.exists(), 'CPU witness output immutable and must not preexist')
    selected = {row['saved_L4_RAW']['path'] for row in plan['cases']}
    index = E.read(E.RETURN / 'archive_index_v1.json')
    raw_bytes, transport = E.stream_selected(E.ARCHIVE, index['members'], selected)
    before = [E.pin(args.r2_artifact), E.pin(args.u96_artifact), export['head_file']]
    ledger = RuntimeLedger(OUT)
    # Numerical imports begin only after review/export/visual/transport guards.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    original_threads = None; results = []; started = time.monotonic()
    def timeout():
        try:
            # Do not let an interrupted atomic status writer block hard exit.
            if ledger.lock.acquire(blocking=False):
                try:
                    ledger.phase = 'RUNTIME_WATCHDOG_TIMEOUT'
                    E.write_json(OUT / 'failure_v1.json', {**ledger.snapshot(),
                        'status': 'RUNTIME_WATCHDOG_TIMEOUT', 'error': '600 second CPU witness cap'})
                    ledger._save()
                finally: ledger.lock.release()
            else:
                E.write_json(OUT / 'failure_watchdog_v1.json', {**ledger.snapshot(),
                    'status': 'RUNTIME_WATCHDOG_TIMEOUT_DURING_STATUS_WRITE',
                    'reserved_model_or_forward_outcome_unknown': True,
                    'error': '600 second cap; preserve durable/pending status without guessing outcome'})
        finally: os._exit(124)
    watchdog = threading.Timer(600, timeout); watchdog.daemon = True; watchdog.start()
    try:
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        E.need(not torch.cuda.is_initialized() and torch.get_default_dtype() == torch.float32, 'CPU FP32 runtime only')
        original_threads = torch.get_num_threads(); torch.set_num_threads(4)
        old_rng = torch.random.get_rng_state().clone()
        spec = importlib.util.spec_from_file_location('reviewed_terminal_lr_inference', E.INFERENCE)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        ledger.update('MODEL_CONSTRUCTION_RESERVED', 'model_construction_attempts')
        net = module.load_frozen_cpu_restorer(args.r2_artifact, args.u96_artifact,
              E.ROOT / export['head_file']['path'], inference_source_sha256=E.INFERENCE_SHA)
        ledger.update('MODEL_CONSTRUCTION_RETURNED', 'model_construction_returns')
        E.need(torch.equal(old_rng, torch.random.get_rng_state()), 'Construction must preserve caller CPU RNG')
        initial = net.validate_invariants(); initial_state = dict(module.source_contract()['canonical_states'])
        for row in plan['cases']:
            case = row['case']; input_path = E.STAGE / case['input']['path']
            actual = E.pin(input_path)
            E.need({k: actual[k] for k in ('bytes', 'sha256')} == {k: case['input'][k] for k in ('bytes', 'sha256')}, 'Exact sealed input file')
            with Image.open(input_path) as image:
                E.need(image.mode == 'RGB' and image.size == (256, 256) and getattr(image, 'n_frames', 1) == 1, 'RGB256 singleton input')
                rgb = np.asarray(image).copy()
            E.need(hashlib.sha256(rgb.tobytes()).hexdigest() == case['input_RGB8_sha256'], 'Input RGB pin')
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).float() / 255
            with torch.no_grad():
                sample = F.interpolate(tensor, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
                ledger.update('FORWARD_RESERVED_' + case['id'], 'forward_attempts')
                triple = net(sample, w=1., detach_16=True, adain=True)
                ledger.update('FORWARD_RETURNED_' + case['id'], 'forward_returns')
                delivered = F.interpolate(((triple[0].float() + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear', align_corners=False)
                cpu = delivered[0].permute(1, 2, 0).contiguous().numpy().copy()
            E.need(cpu.dtype == np.float32 and np.isfinite(cpu).all() and cpu.min() >= 0 and cpu.max() <= 1, 'Finite clipped CPU RAW')
            saved = decode_raw(raw_bytes[row['saved_L4_RAW']['path']])
            differences = difference(cpu, saved)
            raw_path = OUT / (case['id'] + '_CPU_raw.npy'); png_path = OUT / (case['id'] + '_CPU.png')
            with raw_path.open('xb') as stream: np.save(stream, cpu, allow_pickle=False)
            Image.fromarray(np.floor(cpu * np.float32(255)).astype(np.uint8)).save(png_path)
            E.need(png_path.stat().st_size + raw_path.stat().st_size < 2 * 1024**2, 'Bounded per-case output')
            results.append({'id': case['id'], 'reference_id': case['reference_id'], 'role': case['role'],
                'input': actual, 'saved_L4_RAW': row['saved_L4_RAW'], 'saved_L4_PNG': row['saved_L4_PNG'],
                'CPU_RAW': E.pin(raw_path), 'CPU_PNG': E.pin(png_path), 'CPU_vs_saved_L4': differences,
                'completed_ordered_sites': list(net.last_forward_sites), 'base_calls': net.last_base_calls})
            E.need(tuple(net.last_forward_sites) == ('32', '64', '128', '256') and net.last_base_calls == 1, 'Ordered one-base forward witness')
        final = net.validate_invariants()
        E.need(initial == final and initial_state == module.source_contract()['canonical_states'], 'Frozen canonical invariants preserved')
        E.need(torch.equal(old_rng, torch.random.get_rng_state()), 'Inference must preserve caller CPU RNG')
        E.need(before == [E.pin(args.r2_artifact), E.pin(args.u96_artifact), E.pin(E.ROOT / export['head_file']['path'])], 'Original model/head artifacts unchanged')
        E.need(not torch.cuda.is_initialized() and len(results) == 4, 'Exactly four CPU witnesses and no CUDA')
        ledger.update('ALL4_RETURNED_AND_FROZEN_INVARIANTS_VERIFIED')
        receipt = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
            'runtime_integrity_pass': True, 'CPU_L4_bitwise_parity_required': False,
            'CPU_VM_numeric_comparison_complete': True, 'numeric_acceptance_pending_root_review': True,
            'numerical_parity_tolerance_or_quality_gate_changed': False, 'CPU_vs_L4_differences': results,
            'four_first_monitor_photo_roles_only': True, 'CPU_threads': 4, 'forward_calls': 4,
            'frozen_invariants_before': initial, 'frozen_invariants_after': final,
            'original_artifact_pins_unchanged': before, 'CPU_RNG_preserved': True,
            'head_file': E.pin(E.ROOT / export['head_file']['path']),
            'counters': dict(ledger.counters), 'runtime_status': E.pin(OUT / 'status.json'),
            'sources': [E.pin(args.source_review), E.pin(args.export_receipt), E.pin(E.INFERENCE), E.pin(Path(__file__)), E.pin(Path(E.__file__))],
            'archive_saved_RAW_readback': transport, 'source_review': E.pin(args.source_review),
            'export_receipt': E.pin(args.export_receipt), 'inference_source': E.pin(E.INFERENCE),
            'script': E.pin(Path(__file__)), 'export_helper': E.pin(Path(E.__file__)),
            'optimizer_constructions': 0, 'gradient_backward_training_VM_calls': 0,
            'CUDA_initialized': False, 'model_qualified': False, 'main_app_default_changed': False,
            'broader_DEV_native_FINAL_user_acceptance_pending': True,
            'material_limits': 'Four CPU witnesses measure cross-device differences only. Finite adaptive TRAIN success, numerical proximity and source integrity do not establish visual usefulness or model qualification.',
            'seconds': time.monotonic() - started}
        result = E.write_json(OUT / 'cpu_runtime_witness_v1.json', receipt)
        E.need(sum(x.stat().st_size for x in OUT.iterdir() if x.is_file()) < 32 * 1024**2, 'Bounded runtime output')
        print(json.dumps({'complete': True, 'runtime_integrity_pass': True, 'receipt': result, 'model_qualified': False}))
    except BaseException as error:
        ledger.fail('RUNTIME_EXCEPTION_OR_INTERRUPTION', error)
        raise
    finally:
        watchdog.cancel()
        if original_threads is not None: torch.set_num_threads(original_threads)


if __name__ == '__main__':
    main()
