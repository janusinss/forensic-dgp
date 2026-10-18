"""Finite frozen CPU path comparison; no optimization, target scoring or app change."""
import argparse
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / 'scripts/run_codeformer_filter_comparison_v1.py'
HELPER_SHA = '75f96ffe5e7438beca734119c5e763b4ae2fe39ffc6181afc194a5e274252506'
PROTOCOL_SHA = '23772988d16e11f52351365e66ce1b6d8ef77694f873966db98e24512b0aa86c'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError('Trusted local frozen helper source changed')
SPEC = importlib.util.spec_from_file_location('trusted_previous_frozen_helper', HELPER_PATH)
H = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(H)
# Only pure I/O, state validation, array and resource helpers are reused. Never
# mutate H.CAPS or call its run/Output/Watchdog implementations.
CAPS = {'new_forward_calls': 48, 'wall_seconds': 600, 'process_RSS_bytes': 3 * 1024**3,
    'minimum_available_RAM_bytes': 768 * 1024**2, 'minimum_disk_free_bytes': 1024**3,
    'output_payload_bytes': 64 * 1024**2, 'output_files': 170,
    'journal_records': 180, 'journal_bytes': 1024**2, 'watchdog_period_seconds': 1}
ARMS = {'original': 'original0', 'parent': 'parent50', 'r2': 'candidate50'}
SETTINGS = ({'id': 'w0_adain0', 'w': 0., 'adain': False},
    {'id': 'w0_adain1', 'w': 0., 'adain': True}, {'id': 'w1_adain0', 'w': 1., 'adain': False})
BASELINE = {'id': 'w1_adain1', 'w': 1., 'adain': True}
CASE_IDS = ('tr_ffhq_00084__clear', 'tr_ffhq_00084__repeated_rows_4_blend0p75',
    'va_ffhq_00383__local_gaussian_3p0', 'va_ffhq_00383__local_mosaic_8', 'va_ffhq_09056__global_mosaic_8')
FUSIONS = ('32', '64', '128', '256')


def require(value, message):
    if not value:
        raise ValueError(message)


def read_pin(pin, load=True, original=False):
    H.canonical(pin['path'])
    path = ROOT / pin['path']
    require(path.absolute() == Path(pin['absolute_path']).absolute(), 'Physical source pin differs')
    return H.bound(path, pin['sha256'], pin['bytes'], allowed_links=3 if original else 1, load=load)


def enforce(values, elapsed):
    require(0 <= elapsed <= CAPS['wall_seconds'], 'Finite600second wall cap reached')
    require(values['process_RSS_bytes'] <= CAPS['process_RSS_bytes'], 'RSS cap reached')
    require(values['available_RAM_bytes'] >= CAPS['minimum_available_RAM_bytes'], 'Available RAM reserve reached')
    require(values['disk_free_bytes'] >= CAPS['minimum_disk_free_bytes'], 'Disk reserve reached')


def check_observer(observer, setting):
    require(observer['top_level_calls'] == 1 and observer['top_level_settings'] ==
        [{'w': setting['w'], 'adain': setting['adain'], 'detach_16': True}], 'Observed top-level call/setting differs')
    expected = dict.fromkeys(FUSIONS, 1 if setting['w'] == 1 else 0)
    require(observer['fusion_calls'] == expected and observer['fusion_order'] ==
        (list(FUSIONS) if setting['w'] == 1 else []), 'Observed four-fusion bypass/enable differs')


def route_signature(readout):
    return {key: readout[key] for key in ('logits', 'lq_feature', 'recomputed_tokens')}


def exact_parity(raw, png, cached_raw, cached_png):
    return {'RAW_exact': bool(np.array_equal(raw, cached_raw)), 'PNG_exact': bool(np.array_equal(png, cached_png)),
        'RAW_maximum_error': float(np.max(np.abs(raw.astype(np.float64) - cached_raw.astype(np.float64)))),
        'PNG_maximum_error': int(np.max(np.abs(png.astype(np.int16) - cached_png.astype(np.int16))))}


def raw_rgb(payload):
    require(len(payload) <= 3 * 256 * 256 * 4 + 256, 'Bounded RAW payload required')
    stream = io.BytesIO(payload); value = np.load(stream, allow_pickle=False)
    require(not stream.read(1), 'Trailing RAW payload')
    H.delivered(value)
    return value


def shared_decoder_names(schema):
    names = sorted(name for name in schema if name.startswith(('generator.', 'quantize.')))
    require(len(names) == 165 and sum(n.startswith('generator.') for n in names) == 164
        and sum(n.startswith('quantize.') for n in names) == 1, 'Exact shared164 decoder plus one codebook tensors required')
    return names


def shared_decoder_digest(state, schema, torch):
    names = shared_decoder_names(schema); digest = hashlib.sha256()
    for name in names:
        value = state[name].detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {'tensors': 165, 'generator_tensors': 164, 'codebook_tensors': 1,
        'canonical_sha256': digest.hexdigest(), 'tensor_names': names}


class Output:
    def __init__(self, folder):
        self.folder = Path(folder).absolute()
        require(not self.folder.exists() and self.folder.parent.resolve(strict=True).is_relative_to(ROOT / 'outputs'),
            'Exclusive new output namespace required; no partial resume')
        # A checked ordinary sibling certifies parent traversal; explicit parent
        # lstat rejects junctions before the new directory is created.
        import stat
        for p in (self.folder.parent, *self.folder.parent.parents):
            if p == ROOT.parent:
                break
            value = p.lstat()
            require(not stat.S_ISLNK(value.st_mode) and not getattr(value, 'st_file_attributes', 0) & 1024, 'Linked output parent')
        self.folder.mkdir(); self.files = {}; self.bytes = 0
        self.journal = (self.folder / 'journal.jsonl').open('xb')
        self.lock = threading.Lock(); self.journal_rows = 0; self.journal_bytes = 0

    def put(self, name, payload):
        H.canonical(name)
        reserve = 0 if 'journal.jsonl' in self.files else self.journal_bytes
        require(name not in self.files and len(self.files) + 2 <= CAPS['output_files'] and
            self.bytes + len(payload) + reserve <= CAPS['output_payload_bytes'], 'Finite output file/byte cap reached')
        path = self.folder / name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        require(H.ordinary(path).read_bytes() == payload, 'Saved artifact readback differs')
        pin = {'path': name, 'sha256': H.digest(payload), 'bytes': len(payload)}
        self.files[name] = pin; self.bytes += len(payload)
        return dict(pin)

    def record(self, row):
        payload = (json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode()
        with self.lock:
            require(self.journal_rows < CAPS['journal_records'] and self.journal_bytes + len(payload) <= CAPS['journal_bytes'],
                'Finite journal cap reached')
            self.journal.write(payload); self.journal.flush(); os.fsync(self.journal.fileno())
            self.journal_rows += 1; self.journal_bytes += len(payload)

    def close_journal(self):
        with self.lock:
            self.journal.flush(); os.fsync(self.journal.fileno()); self.journal.close()
        payload = (self.folder / 'journal.jsonl').read_bytes()
        self.files['journal.jsonl'] = {'path': 'journal.jsonl', 'sha256': H.digest(payload), 'bytes': len(payload)}
        self.bytes += len(payload)


class Watchdog:
    def __init__(self, output, progress):
        self.output, self.progress = output, progress
        self.started = time.monotonic(); self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def check(self):
        values = H.resource_values(self.output.folder); enforce(values, time.monotonic() - self.started)
        return values

    def failure(self, error, hard=False):
        payload = H.encoded({'complete': False, 'error': str(error), 'watchdog_hard_exit': hard,
            'progress': dict(self.progress), 'partial_preserved_do_not_resume': True,
            'reserved_calls_not_necessarily_committed': True, 'training_updates': 0,
            'seconds': time.monotonic() - self.started})
        try:
            with (self.output.folder / 'failure.json').open('xb') as stream:
                stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            pass

    def loop(self):
        while not self.stop.wait(CAPS['watchdog_period_seconds']):
            try:
                self.check()
            except BaseException as error:
                self.failure(error, hard=True); os._exit(124)


class Observer:
    """Read-only hooks return None; no tensors, weights, RNG or outputs changed."""
    def __init__(self, net):
        self.current = None; self.total_top = 0; self.total_fusion = dict.fromkeys(FUSIONS, 0)
        self.handles = [net.register_forward_pre_hook(self.top, with_kwargs=True)]
        for size in FUSIONS:
            def hook(module, args, kwargs, site=size):
                self.current['fusion_calls'][site] += 1
                self.current['fusion_order'].append(site); self.total_fusion[site] += 1
            self.handles.append(net.fuse_convs_dict[size].register_forward_pre_hook(hook, with_kwargs=True))

    def top(self, module, args, kwargs):
        self.current['top_level_calls'] += 1; self.total_top += 1
        self.current['top_level_settings'].append({k: kwargs[k] for k in ('w', 'adain', 'detach_16')})

    def begin(self):
        self.current = {'top_level_calls': 0, 'top_level_settings': [],
            'fusion_calls': dict.fromkeys(FUSIONS, 0), 'fusion_order': []}

    def remove(self):
        for handle in self.handles:
            handle.remove()


def describe_tokens(tokens):
    require(tokens.shape == (1, 256) and tokens.dtype == np.int64 and tokens.flags.c_contiguous
        and tokens.min() >= 0 and tokens.max() <= 1023, 'Exact256 contiguous int64 recomputed tokens required')
    return {'shape': list(tokens.shape), 'dtype': str(tokens.dtype),
        'array_sha256': H.digest(tokens.tobytes()), 'indices': tokens.reshape(-1).tolist(),
        'rule': 'Exact released F.softmax(returned_logits,dim2) then torch.topk(k1,dim2); squeeze trailing singleton',
        'topk_native_shape': [1, 256, 1], 'actual_internal_codebook_lookup_captured': False}


def internal_readout(logits, lq, torch, F):
    require(logits.shape == (1, 256, 1024) and lq.shape == (1, 256, 16, 16)
        and all(t.dtype == torch.float32 and t.device.type == 'cpu' and not t.requires_grad and
            torch.isfinite(t).all().item() for t in (logits, lq)), 'Returned internal geometry/finite type differs')
    describe = lambda t: {'shape': list(t.shape), 'dtype': str(t.dtype),
        'array_sha256': H.digest(t.detach().contiguous().numpy().tobytes())}
    probabilities = F.softmax(logits, dim=2)
    _, top1 = torch.topk(probabilities, 1, dim=2)
    top2, _ = torch.topk(probabilities, 2, dim=2)
    margins = top2[..., 0] - top2[..., 1]
    require(torch.isfinite(margins).all().item() and torch.all(margins >= 0).item(), 'Finite top2 probability margins required')
    tokens = top1.squeeze(-1).contiguous().cpu().numpy().copy()
    return {'logits': describe(logits), 'lq_feature': describe(lq),
        'recomputed_tokens': describe_tokens(tokens),
        'top2_probability_margin': {'minimum': float(margins.min()), 'mean': float(margins.mean()),
            'maximum': float(margins.max()), 'exact_zero_margin_token_count': int((margins == 0).sum())},
        'readout_only_no_extra_neural_forward': True}


def validate_plan(p):
    require(p['complete'] is True and tuple(c['id'] for c in p['cases']) == CASE_IDS, 'Frozen five-case selection differs')
    require(p['caps'] == dict(CAPS, resource_sampling='aroundeachcall andindependent1s watchdog;sampledpeak limitations disclosed'),
        'Prospective48/600s resources/output caps differ')
    require(p['planned_counts'] == {'models': 3, 'cases': 5, 'factorial_cells': 60,
        'cached_w1_adain_true_cells': 15, 'new_factorial_cells': 45, 'fresh_clear_baseline_parity_calls': 3,
        'total_new_forward_calls': 48, 'optimizer_updates': 0, 'backward_calls': 0,
        'factorial_groups': 15, 'fresh_per_model_calls': 16, 'source_photographs': 3}, 'Exact finite path call plan differs')
    require(p['baseline_reuse']['fresh_clear_case'] == CASE_IDS[0], 'Fresh parity clear case differs')
    for case in p['cases']:
        require(case['input_review']['decision'] in ('readable', 'borderline')
            and case['appearance_review_required'] is False and set(case['cached_baseline']) == set(ARMS.values()),
            'No insufficient/appearance-flag/new admission in targeted five-case comparison')
    require([{k: row[k] for k in ('id', 'w', 'adain')} for row in p['settings']] == [*SETTINGS, BASELINE],
        'Exact2x2 setting plan differs')


def preflight(p):
    """Exercise exact actual metadata and cached arrays before any model import."""
    validate_plan(p)
    metadata = {key: H.json_read(read_pin(pin)) for key, pin in p['bound_metadata'].items()
        if pin['path'].endswith('.json')}
    for pin in p['bound_metadata'].values():
        read_pin(pin, load=False)
    for pin in p['freshly_hashed_dependency_files']:
        original = pin['path'] == H.MODEL_FILES['original'][0] and pin['sha256'] == H.MODEL_FILES['original'][1]
        read_pin(pin, load=False, original=original)
    bindings = metadata['model_binding']; previous_runtime = metadata['comparison_runtime']
    require(metadata['comparison_pixel_audit']['pass'] is True and metadata['comparison_journal_audit']['pass'] is True,
        'Previous216 outputs must have completed independent audits')
    require({m['arm'] for m in p['models']} == set(ARMS) and p['models'] == bindings['models']
        and {r['path']: r['sha256'] for r in p['architecture_sources']} == H.ARCHITECTURE, 'Exact model/source bindings differ')
    for pin in p['architecture_sources']:
        read_pin(pin, load=False)
    inputs = {}; cached = {}
    for case in p['cases']:
        inputs[case['id']] = H.png_pixels(read_pin(case['input']))
        for bank, old in case['cached_baseline'].items():
            cached[bank, case['id']] = (raw_rgb(read_pin(old['raw'])), H.png_pixels(read_pin(old['png'])))
            require(H.digest(cached[bank, case['id']][0].tobytes()) == old['array_digests']['RAW_array_sha256']
                and H.digest(cached[bank, case['id']][1].tobytes()) == old['array_digests']['PNG_RGB8_sha256'], 'Cached array digest differs')
            read_pin(old['evidence'])
    return bindings, previous_runtime, inputs, cached


def run(protocol_path, out):
    require(PROTOCOL_SHA != 'PENDING_FINAL_PROTOCOL', 'Final independently reviewed protocol not frozen yet')
    require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')), 'Start without neural framework')
    source_sha = H.digest(Path(__file__).read_bytes()); helper_caps = dict(H.CAPS)
    protocol_bytes = H.bound(protocol_path, PROTOCOL_SHA); p = H.json_read(protocol_bytes)
    bindings, previous_runtime, inputs, cached = preflight(p)
    output = Output(out); progress = {'reserved': 0, 'returned': 0, 'committed': 0, 'in_flight': None, 'arm': None}
    guard = Watchdog(output, progress)
    rows, parity, model_proofs = [], [], []; observer = None; shared_original = None
    try:
        guard.check(); guard.thread.start()
        output.put('protocol.json', protocol_bytes)
        output.put('source_contract.json', H.encoded({'source_sha256': source_sha, 'trusted_helper_sha256': HELPER_SHA,
            'caps': CAPS, 'new_factorial_outputs': 45, 'fresh_clear_parity_outputs': 3, 'cached_cells_reused': 15,
            'no_training_scoring_or_app_change': True, 'no_extra_forwards_for_readouts': True,
            'hardcoded_settings_only': [*SETTINGS, BASELINE], 'internal_token_readout_is_recomputed_not_actual_lookup': True}))
        import torch
        from torch.nn import functional as F
        from third_party.codeformer.codeformer_arch import CodeFormer
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        require(torch.__version__ == previous_runtime['torch'] and np.__version__ == previous_runtime['NumPy']
            and H.pillow_version == previous_runtime['Pillow'], 'CPU runtime must match cached baseline')
        output.put('runtime.json', H.encoded({'Python': sys.version, 'torch': torch.__version__, 'NumPy': np.__version__,
            'Pillow': H.pillow_version, 'device': 'cpu', 'threads': 4, 'interop_threads': 1, 'resources': guard.check()}))
        for arm, bank in ARMS.items():
            progress['arm'] = arm; guard.check()
            binding = next(row for row in p['models'] if row['arm'] == arm)
            file = binding['file']; expected_path, expected_sha, key = H.MODEL_FILES[arm]
            require(file['path'] == expected_path and file['sha256'] == expected_sha and
                binding['canonical515_tensor_sha256'] == H.STATE_SHA[arm], 'Frozen retained model identity differs')
            model_path = H.bound(ROOT / file['path'], expected_sha, file['bytes'],
                allowed_links=3 if arm == 'original' else 1, load=False)
            saved = torch.load(model_path, map_location='cpu', weights_only=True, mmap=True)
            require(H.canonical_state(saved[key], bindings['full515_tensor_schema'], torch) == H.STATE_SHA[arm], 'Loaded515 state differs')
            shared = shared_decoder_digest(saved[key], bindings['full515_tensor_schema'], torch)
            if shared_original is None:
                shared_original = shared
            require(shared == shared_original, 'Shared generator/codebook165 values differ across retained models')
            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=list(FUSIONS))
            net.load_state_dict(saved[key], strict=True); net.float().eval().requires_grad_(False)
            del saved; gc.collect(); H.frozen(net, torch)
            before = H.canonical_state(net.state_dict(), bindings['full515_tensor_schema'], torch)
            require(before == H.STATE_SHA[arm], 'Constructed515 state differs')
            observer = Observer(net); upstream = {}
            output.record({'event': 'model_loaded_frozen', 'arm': bank, 'state_sha256': before, 'resources': guard.check()})

            def forward(case, setting, kind):
                guard.check(); H.frozen(net, torch)
                require(progress['reserved'] < CAPS['new_forward_calls'], 'Finite48 forward reservation cap reached')
                call = progress['reserved'] + 1; observer.begin()
                output.record({'event': 'forward_reserved', 'call': call, 'arm': bank, 'id': case['id'], 'setting': setting,
                    'kind': kind, 'seconds': time.monotonic() - guard.started})
                progress['reserved'] = call; progress['in_flight'] = {'call': call, 'arm': bank, 'id': case['id'], 'setting': setting['id']}
                started = time.monotonic(); x = torch.from_numpy(H.canonical_input(inputs[case['id']])).permute(2, 0, 1)[None]
                with torch.no_grad():
                    sample = F.interpolate(x, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
                    native, logits, lq = net(sample, w=setting['w'], detach_16=True, adain=setting['adain'])
                    progress['returned'] += 1
                    require(native.shape == (1, 3, 512, 512) and native.dtype == torch.float32 and
                        torch.isfinite(native).all().item() and not native.requires_grad, 'Native output invalid')
                    raw = F.interpolate(((native.float() + 1) / 2).clamp(0, 1), (256, 256), mode='bilinear',
                        align_corners=False)[0].permute(1, 2, 0).contiguous().numpy().copy()
                    internals = internal_readout(logits, lq, torch, F)
                pixels = H.delivered(raw); H.frozen(net, torch)
                signature = route_signature(internals)
                same_upstream = case['id'] not in upstream or upstream[case['id']] == signature
                if case['id'] not in upstream:
                    upstream[case['id']] = signature
                compare = exact_parity(raw, pixels, *cached[bank, case['id']]) if kind == 'fresh_clear_parity' else None
                output.record({'event': 'forward_returned', 'call': call, 'arm': bank, 'id': case['id'], 'setting': setting['id'],
                    'kind': kind, 'observer': observer.current, 'resources': guard.check()})
                prefix = ('parity/' if kind == 'fresh_clear_parity' else 'outputs/') + bank + '/' + case['id'] + '/' + setting['id'] + '/'
                data = io.BytesIO(); np.save(data, raw, allow_pickle=False)
                raw_pin = output.put(prefix + 'raw.npy', data.getvalue()); png_pin = output.put(prefix + 'native.png', H.image_bytes(pixels))
                evidence = {'format': 'codeformer-frozen-path-output-v1', 'id': case['id'], 'reference_id': case['reference_id'],
                    'split': case['split'], 'profile': case['profile'], 'input_review': case['input_review'],
                    'appearance_review_required': case['appearance_review_required'], 'input': case['input'],
                    'targets_not_scored': case['targets'], 'arm': bank, 'model_binding': binding, 'state_sha256': before,
                    'setting': setting, 'kind': kind, 'call': call, 'raw': raw_pin, 'png': png_pin,
                    'RAW_array_sha256': H.digest(raw.tobytes()), 'PNG_RGB8_sha256': H.digest(pixels.tobytes()),
                    'observer': observer.current, 'internal_readout': internals, 'same_upstream_within_input_model': same_upstream,
                    'clear_parity': compare, 'device': 'cpu', 'eval_all_modules': True, 'parameters_require_grad': False,
                    'no_grad': True, 'internal_resolution': 512, 'output_resolution': 256, 'geometry_changed': False,
                    'alignment': 'none', 'display_processing': 'none', 'optimizer_updates': 0, 'backward_calls': 0,
                    'paired_metrics': None, 'model_qualified': False, 'seconds': time.monotonic() - started}
                ev_pin = output.put(prefix + 'evidence.json', H.encoded(evidence))
                output.record({'event': 'output_committed', 'call': call, 'arm': bank, 'id': case['id'], 'setting': setting['id'],
                    'kind': kind, 'evidence': ev_pin, 'observer': observer.current, 'resources': guard.check()})
                progress['committed'] += 1; progress['in_flight'] = None
                result = {'id': case['id'], 'arm': bank, 'setting': setting, 'kind': kind,
                    'raw': raw_pin, 'png': png_pin, 'evidence': ev_pin, 'clear_parity': compare}
                check_observer(observer.current, setting)
                require(same_upstream, 'Same input/model logits, LQ features or recomputed tokens changed with late-route settings')
                if compare is not None:
                    require(compare['RAW_exact'] and compare['PNG_exact'], 'Fresh clear cached baseline parity failed; retain outputs and stop')
                return result

            parity.append(forward(p['cases'][0], BASELINE, 'fresh_clear_parity'))
            output.record({'event': 'baseline_parity_passed', 'arm': bank, 'id': CASE_IDS[0], 'before_interventions': True})
            for case in p['cases']:
                for setting in SETTINGS:
                    rows.append(forward(case, setting, 'new_factorial'))
                rows.append({'id': case['id'], 'arm': bank, 'setting': BASELINE,
                    'kind': 'cached_prior_comparison', 'cached': case['cached_baseline'][bank]})
                print(json.dumps({'reserved': progress['reserved'], 'committed': progress['committed'], 'of': 48,
                    'arm': bank, 'case': case['id'], 'seconds': time.monotonic() - guard.started}), flush=True)
            after = H.canonical_state(net.state_dict(), bindings['full515_tensor_schema'], torch)
            H.frozen(net, torch); require(after == before == H.STATE_SHA[arm], 'Frozen515 state changed')
            require(observer.total_top == 16 and observer.total_fusion == dict.fromkeys(FUSIONS, 6), 'Per-model observed call totals differ')
            H.bound(model_path, expected_sha, file['bytes'], allowed_links=3 if arm == 'original' else 1, load=False)
            model_proofs.append({'arm': bank, 'before_state_sha256': before, 'after_state_sha256': after,
                'top_level_calls': observer.total_top, 'fusion_calls': observer.total_fusion, 'file': file,
                'shared_generator_codebook': shared})
            output.record({'event': 'model_finished_frozen', 'arm': bank, 'proof': model_proofs[-1]})
            observer.remove(); observer = None; del forward, net; gc.collect(); guard.check()
        require(progress['reserved'] == progress['returned'] == progress['committed'] == 48 and
            progress['in_flight'] is None and len(rows) == 60 and len(parity) == 3, 'Exact48/60cell closure differs')
        require(H.CAPS == helper_caps and H.digest(HELPER_PATH.read_bytes()) == HELPER_SHA and
            H.digest(Path(__file__).read_bytes()) == source_sha, 'Trusted helper/source changed')
        for pin in p['architecture_sources']:
            read_pin(pin, load=False)
        output.put('model_state_proofs.json', H.encoded({'complete': True, 'models': model_proofs}))
        output.record({'event': 'complete', 'progress': progress, 'resources': guard.check()}); output.close_journal()
        manifest = {'format': 'codeformer-frozen-path-isolation-results-v1', 'complete': True, 'protocol_sha256': PROTOCOL_SHA,
            'source_sha256': source_sha, 'trusted_helper_sha256': HELPER_SHA, 'caps': CAPS,
            'cells': rows, 'fresh_clear_parity': parity, 'model_proofs': model_proofs, 'files': dict(output.files),
            'shared165_generator_codebook_identical_across_three_models': shared_original,
            'actual_new_forward_calls': 48, 'new_factorial_outputs': 45, 'cached_cells_reused': 15,
            'fresh_clear_parity_outputs': 3, 'actual_optimizer_updates': 0, 'actual_backward_calls': 0,
            'target_scoring': False, 'same_upstream_readouts_all_inputs_models': True,
            'upstream_equality_scope': 'Three new factorial settings per input/model; fresh w1AdaINtrue only on clear. Other cached cells have no captured internal readouts.',
            'settings_are_global_late_path_interventions_not_additive_individual_fusion_layers': True,
            'recomputed_token_readout_not_direct_internal_lookup_capture': True,
            'exploratory_after_known_failures_not_unseen_validation': True,
            'all96_previous_inputs_and24_refusals_unchanged': True, 'app_or_model_changed': False,
            'new_acceptance_thresholds': False, 'model_qualified': False, 'seconds': time.monotonic() - guard.started,
            'manifest_excludes_own_recursive_hash': True}
        pin = output.put('manifest.json', H.encoded(manifest))
        require({x.relative_to(output.folder).as_posix() for x in output.folder.rglob('*') if x.is_file()} == set(output.files), 'Output namespace differs')
        for name, entry in output.files.items():
            guard.check(); H.bound(output.folder / name, entry['sha256'], entry['bytes'], load=False)
        guard.check(); guard.stop.set(); guard.thread.join(timeout=2)
        return {'complete': True, 'actual_new_forward_calls': 48, 'files': len(output.files), 'bytes': output.bytes,
            'manifest': pin, 'seconds': time.monotonic() - guard.started, 'optimizer_updates': 0}
    except BaseException as error:
        guard.stop.set(); guard.failure(error)
        try:
            output.record({'event': 'failure', 'error': str(error), 'progress': progress}); output.close_journal()
        except BaseException:
            pass
        raise
    finally:
        if observer is not None:
            observer.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.out)), flush=True)
