"""Bounded local frozen CPU inference on pre-reviewed fixed synthetic inputs.

No optimization, gradients, training, routing, completion, target scoring or
acceptance-threshold changes. A partial journal is evidence, never a resume plan.
"""
import argparse
from collections import Counter
import ctypes
import gc
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import threading
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_SHA = '7823847ce7a709308fe1f4d35d35e2c14f752b2d5c303f5fa1cb4b7c8de8c3a8'
REVIEW_SHA = '51bff1593cd33ad2b369a7c20d2b5282ed67521df4b62e17a53f082941fc3b50'
CRITERIA_SHA = 'fd0ae5aa624f74775a65474957f42cf3cc69e0838febac4f1feb9d063340f903'
MODELS_SHA = '5a926fc23c58916374d4ad69bf68fc4066bc469c440d920a2046b09a2761a994'
MODEL_NAMES = {'original': 'original0', 'parent': 'parent50', 'r2': 'candidate50'}
MODEL_FILES = {
    'original': ('outputs/codeformer_restoration_pretrained_v1/codeformer.pth',
        '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7', 'params_ema'),
    'parent': ('outputs/codeformer_role_preserving_training_v1_r2_return/outputs/role_preserving_training_v1_r2/snapshot_0000/checkpoint.pt',
        '53778f54589697a14b067033eff8cffc2985985e8a33021092de3c14fbf443bc', 'model'),
    'r2': ('outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth',
        'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896', 'params_ema')}
STATE_SHA = {'original': '6cb93cee937f67814edcf4625a68bae4bf8f1c2351f7306864ceefa63d4803bd',
    'parent': '92ac81750b86ef1da6f1f7c0480b9e64773fafe07b094e4a5ad7e88b3f7a0c8b',
    'r2': 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'}
ARCHITECTURE = {'third_party/codeformer/codeformer_arch.py': 'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb',
    'third_party/codeformer/vqgan_arch.py': 'b47033f66b4a8434e0f04d4df8a69a7ac4b6c5bdf52f5b74912371dff0a20eaf',
    'third_party/codeformer/registry.py': 'bbc3507148101bb202e8f52f5cca75d2cc21d780def757c055d82cb8e6bd594e'}
CAPS = {'forward_calls': 216, 'wall_seconds': 3600, 'process_RSS_bytes': 3 * 1024**3,
    'minimum_available_RAM_bytes': 768 * 1024**2, 'minimum_disk_free_bytes': 1024**3,
    'output_files': 660, 'output_payload_bytes': 256 * 1024**2,
    'journal_records': 700, 'journal_bytes': 2 * 1024**2, 'watchdog_period_seconds': 1}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')


def json_read(data):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data.decode('utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def ordinary(path, allowed_links=1):
    path = Path(path).absolute()
    require(path.resolve(strict=True).is_relative_to(ROOT), 'File escapes workspace')
    for part in (path, *path.parents):
        if part == ROOT.parent:
            break
        info = part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
            'Linked/reparse paths forbidden')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == allowed_links, 'File link/type differs')
    return path


def bound(path, fingerprint, size=None, allowed_links=1, cap=2 * 1024**3, load=True):
    path = ordinary(path, allowed_links)
    before = path.stat(); require(before.st_size <= cap and (size is None or before.st_size == size), 'File size differs')
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            sha.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
        (after.st_size, after.st_mtime_ns, after.st_ino) and sha.hexdigest() == fingerprint, 'File fingerprint changed')
    if not load:
        return path
    require(after.st_size <= 8 * 1024**2, 'Inert metadata/image read is bounded')
    data = path.read_bytes(); require(digest(data) == fingerprint, 'Read changed after hashing')
    return data


def canonical(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and not p.is_absolute() and p.as_posix() == name and ':' not in name and '\\' not in name
        and all(part not in ('', '.', '..') for part in p.parts), 'Canonical relative path required')
    return p


def png_pixels(data):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256), 'Exact RGB256 PNG required')
        return np.asarray(image).copy()


def canonical_input(pixels):
    require(pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8, 'RGB256 input required')
    return np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))


def delivered(raw):
    require(raw.shape == (256, 256, 3) and raw.dtype == np.float32 and raw.flags.c_contiguous
        and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1, 'Finite contiguous RAW256 float32 required')
    return np.floor(raw * np.float32(255)).astype(np.uint8)


def image_bytes(array):
    stream = io.BytesIO(); Image.fromarray(array).save(stream, format='PNG')
    payload = stream.getvalue(); require(np.array_equal(png_pixels(payload), array), 'PNG readback differs')
    return payload


def plan_cases(data, review):
    require(data['complete'] is True and data['pairs'] == 96 and review['complete'] is True
        and review['data_manifest_sha256'] == DATA_SHA and review['criteria_sha256'] == CRITERIA_SHA,
        'Frozen data/input-review bindings differ')
    labels = {}
    for row in review['cases']:
        key = row['reference_id'] + '__' + row['profile']
        require(key not in labels and row['decision'] in ('readable', 'borderline', 'insufficient'), 'Invalid input review')
        labels[key] = row
    cases = []
    for row in data['cases']:
        label = labels.get(row['id']); require(label is not None, 'Input lacks prospective review')
        require(row['id'] == row['reference_id'] + '__' + row['profile'], 'Case identity/profile differs')
        cases.append(dict(row, input_review=label,
            appearance_review_required=row['reference_id'] == 'va_ffhq_08025',
            inference_planned=label['decision'] != 'insufficient', outputs={}))
    require(len(cases) == len(labels) == 96 and len({r['id'] for r in cases}) == 96
        and Counter(r['input_review']['decision'] for r in cases) == {'readable': 28, 'borderline': 44, 'insufficient': 24}
        and Counter(r['split'] for r in cases) == {'TRAIN': 48, 'DEV': 48}, 'Exact96/72infer/24skip plan differs')
    require(all(not r['optimization_permitted_now'] and
        (r['split'] != 'DEV' or r['optimization_allowed_by_split'] is False and r['restoration_training_eligibility'] == 'EXCLUDED_DEV')
        for r in cases), 'Inference does not authorize optimization')
    return cases


def resource_values(folder):
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                *[(name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available',
                    'virtual_total', 'virtual_available', 'extended')]]
        status = Memory(); status.length = ctypes.sizeof(status)
        require(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)), 'RAM measurement unavailable')
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong),
                *[(name, ctypes.c_size_t) for name in ('peak', 'working', 'paged_peak', 'paged',
                    'nonpaged_peak', 'nonpaged', 'pagefile', 'pagefile_peak')]]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess; current.restype = ctypes.c_void_p
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        require(get_info(current(), ctypes.byref(counters), counters.cb), 'RSS measurement unavailable')
        rss, available = int(counters.working), int(status.available)
    else:
        process = Path('/proc/self/status').read_text()
        memory = Path('/proc/meminfo').read_text()
        rss = int(next(row.split()[1] for row in process.splitlines() if row.startswith('VmRSS:'))) * 1024
        available = int(next(row.split()[1] for row in memory.splitlines() if row.startswith('MemAvailable:'))) * 1024
    return {'process_RSS_bytes': rss, 'available_RAM_bytes': available, 'disk_free_bytes': shutil.disk_usage(folder).free}


def enforce_resources(values, elapsed):
    require(elapsed <= CAPS['wall_seconds'], 'Finite3600s wall cap reached')
    require(values['process_RSS_bytes'] <= CAPS['process_RSS_bytes'], 'Process RSS cap reached')
    require(values['available_RAM_bytes'] >= CAPS['minimum_available_RAM_bytes'], 'Available RAM reserve reached')
    require(values['disk_free_bytes'] >= CAPS['minimum_disk_free_bytes'], 'Disk reserve reached')


class Output:
    def __init__(self, folder):
        self.folder = Path(folder).absolute()
        require(not self.folder.exists() and self.folder.parent.resolve(strict=True).is_relative_to(ROOT / 'outputs'),
            'New output namespace required; never resume/overwrite a partial run')
        for part in (self.folder.parent, *self.folder.parent.parents):
            if part == ROOT.parent:
                break
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024, 'Linked output parent')
        self.folder.mkdir(); self.files = {}; self.bytes = 0
        self.journal = (self.folder / 'journal.jsonl').open('xb')
        self.journal_lock = threading.Lock(); self.journal_rows = 0; self.journal_bytes = 0

    def put(self, name, data):
        p = canonical(name)
        require(name not in self.files and len(self.files) + 2 <= CAPS['output_files']
            and self.bytes + len(data) + (0 if 'journal.jsonl' in self.files else self.journal_bytes)
                <= CAPS['output_payload_bytes'], 'Output file/byte cap reached')
        path = self.folder.joinpath(*p.parts); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        require(ordinary(path).read_bytes() == data, 'Output readback differs')
        pin = {'path': name, 'sha256': digest(data), 'bytes': len(data)}
        self.files[name] = pin; self.bytes += len(data)
        return dict(pin)

    def record(self, record):
        data = (json.dumps(record, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        with self.journal_lock:
            require(self.journal_rows + 1 <= CAPS['journal_records'] and
                self.journal_bytes + len(data) <= CAPS['journal_bytes'], 'Journal bounds reached')
            self.journal.write(data); self.journal.flush(); os.fsync(self.journal.fileno()); self.journal_rows += 1
            self.journal_bytes += len(data)

    def close_journal(self):
        with self.journal_lock:
            self.journal.flush(); os.fsync(self.journal.fileno()); self.journal.close()
        data = (self.folder / 'journal.jsonl').read_bytes()
        self.files['journal.jsonl'] = {'path': 'journal.jsonl', 'sha256': digest(data), 'bytes': len(data)}
        self.bytes += len(data)


class Watchdog:
    def __init__(self, output, progress):
        self.output, self.progress = output, progress
        self.started = time.monotonic(); self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def check(self):
        values = resource_values(self.output.folder)
        enforce_resources(values, time.monotonic() - self.started)
        return values

    def failure(self, error, watchdog=False):
        info = {'complete': False, 'failure': str(error), 'watchdog_hard_exit': watchdog,
            'progress': dict(self.progress), 'all_reserved_calls_not_necessarily_committed': True,
            'partial_preserved_do_not_resume': True, 'training_updates': 0,
            'seconds': time.monotonic() - self.started}
        path = self.output.folder / 'failure.json'
        try:
            with path.open('xb') as stream:
                stream.write(encoded(info)); stream.flush(); os.fsync(stream.fileno())
        except FileExistsError:
            pass

    def loop(self):
        while not self.stop.wait(CAPS['watchdog_period_seconds']):
            try:
                self.check()
            except BaseException as error:
                self.failure(error, watchdog=True)
                os._exit(124)


def canonical_state(state, schema, torch):
    require(set(state) == set(schema) and len(state) == 515, 'Exactly the pinned515 tensors required')
    sha = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.device.type == 'cpu'
            and list(value.shape) == schema[name]['shape'] and schema[name]['dtype'] == str(value.dtype)
            and value.numel() == schema[name]['elements'] and torch.isfinite(value).all().item(),
            'Pinned finite FP32 CPU tensor schema differs: ' + name)
        value = value.detach().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        sha.update(len(header).to_bytes(8, 'big')); sha.update(header)
        sha.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def frozen(net, torch):
    require(all(not module.training for module in net.modules()) and
        all(not p.requires_grad and p.grad is None and p.device.type == 'cpu' and p.dtype == torch.float32
            for p in net.parameters()), 'All model modules must stay frozen FP32 CPU eval without gradients')


def run(data_path, review_path, criteria_path, models_path, out):
    require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')), 'Worker starts without a framework')
    worker_sha = digest(Path(__file__).read_bytes())
    blobs = {'data_manifest.json': bound(data_path, DATA_SHA), 'input_review.json': bound(review_path, REVIEW_SHA),
        'input_review_criteria.json': bound(criteria_path, CRITERIA_SHA), 'model_binding.json': bound(models_path, MODELS_SHA)}
    data, review, criteria, bindings = [json_read(blobs[name]) for name in blobs]
    require(criteria['existing_quality_gates_changed'] is False and bindings['complete'] is True, 'Policy/bindings incomplete')
    cases = plan_cases(data, review); selected = [r for r in cases if r['inference_planned']]
    require(len(selected) * len(MODEL_NAMES) == CAPS['forward_calls'], 'Exact216 prospective forwards required')
    data_root = Path(data_path).resolve(strict=True).parent
    for name, pin in data['files'].items():
        canonical(name); bound(data_root / name, pin['sha256'], pin['bytes'], load=False)
    inputs, records = {}, {}
    for case in cases:
        pair = json_read(bound(data_root / case['pair_record']['path'], case['pair_record']['sha256'], case['pair_record']['bytes']))
        require(pair['id'] == case['id'] and pair['reference_id'] == case['reference_id'] and pair['split'] == case['split']
            and pair['profile']['id'] == case['profile'] and pair['observed_input'] == case['input'], 'Pair join differs')
        pixels = png_pixels(bound(data_root / case['input']['path'], case['input']['sha256'], case['input']['bytes']))
        require(digest(pixels.tobytes()) == pair['input256_pixels']['RGB8_sha256'], 'Observed input pixels differ')
        inputs[case['id']], records[case['id']] = pixels, pair
    require({m['arm'] for m in bindings['models']} == set(MODEL_NAMES) and len(bindings['models']) == 3,
        'Exact three retained models required')
    require({p['path']: p['sha256'] for p in bindings['architecture_sources']} == ARCHITECTURE, 'Architecture pins differ')
    for pin in bindings['architecture_sources']:
        bound(ROOT / pin['path'], pin['sha256'], pin['bytes'], load=False)
    output = Output(out); progress = {'reserved_forwards': 0, 'returned_forwards': 0, 'committed_outputs': 0,
        'in_flight': None, 'current_arm': None}
    guard = Watchdog(output, progress)
    try:
        guard.check(); guard.thread.start()
        for name, payload in blobs.items():
            output.put(name, payload)
        output.put('source_contract.json', encoded({'source_sha256': worker_sha,
            'caps': CAPS, 'inputs': 96, 'skip_insufficient': 24, 'models': MODEL_NAMES,
            'scoring': 'none', 'no_completion': True, 'training_updates': 0, 'backward_calls': 0,
            'architecture': ARCHITECTURE, 'fidelity': 1., 'adain': True, 'detach_16': True,
            'input_normalization': 'NumPy float32 division by float32(255), then bilinear512 and *2-1',
            'output_processing': 'native512 +1 /2 clamp0to1 then bilinear256; floor RAWfloat32*255 to PNG',
            'partial_resume_or_automatic_retry': False}))
        output.put('skipped_cases.json', encoded({'complete': True, 'cases': [r for r in cases if not r['inference_planned']],
            'refused_from_prospective_input_review': True, 'automatic_threshold_validated': False}))
        import torch
        from torch.nn import functional as F
        from third_party.codeformer.codeformer_arch import CodeFormer
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        output.put('runtime.json', encoded({'Python': sys.version, 'NumPy': np.__version__, 'Pillow': pillow_version,
            'torch': torch.__version__, 'device': 'cpu', 'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(), 'initial_resources': guard.check()}))
        proofs = []
        for arm in MODEL_NAMES:
            progress['current_arm'] = arm; guard.check()
            binding = next(row for row in bindings['models'] if row['arm'] == arm)
            file = binding['file']; expected_path, expected_sha, expected_key = MODEL_FILES[arm]
            require(file['path'] == expected_path and file['sha256'] == expected_sha and
                binding['canonical515_tensor_sha256'] == STATE_SHA[arm] and
                binding['load_policy']['model_state_key'] == expected_key, 'Frozen model identity differs')
            model_path = bound(ROOT / file['path'], expected_sha, file['bytes'],
                allowed_links=3 if arm == 'original' else 1, load=False)
            saved = torch.load(model_path, map_location='cpu', weights_only=True, mmap=True)
            require(sorted(saved) == sorted(binding['root_keys']), 'Saved root keys differ')
            state = saved[expected_key]; state_hash = canonical_state(state, bindings['full515_tensor_schema'], torch)
            require(state_hash == STATE_SHA[arm], 'Loaded canonical model state differs')
            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                connect_list=['32', '64', '128', '256'])
            net.load_state_dict(state, strict=True); net.float().eval().requires_grad_(False)
            del state, saved; gc.collect(); frozen(net, torch)
            before = canonical_state(net.state_dict(), bindings['full515_tensor_schema'], torch)
            require(before == STATE_SHA[arm], 'Constructed canonical state differs')
            output.record({'event': 'model_loaded_frozen', 'arm': arm, 'state_sha256': before, 'resources': guard.check()})
            for case in selected:
                guard.check(); frozen(net, torch)
                require(progress['reserved_forwards'] < CAPS['forward_calls'], 'Finite216 forward cap reached')
                call = progress['reserved_forwards'] + 1
                output.record({'event': 'forward_reserved', 'call': call, 'arm': arm, 'id': case['id'],
                    'input_sha256': case['input']['sha256'], 'seconds': time.monotonic() - guard.started})
                progress['reserved_forwards'] = call; progress['in_flight'] = {'call': call, 'arm': arm, 'id': case['id']}
                started = time.monotonic(); x = torch.from_numpy(canonical_input(inputs[case['id']])).permute(2, 0, 1)[None]
                with torch.no_grad():
                    sample = F.interpolate(x, (512, 512), mode='bilinear', align_corners=False) * 2 - 1
                    native = net(sample, w=1., detach_16=True, adain=True)[0]
                    progress['returned_forwards'] += 1
                    require(native.shape == (1, 3, 512, 512) and native.dtype == torch.float32
                        and native.device.type == 'cpu' and not native.requires_grad and torch.isfinite(native).all().item(),
                        'Native frozen inference result invalid')
                    raw = F.interpolate(((native.float() + 1) / 2).clamp(0, 1), (256, 256),
                        mode='bilinear', align_corners=False)[0].permute(1, 2, 0).contiguous().numpy().copy()
                frozen(net, torch); resources = guard.check(); pixels = delivered(raw)
                output.record({'event': 'forward_returned', 'call': call, 'arm': arm, 'id': case['id'],
                    'seconds': time.monotonic() - started, 'resources': resources})
                prefix = 'outputs/' + MODEL_NAMES[arm] + '/' + case['id'] + '/'
                stream = io.BytesIO(); np.save(stream, raw, allow_pickle=False)
                raw_pin = output.put(prefix + 'raw.npy', stream.getvalue())
                png_pin = output.put(prefix + 'native.png', image_bytes(pixels))
                evidence = {'format': 'codeformer-filter-frozen-inference-output-v1', 'id': case['id'],
                    'reference_id': case['reference_id'], 'split': case['split'], 'profile': case['profile'],
                    'evidence': 'paired_synthetic_fixed_filter_stress', 'input': case['input'],
                    'pair_record': case['pair_record'], 'source1024': records[case['id']]['source1024'],
                    'target_bindings_not_scored_by_worker': records[case['id']]['targets'],
                    'input_review': case['input_review'], 'appearance_review_required': case['appearance_review_required'],
                    'input_review_does_not_authorize_training': True, 'optimization_permitted_now': False,
                    'arm': MODEL_NAMES[arm], 'model_binding': binding, 'architecture_sources': ARCHITECTURE,
                    'canonical515_state_sha256': before, 'raw': raw_pin, 'png': png_pin,
                    'RAW_array_sha256': digest(raw.tobytes()), 'PNG_RGB8_sha256': digest(pixels.tobytes()),
                    'RAW_shape': [256, 256, 3], 'RAW_dtype': 'float32', 'RAW_min': float(raw.min()), 'RAW_max': float(raw.max()),
                    'observed_support': 'all256x256 pixels', 'geometry': {'crop': None, 'alignment': None, 'warp': None},
                    'fidelity': 1., 'adain': True, 'detach_16': True, 'display_processing': 'none',
                    'forward_reservation': call, 'forward_seconds': time.monotonic() - started,
                    'eval_all_modules': True, 'parameters_require_grad': False, 'no_grad': True,
                    'device': 'cpu', 'internal_resolution': 512, 'output_resolution': 256,
                    'optimizer_updates': 0, 'backward_calls': 0, 'paired_clean_metrics': None, 'model_qualified': False}
                evidence_pin = output.put(prefix + 'evidence.json', encoded(evidence))
                case['outputs'][MODEL_NAMES[arm]] = {'raw': raw_pin, 'png': png_pin, 'evidence': evidence_pin}
                output.record({'event': 'output_committed', 'call': call, 'arm': arm, 'id': case['id'],
                    'evidence': evidence_pin})
                progress['committed_outputs'] += 1; progress['in_flight'] = None
                del x, sample, native, raw, pixels
                if call % 12 == 0:
                    print(json.dumps({'committed': progress['committed_outputs'], 'of': 216, 'arm': arm,
                        'seconds': time.monotonic() - guard.started, 'resources': guard.check()}), flush=True)
            frozen(net, torch); after = canonical_state(net.state_dict(), bindings['full515_tensor_schema'], torch)
            require(after == before == STATE_SHA[arm], 'Frozen state mutated during inference')
            bound(model_path, expected_sha, file['bytes'], allowed_links=3 if arm == 'original' else 1, load=False)
            proofs.append({'arm': MODEL_NAMES[arm], 'file': file, 'before_state_sha256': before,
                'after_state_sha256': after, 'exact515_state_unchanged': True, 'forwards': 72})
            output.record({'event': 'model_finished_frozen', 'arm': arm, 'state_sha256': after})
            del net; gc.collect(); guard.check()
        require(progress['reserved_forwards'] == progress['returned_forwards'] == progress['committed_outputs'] == 216
            and progress['in_flight'] is None and all(len(c['outputs']) == (3 if c['inference_planned'] else 0) for c in cases),
            'Exact216 completed output/24skip closure differs')
        output.put('model_state_proofs.json', encoded({'complete': True, 'models': proofs}))
        for pin in bindings['architecture_sources']:
            bound(ROOT / pin['path'], pin['sha256'], pin['bytes'], load=False)
        require(digest(Path(__file__).read_bytes()) == worker_sha, 'Worker source changed during inference')
        output.record({'event': 'complete', 'progress': progress, 'resources': guard.check()})
        output.close_journal()
        manifest = {'format': 'codeformer-filter-frozen-comparison-v1', 'complete': True,
            'data_manifest_sha256': DATA_SHA, 'root_input_review_sha256': REVIEW_SHA,
            'input_review_criteria_sha256': CRITERIA_SHA, 'model_binding_sha256': MODELS_SHA,
            'source_sha256': worker_sha, 'cases': cases, 'files': dict(output.files),
            'model_state_proofs': proofs, 'caps': CAPS, 'actual_forwards': 216, 'committed_outputs': 216,
            'all96_cases_retained': True, 'input_insufficient_cases_skipped': 24,
            'readable_cases_inferred': 28, 'borderline_cases_inferred': 44,
            'actual_training_updates': 0, 'actual_optimizer_steps': 0, 'actual_backward_calls': 0,
            'paired_metrics_computed': False, 'quality_gates_changed': False, 'model_qualified': False,
            'native_CCTV_or_identity_or_FINAL_claim': False, 'seconds': time.monotonic() - guard.started,
            'manifest_excludes_its_own_recursive_hash': True}
        final = output.put('manifest.json', encoded(manifest))
        require({p.relative_to(output.folder).as_posix() for p in output.folder.rglob('*') if p.is_file()} == set(output.files),
            'Unexpected output namespace member')
        for name, pin in output.files.items():
            guard.check()
            bound(output.folder / name, pin['sha256'], pin['bytes'], load=False)
        guard.check(); guard.stop.set(); guard.thread.join(timeout=2)
        return {'complete': True, 'actual_forwards': 216, 'committed_outputs': 216,
            'files': len(output.files), 'bytes': output.bytes, 'manifest': final,
            'seconds': time.monotonic() - guard.started, 'actual_training_updates': 0}
    except BaseException as error:
        guard.stop.set(); guard.failure(error)
        try:
            output.record({'event': 'failure', 'error': str(error), 'progress': progress})
            output.close_journal()
        except BaseException:
            pass
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-manifest', required=True, type=Path)
    parser.add_argument('--input-review', required=True, type=Path)
    parser.add_argument('--criteria', required=True, type=Path)
    parser.add_argument('--models', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.data_manifest, args.input_review, args.criteria, args.models, args.out)), flush=True)
