"""Guarded manual L4 supervisor and worker for the Unified R2 CodeFormer completion training run."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import io
import json
import math
import os
import random
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.completion_codeformer_unified_r2_v1.contracts import (
    CONTRACT,
    FORMAT,
    GRAY_KEYS,
    KINDS,
    METRIC_KEYS,
    OBJECTIVE_ID,
    ORIGINAL_BYTES,
    ORIGINAL_LOGICAL,
    ORIGINAL_SHA,
    PARENT_BYTES,
    PARENT_LOGICAL,
    PARENT_PROTOCOL_SHA,
    PARENT_SHA,
    STEM,
    SUBSET_SHA,
    TEACHER_INDEX_SHA,
    TYPED_CAPS,
    V4_PROTOCOL_SHA,
    VM_NAME,
    aggregate,
    code_route_name,
    compare as compare_metrics,
    compare_gray,
    frozen_prior_name,
    fusion_route_name,
    gray_style,
    lr_for_cycle,
    need,
    trainable_name,
)
from research.completion_codeformer_unified_r2_v1.optimized_budget import OutputBudget
from research.completion_codeformer_unified_r2_v1.terminal_admission import admit_terminal_zip
from research.completion_codeformer_unified_r2_v1.unified_objectives import (
    anchor_trust_penalty,
    delivered_float,
    inner_boundary,
    spatial_code_latent_loss,
    unified_fusion_loss,
)

WEIGHT_REL = Path('../cctv_completion_codeformer_gradient_v1_vm/weights/codeformer_inpainting.pth')
PARENT_REL = Path('../cctv_completion_codeformer_finetune_v1_r1_vm/outputs/terminal_training_state.pth')
SOURCE_MODULES = {
    'pretrained_completion': 'pretrained_completion.py',
    'third_party.codeformer.codeformer_arch': 'third_party/codeformer/codeformer_arch.py',
    'third_party.codeformer.vqgan_arch': 'third_party/codeformer/vqgan_arch.py',
    'research.completion_codeformer_unified_r2_v1.contracts': 'research/completion_codeformer_unified_r2_v1/contracts.py',
    'research.completion_codeformer_unified_r2_v1.unified_objectives': 'research/completion_codeformer_unified_r2_v1/unified_objectives.py',
    'research.completion_codeformer_unified_r2_v1.optimized_budget': 'research/completion_codeformer_unified_r2_v1/optimized_budget.py',
    'research.completion_codeformer_unified_r2_v1.terminal_admission': 'research/completion_codeformer_unified_r2_v1/terminal_admission.py',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def bad(value):
        raise ValueError('Nonfinite JSON constant ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=bad)


def regular(path):
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not path.is_symlink(), 'Not regular file: ' + str(path))
    return info


def member(root, relative):
    need(
        isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative,
        'Invalid path ' + str(relative),
    )
    pure = PurePosixPath(relative)
    need(not pure.is_absolute() and '..' not in pure.parts, 'Unsafe path ' + relative)
    target = root.joinpath(*pure.parts)
    for item in [target, *target.parents]:
        need(not item.is_symlink(), 'Symlink refused ' + str(item))
        if item == root:
            break
    need(target.resolve().is_relative_to(root.resolve()), 'Escaped root ' + relative)
    return target


def verify(root, expected):
    root = Path(root).resolve(strict=True)
    need(
        not any(x.is_symlink() for x in [root, *root.parents]),
        'Symlink root/parent refused',
    )
    protocol_path = member(root, 'protocol.json')
    regular(protocol_path)
    need(sha(protocol_path) == expected, 'Protocol SHA mismatch')
    p = read(protocol_path)
    need(
        p.get('format') == FORMAT
        and p.get('manual_execution_only') is True
        and p.get('automatic_resume_or_promotion') is False,
        'Invalid protocol header',
    )
    need(p.get('contract') == CONTRACT, 'Contract mismatch')
    need(
        p['source_subset']['source_V4_protocol_sha256'] == V4_PROTOCOL_SHA
        and p['source_subset']['only128_selected_FIT_observations'] is True,
        'Subset provenance mismatch',
    )
    listed = set()
    for row in p['files']:
        path = member(root, row['path'])
        info = regular(path)
        need(
            row['path'] not in listed
            and info.st_size == row['bytes']
            and sha(path) == row['sha256'],
            'Payload mismatch ' + row['path'],
        )
        listed.add(row['path'])
    for relative in SOURCE_MODULES.values():
        need(relative in listed, 'Unlisted module ' + relative)
    need(
        sha(member(root, p['balanced_subset'])) == SUBSET_SHA,
        'Balanced128 subset hash mismatch',
    )
    need(
        sha(member(root, p['original_parent_protocol'])) == PARENT_PROTOCOL_SHA,
        'Parent protocol hash mismatch',
    )
    actual = set()
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*dirs, *files]:
            candidate = current_path / name
            need(not candidate.is_symlink(), 'Symlink inside packet ' + str(candidate))
        for name in files:
            rel = (current_path / name).relative_to(root).as_posix()
            if rel in ('protocol.json', 'trainer.log', 'supervisor.json') or rel.startswith('outputs/'):
                continue
            actual.add(rel)
    need(actual == listed, 'Closed packet file set mismatch')

    refs = p['references']
    fit_refs = {r['id'] for r in refs if r['role'] == 'fit'}
    mon_refs = {r['id'] for r in refs if r['role'] == 'monitor'}
    need(
        len(refs) == 40
        and len(fit_refs) == 32
        and len(mon_refs) == 8
        and not (fit_refs & mon_refs),
        'Reference split mismatch',
    )
    targets = p['teacher_targets']
    target_ids = {r['id'] for r in targets}
    need(len(targets) == 80 and len(target_ids) == 80, 'Expected 80 styled targets')
    labels = p['cached_labels']
    need(
        len(labels) == 80 and {r['teacher_key'] for r in labels} == target_ids,
        'Expected 80 cached teacher label sets',
    )
    need(
        sum(not r['monitor_labels_never_optimized'] for r in labels) == 64
        and sum(r['monitor_labels_never_optimized'] for r in labels) == 16,
        'Cached label 64 FIT / 16 monitor split mismatch',
    )
    for row in labels:
        need(
            row['path'] in listed
            and row['teacher_original_logical_sha256'] == ORIGINAL_LOGICAL
            and row['original_teacher_weights_sha256'] == ORIGINAL_SHA,
            'Cached teacher label provenance mismatch',
        )

    fit = p['fit_cases']
    monitor = p['monitor_cases']
    challenge = p['challenge_cases']
    public = p['development_cases']
    parity = p['parity_cases']
    need(
        len(fit) == 128
        and len(monitor) == 32
        and len(challenge) == 16
        and len(public) == 32
        and len(parity) == 4,
        'Case counts differ',
    )
    subset = read(member(root, p['balanced_subset']))
    need(
        subset['complete']
        and subset['subset_cases'] == 128
        and subset['source_full_pool_cases'] == 256
        and p['fit_cases'] == subset['cases_in_reference_role_order']
        and p['schedule'] == subset['schedule'],
        'Use exact independent balanced128 schedule/cases',
    )
    need(
        sum(r['role'] == 'fit_anchor' for r in fit) == 64
        and sum(r['role'] == 'fit_augmented' for r in fit) == 64,
        'FIT 64 anchor / 64 augmented split differs',
    )
    for kind in KINDS:
        need(
            sum(r['mask_kind'] == kind for r in fit) == 32,
            'FIT geometry balance ' + kind,
        )
    need(
        sum(r['role'] == 'fit_augmented' and r['style'] == 'RGB' for r in fit) == 32
        and sum(r['role'] == 'fit_augmented' and r['style'] == 'GRAY' for r in fit) == 32,
        'Augmented RGB/GRAY balance differs',
    )
    need(
        all(r['reference_id'] in fit_refs and r['teacher_key'] in target_ids for r in fit),
        'FIT reference/teacher target leak',
    )
    need(
        all(r['reference_id'] in mon_refs and r['teacher_key'] in target_ids for r in monitor + challenge),
        'Monitor/challenge reference leak',
    )
    need(
        sum(r.get('style') == 'GRAY' for r in challenge) == 8
        and all(
            sum(r.get('style') == 'GRAY' and r['mask_kind'] == k for r in challenge) == 2
            for k in KINDS
        ),
        'Challenge strict-gray 8-case / 4-geometry split differs',
    )
    need(
        all(
            'reference_id' not in r and 'clean256' not in r and 'clean512' not in r
            for r in public
        ),
        'Public development must remain unpaired',
    )
    schedule = p['schedule']
    flat = [x for batch in schedule for x in batch]
    need(
        len(schedule) == 32
        and all(len(batch) == 4 and len(set(batch)) == 4 for batch in schedule)
        and len(flat) == 128
        and set(flat) == {r['id'] for r in fit},
        '32x4 balanced schedule mismatch',
    )
    hist = p['historical_control']['initial_rows']
    need(
        len(hist['monitor']) == 32 and len(hist['challenge']) == 16,
        '48 historical R1 controls required',
    )
    for rows in hist.values():
        for row in rows:
            need(
                row['result256'] in listed and row['PNG'] in listed,
                'Unlisted historical control',
            )

    checkpoint = None
    parent = None
    w = (root / WEIGHT_REL)
    if w.exists():
        info = regular(w.resolve(strict=True))
        need(
            info.st_size == ORIGINAL_BYTES and sha(w) == ORIGINAL_SHA,
            'Original inpainting checkpoint hash/size mismatch',
        )
        checkpoint = w.resolve(strict=True)
    parent_candidate = (root / PARENT_REL)
    if parent_candidate.exists():
        info = regular(parent_candidate.resolve(strict=True))
        need(
            info.st_size == PARENT_BYTES and sha(parent_candidate) == PARENT_SHA,
            'Parent R1 terminal state hash/size mismatch',
        )
        parent = parent_candidate.resolve(strict=True)
    return p, checkpoint, parent


def metadata_get(suffix):
    req = urllib.request.Request(
        'http://metadata.google.internal/computeMetadata/v1/' + suffix,
        headers={'Metadata-Flavor': 'Google'},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode('utf-8').strip()


def require_vm(root):
    need(
        os.name == 'posix' and Path(root).resolve() == (Path.home() / 'forensic-dgp' / VM_NAME).resolve(),
        'Run only inside ~/forensic-dgp/' + VM_NAME,
    )
    machine = metadata_get('instance/machine-type').rsplit('/', 1)[-1]
    need(machine == 'g2-standard-4', 'Expected g2-standard-4, got ' + machine)
    gpu = [
        x.strip()
        for x in subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True,
            timeout=15,
        ).splitlines()
        if x.strip()
    ]
    need(gpu == ['NVIDIA L4'], 'Expected one NVIDIA L4, got ' + repr(gpu))
    pids = [
        int(x.strip())
        for x in subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            text=True,
            timeout=15,
        ).splitlines()
        if x.strip()
    ]
    need(
        not [pid for pid in pids if pid not in {os.getpid(), os.getppid()}],
        'Competing GPU process detected',
    )
    return {'machine_type': machine, 'GPU': gpu[0]}


def worker(root, p, checkpoint, parent_path, expected):
    hardware = require_vm(root)
    need(
        checkpoint is not None and parent_path is not None,
        'Both original inpainting checkpoint and R1 parent state must exist on VM',
    )

    def stop_signal(signum, frame):
        raise KeyboardInterrupt('Supervisor signal ' + str(signum))

    old_sigterm = signal.signal(signal.SIGTERM, stop_signal)
    need(
        shutil.disk_usage(root).free >= CONTRACT['minimum_free_bytes'],
        'Need 8 GiB free before pilot',
    )
    out = root / 'outputs'
    need(not out.exists(), 'Never overwrite/resume existing outputs')
    out.mkdir()
    started = time.monotonic()
    counts = {
        'forward_slots': 0,
        'backward_calls': 0,
        'code_optimizer_steps': 0,
        'fusion_optimizer_steps': 0,
        'empty_bypasses': 0,
    }
    typed = {k: 0 for k in TYPED_CAPS}
    budget = OutputBudget(out, started)
    budget.metadata_external = [root / 'trainer.log', root / 'protocol.json']

    def write(path, value, *, emergency=False):
        budget.json(path, value, emergency=emergency)

    saved_npz_count = 0

    def save_npz(path, **arrays):
        nonlocal saved_npz_count
        need(
            saved_npz_count < CONTRACT['saved_generated_NPZ_max'] and len(arrays) > 0,
            'Generated NPZ count cap exceeded',
        )
        entries = []
        for name, value in arrays.items():
            need(
                isinstance(name, str)
                and 0 < len((name + '.npy').encode('utf-8')) <= 80
                and '/' not in name
                and '\\' not in name
                and ':' not in name,
                'Invalid NPZ member name',
            )
            arr = np.asarray(value)
            need(
                arr.dtype.kind in ('f', 'i', 'u', 'b') and not arr.dtype.hasobject,
                'Refusing object NPZ array',
            )
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header, np.lib.format.header_data_from_array_1_0(arr)
            )
            need(header.tell() <= 512, 'NPZ NPY header exceeds 512B')
            entries.append((name + '.npy', header.getvalue(), memoryview(np.ascontiguousarray(arr)).cast('B')))
        with budget.open(path) as stream:
            with zipfile.ZipFile(stream, mode='w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for member_name, header_bytes, body_view in entries:
                    with archive.open(member_name, mode='w') as item:
                        item.write(header_bytes)
                        item.write(body_view)
        saved_npz_count += 1

    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F

    need(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(240)
    np.random.seed(240)
    torch.manual_seed(240)
    torch.cuda.manual_seed_all(240)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(root))
    for module, relative in SOURCE_MODULES.items():
        loaded = importlib.import_module(module)
        need(
            Path(loaded.__file__).resolve() == (root / relative).resolve(),
            'Imported module escaped packet: ' + module,
        )
    CodeFormer = importlib.import_module('third_party.codeformer.codeformer_arch').CodeFormer
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    need(
        isinstance(loaded, dict) and isinstance(loaded.get('params_ema'), dict),
        'Expected original params_ema',
    )
    net = CodeFormer(
        dim_embd=512,
        codebook_size=512,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128'],
    )
    net.load_state_dict(loaded['params_ema'], strict=True)
    del loaded
    net = net.to('cuda', dtype=torch.float32).eval().requires_grad_(False)

    def guard(*, terminal_save=False):
        budget.guard(terminal=terminal_save)
        if terminal_save:
            return
        need(
            counts['forward_slots'] <= CONTRACT['hard_forward_cap']
            and counts['backward_calls'] <= CONTRACT['backward_calls_newphase'],
            'Forward/backward cap exceeded',
        )
        need(
            counts['code_optimizer_steps'] <= CONTRACT['code_optimizer_steps_newphase']
            and counts['fusion_optimizer_steps'] <= CONTRACT['fusion_optimizer_steps_newphase']
            and counts['empty_bypasses'] <= CONTRACT['empty_bypasses'],
            'Step/bypass cap exceeded',
        )
        for key, cap in TYPED_CAPS.items():
            need(typed[key] <= cap, 'Typed forward cap exceeded: ' + key)
        need(
            torch.cuda.memory_allocated() <= CONTRACT['CUDA_allocated_bytes_cap'],
            'CUDA allocation cap exceeded',
        )
        need(all(not module.training for module in net.modules()), 'All modules must stay in eval mode')
        need(
            all(
                not value.requires_grad and value.grad is None
                for name, value in net.named_parameters()
                if frozen_prior_name(name)
            ),
            'Frozen 165 prior parameters must keep requires_grad=False and grad=None',
        )

    def forward(kind, call):
        need(not budget.terminal_committed, 'No neural calls after terminal commit')
        need(
            time.monotonic() - started < CONTRACT['cooperative_stop_seconds'],
            'No neural calls after cooperative work deadline',
        )
        guard()
        need(
            typed[kind] + 1 <= TYPED_CAPS[kind]
            and counts['forward_slots'] + 1 <= CONTRACT['hard_forward_cap']
            and model_forwards['calls'] + 1 <= CONTRACT['hard_forward_cap'],
            'Forward cap reached before dispatch: ' + kind,
        )
        counts['forward_slots'] += 1
        typed[kind] += 1
        guard()
        return call()

    def digest_state(prefixes=None, *, check_finite=True):
        digest = hashlib.sha256()
        for name, value in sorted(net.state_dict().items()):
            if prefixes and not any(name.startswith(prefix) for prefix in prefixes):
                continue
            arr = value.detach().cpu().contiguous()
            if check_finite:
                need(torch.isfinite(arr).all().item(), 'Nonfinite model state: ' + name)
            digest.update(name.encode())
            digest.update(str(tuple(arr.shape)).encode())
            digest.update(str(arr.dtype).encode())
            digest.update(arr.numpy().tobytes())
        return digest.hexdigest()

    original_teacher_state = digest_state()
    frozen_state = digest_state(('quantize.', 'generator.'))

    def image(path, size):
        with Image.open(root / path) as im:
            need(
                im.format == 'PNG' and im.size == (size, size) and im.mode == 'RGB',
                'Exact RGB PNG required: ' + str(path),
            )
            arr = np.array(im, copy=True)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to('cuda', torch.float32) / 255.0

    def mask(path):
        with Image.open(root / path) as im:
            need(
                im.format == 'PNG' and im.size == (256, 256) and im.mode == 'L',
                'Exact L256 binary mask required: ' + str(path),
            )
            arr = np.array(im, copy=True)
            need(np.isin(arr, [0, 255]).all(), 'Mask not binary: ' + str(path))
        return torch.from_numpy(arr).to('cuda', torch.float32)[None, None] / 255.0

    def app_input(rgb, m):
        active = m.sum((1, 2, 3)) > 0
        need(
            rgb.shape[0] == 1 and bool(active.all().item()),
            'Single active case required',
        )
        visible = 1.0 - m[active]
        support = F.interpolate(visible, size=(512, 512), mode='bilinear', align_corners=False)
        x = F.interpolate(rgb[active] * visible, size=(512, 512), mode='bilinear', align_corners=False)
        x = x / support.clamp_min(1e-8)
        m512 = F.interpolate(m[active], size=(512, 512), mode='nearest')
        return (x * (1.0 - m512) + m512) * 2.0 - 1.0, m512

    def delivered(rgb, m, raw):
        return delivered_float(raw, rgb, m, interpolate=F.interpolate)

    def save_pixels(path, value):
        arr = np.rint(value.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        with budget.open(path) as stream:
            Image.fromarray(arr).save(stream, format='PNG')
        return arr

    def boundary(m):
        return inner_boundary(m, F=F)

    references = {r['id']: r for r in p['references']}
    clean = {}
    teacher_fit = {}
    teacher_monitor_codes = {}
    fit_cases = {r['id']: r for r in p['fit_cases']}
    schedule = p['schedule']

    code_groups = {
        'encoder': list(net.encoder.parameters()),
        'position': [net.position_emb],
        'feature': list(net.feat_emb.parameters()),
        'transformer': list(net.ft_layers.parameters()),
        'prediction': list(net.idx_pred_layer.parameters()),
    }
    fusion_groups = {size: list(net.fuse_convs_dict[size].parameters()) for size in ('32', '64', '128')}
    code_parameters = [param for group in code_groups.values() for param in group]
    fusion_parameters = [param for group in fusion_groups.values() for param in group]
    parameter_names = {id(parameter): name for name, parameter in net.named_parameters()}
    optimizer_parameter_names = {
        'code': [parameter_names[id(parameter)] for parameter in code_parameters],
        'fusion': [parameter_names[id(parameter)] for parameter in fusion_parameters],
    }
    need(
        len(code_parameters) == 278 and len(fusion_parameters) == 54,
        'Expected 278 code + 54 fusion = 332 trainable parameter tensors',
    )
    code_optimizer = torch.optim.Adam(
        code_parameters, lr=1e-5, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0
    )
    fusion_optimizer = torch.optim.Adam(
        fusion_parameters, lr=1e-5, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0
    )

    def active(groups):
        net.requires_grad_(False)
        for parameters in groups.values():
            for parameter in parameters:
                parameter.requires_grad_(True)

    def set_lr(optimizer, lr_value: float):
        for group in optimizer.param_groups:
            group['lr'] = float(lr_value)

    def gradient_receipt(groups):
        receipt = {}
        for name, parameters in groups.items():
            need(
                all(param.grad is not None for param in parameters),
                'Disconnected parameter group: ' + name,
            )
            need(
                all(torch.isfinite(param.grad).all().item() for param in parameters),
                'Nonfinite gradients in group: ' + name,
            )
            norm = math.sqrt(
                sum(float(param.grad.detach().double().square().sum().item()) for param in parameters)
            )
            receipt[name] = {
                'L2': norm,
                'tensors': len(parameters),
                'finite': True,
                'nonzero': norm > 0,
            }
        return receipt

    def verify_active_adam(optimizer, parameters, names, steps_done):
        need(
            len(optimizer.param_groups) == 1 and len(optimizer.state) == len(parameters),
            'Active Adam map size mismatch',
        )
        for name, parameter in zip(names, parameters):
            slot = optimizer.state[parameter]
            need(
                set(slot) == {'step', 'exp_avg', 'exp_avg_sq'},
                'Adam slot schema mismatch: ' + name,
            )
            step = slot['step']
            need(
                isinstance(step, torch.Tensor)
                and step.shape == ()
                and step.dtype == torch.float32
                and torch.isfinite(step).item()
                and float(step.item()) == 32 + steps_done,
                'Adam step mismatch: ' + name,
            )
            for key in ('exp_avg', 'exp_avg_sq'):
                moment = slot[key]
                need(
                    moment.shape == parameter.shape
                    and moment.dtype == parameter.dtype == torch.float32
                    and moment.device == parameter.device
                    and torch.isfinite(moment).all().item(),
                    'Invalid Adam moment: ' + name + '.' + key,
                )
            need(
                (slot['exp_avg_sq'] >= 0).all().item(),
                'Negative Adam second moment: ' + name,
            )
        return True

    snapshots = []
    terminal_cycle = 0
    stop_reason = None
    parity = []
    fit_log = []
    parent_optimizer_restored = False
    parent_RNG_restored = False
    parent_metadata = {}
    resume_rng = None
    phase_events = []
    terminal_attempted = False
    r1_code_anchors = []
    r1_fusion_anchors = []
    progress = {
        'cycle': 0,
        'route': 'initialization',
        'microbatch': 0,
        'phase': 'before_first_update',
    }
    model_forwards = {'calls': 0, 'generator_calls': 0, 'code_only_calls': 0}

    def model_hook(module, args, kwargs):
        model_forwards['calls'] += 1
        if kwargs.get('code_only', False):
            model_forwards['code_only_calls'] += 1
        else:
            model_forwards['generator_calls'] += 1

    forward_hook = net.register_forward_pre_hook(model_hook, with_kwargs=True)

    def state_receipt():
        tensors = {}
        for name, value in sorted(net.state_dict().items()):
            cpu = value.detach().cpu().contiguous()
            tensors[name] = {
                'shape': list(cpu.shape),
                'dtype': str(cpu.dtype),
                'sha256': hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                'finite': bool(torch.isfinite(cpu).all().item()),
                'frozen_in_newphase': frozen_prior_name(name),
            }
        return {
            'logical_sha256': digest_state(check_finite=False),
            'tensors': tensors,
            'finite': all(v['finite'] for v in tensors.values()),
            'state_tensors': len(tensors),
            'frozen_tensors': sum(v['frozen_in_newphase'] for v in tensors.values()),
            'trainable_tensors': sum(not v['frozen_in_newphase'] for v in tensors.values()),
            'all_modules_eval': all(not module.training for module in net.modules()),
            'frozen_requires_grad_false': all(
                not value.requires_grad
                for name, value in net.named_parameters()
                if frozen_prior_name(name)
            ),
            'frozen_parameter_gradients_none': all(
                value.grad is None
                for name, value in net.named_parameters()
                if frozen_prior_name(name)
            ),
        }

    def phase_start(name, *, emergency=False):
        value = {
            'phase': name,
            'start_seconds': time.monotonic() - started,
            'complete': False,
        }
        phase_events.append(value)
        write(
            out / ('phase_%03d_%s_start.json' % (len(phase_events), name)),
            value,
            emergency=emergency,
        )
        return value

    def phase_end(value, **details):
        value.update(
            end_seconds=time.monotonic() - started,
            complete=True,
            **details,
        )
        value['seconds'] = value['end_seconds'] - value['start_seconds']
        write(
            out / ('phase_%03d_%s_end.json' % (phase_events.index(value) + 1, value['phase'])),
            value,
            emergency=True,
        )

    def cpu_copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: cpu_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu_copy(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu_copy(v) for v in value)
        return value

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(tensor_bytes(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(v) for v in value)
        return 0

    def save_state(cycle, reason):
        nonlocal terminal_attempted
        need(
            not terminal_attempted,
            'Only one bounded terminal save attempt; retain any partial evidence',
        )
        terminal_attempted = True
        actual_prior = digest_state(('quantize.', 'generator.'), check_finite=False)
        current_logical = digest_state(check_finite=False)
        if not (out / 'actual_state_at_save.json').exists():
            write(out / 'actual_state_at_save.json', state_receipt(), emergency=True)
        retention_failures = []
        if actual_prior != frozen_state:
            retention_failures.append('original generator/codebook changed')
        if not all(not module.training for module in net.modules()):
            retention_failures.append('module eval-mode invariant violated')
        if not all(
            not value.requires_grad
            for name, value in net.named_parameters()
            if frozen_prior_name(name)
        ):
            retention_failures.append('frozen 165 prior requires_grad invariant violated')
        unexpected_frozen_gradients = [
            name
            for name, value in net.named_parameters()
            if frozen_prior_name(name) and value.grad is not None
        ]
        if unexpected_frozen_gradients:
            retention_failures.append(
                'unexpected frozen prior gradients: ' + ','.join(unexpected_frozen_gradients)
            )
        if (
            counts['forward_slots'] > CONTRACT['hard_forward_cap']
            or counts['backward_calls'] > CONTRACT['backward_calls_newphase']
            or counts['code_optimizer_steps'] > CONTRACT['code_optimizer_steps_newphase']
            or counts['fusion_optimizer_steps'] > CONTRACT['fusion_optimizer_steps_newphase']
            or counts['empty_bypasses'] > CONTRACT['empty_bypasses']
            or any(typed[k] > TYPED_CAPS[k] for k in typed)
        ):
            retention_failures.append('actual execution counters exceed declared finite bounds')
        if torch.cuda.memory_allocated() > CONTRACT['CUDA_allocated_bytes_cap']:
            retention_failures.append('actual CUDA allocation exceeds cap at save')
        if not all(torch.isfinite(v).all().item() for v in net.state_dict().values()):
            retention_failures.append('nonfinite current model state')

        assembly = phase_start('terminal_CPU_copy_and_assembly', emergency=True)
        delta = {
            n: v
            for n, v in net.state_dict().items()
            if not frozen_prior_name(n)
        }
        pending = {
            'trainable_state': delta,
            'failed_frozen_prior_state': (
                {n: v for n, v in net.state_dict().items() if frozen_prior_name(n)}
                if actual_prior != frozen_state
                else {}
            ),
            'code_optimizer': code_optimizer.state_dict(),
            'fusion_optimizer': fusion_optimizer.state_dict(),
            'partial_accumulated_gradients': {
                n: v.grad
                for n, v in net.named_parameters()
                if v.grad is not None
            },
        }
        planned_tensor_bytes = tensor_bytes(pending)
        need(
            planned_tensor_bytes < CONTRACT['terminal_bytes'],
            'Planned complete recovery payload exceeds 1 GiB bound',
        )
        write(
            out / 'pre_save_receipt.json',
            {
                'complete': False,
                'reason': reason,
                'cycle': cycle,
                'counts': dict(counts),
                'logical_state_sha256': current_logical,
                'frozen_prior_sha256': frozen_state,
                'planned_tensor_payload_bytes': planned_tensor_bytes,
                'remaining_hard_seconds': max(0.0, 1800.0 - (time.monotonic() - started)),
                'serialization_started': False,
                'retained_failure_gates': retention_failures,
            },
            emergency=True,
        )
        snapshot = cpu_copy(pending)
        snapshot.update(
            {
                'format': 'completion-codeformer-unified-r2-v1-trainable-resume-v1',
                'parent_training_state_sha256': PARENT_SHA,
                'parent_protocol_sha256': PARENT_PROTOCOL_SHA,
                'parent_initial_logical_sha256': PARENT_LOGICAL,
                'parent_optimizer_restored': parent_optimizer_restored,
                'parent_RNG_restored': parent_RNG_restored,
                'parent_metadata': parent_metadata,
                'phase_completed_paired_cycles': terminal_cycle,
                'cumulative_completed_paired_cycles': 32 + terminal_cycle,
                'phase_name': 'unified_r2_v1',
                'base_checkpoint_sha256': ORIGINAL_SHA,
                'protocol_sha256': expected,
                'cycle': cycle,
                'progress': dict(progress),
                'resume_progress_requires_new_manual_audit': True,
                'optimizer_parameter_names': optimizer_parameter_names,
                'completed_paired_cycles': 32 + terminal_cycle,
                'frozen_state_sha256': frozen_state,
                'logical_state_sha256': current_logical,
                'actual_prior_sha256': actual_prior,
                'retained_failure_gates': retention_failures,
                'state_is_qualified': False,
                'torch_CPU_RNG': torch.get_rng_state(),
                'CUDA_RNG': torch.cuda.get_rng_state_all(),
                'python_random_state': random.getstate(),
                'numpy_random_state': {
                    'algorithm': np.random.get_state()[0],
                    'keys': np.random.get_state()[1].tolist(),
                    'position': np.random.get_state()[2],
                    'has_gauss': np.random.get_state()[3],
                    'cached_gaussian': np.random.get_state()[4],
                },
                'sampler_schedule': schedule,
                'next_cycle': terminal_cycle + 1,
                'next_clean_sampler_index': terminal_cycle * 4,
                'counts': dict(counts),
                'contract': CONTRACT,
                'reason': reason,
                'automatic_resume_permitted': False,
                'model_qualified': False,
            }
        )
        phase_end(assembly, tensor_payload_bytes=planned_tensor_bytes)
        path = out / 'terminal_training_state.pth'
        need(not path.exists(), 'Terminal state already exists')
        partial = out / 'terminal_training_state.pth.partial'
        serial = phase_start('terminal_serialization', emergency=True)
        before_callbacks = budget.check_counts['write_checks']
        with budget.open(partial, terminal=True) as stream:
            torch.save(snapshot, stream)
            serialize_end = time.monotonic() - started
            flush_start = time.monotonic()
            stream.flush()
            os.fsync(stream.stream.fileno())
            flush_seconds = time.monotonic() - flush_start
        phase_end(
            serial,
            serialized_bytes=partial.stat().st_size,
            write_callbacks=budget.check_counts['write_checks'] - before_callbacks,
            serializer_finished_seconds=serialize_end,
            flush_fsync_seconds=flush_seconds,
        )
        ZIP_admission = admit_terminal_zip(partial, guard=lambda: guard(terminal_save=True))
        write(out / 'terminal_internal_ZIP_admission.json', ZIP_admission, emergency=True)
        guard(terminal_save=True)
        commit = phase_start('terminal_hash_and_atomic_commit', emergency=True)
        digest = sha(partial)
        need(not path.exists(), 'Committed state already exists')
        os.link(partial, path)
        need(
            path.stat().st_ino == partial.stat().st_ino,
            'Exclusive terminal publication identity differs',
        )
        partial.unlink()
        budget.rename(partial, path)
        budget.assert_ledger_consistent()
        phase_end(commit, bytes=path.stat().st_size, sha256=digest, ledger_total=budget.total())
        return {
            'path': path.name,
            'bytes': path.stat().st_size,
            'sha256': digest,
            'cycle': cycle,
            'logical_state_sha256': current_logical,
            'frozen_prior_sha256': frozen_state,
            'retained_failure_gates': retention_failures,
        }

    def evaluate(cycle, rows, kind):
        evaluation_phase = phase_start('evaluate_%s_u%04d' % (kind, cycle))
        folder = out / ('u%04d' % cycle) / kind
        folder.mkdir(parents=True, exist_ok=False)
        net.requires_grad_(False)
        records = []
        with torch.no_grad():
            for case in rows:
                rgb = image(case['input256'], 256)
                target = image(case['clean256'], 256) if kind != 'public_development' else None
                m = mask(case['mask'])
                guard()
                if not m.any().item():
                    counts['empty_bypasses'] += 1
                    out_tensor = rgb.clone()
                    raw = None
                    predicted = None
                else:
                    x, m512 = app_input(rgb, m)
                    raw, logits, _ = forward(
                        kind, lambda: net(x, w=1, detach_16=True, adain=False)
                    )
                    _, indices = torch.topk(F.softmax(logits, dim=2), 1, dim=2)
                    predicted = indices.squeeze(-1)
                    out_tensor = delivered(rgb, m, raw)
                need(
                    torch.isfinite(out_tensor).all().item()
                    and torch.equal(out_tensor * (1.0 - m), rgb * (1.0 - m)),
                    'Visible raw preservation failed',
                )
                array = save_pixels(folder / (case['id'] + '.png'), out_tensor)
                observed = np.rint(rgb[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                support = m[0, 0].cpu().numpy().astype(bool)
                need(
                    np.array_equal(array[~support], observed[~support]),
                    'PNG visible preservation failed',
                )
                record = {
                    'id': case['id'],
                    'mask_kind': case.get('mask_kind'),
                    'family': case.get('family'),
                    'condition': case.get('condition'),
                    'style': case.get('style'),
                    'PNG': str((folder / (case['id'] + '.png')).relative_to(out)),
                    'PNG_sha256': sha(folder / (case['id'] + '.png')),
                    'outside_exact': True,
                    'empty': not bool(support.any()),
                }
                if target is not None and support.any():
                    t_codes = teacher_monitor_codes[case['teacher_key']]
                    record['same_codebook_clean_context_agreement_diagnostic'] = float(
                        (predicted == t_codes).float().mean().item()
                    )
                    b = boundary(m)
                    need(b.sum().item() > 0, 'Empty boundary')
                    target_array = np.rint(target[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(
                        np.uint8
                    )
                    difference = (out_tensor - target).abs()
                    record.update(
                        {
                            'raw_hole_L1': float(
                                (difference * m).sum().item() / (m.sum().item() * 3.0)
                            ),
                            'raw_boundary_L1': float(
                                (difference * b).sum().item() / (b.sum().item() * 3.0)
                            ),
                            'PNG_hole_L1': float(
                                abs(array.astype(np.float64) - target_array)[support].mean() / 255.0
                            ),
                            'PNG_boundary_L1': float(
                                abs(array.astype(np.float64) - target_array)[
                                    b[0, 0].cpu().numpy().astype(bool)
                                ].mean()
                                / 255.0
                            ),
                        }
                    )
                    if case.get('style') == 'GRAY':
                        raw_rgb = out_tensor[0].permute(1, 2, 0).cpu().numpy()
                        record['GRAY_raw_mean_channel_spread'] = float(
                            np.ptp(raw_rgb, axis=-1)[support].mean()
                        )
                        record['GRAY_PNG_mean_channel_spread'] = float(
                            np.ptp(array.astype(np.int16), axis=-1)[support].mean()
                        )
                        record['GRAY_chroma_diagnostic_not_selection_gate'] = False
                        record['GRAY_final32_style_gate_required'] = True
                save_npz(
                    folder / (case['id'] + '.npz'),
                    input256=rgb.detach().cpu().numpy(),
                    mask256=m.detach().cpu().numpy(),
                    result256=out_tensor.detach().cpu().numpy(),
                    raw512=raw.detach().cpu().numpy() if raw is not None else np.empty((0,), np.float32),
                    predicted_codes=(
                        predicted.cpu().numpy() if predicted is not None else np.empty((0,), np.int64)
                    ),
                )
                records.append(record)
                guard()
        summary = {
            'cycle': cycle,
            'kind': kind,
            'records': records,
            'counts': dict(counts),
            'model_qualified': False,
        }
        if kind != 'public_development':
            summary['metrics'] = aggregate(
                [r for r in records if not r['empty']], condition=(kind == 'challenge')
            )
        if kind == 'challenge':
            summary['GRAY_style'] = gray_style(summary)
        write(folder / 'results.json', summary)
        phase_end(evaluation_phase, cases=len(records), counts=dict(counts))
        return summary

    def empty_controls(cycle):
        bypass = importlib.import_module('pretrained_completion').CodeFormerCompletion(net)
        before_calls = model_forwards['calls']
        for ref in references:
            rgb = clean[ref + '::RGB'][0]
            out_rgb = bypass(rgb, torch.zeros_like(rgb[:, :1]))
            need(torch.equal(rgb, out_rgb), 'Empty control mismatch')
            counts['empty_bypasses'] += 1
        need(
            model_forwards['calls'] == before_calls,
            'Empty control unexpectedly called neural model',
        )
        write(
            out / ('empty_controls_u%04d.json' % cycle),
            {
                'references': sorted(references),
                'exact_bypasses': 40,
                'model_calls': 0,
                'clear_glasses_and_uncovered_control': (
                    'No removal means exact source bypass; functional preservation'
                ),
            },
        )

    def compare(summary, baseline, *, conditions=False):
        return compare_metrics(summary, baseline, conditions=conditions)

    try:
        write(
            out / 'initialization_timing.json',
            {
                'complete': True,
                'phase': 'imports_original_model_initialization_and_dual_optimizer_construction',
                'start_seconds': 0.0,
                'end_seconds': time.monotonic() - started,
                'typed_neural_calls': 0,
                'optimizer_steps': 0,
            },
        )
        need(
            original_teacher_state == ORIGINAL_LOGICAL,
            'Original base logical state differs',
        )
        target_phase = phase_start('styled_target_and_cached_teacher_loading')
        for row in p['teacher_targets']:
            c256 = image(row['clean256'], 256)
            c512 = image(row['clean512'], 512)
            clean[row['id']] = (c256, c512)
            if row['style'] == 'GRAY':
                need(
                    torch.equal(c256[:, 0], c256[:, 1])
                    and torch.equal(c256[:, 1], c256[:, 2])
                    and torch.equal(c512[:, 0], c512[:, 1])
                    and torch.equal(c512[:, 1], c512[:, 2]),
                    'GRAY targets not exact gray',
                )
        teacher_folder = out / 'teacher_labels'
        teacher_folder.mkdir()
        fit_keys = {r['teacher_key'] for r in p['fit_cases']}
        mon_keys = {r['teacher_key'] for r in p['monitor_cases'] + p['challenge_cases']}
        need(
            len(fit_keys) == 64 and len(mon_keys) == 16 and not (fit_keys & mon_keys),
            'FIT/monitor teacher key split mismatch',
        )
        label_records = []
        for row in p['cached_labels']:
            src = root / row['path']
            need(
                regular(src).st_size == row['bytes'] and sha(src) == row['sha256'],
                'Cached teacher label hash mismatch: ' + row['path'],
            )
            with np.load(src, allow_pickle=False) as archive:
                need(
                    set(archive.files) == {'nearest_codes', 'prequant_latent'},
                    'Teacher archive schema differs: ' + row['path'],
                )
                codes = archive['nearest_codes'].copy()
                latent = archive['prequant_latent'].copy()
            need(
                codes.dtype == np.int64
                and codes.shape == (1, 256)
                and int(codes.min()) >= 0
                and int(codes.max()) < 512,
                'Invalid teacher nearest_codes: ' + row['path'],
            )
            need(
                latent.dtype == np.float32
                and latent.shape == (1, 256, 16, 16)
                and np.isfinite(latent).all(),
                'Invalid teacher prequant_latent: ' + row['path'],
            )
            dst = teacher_folder / Path(row['path']).name
            with src.open('rb') as incoming, budget.open(dst) as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            need(sha(dst) == row['sha256'], 'Retained teacher copy drift')
            codes_gpu = torch.from_numpy(codes).to('cuda', torch.long)
            latent_gpu = torch.from_numpy(latent).to('cuda', torch.float32)
            key = row['teacher_key']
            if key in fit_keys:
                need(
                    not row['monitor_labels_never_optimized'],
                    'FIT teacher label marked as monitor',
                )
                teacher_fit[key] = (codes_gpu, latent_gpu)
            else:
                need(
                    key in mon_keys and row['monitor_labels_never_optimized'],
                    'Monitor teacher label flag mismatch',
                )
                teacher_monitor_codes[key] = codes_gpu
            label_records.append(
                {
                    'id': key,
                    'path': str(dst.relative_to(out)),
                    'bytes': dst.stat().st_size,
                    'sha256': row['sha256'],
                    'fit_optimized': key in fit_keys,
                    'monitor_diagnostic_only': key in mon_keys,
                }
            )
            guard()
        need(
            len(teacher_fit) == 64 and len(teacher_monitor_codes) == 16,
            'Loaded teacher FIT/monitor counts differ',
        )
        write(
            teacher_folder / 'index.json',
            {
                'complete': True,
                'source_coverage_v2_teacher_index_sha256': TEACHER_INDEX_SHA,
                'teacher_neural_calls': 0,
                'fit_sets_used_for_code_and_latent': 64,
                'monitor_sets_diagnostic_only': 16,
                'labels': label_records,
            },
        )
        phase_end(target_phase, target_sets=len(clean), cached_labels=len(label_records), teacher_neural_calls=0)

        parent_phase = phase_start('parent_state_load')
        parent = torch.load(parent_path, map_location='cpu', weights_only=True)
        phase_end(parent_phase, parent_bytes=parent_path.stat().st_size)

        reconstruction_phase = phase_start('R1_state_reconstruction')
        need(
            parent['protocol_sha256'] == PARENT_PROTOCOL_SHA
            and parent['base_checkpoint_sha256'] == ORIGINAL_SHA
            and parent['logical_state_sha256'] == PARENT_LOGICAL,
            'Parent R1 fingerprint differs',
        )
        need(
            parent['cycle'] == parent['completed_paired_cycles'] == 32
            and parent['partial_accumulated_gradients'] == {}
            and parent['reason'] == 'completed_finite_pilot',
            'Only audited completed R1 parent allowed',
        )
        need(
            parent['optimizer_parameter_names'] == optimizer_parameter_names,
            'Named optimizer order differs from R1',
        )
        need(
            parent['counts']
            == {
                'forward_slots': 528,
                'backward_calls': 256,
                'code_optimizer_steps': 32,
                'fusion_optimizer_steps': 32,
                'empty_bypasses': 168,
            },
            'Parent R1 counters differ',
        )
        parent_protocol = read(root / p['original_parent_protocol'])
        need(
            parent['contract'] == parent_protocol['contract']
            and parent['sampler_schedule'] == parent_protocol['schedule'],
            'Parent contract/sampler differs',
        )
        original_state = net.state_dict()
        expected_names = {n for n in original_state if not frozen_prior_name(n)}
        need(
            set(parent['trainable_state']) == expected_names and len(expected_names) == 332,
            'Parent 332 trainable partition differs',
        )
        net.load_state_dict({**original_state, **parent['trainable_state']}, strict=True)
        need(
            digest_state() == PARENT_LOGICAL
            and digest_state(('quantize.', 'generator.')) == frozen_state == parent['frozen_state_sha256'],
            'R1 reconstruction or frozen prior differs',
        )
        r1_code_anchors = [param.detach().clone() for param in code_parameters]
        r1_fusion_anchors = [param.detach().clone() for param in fusion_parameters]
        phase_end(reconstruction_phase, logical_state_sha256=digest_state())

        optimizer_phase = phase_start('named_Adam_restore')
        optimizer_admission = {}
        for route, optimizer, params in [
            ('code', code_optimizer, code_parameters),
            ('fusion', fusion_optimizer, fusion_parameters),
        ]:
            state = parent[route + '_optimizer']
            need(
                len(state['param_groups']) == 1 and len(state['state']) == len(params),
                'Parent Adam map differs: ' + route,
            )
            group = state['param_groups'][0]
            need(
                group['lr'] == 1e-5
                and tuple(group['betas']) == (0.9, 0.99)
                and group['eps'] == 1e-8
                and group['weight_decay'] == 0,
                'Parent Adam hyperparameters differ: ' + route,
            )
            optimizer.load_state_dict(state)
            verify_active_adam(optimizer, params, optimizer_parameter_names[route], 0)
            optimizer_admission[route] = {
                'parameter_tensors': len(params),
                'step_before': 32,
                'named_order_exact': True,
                'moments_exact': True,
                'lr_in_R1': group['lr'],
            }
        parent_optimizer_restored = True
        phase_end(
            optimizer_phase,
            frozen_prior_tensors=165,
            trainable_code_tensors=278,
            trainable_fusion_tensors=54,
        )

        resume_rng = {
            'CPU': parent['torch_CPU_RNG'].clone(),
            'CUDA': [x.clone() for x in parent['CUDA_RNG']],
            'Python': parent['python_random_state'],
            'NumPy': parent['numpy_random_state'],
        }
        parent_metadata = {
            'terminal_sha256': PARENT_SHA,
            'protocol_sha256': PARENT_PROTOCOL_SHA,
            'logical_state_sha256': PARENT_LOGICAL,
            'counts': parent['counts'],
            'sampler_schedule': parent['sampler_schedule'],
            'completed_cycles': 32,
            'optimizer_admission': optimizer_admission,
        }
        del parent, original_state, state
        base_state = digest_state()
        progress.update(phase='R1_weights_Adam_restored_before_parity_RNG_pending')

        parity_phase = phase_start('fresh_R1_app_parity')
        wrapper = importlib.import_module('pretrained_completion').CodeFormerCompletion(net)
        parity_cases = p['parity_cases']
        need(len(parity_cases) == 4, 'Four fresh baseline parity cases required')
        parity_folder = out / 'fresh_app_parity'
        parity_folder.mkdir()
        with torch.no_grad():
            for case in parity_cases:
                rgb = image(case['input256'], 256)
                m = mask(case['mask'])
                x, _ = app_input(rgb, m)
                captures = []

                def capture_input(module, args):
                    value = args[0]
                    captures.append(
                        {
                            'input': value.detach().cpu().numpy().copy(),
                            'shape': list(value.shape),
                            'stride': list(value.stride()),
                            'dtype': str(value.dtype),
                            'device': str(value.device),
                            'NCHW_contiguous': value.is_contiguous(),
                            'channels_last_contiguous': value.is_contiguous(
                                memory_format=torch.channels_last
                            ),
                        }
                    )

                def capture_output(module, args, output):
                    captures[-1].update(
                        raw=output[0].detach().cpu().numpy().copy(),
                        codes=output[1].argmax(dim=2).detach().cpu().numpy().copy(),
                    )

                input_hook = net.register_forward_pre_hook(capture_input)
                output_hook = net.register_forward_hook(capture_output)
                try:
                    raw, _, _ = forward(
                        'parity', lambda: net(x, w=1, detach_16=True, adain=False)
                    )
                    own = delivered(rgb, m, raw)
                    reference = forward('parity', lambda: wrapper(rgb, m))
                finally:
                    input_hook.remove()
                    output_hook.remove()
                need(
                    len(captures) == 2 and all('raw' in row for row in captures),
                    'Capture both fresh parity calls',
                )
                own_capture, app_capture = captures
                error = float((own - reference).abs().max().item())
                receipt = {
                    'id': case['id'],
                    'max_raw256_error': error,
                    'max_model_input512_error': float(
                        np.max(np.abs(own_capture['input'] - app_capture['input']))
                    ),
                    'input512_exact': bool(
                        np.array_equal(own_capture['input'], app_capture['input'])
                    ),
                    'input_stride_exact': own_capture['stride'] == app_capture['stride'],
                    'model_raw512_max_error': float(
                        np.max(np.abs(own_capture['raw'] - app_capture['raw']))
                    ),
                    'selected_codes_exact': bool(
                        np.array_equal(own_capture['codes'], app_capture['codes'])
                    ),
                    'own_input_layout': {
                        k: v for k, v in own_capture.items() if k not in ('input', 'raw', 'codes')
                    },
                    'app_input_layout': {
                        k: v for k, v in app_capture.items() if k not in ('input', 'raw', 'codes')
                    },
                    'tolerance_raw256': 1e-6,
                    'counts': dict(counts),
                    'optimizer_steps_before_parity': (
                        counts['code_optimizer_steps'] + counts['fusion_optimizer_steps']
                    ),
                }
                save_npz(
                    parity_folder / (case['id'] + '.npz'),
                    input256=rgb.detach().cpu().numpy(),
                    mask256=m.detach().cpu().numpy(),
                    own_input512=own_capture['input'],
                    app_input512=app_capture['input'],
                    own_raw512=own_capture['raw'],
                    app_raw512=app_capture['raw'],
                    own_codes=own_capture['codes'],
                    app_codes=app_capture['codes'],
                    own_result256=own.detach().cpu().numpy(),
                    app_result256=reference.detach().cpu().numpy(),
                )
                write(parity_folder / (case['id'] + '.json'), receipt)
                parity.append(receipt)
        write(
            parity_folder / 'summary.json',
            {
                'complete': True,
                'cases': parity,
                'all_input_values_and_layout_exact': all(
                    row['input512_exact'] and row['input_stride_exact'] for row in parity
                ),
                'max_raw256_error': max(row['max_raw256_error'] for row in parity),
                'tolerance_raw256': 1e-6,
                'counts': dict(counts),
                'training_updates': 0,
            },
        )
        need(
            all(row['input512_exact'] and row['input_stride_exact'] for row in parity),
            'Fresh app input value/layout parity failed',
        )
        need(
            all(row['max_raw256_error'] <= 1e-6 for row in parity),
            'Fresh frozen app pipeline baseline parity failed',
        )
        del wrapper
        phase_end(parity_phase, pairs=len(parity), max_error=max(r['max_raw256_error'] for r in parity))
        need(digest_state() == base_state, 'Initial parity changed model state')

        write(
            out / 'initial_state.json',
            {
                'original_base_logical_state_sha256': original_teacher_state,
                'initial_R1_logical_state_sha256': base_state,
                'parent_training_state_sha256': PARENT_SHA,
                'parent_optimizer_admission': optimizer_admission,
                'frozen_generator_codebook_sha256': frozen_state,
                'checkpoint_sha256': ORIGINAL_SHA,
                'fresh_app_parity': parity,
                'teacher_neural_calls': 0,
                'hardware': hardware,
                'torch': torch.__version__,
                'counts': counts,
            },
        )
        write(out / 'state_before.json', state_receipt())

        baseline = evaluate(0, p['monitor_cases'], 'monitor')
        challenge_baseline = evaluate(0, p['challenge_cases'], 'challenge')
        evaluate(0, p['development_cases'], 'public_development')
        empty_controls(0)

        compatibility = phase_start('historical_R1_48case_compatibility')
        compatibility_rows = []
        for kind, summary in [('monitor', baseline), ('challenge', challenge_baseline)]:
            historical = {r['id']: r for r in p['historical_control']['initial_rows'][kind]}
            for record in summary['records']:
                row = historical[record['id']]
                with np.load(root / row['result256'], allow_pickle=False) as archive:
                    need(
                        set(archive.files) == {'result256'},
                        'Compact historical array schema differs',
                    )
                    prior = archive['result256']
                need(
                    prior.dtype == np.float32
                    and prior.shape == (1, 3, 256, 256)
                    and np.isfinite(prior).all(),
                    'Compact historical values invalid',
                )
                with np.load(out / Path(record['PNG']).with_suffix('.npz'), allow_pickle=False) as archive:
                    current = archive['result256']
                error = float(np.max(np.abs(current - prior)))
                png_exact = sha(out / record['PNG']) == sha(root / row['PNG'])
                compatibility_rows.append(
                    {
                        'kind': kind,
                        'id': record['id'],
                        'max_raw256_error': error,
                        'PNG_bytes_exact': png_exact,
                        'historical_result256_sha256': sha(root / row['result256']),
                        'historical_PNG_sha256': sha(root / row['PNG']),
                    }
                )
        write(
            out / 'historical_R1_compatibility.json',
            {
                'complete': True,
                'cases': compatibility_rows,
                'all48_before_update': len(compatibility_rows) == 48,
                'raw_tolerance': 1e-6,
                'updates_before': counts['code_optimizer_steps'] + counts['fusion_optimizer_steps'],
                'pass': len(compatibility_rows) == 48
                and all(
                    r['max_raw256_error'] <= 1e-6 and r['PNG_bytes_exact']
                    for r in compatibility_rows
                ),
            },
        )
        phase_end(compatibility, cases=len(compatibility_rows))
        need(
            len(compatibility_rows) == 48
            and all(
                r['max_raw256_error'] <= 1e-6 and r['PNG_bytes_exact']
                for r in compatibility_rows
            ),
            'Historical R1 baseline incompatible; retain stop before updates',
        )

        RNG_phase = phase_start('parent_RNG_optimization_boundary')
        torch.set_rng_state(resume_rng['CPU'])
        torch.cuda.set_rng_state_all(resume_rng['CUDA'])
        random.setstate(resume_rng['Python'])
        nr = resume_rng['NumPy']
        np.random.set_state(
            (
                nr['algorithm'],
                np.asarray(nr['keys'], dtype=np.uint32),
                nr['position'],
                nr['has_gauss'],
                nr['cached_gaussian'],
            )
        )
        restored_cuda = torch.cuda.get_rng_state_all()
        need(
            torch.equal(torch.get_rng_state(), resume_rng['CPU'])
            and len(restored_cuda) == len(resume_rng['CUDA']) == 1
            and all(torch.equal(a, b) for a, b in zip(restored_cuda, resume_rng['CUDA']))
            and random.getstate() == resume_rng['Python'],
            'Parent RNG restoration differs',
        )
        actual_nr = np.random.get_state()
        need(
            actual_nr[0] == nr['algorithm']
            and np.array_equal(actual_nr[1], np.asarray(nr['keys'], dtype=np.uint32))
            and actual_nr[2] == nr['position']
            and actual_nr[3] == nr['has_gauss']
            and actual_nr[4] == nr['cached_gaussian'],
            'Full NumPy RNG restoration differs',
        )

        def json_digest(value):
            return hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            ).hexdigest()

        def bytes_digest(value):
            return hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()

        expected_rng_hashes = {
            'CPU': bytes_digest(resume_rng['CPU']),
            'CUDA': [bytes_digest(value) for value in resume_rng['CUDA']],
            'Python': json_digest(resume_rng['Python']),
            'NumPy': json_digest(nr),
        }
        actual_nr_dict = {
            'algorithm': actual_nr[0],
            'keys': actual_nr[1].tolist(),
            'position': actual_nr[2],
            'has_gauss': actual_nr[3],
            'cached_gaussian': actual_nr[4],
        }
        restored_rng_hashes = {
            'CPU': bytes_digest(torch.get_rng_state()),
            'CUDA': [bytes_digest(value) for value in restored_cuda],
            'Python': json_digest(random.getstate()),
            'NumPy': json_digest(actual_nr_dict),
        }
        need(
            expected_rng_hashes == restored_rng_hashes,
            'Parent RNG boundary hashes differ',
        )
        parent_RNG_restored = True
        parent_metadata['RNG_boundary_hashes'] = {
            'expected': expected_rng_hashes,
            'restored': restored_rng_hashes,
        }
        write(
            out / 'RNG_transition.json',
            {
                'complete': True,
                'parent_terminal_sha256': PARENT_SHA,
                'parent_RNG_restored': True,
                'expected_hashes': expected_rng_hashes,
                'restored_hashes': restored_rng_hashes,
                'new_sampler_prematerialized': True,
                'newphase_cycles': 32,
                'global_rng_not_consumed_for_sampler': True,
            },
        )
        phase_end(RNG_phase, hashes=restored_rng_hashes)

        write(
            out / 'objective_contract.json',
            {
                'objective': OBJECTIVE_ID,
                'spatial_code_and_latent': '16x16 support-weighted CE + lq_feat MSE (hole=1.0, ring=0.5, visible=0.15)',
                'joint_coadaptation': '278 code tensors + 54 fusion tensors updated per cycle; 165 quantize/generator frozen',
                'dual_resolution_pixel_seam_grad': 'delivered256 hole/seam/gradient L1 + raw512 hole/seam L1',
                'interchannel_chroma_loss': 'pairwise |dRG|+|dGB|+|dBR| inside mask (0.35 RGB, 0.70 GRAY)',
                'trust_region_and_role_weight': 'anchor_weight=1.25, augmented_weight=0.75, L2 anchor to R1 coef=1e-4, cosine LR 8e-6->2e-6',
                'microbatch_divisor_once': 4,
            },
        )

        fit_logits_folder = out / 'FIT_logits'
        fit_logits_folder.mkdir()
        monitors = [baseline]
        challenges = [challenge_baseline]

        for cycle, case_ids in enumerate(schedule, 1):
            cycle_lr = lr_for_cycle(cycle)
            set_lr(code_optimizer, cycle_lr)
            set_lr(fusion_optimizer, cycle_lr)

            progress.update(
                cycle=cycle,
                route='code',
                microbatch=0,
                microbatches_backward_complete=0,
                phase='accumulating',
            )
            guard()
            active(code_groups)
            code_optimizer.zero_grad(set_to_none=True)
            code_losses = []
            for micro, ident in enumerate(case_ids, 1):
                progress['microbatch'] = micro
                case = fit_cases[ident]
                rgb = image(case['input256'], 256)
                m = mask(case['mask'])
                x, _ = app_input(rgb, m)
                logits, lq_feat = forward('fit_code', lambda: net(x, code_only=True))
                t_codes, t_latent = teacher_fit[case['teacher_key']]
                role_w = (
                    CONTRACT['clear_anchor_role_weight']
                    if case['role'] == 'fit_anchor'
                    else CONTRACT['augmented_role_weight']
                )
                code_obj, code_diag, w16_norm = spatial_code_latent_loss(
                    logits,
                    lq_feat,
                    t_codes,
                    t_latent,
                    m,
                    role_weight=role_w,
                    F=F,
                )
                trust_code = anchor_trust_penalty(
                    code_parameters,
                    r1_code_anchors,
                    coef=CONTRACT['trust_region_l2_coef'],
                )
                loss = (code_obj + trust_code) / 4.0
                need(torch.isfinite(loss).item(), 'Nonfinite code objective')
                logit_rel = 'FIT_logits/c%04d_m%d_%s.npz' % (cycle, micro, ident)
                save_npz(
                    out / logit_rel,
                    logits=logits.detach().cpu().numpy().astype(np.float32),
                    target_codes=t_codes.detach().cpu().numpy().astype(np.int64),
                    spatial_weights16=w16_norm.detach().cpu().numpy().astype(np.float32),
                )
                counts['backward_calls'] += 1
                progress['phase'] = 'backward_started'
                loss.backward()
                progress.update(phase='accumulating', microbatches_backward_complete=micro)
                code_losses.append(
                    {
                        'id': ident,
                        'role': case['role'],
                        'style': case['style'],
                        'mask_kind': case['mask_kind'],
                        'teacher_key': case['teacher_key'],
                        'saved_logits': logit_rel,
                        'saved_logits_sha256': sha(out / logit_rel),
                        'trust_L2': float(trust_code.detach().item()),
                        'backward_loss': float(loss.detach().item()),
                        **code_diag,
                    }
                )
                del logits, lq_feat, code_obj, trust_code, loss
                guard()

            code_gradient = gradient_receipt(code_groups)
            code_clip = float(
                torch.nn.utils.clip_grad_norm_(
                    code_parameters, 1.0, error_if_nonfinite=True
                ).item()
            )
            progress['phase'] = 'optimizer_step_started'
            code_optimizer.step()
            counts['code_optimizer_steps'] += 1
            code_optimizer.zero_grad(set_to_none=True)
            need(
                all(torch.isfinite(parameter).all().item() for parameter in code_parameters),
                'Nonfinite updated code parameters',
            )
            verify_active_adam(
                code_optimizer,
                code_parameters,
                optimizer_parameter_names['code'],
                counts['code_optimizer_steps'],
            )

            progress.update(
                route='fusion',
                microbatch=0,
                microbatches_backward_complete=0,
                phase='accumulating',
            )
            active(fusion_groups)
            fusion_optimizer.zero_grad(set_to_none=True)
            fusion_losses = []
            for micro, ident in enumerate(case_ids, 1):
                progress['microbatch'] = micro
                case = fit_cases[ident]
                rgb = image(case['input256'], 256)
                target256, target512 = clean[case['teacher_key']]
                m = mask(case['mask'])
                x, m512 = app_input(rgb, m)
                raw, _, _ = forward(
                    'fit_RGB', lambda: net(x, w=1, detach_16=True, adain=False)
                )
                role_w = (
                    CONTRACT['clear_anchor_role_weight']
                    if case['role'] == 'fit_anchor'
                    else CONTRACT['augmented_role_weight']
                )
                fusion_obj, _, fusion_diag = unified_fusion_loss(
                    raw,
                    rgb,
                    m,
                    m512,
                    target256,
                    target512,
                    role_weight=role_w,
                    style=case['style'],
                    F=F,
                )
                trust_fusion = anchor_trust_penalty(
                    fusion_parameters,
                    r1_fusion_anchors,
                    coef=CONTRACT['trust_region_l2_coef'],
                )
                loss = (fusion_obj + trust_fusion) / 4.0
                need(torch.isfinite(loss).item(), 'Nonfinite fusion objective')
                counts['backward_calls'] += 1
                progress['phase'] = 'backward_started'
                loss.backward()
                progress.update(phase='accumulating', microbatches_backward_complete=micro)
                fusion_losses.append(
                    {
                        'id': ident,
                        'role': case['role'],
                        'style': case['style'],
                        'mask_kind': case['mask_kind'],
                        'teacher_key': case['teacher_key'],
                        'trust_L2': float(trust_fusion.detach().item()),
                        'backward_loss': float(loss.detach().item()),
                        **fusion_diag,
                    }
                )
                del raw, fusion_obj, trust_fusion, loss
                guard()

            fusion_gradient = gradient_receipt(fusion_groups)
            fusion_clip = float(
                torch.nn.utils.clip_grad_norm_(
                    fusion_parameters, 1.0, error_if_nonfinite=True
                ).item()
            )
            progress['phase'] = 'optimizer_step_started'
            fusion_optimizer.step()
            counts['fusion_optimizer_steps'] += 1
            fusion_optimizer.zero_grad(set_to_none=True)
            need(
                all(torch.isfinite(parameter).all().item() for parameter in fusion_parameters),
                'Nonfinite updated fusion parameters',
            )
            verify_active_adam(
                fusion_optimizer,
                fusion_parameters,
                optimizer_parameter_names['fusion'],
                counts['fusion_optimizer_steps'],
            )
            net.requires_grad_(False)
            terminal_cycle = cycle
            progress.update(phase='paired_cycle_complete', microbatch=4)

            row = {
                'cycle': cycle,
                'cumulative_cycle': 32 + cycle,
                'lr': cycle_lr,
                'microbatch_ids': case_ids,
                'code_losses': code_losses,
                'fusion_losses': fusion_losses,
                'code_gradient_groups': code_gradient,
                'fusion_gradient_groups': fusion_gradient,
                'code_preclip_L2': code_clip,
                'fusion_preclip_L2': fusion_clip,
                'counts': dict(counts),
                'seconds': time.monotonic() - started,
            }
            fit_log.append(row)
            write(out / ('training_cycle_%04d.json' % cycle), row)

            if cycle in (8, 16, 32):
                need(
                    digest_state(('quantize.', 'generator.')) == frozen_state,
                    'Frozen 165 prior changed during cycle ' + str(cycle),
                )
                snapshots.append(
                    {
                        'cycle': cycle,
                        'cumulative_cycle': 32 + cycle,
                        'logical_state_sha256': digest_state(),
                        'state_saved': False,
                        'recoverable_checkpoint': False,
                        'scope': 'Readout metrics/images only; terminal state only is saved.',
                    }
                )
                mon_summary = evaluate(cycle, p['monitor_cases'], 'monitor')
                chal_summary = evaluate(cycle, p['challenge_cases'], 'challenge')
                monitors.append(mon_summary)
                challenges.append(chal_summary)
                clear_screen = compare(mon_summary, baseline, conditions=False)
                chal_screen = compare(chal_summary, challenge_baseline, conditions=True)
                gray_screen = compare_gray(chal_summary, challenge_baseline)
                combined_screen = {
                    'cycle': cycle,
                    'cumulative_cycle': 32 + cycle,
                    'clear': clear_screen,
                    'challenge': chal_screen,
                    'GRAY_style_gate': gray_screen,
                    'catastrophic_regression_stop': bool(
                        clear_screen['catastrophic_regression_stop']
                        or chal_screen['catastrophic_regression_stop']
                    ),
                    'prospective_numeric_screen_pass': bool(
                        cycle == 32
                        and clear_screen['prospective_numeric_screen_pass']
                        and chal_screen['prospective_numeric_screen_pass']
                        and gray_screen['all_GRAY_macro_geometry_chroma_nonworse']
                    ),
                    'model_qualified': False,
                }
                write(out / ('screen_u%04d.json' % cycle), combined_screen)
                empty_controls(cycle)
                if cycle in (8, 16) and combined_screen['catastrophic_regression_stop']:
                    stop_reason = (
                        'Catastrophic >10% subgroup regression at snapshot cycle ' + str(cycle)
                    )
                    break
            print(
                json.dumps(
                    {
                        'cycle': cycle,
                        'maximum': 32,
                        'lr': cycle_lr,
                        'counts': counts,
                        'seconds': time.monotonic() - started,
                    }
                ),
                flush=True,
            )

        evaluate(terminal_cycle, p['development_cases'], 'public_development')
        write(out / 'state_after.json', state_receipt())
        final_clear = compare(monitors[-1], baseline, conditions=False)
        final_challenge = compare(challenges[-1], challenge_baseline, conditions=True)
        final_gray = compare_gray(challenges[-1], challenge_baseline)
        final_screen = {
            'clear': final_clear,
            'challenge': final_challenge,
            'GRAY_style_gate': final_gray,
            'prospective_numeric_screen_pass': bool(
                terminal_cycle == 32
                and not stop_reason
                and final_clear['prospective_numeric_screen_pass']
                and final_challenge['prospective_numeric_screen_pass']
                and final_gray['all_GRAY_macro_geometry_chroma_nonworse']
            ),
            'visual_review_pending': True,
            'model_qualified': False,
        }
        need(
            digest_state(('quantize.', 'generator.')) == frozen_state
            and sha(checkpoint) == ORIGINAL_SHA
            and sha(parent_path) == PARENT_SHA,
            'Frozen prior or source/parent checkpoint changed',
        )
        terminal_state = save_state(
            terminal_cycle,
            'completed_finite_pilot' if not stop_reason else 'quality_stopped_terminal',
        )
        final = {
            'format': 'completion-codeformer-unified-r2-v1-results-v1',
            'complete': True,
            'protocol_sha256': expected,
            'cycles_completed': terminal_cycle,
            'cumulative_cycles_completed': 32 + terminal_cycle,
            'stop_reason': stop_reason,
            'numeric_screen': final_screen,
            'metric_snapshots': snapshots,
            'terminal_training_state': terminal_state,
            'earlier_snapshots_recoverable': False,
            'initial_state_sha256': base_state,
            'final_logical_state_sha256': digest_state(),
            'frozen_state_unchanged': True,
            'original_checkpoint_unchanged': True,
            'parent_checkpoint_unchanged': True,
            'counts': counts,
            'typed_forward_slots': typed,
            'actual_whole_model_forward_calls': model_forwards,
            'seconds': time.monotonic() - started,
            'visual_review_pending': True,
            'user_acceptance_pending': True,
            'model_qualified': False,
            'automatic_app_promotion': False,
        }
        write(out / 'results.json', final)
        budget.guard(terminal=True)
        return (
            0
            if final_screen['prospective_numeric_screen_pass']
            and terminal_cycle == 32
            and not stop_reason
            else 2
        )
    except BaseException as exc:
        if not terminal_attempted and not (out / 'terminal_training_state.pth').exists():
            try:
                snapshots.append(
                    save_state(terminal_cycle, 'failure_retained_current_partial_state')
                )
            except BaseException as save_error:
                if not (out / 'terminal_save_failure.json').exists():
                    write(
                        out / 'terminal_save_failure.json',
                        {
                            'error': repr(save_error),
                            'progress': dict(progress),
                            'counts': dict(counts),
                            'note': 'State may not be recoverable; never auto-resume',
                        },
                        emergency=True,
                    )
        if not (out / 'failure.json').exists():
            write(
                out / 'failure.json',
                {
                    'complete': False,
                    'exception': repr(exc),
                    'traceback': traceback.format_exc(),
                    'last_full_cycle': terminal_cycle,
                    'partial_progress': dict(progress),
                    'counts': counts,
                    'typed_forward_slots': typed,
                    'snapshots': snapshots,
                    'seconds': time.monotonic() - started,
                    'model_qualified': False,
                },
                emergency=True,
            )
        raise
    finally:
        forward_hook.remove()
        signal.signal(signal.SIGTERM, old_sigterm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-transfer', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    p, checkpoint, parent = verify(root, args.protocol_sha)
    if args.verify_transfer:
        print(
            json.dumps(
                {
                    'complete': True,
                    'files': len(p['files']),
                    'selected_fit_cases': 128,
                    'cached_teacher_sets': 80,
                    'monitor_cases': 32,
                    'challenge_cases': 16,
                    'public_cases': 32,
                    'optimizer_updates': 0,
                    'neural_calls': 0,
                }
            )
        )
        return 0
    if args.worker:
        return worker(root, p, checkpoint, parent, args.protocol_sha)

    hardware = require_vm(root)
    need(
        checkpoint is not None and parent is not None,
        'Both original inpainting checkpoint and R1 parent state must exist on VM',
    )
    need(
        not (root / 'outputs').exists()
        and not (root / 'supervisor.json').exists()
        and not (root / 'trainer.log').exists(),
        'Pilot already attempted; preserve existing evidence',
    )
    command = [
        sys.executable,
        '-B',
        '-u',
        str(root / 'scripts/cctv_completion_codeformer_unified_r2_v1_vm.py'),
        '--root',
        str(root),
        '--protocol-sha',
        args.protocol_sha,
        '--worker',
    ]
    started = time.monotonic()
    timeout = False
    interruption = None
    code = 1
    child = None
    shutdown_mode = 'normal_worker_exit'
    captured = 0
    log_cap_hit = False

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor signal ' + str(signum))

    previous = signal.signal(signal.SIGTERM, interrupted)

    def kill_owned(sig):
        if child is None or child.poll() is not None:
            return
        try:
            need(
                os.getpgid(child.pid) == child.pid,
                'Refusing to signal non-owned process group',
            )
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass

    def stop_owned(grace):
        if child is not None and child.poll() is None:
            kill_owned(signal.SIGTERM)
            try:
                child.wait(timeout=max(0, min(grace, 1800 - (time.monotonic() - started))))
            except subprocess.TimeoutExpired:
                kill_owned(signal.SIGKILL)
                child.wait()

    try:
        with (root / 'trainer.log').open('xb') as log:
            child = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                bufsize=0,
            )
            print(
                json.dumps(
                    {
                        'child_pid': child.pid,
                        'log': 'trainer.log',
                        'external_cap_seconds': 1800,
                    }
                ),
                flush=True,
            )
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            try:
                while selector.get_map():
                    if time.monotonic() - started >= 1800:
                        timeout = True
                        shutdown_mode = 'hard_cap_SIGKILL'
                        kill_owned(signal.SIGKILL)
                        child.wait()
                        code = 124
                        break
                    for key, _ in selector.select(timeout=0.1):
                        chunk = os.read(key.fileobj.fileno(), 16384)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        remaining = CONTRACT['trainer_log_bytes'] - captured
                        if len(chunk) > remaining:
                            if remaining:
                                log.write(chunk[:remaining])
                                captured += remaining
                            log_cap_hit = True
                            shutdown_mode = 'stdout_bytecap_SIGTERM'
                            stop_owned(15)
                            code = 125
                            break
                        log.write(chunk)
                        log.flush()
                        captured += len(chunk)
                    if log_cap_hit:
                        break
                if not timeout and not log_cap_hit:
                    code = child.wait(timeout=max(0, 1800 - (time.monotonic() - started)))
            finally:
                selector.close()
                child.stdout.close()
    except subprocess.TimeoutExpired:
        timeout = True
        shutdown_mode = 'hard_cap_SIGKILL'
        kill_owned(signal.SIGKILL)
        if child is not None:
            child.wait()
        code = 124
    except BaseException as exc:
        interruption = repr(exc)
        shutdown_mode = 'interruption_SIGTERM_then_bounded_wait'
        stop_owned(15)
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        if child is not None and child.poll() is None:
            kill_owned(signal.SIGKILL)
            child.wait()
        signal.signal(signal.SIGTERM, previous)
        receipt = {
            'complete': True,
            'worker_exit': code,
            'actual_child_exit': None if child is None else child.returncode,
            'child_pid': None if child is None else child.pid,
            'timeout': timeout,
            'interruption': interruption,
            'shutdown_mode': shutdown_mode,
            'external_cap_seconds': 1800,
            'seconds': time.monotonic() - started,
            'hardware': hardware,
            'terminal_state_present': (root / 'outputs/terminal_training_state.pth').is_file(),
            'terminal_partial_present': (
                root / 'outputs/terminal_training_state.pth.partial'
            ).is_file(),
            'stdout_capture_bytes': captured,
            'stdout_cap_bytes': CONTRACT['trainer_log_bytes'],
            'stdout_cap_reached': log_cap_hit,
            'stdout_prefix_retained_if_capped': True,
            'state_recoverability': (
                'Pending independent return audit; hard kill may prevent save; never automatic resume'
            ),
            'training_success_not_implied': True,
        }
        body = (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode()
        need(len(body) <= 1024 * 1024, 'Supervisor receipt bytecap')
        with (root / 'supervisor.json').open('xb') as stream:
            stream.write(body)
    print(
        json.dumps(
            {
                'worker_exit': code,
                'timeout': timeout,
                'stdout_cap_reached': log_cap_hit,
                'log': 'trainer.log',
                'model_qualified': False,
            }
        )
    )
    return code


if __name__ == '__main__':
    raise SystemExit(main())
