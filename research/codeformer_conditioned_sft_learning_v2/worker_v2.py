"""Manual NVIDIA L4 Conditioned SFT V2 worker; inert on import.

Executes Two-Phase training strictly inside ~/forensic-dgp/cctv_codeformer_conditioned_sft_learning_v2_vm:
- Frozen 575-tensor trained-u64 backbone (515 R2 + 24 U96 + 36 u64 spatial heads)
- Step-0 exact bitwise u64 parity verification
- Phase 1: 40-epoch (1,680-step) Exposure-Matched Clean-SFT Pyramid distillation
- Phase 2: 224-update End-to-End Perceptual + SSIM7 + Multi-Scale Laplacian + Exposure-Split Gradient training
- Checkpoints saved at u0000_init, p1_e040, p2_u0056, p2_u0112, p2_u0168, p2_u0224, and best_degraded_ssim
"""
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

PACKAGE = 'research/codeformer_conditioned_sft_learning_v2'
SEED = 20261026
ENV_SUPERVISED = 'FORENSIC_CONDITIONED_SFT_V2_SUPERVISED'
ENV_PROTOCOL = 'FORENSIC_CONDITIONED_SFT_V2_PROTOCOL'
ENV_PARENT = 'FORENSIC_CONDITIONED_SFT_V2_SUPERVISOR_PID'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def supervisor_context(sha, *, environ, parent_pid):
    need(type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha) is not None
         and environ.get(ENV_SUPERVISED) == '1'
         and environ.get(ENV_PROTOCOL) == sha
         and environ.get(ENV_PARENT) == str(parent_pid),
         'Exact seal and active Conditioned SFT V2 supervisor parent required')


def _cosine_lr(step_idx: int, total_steps: int, warmup_steps: int, lr_max: float, lr_min: float) -> float:
    if step_idx < warmup_steps:
        return lr_max * float(step_idx + 1) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return lr_max
    progress = float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def run(root, protocol, expected_sha):
    from . import (
        architecture_v2 as arch_v2,
        conditioning_v2 as cond_v2,
        contract_v2 as contract,
        objective_v2 as obj_v2,
        packet_v2 as packet,
    )
    root = Path(root).absolute()
    began = time.monotonic()
    out_dir = root / contract.OUTPUT_RELATIVE
    phase = 'static_guards'

    def write_json(rel_path: str, payload):
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + '.pending')
        raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
        with tmp.open('wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
        return packet.pin(root, target)

    def append_jsonl(rel_path: str, row):
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        with target.open('ab') as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def log_journal(kind: str, **fields):
        row = {
            'utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_s': round(time.monotonic() - began, 3),
            'phase': phase,
            'kind': kind,
            **fields,
        }
        append_jsonl('journal.jsonl', row)
        print(json.dumps(row, sort_keys=True), flush=True)

    try:
        p = packet.verify_static(root, expected_sha)
        need(p == protocol, 'Verified protocol differs from supplied protocol object')
        need(Path(__file__).resolve() == root / PACKAGE / 'worker_v2.py', 'Use staged worker_v2.py')
        host_facts = packet.require_vm(root)
        supervisor_context(expected_sha, environ=os.environ, parent_pid=os.getppid())

        # Pre-framework single idle NVIDIA L4 guard
        phase = 'pre_framework_l4_guard'
        names = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=8, check=True,
        )
        need(not names.stderr.strip()
             and tuple(s.strip() for s in names.stdout.splitlines() if s.strip()) == ('NVIDIA L4',),
             'Exactly one pre-framework NVIDIA L4 GPU required')
        procs = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=8, check=True,
        )
        need(not procs.stdout.strip() and not procs.stderr.strip(),
             'Competing GPU workload detected; aborting before framework load')

        out_dir.mkdir(parents=True, exist_ok=False)
        write_json('started.json', {
            'complete': False,
            'protocol_sha256': expected_sha,
            'host': host_facts,
            'model_qualified': False,
        })
        log_journal('worker_started', host=host_facts)

        def _on_sigterm(signum, frame):
            raise TimeoutError('Supervisor SIGTERM received')

        signal.signal(signal.SIGTERM, _on_sigterm)

        # Deterministic CUDA & Torch initialization
        phase = 'init_torch_and_frozen_u64_backbone'
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import numpy as np
        from PIL import Image
        import torch
        from torch.nn import functional as F
        from third_party.codeformer.codeformer_arch import CodeFormer
        from research import codeformer_conditioned_sft_inference_v1 as inf_v1
        from research import codeformer_conditioned_sft_inference_v2 as inf_v2
        from research.codeformer_u96_spatial_feasibility_v1 import metrics as u96_metrics
        from research.codeformer_u64_cleananchors_continuation_v1 import metrics as u64_cont_metrics

        need(torch.cuda.is_available() and torch.cuda.device_count() == 1
             and torch.cuda.get_device_name(0) == 'NVIDIA L4',
             'Exactly one initialized NVIDIA L4 required')
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.cuda.reset_peak_memory_stats(0)

        def check_resources():
            elapsed = time.monotonic() - began
            need(elapsed < contract.LIMITS['worker_seconds'], 'Finite 7200s worker scientific deadline exceeded')
            free_b = shutil.disk_usage(root).free
            need(free_b >= contract.LIMITS['continuing_free_disk_bytes'], 'Disk reserve fell below 4 GiB')
            cuda_res = int(torch.cuda.memory_reserved(0))
            need(cuda_res <= contract.LIMITS['maximum_CUDA_reserved_bytes'], 'CUDA reserved exceeded 20 GiB')

        # Load frozen 575-tensor backbone (515 R2 + 24 U96 + 36 u64 heads)
        base = CodeFormer(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=['32', '64', '128', '256'],
        )
        r2_ckpt = torch.load(str(root / p['model']['r2']['path']), map_location='cpu', weights_only=True)
        r2_sd = r2_ckpt['params_ema'] if 'params_ema' in r2_ckpt else r2_ckpt
        base.load_state_dict(r2_sd, strict=True)
        base.eval().to(device)
        for param in base.parameters():
            param.requires_grad_(False)

        u96_adapters, u64_heads = arch_v2.build_frozen_u96_and_u64_modules(torch=torch)
        u96_ckpt = torch.load(str(root / p['model']['u96']['path']), map_location='cpu', weights_only=True)
        u96_sd = u96_ckpt.get('adapter_state', u96_ckpt.get('adapter_state_dict', u96_ckpt))
        u96_adapters.load_state_dict(
            {k[len('adapters.'):] if k.startswith('adapters.') else k: v for k, v in u96_sd.items()},
            strict=True,
        )
        u64_np = inf_v1.inspect_trainable_npz((root / p['u64_head']['path']).read_bytes())
        u64_heads.load_state_dict(
            {k[len('heads.'):]: torch.from_numpy(v) for k, v in u64_np.items()},
            strict=True,
        )
        u96_adapters.eval().to(device)
        u64_heads.eval().to(device)
        for mod in (u96_adapters, u64_heads):
            for param in mod.parameters():
                param.requires_grad_(False)

        def verify_frozen_backbone_digests():
            r2_sha = arch_v2.tensor_state_sha256(dict(base.named_parameters()), torch=torch)
            u96_sha = arch_v2.tensor_state_sha256(
                {'adapters.' + k: v for k, v in u96_adapters.named_parameters()}, torch=torch
            )
            head_sha = arch_v2.tensor_state_sha256(
                {'heads.' + k: v for k, v in u64_heads.named_parameters()}, torch=torch
            )
            need(r2_sha == arch_v2.R2_CANONICAL_SHA256, 'Frozen 515 R2 digest mutated')
            need(u96_sha == arch_v2.U96_CANONICAL_SHA256, 'Frozen 24 U96 digest mutated')
            need(head_sha == arch_v2.TERMINAL_HEADS_CANONICAL_SHA256, 'Frozen 36 u64 head digest mutated')
            return {'R2_515': r2_sha, 'U96_24': u96_sha, 'u64_heads_36': head_sha}

        frozen_digests = verify_frozen_backbone_digests()

        # Construct V2 trainable bundle
        v2_bundle = arch_v2.build_v2_trainable_modules(torch=torch, seed=arch_v2.INIT_SEED_V2).to(device)
        v2_init_dict = arch_v2.export_v2_trainable_npz_dict(v2_bundle, np=np)
        v2_param_count = sum(param.numel() for param in v2_bundle.parameters())
        v2_tensor_count = sum(1 for _ in v2_bundle.parameters())
        log_journal(
            'backbone_and_v2_initialized',
            frozen_digests=frozen_digests,
            v2_trainable_tensors=v2_tensor_count,
            v2_trainable_parameters=v2_param_count,
        )

        # Helper image & feature routines
        def load_rgb_u8(binding):
            path = packet.regular(root / binding['path'])
            need(packet.pin(root, path) == binding, 'Asset pin mismatch: ' + binding['path'])
            with Image.open(path) as im:
                need(im.mode == 'RGB' and im.size == (256, 256), 'Exact 256x256 RGB PNG required')
                arr = np.asarray(im, dtype=np.uint8).copy()
            return arr

        out_block_indices = set(base.fuse_encoder_block[s] for s in base.connect_list)

        def extract_r2_enc_connected(rgb_nchw_256):
            """Extract connected multi-scale R2 encoder features (for perceptual loss)."""
            x512 = F.interpolate(rgb_nchw_256, (512, 512), mode='bilinear', align_corners=False) * 2.0 - 1.0
            feats = {}
            h = x512
            for idx, block in enumerate(base.encoder.blocks):
                h = block(h)
                if idx in out_block_indices:
                    feats[str(h.shape[-1])] = h
            return feats

        @torch.no_grad()
        def extract_u96_sft_detached_cpu(rgb_u8):
            inp_f32 = rgb_u8.astype(np.float32) / np.float32(255.0)
            t = torch.from_numpy(inp_f32.transpose(2, 0, 1).copy())[None].to(device)
            enc = extract_r2_enc_connected(t)
            sft_cpu = {s: u96_adapters[s](enc[s]).detach().cpu() for s in arch_v2.SITES}
            enc_cpu = {s: enc[s].detach().cpu() for s in arch_v2.SITES}
            cond_t = cond_v2.compute_v2_conditioning_torch(t, torch=torch)
            cond_cpu = {
                'g_dark': cond_t['g_dark'],
                'g_blur': cond_t['g_blur'],
                'g_sensor_noise': cond_t['g_sensor_noise'],
                'site_maps': {s: cond_t['site_maps'][s].detach().cpu() for s in arch_v2.SITES},
            }
            return sft_cpu, enc_cpu, cond_cpu

        @torch.no_grad()
        def forward_u64_raw_f32(rgb_u8):
            inp_f32 = rgb_u8.astype(np.float32) / np.float32(255.0)
            t = torch.from_numpy(inp_f32.transpose(2, 0, 1).copy())[None].to(device)
            s512 = F.interpolate(t, (512, 512), mode='bilinear', align_corners=False) * 2.0 - 1.0
            handles = []
            pending = {}
            coarse = None

            def pre_h(site):
                def _pre(mod, args):
                    nonlocal coarse
                    enc, dec, w_val = args
                    used = u96_adapters[site](enc.detach())
                    if site == '32':
                        coarse = used.detach()
                    pending[site] = (used, dec)
                    return used, dec, w_val
                return _pre

            def post_h(site):
                def _post(mod, args, fused):
                    used, dec = pending.pop(site)
                    return u64_heads[site](used, dec, fused, coarse)
                return _post

            try:
                for s in arch_v2.SITES:
                    m = base.fuse_convs_dict[s]
                    handles.append(m.register_forward_pre_hook(pre_h(s)))
                    handles.append(m.register_forward_hook(post_h(s)))
                triple = base(s512, w=1.0, detach_16=True, adain=True)
            finally:
                for h in reversed(handles):
                    h.remove()
            norm = (triple[0] + 1.0) / 2.0
            raw256 = F.interpolate(norm.clamp(0.0, 1.0), (256, 256), mode='bilinear', align_corners=False)
            return raw256[0].detach().cpu().numpy().transpose(1, 2, 0).copy()

        def forward_v2_connected(inp_nchw_256, base_u64_nchw_256, gamma_val: float):
            """Connected FP32 forward through frozen CodeFormer + trainable v2_bundle."""
            cond_t = cond_v2.compute_v2_conditioning_torch(inp_nchw_256, torch=torch)
            s512 = F.interpolate(inp_nchw_256, (512, 512), mode='bilinear', align_corners=False) * 2.0 - 1.0
            handles = []
            pending = {}
            used_features = {}
            deg_u96_features = {}
            coarse = None
            prev_h = None

            def pre_h(site):
                def _pre(mod, args):
                    nonlocal coarse, prev_h
                    enc, dec, w_val = args
                    e_u96 = u96_adapters[site](enc.detach())
                    deg_u96_features[site] = e_u96
                    used, prev_h = v2_bundle.refine_site_sequential(site, e_u96, cond_t, prev_h)
                    used_features[site] = used
                    if site == '32':
                        coarse = used.detach()
                    pending[site] = (used, dec)
                    return used, dec, w_val
                return _pre

            def post_h(site):
                def _post(mod, args, fused):
                    used, dec = pending.pop(site)
                    base_post = u64_heads[site](used, dec, fused, coarse)
                    return v2_bundle.refine_post_sft(site, base_post, used, dec, fused, coarse, cond_t)
                return _post

            try:
                for s in arch_v2.SITES:
                    m = base.fuse_convs_dict[s]
                    handles.append(m.register_forward_pre_hook(pre_h(s)))
                    handles.append(m.register_forward_hook(post_h(s)))
                triple = base(s512, w=1.0, detach_16=True, adain=True)
            finally:
                for h in reversed(handles):
                    h.remove()
            norm = (triple[0] + 1.0) / 2.0
            raw256 = F.interpolate(norm.clamp(0.0, 1.0), (256, 256), mode='bilinear', align_corners=False)
            cond_out, _ = cond_v2.condition_output_v2_torch(
                raw256, inp_nchw_256, torch=torch, baseline_nchw=base_u64_nchw_256,
                beta=cond_v2.DEFAULT_ANCHOR_BETA, gamma=gamma_val,
            )
            return raw256, cond_out, used_features, deg_u96_features, cond_t

        @torch.no_grad()
        def infer_v2_eval(rgb_u8, u64_raw_cached, gamma_val: float):
            inp_f32 = rgb_u8.astype(np.float32) / np.float32(255.0)
            if gamma_val == 0.0:
                cond_np = cond_v2.compute_v2_conditioning_numpy(inp_f32, np=np)
                u8 = (np.clip(u64_raw_cached, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
                return u64_raw_cached.copy(), u8, cond_np
            t = torch.from_numpy(inp_f32.transpose(2, 0, 1).copy())[None].to(device)
            b_t = torch.from_numpy(u64_raw_cached.transpose(2, 0, 1).copy())[None].to(device)
            raw_neural_t, _, _, _, _ = forward_v2_connected(t, b_t, gamma_val=0.0)
            raw_neural_np = raw_neural_t[0].detach().cpu().numpy().transpose(1, 2, 0).copy()
            cond_raw, cond_np = cond_v2.condition_output_v2_numpy(
                raw_neural_np, inp_f32, np=np, baseline_rgb=u64_raw_cached,
                beta=cond_v2.DEFAULT_ANCHOR_BETA, gamma=gamma_val, dark_drift_align=True,
            )
            u8 = (np.clip(cond_raw, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
            return cond_raw, u8, cond_np

        def ssim7_numpy(pred_f32, targ_f32) -> float:
            pt = torch.from_numpy(pred_f32.transpose(2, 0, 1).copy())[None]
            tt = torch.from_numpy(targ_f32.transpose(2, 0, 1).copy())[None]
            return float(1.0 - obj_v2.ssim7_loss_torch(pt, tt, torch=torch).item())

        def paired_case_metrics(raw_f32, rgb_u8, target_u8):
            targ_f32 = target_u8.astype(np.float32) / np.float32(255.0)
            png_f32 = rgb_u8.astype(np.float32) / np.float32(255.0)
            return {
                'raw': {
                    'mae': float(np.mean(np.abs(raw_f32 - targ_f32))),
                    'ssim7': ssim7_numpy(raw_f32, targ_f32),
                    'central_lap': inf_v2.central_laplacian_sharpness_numpy(raw_f32, np=np),
                },
                'native_png': {
                    'mae': float(np.mean(np.abs(png_f32 - targ_f32))),
                    'ssim7': ssim7_numpy(png_f32, targ_f32),
                    'central_lap': inf_v2.central_laplacian_sharpness_numpy(png_f32, np=np),
                },
            }

        # Load validation images & cache u64 baseline predictions
        phase = 'step0_u64_baseline_and_parity'
        cohorts = p['selection']
        parent_raw = (root / p['evidence']['u64_parent_protocol']['path']).read_bytes()
        audit_raw = (root / p['evidence']['u64_historical_pixel_audit']['path']).read_bytes()
        frozen_U96_records = [
            packet.strict_json((root / c['record']['path']).read_bytes())
            for c in p['controls']
        ]
        parity_witnesses = {
            w['id']: np.load(root / w['raw']['path'], allow_pickle=False)
            for w in p['u64_parity']
        }

        val_inputs = {}
        val_targets = {}
        u64_val_raw = {}
        initial_old_quality_records = []

        for bank_name in ('old_quality', 'v1_monitors', 'v2_monitors', 'native_cctv_dev'):
            for row in cohorts[bank_name]:
                cid = row['id']
                inp_u8 = load_rgb_u8(row['input'])
                val_inputs[cid] = inp_u8
                if row.get('target') is not None:
                    val_targets[cid] = load_rgb_u8(row['target'])
                u64_r = forward_u64_raw_f32(inp_u8)
                u64_val_raw[cid] = u64_r
                if cid in parity_witnesses:
                    max_w_diff = float(np.max(np.abs(u64_r - parity_witnesses[cid])))
                    need(max_w_diff <= 2e-5, f'Historical u64 parity witness exceeded tolerance: {cid} ({max_w_diff})')

        # Build initial u64 old_quality records for exact inherited 56-case evaluation
        for row in cohorts['old_quality']:
            cid = row['id']
            u64_r = u64_val_raw[cid]
            u64_u8 = (np.clip(u64_r, 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)
            inp_f32 = val_inputs[cid].astype(np.float32) / np.float32(255.0)
            ref_f32 = (val_targets[cid].astype(np.float32) / np.float32(255.0)) if cid in val_targets else inp_f32
            rec = {
                'id': cid,
                'reference_id': row['reference_id'],
                'role': row['role'],
                'split': row['split'],
                'metrics': {
                    'raw': u96_metrics.case_view_metrics(u64_r, ref_f32, row['role'], np=np),
                    'native_png': u96_metrics.case_view_metrics(
                        u64_u8.astype(np.float32) / np.float32(255.0), ref_f32, row['role'], np=np
                    ),
                },
            }
            initial_old_quality_records.append(rec)

        # Verify step-0 bitwise parity of v2_bundle neural forward against u64
        v2_bundle.eval()
        max_step0_neural_diff = 0.0
        for row in cohorts['old_quality'][:8]:
            cid = row['id']
            inp_f32 = val_inputs[cid].astype(np.float32) / np.float32(255.0)
            t = torch.from_numpy(inp_f32.transpose(2, 0, 1).copy())[None].to(device)
            b_t = torch.from_numpy(u64_val_raw[cid].transpose(2, 0, 1).copy())[None].to(device)
            with torch.no_grad():
                raw_t, _, _, _, _ = forward_v2_connected(t, b_t, gamma_val=0.0)
            raw_np = raw_t[0].detach().cpu().numpy().transpose(1, 2, 0)
            max_step0_neural_diff = max(max_step0_neural_diff, float(np.max(np.abs(raw_np - u64_val_raw[cid]))))
        need(max_step0_neural_diff == 0.0, f'Exact step-0 bitwise u64 parity required, got {max_step0_neural_diff}')

        def evaluate_and_save_checkpoint(tag: str, inherited_phase: str):
            check_resources()
            verify_frozen_backbone_digests()
            v2_bundle.eval()
            disp = arch_v2.max_linf_displacement_from_init(v2_bundle, v2_init_dict, np=np)
            gamma_now = 0.0 if tag == 'u0000_init' else cond_v2.displacement_gamma(disp)

            npz_bytes = inf_v2.save_v2_trainable_npz_bytes(v2_bundle, np=np)
            ckpt_rel = f'checkpoints/{tag}/trainable_v2.npz'
            ckpt_path = out_dir / ckpt_rel
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            with ckpt_path.open('wb') as f:
                f.write(npz_bytes)
                f.flush()
                os.fsync(f.fileno())
            ckpt_pin = packet.pin(root, ckpt_path)

            # 1) Inherited 56-case old_quality bank
            cur_old_records = []
            for row in cohorts['old_quality']:
                cid = row['id']
                r_f32, r_u8, _ = infer_v2_eval(val_inputs[cid], u64_val_raw[cid], gamma_now)
                inp_f32 = val_inputs[cid].astype(np.float32) / np.float32(255.0)
                ref_f32 = (val_targets[cid].astype(np.float32) / np.float32(255.0)) if cid in val_targets else inp_f32
                cur_old_records.append({
                    'id': cid,
                    'reference_id': row['reference_id'],
                    'role': row['role'],
                    'split': row['split'],
                    'metrics': {
                        'raw': u96_metrics.case_view_metrics(r_f32, ref_f32, row['role'], np=np),
                        'native_png': u96_metrics.case_view_metrics(
                            r_u8.astype(np.float32) / np.float32(255.0), ref_f32, row['role'], np=np
                        ),
                    },
                })
            inherited_eval = u64_cont_metrics.evaluate_bank(
                cur_old_records, initial_old_quality_records, frozen_U96_records,
                parent_raw, phase=inherited_phase, saved_audit_bytes=audit_raw,
            )

            # 2) Paired V2 monitors (48 cases across 6 V2 roles)
            v2_mon_by_role = {r: [] for r in obj_v2.ROLES_V2}
            v2_mon_cases = {}
            for row in cohorts['v2_monitors']:
                cid = row['id']
                r_f32, r_u8, c_info = infer_v2_eval(val_inputs[cid], u64_val_raw[cid], gamma_now)
                m = paired_case_metrics(r_f32, r_u8, val_targets[cid])
                v2_mon_by_role[row['role']].append(m)
                v2_mon_cases[cid] = {'role': row['role'], 'metrics': m, 'g_blur': c_info['g_blur'], 'g_dark': c_info['g_dark']}

            v2_role_means = {}
            for role_name, items in v2_mon_by_role.items():
                v2_role_means[role_name] = {
                    view: {
                        metric_k: float(np.mean([it[view][metric_k] for it in items]))
                        for metric_k in ('mae', 'ssim7', 'central_lap')
                    }
                    for view in ('raw', 'native_png')
                }

            # 3) Paired V1 monitors (40 cases across 5 V1 roles)
            v1_roles = sorted({r['role'] for r in cohorts['v1_monitors']})
            v1_mon_by_role = {r: [] for r in v1_roles}
            for row in cohorts['v1_monitors']:
                cid = row['id']
                r_f32, r_u8, _ = infer_v2_eval(val_inputs[cid], u64_val_raw[cid], gamma_now)
                v1_mon_by_role[row['role']].append(paired_case_metrics(r_f32, r_u8, val_targets[cid]))
            v1_role_means = {
                role_name: {
                    view: {
                        metric_k: float(np.mean([it[view][metric_k] for it in items]))
                        for metric_k in ('mae', 'ssim7', 'central_lap')
                    }
                    for view in ('raw', 'native_png')
                }
                for role_name, items in v1_mon_by_role.items()
            }

            # 4) Native CCTV DEV sharpness (4 crops)
            native_summary = {}
            for row in cohorts['native_cctv_dev']:
                cid = row['id']
                r_f32, _, c_info = infer_v2_eval(val_inputs[cid], u64_val_raw[cid], gamma_now)
                native_summary[cid] = {
                    'central_lap': inf_v2.central_laplacian_sharpness_numpy(r_f32, np=np),
                    'u64_central_lap': inf_v2.central_laplacian_sharpness_numpy(u64_val_raw[cid], np=np),
                    'g_blur': c_info['g_blur'],
                    'g_dark': c_info['g_dark'],
                }

            degraded_ssim7_mean = float(np.mean([
                v2_role_means[r]['native_png']['ssim7'] for r in obj_v2.NORMAL_DEGRADED_ROLES
            ]))
            degraded_mae_mean = float(np.mean([
                v2_role_means[r]['native_png']['mae'] for r in obj_v2.NORMAL_DEGRADED_ROLES
            ]))
            degraded_lap_mean = float(np.mean([
                v2_role_means[r]['native_png']['central_lap'] for r in obj_v2.NORMAL_DEGRADED_ROLES
            ]))

            summary = {
                'tag': tag,
                'checkpoint': ckpt_pin,
                'max_linf_displacement': disp,
                'gamma': gamma_now,
                'inherited_56_pass': bool(inherited_eval['pass']),
                'inherited_56': inherited_eval,
                'v2_monitor_role_means': v2_role_means,
                'v1_monitor_role_means': v1_role_means,
                'native_cctv_dev': native_summary,
                'v2_degraded_4role_png_ssim7': degraded_ssim7_mean,
                'v2_degraded_4role_png_mae': degraded_mae_mean,
                'v2_degraded_4role_png_lap': degraded_lap_mean,
            }
            write_json(f'checkpoints/{tag}/summary.json', summary)
            log_journal(
                'checkpoint_evaluated',
                tag=tag,
                disp=round(disp, 6),
                inherited_56_pass=bool(inherited_eval['pass']),
                v2_deg_ssim7=round(degraded_ssim7_mean, 5),
                v2_deg_mae=round(degraded_mae_mean, 5),
                v2_deg_lap=round(degraded_lap_mean, 5),
                v2_clear_mae=round(v2_role_means['clear']['native_png']['mae'], 6),
                v2_lowlight_mae=round(v2_role_means['lowlight_sensor_noise']['native_png']['mae'], 6),
            )
            return summary

        summaries = {}
        summaries['u0000_init'] = evaluate_and_save_checkpoint('u0000_init', 'initializer')

        # =====================================================================
        # PHASE 1: Pre-Extract Frozen Encoder Features & 40-Epoch Pyramid Train
        # =====================================================================
        phase = 'phase1_preextract_sft_features'
        clean_target_sft = {}
        clean_target_r2_enc = {}
        clean_target_u8 = {}
        opt_by_ref = {}
        for row in cohorts['v2_optimization']:
            opt_by_ref.setdefault(row['reference_id'], {})[row['role']] = row

        for ref_id in cohorts['optimization_reference_order']:
            t_u8 = load_rgb_u8(opt_by_ref[ref_id]['clear']['target'])
            clean_target_u8[ref_id] = t_u8
            sft_c, enc_c, _ = extract_u96_sft_detached_cpu(t_u8)
            clean_target_sft[ref_id] = sft_c
            clean_target_r2_enc[ref_id] = enc_c

        opt_cached = []
        u64_opt_raw = {}
        for row in cohorts['v2_optimization']:
            inp_u8 = load_rgb_u8(row['input'])
            sft_deg, _, cond_c = extract_u96_sft_detached_cpu(inp_u8)
            u64_opt_raw[row['id']] = forward_u64_raw_f32(inp_u8)
            opt_cached.append({
                'id': row['id'],
                'reference_id': row['reference_id'],
                'role': row['role'],
                'inp_u8': inp_u8,
                'deg_sft': sft_deg,
                'cond': cond_c,
            })
        log_journal('phase1_features_cached', opt_cases=len(opt_cached), clean_targets=len(clean_target_sft))

        phase = 'phase1_clean_sft_distillation'
        p1_cfg = contract.PHASE1_SCHEDULE
        p1_params = list(v2_bundle.pyramid.parameters()) + list(v2_bundle.dark_refiner.parameters())
        opt_p1 = torch.optim.AdamW(
            p1_params,
            lr=p1_cfg['lr_max'],
            betas=tuple(p1_cfg['betas']),
            weight_decay=p1_cfg['weight_decay'],
        )
        rng_p1 = random.Random(SEED + 101)
        p1_step = 0
        total_p1_steps = p1_cfg['total_optimizer_updates']
        warmup_p1_steps = p1_cfg['warmup_epochs'] * p1_cfg['steps_per_epoch']

        v2_bundle.train()
        for epoch in range(1, p1_cfg['epochs'] + 1):
            check_resources()
            order = list(range(len(opt_cached)))
            rng_p1.shuffle(order)
            epoch_losses = []
            for start_i in range(0, len(order), p1_cfg['batch_size']):
                batch_idx = order[start_i:start_i + p1_cfg['batch_size']]
                lr_now = _cosine_lr(p1_step, total_p1_steps, warmup_p1_steps, p1_cfg['lr_max'], p1_cfg['lr_min'])
                for pg in opt_p1.param_groups:
                    pg['lr'] = lr_now

                deg_b = {
                    s: torch.cat([opt_cached[i]['deg_sft'][s] for i in batch_idx], dim=0).to(device)
                    for s in arch_v2.SITES
                }
                clean_b = {
                    s: torch.cat([clean_target_sft[opt_cached[i]['reference_id']][s] for i in batch_idx], dim=0).to(device)
                    for s in arch_v2.SITES
                }
                smaps_b = {
                    s: torch.cat([opt_cached[i]['cond']['site_maps'][s] for i in batch_idx], dim=0).to(device)
                    for s in arch_v2.SITES
                }
                gdark_b = torch.tensor(
                    [opt_cached[i]['cond']['g_dark'] for i in batch_idx], dtype=torch.float32, device=device
                ).view(-1, 1, 1, 1)
                gsn_b = torch.tensor(
                    [opt_cached[i]['cond']['g_sensor_noise'] for i in batch_idx], dtype=torch.float32, device=device
                ).view(-1, 1, 1, 1)

                opt_p1.zero_grad(set_to_none=True)
                pred_sft = v2_bundle.refine_sft_features(
                    deg_b, {'site_maps': smaps_b, 'g_sensor_noise': gsn_b}
                )
                loss_p1, _ = obj_v2.sft_pyramid_distillation_loss_torch(
                    pred_sft, clean_b, deg_b, gdark_b, torch=torch
                )
                loss_p1.backward()
                torch.nn.utils.clip_grad_norm_(p1_params, max_norm=p1_cfg['grad_clip_norm'])
                opt_p1.step()
                p1_step += 1
                epoch_losses.append(float(loss_p1.detach().item()))

            if epoch == 1 or epoch % 5 == 0 or epoch == p1_cfg['epochs']:
                log_journal(
                    'phase1_epoch',
                    epoch=epoch,
                    step=p1_step,
                    mean_sft_loss=round(float(np.mean(epoch_losses)), 6),
                    lr=round(lr_now, 7),
                )

        summaries['p1_e040'] = evaluate_and_save_checkpoint('p1_e040', 'u8')

        # =====================================================================
        # PHASE 2: 224-Update End-to-End Perceptual + SSIM7 + Laplacian Training
        # =====================================================================
        phase = 'phase2_end_to_end_perceptual_training'
        p2_cfg = contract.PHASE2_SCHEDULE
        all_v2_named = sorted(v2_bundle.named_parameters())
        all_v2_params = [param for _, param in all_v2_named]
        opt_p2 = torch.optim.AdamW(
            all_v2_params,
            lr=p2_cfg['lr_max'],
            betas=tuple(p2_cfg['betas']),
            weight_decay=p2_cfg['weight_decay'],
        )

        opt_lookup = {(it['reference_id'], it['role']): it for it in opt_cached}
        rng_p2 = random.Random(SEED + 202)
        p2_schedule_refs = []
        for pass_idx in range(p2_cfg['identity_passes']):
            refs = list(cohorts['optimization_reference_order'])
            if pass_idx > 0:
                rng_p2.shuffle(refs)
            p2_schedule_refs.extend(refs)
        need(len(p2_schedule_refs) == p2_cfg['total_optimizer_updates'], 'Exact 224 Phase-2 updates required')

        perc_weights = {'32': 0.20, '64': 0.25, '128': 0.30, '256': 0.25}

        for u_idx, ref_id in enumerate(p2_schedule_refs, start=1):
            check_resources()
            v2_bundle.train()
            lr_p2 = _cosine_lr(
                u_idx - 1, p2_cfg['total_optimizer_updates'], p2_cfg['warmup_updates'],
                p2_cfg['lr_max'], p2_cfg['lr_min'],
            )
            for pg in opt_p2.param_groups:
                pg['lr'] = lr_p2

            targ_f32 = clean_target_u8[ref_id].astype(np.float32) / np.float32(255.0)
            targ_t = torch.from_numpy(targ_f32.transpose(2, 0, 1).copy())[None].to(device)
            clean_sft_t = {s: clean_target_sft[ref_id][s].to(device) for s in arch_v2.SITES}
            clean_enc_t = {s: clean_target_r2_enc[ref_id][s].to(device) for s in arch_v2.SITES}

            disp_cur = arch_v2.max_linf_displacement_from_init(v2_bundle, v2_init_dict, np=np)
            gamma_cur = max(0.05, cond_v2.displacement_gamma(disp_cur))

            role_grads = {}
            role_loss_log = {}
            for role_name in obj_v2.ROLES_V2:
                item = opt_lookup[(ref_id, role_name)]
                c_stat = item['cond']
                # On clear inputs, g_blur == 0 and g_sensor_noise == 0 -> exact zero structural gradient
                if float(c_stat['g_blur']) == 0.0 and float(c_stat['g_sensor_noise']) == 0.0:
                    role_grads[role_name] = {
                        name: np.zeros(tuple(param.shape), dtype=np.float32)
                        for name, param in all_v2_named
                    }
                    role_loss_log[role_name] = 0.0
                    continue

                inp_f32 = item['inp_u8'].astype(np.float32) / np.float32(255.0)
                inp_t = torch.from_numpy(inp_f32.transpose(2, 0, 1).copy())[None].to(device)
                base_t = torch.from_numpy(u64_opt_raw[item['id']].transpose(2, 0, 1).copy())[None].to(device)

                opt_p2.zero_grad(set_to_none=True)
                _, cond_pred_t, used_f, deg_u96_f, cond_t = forward_v2_connected(inp_t, base_t, gamma_val=gamma_cur)

                pred_enc = extract_r2_enc_connected(cond_pred_t)
                r2_perc = sum(
                    perc_weights[s] * (pred_enc[s] - clean_enc_t[s]).abs().mean()
                    for s in arch_v2.SITES
                )
                sft_ret, _ = obj_v2.sft_pyramid_distillation_loss_torch(
                    used_f, clean_sft_t, deg_u96_f, cond_t['g_dark'], torch=torch
                )
                total_loss, _ = obj_v2.joint_phase2_loss_torch(
                    cond_pred_t, targ_t, torch=torch,
                    g_dark=cond_t['g_dark'], r2_perc_loss=r2_perc, sft_ret_loss=sft_ret,
                )
                total_loss.backward()
                role_loss_log[role_name] = float(total_loss.detach().item())
                role_grads[role_name] = {
                    name: (
                        param.grad.detach().cpu().numpy().astype(np.float32)
                        if param.grad is not None
                        else np.zeros(tuple(param.shape), dtype=np.float32)
                    )
                    for name, param in all_v2_named
                }

            combined_grads, proj_info = obj_v2.project_exposure_split_gradients_numpy(role_grads, np=np)
            opt_p2.zero_grad(set_to_none=True)
            for name, param in all_v2_named:
                param.grad = torch.from_numpy(combined_grads[name]).to(device=device, dtype=torch.float32)
            torch.nn.utils.clip_grad_norm_(all_v2_params, max_norm=p2_cfg['grad_clip_norm'])
            opt_p2.step()

            if u_idx == 1 or u_idx % 8 == 0 or u_idx == p2_cfg['total_optimizer_updates']:
                log_journal(
                    'phase2_step',
                    update=u_idx,
                    ref=ref_id,
                    lr=round(lr_p2, 7),
                    mean_active_loss=round(float(np.mean([v for k, v in role_loss_log.items() if k != 'clear'])), 5),
                    normal_conflicts=proj_info['normal_projected_conflicts'],
                )

            if u_idx in p2_cfg['snapshots']:
                tag = f'p2_u{u_idx:04d}'
                inh_phase = 'u8' if u_idx == 56 else 'u16'
                summaries[tag] = evaluate_and_save_checkpoint(tag, inh_phase)

        # =====================================================================
        # SELECT BEST CHECKPOINT & RENDER L4 RESTORED OUTPUTS
        # =====================================================================
        phase = 'select_best_and_render_outputs'
        init_clear_mae = summaries['u0000_init']['v2_monitor_role_means']['clear']['native_png']['mae']
        candidate_tags = ['p1_e040', 'p2_u0056', 'p2_u0112', 'p2_u0168', 'p2_u0224']
        admissible = [
            t for t in candidate_tags
            if summaries[t]['v2_monitor_role_means']['clear']['native_png']['mae'] <= init_clear_mae + 1e-6
        ]
        pool = admissible if admissible else candidate_tags
        best_tag = max(pool, key=lambda t: summaries[t]['v2_degraded_4role_png_ssim7'])

        best_src = out_dir / f'checkpoints/{best_tag}/trainable_v2.npz'
        best_dst = out_dir / 'checkpoints/best_degraded_ssim/trainable_v2.npz'
        best_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_src, best_dst)
        best_pin = packet.pin(root, best_dst)

        # Load winning weights & render all 148 validation PNGs on L4 GPU
        best_arrays = inf_v2.inspect_v2_trainable_npz(best_dst.read_bytes(), torch=torch, np=np)
        arch_v2.load_v2_trainable_npz_dict(v2_bundle, best_arrays, torch=torch, np=np)
        v2_bundle.eval()
        best_disp = arch_v2.max_linf_displacement_from_init(v2_bundle, v2_init_dict, np=np)
        best_gamma = cond_v2.displacement_gamma(best_disp)

        restored_manifest = {}
        for bank_name in ('v2_monitors', 'v1_monitors', 'old_quality', 'native_cctv_dev'):
            restored_manifest[bank_name] = []
            for row in cohorts[bank_name]:
                cid = row['id']
                _, out_u8, _ = infer_v2_eval(val_inputs[cid], u64_val_raw[cid], best_gamma)
                rel_png = f'restored_best/{bank_name}/{cid}.png'
                png_path = out_dir / rel_png
                png_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(out_u8, mode='RGB').save(png_path, format='PNG', optimize=False)
                restored_manifest[bank_name].append({
                    'id': cid,
                    'role': row.get('role', row.get('dataset', 'native')),
                    'png': packet.pin(root, png_path),
                })

        final_digests = verify_frozen_backbone_digests()
        result_payload = {
            'complete': True,
            'protocol_sha256': expected_sha,
            'status': 'CONDITIONED_SFT_V2_L4_TRAINING_COMPLETE_NOT_AUTO_QUALIFIED',
            'elapsed_seconds': time.monotonic() - began,
            'step0_exact_u64_max_abs_diff': max_step0_neural_diff,
            'frozen_backbone_digests': final_digests,
            'v2_trainable_tensors': v2_tensor_count,
            'v2_trainable_parameters': v2_param_count,
            'best_checkpoint_tag': best_tag,
            'best_checkpoint': best_pin,
            'best_summary': summaries[best_tag],
            'init_summary': summaries['u0000_init'],
            'all_checkpoint_summaries': {
                k: {
                    'v2_degraded_4role_png_ssim7': v['v2_degraded_4role_png_ssim7'],
                    'v2_degraded_4role_png_mae': v['v2_degraded_4role_png_mae'],
                    'v2_degraded_4role_png_lap': v['v2_degraded_4role_png_lap'],
                    'inherited_56_pass': v['inherited_56_pass'],
                    'max_linf_displacement': v['max_linf_displacement'],
                }
                for k, v in summaries.items()
            },
            'restored_best_manifest': restored_manifest,
            'model_qualified': False,
        }
        write_json('result.json', result_payload)
        write_json('status.json', {
            'complete': True,
            'protocol_sha256': expected_sha,
            'status': 'FINISHED',
            'best_checkpoint_tag': best_tag,
            'elapsed_seconds': time.monotonic() - began,
            'model_qualified': False,
        })
        log_journal(
            'worker_complete',
            best_checkpoint_tag=best_tag,
            init_ssim7=round(summaries['u0000_init']['v2_degraded_4role_png_ssim7'], 5),
            best_ssim7=round(summaries[best_tag]['v2_degraded_4role_png_ssim7'], 5),
            init_mae=round(summaries['u0000_init']['v2_degraded_4role_png_mae'], 5),
            best_mae=round(summaries[best_tag]['v2_degraded_4role_png_mae'], 5),
        )
        return 0

    except BaseException as exc:
        err_payload = {
            'complete': False,
            'fatal': True,
            'phase': phase,
            'protocol_sha256': expected_sha,
            'error_type': type(exc).__name__,
            'error': str(exc)[:4000],
            'traceback': traceback.format_exc()[:8000],
            'elapsed_seconds': time.monotonic() - began,
            'model_qualified': False,
        }
        if out_dir.exists():
            try:
                write_json('failure.json', err_payload)
                write_json('status.json', {
                    'complete': False,
                    'protocol_sha256': expected_sha,
                    'status': 'FAILED',
                    'phase': phase,
                    'error_type': type(exc).__name__,
                    'error': str(exc)[:2000],
                    'model_qualified': False,
                })
            except Exception:
                pass
        print(json.dumps(err_payload, sort_keys=True), file=sys.stderr, flush=True)
        return 1
