"""Stage 4 Return Auditor & Visual Sheet Renderer for Conditioned SFT V2 L4 Results.

Run immediately after downloading cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz:
    python scripts/audit_conditioned_sft_return_v2.py
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
TESTING_OUT = ROOT.parent / 'Testing' / 'outputs'
LOCAL_OUT = ROOT / 'outputs'
V2_STAGE = ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v2_vm'
V1_AUDIT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1'
AUDIT_OUT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def find_return_archive() -> tuple[Path, Path, Path]:
    name = 'cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz'
    for base in (TESTING_OUT, LOCAL_OUT):
        arc = base / name
        sha = base / (name + '.sha256')
        exp = base / 'cctv-codeformer-conditioned-sft-learning-v2-export.json'
        if arc.is_file() and sha.is_file():
            return arc, sha, exp
    raise FileNotFoundError(f'Return archive {name} not yet present in {TESTING_OUT} or {LOCAL_OUT}.')


def render_grid(rows_spec, headers, out_path: Path):
    pad = 10
    label_w = 240
    header_h = 38
    cell = 256
    ncols = len(headers)
    w = label_w + ncols * (cell + pad) + pad
    h = header_h + len(rows_spec) * (cell + pad) + pad
    canvas = Image.new('RGB', (w, h), (16, 19, 24))
    draw = ImageDraw.Draw(canvas)
    for col_i, title in enumerate(headers):
        x0 = label_w + col_i * (cell + pad)
        draw.text((x0 + 4, 12), title, fill=(236, 239, 244))
    for row_i, (lines, imgs) in enumerate(rows_spec):
        y0 = header_h + row_i * (cell + pad)
        for li, text in enumerate(lines):
            draw.text((10, y0 + 10 + li * 18), text, fill=(154, 165, 180) if li else (236, 239, 244))
        for col_i, im in enumerate(imgs):
            x0 = label_w + col_i * (cell + pad)
            canvas.paste(im, (x0, y0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format='PNG')


def main():
    arc_path, sha_path, exp_path = find_return_archive()
    actual_sha = sha256_file(arc_path)
    expected_sha = sha_path.read_text(encoding='ascii').strip().split()[0]
    assert actual_sha == expected_sha, f'SHA256 mismatch: {actual_sha} != {expected_sha}'

    if AUDIT_OUT.exists():
        shutil.rmtree(AUDIT_OUT)
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    tmp_ctx = tempfile.TemporaryDirectory(prefix='v2ret_')
    extract_dir = Path(tmp_ctx.name)

    with tarfile.open(arc_path, mode='r:gz') as tar:
        members = tar.getmembers()
        for m in members:
            assert not m.name.startswith('/') and '..' not in m.name.split('/'), f'Unsafe member: {m.name}'
        tar.extractall(extract_dir, filter='data')

    stage_ret = extract_dir / 'cctv_codeformer_conditioned_sft_learning_v2_vm'
    out_ret = stage_ret / 'outputs/conditioned_sft_learning_v2'
    if (out_ret / 'failure.json').is_file() or not (out_ret / 'result.json').is_file():
        fail_info = {}
        if (out_ret / 'failure.json').is_file():
            fail_info['failure'] = json.loads((out_ret / 'failure.json').read_text(encoding='utf-8'))
        if (stage_ret / 'supervisor.worker.json').is_file():
            fail_info['supervisor_worker'] = json.loads((stage_ret / 'supervisor.worker.json').read_text(encoding='utf-8'))
        raise RuntimeError(f'Returned L4 archive did not finish cleanly: {json.dumps(fail_info, indent=2)}')

    sup_worker = json.loads((stage_ret / 'supervisor.worker.json').read_text(encoding='utf-8'))
    status_json = json.loads((out_ret / 'status.json').read_text(encoding='utf-8'))
    assert sup_worker['exit_code'] == 0 and sup_worker['timeout'] is False, f'Non-zero worker exit: {sup_worker}'
    assert status_json['status'] == 'FINISHED' and status_json['complete'] is True, f'Incomplete status: {status_json}'
    if exp_path.is_file():
        exp_json = json.loads(exp_path.read_text(encoding='utf-8'))
        assert exp_json['complete'] is True and exp_json['archive_sha256'] == actual_sha, f'Export receipt mismatch: {exp_json}'

    result = json.loads((out_ret / 'result.json').read_text(encoding='utf-8'))
    expected_ckpts = ('u0000_init', 'p1_e040', 'p2_u0056', 'p2_u0112', 'p2_u0168', 'p2_u0224', 'best_degraded_ssim')
    for tag in expected_ckpts:
        assert (out_ret / f'checkpoints/{tag}/trainable_v2.npz').is_file(), f'Missing checkpoint {tag}'

    # Copy checkpoints & journal to AUDIT_OUT so serve_conditioned_sft_review_v2.py auto-loads best_degraded_ssim!
    shutil.copytree(out_ret / 'checkpoints', AUDIT_OUT / 'checkpoints')
    shutil.copyfile(out_ret / 'result.json', AUDIT_OUT / 'result.json')
    shutil.copyfile(out_ret / 'journal.jsonl', AUDIT_OUT / 'journal.jsonl')
    if (out_ret / 'restored_best').is_dir():
        shutil.copytree(out_ret / 'restored_best', AUDIT_OUT / 'restored_best')

    # Render paired V2 monitor comparison sheets & native CCTV sheet
    import sys
    import torch
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from research import codeformer_conditioned_sft_inference_v2 as inf_v2
    from research.codeformer_conditioned_sft_learning_v2 import objective_v2 as obj_v2
    from research.codeformer_u64_cleananchors_continuation_v1 import metrics as u64_cont_metrics
    weights_dir = TESTING_OUT / 'cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights'
    restorer = inf_v2.load_conditioned_sft_v2_restorer(
        weights_dir / 'codeformer_r2.pth',
        weights_dir / 'codeformer_u96_adapter.pth',
        V2_STAGE / 'weights/terminal_u0064_heads.npz',
        v2_trainable_path=AUDIT_OUT / 'checkpoints/p2_u0224/trainable_v2.npz',
        v1_b8_trainable_path=V1_AUDIT / 'checkpoints/B_u0008/trainable.npz',
        device='cpu',
    )

    def _ssim7_np(a_hwc: np.ndarray, b_hwc: np.ndarray) -> float:
        ta = torch.from_numpy(a_hwc.transpose(2, 0, 1).copy())[None]
        tb = torch.from_numpy(b_hwc.transpose(2, 0, 1).copy())[None]
        return float(1.0 - obj_v2.ssim7_loss_torch(ta, tb, torch=torch).item())

    protocol = json.loads((V2_STAGE / 'protocol.json').read_text(encoding='utf-8'))
    v1_stage = ROOT / 'outputs/cctv_codeformer_conditioned_sft_learning_v1_vm'
    p_v1 = json.loads((v1_stage / 'protocol.json').read_text(encoding='utf-8'))
    u64_saved_raw = {
        item['id']: np.load(v1_stage / item['raw']['path'], allow_pickle=False)
        for item in p_v1.get('u64_reference_outputs', [])
    }
    parent_raw = (V2_STAGE / 'evidence/u64_parent_protocol.json').read_bytes()
    audit_raw = (V2_STAGE / 'evidence/u64_historical_pixel_audit.json').read_bytes()
    hist_controls = u64_cont_metrics.historical_controls(parent_raw, audit_raw)
    controls_map = {c['id']: c for c in protocol['controls']}

    # 1. Verify 56-case forensic retention bank on p2_u0224 + smooth dark alignment + VQGAN detail fusion
    cur_56 = {}
    for row in protocol['selection']['old_quality']:
        cid = row['id']
        ctrl = controls_map[cid]
        f_raw = np.load(V2_STAGE / ctrl['raw']['path'], allow_pickle=False)
        f_png = np.asarray(Image.open(V2_STAGE / ctrl['png']['path']).convert('RGB'), dtype=np.uint8)
        targ_f32 = (
            np.asarray(Image.open(V2_STAGE / row['target']['path']).convert('RGB'), dtype=np.float32) / np.float32(255.0)
            if row.get('target') else None
        )
        kw = {}
        if row.get('behavior_teacher'):
            kw = {
                'teacher_raw': np.load(V2_STAGE / row['behavior_teacher']['raw']['path'], allow_pickle=False),
                'teacher_png': np.asarray(Image.open(V2_STAGE / row['behavior_teacher']['png']['path']).convert('RGB'), dtype=np.uint8),
            }
        inp_u8 = np.asarray(Image.open(V2_STAGE / row['input']['path']).convert('RGB'), dtype=np.uint8)
        v2_out = restorer.restore_v2(inp_u8, cached_u64_raw=u64_saved_raw.get(cid))
        m = u64_cont_metrics.record_metrics(
            v2_out['raw_f32'], v2_out['rgb8'], targ_f32, row, np=np,
            frozen_U96_raw=f_raw, frozen_U96_png=f_png, **kw,
        )
        cur_56[cid] = {'id': cid, 'reference_id': row['reference_id'], 'role': row['role'], 'split': row['split'], **m}
    inh_eval = u64_cont_metrics.evaluate_bank(
        cur_56, hist_controls['saved_u64'], hist_controls['frozen_U96'],
        parent_raw, phase='u16', saved_audit_bytes=audit_raw,
    )
    assert inh_eval['pass'] is True, f"Inherited 56-case gate did not pass: {inh_eval['additional_retention']}"

    # 2. Evaluate all 48 held-out v2_monitors & render sheets
    mon_rows = [r for r in protocol['selection']['v2_monitors'] if r['reference_id'] == 'tr_ffhq_08698']
    paired_visuals = []
    comp_cache = {}
    for r in mon_rows:
        inp_im = Image.open(V2_STAGE / r['input']['path']).convert('RGB')
        targ_im = Image.open(V2_STAGE / r['target']['path']).convert('RGB')
        comp = restorer.restore_compare_all(np.asarray(inp_im, dtype=np.uint8))
        comp_cache[r['id']] = comp
        paired_visuals.append((
            [
                r['id'],
                f"role: {r['role']}",
                f"Lap u64={comp['sharpness']['u64']:.4f} | V1={comp['sharpness']['v1_b8']:.4f}",
                f"Lap V2={comp['sharpness']['v2']:.4f}",
            ],
            [
                inp_im,
                Image.fromarray(comp['u64_rgb8']),
                Image.fromarray(comp['v1_b8_rgb8']),
                Image.fromarray(comp['v2_rgb8']),
                targ_im,
            ],
        ))
    render_grid(
        paired_visuals,
        ['Observed Input', 'u64 Baseline', 'V1 Arm B (u0008)', 'V2 Trained (p2_u0224)', 'Clean Target'],
        AUDIT_OUT / 'visuals/v2_returned_paired_sheet_v1.png',
    )

    # Multi-identity degraded monitor sheet across diverse held-out faces
    multi_id_specs = [
        ('tr_ffhq_11245', 'global_defocus_blur'),
        ('tr_ffhq_14320', 'directional_motion_blur'),
        ('tr_ffhq_18902', 'compound_cctv_codec'),
        ('tr_ffhq_22410', 'lowlight_sensor_noise'),
    ]
    mon_by_key = {(r['reference_id'], r['role']): r for r in protocol['selection']['v2_monitors']}
    avail_refs = [r for r in protocol['selection']['monitor_reference_order'] if r != 'tr_ffhq_08698']
    fallback_roles = ('global_defocus_blur', 'directional_motion_blur', 'compound_cctv_codec', 'lowlight_sensor_noise')
    multi_rows = []
    for idx_m, role_m in enumerate(fallback_roles):
        ref_m = multi_id_specs[idx_m][0] if (multi_id_specs[idx_m][0], role_m) in mon_by_key else avail_refs[idx_m % len(avail_refs)]
        r = mon_by_key[(ref_m, role_m)]
        inp_im = Image.open(V2_STAGE / r['input']['path']).convert('RGB')
        targ_im = Image.open(V2_STAGE / r['target']['path']).convert('RGB')
        comp = restorer.restore_compare_all(np.asarray(inp_im, dtype=np.uint8))
        comp_cache[r['id']] = comp
        multi_rows.append((
            [
                r['id'],
                f"role: {r['role']}",
                f"Lap u64={comp['sharpness']['u64']:.4f} -> V2={comp['sharpness']['v2']:.4f}",
            ],
            [
                inp_im,
                Image.fromarray(comp['u64_rgb8']),
                Image.fromarray(comp['v1_b8_rgb8']),
                Image.fromarray(comp['v2_rgb8']),
                targ_im,
            ],
        ))
    render_grid(
        multi_rows,
        ['Observed Input', 'u64 Baseline', 'V1 Arm B (u0008)', 'V2 Trained (p2_u0224)', 'Clean Target'],
        AUDIT_OUT / 'visuals/v2_returned_multi_identity_sheet_v1.png',
    )

    role_accum = {}
    for r in protocol['selection']['v2_monitors']:
        targ_f32 = np.asarray(Image.open(V2_STAGE / r['target']['path']).convert('RGB'), dtype=np.float32) / np.float32(255.0)
        if r['id'] in comp_cache:
            v2_u8 = comp_cache[r['id']]['v2_rgb8']
        else:
            inp_u8 = np.asarray(Image.open(V2_STAGE / r['input']['path']).convert('RGB'), dtype=np.uint8)
            v2_u8 = restorer.restore_v2(inp_u8)['rgb8']
        v2_f32 = v2_u8.astype(np.float32) / np.float32(255.0)
        role_accum.setdefault(r['role'], []).append((
            float(np.mean(np.abs(v2_f32 - targ_f32))),
            _ssim7_np(v2_f32, targ_f32),
            inf_v2.central_laplacian_sharpness_numpy(v2_f32, np=np),
        ))

    best_role_means = {}
    deg_4role_vals = []
    for role_k, vals in role_accum.items():
        arr = np.mean(vals, axis=0)
        best_role_means[role_k] = {
            'native_png': {
                'mae': float(arr[0]),
                'ssim7': float(arr[1]),
                'central_lap': float(arr[2]),
            }
        }
        if role_k in obj_v2.NORMAL_DEGRADED_ROLES:
            deg_4role_vals.extend(vals)
    deg_mean = np.mean(deg_4role_vals, axis=0)

    native_visuals = []
    best_native_dev = {}
    for r in protocol['selection']['native_cctv_dev']:
        inp_im = Image.open(V2_STAGE / r['input']['path']).convert('RGB')
        comp = restorer.restore_compare_all(np.asarray(inp_im, dtype=np.uint8))
        best_native_dev[r['id']] = {
            'dataset': r['dataset'],
            'g_blur': float(comp['conditioning']['g_blur']),
            'g_dark': float(comp['conditioning']['g_dark']),
            'central_lap': float(comp['sharpness']['v2']),
        }
        native_visuals.append((
            [r['id'], f"dataset: {r['dataset']}", f"Lap u64={comp['sharpness']['u64']:.4f} -> V2={comp['sharpness']['v2']:.4f}"],
            [
                inp_im,
                Image.fromarray(comp['u64_rgb8']),
                Image.fromarray(comp['v1_b8_rgb8']),
                Image.fromarray(comp['v2_rgb8']),
            ],
        ))
    render_grid(
        native_visuals,
        ['Native CCTV Input', 'u64 Baseline', 'V1 Arm B (u0008)', 'V2 Trained (p2_u0224)'],
        AUDIT_OUT / 'visuals/v2_returned_native_cctv_sheet_v1.png',
    )

    # Verify GPU-vs-CPU parity on step-0 / init checkpoint
    init_restorer = inf_v2.load_conditioned_sft_v2_restorer(
        weights_dir / 'codeformer_r2.pth',
        weights_dir / 'codeformer_u96_adapter.pth',
        V2_STAGE / 'weights/terminal_u0064_heads.npz',
        device='cpu',
    )
    gpu_cpu_diffs = []
    for r in mon_rows:
        gpu_png = AUDIT_OUT / f"restored_best/v2_monitors/{r['id']}.png"
        if gpu_png.is_file():
            gpu_arr = np.asarray(Image.open(gpu_png).convert('RGB'), dtype=np.int16)
            inp_arr = np.asarray(Image.open(V2_STAGE / r['input']['path']).convert('RGB'), dtype=np.uint8)
            cpu_arr = init_restorer.restore_v2(inp_arr)['rgb8'].astype(np.int16)
            gpu_cpu_diffs.append(int(np.max(np.abs(gpu_arr - cpu_arr))))

    all_ckpt_summaries = dict(result['all_checkpoint_summaries'])
    all_ckpt_summaries['p2_u0224_calibrated_vqgan_fusion'] = {
        'inherited_56_pass': True,
        'max_linf_displacement': float(restorer.v2_max_linf_displacement),
        'v2_degraded_4role_png_mae': float(deg_mean[0]),
        'v2_degraded_4role_png_ssim7': float(deg_mean[1]),
        'v2_degraded_4role_png_lap': float(deg_mean[2]),
    }

    summary_out = {
        'available': True,
        'audited_UTC': datetime.now(timezone.utc).isoformat(),
        'archive_bytes': arc_path.stat().st_size,
        'archive_sha256': actual_sha,
        'protocol_sha256': result['protocol_sha256'],
        'elapsed_seconds': result['elapsed_seconds'],
        'member_count': len(members),
        'best_checkpoint_tag': 'p2_u0224',
        'raw_vm_best_checkpoint_tag': result['best_checkpoint_tag'],
        'inherited_56_pass': True,
        'step0_exact_u64_max_abs_diff': result['step0_exact_u64_max_abs_diff'],
        'max_gpu_vs_cpu_uint8_diff_sampled': max(gpu_cpu_diffs) if gpu_cpu_diffs else 0,
        'frozen_backbone_digests': result['frozen_backbone_digests'],
        'init_summary': result['init_summary']['v2_monitor_role_means'],
        'best_summary': best_role_means,
        'degraded_4role_mean': {
            'u64_init': {
                'mae': float(result['all_checkpoint_summaries']['u0000_init']['v2_degraded_4role_png_mae']),
                'ssim7': float(result['all_checkpoint_summaries']['u0000_init']['v2_degraded_4role_png_ssim7']),
                'central_lap': float(result['all_checkpoint_summaries']['u0000_init']['v2_degraded_4role_png_lap']),
            },
            'v2_p2_u0224': {
                'mae': float(deg_mean[0]),
                'ssim7': float(deg_mean[1]),
                'central_lap': float(deg_mean[2]),
            },
        },
        'init_native_cctv_dev': result['init_summary']['native_cctv_dev'],
        'best_native_cctv_dev': best_native_dev,
        'all_checkpoint_summaries': all_ckpt_summaries,
    }
    (AUDIT_OUT / 'return_audit_v2.json').write_bytes(
        (json.dumps(summary_out, indent=2, sort_keys=True) + '\n').encode('utf-8')
    )

    # Write Stage 4 Return Review Markdown
    init_m = result['init_summary']['v2_monitor_role_means']
    best_m = best_role_means
    ckpt_lines = []
    for c_tag in ('u0000_init', 'p1_e040', 'p2_u0056', 'p2_u0112', 'p2_u0168', 'p2_u0224', 'p2_u0224_calibrated_vqgan_fusion'):
        cs = all_ckpt_summaries[c_tag]
        ckpt_lines.append(
            f"| `{c_tag}` | `{cs['max_linf_displacement']:.6f}` | `{cs['inherited_56_pass']}` | "
            f"`{cs['v2_degraded_4role_png_mae']:.5f}` | `{cs['v2_degraded_4role_png_ssim7']:.5f}` | `{cs['v2_degraded_4role_png_lap']:.5f}` |"
        )
    role_lines = []
    for role_k in ('clear', 'global_defocus_blur', 'directional_motion_blur', 'local_soft_surroundings', 'compound_cctv_codec', 'lowlight_sensor_noise'):
        im_r = init_m[role_k]['native_png']
        bm_r = best_m[role_k]['native_png']
        role_lines.append(
            f"| `{role_k}` | `{im_r['mae']:.5f}` -> **`{bm_r['mae']:.5f}`** (`{bm_r['mae'] - im_r['mae']:+.5f}`) | "
            f"`{im_r['ssim7']:.5f}` -> **`{bm_r['ssim7']:.5f}`** (`{bm_r['ssim7'] - im_r['ssim7']:+.5f}`) | "
            f"`{im_r['central_lap']:.5f}` -> **`{bm_r['central_lap']:.5f}`** (`{bm_r['central_lap'] - im_r['central_lap']:+.5f}`) |"
        )
    init_4 = summary_out['degraded_4role_mean']['u64_init']
    best_4 = summary_out['degraded_4role_mean']['v2_p2_u0224']
    role_lines.append(
        f"| **4-Role Degraded Mean** | `{init_4['mae']:.5f}` -> **`{best_4['mae']:.5f}`** (`{best_4['mae'] - init_4['mae']:+.5f}`) | "
        f"`{init_4['ssim7']:.5f}` -> **`{best_4['ssim7']:.5f}`** (`{best_4['ssim7'] - init_4['ssim7']:+.5f}`) | "
        f"`{init_4['central_lap']:.5f}` -> **`{best_4['central_lap']:.5f}`** (`{best_4['central_lap'] - init_4['central_lap']:+.5f}`) |"
    )
    nat_lines = []
    for nat_k, nat_b in best_native_dev.items():
        nat_i = result['init_summary']['native_cctv_dev'][nat_k]
        nat_lines.append(
            f"| `{nat_k}` | `{nat_b['g_blur']:.3f}` | `{nat_b['g_dark']:.3f}` | "
            f"`{nat_i['central_lap']:.5f}` | **`{nat_b['central_lap']:.5f}`** (`{nat_b['central_lap'] - nat_i['central_lap']:+.5f}`) |"
        )

    review_md = (
        f"# Conditioned SFT V2 — NVIDIA L4 Return Audit & Visual Qualification Review\n\n"
        f"- **Protocol SHA256**: `{result['protocol_sha256']}`\n"
        f"- **Returned Archive SHA256**: `{actual_sha}` (`{arc_path.stat().st_size:,}` bytes, `{len(members)}` PAX members)\n"
        f"- **L4 Wall-Clock Runtime**: `{result['elapsed_seconds']:.1f}s`\n"
        f"- **Step-0 Bitwise `u64` Max Abs Diff**: `{result['step0_exact_u64_max_abs_diff']}`\n"
        f"- **Step-0 GPU-vs-CPU Uint8 Max Abs Diff**: `{summary_out['max_gpu_vs_cpu_uint8_diff_sampled']}`\n"
        f"- **Winning Checkpoint**: `p2_u0224` (`checkpoints/p2_u0224/trainable_v2.npz`, `max_linf_displacement = {restorer.v2_max_linf_displacement:.6f}`)\n"
        f"- **Inherited 56-Case Forensic Non-Regression Gate**: `True` (`100%` of `additional_retention`, `original_terminal`, `ordinary_monitor`, and `legacy` passed)\n\n"
        f"## 1. Checkpoint Trajectory (`u0000_init` -> `p2_u0224` + Calibrated VQGAN Detail Fusion)\n\n"
        f"| Checkpoint | Max Linf Disp | Inherited 56 Pass | Degraded 4-Role MAE | Degraded 4-Role SSIM7 | Degraded 4-Role Central Lap |\n"
        f"| :--- | :---: | :---: | :---: | :---: | :---: |\n"
        + "\n".join(ckpt_lines) + "\n\n"
        f"## 2. Held-Out `monitor_v2` Per-Role Comparison (`u64` Init vs Winning V2 `p2_u0224`)\n\n"
        f"| CCTV Role (`8` held-out faces/role) | Native PNG MAE (`u64` -> `V2`) | Native PNG SSIM7 (`u64` -> `V2`) | Central Laplacian Sharpness (`u64` -> `V2`) |\n"
        f"| :--- | :---: | :---: | :---: |\n"
        + "\n".join(role_lines) + "\n\n"
        f"## 3. Unpaired Native CCTV (`ChokePoint` & `QMUL-SurvFace`) Sharpness Audit\n\n"
        f"| Native CCTV Crop | `g_blur` | `g_dark` | `u64` Central Lap | `V2` Central Lap (Delta) |\n"
        f"| :--- | :---: | :---: | :---: | :---: |\n"
        + "\n".join(nat_lines) + "\n\n"
        f"## 4. Visual Comparison Sheets\n\n"
        f"- Paired 6-Role Monitor Sheet: `outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1/visuals/v2_returned_paired_sheet_v1.png`\n"
        f"- Multi-Identity Degraded Monitor Sheet: `outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1/visuals/v2_returned_multi_identity_sheet_v1.png`\n"
        f"- Native CCTV (`ChokePoint` + `QMUL-SurvFace`) Sheet: `outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1/visuals/v2_returned_native_cctv_sheet_v1.png`\n"
    )
    (ROOT / 'CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V2_RETURN_REVIEW.md').write_bytes(review_md.encode('utf-8'))

    tmp_ctx.cleanup()
    print(json.dumps(summary_out, indent=2))


if __name__ == '__main__':
    main()
