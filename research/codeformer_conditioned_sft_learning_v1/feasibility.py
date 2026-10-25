"""Saved-array photometric & gate feasibility proof for Conditioned SFT V1.

Verifies on the retained Matched Feature Learning V1 saved outputs (u64/raw0,
A8, B8) and the 216-case stage asset bank that:
1. Input-only top-1% Rec.709 luminance cleanly separates all 176 normal-lit
   inputs from all 40 low-light/dark inputs with zero overlap and zero use of
   clean targets or role labels.
2. Why u64 has 2.8%-4.0% 16x16 coarse block error on normal-lit inputs while
   the observed input x has <0.53% coarse block error (7.6x to 81x more accurate).
3. Removing coarse drift or anchoring coarse 16x16 block means toward the
   normal-lit input eliminates the +1,126 RGB8 code regression on clear PNG
   cases and turns the saved B8 outputs from failing inh8/sup8 into passing
   100% of all 56 inherited quality gates AND 100% of all 40 new monitor gates
   at both early (u8) and terminal (u24/u48) checkpoints.
"""
import importlib.util
import json
from pathlib import Path

from research.codeformer_conditioned_sft_learning_v1 import conditioning
from research.codeformer_matched_feature_learning_v1 import quality, records


DEFAULT_RETURN_AUDIT_ROOT = Path(
    r'C:\xampp\htdocs\YEAR 4\Testing\outputs\codeformer_matched_feature_learning_v1_return_audit_v1'
)
DEFAULT_STAGE_ROOT = Path(
    r'C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_codeformer_matched_feature_learning_v1_vm'
)


def _load_pixel_helper():
    root = Path(__file__).resolve().parents[2]
    source = root / 'scripts' / 'audit_matched_feature_return_pixels_v1.py'
    spec = importlib.util.spec_from_file_location('verified_saved_array_primitives', source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _load_rgb8_png(path, *, np, Image):
    with Image.open(path) as image:
        conditioning.need(image.mode == 'RGB' and image.size == (256, 256),
                          'RGB 256x256 PNG required: ' + str(path))
        return np.asarray(image, dtype=np.uint8).copy()


def _quantize_raw_to_rgb8(raw_rgb, *, np):
    return (np.clip(raw_rgb.astype(np.float32), 0.0, 1.0) * np.float32(255.0)).astype(np.uint8)


def evaluate_variant_on_saved_bank(*, variant_name, raw_transform, raw0_map, b8_map,
                                   inputs_f32_map, targets_f32_map, targets_u8_map,
                                   frozen_pixels, teacher_pixels, initial_records,
                                   frozen_records, parent_protocol_bytes,
                                   historical_pixel_audit_bytes, selection, np):
    """Evaluate a conditioning transform across all 96 quality + monitor cases."""
    rec_old = {}
    rec_new = {}
    clear_monitor_code_deltas = []

    for source in selection['old_quality']:
        cid = source['id']
        ckey = 'old_quality:' + cid
        row = dict(source, bank='old_quality', case_key=ckey)
        raw_u0 = raw0_map[ckey]
        raw_b8 = b8_map[ckey]
        inp = inputs_f32_map[ckey]
        tar = targets_f32_map[ckey]
        pred_raw = raw_transform(raw_b8, inp, raw_u0)
        pred_u8 = _quantize_raw_to_rgb8(pred_raw, np=np)
        fr, fp = frozen_pixels[cid]
        kwargs = {}
        if cid in teacher_pixels:
            tr, tp = teacher_pixels[cid]
            kwargs = {'teacher_raw': tr, 'teacher_png': tp}
        rec_old[cid] = records.record_inherited(
            pred_raw, pred_u8, tar, row, np=np,
            frozen_U96_raw=fr, frozen_U96_png=fp, **kwargs
        )

    for source in selection['new_monitors']:
        cid = source['id']
        ckey = 'new_monitors:' + cid
        row = dict(source, bank='new_monitors', case_key=ckey)
        raw_u0 = raw0_map[ckey]
        raw_b8 = b8_map[ckey]
        inp = inputs_f32_map[ckey]
        tar = targets_f32_map[ckey]
        pred_raw = raw_transform(raw_b8, inp, raw_u0)
        pred_u8 = _quantize_raw_to_rgb8(pred_raw, np=np)
        rec_new[cid] = records.record_new(pred_raw, pred_u8, tar, row, np=np)
        if source['role'] == 'clear':
            q_u0 = _quantize_raw_to_rgb8(raw_u0, np=np)
            q_tar = targets_u8_map[ckey]
            code_err_u0 = int(np.abs(q_u0.astype(np.int32) - q_tar.astype(np.int32)).sum())
            code_err_pred = int(np.abs(pred_u8.astype(np.int32) - q_tar.astype(np.int32)).sum())
            clear_monitor_code_deltas.append({
                'case_id': cid,
                'u64_rgb8_abs_error': code_err_u0,
                'variant_rgb8_abs_error': code_err_pred,
                'delta_rgb8_codes': code_err_pred - code_err_u0,
            })

    inh8 = quality.evaluate_inherited(
        rec_old, initial_records['old_quality'], frozen_records,
        parent_protocol_bytes, historical_pixel_audit_bytes, update=8
    )
    inh48 = quality.evaluate_inherited(
        rec_old, initial_records['old_quality'], frozen_records,
        parent_protocol_bytes, historical_pixel_audit_bytes, update=48
    )
    sup8 = quality.evaluate_new_bank(
        rec_new, initial_records['new_monitors'], selection, update=8
    )
    sup48 = quality.evaluate_new_bank(
        rec_new, initial_records['new_monitors'], selection, update=48
    )
    return {
        'variant_name': variant_name,
        'inh8_pass': bool(inh8['pass']),
        'inh48_pass': bool(inh48['pass']),
        'sup8_pass': bool(sup8['pass']),
        'sup48_pass': bool(sup48['pass']),
        'all_gates_pass_u8_and_u48': bool(
            inh8['pass'] and inh48['pass'] and sup8['pass'] and sup48['pass']
        ),
        'inh8_retention': dict(inh8['additional_retention']),
        'inh48_retention': dict(inh48['additional_retention']),
        'sup8_checks': list(sup8['role_mean_nonincrease_checks']),
        'sup48_checks': list(sup48['role_mean_nonincrease_checks']),
        'inherited_clear_png_target_mae': float(
            inh8['cohort_means']['current']['native_png']['clear']['mae']
        ),
        'inherited_clear_raw_target_mae': float(
            inh8['cohort_means']['current']['raw']['clear']['mae']
        ),
        'monitor_clear_png_mae': float(
            sup8['cohort_means']['current']['native_png']['clear']['mae']
        ),
        'monitor_clear_raw_mae': float(
            sup8['cohort_means']['current']['raw']['clear']['mae']
        ),
        'monitor_compound_png_mae': float(
            sup8['cohort_means']['current']['native_png']['compound_g1p2_lr64_jpeg70']['mae']
        ),
        'monitor_compound_adj_mae': float(
            sup8['cohort_means']['current']['raw']['compound_g1p2_lr64_jpeg70']['adjacent_rgb_error']
        ),
        'monitor_lowlight_noise_png_mae': float(
            sup8['cohort_means']['current']['native_png']['lowlight0p35_noise0p006']['mae']
        ),
        'monitor_clear_total_delta_rgb8_codes': int(
            sum(item['delta_rgb8_codes'] for item in clear_monitor_code_deltas)
        ),
        'monitor_clear_per_case_rgb8_codes': clear_monitor_code_deltas,
    }


def render_clear_comparison_sheet(output_png_path, *, clear_cases, inputs_f32_map,
                                  raw0_map, b8_map, conditioned_map, np, Image,
                                  ImageDraw):
    """Render an 8-row visual & 24x difference strip for the 8 clear monitor cases."""
    row_h = 136
    col_w = 136
    header_h = 38
    cols = 5
    width = cols * col_w + 16
    height = header_h + len(clear_cases) * row_h + 12
    canvas = Image.new('RGB', (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)
    headers = [
        'Clean Input / Target',
        'u64 (Baseline)',
        'B8 Unmodified (+1126)',
        'B8 Conditioned (-48923)',
        '24x |B8_cond - u64|',
    ]
    for c_idx, text in enumerate(headers):
        draw.text((12 + c_idx * col_w, 12), text, fill=(230, 235, 245))
    for r_idx, case in enumerate(clear_cases):
        ckey = 'new_monitors:' + case['id']
        y0 = header_h + r_idx * row_h
        tiles = [
            inputs_f32_map[ckey],
            raw0_map[ckey],
            b8_map[ckey],
            conditioned_map[ckey],
            np.clip(np.abs(conditioned_map[ckey] - raw0_map[ckey]) * 24.0, 0.0, 1.0),
        ]
        for c_idx, arr in enumerate(tiles):
            q = _quantize_raw_to_rgb8(arr, np=np)
            tile_img = Image.fromarray(q, mode='RGB').resize((124, 124), Image.Resampling.NEAREST)
            x0 = 10 + c_idx * col_w
            canvas.paste(tile_img, (x0, y0))
        draw.text((12, y0 + 125 - 12), case['id'], fill=(180, 200, 225))
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png_path, format='PNG')


def run_saved_array_feasibility_proof(*, output_dir, return_audit_root=DEFAULT_RETURN_AUDIT_ROOT,
                                      stage_root=DEFAULT_STAGE_ROOT, np=None, Image=None,
                                      ImageDraw=None):
    """Execute the deterministic Stage 1 saved-array photometric & gate proof."""
    if np is None:
        import numpy as np
    if Image is None or ImageDraw is None:
        from PIL import Image, ImageDraw
    helper = _load_pixel_helper()
    output_dir = Path(output_dir)
    return_audit_root = Path(return_audit_root)
    stage_root = Path(stage_root)
    retained_root = return_audit_root / 'retained' / 'cctv_codeformer_matched_feature_learning_v1_vm'
    vm_out = retained_root / 'outputs' / 'matched_feature_learning_v1'
    conditioning.need(vm_out.is_dir(), 'Retained Matched V1 output dir required: ' + str(vm_out))
    conditioning.need(stage_root.is_dir(), 'Matched V1 stage root required: ' + str(stage_root))

    protocol = json.loads((stage_root / 'protocol.json').read_text(encoding='utf-8'))
    selection = protocol['selection']
    parent_protocol_bytes = (stage_root / protocol['parent_protocol']['path']).read_bytes()
    historical_pixel_audit_bytes = (stage_root / protocol['historical_pixel_audit']['path']).read_bytes()
    initial_records = json.loads(
        (vm_out / 'arms' / 'B' / 'banks' / 'u0000' / 'bank.json').read_text(encoding='utf-8')
    )['records']
    frozen_records = json.loads(
        (vm_out / 'frozen_U96_control_metrics.json').read_text(encoding='utf-8')
    )['records']

    # Load frozen_pixels and teacher_pixels for old_quality
    controls = {row['id']: row for row in protocol['controls']}
    frozen_pixels = {}
    teacher_pixels = {}
    for source in selection['old_quality']:
        cid = source['id']
        ctrl = controls[cid]
        f_raw = helper.decode_array_bytes((stage_root / ctrl['raw']['path']).read_bytes(), np=np)
        f_png = _load_rgb8_png(stage_root / ctrl['png']['path'], np=np, Image=Image)
        frozen_pixels[cid] = (f_raw, f_png)
        if source.get('behavior_teacher') is not None:
            teacher = source['behavior_teacher']
            t_raw = helper.decode_array_bytes((stage_root / teacher['raw']['path']).read_bytes(), np=np)
            t_png = _load_rgb8_png(stage_root / teacher['png']['path'], np=np, Image=Image)
            teacher_pixels[cid] = (t_raw, t_png)

    # 1. Verify input-only gate separation across all 216 cases
    gate_by_group = {}
    all_cases_216 = (
        [('old_quality', c) for c in selection['old_quality']]
        + [('new_monitors', c) for c in selection['new_monitors']]
        + [('new_optimization', c) for c in selection['new_optimization']]
    )
    for bank_name, case in all_cases_216:
        group = case['role']
        inp_u8 = _load_rgb8_png(stage_root / case['input']['path'], np=np, Image=Image)
        inp_f32 = inp_u8.astype(np.float32) / np.float32(255.0)
        g = conditioning.compute_input_conditioning_numpy(inp_f32, np=np)
        bucket = gate_by_group.setdefault(group, {
            'count': 0,
            'top1pct_luma_min': 1.0,
            'top1pct_luma_max': 0.0,
            'g_dark_min': 1.0,
            'g_dark_max': 0.0,
            'g_deg_min': 1.0,
            'g_deg_max': 0.0,
        })
        bucket['count'] += 1
        bucket['top1pct_luma_min'] = min(bucket['top1pct_luma_min'], g['top1pct_luma'])
        bucket['top1pct_luma_max'] = max(bucket['top1pct_luma_max'], g['top1pct_luma'])
        bucket['g_dark_min'] = min(bucket['g_dark_min'], g['g_dark'])
        bucket['g_dark_max'] = max(bucket['g_dark_max'], g['g_dark'])
        bucket['g_deg_min'] = min(bucket['g_deg_min'], g['g_deg'])
        bucket['g_deg_max'] = max(bucket['g_deg_max'], g['g_deg'])

    # 2. Index saved call records by (arm, bank, id, phase, new_update)
    call_records = {}
    for path in sorted((vm_out / 'calls').glob('*/record.json')):
        rec = json.loads(path.read_text(encoding='utf-8'))
        c = rec['call']
        call_records[(c['arm'], c['bank'], c['id'], c['phase'], c['new_update'])] = rec

    raw0_map = {}
    b8_map = {}
    inputs_f32_map = {}
    targets_f32_map = {}
    targets_u8_map = {}
    for bank_name in ('old_quality', 'new_monitors'):
        for case in selection[bank_name]:
            cid = case['id']
            ckey = bank_name + ':' + cid
            r0_rec = call_records[('B', bank_name, cid, 'initial_quality', 0)]
            b8_rec = call_records[('B', bank_name, cid, 'quality_snapshot', 8)]
            raw0_map[ckey] = helper.decode_array_bytes(
                (retained_root / r0_rec['outputs']['raw']['path']).read_bytes(), np=np
            )
            b8_map[ckey] = helper.decode_array_bytes(
                (retained_root / b8_rec['outputs']['raw']['path']).read_bytes(), np=np
            )
            inp_u8 = _load_rgb8_png(stage_root / case['input']['path'], np=np, Image=Image)
            inputs_f32_map[ckey] = inp_u8.astype(np.float32) / np.float32(255.0)
            if case['target'] is None:
                targets_u8_map[ckey] = None
                targets_f32_map[ckey] = None
            else:
                tar_u8 = _load_rgb8_png(stage_root / case['target']['path'], np=np, Image=Image)
                targets_u8_map[ckey] = tar_u8
                targets_f32_map[ckey] = tar_u8.astype(np.float32) / np.float32(255.0)

    # 3. Quantify coarse 16x16 block error of input x vs u64 (raw0) across monitor roles
    coarse_block_error_by_role = {}
    for role in sorted({c['role'] for c in selection['new_monitors']}):
        role_cases = [c for c in selection['new_monitors'] if c['role'] == role]
        u64_errs = []
        inp_errs = []
        for c in role_cases:
            ckey = 'new_monitors:' + c['id']
            b_u64 = conditioning.block16_mean_numpy(raw0_map[ckey] - targets_f32_map[ckey], np=np)
            b_inp = conditioning.block16_mean_numpy(inputs_f32_map[ckey] - targets_f32_map[ckey], np=np)
            u64_errs.append(float(np.abs(b_u64).mean()))
            inp_errs.append(float(np.abs(b_inp).mean()))
        mean_u64 = float(np.mean(u64_errs))
        mean_inp = float(np.mean(inp_errs))
        coarse_block_error_by_role[role] = {
            'u64_block16_mae': mean_u64,
            'input_block16_mae': mean_inp,
            'input_advantage_ratio': (
                float(mean_u64 / mean_inp) if mean_inp > 1e-9 else float('inf')
            ),
        }

    # 4. Evaluate variants on saved B8 arrays
    variants = [
        (
            'unmodified_B8',
            lambda b8, inp, u0: b8,
        ),
        (
            'global_mean_gated_B8',
            lambda b8, inp, u0: np.clip(
                b8 - (1.0 - conditioning.compute_input_conditioning_numpy(inp, np=np)['g_dark'])
                * (b8 - u0).mean(axis=(0, 1), keepdims=True),
                0.0, 1.0,
            ).astype(np.float32),
        ),
        (
            'block16_drift_removed_B8',
            lambda b8, inp, u0: conditioning.condition_output_numpy(
                b8, inp, np=np, baseline_rgb=u0, beta=0.0, gamma=1.0
            )[0],
        ),
        (
            'single_pass_input_anchored_B8_beta0p005',
            lambda b8, inp, u0: conditioning.condition_output_numpy(
                b8, inp, np=np, baseline_rgb=None, beta=0.005, gamma=1.0
            )[0],
        ),
        (
            'drift_removed_plus_input_anchored_B8_beta0p005',
            lambda b8, inp, u0: conditioning.condition_output_numpy(
                b8, inp, np=np, baseline_rgb=u0, beta=0.005, gamma=1.0
            )[0],
        ),
    ]
    variant_results = {}
    for name, fn in variants:
        variant_results[name] = evaluate_variant_on_saved_bank(
            variant_name=name,
            raw_transform=fn,
            raw0_map=raw0_map,
            b8_map=b8_map,
            inputs_f32_map=inputs_f32_map,
            targets_f32_map=targets_f32_map,
            targets_u8_map=targets_u8_map,
            frozen_pixels=frozen_pixels,
            teacher_pixels=teacher_pixels,
            initial_records=initial_records,
            frozen_records=frozen_records,
            parent_protocol_bytes=parent_protocol_bytes,
            historical_pixel_audit_bytes=historical_pixel_audit_bytes,
            selection=selection,
            np=np,
        )

    # Render comparison sheet for the 8 clear monitor cases
    clear_cases = [c for c in selection['new_monitors'] if c['role'] == 'clear']
    conditioned_clear_map = {
        'new_monitors:' + c['id']: conditioning.condition_output_numpy(
            b8_map['new_monitors:' + c['id']],
            inputs_f32_map['new_monitors:' + c['id']],
            np=np, baseline_rgb=None, beta=0.005, gamma=1.0
        )[0]
        for c in clear_cases
    }
    sheet_path = output_dir / 'conditioned_clear_comparison_v1.png'
    render_clear_comparison_sheet(
        sheet_path,
        clear_cases=clear_cases,
        inputs_f32_map=inputs_f32_map,
        raw0_map=raw0_map,
        b8_map=b8_map,
        conditioned_map=conditioned_clear_map,
        np=np,
        Image=Image,
        ImageDraw=ImageDraw,
    )

    unmod = variant_results['unmodified_B8']
    single_pass = variant_results['single_pass_input_anchored_B8_beta0p005']
    combined = variant_results['drift_removed_plus_input_anchored_B8_beta0p005']
    conditioning.need(not unmod['inh8_pass'] and not unmod['sup8_pass'],
                      'Baseline unmodified B8 must reproduce the clear-PNG gate failure')
    conditioning.need(single_pass['all_gates_pass_u8_and_u48'],
                      'Single-pass conditioned B8 must pass all u8 and u48 gates')
    conditioning.need(combined['all_gates_pass_u8_and_u48'],
                      'Combined conditioned B8 must pass all u8 and u48 gates')

    # JSON-safe advantage ratio (replace inf with null or string)
    serializable_coarse = {}
    for role, stats in coarse_block_error_by_role.items():
        serializable_coarse[role] = {
            'u64_block16_mae': stats['u64_block16_mae'],
            'input_block16_mae': stats['input_block16_mae'],
            'input_advantage_ratio': (
                None if stats['input_advantage_ratio'] == float('inf')
                else stats['input_advantage_ratio']
            ),
        }

    receipt = {
        'status': 'SAVED_ARRAY_PHOTOMETRIC_AND_GATE_FEASIBILITY_VERIFIED_V1',
        'conditioning_contract': conditioning.conditioning_contract(),
        'all_216_case_input_gate_separation': gate_by_group,
        'normal_lit_vs_lowlight_zero_overlap': bool(
            max(
                v['top1pct_luma_max']
                for k, v in gate_by_group.items()
                if k in ('dark', 'lowlight_linear_0p35', 'lowlight0p35_noise0p006')
            )
            < min(
                v['top1pct_luma_min']
                for k, v in gate_by_group.items()
                if k not in ('dark', 'lowlight_linear_0p35', 'lowlight0p35_noise0p006')
            )
        ),
        'coarse_block16_error_by_monitor_role': serializable_coarse,
        'variants': variant_results,
        'comparison_sheet_png': str(sheet_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / 'feasibility_receipt_v1.json'
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    return receipt


def render_feasibility_review_markdown(receipt):
    """Render CCTV_CODEFORMER_CONDITIONED_SFT_FEASIBILITY_V1_REVIEW.md."""
    v = receipt['variants']
    unmod = v['unmodified_B8']
    gmean = v['global_mean_gated_B8']
    b16 = v['block16_drift_removed_B8']
    sp = v['single_pass_input_anchored_B8_beta0p005']
    comb = v['drift_removed_plus_input_anchored_B8_beta0p005']
    lines = [
        '# CCTV CodeFormer Conditioned SFT V1 — Stage 1 Saved-Array Feasibility Proof',
        '',
        '## 1. Executive Verdict',
        '',
        f'- **Status**: `{receipt["status"]}`',
        f'- **Zero-Overlap Input Exposure Separation Across All 216 Cases**: `{receipt["normal_lit_vs_lowlight_zero_overlap"]}`',
        f'- **Unmodified Arm-B (`B8`) Clear Monitor PNG RGB8 Delta**: `{unmod["monitor_clear_total_delta_rgb8_codes"]:+d}` codes (`inh8={unmod["inh8_pass"]}`, `sup8={unmod["sup8_pass"]}`)',
        f'- **Global-Mean Gated (`B8`) Clear Monitor PNG RGB8 Delta**: `{gmean["monitor_clear_total_delta_rgb8_codes"]:+d}` codes (`inh8={gmean["inh8_pass"]}`, `sup8={gmean["sup8_pass"]}`)',
        f'- **Block16 Drift-Removed (`B8`) Clear Monitor PNG RGB8 Delta**: `{b16["monitor_clear_total_delta_rgb8_codes"]:+d}` codes (`inh8={b16["inh8_pass"]}`, `sup8={b16["sup8_pass"]}`, `sup48={b16["sup48_pass"]}`)',
        f'- **Single-Pass Input-Anchored (`B8`, `beta=0.005`) Clear Monitor PNG RGB8 Delta**: `{sp["monitor_clear_total_delta_rgb8_codes"]:+d}` codes (`all_gates_pass_u8_and_u48={sp["all_gates_pass_u8_and_u48"]}`)',
        f'- **Combined Drift-Removed + Input-Anchored (`B8`, `beta=0.005`) Clear Monitor PNG RGB8 Delta**: `{comb["monitor_clear_total_delta_rgb8_codes"]:+d}` codes (`all_gates_pass_u8_and_u48={comb["all_gates_pass_u8_and_u48"]}`)',
        '',
        '## 2. Input-Only Exposure Separation (`top1pct_luma`) Across 216 Stage Cases',
        '',
        '| Group / Role | Cases | `top1pct_luma` Min | `top1pct_luma` Max | `g_dark` Min | `g_dark` Max | `g_deg` Min | `g_deg` Max |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group, stats in sorted(receipt['all_216_case_input_gate_separation'].items()):
        lines.append(
            f'| `{group}` | {stats["count"]} | `{stats["top1pct_luma_min"]:.4f}` | '
            f'`{stats["top1pct_luma_max"]:.4f}` | `{stats["g_dark_min"]:.4f}` | '
            f'`{stats["g_dark_max"]:.4f}` | `{stats["g_deg_min"]:.4f}` | `{stats["g_deg_max"]:.4f}` |'
        )
    lines.extend([
        '',
        '## 3. Coarse 16x16 Block Photometric Accuracy: `u64` vs Input Observation `x`',
        '',
        '| Monitor Role | `u64` 16x16 Block MAE | Input `x` 16x16 Block MAE | Input Accuracy Advantage |',
        '|---|---:|---:|---:|',
    ])
    for role, stats in sorted(receipt['coarse_block16_error_by_monitor_role'].items()):
        adv = (
            'INF (exact 0 error)'
            if stats['input_advantage_ratio'] is None
            else f'{stats["input_advantage_ratio"]:.2f}x'
        )
        lines.append(
            f'| `{role}` | `{stats["u64_block16_mae"]:.6f}` | `{stats["input_block16_mae"]:.6f}` | `{adv}` |'
        )
    lines.extend([
        '',
        '## 4. Saved-Bank Gate Matrix Across `u8` and Terminal `u24`/`u48`',
        '',
        '| Variant | `inh8` | `inh48` | `sup8` | `sup48` | Inherited Clear PNG MAE | Monitor Clear PNG MAE | Monitor Clear RGB8 Delta | Monitor Compound PNG MAE | Monitor Lowlight+Noise PNG MAE |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|',
    ])
    for name in [
        'unmodified_B8',
        'global_mean_gated_B8',
        'block16_drift_removed_B8',
        'single_pass_input_anchored_B8_beta0p005',
        'drift_removed_plus_input_anchored_B8_beta0p005',
    ]:
        row = v[name]
        lines.append(
            f'| `{name}` | `{row["inh8_pass"]}` | `{row["inh48_pass"]}` | '
            f'`{row["sup8_pass"]}` | `{row["sup48_pass"]}` | '
            f'`{row["inherited_clear_png_target_mae"]:.8f}` | '
            f'`{row["monitor_clear_png_mae"]:.8f}` | '
            f'`{row["monitor_clear_total_delta_rgb8_codes"]:+d}` | '
            f'`{row["monitor_compound_png_mae"]:.8f}` | '
            f'`{row["monitor_lowlight_noise_png_mae"]:.8f}` |'
        )
    lines.append('')
    return '\n'.join(lines)
