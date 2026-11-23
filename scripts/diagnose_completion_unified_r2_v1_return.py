"""Detailed subgroup and trajectory diagnostic for Unified R2 v1 return."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / 'outputs/completion_codeformer_unified_r2_v1_return/'
    'cctv_completion_codeformer_unified_r2_v1_results/outputs'
)


def main():
    res = json.loads((OUT / 'results.json').read_text(encoding='utf-8'))
    print(
        f"Worker seconds: {res['seconds']:.2f}, cycles: {res['cycles_completed']}, stop_reason: {res['stop_reason']}"
    )
    print('Counts:', res['counts'])

    u0_m_full = json.loads((OUT / 'u0000/monitor/results.json').read_text(encoding='utf-8'))
    u0_c_full = json.loads((OUT / 'u0000/challenge/results.json').read_text(encoding='utf-8'))
    u0_m = u0_m_full['metrics']
    u0_c = u0_c_full['metrics']
    u0_m_agree = sum(
        r['same_codebook_clean_context_agreement_diagnostic'] for r in u0_m_full['records']
    ) / len(u0_m_full['records'])
    u0_c_agree = sum(
        r['same_codebook_clean_context_agreement_diagnostic'] for r in u0_c_full['records']
    ) / len(u0_c_full['records'])
    print(f'Cycle 0 code agreement: clear={u0_m_agree:.4f}, challenge={u0_c_agree:.4f}')

    for cyc in (8, 16, 32):
        s = json.loads((OUT / f'screen_u{cyc:04d}.json').read_text(encoding='utf-8'))
        m_full = json.loads((OUT / f'u{cyc:04d}/monitor/results.json').read_text(encoding='utf-8'))
        c_full = json.loads((OUT / f'u{cyc:04d}/challenge/results.json').read_text(encoding='utf-8'))
        m = m_full['metrics']
        c = c_full['metrics']
        m_agree = sum(
            r['same_codebook_clean_context_agreement_diagnostic'] for r in m_full['records']
        ) / len(m_full['records'])
        c_agree = sum(
            r['same_codebook_clean_context_agreement_diagnostic'] for r in c_full['records']
        ) / len(c_full['records'])
        print(f'\n=== Cycle {cyc} (code agree: clear={m_agree:.4f}, challenge={c_agree:.4f}) ===')
        print(
            f"  Clear PNG hole gain: {s['clear']['PNG_macro_hole_gain']*100:+.4f}% | nonworse: {s['clear']['all_raw_PNG_hole_boundary_subgroups_nonworse']}"
        )
        print(
            f"  Challenge PNG hole gain: {s['challenge']['PNG_macro_hole_gain']*100:+.4f}% | nonworse: {s['challenge']['all_raw_PNG_hole_boundary_subgroups_nonworse']}"
        )
        print(
            f"  GRAY chroma pass: {s['GRAY_style_gate']['all_GRAY_macro_geometry_chroma_nonworse']} | macro PNG spread: {s['GRAY_style_gate']['macro_current']['GRAY_PNG_mean_channel_spread']:.4f} (R1: {s['GRAY_style_gate']['macro_baseline_R1']['GRAY_PNG_mean_channel_spread']:.4f})"
        )
        for k in ('raw_hole_L1', 'raw_boundary_L1', 'PNG_hole_L1', 'PNG_boundary_L1'):
            d_m = (m['macro'][k] - u0_m['macro'][k]) / u0_m['macro'][k] * 100
            d_c = (c['macro'][k] - u0_c['macro'][k]) / u0_c['macro'][k] * 100
            print(f'    macro {k:16s}: clear {d_m:+.4f}% | challenge {d_c:+.4f}%')
        for g in ('lower_face', 'eye_band', 'image_left_lens', 'cheek_object'):
            for k in ('raw_hole_L1', 'raw_boundary_L1', 'PNG_hole_L1', 'PNG_boundary_L1'):
                d_m = (m['geometry'][g][k] - u0_m['geometry'][g][k]) / u0_m['geometry'][g][k] * 100
                d_c = (c['geometry'][g][k] - u0_c['geometry'][g][k]) / u0_c['geometry'][g][k] * 100
                flag_m = ' [REGR]' if m['geometry'][g][k] > u0_m['geometry'][g][k] + 1e-12 else ''
                flag_c = ' [REGR]' if c['geometry'][g][k] > u0_c['geometry'][g][k] + 1e-12 else ''
                print(
                    f'    geom/{g:15s} {k:16s}: clear {d_m:+.4f}%{flag_m} | chal {d_c:+.4f}%{flag_c}'
                )
        for cond in sorted(c['condition']):
            for k in ('raw_hole_L1', 'raw_boundary_L1', 'PNG_hole_L1', 'PNG_boundary_L1'):
                d_c = (
                    (c['condition'][cond][k] - u0_c['condition'][cond][k])
                    / u0_c['condition'][cond][k]
                    * 100
                )
                flag_c = (
                    ' [REGR]'
                    if c['condition'][cond][k] > u0_c['condition'][cond][k] + 1e-12
                    else ''
                )
                if flag_c or k == 'PNG_hole_L1':
                    print(f'    cond/{cond:24s} {k:16s}: chal {d_c:+.4f}%{flag_c}')

    u32_m_map = {
        r['id']: r
        for r in json.loads((OUT / 'u0032/monitor/results.json').read_text(encoding='utf-8'))[
            'records'
        ]
    }
    u32_c_map = {
        r['id']: r
        for r in json.loads((OUT / 'u0032/challenge/results.json').read_text(encoding='utf-8'))[
            'records'
        ]
    }
    print('\n--- Clear Monitor per-case PNG_hole_L1 deltas (Cycle 32 vs R1) ---')
    for r0 in u0_m_full['records']:
        cid = r0['id']
        r32 = u32_m_map[cid]
        d = (r32['PNG_hole_L1'] - r0['PNG_hole_L1']) / r0['PNG_hole_L1'] * 100
        da = (
            r32['same_codebook_clean_context_agreement_diagnostic']
            - r0['same_codebook_clean_context_agreement_diagnostic']
        )
        print(
            f"{cid:36s} ({r0['mask_kind']:15s}): hole {d:+6.2f}% ({r0['PNG_hole_L1']:.4f} -> {r32['PNG_hole_L1']:.4f}) | code_agree {da:+.4f}"
        )

    print('\n--- Challenge per-case PNG_hole_L1 deltas (Cycle 32 vs R1) ---')
    for r0 in u0_c_full['records']:
        cid = r0['id']
        r32 = u32_c_map[cid]
        d = (r32['PNG_hole_L1'] - r0['PNG_hole_L1']) / r0['PNG_hole_L1'] * 100
        da = (
            r32['same_codebook_clean_context_agreement_diagnostic']
            - r0['same_codebook_clean_context_agreement_diagnostic']
        )
        print(
            f"{cid:36s} ({r0['mask_kind']:15s}, {r0['condition']:28s}): hole {d:+6.2f}% ({r0['PNG_hole_L1']:.4f} -> {r32['PNG_hole_L1']:.4f}) | code_agree {da:+.4f}"
        )

    import numpy as np

    print('\n--- Saved token agreement inside 16x16 hole vs visible (Clear & Challenge) ---')
    for kind, records0, records32 in [
        ('monitor', u0_m_full['records'], u32_m_map),
        ('challenge', u0_c_full['records'], u32_c_map),
    ]:
        hole_ag0, hole_ag32 = [], []
        vis_ag0, vis_ag32 = [], []
        for r0 in records0:
            cid = r0['id']
            z0 = np.load(OUT / f'u0000/{kind}/{cid}.npz')
            z32 = np.load(OUT / f'u0032/{kind}/{cid}.npz')
            m256 = z0['mask256'][0, 0]
            m16 = m256.reshape(16, 16, 16, 16).max(axis=(1, 3)).reshape(1, 256) > 0.5
            tkey = cid.split('__')[0] + '::RGB' if kind == 'monitor' else None
            c0 = z0['predicted_codes']
            c32 = z32['predicted_codes']
            diff_hole = int((c0[m16] != c32[m16]).sum())
            diff_vis = int((c0[~m16] != c32[~m16]).sum())
            d_hole = (records32[cid]['PNG_hole_L1'] - r0['PNG_hole_L1']) / r0['PNG_hole_L1'] * 100
            if r0['mask_kind'] in ('eye_band', 'cheek_object'):
                print(
                    f"[{kind:9s}] {cid:32s} ({r0['mask_kind']:12s}): hole_tokens_changed={diff_hole:2d}/{int(m16.sum()):2d} | vis_tokens_changed={diff_vis:2d} | hole_L1={d_hole:+6.2f}%"
                )


if __name__ == '__main__':
    main()
