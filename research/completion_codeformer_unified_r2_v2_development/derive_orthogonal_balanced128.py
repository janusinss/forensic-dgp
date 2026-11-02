"""Deterministic de-aliased orthogonal balanced128-case subset from exact retained256FIT pool; noNN."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / 'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/'
    'cctv_completion_codeformer_delivered_objective_v4_vm/protocol.json'
)
OUT = (
    ROOT
    / 'research/completion_codeformer_unified_r2_v2_development/'
    'orthogonal_balanced128_subset.json'
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def need(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def pin(p: Path) -> dict:
    return {'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def main() -> None:
    need(
        sha(SOURCE) == '8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d',
        'Exact frozen pool protocol',
    )
    p = json.loads(SOURCE.read_text(encoding='utf-8'))
    pool = p['fit_cases']
    files = {r['path']: r for r in p['files']}
    refs = sorted({r['reference_id'] for r in pool})
    need(len(refs) == 32 and len(pool) == 256, 'Expected full pool')
    allgeometry = ('lower_face', 'eye_band', 'image_left_lens', 'cheek_object')
    grayconditions = ('strict_gray_clear', 'strict_gray_mild_degraded')
    rgbconditions = ('RGB_mild_degraded', 'RGB_clear_irregular')

    selected = []
    roles = {}
    for i, ref in enumerate(refs):
        cases = [r for r in pool if r['reference_id'] == ref]
        need(len(cases) == 8, 'Eight fixed cases/ref')
        gray = next(r for r in cases if r['condition'] == grayconditions[(i // 4) % 2])
        rgb = next(r for r in cases if r['condition'] == rgbconditions[(i // 8) % 2])
        need(gray['mask_kind'] != rgb['mask_kind'], 'Chosen augmentation geometries unique')
        anchors = [
            r
            for r in cases
            if r['role'] == 'fit_anchor'
            and r['mask_kind'] not in (gray['mask_kind'], rgb['mask_kind'])
        ]
        need(len(anchors) == 2, 'Two complementary clear anchors')
        role = sorted([*anchors, gray, rgb], key=lambda r: allgeometry.index(r['mask_kind']))
        need({r['mask_kind'] for r in role} == set(allgeometry), 'One each geometry/ref')
        roles[ref] = role
        selected.extend(role)

    patterns = []
    pat_map = {}
    for ref in refs:
        role = roles[ref]
        pat = tuple(
            'GRAY'
            if r['style'] == 'GRAY'
            else ('RGB_AUG' if r['role'] == 'fit_augmented' else 'ANCHOR')
            for r in role
        )
        if pat not in pat_map:
            pat_map[pat] = []
            patterns.append(pat)
        pat_map[pat].append(ref)
    need(len(patterns) == 8 and all(len(v) == 4 for v in pat_map.values()), 'Eight 4-ref patterns')

    order = [0, 2, 1, 3, 4, 6, 5, 7]
    schedule = []
    for idx in order:
        grouprefs = pat_map[patterns[idx]]
        for turn in range(4):
            schedule.append([roles[ref][(turn + j) % 4]['id'] for j, ref in enumerate(grouprefs)])

    byid = {r['id']: r for r in selected}
    flat = [i for row in schedule for i in row]
    need(
        len(selected) == len(set(flat)) == len(flat) == 128 and set(flat) == set(byid),
        '128 unique exact once',
    )
    need(len(schedule) == 32, '32 four-case cycles')

    for row in schedule:
        cases = [byid[i] for i in row]
        need(
            len({r['reference_id'] for r in cases}) == 4
            and len({r['mask_kind'] for r in cases}) == 4
            and sum(r['role'] == 'fit_anchor' for r in cases) == 2
            and sum(r['style'] == 'GRAY' for r in cases) == 1
            and sum(r['role'] == 'fit_augmented' and r['style'] == 'RGB' for r in cases) == 1,
            'Every cycle 4 refs, 4 geometries, 2 anchors, 1 gray aug, 1 RGB aug',
        )

    for w in range(4):
        w_cases = [byid[cid] for row in schedule[w * 8 : (w + 1) * 8] for cid in row]
        for g in allgeometry:
            need(
                sum(r['role'] == 'fit_anchor' and r['mask_kind'] == g for r in w_cases) == 4
                and sum(r['role'] == 'fit_augmented' and r['mask_kind'] == g for r in w_cases) == 4,
                f'Window {w} 4 anchor / 4 augmented per geometry {g}',
            )

    counts = {
        k: dict(Counter(r[k] for r in selected))
        for k in ('role', 'style', 'mask_kind', 'condition', 'reference_id')
    }
    role_by_geometry = {
        f"{role}::{g}": sum(r['role'] == role and r['mask_kind'] == g for r in selected)
        for role in ('fit_anchor', 'fit_augmented')
        for g in allgeometry
    }
    condition_by_geometry = {
        f"{cond}::{g}": sum(r['condition'] == cond and r['mask_kind'] == g for r in selected)
        for cond in ('original_clear_anchor', *grayconditions, *rgbconditions)
        for g in allgeometry
    }
    need(
        counts['role'] == {'fit_anchor': 64, 'fit_augmented': 64}
        and counts['style'] == {'RGB': 96, 'GRAY': 32}
        and set(counts['mask_kind'].values()) == {32}
        and set(counts['reference_id'].values()) == {4}
        and set(role_by_geometry.values()) == {16},
        'Orthogonal balanced totals',
    )
    need(
        counts['condition']['original_clear_anchor'] == 64
        and all(counts['condition'][c] == 16 for c in (*grayconditions, *rgbconditions))
        and all(
            condition_by_geometry[f"{c}::{g}"] == 4
            for c in (*grayconditions, *rgbconditions)
            for g in allgeometry
        ),
        'Aug conditions 16 total and 4 per geometry each',
    )
    need(
        len({r['teacher_key'] for r in selected}) == 64
        and not (
            {r['reference_id'] for r in selected}
            & {r['reference_id'] for r in p['monitor_cases'] + p['challenge_cases']}
        ),
        'All 64 FIT labels consumed; monitor disjoint',
    )

    data = {}
    for r in selected:
        for field in ('input256', 'clean256', 'clean512', 'mask'):
            row = files[r[field]]
            q = SOURCE.parent / r[field]
            need(
                q.stat().st_size == row['bytes'] and sha(q) == row['sha256'],
                'Existing asset drift',
            )
            data[r[field]] = row

    report = {
        'format': 'independent-orthogonal-balanced128-retained-subset-design-v2',
        'complete': True,
        'independent': True,
        'created_UTC': '2026-10-29T00:00:00+00:00',
        'deriver': pin(Path(__file__)),
        'source_protocol': pin(SOURCE),
        'source_full_pool_cases': 256,
        'retained_full_pool_modified': False,
        'subset_cases': 128,
        'references': 32,
        'teacher_keys_consumed': 64,
        'deterministic_selection': (
            'Sort 32 FIT reference IDs. At reference index i choose graycondition[(i//4)%2] (period 8) '
            'and RGBcondition[(i//8)%2] (period 16) to eliminate modulo-4 resonance with (i+j)%4. '
            'Choose clear anchor cases in the 2 complementary mask geometries and order each reference '
            'by canonical geometry index (lower_face, eye_band, image_left_lens, cheek_object).'
        ),
        'deterministic_schedule': (
            'Group the 32 references into 8 four-reference groups by their 4-geometry role pattern in '
            'paired complementary order [0,2,1,3,4,6,5,7]. For turn 0..3, reference position j takes '
            'geometry index (turn+j)%4. Every cycle has 4 distinct refs, all 4 distinct geometries, '
            '2 clear anchors, 1 GRAY augmented, and 1 RGB augmented.'
        ),
        'cases_in_reference_role_order': selected,
        'schedule': schedule,
        'schedule_flat_ids': flat,
        'balanced_counts': counts,
        'role_by_geometry_counts': role_by_geometry,
        'condition_by_geometry_counts': condition_by_geometry,
        'all_cases_drawn_from_existing_FIT_pool_exact': True,
        'every_reference_oneeach4geometries': True,
        'every_cycle_all4geometries_exact': True,
        'every_8cycle_window_4anchor_4augmented_per_geometry': True,
        'every_selected_case_once': True,
        'monitor_challenge_FINAL_never_optimized': True,
        'pinned_existing_assets': list(data.values()),
        'optimizer_steps_prospective': 32,
        'backwards_prospective': 128,
        'all256FITcases_used_in_this_pilot': False,
        'full256pool_preserved_for_other_future_designs': True,
        'subset_selection_uses_no_monitor_or_quality_score': True,
        'local_actions': {
            'Torch_imported': False,
            'neural_calls': 0,
            'gradients': 0,
            'optimizer_objects_or_steps': 0,
            'VM_calls': 0,
            'new_training_targets': 0,
        },
        'runtime_or_packet_admitted': False,
        'model_qualified': False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(
        json.dumps(
            {
                'complete': True,
                'subset': pin(OUT),
                'role_by_geometry': role_by_geometry,
                'steps': 32,
                'cases': 128,
                'labels': 64,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
