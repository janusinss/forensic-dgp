"""Bind actual all-page observations and explicitly secondary saved-score analysis."""
import numpy as np
from automatic_proposal_score_v1_common import ROOT, OUT, sha, read, write, binary, bindings

NOTES = {
    '00_cloth_mask': ['Fragmented covering edges; ordinary hair, brow and background are also marked. Much central cloth is missed.',
        'Most cloth is missed; markings shift to forehead, temple and lower side of the face.'],
    '01_pink_mask': ['Partial cloth/lace proposal also marks visible eye regions; broad core gaps remain.',
        'Empty automatic area despite a visible covering.'],
    '02_mask_with_clear_glasses': ['Mask core missed; tiny frame/temple marks. Retain the unsupported-pose exclusion.',
        'Mask core missed; tiny nose/cheek marks. Retain the unsupported-pose exclusion.'],
    '03_dark_sunglasses': ['Empty automatic area despite dark lenses hiding eyes.', 'Empty automatic area despite dark lenses hiding eyes.'],
    '04_sunglasses': ['Empty automatic area despite dark lenses hiding eyes.', 'Empty automatic area despite dark lenses hiding eyes.'],
    '05_white_glare': ['Empty automatic area despite opaque lens glare; visible frame remains to preserve.',
        'Empty automatic area despite opaque lens glare.'],
    '06_mirrored_glare': ['Both mirrored lenses substantially marked, with small core gaps and frame/skin overlap.',
        'Broader lens proposal with small peripheral hair/skin marks; no hidden identity claim.'],
    '07_hand_over_mask': ['Nearly all hand/mask core missed; small ear/strap mark.', 'Empty automatic area despite hand and mask.'],
    '08_uncovered': ['False proposal marks visible eyes, hair edges and background on an uncovered control.',
        'Appropriate empty proposal on an uncovered control; not evidence of a reliable covering classifier.'],
    '09_clear_glasses': ['Appropriate empty proposal preserves clear glasses and ordinary hair.',
        'Appropriate empty proposal preserves clear glasses and ordinary hair.'],
    'val_18_hand_eyes': ['Fragmented finger marks miss broad hand areas and also mark mouth, nose, earrings and non-obstructing head accessories.',
        'Empty automatic area despite both hands covering the eyes.'],
    'val_25_hand_mouth': ['Most hands missed; sparse hair/boundary and finger marks.', 'Empty automatic area despite hands covering mouth.'],
    'val_362_hair_eye': ['Obstructing hair at eye missed; tiny unrelated hair-side mark.',
        'Empty automatic area despite obstructing hair at eye.'],
    'val_244_knit_scarf': ['Substantial scarf marks with gaps, but visible eyes and neck scarf outside facial support are also marked.',
        'Empty automatic area despite scarf covering the lower face.'],
    'val_336_scarf_gloves': ['Fragmented scarf/glove texture marks leave core gaps.',
        'Empty automatic area despite scarf/gloves covering lower face.'],
    'val_6_flower_mouth': ['Empty automatic area despite flower obstructing mouth.', 'Empty automatic area despite flower obstructing mouth.'],
    'val_7_leaf_eye': ['Most leaf missed; a narrow leaf edge and ordinary blue hair are marked.',
        'Most leaf marked, with limited adjacent visible-feature overlap; all-case usefulness remains unqualified.'],
    'val_26_nearly_hidden_hands': ['Nearly hidden face is not represented by the sparse finger/wrist proposal. Retain less-covered-input request.',
        'Empty proposal does not imply usable face. Retain less-covered-input request.']}


def main():
    p, r, audit = [read(OUT/name) for name in ['protocol.json', 'results.json', 'independent_audit.json']]
    bindings(p); assert audit['complete'] and audit['results_sha256'] == sha(OUT/'results.json')
    assert audit['exact256_page_cells'] == 180 and audit['fresh_exact_detector_replays'] == 2
    assert len(NOTES) == 18 and set(NOTES) == {c['base_id'] for c in p['cases']}
    rows = []; witnesses = []
    for case, measurement in zip(p['cases'], r['rows']):
        note = NOTES[case['base_id']][int(case['condition'] != 'original_photo')]
        rows.append({'id': case['id'], 'family': case['family'], 'condition': case['condition'],
            'actually_viewed': True, 'input_exclusion_retained': case['rejected'], 'observation': note,
            'generation_outputs_reviewed': 0, 'automatic_completion_quality_qualified': False})
        if not case.get('masks'): continue
        core = binary(ROOT/case['masks']['core']); protected = binary(ROOT/case['masks']['protected'])
        assert not (core & protected).any()
        if not core.any() or not protected.any(): continue
        with np.load(OUT/'stages'/(case['id']+'.npz'), allow_pickle=False) as saved: score = saved['canvas_probability']
        core_yx = np.argwhere(core)[int(np.argmin(score[core]))]
        protected_yx = np.argwhere(protected)[int(np.argmax(score[protected]))]
        a, b = float(score[tuple(core_yx)]), float(score[tuple(protected_yx)])
        witnesses.append({'id': case['id'], 'core_xy': core_yx[::-1].tolist(),
            'protected_xy': protected_yx[::-1].tolist(), 'core_score': a, 'protected_score': b,
            'strict_order_inversion': b > a,
            'scope': 'Two fixed approximate operator-footprint pixels; not expert segmentation truth'})
    condition_summary = {}
    for condition in ['original_photo', 'synthetic_degraded_photo']:
        subset = [row for row in r['rows'] if row['condition'] == condition]
        covering = [row for row in subset if row['regions'].get('assisted_core', {}).get('pixels', 0) > 0]
        condition_summary[condition] = {'requests': len(subset), 'empty_proposals_all_cases': sum(v['proposal_pixels'] == 0 for v in subset),
            'eligible_nonempty_core_cases': len(covering), 'empty_proposals_covering_cases': sum(v['proposal_pixels'] == 0 for v in covering),
            'protected_pixels_proposed': sum(v['proposal_region_pixels'].get('protected_appearance', 0) for v in subset),
            'cases_with_protected_pixels_proposed': sum(v['proposal_region_pixels'].get('protected_appearance', 0) > 0 for v in subset),
            'mean_core_fraction_above_fixed_threshold': float(np.mean([v['regions']['assisted_core']['fraction_ge_0_5'] for v in covering]))}
    write(OUT/'secondary_score_order_analysis.json', {'complete': True, 'post_hoc_saved_score_analysis': True,
        'not_a_preregistered_quality_gate': True, 'protocol_sha256': sha(OUT/'protocol.json'),
        'results_sha256': sha(OUT/'results.json'), 'witnesses': witnesses,
        'witness_count': len(witnesses), 'strict_inversions': sum(v['strict_order_inversion'] for v in witnesses),
        'condition_summary': condition_summary, 'threshold_search': False, 'new_annotations': False,
        'new_neural_calls': 0, 'optimizer_updates': 0, 'app_adoption': False, 'goal_complete': False})
    write(OUT/'visual_review.json', {'complete': True, 'reviewer': 'Primary assistant input/proposal development review',
        'independent_final_reviewer': False, 'all9_pages_actually_viewed_at_original_resolution': True,
        'all36_cases_actually_viewed': True, 'page_sha256': {name: digest for name, digest in r['artifact_sha256'].items() if name.startswith('pages/')},
        'protocol_sha256': sha(OUT/'protocol.json'), 'results_sha256': sha(OUT/'results.json'),
        'independent_audit_sha256': sha(OUT/'independent_audit.json'),
        'secondary_analysis_sha256': sha(OUT/'secondary_score_order_analysis.json'), 'cases': rows,
        'automatic_generation_outputs': 0, 'assisted_generation_outputs': 0,
        'automatic_completion_quality_qualified': False, 'assisted_completion_quality_qualified': False,
        'app_adoption': False, 'native_or_final_used': False, 'goal_complete': False,
        'recording_neural_calls': 0, 'recording_gradient_queries': 0, 'recording_optimizer_updates': 0})
    print({'complete': True, 'reviewed_proposals': 36, 'strict_score_order_inversions': sum(v['strict_order_inversion'] for v in witnesses)})


if __name__ == '__main__': main()
