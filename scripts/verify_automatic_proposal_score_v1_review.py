"""Independent recording, witness-coordinate and aggregate arithmetic verification."""
import numpy as np
from pathlib import Path
from automatic_proposal_score_v1_common import ROOT, OUT, read, write, sha, binary, bindings


def main():
    p, r, v, s, audit = [read(OUT/name) for name in ['protocol.json', 'results.json', 'visual_review.json',
        'secondary_score_order_analysis.json', 'independent_audit.json']]
    bindings(p); assert audit['complete']
    assert v['complete'] and v['all36_cases_actually_viewed'] and v['all9_pages_actually_viewed_at_original_resolution']
    assert not v['independent_final_reviewer'] and not v['app_adoption'] and not v['native_or_final_used'] and not v['goal_complete']
    assert [row['id'] for row in v['cases']] == [c['id'] for c in p['cases']]
    for c, row in zip(p['cases'], v['cases']):
        assert row['family'] == c['family'] and row['condition'] == c['condition']
        assert row['input_exclusion_retained'] == c['rejected'] and row['actually_viewed'] and row['observation']
        assert row['generation_outputs_reviewed'] == 0 and not row['automatic_completion_quality_qualified']
    assert len(v['page_sha256']) == 9
    for name, digest in v['page_sha256'].items(): assert sha(OUT/name) == digest == r['artifact_sha256'][name]
    assert v['independent_audit_sha256'] == sha(OUT/'independent_audit.json')
    assert v['secondary_analysis_sha256'] == sha(OUT/'secondary_score_order_analysis.json')
    assert v['results_sha256'] == s['results_sha256'] == sha(OUT/'results.json')
    assert s['post_hoc_saved_score_analysis'] and s['not_a_preregistered_quality_gate'] and not s['threshold_search']
    cases = {c['id']: c for c in p['cases']}
    expected_ids = set()
    for c in p['cases']:
        if c.get('masks') and binary(ROOT/c['masks']['core']).any() and binary(ROOT/c['masks']['protected']).any(): expected_ids.add(c['id'])
    assert {w['id'] for w in s['witnesses']} == expected_ids and len(s['witnesses']) == len(expected_ids)
    for w in s['witnesses']:
        c = cases[w['id']]; core = binary(ROOT/c['masks']['core']); protected = binary(ROOT/c['masks']['protected'])
        x, y = w['core_xy']; px, py = w['protected_xy']; assert core[y, x] and protected[py, px]
        with np.load(OUT/'stages'/(w['id']+'.npz'), allow_pickle=False) as saved: score = saved['canvas_probability']
        assert float(score[y, x]) == w['core_score'] == float(score[core].min())
        assert float(score[py, px]) == w['protected_score'] == float(score[protected].max())
        assert w['strict_order_inversion'] == (w['protected_score'] > w['core_score'])
    assert s['witness_count'] == len(expected_ids) and s['strict_inversions'] == sum(w['strict_order_inversion'] for w in s['witnesses'])
    for condition, summary in s['condition_summary'].items():
        rows = [row for row in r['rows'] if row['condition'] == condition]
        covering = [row for row in rows if row['regions'].get('assisted_core', {}).get('pixels', 0) > 0]
        expected = {'requests': len(rows), 'empty_proposals_all_cases': len([row for row in rows if row['proposal_pixels'] == 0]),
            'eligible_nonempty_core_cases': len(covering), 'empty_proposals_covering_cases': len([row for row in covering if row['proposal_pixels'] == 0]),
            'protected_pixels_proposed': sum(row['proposal_region_pixels'].get('protected_appearance', 0) for row in rows),
            'cases_with_protected_pixels_proposed': len([row for row in rows if row['proposal_region_pixels'].get('protected_appearance', 0) > 0]),
            'mean_core_fraction_above_fixed_threshold': float(np.mean([row['regions']['assisted_core']['fraction_ge_0_5'] for row in covering]))}
        assert summary == expected
    write(OUT/'review_independent_audit.json', {'complete': True, 'visual_review_sha256': sha(OUT/'visual_review.json'),
        'secondary_analysis_sha256': sha(OUT/'secondary_score_order_analysis.json'), 'page_hashes_verified': 9,
        'observations_bound': 36, 'input_exclusions_preserved': 4, 'score_witnesses_verified': len(expected_ids),
        'secondary_statistics_verified': True, 'subjective_visual_judgments_proved_by_code': False,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'app_adoption': False,
        'native_or_final_used': False, 'goal_complete': False, 'checker_sha256': sha(Path(__file__))})
    print({'complete': True, 'observations': 36, 'secondary_statistics_verified': True})


if __name__ == '__main__': main()
