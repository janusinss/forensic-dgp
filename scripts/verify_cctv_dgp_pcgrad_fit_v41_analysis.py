"""Independent dot-product/KKT and exact source-cell audit; no neural imports."""
from pathlib import Path
import hashlib
import sys
import time
import numpy as np
from PIL import Image
from cctv_dgp_spatial_fit_v40_contract import read, write, sha, PROFILES

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41'
RETURN = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return'
OUT = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis'


def pixels(path):
    with Image.open(path) as im: return np.array(im.convert('RGB'))


def main():
    start = time.monotonic(); a = read(OUT/'analysis.json'); p = read(BUNDLE/'protocol.json')
    audit = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json')
    assert a['complete'] and audit['complete'] and audit['failure_retained'] and not audit['necessary_capacity_pass']
    assert a['original_gate_unchanged'] == audit['gates'][0] and not a['original_gate_unchanged']['pass']
    assert a['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    for name, digest in a['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
        assert time.monotonic()-start < 300
    rows = a['step_rows']; assert len(rows) == 50 and a['descriptive_cosine_roundoff'] == 1e-8
    for update, row in enumerate(rows, 1):
        assert row['update'] == update
        path = RETURN/'outputs/steps'/('step'+str(update).zfill(4)+'.npz')
        record = read(path.with_suffix('.json'))
        with np.load(path, allow_pickle=False) as arrays:
            gradients = arrays['components'].astype(np.float64)
            step = arrays['after'].astype(np.float64)-arrays['before'].astype(np.float64)
            direction = -arrays['combined'].astype(np.float64)
            no_decay = step+.0003*.01*arrays['before'].astype(np.float64)
        assert row['case_ids'] == record['case_ids'] and row['component_means'] == record['component_means']
        lengths = np.sqrt(np.sum(gradients*gradients, axis=1, dtype=np.float64))
        np.testing.assert_allclose(lengths, row['component_gradient_norms'], rtol=2e-12, atol=1e-15)
        norms = [float(np.sqrt(np.sum(vector*vector))) for vector in [step, direction]]
        np.testing.assert_allclose(norms, [row['actual_step_norm'], row['PCGrad_direction_norm']], rtol=2e-12, atol=1e-15)
        for vector, key, cosine_key in [
            (step, 'actual_parameter_step_component_derivatives', 'actual_step_cosines'),
            (direction, 'PCGrad_negative_direction_component_derivatives', 'PCGrad_direction_cosines'),
            (no_decay, 'decay_free_step_component_derivatives', 'decay_free_step_cosines')]:
            dots = np.sum(gradients*vector[None, :], axis=1, dtype=np.float64)
            np.testing.assert_allclose(dots, row[key], rtol=2e-12, atol=1e-15)
            norm = np.sqrt(np.sum(vector*vector))
            cosines = np.divide(dots, lengths*norm, out=np.zeros(7), where=lengths > 0)
            np.testing.assert_allclose(cosines, row[cosine_key], rtol=2e-12, atol=1e-12)
        proposal = row['one_fixed_saved_array_cone_proposal']; dual = np.asarray(proposal['dual'], np.float64)
        assert dual.shape == (7,) and np.isfinite(dual).all() and dual.min() >= 0 and proposal['maximum_iterations'] == 21
        q = np.divide(gradients, lengths[:, None], out=np.zeros_like(gradients), where=lengths[:, None] > 0)
        correction = np.sum(q*dual[:, None], axis=0, dtype=np.float64)
        projected = step-correction
        primal = np.sum(q*projected[None, :], axis=1, dtype=np.float64)
        allowance = 1e-10*max(norms[0], 1e-12)
        assert max(primal) <= allowance
        # KKT certifies the convex projection independently of the producer's NNLS solver.
        assert max(np.abs(dual*primal)) <= allowance*max(float(dual.max()), 1e-12)
        projected_norm = float(np.sqrt(np.sum(projected*projected)))
        assert projected_norm <= norms[0]*(1+1e-12)
        np.testing.assert_allclose(primal, proposal['normalized_constraint_dots'], rtol=2e-12, atol=1e-15)
        np.testing.assert_allclose(np.sum(gradients*projected[None, :], axis=1), proposal['component_derivatives'], rtol=2e-12, atol=1e-15)
        np.testing.assert_allclose([projected_norm, projected_norm/norms[0]], [proposal['norm'], proposal['retained_norm_ratio']], rtol=2e-12, atol=1e-15)
        assert proposal['finite_model_loss_or_quality_not_tested']
    for k, summary in enumerate(a['direction_summary']):
        active = [r for r in rows if r['component_gradient_norms'][k] > 0]
        assert summary['term'] == p['terms'][k] and summary['active_updates'] == len(active)
        for key, cosine_key in [('PCGrad_direction_ascent_updates', 'PCGrad_direction_cosines'),
                                ('actual_step_ascent_updates', 'actual_step_cosines'),
                                ('decay_free_step_ascent_updates', 'decay_free_step_cosines')]:
            assert summary[key] == [r['update'] for r in active if r[cosine_key][k] > 1e-8]
        assert summary['actual_ascent_despite_projected_nonincrease'] == [r['update'] for r in active if
            r['actual_step_cosines'][k] > 1e-8 and r['PCGrad_direction_cosines'][k] <= 1e-8]
    expected_cohorts = read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm/protocol.json')['cohorts']
    assert a['cohorts'] == expected_cohorts and expected_cohorts['not_yet_optimized'] == p['preview_case_ids']
    cases = {c['id']: c for c in p['cases']}; batches = read(BUNDLE/'schedule.json')['batches']
    exposed = {p['cases'][i]['id'] for batch in batches[:50] for i in batch}
    before, after = [{row['id']: row for row in read(RETURN/('outputs/update'+str(n)+'/metrics.json'))['rows']} for n in [0, 50]]
    selections = {'all_TRAIN': set(cases), 'exposed_by50_TRAIN': exposed, 'not_yet_exposed_TRAIN': set(cases)-exposed,
                  'fixed50_TRAIN': set(p['preview_case_ids']), 'optimized50_TRAIN': set(expected_cohorts['optimized'])}
    for name, selection in selections.items():
        ids = [cid for cid in before if cid in selection and cases[cid]['profile'] != 'clear']
        baseline, candidate = [float(np.mean([table[cid]['metrics']['landmark_high_frequency_MSE'] for cid in ids])) for table in [before, after]]
        saved = a['exposure_buckets'][name]
        assert saved['degraded_cases'] == len(ids) and saved['TRAIN_not_independent_evaluation']
        assert baseline == saved['baseline_feature_MSE'] and candidate == saved['V41_feature_MSE']
        assert saved['relative_feature_gain'] == 1-candidate/baseline
    assert len(a['pages']) == 20 and a['unscaled_cell_size'] == [256, 256]
    ids_seen = {name: [] for name in expected_cohorts}; cell_count = 0
    for page in a['pages']:
        assert sha(OUT/page['path']) == page['sha256']
        canvas = pixels(OUT/page['path']); assert canvas.shape == (1516, 1336, 3)
        assert [cases[cid]['profile'] for cid in page['ids']] == PROFILES and len(page['cells']) == 25
        assert {cases[cid]['source_person_or_reference'] for cid in page['ids']} == {page['reference']}
        for cell in page['cells']:
            cid = cell['id']; c = cases[cid]; column = cell['column']; index = page['ids'].index(cid)
            paths = [BUNDLE/c['input'], RETURN/('outputs/update0/'+cid+'.png'),
                     ROOT/('outputs/cctv_dgp_spatial_fit_v40_return/outputs/update50/'+cid+'.png'),
                     RETURN/('outputs/update50/'+cid+'.png'), BUNDLE/c['target']]
            assert cell['source'] == paths[column].relative_to(ROOT).as_posix() and cell['xy'] == [12+264*column, 88+288*index]
            original = pixels(paths[column]); assert original.shape == (256, 256, 3)
            assert cell['RGB_sha256'] == hashlib.sha256(original.tobytes()).hexdigest()
            x, y = cell['xy']; np.testing.assert_array_equal(canvas[y:y+256, x:x+256], original)
            cell_count += 1
        ids_seen[page['cohort']].extend(page['ids'])
    assert ids_seen == expected_cohorts and cell_count == 500
    for c in p['cases']:
        np.testing.assert_array_equal(pixels(RETURN/('outputs/update0/'+c['id']+'.png')),
            pixels(ROOT/('outputs/cctv_dgp_spatial_fit_v40_return/outputs/update0/'+c['id']+'.png')))
        assert time.monotonic()-start < 300
    assert a['initial_PNG_case_parity_exact'] == 3905
    assert a['no_proposal_loaded_into_model_or_training_checkpoint'] and a['native_or_reserved_used'] is False
    assert a['new_neural_calls'] == a['gradient_calls'] == a['optimizer_updates'] == a['VM_calls'] == 0
    assert not a['app_promotion'] and not a['goal_complete'] and 'torch' not in sys.modules
    write(OUT/'independent_analysis_audit.json', {'complete': True, 'analysis_sha256': sha(OUT/'analysis.json'),
          'checker_sha256': sha(Path(__file__)), 'saved_updates': 50, 'all350_component_direction_sets_checked': True,
          'all50_convex_projection_KKT_certificates_checked': True, 'exact_unscaled_sheet_cells': 500,
          'both_metadata_selected_TRAIN_cohorts_checked': True, 'initial_PNG_case_parity_exact': 3905,
          'original_failed_structure_and_preservation_gates_unchanged': True, 'finite_model_proposals_not_tested': True,
          'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
          'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 300})
    print({'complete': True, 'saved_updates': 50, 'exact_cells': 500, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
