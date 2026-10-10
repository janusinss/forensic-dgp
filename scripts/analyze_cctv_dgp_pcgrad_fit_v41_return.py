"""Saved-array direction/exposure analysis and exact TRAIN comparison pages."""
from pathlib import Path
import hashlib
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import nnls
from cctv_dgp_spatial_fit_v40_contract import read, write, sha, PROFILES

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41'
RETURN = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return'
OLDER = ROOT/'outputs/cctv_dgp_spatial_fit_v40_return'
OUT = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis'
ANGLE_ROUNDOFF = 1e-8  # Descriptive direction counts only; never an image/gate tolerance.


def pixels(path):
    with Image.open(path) as im:
        assert im.mode == 'RGB' and im.size == (256, 256)
        return np.asarray(im).copy()


def main():
    start = time.monotonic(); assert not OUT.exists()
    audit_path = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json'
    audit = read(audit_path); p = read(BUNDLE/'protocol.json')
    assert audit['complete'] and audit['failure_retained'] and not audit['necessary_capacity_pass']
    assert audit['per_update_learning_evidence']['all_logged_updates_verified'] == 50
    assert [s['update'] for s in audit['snapshots']] == [0, 50]
    assert audit['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    assert audit['checker_sha256'] == sha(ROOT/'scripts/audit_cctv_dgp_pcgrad_fit_v41_return_r1.py')
    previous_audit = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json')
    assert previous_audit['complete'] and previous_audit['failure_retained']
    OUT.mkdir(); bindings = {}

    def bind(path): bindings[path.relative_to(ROOT).as_posix()] = sha(path)

    for path in [Path(__file__), ROOT/'scripts/verify_cctv_dgp_pcgrad_fit_v41_analysis.py', audit_path,
                 BUNDLE/'protocol.json', BUNDLE/'schedule.json', RETURN/'outputs/failure.json',
                 RETURN/'outputs/capacity_update50.json', ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json',
                 ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json']:
        bind(path)
    rows = []
    for update in range(1, 51):
        path = RETURN/'outputs/steps'/('step'+str(update).zfill(4)+'.npz'); bind(path); bind(path.with_suffix('.json'))
        record = read(path.with_suffix('.json'))
        with np.load(path, allow_pickle=False) as arrays:
            g = arrays['components'].astype(np.float64)
            d = arrays['after'].astype(np.float64)-arrays['before'].astype(np.float64)
            projected = -arrays['combined'].astype(np.float64)
            decay_free = d+.0003*.01*arrays['before'].astype(np.float64)
        norms = np.linalg.norm(g, axis=1); dnorm = float(np.linalg.norm(d)); pnorm = float(np.linalg.norm(projected))
        assert dnorm > 0 and pnorm > 0
        cosine = lambda value: np.divide(g@value, norms*np.linalg.norm(value), out=np.zeros(7), where=norms > 0)
        q = np.divide(g, norms[:, None], out=np.zeros_like(g), where=norms[:, None] > 0)
        # One fixed convex arithmetic proposal on SAVED arrays. No network is instantiated or assigned.
        dual, _ = nnls(q.T, d, maxiter=21)
        hypothetical = d-q.T@dual
        assert max(q@hypothetical) <= 1e-10*max(dnorm, 1e-12)
        rows.append({'update': update, 'case_ids': record['case_ids'], 'component_means': record['component_means'],
                     'different_references_each_batch_not_a_loss_curve': True,
                     'component_gradient_norms': norms.tolist(), 'conflict_projections': record['projection']['conflict_projections'],
                     'PCGrad_negative_direction_component_derivatives': (g@projected).tolist(),
                     'actual_parameter_step_component_derivatives': (g@d).tolist(),
                     'decay_free_step_component_derivatives': (g@decay_free).tolist(),
                     'PCGrad_direction_cosines': cosine(projected).tolist(), 'actual_step_cosines': cosine(d).tolist(),
                     'decay_free_step_cosines': cosine(decay_free).tolist(), 'actual_step_norm': dnorm,
                     'PCGrad_direction_norm': pnorm,
                     'one_fixed_saved_array_cone_proposal': {'dual': dual.tolist(), 'maximum_iterations': 21,
                         'component_derivatives': (g@hypothetical).tolist(),
                         'normalized_constraint_dots': (q@hypothetical).tolist(),
                         'norm': float(np.linalg.norm(hypothetical)), 'retained_norm_ratio': float(np.linalg.norm(hypothetical)/dnorm),
                         'finite_model_loss_or_quality_not_tested': True}})
    summary = []
    for k, term in enumerate(p['terms']):
        active = [r for r in rows if r['component_gradient_norms'][k] > 0]
        summary.append({'term': term, 'active_updates': len(active),
                        'PCGrad_direction_ascent_updates': [r['update'] for r in active if r['PCGrad_direction_cosines'][k] > ANGLE_ROUNDOFF],
                        'actual_step_ascent_updates': [r['update'] for r in active if r['actual_step_cosines'][k] > ANGLE_ROUNDOFF],
                        'decay_free_step_ascent_updates': [r['update'] for r in active if r['decay_free_step_cosines'][k] > ANGLE_ROUNDOFF],
                        'actual_ascent_despite_projected_nonincrease': [r['update'] for r in active if
                            r['actual_step_cosines'][k] > ANGLE_ROUNDOFF and r['PCGrad_direction_cosines'][k] <= ANGLE_ROUNDOFF]})
    metrics = [read(RETURN/('outputs/update'+str(n)+'/metrics.json')) for n in [0, 50]]
    for n in [0, 50]: bind(RETURN/('outputs/update'+str(n)+'/metrics.json'))
    before, after = [{row['id']: row for row in receipt['rows']} for receipt in metrics]
    cases = {c['id']: c for c in p['cases']}; batches = read(BUNDLE/'schedule.json')['batches']
    exposed = {p['cases'][i]['id'] for batch in batches[:50] for i in batch}
    cohorts_path = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm/protocol.json'; bind(cohorts_path)
    cohorts = read(cohorts_path)['cohorts']
    assert cohorts['not_yet_optimized'] == p['preview_case_ids'] and not exposed.intersection(p['preview_case_ids'])
    assert set(cohorts['optimized']).issubset(exposed) and len(exposed) == 250
    buckets = {}
    for name, selected in [('all_TRAIN', set(cases)), ('exposed_by50_TRAIN', exposed),
                           ('not_yet_exposed_TRAIN', set(cases)-exposed),
                           ('fixed50_TRAIN', set(p['preview_case_ids'])), ('optimized50_TRAIN', set(cohorts['optimized']))]:
        ids = [cid for cid in before if cid in selected and cases[cid]['profile'] != 'clear']
        base, current = [float(np.mean([table[cid]['metrics']['landmark_high_frequency_MSE'] for cid in ids])) for table in [before, after]]
        buckets[name] = {'degraded_cases': len(ids), 'baseline_feature_MSE': base, 'V41_feature_MSE': current,
                         'relative_feature_gain': 1-current/base, 'TRAIN_not_independent_evaluation': True}
    baseline_exact = 0
    for c in p['cases']:
        assert time.monotonic()-start < 300
        old, current = [folder/('outputs/update0/'+c['id']+'.png') for folder in [OLDER, RETURN]]
        assert np.array_equal(pixels(old), pixels(current)); baseline_exact += 1
    pages = []; font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    columns = ['Input256', 'Retained DGP / initial', 'V40 stopped50', 'V41 stopped50', 'Paired TRAIN target']
    for cohort, ids in cohorts.items():
        for number, begin in enumerate(range(0, 50, 5), 1):
            chosen = [cases[cid] for cid in ids[begin:begin+5]]
            assert len({c['source_person_or_reference'] for c in chosen}) == 1 and [c['profile'] for c in chosen] == PROFILES
            canvas = Image.new('RGB', (1336, 1516), '#f1f2f4'); draw = ImageDraw.Draw(canvas); cells = []
            draw.text((12, 5), 'Paired TRAIN | '+cohort+' | '+chosen[0]['source_person_or_reference'], font=font, fill='#111111')
            for k, label in enumerate(columns): draw.text((12+264*k, 30), label, font=font, fill='#111111')
            for row, c in enumerate(chosen):
                cid = c['id']; y = 64+288*row; draw.text((12, y), cid, font=font, fill='#111111')
                paths = [BUNDLE/c['input'], RETURN/('outputs/update0/'+cid+'.png'), OLDER/('outputs/update50/'+cid+'.png'),
                         RETURN/('outputs/update50/'+cid+'.png'), BUNDLE/c['target']]
                for k, path in enumerate(paths):
                    bind(path); image = pixels(path); xy = [12+264*k, y+24]; canvas.paste(Image.fromarray(image), tuple(xy))
                    cells.append({'id': cid, 'column': k, 'xy': xy, 'source': path.relative_to(ROOT).as_posix(),
                                  'RGB_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
            destination = OUT/(cohort+'_sheet_'+str(number).zfill(2)+'.png'); canvas.save(destination)
            pages.append({'path': destination.name, 'sha256': sha(destination), 'cohort': cohort,
                          'reference': chosen[0]['source_person_or_reference'], 'source': chosen[0]['source'],
                          'ids': [c['id'] for c in chosen], 'cells': cells})
    assert baseline_exact == 3905 and len(rows) == 50 and len(pages) == 20
    assert 'torch' not in sys.modules and time.monotonic()-start < 300
    write(OUT/'analysis.json', {'complete': True, 'protocol_sha256': sha(BUNDLE/'protocol.json'), 'source_sha256': bindings,
          'step_rows': rows, 'direction_summary': summary, 'descriptive_cosine_roundoff': ANGLE_ROUNDOFF,
          'original_gate_unchanged': audit['gates'][0], 'exposure_buckets': buckets, 'initial_PNG_case_parity_exact': baseline_exact,
          'cohorts': cohorts, 'columns': columns, 'pages': pages, 'unscaled_cell_size': [256, 256],
          'all_first_order_observations_not_unique_cause_or_finite_guarantee': True,
          'no_proposal_loaded_into_model_or_training_checkpoint': True, 'different_batches_not_a_loss_curve': True,
          'visual_review_pending': True, 'native_or_reserved_used': False, 'source_names_not_ethnicity': True,
          'new_neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0,
          'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 300})
    print({'complete': True, 'saved_updates': 50, 'pages': 20, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
