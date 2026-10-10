"""Analyze audited saved endpoint gradients and prepare exact TRAIN comparisons."""
from pathlib import Path
import hashlib
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cctv_dgp_spatial_fit_v40_contract import read, write, sha
from cctv_dgp_pcgrad_v41 import combine, SEED

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm'
RETURN = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return'
OUT = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis'


def main():
    start = time.monotonic(); assert not OUT.exists()
    p = read(BUNDLE/'protocol.json'); a = read(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_independent_audit.json')
    assert a['complete'] and a['diagnostic_complete'] and a['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    assert a['checker_sha256'] == p['local_basis_sha256']['scripts/audit_cctv_dgp_v40_learning_signal_v1_return.py']
    summary = read(RETURN/'outputs/gradient_summary.json'); OUT.mkdir(); bindings = {}
    def bind(path): bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in [Path(__file__), ROOT/'scripts/cctv_dgp_pcgrad_v41.py', BUNDLE/'protocol.json',
                 ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_independent_audit.json', RETURN/'outputs/results.json',
                 RETURN/'outputs/gradient_summary.json', ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return_import.json']:
        bind(path)
    reports = {}; batches = []
    for state, cohorts in summary['states'].items():
        reports[state] = {}
        for cohort, row in cohorts.items():
            path = RETURN/('outputs/'+state+'/'+cohort+'_gradient_components.npy'); bind(path)
            components = np.load(path, allow_pickle=False)
            _, merged = combine(components, 1)
            initial = summary['states']['initial'][cohort]['values'][0]
            norm = row['component_norms']
            reports[state][cohort] = {
                'seven_weighted_mean_losses': row['values'], 'component_norms': norm,
                'identity_to_landmark_gradient_norm_ratio': norm[6]/norm[0],
                'landmark_loss_gain_fraction_from_initial': 1-row['values'][0]/initial,
                'original_direction_derivatives': row['negative_total_direction_component_derivatives'],
                'fixed_PCGrad_arithmetic': merged,
            }
            for number in range(10):
                path = RETURN/('outputs/'+state+'/gradients/'+cohort+'_batch'+str(number)+'.npy'); bind(path)
                _, math = combine(np.load(path, allow_pickle=False), number+1)
                batches.append({'state': state, 'cohort': cohort, 'batch': number, 'diagnostic_scalars_include_one_tenth_cohort_factor': True, **math})
    assert len(batches) == 40 and all(max(r['projected_direction_derivatives']) <= 1e-12 for r in batches)
    lookup = {c['id']: c for c in p['cases']}; ids = p['cohorts']['optimized']; pages = []
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for number, begin in enumerate(range(0, 50, 5), 1):
        group = [lookup[cid] for cid in ids[begin:begin+5]]
        assert len({c['source_person_or_reference'] for c in group}) == 1
        page = Image.new('RGB', (1072, 1516), '#f1f2f4'); draw = ImageDraw.Draw(page); cells = []
        draw.text((12, 5), 'Previously optimized TRAIN | '+group[0]['source_person_or_reference']+' | '+group[0]['source'], font=font, fill='#111111')
        for slot, label in enumerate(['Input', 'Original DGP / initial', 'V40 stopped50', 'Aligned photographic target']):
            draw.text((12+264*slot, 30), label, font=font, fill='#111111')
        for row, c in enumerate(group):
            y = 64+288*row; draw.text((12, y), c['id'], font=font, fill='#111111')
            paths = [BUNDLE/c['input'], RETURN/('outputs/initial/'+c['id']+'.png'),
                     RETURN/('outputs/stopped50/'+c['id']+'.png'), BUNDLE/c['target']]
            for slot, path in enumerate(paths):
                bind(path)
                with Image.open(path) as im:
                    assert im.mode == 'RGB' and im.size == (256, 256); pixels = np.asarray(im).copy()
                xy = [12+264*slot, y+24]; page.paste(Image.fromarray(pixels), tuple(xy))
                cells.append({'id': c['id'], 'column': slot, 'xy': xy, 'source': path.relative_to(ROOT).as_posix(), 'RGB_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        dest = OUT/('sheet_'+str(number).zfill(2)+'.png'); page.save(dest)
        pages.append({'path': dest.name, 'sha256': sha(dest), 'reference': group[0]['source_person_or_reference'], 'source': group[0]['source'], 'ids': [c['id'] for c in group], 'cells': cells})
    assert time.monotonic()-start < 180
    write(OUT/'analysis.json', {'complete': True, 'protocol_sha256': sha(BUNDLE/'protocol.json'), 'sources_sha256': bindings,
        'reports': reports, 'fixed_PCGrad_seed': SEED, 'batches': batches, 'pages': pages,
        'all40_saved_batches_first_order_nonincrease': True, 'finite_learning_or_preservation_not_demonstrated': True,
        'cohorts_are_exposed_TRAIN_not_independent_evaluation': True, 'visual_review_pending': True,
        'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
        'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 180})
    print({'complete': True, 'saved_batches': 40, 'optimized_TRAIN_pages': 10, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
