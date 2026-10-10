"""Fixed TRAIN visual pages and exposure diagnostics from audited saved arrays."""
from pathlib import Path
import time
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cctv_dgp_spatial_fit_v40_contract import read, write, sha, PROFILES, validate_schedule

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40'
RETURNED = ROOT/'outputs/cctv_dgp_spatial_fit_v40_return'
OUT = ROOT/'outputs/cctv_dgp_spatial_fit_v40_failure_review'


def pixels(path):
    with Image.open(path) as im:
        assert im.size == (256, 256) and im.mode == 'RGB'
        return np.array(im)


def main():
    start = time.monotonic(); assert not OUT.exists()
    audit_path = ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json'
    a = read(audit_path); p = read(BUNDLE/'protocol.json')
    assert a['complete'] and a['failure_retained'] and not a['necessary_capacity_pass'] and not a['training_finished800']
    assert [s['update'] for s in a['snapshots']] == [0, 50] and [r['cases'] for r in a['CPU_replays']] == [50, 50]
    assert a['checker_sha256'] == sha(ROOT/'scripts/audit_cctv_dgp_spatial_fit_v40_return.py')
    assert a['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    initial, stopped = [read(RETURNED/('outputs/update'+str(n)+'/metrics.json')) for n in [0, 50]]
    before, after = [{row['id']: row for row in v['rows']} for v in [initial, stopped]]
    cases = {c['id']: c for c in p['cases']}; ids = p['preview_case_ids']
    assert len(ids) == len(set(ids)) == 50
    schedule = read(BUNDLE/'schedule.json')['batches']; validate_schedule(p['cases'], schedule)
    exposed = {p['cases'][index]['id'] for batch in schedule[:50] for index in batch}
    assert len(exposed) == 250
    OUT.mkdir()
    plan = {'format': 'V40 stopped50 fixed photographic TRAIN review', 'case_ids': ids,
            'selection': 'Same prospective50 preview IDs fixed in V39/V40 before new learning; no output selection',
            'columns': ['Input256', 'Retained DGP / update0', 'V40 stopped50', 'Mean-shift-only diagnostic', 'Paired TRAIN target'],
            'regions': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
            'criteria': ['Preserve all visible facial structure; accept softness.',
                         'Assess feature definition separately from brightness or texture changes.',
                         'Preserve clear glasses, ordinary hair, gaze and expression.',
                         'No visual impression waives the unchanged failed capacity gate.',
                         'Photographic paired TRAIN is separate from native CCTV and independent final evidence.'],
            'protocol_sha256': sha(BUNDLE/'protocol.json'), 'return_audit_sha256': sha(audit_path),
            'source_sha256': sha(Path(__file__)),
            'checker_sha256': sha(ROOT/'scripts/audit_cctv_dgp_spatial_fit_v40_failure_review.py'),
            'cell_size': [256, 256], 'image_processing_or_resize': False, 'new_neural_calls': 0,
            'local_gradients': 0, 'optimizer_updates': 0, 'native_or_reserved_used': False,
            'independent_final_review': False, 'app_promotion': False, 'goal_complete': False}
    write(OUT/'plan.json', plan)
    buckets = {}
    for label, selected in [('all_TRAIN', set(cases)), ('exposed_by50_TRAIN', exposed),
                            ('not_yet_exposed_TRAIN', set(cases)-exposed), ('fixed50_TRAIN', set(ids))]:
        chosen = [cid for cid in before if cid in selected and cases[cid]['profile'] != 'clear']
        baseline, current = [float(np.mean([table[cid]['metrics']['landmark_high_frequency_MSE'] for cid in chosen])) for table in [before, after]]
        buckets[label] = {'degraded_cases': len(chosen), 'baseline_feature_MSE': baseline,
                          'stopped50_feature_MSE': current, 'relative_feature_gain': 1-current/baseline,
                          'all_are_TRAIN_not_held_out_evaluation': True}
    assert abs(buckets['all_TRAIN']['relative_feature_gain']-a['gates'][0]['relative_feature_gain']) < 1e-12
    bindings = {}; rows = []; pages = []
    def bind(path): bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in [audit_path, BUNDLE/'protocol.json', BUNDLE/'schedule.json', RETURNED/'outputs/failure.json',
                 RETURNED/'outputs/capacity_update50.json', RETURNED/'outputs/update0/metrics.json', RETURNED/'outputs/update50/metrics.json']:
        bind(path)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for page_number, begin in enumerate(range(0, 50, 5), 1):
        page_ids = ids[begin:begin+5]; selected = [cases[cid] for cid in page_ids]
        assert len({c['source_person_or_reference'] for c in selected}) == 1 and [c['profile'] for c in selected] == PROFILES
        page = Image.new('RGB', (1336, 1516), '#f1f2f4'); draw = ImageDraw.Draw(page); cells = []
        draw.text((12, 5), 'Paired TRAIN | stopped50 | '+selected[0]['source_person_or_reference']+' | '+selected[0]['source'], font=font, fill='#111111')
        for k, label in enumerate(plan['columns']): draw.text((12+k*264, 30), label, font=font, fill='#111111')
        for row_index, c in enumerate(selected):
            cid = c['id']; y = 64+row_index*288
            draw.text((12, y), c['profile']+' | '+cid+(' | optimized by50' if cid in exposed else ' | not-yet-optimized TRAIN'), font=font, fill='#111111')
            paths = [BUNDLE/c['input'], RETURNED/('outputs/update0/'+cid+'.png'), RETURNED/('outputs/update50/'+cid+'.png'),
                     RETURNED/('outputs/update50/'+cid+'_mean_only.png'), BUNDLE/c['target']]
            for k, path in enumerate(paths):
                array = pixels(path); bind(path); xy = [12+k*264, y+24]; page.paste(Image.fromarray(array), tuple(xy))
                cells.append({'id': cid, 'column': k, 'xy': xy, 'source': path.relative_to(ROOT).as_posix(),
                              'RGB_sha256': hashlib.sha256(array.tobytes()).hexdigest()})
            support_path = BUNDLE/c['observed']; bind(support_path)
            with Image.open(support_path) as im: assert im.mode == 'L'; support = np.array(im) > 0
            raw_paths = [RETURNED/('outputs/update'+str(n)+'/'+cid+'.npy') for n in [0, 50]]
            for path in raw_paths: bind(path)
            raw0, raw50 = [np.load(path, allow_pickle=False) for path in raw_paths]
            difference = (raw50-raw0)[support].astype(np.float64)
            byte_difference = pixels(paths[2]).astype(np.int16)-pixels(paths[1]).astype(np.int16)
            rows.append({'id': cid, 'source': c['source'], 'profile': c['profile'], 'optimized_by50': cid in exposed,
                         'raw_correction_RMS': float(np.sqrt(np.square(difference).mean())),
                         'mean_postclip_RGB_shift': difference.mean(0).tolist(),
                         'changed_PNG_pixels': int(np.any(byte_difference != 0, -1).sum()),
                         'maximum_PNG_byte_change': int(np.abs(byte_difference).max()),
                         'baseline_metrics': before[cid]['metrics'], 'stopped50_metrics': after[cid]['metrics']})
        path = OUT/('sheet_'+str(page_number).zfill(2)+'.png'); page.save(path)
        pages.append({'path': path.relative_to(OUT).as_posix(), 'reference': selected[0]['source_person_or_reference'],
                      'source': selected[0]['source'], 'ids': page_ids, 'cells': cells, 'sha256': sha(path)})
    assert len(rows) == 50 and len(pages) == 10 and time.monotonic()-start < 180
    write(OUT/'preparation.json', {'complete': True, 'plan_sha256': sha(OUT/'plan.json'), 'rows': rows, 'pages': pages,
          'source_bindings_sha256': bindings, 'exposure_buckets': buckets, 'unique_cases_optimized_by50': len(exposed),
          'fixed50_cases_optimized_by50': len(set(ids)&exposed), 'early_gate': a['gates'][0],
          'visual_review_pending_at_preparation': True, 'new_neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0,
          'native_or_reserved_used': False, 'app_promotion': False, 'independent_final_review': False,
          'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 180})
    print({'complete': True, 'pages': 10, 'preview_cases': 50, 'fixed50_optimized_by50': len(set(ids)&exposed), 'buckets': buckets}, flush=True)


if __name__ == '__main__': main()
