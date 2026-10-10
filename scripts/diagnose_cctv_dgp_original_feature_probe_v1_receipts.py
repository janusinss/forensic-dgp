"""Diagnose derived receipt sensitivity without neural calls or gate changes."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm'
RETURN = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm_return'
OUT = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1'
sys.path.insert(0, str(BUNDLE))
from frozen_raw_metrics import mean_only, review_groups
from frozen_capacity_contract import capacity


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def mse(actual, target, mask):
    reference = target.astype(np.float32)/np.float32(255)
    return float(np.square((actual-reference)[mask]).astype(np.float64).mean())


def differences(a, b, prefix=''):
    if isinstance(b, dict):
        assert a.keys() == b.keys()
        return [d for k in b for d in differences(a[k], b[k], prefix+'/'+k)]
    if isinstance(b, list):
        assert len(a) == len(b)
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, prefix+'/'+str(i))]
    if isinstance(b, (float, np.floating)):
        delta = abs(a-b)
        return [{'path': prefix, 'saved': a, 'fresh': b, 'absolute_difference': delta}] if delta else []
    assert a == b, (prefix, a, b)
    return []


def main():
    start = time.monotonic(); p = read(BUNDLE/'protocol.json')
    result = read(RETURN/'outputs/results.json'); folder = RETURN/'outputs'
    labels = ['baseline']+[r['variant'] for r in result['trial_summaries']]
    inputs = {}
    for c in p['cases']:
        inputs[c['id']] = (pixels(BUNDLE/c['input']), pixels(BUNDLE/c['target']), pixels(BUNDLE/c['observed'], 'L') > 0,
            np.load(folder/'baseline'/(c['id']+'.npy'), allow_pickle=False))
    groups = {}; changes = []; stored_decisions = {}; fresh_decisions = {}; derived = []
    for label in labels:
        saved = read(folder/label/'metrics.json'); rows = copy.deepcopy(saved['rows'])
        for row in rows:
            assert time.monotonic()-start <= 180
            cid = row['id']; camera, target, mask, baseline = inputs[cid]
            raw = np.load(folder/label/(cid+'.npy'), allow_pickle=False)
            png = pixels(folder/label/(cid+'.png')); mraw, mpng, _ = mean_only(raw, baseline, camera, mask)
            assert np.array_equal(mpng, pixels(folder/label/(cid+'_mean_only.png')))
            for stage, values in [('raw', (raw, mraw)), ('png', (png.astype(np.float32)/np.float32(255), mpng.astype(np.float32)/np.float32(255)))]:
                for metric, actual in zip(['MSE', 'constant_mean_shift_only_MSE'], values):
                    before = row[stage][metric]; fresh = mse(actual, target, mask)
                    if before != fresh: changes.append({'variant': label, 'id': cid, 'stage': stage, 'metric': metric,
                        'saved': before, 'fresh': fresh, 'absolute_difference': abs(before-fresh)})
                    row[stage][metric] = fresh
        stored_groups = {co['name']: {s: review_groups([r for r in saved['rows'] if r['id'] in co['case_ids']], s) for s in ['raw', 'png']} for co in p['cohorts']}
        assert stored_groups == saved['groups']
        groups[label] = {co['name']: {s: review_groups([r for r in rows if r['id'] in co['case_ids']], s) for s in ['raw', 'png']} for co in p['cohorts']}
        if label == 'baseline': continue
        summary = next(r for r in result['trial_summaries'] if r['variant'] == label)
        comparisons = {co['name']: {s: capacity(groups['baseline'][co['name']][s], groups[label][co['name']][s], .01) for s in ['raw', 'png']} for co in p['cohorts']}
        exact_from_rows = {co['name']: {s: capacity(read(folder/'baseline/metrics.json')['groups'][co['name']][s], stored_groups[co['name']][s], .01) for s in ['raw', 'png']} for co in p['cohorts']}
        assert exact_from_rows == summary['comparisons']
        for co in p['cohorts']:
            name = co['name']
            for s in ['raw', 'png']:
                before = exact_from_rows[name][s]; after = comparisons[name][s]
                key = label+'/'+name+'/'+s
                stored_decisions[key] = {'pass': before['pass'], 'failures': [(x['group'], x['metric']) for x in before['preservation_failures']]}
                fresh_decisions[key] = {'pass': after['pass'], 'failures': [(x['group'], x['metric']) for x in after['preservation_failures']]}
                for d in differences(before, after): derived.append({'variant': label, 'cohort': name, 'stage': s, **d})
    max_metric = max((r['absolute_difference'] for r in changes), default=0)
    assert max_metric <= 1e-15 and stored_decisions == fresh_decisions
    outside = [r for r in derived if r['absolute_difference'] > 1e-10]
    assert outside and all('brightness_gain_fraction' in r['path'] or r['path'].endswith('/candidate') for r in outside)
    payload = {'complete': True, 'scope': 'MSE and mean-only MSE sensitivity; full independent audit remains pending',
        'checker_R1_failure_preserved': True, 'protocol_sha256': sha(BUNDLE/'protocol.json'),
        'VM_results_sha256': sha(folder/'results.json'), 'diagnostic_sha256': sha(Path(__file__)),
        'variants': len(labels), 'raw_PNG_MSE_sets': 2000, 'mean_only_pngs_exact': 1000,
        'maximum_individual_metric_difference': max_metric, 'changed_individual_metrics': len(changes),
        'individual_metric_differences': changes, 'derived_differences': derived,
        'derived_differences_outside_original_arithmetic_tolerance': outside,
        'maximum_derived_difference': max((r['absolute_difference'] for r in derived), default=0),
        'all36_failure_memberships_and_pass_decisions_exact': stored_decisions == fresh_decisions,
        'stored_row_groups_and_all36_comparisons_exact': True,
        'scientific_thresholds_changed': False, 'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0,
        'model_qualification': False, 'seconds': time.monotonic()-start}
    OUT.mkdir(exist_ok=True)
    with (OUT/'audit_r1_derived_receipt_diagnostic.json').open('x', encoding='utf-8') as f: json.dump(payload, f, indent=2); f.write('\n')
    print({k: payload[k] for k in ['complete', 'maximum_individual_metric_difference', 'maximum_derived_difference', 'changed_individual_metrics', 'all36_failure_memberships_and_pass_decisions_exact', 'seconds']})
    print(outside[:5])


if __name__ == '__main__': main()
