"""Independent fifty-case saved-output processing audit; no network/model calls."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgp_input_selector_v19 import DESIGN, select_restoration


def audit():
    start = time.monotonic()
    sys.path.insert(0, str(ROOT / 'outputs/cctv_dgp_structure_vm_v18'))
    import cctv_dgp_structure_v18 as q
    sys.path.insert(0, str(ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'))
    import cctv_dgp_generalization_v15 as v
    import numpy as np
    from PIL import Image
    out = ROOT / 'outputs/cctv_dgp_input_selector_v19'
    returned = ROOT / 'outputs/cctv_dgp_structure_return_v18/outputs/structure_v18'
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    p, result = v.read(out / 'processing_protocol_v19.json'), v.read(out / 'results.json')
    v.require(result['complete'] and result['protocol_sha256'] == v.sha(out / 'processing_protocol_v19.json')
              and p['design'] == DESIGN and v.sha(ROOT / 'dgp_input_selector_v19.py') == p['selector_sha256'], 'Frozen processing rule differs')
    v.require(p['parent_results_sha256'] == v.sha(returned / 'results.json'), 'Original V18 result differs')
    original = v.read(returned / 'results.json')
    for key in ['validation_used', 'native_used', 'native_reserved_used', 'production_promoted']:
        v.require(p[key] is False and result[key] is False, 'Training-only processing scope differs:' + key)
    v.require(p['neural_forwards'] == p['optimizer_updates'] == result['neural_forwards']
              == result['backward_calls'] == result['optimizer_updates'] == 0, 'No neural/fitting calls allowed')
    for name, digest in p['data_assets_sha256'].items():
        v.require(v.sha(mixed / name) == digest, 'Input/reference support fingerprint differs')
    b0, b600 = v.read(returned / 'update0/metrics.json'), v.read(returned / 'update600/metrics.json')
    lookup = {'retained_dgp_v2': {r['id']: r for r in b0['rows']},
              'structure_v18_update600': {r['id']: r for r in b600['rows']}}
    refs = {r['id']: r for r in p['references']}
    v.require([r['id'] for r in result['rows']] == [c['id'] for c in p['cases']]
              == [d['id'] for d in result['decisions']], 'All50 ordered cases/decisions required')
    for c, actual, row in zip(p['cases'], result['decisions'], result['rows']):
        v.require(time.monotonic() - start <= 60, 'Processing audit exceeds60s')
        camera = v.rgb(mixed / c['input'])
        with Image.open(mixed / refs[c['reference_id']]['observed']) as im: support = np.asarray(im).copy() > 0
        decision = select_restoration(camera, support)
        v.require(all(actual[k] == x for k, x in decision.items())
                  and row == lookup[decision['branch']][c['id']], 'Independent branch/output alias differs')
        for key, name in [('raw_sha256', row['raw']), ('PNG_sha256', row['prediction'])]:
            v.require(actual[key] == v.sha(returned / name) == original['artifacts_sha256'][name], 'Selected artifact differs')
        raw = np.load(returned / row['raw'], allow_pickle=False)
        image = v.rgb(returned / row['prediction'])
        v.require(np.array_equal(image, v.png(raw, camera, support)), 'Selected raw/PNG composition differs')
        target = v.rgb(mixed / refs[c['reference_id']]['target'])
        for key, expected in v.metrics(image, target, support).items():
            v.require(expected == row[key] if expected is None or isinstance(expected, bool)
                      else abs(expected - row[key]) <= 1e-9, 'Selected metric differs')
    summary = v.aggregate(result['rows'])
    guard = q.strict_preservation(summary, b0['summary'])
    v.require(summary == result['summary'] and guard == result['preservation'], 'Original processing guard arithmetic differs')
    v.require(sum(d['branch'] == 'retained_dgp_v2' for d in result['decisions']) == result['retained_cases'], 'Recorded branch count differs')
    v.write(out / 'independent_processing_audit.json', {'complete': True,
        'protocol_sha256': result['protocol_sha256'], 'results_sha256': v.sha(out / 'results.json'),
        'auditor_sha256': v.sha(Path(__file__)), 'input_decisions': 50, 'exact_output_aliases': 50,
        'PNG_metrics': 50, 'raw_compositions': 50, 'unchanged_quality_guard': guard,
        'neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'seconds': time.monotonic() - start, 'scope': 'Training-only processing verification, not development/native acceptance.'})
    print(json.dumps({'complete': True, 'cases': 50, 'quality_guard': guard, 'seconds': time.monotonic() - start}))


if __name__ == '__main__': audit()
