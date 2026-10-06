"""Freeze and verify V19 training-only saved-output processing; no neural calls."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dgp_input_selector_v19 import DESIGN, select_restoration

V18 = ROOT / 'outputs/cctv_dgp_structure_vm_v18'
RETURN = ROOT / 'outputs/cctv_dgp_structure_return_v18/outputs/structure_v18'
OUT = ROOT / 'outputs/cctv_dgp_input_selector_v19'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run():
    start = time.monotonic()
    if OUT.exists(): raise ValueError('Keep previous control/failure; no overwrite')
    p = read(V18 / 'structure_protocol_v18.json')
    verified = read(ROOT / 'outputs/cctv_dgp_structure_v18_verified_summary.json')
    if sha(RETURN / 'results.json') != verified['results_sha256']:
        raise ValueError('Audited fixed V18 return required')
    protocol = {'format': 'training-only-input-selection-processing-v19', 'design': DESIGN,
        'selector_sha256': sha(ROOT / 'dgp_input_selector_v19.py'),
        'runner_sha256': sha(Path(__file__)), 'parent_protocol_sha256': sha(V18 / 'structure_protocol_v18.json'),
        'parent_results_sha256': verified['results_sha256'],
        'terminal_checkpoint_sha256': verified['snapshots']['600']['checkpoint_sha256'],
        'input_diagnosis_sha256': sha(ROOT / 'outputs/cctv_dgp_clear_processing_diagnostic_v18.json'),
        'references': p['references'], 'cases': p['training_cases'], 'data_assets_sha256': p['data_assets_sha256'],
        'arithmetic_cap_seconds': 60, 'neural_forwards': 0, 'optimizer_updates': 0,
        'validation_used': False, 'native_used': False, 'native_reserved_used': False, 'production_promoted': False,
        'guard_policy': 'Unchanged V18 MSE1e-12/SSIM-cosine1e-6/PSNR>=0.1dB/degraded-MSE>=10%; V18 failures remain'}
    OUT.mkdir()
    # Freeze design before reading any output quality to evaluate this control.
    write_new(OUT / 'processing_protocol_v19.json', protocol)
    sys.path.insert(0, str(V18))
    import cctv_dgp_structure_v18 as q
    sys.path.insert(0, str(ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'))
    import cctv_dgp_generalization_v15 as v
    import numpy as np
    from PIL import Image
    p0, p600 = read(RETURN / 'update0/metrics.json'), read(RETURN / 'update600/metrics.json')
    before = {r['id']: r for r in p0['rows']}
    terminal = {r['id']: r for r in p600['rows']}
    refs = {r['id']: r for r in p['references']}
    mixed = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
    rows, decisions, artifacts = [], [], {}
    for c in p['training_cases']:
        v.require(time.monotonic() - start < 60, 'Finite processing audit exceeds60s')
        camera = v.rgb(mixed / c['input'])
        with Image.open(mixed / refs[c['reference_id']]['observed']) as image:
            support = np.asarray(image).copy() > 0
        decision = select_restoration(camera, support)
        row = (before if decision['branch'] == 'retained_dgp_v2' else terminal)[c['id']]
        raw = np.load(RETURN / row['raw'], allow_pickle=False)
        png = v.rgb(RETURN / row['prediction'])
        v.require(np.array_equal(png, v.png(raw, camera, support)), 'Selected raw/PNG composition differs')
        target = v.rgb(mixed / refs[c['reference_id']]['target'])
        recomputed = v.metrics(png, target, support)
        for key, actual in recomputed.items():
            v.require(actual == row[key] if isinstance(actual, bool) or actual is None else abs(actual - row[key]) <= 1e-9,
                      'Selected PNG metric differs:' + key)
        rows.append(row.copy())
        decisions.append({'id': c['id'], 'input_sha256': sha(mixed / c['input']), **decision,
                          'raw_sha256': sha(RETURN / row['raw']), 'PNG_sha256': sha(RETURN / row['prediction'])})
    summary = v.aggregate(rows)
    guard = q.strict_preservation(summary, p0['summary'])
    # Retain output and any guard failure rather than asserting scientific pass.
    result = {'complete': True, 'protocol_sha256': sha(OUT / 'processing_protocol_v19.json'),
        'rows': rows, 'decisions': decisions, 'summary': summary, 'preservation': guard,
        'retained_cases': sum(d['branch'] == 'retained_dgp_v2' for d in decisions),
        'spatial_cases': sum(d['branch'] == 'structure_v18_update600' for d in decisions),
        'neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'validation_used': False, 'native_used': False, 'native_reserved_used': False,
        'production_promoted': False, 'seconds': time.monotonic() - start,
        'limitation': 'Training-calibrated processing on fitted faces; no face/information qualification, native effectiveness or generalization proof.'}
    write_new(OUT / 'results.json', result)
    print(json.dumps({'complete': True, 'retained_cases': result['retained_cases'], 'spatial_cases': result['spatial_cases'],
                      'preservation': guard, 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__': run()
