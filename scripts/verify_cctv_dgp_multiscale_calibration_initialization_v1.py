"""Prospective 100-case CPU initializer parity; no gradients or training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_parity'
CP = ROOT / 'outputs/cctv_dgp_head4_capacity_vm_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare():
    assert not OUT.exists()
    cp = read(CP / 'protocol.json')
    gp = read(ROOT / 'outputs/cctv_dgp_head4_reactivation_vm_v1/protocol.json')
    by_id = {c['id']: c for c in cp['cases']}
    cases = [by_id[c['id']] for c in gp['cases']]
    refs = {r['id']: r for r in cp['references']}
    assert len(cases) == 100 and all(c['role'] == 'train' for c in cases)
    paths = [Path(__file__), ROOT / 'scripts/cctv_dgp_multiscale_calibration_model_v1.py',
             ROOT / 'dgp_frozen_inference_v2.py', ROOT / 'cctv_dgp_frozen_norm.py',
             ROOT / 'dgp_face_restoration.py', ROOT / 'cctv_dgp_pilot.py',
             CP / 'protocol.json', CP / 'weights/dgp_v2.pth',
             ROOT / 'outputs/cctv_dgp_head4_learning_signal_review_v1/independent_audit.json']
    paths += sorted((ROOT / 'models').glob('*.py'))
    paths += [CP / c['input'] for c in cases]
    paths += [CP / refs[cases[i]['source_person_or_reference']]['observed'] for i in range(0, 100, 5)]
    OUT.mkdir()
    write(OUT / 'plan.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'cases': cases,
                            'references': [refs[cases[i]['source_person_or_reference']] for i in range(0, 100, 5)],
                            'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                            'model_calls': 40, 'maximum_seconds': 180, 'exact_initial_raw_error': 0,
                            'optimizer_updates': 0, 'gradient_queries': 0, 'reserved_final_pixels': 0,
                            'model_qualification': False, 'goal_complete': False})
    print({'prepared': True, 'plan_sha256': sha(OUT / 'plan.json')}, flush=True)


def run():
    start = time.monotonic()
    assert not (OUT / 'results.json').exists()
    p = read(OUT / 'plan.json')
    for n, d in p['source_sha256'].items():
        assert sha(ROOT / n) == d, n
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_pilot import state_hash, buffer_hash
    from cctv_dgp_multiscale_calibration_model_v1 import MultiscaleCalibrationDGP
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(CP / 'weights/dgp_v2.pth',
                                          expected_sha256='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b', device='cpu')
    candidate = MultiscaleCalibrationDGP(original.net).eval().requires_grad_(False)
    assert all(not m.training for m in candidate.modules())
    before = {'original': state_hash(original.net), 'candidate': state_hash(candidate), 'buffers': buffer_hash(candidate)}
    assert before['original'] == 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
    refs = {r['id']: r for r in p['references']}
    rows = []; calls = 0; last_x = None
    with torch.inference_mode():
        for begin in range(0, 100, 5):
            cases = p['cases'][begin:begin + 5]; cameras = []; masks = []
            for c in cases:
                with Image.open(CP / c['input']) as im:
                    assert im.size == (256, 256); cameras.append(np.asarray(im.convert('RGB')).copy())
                with Image.open(CP / refs[c['source_person_or_reference']]['observed']) as im:
                    assert im.size == (256, 256); masks.append(np.asarray(im.convert('L')) > 0)
            x = torch.from_numpy(np.stack(cameras).astype(np.float32) / np.float32(255)).permute(0, 3, 1, 2)
            mask = torch.from_numpy(np.stack(masks))[:, None]
            a = torch.where(mask, original.net(x), x); b = torch.where(mask, candidate(x), x); calls += 2
            assert torch.equal(a, b), 'Initializer changes current output'
            for i, c in enumerate(cases):
                rows.append({'id': c['id'], 'source': c['source'], 'profile': c['profile'], 'raw_max_abs': 0.,
                             'RGB_float32_sha256': hashlib.sha256(a[i].permute(1, 2, 0).contiguous().numpy().tobytes()).hexdigest()})
            last_x = x
    # Guard tests reject before any forward or metadata HTTP request.
    rejected = []
    for partition in ['deep3', 'decoder15']:
        try:
            candidate.enable_vm_learning(ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1', partition)
        except AssertionError:
            rejected.append(partition)
        else:
            raise AssertionError('Local training must be refused')
    try:
        candidate(last_x)
    except AssertionError:
        rejected.append('gradient_enabled_forward')
    else:
        raise AssertionError('Gradient-enabled local forward must be refused')
    after = {'original': state_hash(original.net), 'candidate': state_hash(candidate), 'buffers': buffer_hash(candidate)}
    assert before == after and calls == 40 and len(rows) == 100
    assert all(v.grad is None and not v.requires_grad for v in candidate.parameters())
    assert time.monotonic() - start < p['maximum_seconds']
    for n, d in p['source_sha256'].items():
        assert sha(ROOT / n) == d, n
    result = {'complete': True, 'plan_sha256': sha(OUT / 'plan.json'), 'rows': rows,
              'exact_initial_parity_cases': 100, 'maximum_initial_raw_error': 0.,
              'states_before': before, 'states_after': after, 'neural_calls': calls,
              'partition_elements': {'deep3': 147456, 'decoder15': 609219},
              'partition_tensors': {'deep3': 3, 'decoder15': 15}, 'local_training_guards_refused': rejected,
              'optimizer_updates': 0, 'gradient_queries': 0, 'model_qualification': False,
              'goal_complete': False, 'seconds': time.monotonic() - start}
    write(OUT / 'results.json', result)
    print({k: v for k, v in result.items() if k not in ['rows', 'states_before', 'states_after']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--run', action='store_true')
    a = parser.parse_args()
    prepare() if a.prepare else run()
