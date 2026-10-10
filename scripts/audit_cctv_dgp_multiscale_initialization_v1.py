"""Independent retained-row audit and ten-case CPU initializer replay; no gradients."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_parity'
CP = ROOT / 'outputs/cctv_dgp_head4_capacity_vm_v1'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, d):
    with p.open('x', encoding='utf-8') as f:
        f.write(json.dumps(d, indent=2, allow_nan=False) + '\n')


def prepare():
    paths = [OUT / 'plan.json', OUT / 'results.json', Path(__file__)]
    write(OUT / 'independent_plan.json', {'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        'replay_indices': list(range(5)) + list(range(25, 30)), 'model_calls': 4,
        'maximum_seconds': 100, 'gradient_queries': 0, 'optimizer_updates': 0})


def run():
    start = time.monotonic(); plan = read(OUT / 'independent_plan.json')
    for n, d in plan['source_sha256'].items():
        assert sha(ROOT / n) == d, n
    p = read(OUT / 'plan.json'); r = read(OUT / 'results.json')
    for n, d in p['source_sha256'].items():
        assert sha(ROOT / n) == d, n
    assert r['complete'] and r['plan_sha256'] == sha(OUT / 'plan.json')
    assert [x['id'] for x in r['rows']] == [x['id'] for x in p['cases']]
    assert len(set(x['id'] for x in r['rows'])) == 100
    assert all(x['role'] == 'train' for x in p['cases'])
    assert all(x['raw_max_abs'] == 0 and len(x['RGB_float32_sha256']) == 64 for x in r['rows'])
    assert r['states_before'] == r['states_after'] and r['neural_calls'] == 40
    assert r['gradient_queries'] == r['optimizer_updates'] == 0
    assert r['local_training_guards_refused'] == ['deep3', 'decoder15', 'gradient_enabled_forward']
    assert not r['model_qualification'] and not r['goal_complete']
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_multiscale_calibration_model_v1 import MultiscaleCalibrationDGP
    from cctv_dgp_pilot import state_hash, buffer_hash
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(CP / 'weights/dgp_v2.pth',
        expected_sha256='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b', device='cpu')
    candidate = MultiscaleCalibrationDGP(original.net).eval().requires_grad_(False)
    assert state_hash(original.net) == r['states_before']['original']
    assert state_hash(candidate) == r['states_before']['candidate']
    assert buffer_hash(candidate) == r['states_before']['buffers']
    values = dict(candidate.named_parameters())
    for partition, count, elements in [('deep3', 3, 147456), ('decoder15', 15, 609219)]:
        names = candidate.selected_names(partition)
        assert len(names) == count and sum(values[n].numel() for n in names) == elements
        assert len({values[n].data_ptr() for n in names}) == count
        assert values['net.smooth.0.weight'].data_ptr() not in [values[n].data_ptr() for n in names]
    assert torch.equal(candidate.net.smooth[0].weight, original.net.smooth[0].weight)
    refs = {x['id']: x for x in p['references']}; calls = 0; replayed = []
    with torch.inference_mode():
        for indices in [plan['replay_indices'][:5], plan['replay_indices'][5:]]:
            cameras = []; masks = []
            for i in indices:
                c = p['cases'][i]
                with Image.open(CP / c['input']) as im:
                    cameras.append(np.asarray(im.convert('RGB')).copy())
                with Image.open(CP / refs[c['source_person_or_reference']]['observed']) as im:
                    masks.append(np.asarray(im.convert('L')) > 0)
            x = torch.from_numpy(np.stack(cameras).astype(np.float32) / np.float32(255)).permute(0, 3, 1, 2)
            mask = torch.from_numpy(np.stack(masks))[:, None]
            baseline = torch.where(mask, original.net(x), x)
            current = torch.where(mask, candidate(x), x); calls += 2
            assert torch.equal(baseline, current)
            for j, i in enumerate(indices):
                a = current[j].permute(1, 2, 0).contiguous().numpy()
                assert hashlib.sha256(a.tobytes()).hexdigest() == r['rows'][i]['RGB_float32_sha256']
                replayed.append(p['cases'][i]['id'])
    assert calls == 4 and all(v.grad is None and not v.requires_grad for v in candidate.parameters())
    assert state_hash(candidate) == r['states_before']['candidate']
    assert buffer_hash(candidate) == r['states_before']['buffers']
    for n, d in p['source_sha256'].items():
        assert sha(ROOT / n) == d, n
    assert time.monotonic() - start < plan['maximum_seconds']
    write(OUT / 'independent_audit.json', {'complete': True, 'independent_plan_sha256': sha(OUT / 'independent_plan.json'),
        'worker_results_sha256': sha(OUT / 'results.json'), 'retained_rows_checked': 100,
        'fresh_exact_replays': replayed, 'neural_calls': calls, 'maximum_raw_error': 0,
        'original_fusion_storage_unchanged': True, 'independent_learning_ownership_verified': True,
        'gradient_queries': 0, 'optimizer_updates': 0, 'model_qualification': False,
        'goal_complete': False, 'seconds': time.monotonic() - start})
    print({'complete': True, 'retained_rows': 100, 'exact_CPU_replay_cases': 10, 'gradient_queries': 0, 'optimizer_updates': 0}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare', action='store_true'); g.add_argument('--run', action='store_true')
    a = parser.parse_args(); prepare() if a.prepare else run()
