"""Freeze feature-loss scales on64 training references; forward-only, no validation fit."""
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import read, write, sha, state_hash, verify_bundle, PilotDataset, composite
from cctv_dgp_perceptual_v3 import PerceptualV3
from dgp_face_restoration import load_dgp_restorer

DATA = ROOT / 'outputs/cctv_dgp_vm_bundle_v1'
RETURN = ROOT / 'outputs/cctv_dgp_normfix_return_v2'
OUT = ROOT / 'outputs/cctv_dgp_perceptual_calibration_v3'
START_SHA = 'b30aeabecc60dff2fbd289575ce8691be9e670c653cacb6721bc154b618c3916'


def main():
    if OUT.exists():
        raise ValueError('Preserve the completed/partial scale calibration')
    receipt = read(RETURN / 'local_independent_audit.json')
    if not receipt['complete'] or sha(RETURN / 'local_independent_audit.json') != '8eb3ec00c31e201689a22d4682c6fa21345653990529bf9d0986133a50eae685':
        raise ValueError('Original independently audited return changed')
    protocol = verify_bundle(DATA)
    refs = {r['id']: r for r in protocol['references']}
    epoch = protocol['training_epochs']['1']
    sources = sorted({r['source'] for r in refs.values()})
    chosen = [c for source in sources for c in
              [c for c in epoch if c['source'] == source][:32]]
    if len(chosen) != 64 or len({c['reference_id'] for c in chosen}) != 64 or any(refs[c['reference_id']]['role'] != 'train' for c in chosen):
        raise ValueError('Training-only fixed calibration cohort differs')
    OUT.mkdir()
    write(OUT / 'protocol.json', {
        'format': 'cctv-perceptual-training-only-calibration-v3', 'date':'2026-10-04',
        'parent_protocol_sha256': sha(DATA / 'protocol.json'),
        'return_audit_sha256': sha(RETURN / 'local_independent_audit.json'),
        'starting_checkpoint_sha256': START_SHA, 'cases': chosen,
        'sampling':'First32 original epoch1 cases per source; no validation or native fitting',
        'rule':'Per-tap ratio of mean postactivation feature error to mean preactivation error at the fixed start',
        'source_sha256':{n:sha(ROOT/n) for n in ['cctv_dgp_perceptual_v3.py','scripts/calibrate_cctv_dgp_perceptual_v3.py']},
        'maximum_dgp_batch_forwards':8, 'maximum_vgg_batch_forwards':32,
        'maximum_seconds_after_loading':240, 'cpu_threads':4,
        'validation_references_used':0, 'native_cases_used':0, 'native_reserved_used':False,
        'optimizer_updates':0, 'backward_calls':0,
    })
    torch.set_num_threads(4)
    model, _ = load_dgp_restorer(RETURN / 'outputs/cctv_dgp_pilot/camera_identity/best.pth', 'cpu', START_SHA)
    teacher = PerceptualV3(DATA / protocol['weights']['vgg_trunk'], 'cpu')
    before, teacher_before = state_hash(model.net), state_hash(teacher)
    loader = torch.utils.data.DataLoader(PilotDataset(DATA, protocol, chosen), batch_size=8, shuffle=False, num_workers=0)
    sums = {p:np.zeros(4, np.float64) for p in ('postactivation','preactivation')}
    dgp_calls = vgg_calls = 0
    started = time.monotonic()
    for batch in loader:
        if time.monotonic()-started > 240:
            raise TimeoutError('Finite training-only calibration budget exceeded')
        with torch.inference_mode():
            predicted = composite(model(batch['low']), batch['low'], batch['mask'])
            dgp_calls += 1
            for policy in sums:
                teacher.select_policy(policy)
                actual, expected = teacher.taps(predicted), teacher.taps(batch['target'])
                vgg_calls += 2
                for i, (a,b) in enumerate(zip(actual,expected)):
                    sums[policy][i] += (a-b).abs().mean((1,2,3)).double().sum().item()
        print(f'Training-only calibration {dgp_calls*8}/64; no updates', flush=True)
    means = {p:v/64 for p,v in sums.items()}
    scales = means['postactivation']/means['preactivation']
    if not np.isfinite(scales).all() or (scales<=0).any() or (scales>10).any():
        raise ValueError('Unusable fixed feature scales')
    if state_hash(model.net) != before or state_hash(teacher) != teacher_before or any(p.grad is not None or p.requires_grad for m in (model,teacher) for p in m.parameters()):
        raise ValueError('Frozen calibration state changed')
    elapsed = time.monotonic()-started
    if elapsed>240 or dgp_calls!=8 or vgg_calls!=32:
        raise ValueError('Finite calibration counts/timing differ')
    report = {
        'complete':True, 'protocol_sha256':sha(OUT/'protocol.json'),
        'starting_checkpoint_sha256':START_SHA, 'starting_state_hash':before,
        'teacher_state_hash':teacher_before, 'teacher_states_unchanged':True,
        'postactivation_mean_error_per_tap':means['postactivation'].tolist(),
        'preactivation_mean_error_per_tap':means['preactivation'].tolist(),
        'preactivation_scales':scales.tolist(), 'postactivation_scales':[1.,1.,1.,1.],
        'reference_count':64, 'references_per_source':32,
        'dgp_batch_forwards':dgp_calls, 'vgg_batch_forwards':vgg_calls,
        'seconds_after_loading':elapsed, 'validation_references_used':0,
        'native_cases_used':0, 'native_reserved_used':False,
        'backward_calls':0, 'optimizer_updates':0,
        'gradient_scale_equivalence_claimed':False, 'trained_improvement_claimed':False,
    }
    write(OUT/'scales.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
