"""Bounded forward-only VGG diagnostic on the immutable ten paired preview cases."""
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cctv_dgp_pilot import read, write, sha, state_hash, image_tensor, PilotPerceptual, verify_bundle
from cctv_dgp_perceptual_v3 import PerceptualV3

DATA = ROOT / 'outputs/cctv_dgp_vm_bundle_v1'
RETURN = ROOT / 'outputs/cctv_dgp_normfix_return_v2'
OUT = ROOT / 'outputs/cctv_dgp_perceptual_diagnostic_v3'


def main():
    if OUT.exists():
        raise ValueError('Preserve the completed/partial diagnostic')
    receipt = read(RETURN / 'local_independent_audit.json')
    result = read(RETURN / 'outputs/cctv_dgp_pilot/results.json')
    if (not receipt['complete'] or sha(RETURN / 'local_independent_audit.json') !=
            '8eb3ec00c31e201689a22d4682c6fa21345653990529bf9d0986133a50eae685' or
            sha(RETURN / 'outputs/cctv_dgp_pilot/results.json') != receipt['returned_results_sha256']):
        raise ValueError('The audited return binding changed')
    protocol = verify_bundle(DATA)
    cases = {c['id']: c for c in protocol['validation_cases']}
    refs = {r['id']: r for r in protocol['references']}
    OUT.mkdir()
    write(OUT / 'protocol.json', {
        'format': 'cctv-perceptual-forward-diagnostic-v3', 'date': '2026-10-04',
        'parent_protocol_sha256': sha(DATA / 'protocol.json'),
        'return_audit_sha256': sha(RETURN / 'local_independent_audit.json'),
        'returned_results_sha256': receipt['returned_results_sha256'],
        'source_sha256': {n: sha(ROOT / n) for n in
            ['cctv_dgp_perceptual_v3.py', 'scripts/diagnose_cctv_dgp_perceptual_v3.py']},
        'cases': protocol['preview_case_ids'], 'preview_reference_count': 2,
        'feature_policies': ['postactivation', 'preactivation'],
        'tap_indices': {'postactivation': [3, 8, 17, 26], 'preactivation': [2, 7, 16, 25]},
        'maximum_vgg_forwards': 42, 'maximum_seconds_after_loading': 120,
        'native_reserved_used': False, 'optimizer_updates': 0, 'backward_calls': 0,
        'purpose': 'Measure feature sparsity/loss scaling; no causal or quality claim',
    })
    torch.set_num_threads(4)
    teacher = PerceptualV3(DATA / protocol['weights']['vgg_trunk'], 'cpu')
    before = state_hash(teacher)
    started = time.monotonic()
    rows, count = [], 0
    control_error = 0.
    for index, caseid in enumerate(protocol['preview_case_ids']):
        if time.monotonic() - started > 120:
            raise TimeoutError('Finite forward-only diagnostic budget exceeded')
        case = cases[caseid]
        relative = 'camera_identity_epoch2/images/' + caseid + '.png'
        prediction = RETURN / 'outputs/cctv_dgp_pilot' / relative
        if sha(prediction) != result['artifacts_sha256'][relative]:
            raise ValueError('Audited preview prediction changed')
        with Image.open(prediction) as image:
            generated = image_tensor(image.convert('RGB'))[None]
        with Image.open(DATA / refs[case['reference_id']]['target']) as image:
            target = image_tensor(image.convert('RGB'))[None]
        row = {'id': caseid, 'reference_id': case['reference_id'],
               'source': case['source'], 'profile': case['profile'], 'policies': {}}
        with torch.no_grad():
            for policy in ('postactivation', 'preactivation'):
                teacher.select_policy(policy)
                actual, expected = teacher.taps(generated), teacher.taps(target)
                count += 2
                if index == 0 and policy == 'postactivation':
                    old_actual = PilotPerceptual.taps(teacher, generated)
                    old_expected = PilotPerceptual.taps(teacher, target)
                    count += 2
                    for new, old in zip(actual + expected, old_actual + old_expected):
                        torch.testing.assert_close(new, old, rtol=0, atol=0)
                        control_error = max(control_error, float((new - old).abs().max()))
                taps = [{'mean_absolute_feature_error': float((a-b).abs().mean()),
                         'target_zero_fraction': float((b == 0).float().mean()),
                         'target_negative_fraction': float((b < 0).float().mean())}
                        for a, b in zip(actual, expected)]
                row['policies'][policy] = {
                    'taps': taps, 'weighted_perceptual_loss':
                    sum(w*t['mean_absolute_feature_error'] for w, t in zip((.1,.2,1.,1.), taps))}
        rows.append(row)
        if (index+1) % 2 == 0:
            print(f'Perceptual diagnostic {index+1}/10; VGG forwards={count}', flush=True)
    unchanged = state_hash(teacher) == before
    if not unchanged or any(p.requires_grad or p.grad is not None for p in teacher.parameters()) or count != 42:
        raise ValueError('Frozen teacher/finite-count invariant failed')
    elapsed = time.monotonic()-started
    if elapsed > 120:
        raise TimeoutError('Finite diagnostic budget exceeded')
    summary = {}
    for policy in ('postactivation', 'preactivation'):
        summary[policy] = {
            'mean_weighted_perceptual_loss': float(np.mean([
                r['policies'][policy]['weighted_perceptual_loss'] for r in rows])),
            'mean_target_zero_fraction_per_tap': [float(np.mean([
                r['policies'][policy]['taps'][i]['target_zero_fraction'] for r in rows])) for i in range(4)],
            'mean_target_negative_fraction_per_tap': [float(np.mean([
                r['policies'][policy]['taps'][i]['target_negative_fraction'] for r in rows])) for i in range(4)],
        }
    report = {'complete': True, 'protocol_sha256': sha(OUT / 'protocol.json'),
              'rows': rows, 'summary': summary, 'vgg_forwards': count,
              'seconds_after_loading': elapsed, 'teacher_state_before': before,
              'teacher_state_after': state_hash(teacher), 'teacher_state_unchanged': True,
              'postactivation_control_maximum_error': control_error,
              'native_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
              'native_reserved_used': False, 'trained_quality_claim': False,
              'causal_quality_improvement_established': False}
    write(OUT / 'results.json', report)
    print({k: v for k, v in report.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
